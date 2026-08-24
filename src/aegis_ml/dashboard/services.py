"""Start, probe and stop the two premade UIs the hub points at.

The hub does not re-implement MLflow's run comparison or Optuna's parallel-coordinate plot.
Those are years of work by people who do nothing else, they are already installed, and this
package already writes the two stores they read:

* **MLflow** reads ``registry_store/mlflow/mlflow.db``, populated by
  :func:`aegis_ml.registry.mlflow_mirror.backfill` from the filesystem registry — which
  stays the source of truth. Deleting the MLflow store loses a browser tab and nothing else.
* **Optuna Dashboard** reads ``registry_store/optuna/studies.db`` directly. That file is not
  written for the dashboard's benefit — :mod:`aegis_ml.automl.hpo` persists every study
  there so a search is resumable — so the dashboard is reading the real search, complete
  with its pruned trials, not an export of it.

Both are launched as child processes rather than imported into this one. Two reasons, and
both have bitten this kind of code before: MLflow's server binds a socket and installs
signal handlers at import time, so an in-process launch cannot be shut down cleanly; and a
crash inside either UI must not take the hub down with it, because the hub is the thing
that explains what went wrong.

**Every refusal here is explicit and carries its remedy.** A missing package, a held port
and an absent store are three different failures with three different fixes, and the panel
the reader sees names which one happened and the exact command that clears it. The one
thing this module never does is return a healthy-looking state for a service that is not
running — a blank iframe with no explanation is the failure this whole file exists to
prevent.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from aegis_ml._require import is_available

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

__all__ = [
    "MLFLOW_KEY",
    "OPTUNA_KEY",
    "Probe",
    "ServiceState",
    "Supervisor",
    "free_port_from",
    "mlflow_store_uri",
    "optuna_storage_uri",
    "port_in_use",
    "probe",
    "supervise",
]

MLFLOW_KEY = "mlflow"
"""Stable key used by the hub's JSON payload and by ``/api/services.json``."""

OPTUNA_KEY = "optuna"
"""Stable key for the Optuna Dashboard panel."""

_STARTUP_GRACE = 40.0
"""Seconds to wait for a child to answer its first request.

MLflow's first start against a fresh SQLite store runs the full Alembic migration chain,
which takes tens of seconds on a cold filesystem cache. A shorter budget reports "MLflow
did not come up" for a server that was three seconds from being ready — a diagnosis worse
than no diagnosis, because it sends the reader after the wrong problem.
"""

_POLL_INTERVAL = 0.4
"""Gap between readiness polls. Short enough that a fast start feels instant."""


@dataclass(frozen=True)
class Probe:
    """The result of one HTTP request against a service root.

    Attributes:
        status: The HTTP status code, or ``None`` when the connection itself failed.
        error: The transport-level failure, when there was one.
        frame_blocked: Whether response headers forbid embedding this origin in the hub's
            iframe. Determined from the *server's own headers* rather than guessed at in
            the browser: a frame blocked by ``X-Frame-Options`` still fires ``load`` in
            Chrome, so client-side detection reports success for a blank grey rectangle.
        frame_reason: The header that forbade embedding, verbatim, for the panel to quote.
    """

    status: int | None
    error: str | None = None
    frame_blocked: bool = False
    frame_reason: str | None = None

    @property
    def up(self) -> bool:
        """Whether the service answered at all.

        Any HTTP status counts as up. A 403 from a server that is listening is a
        configuration question; a refused connection is an "it is not running" question,
        and the panel must not conflate them.
        """
        return self.status is not None


@dataclass
class ServiceState:
    """Everything the hub needs to render one service panel, honestly.

    Attributes:
        key: :data:`MLFLOW_KEY` or :data:`OPTUNA_KEY`.
        label: Human name, as printed on the panel.
        blurb: One line saying what this UI is for and which store it reads.
        url: Where it is (or would be) reachable. Always populated so the panel can show
            the address even when the service is down.
        landing: Fragment appended to ``url`` to open the view that actually holds this
            repository's data. MLflow 3 opens on a GenAI welcome screen by default, and a
            panel that lands a viewer on a product tour instead of the run they came to see
            is worse than no panel — the fragment sends them straight to the experiments.
        identity_path: A path only *this* service answers with 200. Used before adopting a
            server that is already on the port: "something is listening" and "MLflow is
            listening" are different facts, and the panel must not report the second when
            it only established the first.
        port: The TCP port.
        managed: Whether *this* process started it. A service already listening on the
            port when we arrived is used as-is and never terminated on exit.
        running: Whether it is answering right now.
        reason: Why it is not running, when it is not. Empty when it is.
        remedy: The exact shell command that fixes ``reason``.
        embeddable: Whether the panel may use an iframe. ``None`` until probed.
        frame_reason: The response header that forbade embedding, when one did.
        pid: The child process id, when we started it.
        log_path: Where the child's stdout and stderr are being written.
    """

    key: str
    label: str
    blurb: str
    url: str
    port: int
    landing: str = ""
    identity_path: str = ""
    managed: bool = False
    running: bool = False
    reason: str = ""
    remedy: str = ""
    embeddable: bool | None = None
    frame_reason: str | None = None
    pid: int | None = None
    log_path: str | None = None
    _process: subprocess.Popen[bytes] | None = field(default=None, repr=False, compare=False)

    def as_json(self) -> dict[str, object]:
        """Return the JSON-safe view the hub page and ``/api/services.json`` share.

        The private process handle is excluded deliberately: the same dict is embedded in
        the page and served over HTTP, and a payload that differs between those two paths
        is how a status indicator ends up disagreeing with itself after a refresh.
        """
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "url": self.url,
            "entry_url": self.url + self.landing,
            "identity_path": self.identity_path,
            "port": self.port,
            "managed": self.managed,
            "running": self.running,
            "reason": self.reason,
            "remedy": self.remedy,
            "embeddable": self.embeddable,
            "frame_reason": self.frame_reason,
            "pid": self.pid,
            "log_path": self.log_path,
        }


def port_in_use(host: str, port: int, timeout: float = 0.35) -> bool:
    """Return whether something is already listening on ``host:port``.

    Args:
        host: Bind address to test. ``0.0.0.0`` is probed as ``127.0.0.1`` because a
            connect() to the wildcard address is not meaningful on every platform.
        port: TCP port.
        timeout: Connect timeout in seconds.

    Returns:
        True when a connection succeeds.

    On macOS this is not a rare edge: port 5000 — MLflow's own default — is held by the
    AirPlay Receiver in Control Centre out of the box, so the very first run of this
    command on an unmodified laptop hits this path.
    """
    target = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host  # noqa: S104
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.settimeout(timeout)
        return probe_socket.connect_ex((target, port)) == 0


def free_port_from(host: str, start: int, span: int = 12) -> int | None:
    """Return the first free port at or after ``start``, or ``None`` within ``span``.

    Args:
        host: Bind address the ports are tested against.
        start: First candidate.
        span: How many consecutive ports to try.

    Returns:
        A free port, or ``None`` when every candidate in the window is taken.

    Only used when the caller did **not** name a port. An explicitly requested port that
    is busy is reported as a failure with its remedy, never quietly moved: a reader who
    typed ``--mlflow-port 5001`` and got 5003 has been lied to.
    """
    for candidate in range(start, start + span):
        if not port_in_use(host, candidate):
            return candidate
    return None


def _frame_verdict(headers: object, page_origin: str) -> tuple[bool, str | None]:
    """Decide from response headers whether ``page_origin`` may iframe this service.

    Args:
        headers: The ``email.message.Message`` an :mod:`http.client` response carries.
        page_origin: The hub's own origin, e.g. ``http://127.0.0.1:8000``.

    Returns:
        ``(blocked, reason)``. ``reason`` quotes the offending header so the fallback card
        can say *why* it is a link instead of a frame, rather than asserting it generically.

    The hub and the services are on different ports, therefore different origins, so
    ``SAMEORIGIN`` blocks just as firmly as ``DENY`` does. That distinction is the entire
    reason this check exists — ``SAMEORIGIN`` reads like it should work and does not.
    """
    get = getattr(headers, "get", None)
    if get is None:
        return False, None
    xfo = (get("X-Frame-Options") or "").strip()
    if xfo and xfo.upper() in {"DENY", "SAMEORIGIN"}:
        return True, f"X-Frame-Options: {xfo}"
    csp = (get("Content-Security-Policy") or "").strip()
    for directive in csp.split(";"):
        clean = directive.strip()
        if not clean.lower().startswith("frame-ancestors"):
            continue
        allowed = clean[len("frame-ancestors") :].strip()
        if "*" in allowed or page_origin in allowed:
            return False, None
        return True, f"Content-Security-Policy: {clean}"
    return False, None


def probe(url: str, page_origin: str, timeout: float = 2.0) -> Probe:
    """Make one GET against ``url`` and report status plus frame-embedding headers.

    Args:
        url: The service root to request.
        page_origin: The hub's origin, used to evaluate ``frame-ancestors``.
        timeout: Socket timeout in seconds.

    Returns:
        A :class:`Probe`. A non-2xx status is still "up" — the service is listening, which
        is a different fact from the service being healthy, and the panel shows both.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="GET")  # noqa: S310 - loopback, our own child
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            response.read(2048)
            blocked, reason = _frame_verdict(response.headers, page_origin)
            return Probe(status=int(response.status), frame_blocked=blocked, frame_reason=reason)
    except urllib.error.HTTPError as exc:
        blocked, reason = _frame_verdict(exc.headers, page_origin)
        return Probe(status=int(exc.code), frame_blocked=blocked, frame_reason=reason)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return Probe(status=None, error=f"{type(exc).__name__}: {exc}")


def mlflow_store_uri(registry_dir: Path) -> tuple[str, Path]:
    """Return the SQLite tracking URI and artifact root for the local MLflow store.

    Args:
        registry_dir: The registry root, i.e. ``settings.registry_dir``.

    Returns:
        ``(backend_store_uri, artifact_root)``. SQLite rather than the bare ``mlruns/``
        file store because the file store cannot serve MLflow's model registry, and the
        registry tab is the one a viewer clicks first after seeing the word "registry" on
        the hub.
    """
    root = registry_dir / "mlflow"
    root.mkdir(parents=True, exist_ok=True)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}", artifacts


def optuna_storage_uri(registry_dir: Path) -> tuple[str, Path]:
    """Return the Optuna storage URL and the SQLite file it points at.

    Args:
        registry_dir: The registry root.

    Returns:
        ``(storage_url, db_path)``. The file is **not** created here: its absence is a
        real answer — no hyper-parameter search has been persisted yet — and the panel
        says so and names the command that produces one. Creating an empty database would
        replace that answer with an Optuna Dashboard showing zero studies, which looks
        like a broken dashboard rather than an unrun search.
    """
    db_path = registry_dir / "optuna" / "studies.db"
    return f"sqlite:///{db_path.as_posix()}", db_path


class Supervisor:
    """Owns the child processes for the lifetime of one ``aegis-ml dashboard`` invocation.

    Used as a context manager so that every exit path — a clean Ctrl-C, an exception in
    the hub server, an unhandled failure anywhere — runs :meth:`shutdown`. Orphaned MLflow
    servers are not a theoretical tidiness concern: each one keeps its port, so the *next*
    invocation finds the port busy and degrades, and the reader is left debugging a
    dashboard that worked five minutes ago.

    Only processes this supervisor started are terminated. A service that was already
    listening when we arrived belongs to whoever started it.
    """

    def __init__(self, host: str, page_port: int, registry_dir: Path) -> None:
        """Record where the hub itself lives, so frame headers can be judged against it.

        Args:
            host: Bind address shared by the hub and the services it starts.
            page_port: The hub's own port, which forms the origin an iframe would load from.
            registry_dir: Registry root; the two stores and the child logs live under it.
        """
        self.host = host
        self.page_port = page_port
        self.registry_dir = Path(registry_dir)
        self.states: dict[str, ServiceState] = {}
        self._logs: list[BinaryIO] = []

    @property
    def page_origin(self) -> str:
        """The hub's origin, as a browser would compute it."""
        display = "127.0.0.1" if self.host in {"0.0.0.0", "::", ""} else self.host  # noqa: S104
        return f"http://{display}:{self.page_port}"

    def _display_host(self) -> str:
        """The address a human types, which is loopback even when we bind the wildcard."""
        return "127.0.0.1" if self.host in {"0.0.0.0", "::", ""} else self.host  # noqa: S104

    def _log_file(self, key: str) -> Path:
        """Return the path a child's combined stdout/stderr is written to."""
        log_dir = self.registry_dir / "dashboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{key}.log"

    def _spawn(self, key: str, argv: list[str], cwd: Path) -> subprocess.Popen[bytes]:
        """Start one child with its output captured to a file, and remember the handle."""
        log_path = self._log_file(key)
        handle = log_path.open("wb")
        self._logs.append(handle)
        return subprocess.Popen(  # noqa: S603 - argv is built here from vetted parts
            argv,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd),
            start_new_session=True,
        )

    def _await_ready(self, state: ServiceState, deadline: float) -> None:
        """Poll ``state.url`` until it answers, the child dies, or ``deadline`` passes."""
        while time.monotonic() < deadline:
            process = state._process  # noqa: SLF001 - same class, private by convention only
            if process is not None and process.poll() is not None:
                state.running = False
                state.reason = (
                    f"{state.label} exited with code {process.returncode} during startup. "
                    f"Its output was captured — read it, the failure is named there."
                )
                state.remedy = f"cat {state.log_path}"
                return
            result = probe(state.url, self.page_origin, timeout=1.5)
            if result.up:
                state.running = True
                state.reason = ""
                state.embeddable = not result.frame_blocked
                state.frame_reason = result.frame_reason
                return
            time.sleep(_POLL_INTERVAL)
        state.running = False
        state.reason = (
            f"{state.label} was started but did not answer on port {state.port} within "
            f"{_STARTUP_GRACE:.0f}s. It may still be migrating its store."
        )
        state.remedy = f"cat {state.log_path}"

    def _adopt_existing(self, state: ServiceState) -> bool:
        """Adopt a server already on the port, but only once it proves it is the right one.

        Args:
            state: The service being brought up. Its ``identity_path`` is what settles it.

        Returns:
            True when the listener answered ``identity_path`` with 200 and the state was
            filled in from it; False when the port is held by something else.

        The identity request is the point of this method. On macOS, port 5000 — MLflow's
        documented default — is held by the AirPlay Receiver, which answers every request
        with a 403. Adopting on "the port responded" alone would put a green dot and the
        word "live" next to a panel showing an empty grey frame served by Control Centre,
        which is exactly the confident-and-wrong failure this dashboard exists to avoid.
        """
        result = probe(state.url, self.page_origin, timeout=1.5)
        if not result.up:
            return False
        identity = probe(state.url.rstrip("/") + state.identity_path, self.page_origin, 1.5)
        if identity.status != 200:  # noqa: PLR2004 - 200 is the whole assertion here
            return False
        state.running = True
        state.managed = False
        state.embeddable = not result.frame_blocked
        state.frame_reason = result.frame_reason
        state.reason = ""
        return True

    def _port_held_by_stranger(self, state: ServiceState, flag: str) -> ServiceState:
        """Record that the chosen port belongs to something that is not this service."""
        state.running = False
        state.reason = (
            f"Port {state.port} is already held by a process that does not answer "
            f"{state.identity_path} the way {state.label} does, so {state.label} was not "
            f"started — binding that port would have failed. On macOS, port 5000 is the "
            f"AirPlay Receiver unless it has been switched off in System Settings."
        )
        state.remedy = f"aegis-ml dashboard --{flag} <a free port>"
        return state

    def start_mlflow(self, port: int | None, default_port: int) -> ServiceState:
        """Bring up the MLflow tracking UI against the registry's local store.

        Args:
            port: The port the caller explicitly asked for, or ``None`` for "choose one".
            default_port: The port to prefer when ``port`` is ``None``.

        Returns:
            A :class:`ServiceState` that is either running or carries the reason it is not.

        When ``port`` is ``None`` and ``default_port`` is busy, the next free port is used
        and printed. That is not a silent downgrade — the address is shown in the banner,
        in the rail and on the panel, and the caller expressed no preference to override.
        An explicit ``port`` that is busy fails with its remedy instead.
        """
        chosen, refusal = self._resolve_port(port, default_port, "mlflow-port")
        state = ServiceState(
            key=MLFLOW_KEY,
            label="MLflow",
            blurb=(
                "Experiment tracking, run comparison and the model registry, reading the "
                "mirror of this repository's filesystem registry."
            ),
            port=chosen,
            url=f"http://{self._display_host()}:{chosen}/",
            landing="#/experiments",
            identity_path="/version",
        )
        self.states[MLFLOW_KEY] = state
        if refusal is not None:
            state.reason, state.remedy = refusal
            return state
        if port_in_use(self.host, chosen):
            if self._adopt_existing(state):
                return state
            return self._port_held_by_stranger(state, "mlflow-port")
        if not is_available("mlflow"):
            state.reason = (
                "mlflow is not importable in this interpreter, so the tracking UI cannot "
                "be started. Every number the hub shows is read from the filesystem "
                "registry and is unaffected."
            )
            state.remedy = "uv pip install 'aegis-ml[dashboard]'"
            return state

        store_uri, artifacts = mlflow_store_uri(self.registry_dir)
        state.log_path = str(self._log_file(MLFLOW_KEY))
        argv = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            store_uri,
            "--default-artifact-root",
            artifacts.as_uri(),
            "--host",
            self.host,
            "--port",
            str(chosen),
            # SAMEORIGIN — MLflow's default — blocks the hub, because a different port is
            # a different origin. NONE drops the header so the panel can embed the real UI
            # rather than degrading to a link on a purely local, loopback-bound server.
            "--x-frame-options",
            "NONE",
        ]
        state._process = self._spawn(MLFLOW_KEY, argv, self.registry_dir)  # noqa: SLF001
        state.pid = state._process.pid  # noqa: SLF001
        state.managed = True
        self._await_ready(state, time.monotonic() + _STARTUP_GRACE)
        return state

    def start_optuna(self, port: int | None, default_port: int) -> ServiceState:
        """Bring up Optuna Dashboard against the persisted study database.

        Args:
            port: The explicitly requested port, or ``None``.
            default_port: The port to prefer when none was requested.

        Returns:
            A :class:`ServiceState`, running or carrying its reason.
        """
        chosen, refusal = self._resolve_port(port, default_port, "optuna-port")
        state = ServiceState(
            key=OPTUNA_KEY,
            label="Optuna Dashboard",
            blurb=(
                "Every hyper-parameter trial this repository has run — including the "
                "pruned ones — with importance, history and parallel-coordinate views."
            ),
            port=chosen,
            url=f"http://{self._display_host()}:{chosen}/",
            identity_path="/api/meta",
        )
        self.states[OPTUNA_KEY] = state
        if refusal is not None:
            state.reason, state.remedy = refusal
            return state
        if port_in_use(self.host, chosen):
            if self._adopt_existing(state):
                return state
            return self._port_held_by_stranger(state, "optuna-port")

        storage_uri, db_path = optuna_storage_uri(self.registry_dir)
        if not is_available("optuna_dashboard"):
            state.reason = (
                "optuna-dashboard is not importable in this interpreter. The studies "
                "themselves are intact in the database below — only the viewer is missing."
            )
            state.remedy = "uv pip install 'aegis-ml[dashboard]'"
            return state
        if not db_path.is_file():
            state.reason = (
                f"No Optuna study database at {db_path}. Nothing has been searched yet, so "
                f"there is nothing to show — this is an empty registry, not a broken panel."
            )
            state.remedy = "aegis-ml train --adapter <module>   # persists studies as it searches"
            return state

        state.log_path = str(self._log_file(OPTUNA_KEY))
        argv = [
            sys.executable,
            "-c",
            "from optuna_dashboard._cli import main; main()",
            storage_uri,
            "--host",
            self.host,
            "--port",
            str(chosen),
        ]
        state._process = self._spawn(OPTUNA_KEY, argv, self.registry_dir)  # noqa: SLF001
        state.pid = state._process.pid  # noqa: SLF001
        state.managed = True
        self._await_ready(state, time.monotonic() + _STARTUP_GRACE)
        return state

    def skip(self, key: str, label: str, blurb: str, port: int, why: str) -> ServiceState:
        """Record a service the caller switched off, so the panel says so rather than lying.

        Args:
            key: :data:`MLFLOW_KEY` or :data:`OPTUNA_KEY`.
            label: Human name.
            blurb: What the service would have shown.
            port: The port it would have used.
            why: The flag that switched it off, in words.

        Returns:
            A stopped :class:`ServiceState` whose reason is the caller's own choice.
        """
        state = ServiceState(
            key=key,
            label=label,
            blurb=blurb,
            port=port,
            url=f"http://{self._display_host()}:{port}/",
            reason=why,
            remedy="aegis-ml dashboard   # without that flag",
        )
        self.states[key] = state
        return state

    def _resolve_port(
        self, requested: int | None, default_port: int, flag: str
    ) -> tuple[int, tuple[str, str] | None]:
        """Pick the port to bind, or explain why no port could be picked.

        Args:
            requested: What the caller typed, or ``None``.
            default_port: The documented default for this service.
            flag: The CLI flag name, quoted back in the remedy.

        Returns:
            ``(port, refusal)`` where ``refusal`` is ``(reason, remedy)`` or ``None``.
        """
        if requested is not None:
            return requested, None
        if not port_in_use(self.host, default_port):
            return default_port, None
        alternative = free_port_from(self.host, default_port + 1)
        if alternative is None:
            return default_port, (
                f"Port {default_port} is in use and so are the twelve ports after it, so "
                f"no address was available to bind.",
                f"aegis-ml dashboard --{flag} <a free port>",
            )
        return alternative, None

    def refresh(self) -> dict[str, ServiceState]:
        """Re-probe every service and return the updated states.

        Called by ``/api/services.json`` on the hub's polling interval, so a service that
        dies mid-demo turns its indicator red instead of leaving a stale green dot next to
        a frame that stopped loading.
        """
        for state in self.states.values():
            process = state._process  # noqa: SLF001
            if process is not None and process.poll() is not None:
                state.running = False
                state.reason = (
                    f"{state.label} exited with code {process.returncode} after starting."
                )
                state.remedy = f"cat {state.log_path}"
                continue
            if not state.managed and not state.running and state.reason:
                continue
            result = probe(state.url, self.page_origin, timeout=1.0)
            state.running = result.up
            if result.up:
                state.reason = ""
                state.embeddable = not result.frame_blocked
                state.frame_reason = result.frame_reason
        return self.states

    def _signal_group(self, process: subprocess.Popen[bytes], sig: int) -> None:
        """Send ``sig`` to the child's whole process group, falling back to the child.

        Args:
            process: The child, started with ``start_new_session=True`` and therefore the
                leader of its own process group.
            sig: ``SIGTERM`` or ``SIGKILL``.

        MLflow 3's server is not one process. It forks gunicorn workers, a job runner and
        a set of queue consumers, and signalling only the process we hold a handle to
        leaves every one of those alive — still holding the port, so the *next*
        ``aegis-ml dashboard`` finds it busy and degrades for no visible reason. Signalling
        the group is the only thing that reaps the tree, and it is why the children are
        spawned into their own session in the first place.
        """
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(process.pid), sig)
            return
        with contextlib.suppress(ProcessLookupError, OSError):
            process.send_signal(sig)

    def shutdown(self, grace: float = 6.0) -> None:
        """Terminate every child this supervisor started, then close the log handles.

        Args:
            grace: Seconds to wait for a polite ``SIGTERM`` before ``SIGKILL``.

        Idempotent: safe to call from both the context manager's exit and an explicit
        cleanup path, which is what makes a Ctrl-C during startup as clean as one during
        ``serve_forever``.
        """
        for state in self.states.values():
            process = state._process  # noqa: SLF001
            if process is None or process.poll() is not None:
                continue
            self._signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self._signal_group(process, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=grace)
            state.running = False
        for handle in self._logs:
            with contextlib.suppress(OSError, ValueError):
                handle.close()
        self._logs.clear()

    def __enter__(self) -> Supervisor:
        """Return self; children are started by the explicit ``start_*`` calls."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Terminate children on every exit path, including ``KeyboardInterrupt``."""
        self.shutdown()


@contextlib.contextmanager
def supervise(host: str, page_port: int, registry_dir: Path) -> Iterator[Supervisor]:
    """Yield a :class:`Supervisor` that terminates its children when the block ends.

    Args:
        host: Bind address for the hub and the services.
        page_port: The hub's port, used as the origin for frame-header checks.
        registry_dir: Registry root.

    Yields:
        The supervisor, with no services started yet.
    """
    supervisor = Supervisor(host=host, page_port=page_port, registry_dir=registry_dir)
    try:
        yield supervisor
    finally:
        supervisor.shutdown()

"""The local HTTP server: the hub page, its live state, and the run directory's own files.

Built on :mod:`http.server` rather than FastAPI, and that is a deliberate downgrade in
sophistication. This process must start in under a second on a machine where the only
certainty is that Python 3.11 exists, it must have no dependency that the ``serve`` extra's
caps could refuse, and it must be readable by whoever inherits it. It serves a handful of
files to one browser on loopback; an ASGI stack would add a dependency, a startup cost and
an event loop to a problem that has none of those.

What it serves, and nothing else:

``/``
    The hub, rebuilt from the registry on every request. Rebuilding rather than caching is
    what makes a training run that finishes mid-demo appear on the next refresh — a cached
    page would show yesterday's champion next to a green "live" dot, which is precisely the
    kind of confident-and-wrong that this repository is written against.
``/api/state.json``
    The same payload the page embeds, for anything that wants to read it programmatically.
``/api/services.json``
    Re-probed MLflow and Optuna Dashboard status, polled by the page every five seconds.
``/runs/<run_id>/files/<path>``
    Static bytes from ``registry_store/runs/<run_id>/`` — the nine PNGs, ``card.html``,
    ``shap.html``, ``profile.html``, ``visuals/interactive.html``. Served from the registry
    rather than copied into the page, so what a viewer opens is the artifact itself.
``/reports/<name>``
    The Evidently drift reports under ``registry_store/reports/``.
``/healthz``
    A single JSON line, so a script can wait for the hub instead of sleeping.

**Every path is resolved and re-checked against its root before a byte is read.** The run
id is additionally required to be a single path component. This binds to loopback by
default and the artifacts are model cards and dataset digests rather than secrets, but a
path-traversal hole in a server that ships next to a registry is not a defect anyone wants
to explain afterwards.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Mapping

__all__ = ["HubServer", "build_server"]

_LOG = logging.getLogger(__name__)

_TEXT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".svg": "image/svg+xml",
}
"""Content types we state explicitly.

``mimetypes`` is driven by the host's registry, which on a stripped container can map
``.json`` to ``application/octet-stream`` — and a browser then downloads the leaderboard
instead of showing it. The types that matter to this page are pinned here; everything else
falls through to :mod:`mimetypes` and finally to ``application/octet-stream``.
"""

_CHUNK = 1 << 16
"""Copy buffer. ``interactive.html`` is several megabytes; streaming it keeps the server's
memory flat regardless of how large a run's bundle grows."""


class HubServer(ThreadingHTTPServer):
    """A threading HTTP server that closes its socket promptly on shutdown.

    Threaded because one browser opens several requests at once — the page, the PNGs in the
    gallery, and the service poll — and a single-threaded server serialises them behind the
    slowest one, which on a first load is a multi-megabyte figure.

    ``daemon_threads`` means a Ctrl-C is not held up by an in-flight download of a 4 MB
    interactive report, and ``allow_reuse_address`` means the *next* invocation is not
    refused by a socket still in ``TIME_WAIT`` from this one.
    """

    daemon_threads = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    """Routes the six paths above and refuses everything else with a readable message."""

    server_version = "aegis-ml-hub"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    page_source: Callable[[], str]
    state_source: Callable[[], Mapping[str, Any]]
    services_source: Callable[[], Mapping[str, Any]]
    runs_root: Path
    reports_root: Path

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401 - stdlib signature
        """Send request logs to the logger instead of stderr.

        The terminal running this command shows a banner with three URLs on it, and that
        banner is what someone reads out loud during a demo. Two hundred lines of access
        log scrolling it away is not an acceptable default; ``-v`` on the logger brings it
        back for anyone debugging.
        """
        _LOG.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        """Write one complete response with an accurate ``Content-Length``."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:  # noqa: ANN401 - any JSON
        """Serialise ``payload`` and send it as ``application/json``."""
        body = json.dumps(payload, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _refuse(self, status: int, message: str) -> None:
        """Answer with a short, readable explanation rather than the stdlib's HTML page.

        Args:
            status: The HTTP status.
            message: What was wrong with the request, in a sentence.
        """
        self._send(status, message.encode("utf-8"), "text/plain; charset=utf-8")

    def _resolve(self, root: Path, relative: str) -> Path | None:
        """Resolve ``relative`` under ``root``, or return ``None`` if it escapes.

        Args:
            root: The directory the request is confined to.
            relative: The user-supplied, already URL-decoded path fragment.

        Returns:
            An existing regular file inside ``root``, or ``None``.

        The check is done after :meth:`~pathlib.Path.resolve` so that symlinks are followed
        *before* containment is judged. Comparing the unresolved strings would let a symlink
        inside the run directory point anywhere on the filesystem and still look contained.
        """
        if relative.startswith("/") or "\x00" in relative:
            return None
        try:
            candidate = (root / relative).resolve()
            base = root.resolve()
        except OSError:
            return None
        if not candidate.is_relative_to(base) or not candidate.is_file():
            return None
        return candidate

    def _content_type(self, path: Path) -> str:
        """Return the content type for a file, preferring the pinned table."""
        suffix = path.suffix.lower()
        if suffix in _TEXT_TYPES:
            return _TEXT_TYPES[suffix]
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    def _send_file(self, path: Path) -> None:
        """Stream a file to the client without reading it into memory."""
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", self._content_type(path))
        self.send_header("Content-Length", str(size))
        # Run artifacts are immutable once registered, so the browser may keep them for the
        # session. The hub page itself is `no-store` — that one must never be stale.
        self.send_header("Cache-Control", "private, max-age=300")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile, _CHUNK)

    def _serve_run_file(self, parts: list[str]) -> None:
        """Serve ``/runs/<run_id>/files/<path>`` from the run directory."""
        min_parts = 4
        if len(parts) < min_parts or parts[2] != "files":
            self._refuse(404, "Expected /runs/<run_id>/files/<path>.")
            return
        run_id = parts[1]
        if run_id != Path(run_id).name or run_id in {"", ".", ".."}:
            self._refuse(400, "A run id is a single path component.")
            return
        target = self._resolve(self.runs_root / run_id, "/".join(parts[3:]))
        if target is None:
            self._refuse(404, f"No such artifact under run {run_id}.")
            return
        self._send_file(target)

    def _serve_report(self, parts: list[str]) -> None:
        """Serve ``/reports/<name>`` from the registry's reports directory."""
        target = self._resolve(self.reports_root, "/".join(parts[1:]))
        if target is None:
            self._refuse(404, "No such report.")
            return
        self._send_file(target)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's dispatch name
        """Route one request. Unknown paths get a 404 that names the paths that exist."""
        path = unquote(urlparse(self.path).path)
        parts = [p for p in path.split("/") if p]

        try:
            if path in {"/", "/index.html"}:
                self._send(200, self.page_source().encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/healthz":
                self._send_json({"service": "aegis-ml-hub", "ok": True})
                return
            if path == "/api/state.json":
                self._send_json(self.state_source())
                return
            if path == "/api/services.json":
                self._send_json(self.services_source())
                return
            if parts and parts[0] == "runs":
                self._serve_run_file(parts)
                return
            if parts and parts[0] == "reports":
                self._serve_report(parts)
                return
            self._refuse(
                404,
                "Not a route on this server. Available: /, /healthz, /api/state.json, "
                "/api/services.json, /runs/<run_id>/files/<path>, /reports/<name>.",
            )
        except BrokenPipeError:
            # The browser navigated away mid-transfer. Nothing is wrong with the server and
            # nothing is wrong with the file; re-raising would print a traceback that reads
            # like a crash over the banner someone is trying to read.
            _LOG.debug("client disconnected while receiving %s", path)
        except OSError as exc:
            _LOG.warning("failed to serve %s: %s", path, exc)
            self._refuse(500, f"Could not read the artifact behind {path}: {exc}")

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's dispatch name
        """Answer HEAD through the same routing, so a probe cannot disagree with a GET."""
        self.do_GET()


def build_server(
    *,
    host: str,
    port: int,
    page_source: Callable[[], str],
    state_source: Callable[[], Mapping[str, Any]],
    services_source: Callable[[], Mapping[str, Any]],
    runs_root: Path,
    reports_root: Path,
) -> HubServer:
    """Construct the hub's HTTP server, bound and ready for ``serve_forever``.

    Args:
        host: Bind address. The CLI defaults it to loopback.
        port: Bind port.
        page_source: Returns the hub's HTML. Called per request, so the page reflects the
            registry as it is now rather than as it was when the command started.
        state_source: Returns the JSON payload behind ``/api/state.json``.
        services_source: Returns freshly probed service states for ``/api/services.json``.
        runs_root: ``registry_store/runs``; the only directory ``/runs/`` can reach.
        reports_root: ``registry_store/reports``; the only directory ``/reports/`` can reach.

    Returns:
        A bound :class:`HubServer`. The caller owns its lifetime and must call
        ``server_close``.

    Raises:
        OSError: When the port cannot be bound — raised rather than swallowed, because the
            caller has already checked the port and a failure here means something took it
            in between, which the operator needs to be told about.

    The callables are attached to the handler class rather than captured in a closure so
    that the handler stays a plain class: :mod:`http.server` instantiates it per request,
    and a subclass with class attributes is both cheaper and easier to read than a factory.
    """
    runs = Path(runs_root)
    reports = Path(reports_root)
    runs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    handler = type(
        "_BoundHubHandler",
        (_Handler,),
        {
            "page_source": staticmethod(page_source),
            "state_source": staticmethod(state_source),
            "services_source": staticmethod(services_source),
            "runs_root": runs,
            "reports_root": reports,
        },
    )
    return HubServer((host, port), handler)

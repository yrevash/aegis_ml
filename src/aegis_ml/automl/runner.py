"""The subprocess bridge: run the AutoML search in the trainer venv, get JSON back.

This is decision D1 made executable. The backend venv carries hard caps —
``pandas>=2.2,<2.4`` (nemoguardrails), ``numpy>=1.26,<2.5`` (presidio, and numba/llvmlite
via shap), ``numba==0.67.0`` — and AutoGluon 1.6 + TabPFN-2.5 + torch will not resolve
under them. Installing them into the backend venv is the single most likely way to lose a
hackathon morning, so they live in ``settings.trainer_venv`` and are reached the only way
two incompatible dependency graphs can safely talk: a process boundary with a serialised
contract across it.

**What crosses, and why it is only these three things.** Parquet in (the frame), JSON in
(the problem and the search request), JSON out (the recipe and the leaderboard). Never a
pickle, in either direction: ``joblib.load`` of a model built against a different
numpy/sklearn is either an exception or — worse — a silently wrong object, and the whole
point of the split is that the two sides do *not* share those libraries. The recipe is
constructor kwargs, which are version-independent by construction.

**Why stderr is streamed rather than captured.** An AutoGluon fit prints its progress and
its failures to stderr, and a search that dies twelve minutes in with a captured-and-
discarded traceback is undebuggable at exactly the moment you cannot afford it. Every line
the child writes appears live on this process's stderr, and the tail is retained so the
raised error carries the last thing the child said.

**Why there is a timeout.** ``time_budget`` bounds what FLAML and AutoGluon *intend* to
spend, not what they do: a hung fit, a torch import stuck on a network mount, or a worker
deadlock will otherwise wait forever holding the pipeline open. The guard kills the child
and raises; it never returns a partial result as if it were a complete one.
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 - the whole module exists to run one known interpreter
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import aegis_ml
from aegis_ml._require import require
from aegis_ml.contracts.errors import AegisMLError, TrainerVenvMissingError
from aegis_ml.contracts.protocols import Leaderboard, Recipe, TierName
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import light
    import pandas as pd

__all__ = [
    "ERROR_FILENAME",
    "FRAME_FILENAME",
    "LEADERBOARD_FILENAME",
    "RECIPE_FILENAME",
    "REQUEST_FILENAME",
    "STDERR_TAIL_LINES",
    "run_in_trainer_venv",
    "trainer_available",
    "trainer_python",
]

FRAME_FILENAME = "frame.parquet"
"""The training frame, written as parquet.

Parquet and not CSV: a CSV round-trip loses dtypes, and a categorical column that comes
back as ``object`` while a boolean comes back as the string ``"True"`` changes what the
search encodes — the trainer venv would be searching over a subtly different problem than
the one the serving venv validated.
"""

REQUEST_FILENAME = "request.json"
"""The problem spec plus the search parameters (tiers, budget, seed)."""

RECIPE_FILENAME = "recipe.json"
"""The portable recipe the worker produced — the answer that crosses back."""

LEADERBOARD_FILENAME = "leaderboard.json"
"""Every candidate the worker scored, including the losers and the skipped tiers."""

ERROR_FILENAME = "error.json"
"""Written by the worker when the search raises, carrying the full child traceback."""

STDERR_TAIL_LINES = 200
"""How many of the child's last stderr lines are retained for the raised error message."""


def trainer_python() -> Path:
    """Return the trainer venv's interpreter, or raise naming the command that builds it.

    Returns:
        Path to the interpreter inside ``settings.trainer_venv``.

    Raises:
        TrainerVenvMissingError: If the interpreter does not exist. Its message carries the
            two ``uv`` commands that create and populate the venv, because "trainer venv
            missing" without them sends the reader to the docs mid-demo.
    """
    interpreter = settings.trainer_python
    if not interpreter.exists():
        raise TrainerVenvMissingError(str(settings.trainer_venv))
    return interpreter


def trainer_available() -> bool:
    """Return whether the trainer venv exists, for capability reporting only.

    ``aegis-ml doctor`` prints this. Nothing branches on it silently: a caller that needs
    the trainer venv calls :func:`trainer_python` and gets the typed refusal.

    Returns:
        ``True`` if the trainer interpreter is present.
    """
    return settings.trainer_python.exists()


def _child_env() -> dict[str, str]:
    """Return the child's environment: ours, plus this checkout on ``PYTHONPATH``.

    The trainer venv is normally an editable install of this package, but it need not be —
    a venv built only for the heavy wheels still has to import ``aegis_ml.automl._worker``.
    Prepending this checkout's ``src`` makes the worker importable either way, and putting
    it *first* means the code that runs in the child is the code you are editing, not a
    stale wheel.
    """
    env = dict(os.environ)
    src_root = str(Path(aegis_ml.__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing}" if existing else src_root
    # Line-buffered child output, so a long AutoGluon fit's progress appears while it runs
    # instead of arriving in one block after it finishes (or never, if it is killed).
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _default_timeout(time_budget: int) -> int:
    """Return the wall-clock ceiling for the child process.

    Four times the per-tier budget plus ten minutes of slack. The multiplier is the tier
    count: FLAML and AutoGluon each take ``time_budget`` and the other two tiers are
    unbudgeted, so a search that honours its budgets stays well inside this. The slack
    covers the one cost no budget accounts for — importing torch and loading TabPFN's
    weights on a cold filesystem.
    """
    return max(4 * int(time_budget) + 600, 900)


def _stream_stderr(stream: object, tail: deque[str]) -> None:
    """Echo the child's stderr to ours line by line, retaining the tail.

    Runs on a thread so a chatty child cannot fill its pipe buffer and deadlock while the
    parent waits on ``proc.wait()`` — the classic subprocess hang, and one that would look
    exactly like "AutoGluon is slow".
    """
    for raw in stream:  # type: ignore[attr-defined]
        line = raw.rstrip("\n")
        tail.append(line)
        print(f"[trainer] {line}", file=sys.stderr, flush=True)


def run_in_trainer_venv(
    frame: pd.DataFrame,
    problem: MLProblem,
    *,
    tiers: list[TierName] | tuple[TierName, ...] | None = None,
    time_budget: int | None = None,
    seed: int | None = None,
    timeout: int | None = None,
    workdir: str | Path | None = None,
) -> tuple[Recipe, Leaderboard]:
    """Run :func:`aegis_ml.automl.search.run_search` in the trainer venv and read it back.

    Args:
        frame: The training frame; written to parquet for the child.
        problem: The spec; written to JSON for the child.
        tiers: Tiers to attempt, or ``None`` for all four. The child skips the ones it
            cannot run *with a reason*, which arrives in ``Leaderboard.tiers_skipped``.
        time_budget: Per-budgeted-tier wall clock; defaults to
            ``settings.automl_time_budget``.
        seed: Split and estimator seed; defaults to ``settings.random_seed``.
        timeout: Hard ceiling on the child process. Defaults to :func:`_default_timeout`.
        workdir: Directory for the exchange files. When given it is *kept* after the run,
            which is how you inspect a failing search's inputs; when omitted a temporary
            directory is used and removed.

    Returns:
        ``(recipe, leaderboard)`` — the same pair :func:`aegis_ml.automl.search.run_search`
        returns in-process, having crossed the venv boundary as JSON.

    Raises:
        TrainerVenvMissingError: If the trainer interpreter is absent.
        AegisMLError: If the child exits non-zero (carrying its traceback), is killed by
            the timeout, or exits zero without writing both result files.
        ImportError: If pyarrow is missing, so the frame cannot be written.
    """
    interpreter = trainer_python()
    require("aegis-ml[serve]", "pyarrow")  # frame.to_parquet's engine, named before use
    time_budget = settings.automl_time_budget if time_budget is None else time_budget
    seed = settings.random_seed if seed is None else seed
    timeout = _default_timeout(time_budget) if timeout is None else timeout

    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="aegis_ml_trainer_") as tmp:
            return _run(interpreter, Path(tmp), frame, problem, tiers, time_budget, seed, timeout)
    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    return _run(interpreter, directory, frame, problem, tiers, time_budget, seed, timeout)


def _run(  # noqa: PLR0913 - every argument is already resolved; bundling them would hide them
    interpreter: Path,
    directory: Path,
    frame: pd.DataFrame,
    problem: MLProblem,
    tiers: list[TierName] | tuple[TierName, ...] | None,
    time_budget: int,
    seed: int,
    timeout: int,
) -> tuple[Recipe, Leaderboard]:
    """Write the inputs, run the child, and read the results back."""
    frame.to_parquet(directory / FRAME_FILENAME, index=False)
    request = {
        "problem": problem.model_dump(mode="json"),
        "tiers": list(tiers) if tiers is not None else None,
        "time_budget": int(time_budget),
        "seed": int(seed),
    }
    (directory / REQUEST_FILENAME).write_text(json.dumps(request, indent=2), encoding="utf-8")

    command = [str(interpreter), "-m", "aegis_ml.automl._worker", str(directory)]
    print(
        f"[aegis-ml] AutoML search in trainer venv: {' '.join(command)} "
        f"(timeout {timeout}s, budget {time_budget}s/tier)",
        file=sys.stderr,
        flush=True,
    )

    tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
    process = subprocess.Popen(  # noqa: S603 - argv is built here from a validated path
        command,
        env=_child_env(),
        stdout=None,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    killed: list[str] = []

    def _kill() -> None:
        """Kill the child and record that the timeout, not the child, ended the run."""
        killed.append("timeout")
        process.kill()

    guard = threading.Timer(timeout, _kill)
    guard.start()
    reader = threading.Thread(target=_stream_stderr, args=(process.stderr, tail), daemon=True)
    reader.start()
    try:
        returncode = process.wait()
    finally:
        guard.cancel()
        reader.join(timeout=5)
        if process.stderr is not None:
            process.stderr.close()

    if killed:
        raise AegisMLError(
            f"AutoML search in the trainer venv exceeded its {timeout}s ceiling and was "
            f"killed. Lower --time-budget, drop the heaviest tier, or raise the timeout — "
            f"a partial search is never returned as a complete one.\n"
            f"Last output:\n  " + "\n  ".join(tail)
        )
    if returncode != 0:
        raise AegisMLError(_failure_message(directory, returncode, tail))

    recipe_path = directory / RECIPE_FILENAME
    leaderboard_path = directory / LEADERBOARD_FILENAME
    missing = [p.name for p in (recipe_path, leaderboard_path) if not p.exists()]
    if missing:
        raise AegisMLError(
            f"The trainer venv's search exited 0 but wrote no {missing}. Treating a "
            f"missing result as an empty one would publish a leaderboard that says the "
            f"strong tiers found nothing.\nLast output:\n  " + "\n  ".join(tail)
        )
    recipe = Recipe.model_validate_json(recipe_path.read_text(encoding="utf-8"))
    leaderboard = Leaderboard.model_validate_json(leaderboard_path.read_text(encoding="utf-8"))
    return recipe, leaderboard


def _failure_message(directory: Path, returncode: int, tail: deque[str]) -> str:
    """Compose the error for a non-zero child, preferring the traceback it wrote itself."""
    error_path = directory / ERROR_FILENAME
    detail = ""
    if error_path.exists():
        payload = json.loads(error_path.read_text(encoding="utf-8"))
        detail = (
            f"\nChild raised {payload.get('type', '?')}: {payload.get('message', '')}\n"
            f"{payload.get('traceback', '')}"
        )
    return (
        f"AutoML search in the trainer venv failed with exit code {returncode}. The "
        f"serving venv's own tiers were not run, so there is no partial result to fall "
        f"back on — re-run with `--tier baseline` for an in-process search that does not "
        f"need the trainer venv.{detail}\nLast output:\n  " + "\n  ".join(tail)
    )

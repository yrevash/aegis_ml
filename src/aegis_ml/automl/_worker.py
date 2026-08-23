"""The search, executed inside the trainer venv. Run as ``python -m aegis_ml.automl._worker``.

This is the far side of the process boundary described in :mod:`aegis_ml.automl.runner`.
It is a module rather than a generated script so that it is linted, type-checked and
reviewed like everything else — a bridge whose remote half is a string of Python built at
call time is a bridge nobody can read.

Its contract is deliberately tiny, because everything it does happens where the parent
cannot see it:

* read ``request.json`` and ``frame.parquet`` from the directory named on the command line;
* run :func:`aegis_ml.automl.search.search` with exactly the arguments requested;
* write ``recipe.json`` and ``leaderboard.json`` back into the same directory;
* on any failure, write ``error.json`` with the full traceback and exit non-zero.

**Why it writes ``error.json`` rather than only printing.** stderr is streamed and tailed
by the parent, but a torch or AutoGluon crash can emit thousands of lines and push the
actual traceback out of the tail. The file survives that.

**Why it never falls back.** If AutoGluon is missing here, that is a fact about the trainer
venv the leaderboard must report (``tiers_skipped``), not something to paper over by
quietly running the baseline and returning a recipe that says ``tier="autogluon"``. The
search function already enforces this; the worker adds nothing on top.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from aegis_ml.automl.runner import (
    ERROR_FILENAME,
    FRAME_FILENAME,
    LEADERBOARD_FILENAME,
    RECIPE_FILENAME,
    REQUEST_FILENAME,
)
from aegis_ml.automl.search import search
from aegis_ml.automl.tiers import TABPFN_LICENSE_NOTICE, tier_status
from aegis_ml.contracts.spec import MLProblem

__all__ = ["main"]


def _log(message: str) -> None:
    """Write one line to stderr, which the parent streams live and retains a tail of."""
    print(message, file=sys.stderr, flush=True)


def _run(directory: Path) -> int:
    """Execute one search in this interpreter and write its two result files.

    Args:
        directory: The exchange directory prepared by
            :func:`aegis_ml.automl.runner.run_in_trainer_venv`.

    Returns:
        Process exit code: 0 on success.
    """
    import pandas as pd  # noqa: PLC0415 - heavy import, and only the child needs it

    request = json.loads((directory / REQUEST_FILENAME).read_text(encoding="utf-8"))
    problem = MLProblem.model_validate(request["problem"])
    frame = pd.read_parquet(directory / FRAME_FILENAME)

    _log(f"aegis-ml worker: python {sys.version.split()[0]} at {sys.executable}")
    for tier, status in tier_status().items():
        _log(f"aegis-ml worker: tier {tier:<10} {status}")
    if tier_status().get("tabpfn") == "available":
        _log(f"aegis-ml worker: {TABPFN_LICENSE_NOTICE}")
    _log(
        f"aegis-ml worker: searching {len(frame)} rows × {len(problem.features)} features, "
        f"target {problem.target.name!r} ({problem.target.task}), metric {problem.metric!r}"
    )

    recipe, leaderboard = search(
        frame,
        problem,
        tiers=request.get("tiers"),
        time_budget=request.get("time_budget"),
        seed=request.get("seed"),
    )

    (directory / RECIPE_FILENAME).write_text(recipe.model_dump_json(indent=2), encoding="utf-8")
    (directory / LEADERBOARD_FILENAME).write_text(
        leaderboard.model_dump_json(indent=2), encoding="utf-8"
    )

    winner = next((c for c in leaderboard.candidates if c.selected), None)
    if winner is None:
        selection = f"no portable candidate; fell back to the {recipe.tier} recipe, see its notes"
    else:
        selection = (
            f"selected {winner.name!r} ({leaderboard.metric_name}={winner.metric_value:.4f})"
        )
    _log(
        f"aegis-ml worker: {len(leaderboard.candidates)} candidates across "
        f"{leaderboard.tiers_run}; {selection}"
    )
    for tier, reason in sorted(leaderboard.tiers_skipped.items()):
        _log(f"aegis-ml worker: skipped {tier}: {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m aegis_ml.automl._worker <exchange-dir>``.

    Args:
        argv: Command-line arguments after the module name; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a usage error, 1 if the search raised (with ``error.json``
        written next to the inputs).
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        _log("usage: python -m aegis_ml.automl._worker <exchange-dir>")
        return 2
    directory = Path(args[0])
    if not directory.is_dir():
        _log(f"exchange directory {directory} does not exist")
        return 2

    try:
        return _run(directory)
    except Exception as exc:  # audit-ok: the traceback is written out, never swallowed
        payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (directory / ERROR_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess, not by import
    raise SystemExit(main())

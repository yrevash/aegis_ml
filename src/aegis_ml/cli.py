"""``aegis-ml`` — the command line, wired to the real functions.

Every command here calls the same code the library exposes; none of them has a private
path, a demo mode or a shortcut. ``aegis-ml train`` runs
:func:`aegis_ml.pipelines.flows.train_flow` and nothing else, so a number printed at the
terminal and a number in the registry cannot disagree.

``doctor`` is the command this file exists for. It is the first thing run on hackathon
morning and it answers, in one screen, every question that otherwise costs an hour:

* which Python, which ``aegis_ml``, and is ``aegis`` importable — **and from where**, because
  two checkouts on one machine is how a fix gets applied to the wrong tree;
* the *resolved* version of every library whose API this package depends on, since
  "evidently is installed" and "evidently 0.7's ``Report`` API is available" are different
  facts and only the second one matters;
* which AutoML tiers will actually run, **and the reason each unavailable one will not** —
  because a leaderboard missing AutoGluon and a leaderboard where AutoGluon lost look
  identical, and one of them means "install it";
* the trainer venv, the artifact path the backend actually loads from, and whether the
  directory is writable — a training run that discovers this at the end has spent the budget;
* Postgres reachability when a DSN is configured;
* the TabPFN licence notice whenever that tier is enabled;
* the **realism band**, and whether the registered reference frame falls inside it. This is
  the fastest way on the day to tell a broken generator from a working one: a held-out score
  under the floor means the label is noise, and one over the ceiling means the generator
  forgot the noise, the confounders and the missingness — the model will look perfect here
  and collapse on the first real frame.

It exits non-zero when something essential is broken, so it can gate a Makefile target.
"""

# ruff: noqa: B008 - `typer.Option(...)` in a parameter default is Typer's declaration
# syntax, not an accidental call-at-import: Typer reads the OptionInfo object off the
# signature to build the parser. Hoisting them to module-level singletons, which is what
# B008 asks for, would scatter every command's help text away from the command.

from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from aegis_ml import __version__
from aegis_ml._require import is_available
from aegis_ml.contracts.spec import MLProblem
from aegis_ml.settings import settings

__all__ = ["app", "main"]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SOTA ML/MLOps adapter factory for the Aegis agentic-AI platform.",
)

#: Import name → distribution name, for libraries where the two differ. Version is read
#: from installed metadata rather than a ``__version__`` attribute: metadata is what the
#: resolver actually installed, and a stale ``__version__`` in a shadowed copy is exactly
#: the situation ``doctor`` exists to expose.
_DISTRIBUTIONS: dict[str, str] = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "shap": "shap",
    "mapie": "mapie",
    "pandera": "pandera",
    "skrub": "skrub",
    "optuna": "optuna",
    "flaml": "flaml",
    "evidently": "evidently",
    "nannyml": "nannyml",
    "joblib": "joblib",
    "pyarrow": "pyarrow",
    "statsforecast": "statsforecast",
    "mlforecast": "mlforecast",
    "lightgbm": "lightgbm",
    "autogluon.tabular": "autogluon.tabular",
    "tabpfn": "tabpfn",
    "onnxruntime": "onnxruntime",
    "prefect": "prefect",
    "mlflow": "mlflow",
    "fastapi": "fastapi",
}

#: The floor this package is written against. Held as a constant rather than an inline
#: literal so the check still runs when `aegis-ml` is invoked by an older interpreter that
#: a `requires-python` metadata bound never got the chance to reject.
_MIN_PYTHON = (3, 11)

#: Libraries without which nothing in this package can run. Their absence is a non-zero exit.
_ESSENTIAL = ("pandas", "numpy", "sklearn", "joblib")

TABPFN_NOTICE = (
    "TabPFN-2.5 weights are distributed under the Prior Labs License: research and "
    "EVALUATION use are permitted; commercial and production use are NOT. A demo is "
    "evaluation use, which is why this tier is on by default — every model card it touches "
    "prints this notice. Set AEGIS_ML_ENABLE_TABPFN=0 to switch it off."
)


def _echo(text: str = "") -> None:
    """Write one line to stdout."""
    typer.echo(text)


def _fail(message: str, code: int = 1) -> None:
    """Print an error to stderr and exit non-zero.

    Args:
        message: What went wrong, and where possible what fixes it.
        code: Process exit code.

    Raises:
        typer.Exit: Always.
    """
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _version_of(import_name: str) -> str | None:
    """Return the installed version of a library, or ``None`` when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    if not is_available(import_name.split(".")[0]):
        return None
    try:
        return version(_DISTRIBUTIONS.get(import_name, import_name))
    except PackageNotFoundError:
        return "installed (no metadata)"


def _load_problem(problem: Path | None, adapter: str | None) -> MLProblem:
    """Load the :class:`~aegis_ml.contracts.spec.MLProblem` a command operates on.

    Args:
        problem: Path to a JSON file holding a serialised ``MLProblem``.
        adapter: Dotted module path exposing ``PROBLEM`` or ``ML_PROBLEM``.

    Returns:
        The problem.

    Raises:
        typer.Exit: When neither is supplied, or the module exposes no problem — a command
            that guessed the target column would fit a model on the wrong thing and report
            a perfectly formatted metric about it.
    """
    if problem is not None:
        return MLProblem.model_validate_json(Path(problem).read_text(encoding="utf-8"))
    if adapter is not None:
        import importlib

        module = importlib.import_module(adapter)
        for attribute in ("PROBLEM", "ML_PROBLEM", "problem"):
            found = getattr(module, attribute, None)
            if isinstance(found, MLProblem):
                return found
        _fail(
            f"module {adapter!r} exposes no MLProblem under PROBLEM / ML_PROBLEM / problem. "
            f"Generate one with `aegis-ml init` and pass it with --problem."
        )
    _fail("pass --problem <problem.json> or --adapter <module exposing PROBLEM>.")
    raise AssertionError("unreachable")  # pragma: no cover - _fail always raises


def _load_frame(path: Path) -> Any:  # noqa: ANN401 - a pandas.DataFrame
    """Read a ``.csv`` or ``.parquet`` frame, or exit naming the supported formats."""
    from aegis_ml._require import require

    pd = require("aegis-ml[serve]", "pandas")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    _fail(f"unsupported data file {path.suffix!r}; use .csv or .parquet")
    raise AssertionError("unreachable")  # pragma: no cover - _fail always raises


# ────────────────────────────────────────────────────────────────────────── doctor ──


def _tier_report() -> list[tuple[str, bool, str]]:
    """Return ``(tier, available, reason)`` for each AutoML tier.

    A tier is available only when its dependency imports **and** its settings flag is on.
    Both halves are reported, because "AutoGluon is installed but AEGIS_ML_ENABLE_AUTOGLUON=0"
    and "AutoGluon is not installed" need different fixes and produce the same empty slot on
    a leaderboard.
    """
    rows: list[tuple[str, bool, str]] = []

    sklearn_ok = is_available("sklearn")
    rows.append(
        (
            "baseline",
            sklearn_ok,
            "sklearn + xgboost" if sklearn_ok else "scikit-learn is not importable",
        )
    )

    for tier, module, enabled, extra in (
        ("flaml", "flaml", settings.enable_flaml, "aegis-ml[serve]"),
        ("autogluon", "autogluon.tabular", settings.enable_autogluon, "aegis-ml[strong]"),
        ("tabpfn", "tabpfn", settings.enable_tabpfn, "aegis-ml[strong]"),
    ):
        importable = is_available(module.split(".")[0])
        if not enabled:
            rows.append((tier, False, f"disabled by settings (AEGIS_ML_ENABLE_{tier.upper()}=0)"))
        elif not importable:
            rows.append((tier, False, f"{module} not importable — `uv pip install '{extra}'`"))
        else:
            rows.append((tier, True, f"{module} {_version_of(module) or 'installed'}"))
    return rows


def _postgres_reachable(dsn: str, timeout: float = 2.0) -> tuple[bool, str]:
    """Probe a Postgres DSN with a plain TCP connect.

    A socket connect rather than a driver handshake on purpose: ``doctor`` must not require
    psycopg to be installed to answer "is the database reachable from this machine", and the
    common failure on the day is a host or port that is wrong or firewalled, which a TCP
    connect detects exactly.

    Args:
        dsn: A ``postgresql://`` URL.
        timeout: Connect timeout in seconds.

    Returns:
        ``(reachable, detail)``.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(dsn)
        host = parts.hostname or "localhost"
        port = parts.port or 5432
    except ValueError as exc:
        return False, f"unparseable DSN: {exc}"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"TCP connect to {host}:{port} succeeded (no auth attempted)"
    except OSError as exc:
        return False, f"cannot reach {host}:{port} — {type(exc).__name__}: {exc}"


def _writable(directory: Path) -> tuple[bool, str]:
    """Return whether ``directory`` can be created and written to."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".aegis_ml_write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, "writable"
    except OSError as exc:
        return False, f"NOT writable: {type(exc).__name__}: {exc}"


def _realism_check(domain_id: str | None) -> tuple[str, bool | None]:
    """Score the registered reference frame against the realism band.

    Args:
        domain_id: Which domain's champion to check; the most recent run when ``None``.

    Returns:
        ``(message, inside_band)`` where ``inside_band`` is ``None`` when nothing could be
        measured — which is reported as "unknown", never as a pass.
    """
    from aegis_ml._require import require
    from aegis_ml.registry import store

    try:
        entry = store.champion(domain_id) if domain_id else None
        if entry is None:
            runs = store.list_runs(domain_id=domain_id, limit=1)
            entry = runs[0] if runs else None
    except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
        return f"registry unreadable: {type(exc).__name__}: {exc}", None
    if entry is None:
        return "no registered run — nothing to check yet (`aegis-ml train` first)", None

    reference = entry.paths.get("reference_frame")
    problem_path = entry.paths.get("problem")
    if not reference or not Path(reference).exists():
        return f"run {entry.run_id} froze no reference frame", None
    if not problem_path or not Path(problem_path).exists():
        return f"run {entry.run_id} stored no problem spec", None

    try:
        pd = require("aegis-ml[serve]", "pandas")
        problem = MLProblem.model_validate_json(Path(problem_path).read_text(encoding="utf-8"))
        frame = pd.read_parquet(reference)
        from aegis_ml.data import latent as latent_mod
        from aegis_ml.pipelines.flows import realism_band_for

        score = float(latent_mod.assert_learnable(frame, problem))
        floor, ceiling = realism_band_for(problem)
        inside = floor <= score <= ceiling
        verdict = "INSIDE" if inside else ("BELOW — the label is closer to noise than signal"
                                           if score < floor else
                                           "ABOVE — the generator forgot the noise, the "
                                           "confounders or the missingness; this will look "
                                           "perfect here and collapse on real data")
        return (
            f"run {entry.run_id} ({problem.domain_id}): held-out score {score:.3f} vs band "
            f"[{floor:.2f}, {ceiling:.2f}] — {verdict}",
            inside,
        )
    except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
        return f"could not score the reference frame: {type(exc).__name__}: {exc}", None


@app.command()
def doctor(
    problem: Path = typer.Option(None, help="Problem JSON, to check its domain's reference frame."),
    domain_id: str = typer.Option(None, help="Domain whose reference frame to check."),
    strict: bool = typer.Option(
        False, help="Also exit non-zero when the reference frame is outside the realism band."
    ),
) -> None:
    """Print the environment, the tiers, the paths and the realism band; exit non-zero if broken.

    Args:
        problem: A problem JSON whose ``domain_id`` selects the run to realism-check.
        domain_id: The domain directly, when there is no problem file to hand.
        strict: Treat a reference frame outside the realism band as an essential failure.

    Raises:
        typer.Exit: Non-zero when something essential is broken — a missing core library, an
            unwritable artifact directory or registry, or (with ``--strict``) a frame outside
            the realism band. Exit code 0 means the morning can proceed.
    """
    problems: list[str] = []

    _echo("── environment " + "─" * 64)
    _echo(f"  python           {sys.version.split()[0]}  ({platform.platform()})")
    _echo(f"  executable       {sys.executable}")
    _echo(f"  aegis_ml         {__version__}")
    if sys.version_info[:2] < _MIN_PYTHON:
        problems.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.11")

    if is_available("aegis"):
        import importlib

        module = importlib.import_module("aegis")
        where = getattr(module, "__file__", "<namespace package>")
        _echo(f"  aegis            importable from {where}")
    else:
        _echo("  aegis            NOT importable — the host platform is not on this path")
        _echo("                   fix: install it, or run from the backend venv where it lives")

    _echo()
    _echo("── resolved versions " + "─" * 58)
    for name in _DISTRIBUTIONS:
        found = _version_of(name)
        marker = "  " if found else "!!"
        _echo(f"{marker} {name:<20} {found or 'not installed'}")
        if found is None and name in _ESSENTIAL:
            problems.append(f"{name} is not installed (`uv pip install 'aegis-ml[serve]'`)")

    _echo()
    _echo("── AutoML tiers " + "─" * 63)
    for tier, available, reason in _tier_report():
        _echo(f"  {'RUNS    ' if available else 'skipped '} {tier:<12} {reason}")
    if settings.enable_tabpfn:
        _echo()
        _echo("  LICENCE  " + TABPFN_NOTICE)

    _echo()
    _echo("── paths " + "─" * 70)
    trainer_exists = settings.trainer_python.exists()
    _echo(f"  trainer venv     {settings.trainer_venv}")
    _echo(
        f"  trainer python   {settings.trainer_python} "
        f"{'(exists)' if trainer_exists else '(MISSING — `uv venv .venv-ml --python 3.11`)'}"
    )
    artifact_ok, artifact_detail = _writable(settings.artifact_path.parent)
    _echo(f"  artifact_path    {settings.artifact_path}")
    _echo(f"                   directory {artifact_detail}")
    trained = "present" if settings.artifact_path.exists() else "not yet trained"
    _echo(f"                   artifact  {trained}")
    if not artifact_ok:
        problems.append(f"artifact directory {settings.artifact_path.parent} is not writable")
    registry_ok, registry_detail = _writable(settings.registry_dir)
    _echo(f"  registry_dir     {settings.registry_dir} ({registry_detail})")
    if not registry_ok:
        problems.append(f"registry directory {settings.registry_dir} is not writable")
    reports_ok, reports_detail = _writable(settings.reports_dir)
    _echo(f"  reports_dir      {settings.reports_dir} ({reports_detail})")
    if not reports_ok:
        problems.append(f"reports directory {settings.reports_dir} is not writable")
    _echo(f"  adapter_dir      {settings.adapter_dir}")

    _echo()
    _echo("── orchestration & storage " + "─" * 52)
    from aegis_ml.pipelines.prefect_shim import prefect_active

    orchestration = (
        "ACTIVE — flows register with the server"
        if prefect_active()
        else "inactive — flows run as plain functions (artifacts are identical)"
    )
    mirror = (
        "enabled"
        if settings.enable_mlflow
        else "disabled (filesystem registry is the source of truth)"
    )
    _echo(f"  prefect          {orchestration}")
    _echo(f"  mlflow mirror    {mirror}")
    if settings.postgres_dsn:
        reachable, detail = _postgres_reachable(settings.postgres_dsn)
        _echo(f"  postgres         {'reachable' if reachable else 'UNREACHABLE'} — {detail}")
        if not reachable:
            _echo(
                "                   (not essential: the filesystem registry is the "
                "source of truth)"
            )
    else:
        _echo("  postgres         no DSN configured (AEGIS_ML_POSTGRES_DSN unset)")

    _echo()
    _echo("── data realism " + "─" * 63)
    selected_domain = domain_id
    if selected_domain is None and problem is not None:
        selected_domain = _load_problem(problem, None).domain_id
    message, inside = _realism_check(selected_domain)
    from aegis_ml.pipelines.flows import REALISM_ACCURACY_BAND, REALISM_R2_BAND

    _echo(
        f"  bands            regression R² {REALISM_R2_BAND}, "
        f"classification accuracy {REALISM_ACCURACY_BAND}"
    )
    _echo(f"  reference frame  {message}")
    if inside is False:
        _echo(
            "                   A frame outside the band is the single fastest signal that "
            "the generator is wrong — fix it before anything expensive runs."
        )
        if strict:
            problems.append("reference frame is outside the realism band")

    _echo()
    if problems:
        _echo("── VERDICT: not ready " + "─" * 57)
        for issue in problems:
            typer.secho(f"  ✗ {issue}", fg=typer.colors.RED)
        raise typer.Exit(1)
    _echo("── VERDICT: ready " + "─" * 61)
    typer.secho("  ✓ nothing essential is broken", fg=typer.colors.GREEN)


# ──────────────────────────────────────────────────────────────────────────── init ──


@app.command()
def init(
    domain_id: str = typer.Option(..., help="Stable machine id; matches the adapter DOMAIN_ID."),
    out: Path = typer.Option(Path("problem.json"), help="Where to write the problem scaffold."),
    target: str = typer.Option("outcome", help="Name of the predicted column."),
    task: str = typer.Option("regression", help="'regression' or 'classification'."),
    unit: str = typer.Option(None, help="Unit of a regression target, e.g. 'percent'."),
    templates: Path = typer.Option(None, help="Copy templates/adapter/ into this directory."),
    force: bool = typer.Option(False, help="Overwrite an existing problem file."),
) -> None:
    """Write an :class:`MLProblem` scaffold, and optionally copy the adapter templates.

    Args:
        domain_id: The domain id every consumer keys on.
        out: Destination for the problem JSON.
        target: The predicted column's name.
        task: ``regression`` or ``classification``.
        unit: The target's unit — not decoration: the sanity probe and every explanation
            render the prediction with it, and a target without one prints bare floats into
            a decision-support sentence.
        templates: Copy this package's ``templates/adapter/`` tree here, if present.
        force: Overwrite ``out`` if it exists.

    Raises:
        typer.Exit: When ``out`` exists and ``--force`` was not passed, or the task/unit
            combination is one :class:`MLProblem` refuses.
    """
    destination = Path(out)
    if destination.exists() and not force:
        _fail(f"{destination} already exists; pass --force to overwrite.")

    if task == "classification":
        target_spec = {
            "name": target,
            "task": "classification",
            "description": f"What {domain_id} predicts, in the client's own words.",
            "levels": ["low", "high"],
        }
    else:
        target_spec = {
            "name": target,
            "task": "regression",
            "description": f"What {domain_id} predicts, in the client's own words.",
            "unit": unit or "unit",
        }

    scaffold = {
        "domain_id": domain_id,
        "requested_coverage": settings.requested_coverage,
        "primary_metric": "",
        "features": [
            {
                "name": "driver_numeric",
                "dtype": "numeric",
                "description": "Rename me: a monotone driver of the target.",
                "unit": "unit",
                "minimum": 0.0,
                "maximum": 100.0,
                "nullable": True,
            },
            {
                "name": "driver_categorical",
                "dtype": "categorical",
                "description": "Rename me: a categorical driver.",
                "levels": ["alpha", "beta", "gamma"],
            },
        ],
        "target": target_spec,
    }
    try:
        MLProblem.model_validate(scaffold)
    except Exception as exc:  # noqa: BLE001 - the validation message IS the guidance
        _fail(f"the scaffold does not validate: {exc}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(scaffold, indent=2), encoding="utf-8")
    _echo(f"wrote {destination}")
    _echo(
        "Next: rename the two example features to the real ones — a categorical feature "
        "MUST declare its levels, because an unseen level otherwise one-hot-encodes to all "
        "zeros without raising."
    )

    if templates is not None:
        import shutil

        source = Path(__file__).resolve().parents[2] / "templates" / "adapter"
        if not source.exists():
            _fail(f"no adapter templates at {source}")
        shutil.copytree(source, templates, dirs_exist_ok=True)
        _echo(f"copied adapter templates to {templates}")


# ──────────────────────────────────────────────────────────────────────── contract ──


@app.command()
def contract(
    data: Path = typer.Option(..., help="Training frame (.csv or .parquet)."),
    problem: Path = typer.Option(None, help="Problem JSON."),
    adapter: str = typer.Option(None, help="Module exposing PROBLEM."),
) -> None:
    """Validate a frame against its contract, scan for leakage, and prove the label is learnable.

    Run this **before** anything expensive. An AutoML search over a target that carries no
    signal spends its whole budget discovering that and reports it as a leaderboard where
    everything failed equally — which reads like a hard problem rather than a broken
    generator.

    Args:
        data: The training frame.
        problem: Problem JSON.
        adapter: Module exposing ``PROBLEM``, as an alternative to ``--problem``.

    Raises:
        typer.Exit: Non-zero when the contract fails, leakage is found, or the label does not
            clear its learnability floor.
    """
    spec = _load_problem(problem, adapter)
    frame = _load_frame(data)
    failures: list[str] = []

    from aegis_ml.data import contract_check
    from aegis_ml.data import latent as latent_mod
    from aegis_ml.features import leakage as leakage_mod
    from aegis_ml.pipelines.flows import realism_band_for

    report = contract_check.check(frame, spec)
    ok = bool(getattr(report, "ok", False))
    _echo(
        f"contract      {'PASS' if ok else 'FAIL'}  "
        f"({len(frame)} rows, {len(frame.columns)} columns)"
    )
    if not ok:
        failures.append("data contract failed")
        for line in str(getattr(report, "errors", report)).splitlines()[:20]:
            _echo(f"              {line}")

    found = list(leakage_mod.detect_leakage(frame, spec))
    _echo(f"leakage       {'none' if not found else 'FLAGGED: ' + ', '.join(map(str, found))}")
    if found:
        failures.append("target leakage flagged")

    floor, ceiling = realism_band_for(spec)
    try:
        score = float(latent_mod.assert_learnable(frame, spec))
        band = "inside" if floor <= score <= ceiling else ("BELOW" if score < floor else "ABOVE")
        _echo(f"learnability  {score:.4f}  band [{floor:.2f}, {ceiling:.2f}] — {band}")
        if score > ceiling:
            _echo(
                "              ABOVE the ceiling means the generator sampled the label with "
                "too little noise: everything downstream will look better than it will on "
                "real data."
            )
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _echo(f"learnability  FAILED — {exc}")
        failures.append("label is not learnable")

    if failures:
        _fail("contract check failed: " + "; ".join(failures))
    typer.secho("contract check passed", fg=typer.colors.GREEN)


# ─────────────────────────────────────────────────────────────────────────── synth ──


@app.command()
def synth(
    data: Path = typer.Option(..., help="Real frame to learn the joint distribution from."),
    out: Path = typer.Option(..., help="Where to write the synthetic frame (.parquet)."),
    rows: int = typer.Option(2000, help="How many synthetic rows to generate."),
    problem: Path = typer.Option(None, help="Problem JSON."),
    adapter: str = typer.Option(None, help="Module exposing PROBLEM."),
    model: str = typer.Option("gaussian_copula", help="'gaussian_copula' or 'ctgan'."),
) -> None:
    """Fit SDV on a real frame and sample more rows from it — the "make 10× more" path.

    Args:
        data: The real frame to learn from.
        out: Destination parquet.
        rows: How many rows to sample.
        problem: Problem JSON.
        adapter: Module exposing ``PROBLEM``.
        model: Which SDV synthesiser to fit — the cheap copula, or CTGAN.

    Raises:
        typer.Exit: When SDV is not installed, naming the install command.

    Synthetic rows are labelled synthetic wherever they travel. A model fitted on them is a
    model fitted on a *copula's opinion* of the data, and the model card must say so.
    """
    spec = _load_problem(problem, adapter)
    frame = _load_frame(data)
    from aegis_ml.data import synth as synth_mod

    produced, quality = synth_mod.synthesize(frame, n=rows, model=model, problem=spec)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    produced.to_parquet(out, index=False)
    _echo(f"wrote {len(produced)} synthetic rows to {out}")
    _echo("SDMetrics quality report:")
    _echo(json.dumps(quality, indent=2, default=str))
    _echo(
        "These rows are a copula's opinion of your data, not your data. Any model fitted "
        "on them must say so on its card."
    )


# ─────────────────────────────────────────────────────────────────────────── train ──


@app.command()
def train(
    problem: Path = typer.Option(None, help="Problem JSON."),
    adapter: str = typer.Option(None, help="Module exposing PROBLEM."),
    data: Path = typer.Option(None, help="Training frame (.csv or .parquet)."),
    tier: list[str] = typer.Option(None, "--tier", help="AutoML tiers to run; repeatable."),
    time_budget: int = typer.Option(None, help="Search budget in seconds."),
    seed: int = typer.Option(None, help="Random seed."),
    trainer_venv: bool = typer.Option(False, help="Run the search in the isolated trainer venv."),
    hpo: bool = typer.Option(True, help="Run the Optuna study over the winning recipe."),
    force: bool = typer.Option(False, help="Bypass the stage cache and re-run every stage."),
    resume_from: str = typer.Option(None, help="Adopt a previous run's recipe (same data only)."),
    full: bool = typer.Option(False, help="Also promote and drift-check, writing RUN_SUMMARY.md."),
) -> None:
    """Search, tune, fit, measure and register a model — every number from a real fit.

    Args:
        problem: Problem JSON.
        adapter: Module exposing ``PROBLEM``.
        data: The training frame; falls back to the champion's frozen reference frame.
        tier: AutoML tiers to run. Omit to let the search decide from what is installed and
            enabled, recording every skipped tier with its reason.
        time_budget: Search budget in seconds.
        seed: Random seed for split, search and fit.
        trainer_venv: Run the search through the subprocess bridge into ``.venv-ml``.
        hpo: Run the Optuna study.
        force: Bypass the content-addressed stage cache.
        resume_from: Adopt a previous run's recipe; refuses if that run's dataset digest
            differs from this frame's.
        full: Run ``full_flow`` instead — train, promote, drift-check and write the bundle.

    Raises:
        typer.Exit: Non-zero on any refusal (unlearnable label, resume mismatch, missing
            frame), with the typed error's own message printed.
    """
    spec = _load_problem(problem, adapter)
    from aegis_ml.pipelines.flows import full_flow, train_flow

    kwargs: dict[str, Any] = {
        "tiers": list(tier) if tier else None,
        "time_budget": time_budget,
        "seed": seed,
        "use_trainer_venv": trainer_venv,
        "do_hpo": hpo,
        "force": force,
        "resume_from": resume_from,
        "source": data,
    }
    try:
        if full:
            bundle = full_flow(spec, **kwargs)
            _echo(f"\nrun_id: {bundle['run_id']}")
            _echo(f"summary: {bundle['summary_path']}")
            if not bundle["decision"]["promoted"]:
                raise typer.Exit(2)
            return
        result = train_flow(spec, **kwargs)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    coverage = (
        f"{result.empirical_coverage:.2%}"
        if result.empirical_coverage is not None
        else "not measured"
    )
    _echo()
    _echo(f"run_id            {result.run_id}")
    _echo(f"{result.metric_name:<17} {result.metric_value:.4g}  (held-out test split)")
    _echo(f"coverage          requested {result.requested_coverage:.0%} / achieved {coverage}")
    _echo(
        f"splits            {result.training_size} train / "
        f"{result.calibration_size} calib / {result.test_size} test"
    )
    _echo(f"digest            {result.dataset_digest}")
    _echo(f"artifact          {result.artifact_path}")
    _echo("\nNext: `aegis-ml promote --run-id " + result.run_id + "`")


# ──────────────────────────────────────────────────────────────────────────── eval ──


@app.command(name="eval")
def eval_command(
    run_id: str = typer.Option(..., help="The registered run to re-score."),
    data: Path = typer.Option(None, help="Fresh labelled data; defaults to the run's reference."),
) -> None:
    """Re-score a registered run on data it has never seen.

    Args:
        run_id: The run to re-score.
        data: Fresh labelled data. Omitted, it re-scores on the run's own frozen reference
            frame — an integrity check that should reproduce the registered number, and it
            is labelled as such rather than presented as fresh evidence.

    Raises:
        typer.Exit: Non-zero when the run has no persisted model or problem spec.
    """
    from aegis_ml.pipelines.flows import eval_flow

    try:
        result = eval_flow(run_id, source=data)
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return
    _echo()
    _echo(f"{result.metric_name:<17} {result.metric_value:.4g}  on {result.test_size} rows")
    _echo(f"digest            {result.dataset_digest}")
    for note in result.notes[-4:]:
        _echo(f"  - {note}")


# ───────────────────────────────────────────────────────────────────────── promote ──


@app.command()
def promote(
    run_id: str = typer.Option(..., help="The challenger run."),
    force: bool = typer.Option(False, help="Promote despite a failed gate; records it."),
) -> None:
    """Judge a challenger against the champion and replace the served artifact if it wins.

    Args:
        run_id: The challenger run.
        force: Promote despite a refusal. The decision still records every failed check and
            the override is written into its reasons — an override a reader cannot see is
            indistinguishable from a pass.

    Raises:
        typer.Exit: Exit code 2 when the gate refused and ``--force`` was not passed, so a
            Makefile or CI step fails on a rejected promotion.
    """
    from aegis_ml.pipelines.flows import promote_flow

    try:
        decision = promote_flow(run_id, force=force)
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    _echo()
    colour = typer.colors.GREEN if decision.promoted else typer.colors.YELLOW
    typer.secho(f"promoted: {decision.promoted}", fg=colour)
    if decision.champion_run_id:
        _echo(f"champion: {decision.champion_run_id}")
    for name, passed in decision.checks.items():
        _echo(f"  [{'x' if passed else ' '}] {name}")
    for key, value in decision.metrics.items():
        _echo(f"      {key} = {value:.4g}")
    for reason in decision.reasons:
        _echo(f"  - {reason}")
    if not decision.promoted and not force:
        raise typer.Exit(2)


@app.command()
def rollback(
    domain_id: str = typer.Option(..., help="The domain whose champion to roll back."),
) -> None:
    """Restore the previously promoted artifact for a domain.

    Args:
        domain_id: The domain to roll back.

    Raises:
        typer.Exit: Non-zero when there is no previous artifact to restore — a rollback that
            silently did nothing is the worst possible outcome of a rollback.
    """
    from aegis_ml.registry import promote as promote_mod

    try:
        restored = promote_mod.rollback(domain_id)
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return
    _echo(f"rolled back {domain_id}: {restored}")
    _echo(f"served artifact is now {settings.artifact_path}")


# ─────────────────────────────────────────────────────────────────────────── drift ──


@app.command()
def drift(
    run_id: str = typer.Option(..., help="The run whose frozen reference frame is the baseline."),
    data: Path = typer.Option(..., help="The current frame to compare (.csv or .parquet)."),
) -> None:
    """Measure drift against a run's reference frame and estimate performance without labels.

    Args:
        run_id: The registered run.
        data: The current frame.

    Raises:
        typer.Exit: Exit code 2 on a ``block`` verdict, so a scheduled check fails loudly.
            The served model is **not** withdrawn: Aegis serves the model it has and flags
            it; what this blocks is the promotion of anything calibrated on a stale reference.
    """
    from aegis_ml.pipelines.flows import drift_flow

    frame = _load_frame(data)
    try:
        report = drift_flow(run_id, frame)
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    _echo()
    _echo(f"verdict           {report.verdict}")
    _echo(f"dataset drift     {report.dataset_drift}")
    _echo(f"drifted share     {report.drifted_share:.1%} ({len(report.drifted_features)} features)")
    if report.drifted_features:
        _echo(f"drifted features  {', '.join(report.drifted_features)}")
    if report.estimated_metric_name:
        _echo(
            f"estimated {report.estimated_metric_name:<7} {report.estimated_metric_value} "
            f"(ESTIMATE — no ground truth was used; not a measurement)"
        )
    if report.html_report_path:
        _echo(f"report            {report.html_report_path}")
    if report.verdict == "block":
        raise typer.Exit(2)


# ──────────────────────────────────────────────────────────────────────── forecast ──


@app.command()
def forecast(
    data: Path = typer.Option(..., help="Series file (.csv/.parquet) with timestamp and value."),
    label: str = typer.Option(..., help="Human label, e.g. 'Shipments dispatched per day'."),
    ts_column: str = typer.Option("ts", help="Timestamp column name."),
    value_column: str = typer.Option("value", help="Value column name."),
    horizon: int = typer.Option(14, help="Steps to forecast beyond the last observation."),
    freq: str = typer.Option(None, help="Frequency alias 'h'|'D'|'W'|'MS'; inferred when omitted."),
    unit: str = typer.Option(None, help="Unit of the values."),
    level: float = typer.Option(None, help="Coverage level to REQUEST."),
    ml_candidates: bool = typer.Option(False, help="Also score the mlforecast roster."),
    out: Path = typer.Option(None, help="Write the full JSON payload here."),
) -> None:
    """Forecast a series and report the coverage the band actually achieved.

    Args:
        data: A file with a timestamp column and a value column.
        label: Human label for the series.
        ts_column: Timestamp column name.
        value_column: Value column name.
        horizon: Steps to forecast.
        freq: Frequency alias; inferred when omitted.
        unit: Unit of the values.
        level: Coverage level to REQUEST.
        ml_candidates: Also score global-ML candidates on the same rolling-origin windows.
        out: Write the full payload as JSON here.

    Raises:
        typer.Exit: Non-zero on any of the three refusals — too little history, a perfectly
            flat series, or a total fit failure. Nothing is substituted: a naive line through
            this history would look like a forecast and carry none of the guarantees of one.
    """
    from aegis_ml.pipelines.flows import forecast_flow

    frame = _load_frame(data)
    for column in (ts_column, value_column):
        if column not in frame.columns:
            _fail(f"column {column!r} not in {data} (columns: {list(frame.columns)})")
    points = [
        (row_ts.to_pydatetime() if hasattr(row_ts, "to_pydatetime") else row_ts, float(row_value))
        for row_ts, row_value in zip(frame[ts_column], frame[value_column], strict=True)
    ]

    try:
        payload = forecast_flow(
            points,
            label=label,
            unit=unit,
            horizon=horizon,
            freq=freq,
            level=level,
            data_source=f"file:{data.name}",
            include_ml_candidates=ml_candidates,
        )
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    run = payload["forecast"]
    _echo()
    _echo(f"model             {run['model']}  (selected on measured sMAPE)")
    _echo(f"interval          {run['interval_method']} — {run['interval_method_detail']}")
    _echo(
        f"coverage          requested {run['requested_coverage']:.0%} / "
        f"achieved {run['empirical_coverage']:.1%}"
    )
    _echo(f"sMAPE / MAE       {run['smape']:.3f}% / {run['mae']:.4g}")
    _echo("candidates:")
    for candidate in payload["backtest"]["candidates"]:
        _echo(
            f"  {'*' if candidate['selected'] else ' '} {candidate['model']:<16} "
            f"{candidate['family']:<14} sMAPE {candidate['smape']:.3f}  "
            f"coverage {candidate['empirical_coverage']:.1%}"
        )
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _echo(f"\nwrote {out}")


# ──────────────────────────────────────────────────────────────────────────── card ──


@app.command()
def card(
    run_id: str = typer.Option(..., help="The registered run."),
    fmt: str = typer.Option("md", "--format", help="'md', 'html' or 'json'."),
    out: Path = typer.Option(None, help="Write to this file instead of stdout."),
) -> None:
    """Print or write a run's model card.

    Args:
        run_id: The registered run.
        fmt: ``md``, ``html`` or ``json``.
        out: Destination file; stdout when omitted.

    Raises:
        typer.Exit: Non-zero when the run is unknown or the card cannot be rendered.
    """
    from aegis_ml.registry import store

    try:
        entry = store.load_entry(run_id)
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    if fmt == "json":
        text = entry.result.model_dump_json(indent=2)
    else:
        key = "card_html" if fmt == "html" else "card_md"
        stored = entry.paths.get(key)
        if stored and Path(stored).exists():
            text = Path(stored).read_text(encoding="utf-8")
        else:
            from aegis_ml.explain import card as card_mod

            built = card_mod.build_card(entry.result)
            renderer = card_mod.render_html if fmt == "html" else card_mod.render_markdown
            text = str(renderer(built))

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        _echo(f"wrote {out}")
    else:
        _echo(text)


# ────────────────────────────────────────────────────────────────────────── export ──


@app.command()
def export(
    run_id: str = typer.Option(..., help="The registered run to export."),
    out: Path = typer.Option(None, help="Destination .onnx file; defaults inside the run dir."),
    validate: bool = typer.Option(True, help="Validate the round-trip on the reference frame."),
) -> None:
    """Export a run's fitted model to ONNX and validate the round-trip.

    Args:
        run_id: The registered run.
        out: Destination ``.onnx`` path.
        validate: Compare ONNX Runtime's outputs against the fitted model's on real rows.

    Raises:
        typer.Exit: Non-zero when the run has no persisted model, or the round-trip differs.

    What crosses to ONNX is the **point predictor and nothing else**. MAPIE's conformal
    intervals and the SHAP attributions do not export — they are not part of the graph — so
    an ONNX artifact is a fast portable predictor, never a replacement for the served spine.
    Quoting an ONNX prediction with the joblib model's interval would attach a calibration to
    a model that never received it.
    """
    from aegis_ml._require import require
    from aegis_ml.export import onnx as onnx_mod
    from aegis_ml.registry import store

    try:
        entry = store.load_entry(run_id)
        joblib = require("aegis-ml[serve]", "joblib")
        model_path = Path(entry.paths.get("model", Path(store.run_dir(run_id)) / "model.joblib"))
        if not model_path.exists():
            _fail(f"run {run_id} has no persisted model at {model_path}")
        model = joblib.load(model_path)
        problem = MLProblem.model_validate_json(
            Path(entry.paths["problem"]).read_text(encoding="utf-8")
        )
        destination = Path(out) if out else Path(store.run_dir(run_id)) / "model.onnx"
        destination.parent.mkdir(parents=True, exist_ok=True)
        onnx_mod.to_onnx(model, problem, destination)
        _echo(f"wrote {destination}")

        if validate:
            reference = entry.paths.get("reference_frame")
            if not reference or not Path(reference).exists():
                _echo("skipped round-trip validation: this run froze no reference frame")
            else:
                pd = require("aegis-ml[serve]", "pandas")
                sample = pd.read_parquet(reference).head(256)
                onnx_mod.validate_roundtrip(model, destination, sample)
                _echo(f"round-trip validated on {len(sample)} real rows")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    _echo(
        "NOTE: the conformal interval and the SHAP attributions did NOT export — they are "
        "not part of the ONNX graph. This artifact is a portable point predictor."
    )


# ──────────────────────────────────────────────────────────────────────── registry ──


@app.command()
def registry(
    domain_id: str = typer.Option(None, help="Restrict to one domain."),
    limit: int = typer.Option(20, help="Maximum rows."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List registered runs, newest first, with both coverage numbers.

    Args:
        domain_id: Restrict to one domain.
        limit: Maximum rows.
        as_json: Emit JSON.

    Raises:
        typer.Exit: Non-zero when the registry cannot be read.
    """
    from aegis_ml.registry import store

    try:
        entries = store.list_runs(domain_id=domain_id, limit=limit)
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    if as_json:
        _echo(json.dumps([entry.model_dump(mode="json") for entry in entries], indent=2))
        return

    if not entries:
        _echo(f"no runs in {settings.registry_dir}")
        return

    width = max(6, *(len(entry.run_id) for entry in entries))
    _echo(
        f"{'run_id':<{width}} {'stage':<11} {'metric':<10} "
        f"{'value':>9} {'req':>5} {'emp':>7}  created"
    )
    _echo("-" * (width + 52))
    for entry in entries:
        result = entry.result
        empirical = (
            f"{result.empirical_coverage:.1%}" if result.empirical_coverage is not None else "  n/a"
        )
        _echo(
            f"{entry.run_id:<{width}} {entry.stage:<11} {result.metric_name:<10} "
            f"{result.metric_value:>9.4g} {result.requested_coverage:>5.0%} {empirical:>7}  "
            f"{entry.created_at}"
        )
    _echo(f"\nregistry: {settings.registry_dir}")


# ─────────────────────────────────────────────────────────────────────────── serve ──


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8099, help="Bind port."),
    prefix: str = typer.Option("/ml", help="URL prefix for the router."),
    reload: bool = typer.Option(False, help="Reload on source change (development only)."),
) -> None:
    """Serve the ML router standalone, for the console or a quick look at the registry.

    Args:
        host: Bind address. Defaults to loopback: this surface exposes model cards, dataset
            digests and drift reports, and none of that should reach a network by accident.
        port: Bind port.
        prefix: URL prefix the router is mounted under.
        reload: Uvicorn auto-reload.

    Raises:
        typer.Exit: Non-zero when FastAPI or uvicorn is not installed.

    In production the host mounts :func:`aegis_ml.serve.router.build_router` into its
    own FastAPI app instead — that keeps one process, one auth layer and one OpenAPI schema.
    This command is the standalone convenience, not the deployment path.
    """
    from aegis_ml._require import require

    try:
        fastapi = require("fastapi", "fastapi")
        uvicorn = require("uvicorn", "uvicorn")
    except ImportError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    from aegis_ml.serve.router import build_router

    application = fastapi.FastAPI(
        title="aegis-ml", version=__version__, description="ML registry, cards, drift and what-if."
    )
    application.include_router(build_router(prefix=prefix))

    @application.get("/")
    async def index() -> dict[str, Any]:
        """Name the mounted routes so an operator does not have to guess them."""
        return {
            "service": "aegis-ml",
            "version": __version__,
            "started_at": datetime.now(UTC).isoformat(),
            "routes": [
                f"{prefix}/registry",
                f"{prefix}/runs/{{run_id}}/card",
                f"{prefix}/leaderboard",
                f"{prefix}/drift/{{run_id}}",
                f"{prefix}/whatif",
                f"{prefix}/health",
            ],
        }

    _echo(f"serving on http://{host}:{port}{prefix} (registry: {settings.registry_dir})")
    uvicorn.run(application, host=host, port=port, reload=reload)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - module execution path
    main()

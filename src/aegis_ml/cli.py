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

    This is a thin projection of :func:`aegis_ml.automl.tiers.tier_status`, and deliberately
    holds no availability logic of its own. It used to: it re-derived availability from
    importability plus the settings flag, which was right until the tabpfn tier grew a third
    condition — weights on disk or ``TABPFN_TOKEN`` set. ``doctor`` then printed
    ``RUNS tabpfn`` on a machine where ``.fit()`` raises ``TabPFNLicenseError``, while
    ``unavailable_reason`` correctly said otherwise. Two answers to one question, and the
    wrong one was the one a human reads first.

    Returns:
        ``(tier, available, reason)`` in :data:`TIER_ORDER`, where ``reason`` is a short
        capability line when available and the remedy-carrying explanation when not.
    """
    from aegis_ml.automl.tiers import TIER_ORDER, tier_status

    labels = {
        "baseline": "sklearn + xgboost",
        "flaml": lambda: f"flaml {_version_of('flaml') or 'installed'}",
        "autogluon": lambda: f"autogluon.tabular {_version_of('autogluon.tabular') or 'installed'}",
        "tabpfn": lambda: f"tabpfn {_version_of('tabpfn') or 'installed'}",
    }
    status = tier_status()
    rows: list[tuple[str, bool, str]] = []
    for tier in TIER_ORDER:
        reason = status[tier]
        available = reason == "available"
        if available:
            label = labels[tier]
            reason = label() if callable(label) else label
        rows.append((tier, available, reason))
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
    _echo("── config files " + "─" * 63)
    from aegis_ml.config import CONFIG_DIR, load_config_overrides, unknown_keys

    overrides = load_config_overrides()
    if CONFIG_DIR.is_dir():
        _echo(f"  config dir       {CONFIG_DIR}")
        _echo(f"  applied          {len(overrides)} setting(s) loaded from config/*.toml")
    else:
        _echo(f"  config dir       {CONFIG_DIR} (absent — field defaults in use)")
    stray = unknown_keys()
    if stray:
        _echo(
            f"  NOT CONSUMED     {len(stray)} key(s) no setting reads — "
            f"editing them does nothing:"
        )
        for key in stray:
            _echo(f"                     {key}")
    _echo()
    _echo("── data realism " + "─" * 63)
    selected_domain = domain_id
    if selected_domain is None and problem is not None:
        selected_domain = _load_problem(problem, None).domain_id
    message, inside = _realism_check(selected_domain)
    REALISM_R2_BAND = settings.realism_r2_band
    REALISM_ACCURACY_BAND = settings.realism_accuracy_band

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
    data: Path = typer.Option(None, help="Fresh labelled data to re-score on."),
    allow_in_sample: bool = typer.Option(
        False, help="Permit re-scoring on the run's own reference frame (IN-SAMPLE)."
    ),
) -> None:
    """Re-score a registered run on data it has never seen.

    Args:
        run_id: The run to re-score.
        data: Fresh labelled data. Omitted, the flow REFUSES unless --allow-in-sample is
            given, because the fallback measures the model on its own training rows.
        allow_in_sample: Ask for that fallback on purpose. It is an integrity check that
            the artifact loads and predicts, not evidence about unseen data.
            is labelled as such rather than presented as fresh evidence.

    Raises:
        typer.Exit: Non-zero when the run has no persisted model or problem spec.
    """
    from aegis_ml.pipelines.flows import eval_flow

    try:
        result = eval_flow(run_id, source=data, allow_in_sample=allow_in_sample)
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


# ───────────────────────────────────────────────────────────────────────── visuals ──


@app.command()
def visuals(
    run_id: str = typer.Option(None, help="The registered run to draw. Omit with --all."),
    rebuild_all: bool = typer.Option(False, "--all", help="Rebuild every registered run."),
    domain_id: str = typer.Option(None, help="With --all, restrict to one domain."),
    shap_samples: int = typer.Option(300, help="Rows to explain when recomputing SHAP."),
    open_it: bool = typer.Option(False, "--open", help="Open index.html when finished."),
) -> None:
    """(Re)build a run's visual bundle: ``registry_store/runs/<run_id>/visuals/``.

    The flows write this bundle automatically as their last stage, so this command exists
    for the two cases that are not a fresh training run: a run registered before the visuals
    stage existed, and a run whose figures you want back after deleting them. It is a pure
    function of the run directory, so re-running it on unchanged artifacts reproduces the
    same bundle rather than a slightly different one.

    A figure whose input is missing is **omitted and explained** in ``manifest.json`` and on
    the page, never drawn empty — so a non-zero omission count in the output below is
    information about the run, not a failure of this command.

    Args:
        run_id: The registered run to draw.
        rebuild_all: Rebuild every registered run instead of one.
        domain_id: With ``--all``, restrict to one domain.
        shap_samples: Rows to explain when recomputing global SHAP attribution.
        open_it: Open the rendered page in the default browser when finished.

    Raises:
        typer.Exit: Non-zero when neither ``--run-id`` nor ``--all`` was given, when the run
            is unknown, or when every requested run failed to build.
    """
    from aegis_ml.registry import store

    if not run_id and not rebuild_all:
        _fail("pass --run-id <run> or --all; there is nothing to draw otherwise")
        return

    try:
        targets = (
            [entry.run_id for entry in store.list_runs(domain_id=domain_id, limit=10_000)]
            if rebuild_all
            else [run_id]
        )
    except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
        _fail(f"{type(exc).__name__}: {exc}")
        return

    if not targets:
        _echo(f"no runs in {settings.registry_dir}")
        return

    from aegis_ml.report.bundle import build_bundle

    built: list[Path] = []
    failures: list[tuple[str, str]] = []
    for target in targets:
        try:
            directory = build_bundle(target, shap_max_samples=shap_samples)
        except Exception as exc:  # noqa: BLE001 - the typed refusal's message is the output
            failures.append((target, f"{type(exc).__name__}: {exc}"))
            typer.secho(f"  ✗ {target}: {type(exc).__name__}: {exc}", fg=typer.colors.RED)
            continue
        summary = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        built.append(directory / "index.html")
        tone = typer.colors.GREEN if summary["omitted"] == 0 else typer.colors.YELLOW
        typer.secho(
            f"  ✓ {target}: {summary['rendered']} figures rendered, "
            f"{summary['omitted']} omitted",
            fg=tone,
        )
        for figure in summary["plots"]:
            if figure["status"] != "rendered":
                _echo(f"      - {figure['file']}: {figure['reason']}")

    for path in built:
        _echo(f"\nopen {path}")
    if failures and not built:
        _fail(f"every requested run failed to build; first: {failures[0][1]}")
        return
    if open_it and built:
        import webbrowser

        webbrowser.open(built[0].as_uri())


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


# ─────────────────────────────────────────────────────────────────────── dashboard ──


DASHBOARD_HUB_PORT = 8000
"""Default port for the hub page."""

DASHBOARD_MLFLOW_PORT = 5000
"""MLflow's own documented default. On macOS it is usually held by the AirPlay Receiver,
which is why an unspecified port is allowed to move to the next free one and say so."""

DASHBOARD_OPTUNA_PORT = 8080
"""Optuna Dashboard's own documented default."""


@app.command()
def dashboard(  # noqa: PLR0913 - one flag per service and per port; collapsing them hides them
    host: str = typer.Option("127.0.0.1", help="Bind address for all three servers."),
    port: int = typer.Option(None, help=f"Hub port (default {DASHBOARD_HUB_PORT})."),
    mlflow_port: int = typer.Option(
        None, help=f"MLflow UI port (default {DASHBOARD_MLFLOW_PORT})."
    ),
    optuna_port: int = typer.Option(
        None, help=f"Optuna Dashboard port (default {DASHBOARD_OPTUNA_PORT})."
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser."),
    no_mlflow: bool = typer.Option(False, "--no-mlflow", help="Do not start the MLflow UI."),
    no_optuna: bool = typer.Option(False, "--no-optuna", help="Do not start Optuna Dashboard."),
    backfill_mlflow: bool = typer.Option(
        True,
        "--backfill-mlflow/--no-backfill-mlflow",
        help="Log registry runs into the local MLflow store before starting it. Idempotent.",
    ),
) -> None:
    """Bring up the hub, the MLflow UI and Optuna Dashboard, and open a browser.

    Args:
        host: Bind address. Loopback by default: the hub exposes model cards, dataset
            digests and drift reports, none of which should reach a network by accident.
        port: Hub port. Unset means :data:`DASHBOARD_HUB_PORT`.
        mlflow_port: MLflow port. Unset means :data:`DASHBOARD_MLFLOW_PORT`, or the next
            free port when that one is held — announced in the banner and on the panel. A
            port given explicitly is never moved; if it is busy, the panel says so.
        optuna_port: Optuna Dashboard port, with the same rule.
        no_browser: Suppress the browser launch.
        no_mlflow: Skip MLflow entirely. Its panel then says the flag switched it off.
        no_optuna: Skip Optuna Dashboard, likewise.
        backfill_mlflow: Mirror every registry run into ``registry_store/mlflow/`` first,
            so the MLflow UI has this repository's runs in it rather than being empty.
            Idempotent — a run already present is skipped, not duplicated.

    Raises:
        typer.Exit: Non-zero when the hub's own port cannot be bound. That one is fatal
            because the hub is the page that explains everything else; the two service
            panels degrade in place instead, each naming its own reason and remedy.

    Ctrl-C terminates the child processes before exiting. An orphaned MLflow server keeps
    its port, so the next invocation would find it busy and degrade — a failure that looks
    like the dashboard breaking itself.
    """
    import signal
    import webbrowser

    from aegis_ml.dashboard import hub, server, services
    from aegis_ml.registry import store

    def _on_terminate(_signum: int, _frame: Any) -> None:  # noqa: ANN401 - a frame object
        """Turn a SIGTERM into the same clean stop that Ctrl-C already produces.

        Without this, ``kill <pid>`` — what a supervisor, a Makefile target or a shell's
        job control sends — takes the default action and the process dies before the
        supervisor's context manager can run. The MLflow and Optuna children then survive,
        still holding their ports, and the next invocation degrades for a reason nobody
        can see.
        """
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_terminate)

    hub_port = port if port is not None else DASHBOARD_HUB_PORT
    registry_dir = Path(settings.registry_dir)

    if services.port_in_use(host, hub_port):
        _fail(
            f"port {hub_port} is already in use, so the hub cannot bind it. Free it, or "
            f"choose another: aegis-ml dashboard --port {hub_port + 1}"
        )
        return

    def read_registry() -> tuple[list[Any], str | None]:
        """Read the registry index, returning the rows and any failure to read them."""
        try:
            return list(store.list_runs()), None
        except Exception as exc:  # noqa: BLE001 - surfaced on the page, not swallowed
            return [], f"{type(exc).__name__}: {exc}"

    entries, registry_error = read_registry()
    if registry_error:
        typer.secho(f"registry: {registry_error}", fg=typer.colors.YELLOW, err=True)
    _echo(f"registry: {len(entries)} run(s) in {registry_dir}")

    with services.supervise(host=host, page_port=hub_port, registry_dir=registry_dir) as sup:
        if no_mlflow:
            sup.skip(
                services.MLFLOW_KEY,
                "MLflow",
                "Experiment tracking, run comparison and the model registry.",
                mlflow_port if mlflow_port is not None else DASHBOARD_MLFLOW_PORT,
                "Switched off for this session with --no-mlflow.",
            )
        else:
            if backfill_mlflow and entries:
                _backfill_mlflow(entries, registry_dir)
            _echo("starting MLflow UI …")
            state = sup.start_mlflow(mlflow_port, DASHBOARD_MLFLOW_PORT)
            _report_service(state)

        if no_optuna:
            sup.skip(
                services.OPTUNA_KEY,
                "Optuna Dashboard",
                "Every hyper-parameter trial, including the pruned ones.",
                optuna_port if optuna_port is not None else DASHBOARD_OPTUNA_PORT,
                "Switched off for this session with --no-optuna.",
            )
        else:
            _echo("starting Optuna Dashboard …")
            state = sup.start_optuna(optuna_port, DASHBOARD_OPTUNA_PORT)
            _report_service(state)

        def payload() -> dict[str, Any]:
            """Rebuild the page payload from the registry as it is at this instant."""
            rows, error = read_registry()
            return hub.collect(
                rows,
                registry_dir=registry_dir,
                services={k: s.as_json() for k, s in sup.states.items()},
                registry_error=error,
            )

        httpd = server.build_server(
            host=host,
            port=hub_port,
            page_source=lambda: hub.render(payload()),
            state_source=payload,
            services_source=lambda: {k: s.as_json() for k, s in sup.refresh().items()},
            runs_root=registry_dir / "runs",
            reports_root=registry_dir / "reports",
        )
        url = f"http://{'127.0.0.1' if host in {'0.0.0.0', '::', ''} else host}:{hub_port}/"
        _echo("")
        _echo(f"  hub      {url}")
        for state in sup.states.values():
            mark = "" if state.running else "  (down — see the panel)"
            _echo(f"  {state.key:<8} {state.url}{mark}")
        _echo("")
        _echo("Ctrl-C to stop everything.")
        if not no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            _echo("\nstopping …")
        finally:
            httpd.shutdown()
            httpd.server_close()


def _report_service(state: Any) -> None:  # noqa: ANN401 - dashboard.services.ServiceState
    """Print one service's outcome, with its remedy when it did not come up."""
    if state.running:
        owner = "started" if state.managed else "already running"
        typer.secho(f"  {state.label}: {owner} at {state.url}", fg=typer.colors.GREEN)
        return
    typer.secho(f"  {state.label}: not running — {state.reason}", fg=typer.colors.YELLOW)
    if state.remedy:
        typer.secho(f"    fix: {state.remedy}", fg=typer.colors.YELLOW)


def _backfill_mlflow(entries: list[Any], registry_dir: Path) -> None:
    """Mirror the registry into the dashboard's local MLflow store, reporting the counts.

    Args:
        entries: Registry rows to log.
        registry_dir: Registry root; the store is written under ``<registry_dir>/mlflow``.

    A failure here is printed and the dashboard continues: MLflow is a viewer, and losing
    it must not cost the reader the hub, which is where the authoritative numbers are.
    """
    from aegis_ml.dashboard import services as dashboard_services
    from aegis_ml.registry import mlflow_mirror

    if not is_available("mlflow"):
        typer.secho(
            "  mlflow is not installed, so there is nothing to backfill. "
            "Install it with: uv pip install 'aegis-ml[dashboard]'",
            fg=typer.colors.YELLOW,
        )
        return
    tracking_uri, _ = dashboard_services.mlflow_store_uri(registry_dir)
    _echo(f"backfilling MLflow at {tracking_uri} …")
    try:
        report = mlflow_mirror.backfill(
            entries, tracking_uri=tracking_uri, registry_dir=registry_dir
        )
    except Exception as exc:  # noqa: BLE001 - reported, and the hub continues regardless
        typer.secho(f"  backfill failed: {type(exc).__name__}: {exc}", fg=typer.colors.YELLOW)
        return
    _echo(f"  {report.summary()}")
    for run_id, reason in report.failed:
        typer.secho(f"    {run_id}: {reason}", fg=typer.colors.YELLOW)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover - module execution path
    main()

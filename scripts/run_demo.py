#!/usr/bin/env python3
"""End-to-end demonstration of ``aegis_ml`` on the worked reference domain.

``make demo`` runs this. It is the proof that the whole system works: it generates a
synthetic pharmaceutical cold-chain world, admits it through the data contract, proves the
label is *learnable* and *not too easy*, searches for a model, fits it, measures it on rows
nothing touched, judges it at the promotion gate, and then deliberately breaks the world and
measures the drift.

Every number this prints is measured in this process. Nothing is quoted from a docstring,
nothing is asserted without being computed, and nothing is fabricated when a stage cannot
run — a stage that fails takes the whole script down with a non-zero exit code.

WHAT IT ACTUALLY DOES

    1. **Generate.** ``reference.adapter.generator`` fabricates a seeded world and runs its
       own dependency-free quality gate over it (referential integrity, class coverage on
       both targets, temporal consistency, PII).
    2. **Realism, front and centre.** ``assert_learnable`` and ``realism_report`` are run
       and printed *first*, before anything expensive, because the single most damaging
       failure in this stack is a target that is either noise or trivial, and both are
       cheap to detect and expensive to discover late.
    3. ``data_flow`` — contract, profile, learnability, realism, leakage, three-way split,
       frozen reference frame.
    4. ``train_flow`` — AutoML search across the installed tiers, Optuna HPO over the
       winner, fit, conformal calibration on a split neither the fit nor the test saw, and
       a slice sweep.
    5. ``promote_flow`` — the five-criterion gate. A refusal here is a **successful demo**,
       not a failure, and the script says so and keeps going.
    6. ``drift_flow`` — against a frame deliberately shifted the way this domain actually
       degrades: a hot season on longer, cheaper lanes.
    7. The **secondary classification target** is measured too, so the accuracy band is
       evidenced rather than claimed.

    Finally it writes ``registry_store/RUN_SUMMARY.md`` — the same numbers, in a file a
    reviewer can read without re-running anything.

WHY THE REALISM BLOCK IS THE HEADLINE
    A latent function plus a whisper of noise gives held-out R² ≈ 0.99, and that number is
    a tell: it says the data is a toy, and it collapses the entire "uncertainty you can
    audit" story because a conformal interval calibrated on near-zero residuals is a
    hairline that impresses nobody and informs no decision. This domain calibrates its noise
    to a declared ceiling instead, and prints the evidence: the achieved score, the oracle
    score, the headroom between them, the share of variance carried by unobserved
    confounders, the realised missingness and the columns that were deliberately given no
    driver at all.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Python puts *this file's* directory on ``sys.path``, not the working directory, so
    # ``import reference`` fails for ``python scripts/run_demo.py`` unless the repository
    # root is added here. ``aegis_ml`` resolves through the installed editable package.
    sys.path.insert(0, str(REPO_ROOT))

from reference.adapter import (  # noqa: E402 - same reason
    DOMAIN_DESCRIPTION,
    DOMAIN_ID,
    DOMAIN_SERIES_LABEL,
    DOMAIN_SERIES_UNIT,
    TOOL_REGISTRY,
    GeneratorConfig,
    assess_quality,
    domain_series_events,
    generate_synthetic_sync,
    ml_spec,
)
from reference.problem import (  # noqa: E402 - same reason
    EXCURSION_LATENT,
    EXCURSION_PROBLEM,
    LATENT,
    PROBLEM,
    SEED,
)

from aegis_ml.data.latent import (  # noqa: E402 - must follow the sys.path bootstrap above
    assert_learnable,
    measure_learnability,
    realism_report,
)
from aegis_ml.pipelines.flows import (  # noqa: E402 - same reason
    data_flow,
    drift_flow,
    promote_flow,
    realism_band_for,
    train_flow,
)
from aegis_ml.settings import settings  # noqa: E402 - same reason

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from aegis_ml.contracts.protocols import DriftReport, GateDecision, TrainResult
    from aegis_ml.pipelines.flows import DataBundle

__all__ = ["DemoConfig", "main", "run_demo"]

RULE = "─" * 88
"""Section rule. Plain box-drawing, so the output is readable in a terminal and in a log."""


@dataclass
class DemoConfig:
    """Knobs for one demo run.

    Defaults are chosen so the whole script finishes in a few minutes on a laptop while
    still exercising every stage for real. The two that matter most:

    * ``num_shipments`` — the *generated* world size. Roughly 78% of these are received and
      assayed, so the labelled frame is about four fifths of this number. Enough rows that
      the held-out measurement is a measurement rather than a coin flip.
    * ``time_budget`` — seconds for the AutoML search. Small enough to be a demo, large
      enough that more than one tier actually gets to run and lose.

    Attributes:
        num_shipments: Shipments to fabricate for the training world.
        seed: Seed for generation, splitting, search and fit.
        time_budget: AutoML search budget, in seconds.
        do_hpo: Whether to run the Optuna study over the winning recipe.
        drift_shipments: Shipments to fabricate for the deliberately drifted frame.
        summary_path: Where the run summary is written.
    """

    num_shipments: int = 2600
    seed: int = SEED
    time_budget: int = 45
    do_hpo: bool = True
    drift_shipments: int = 1200
    summary_path: Path = field(
        default_factory=lambda: settings.registry_dir / "RUN_SUMMARY.md"
    )


def _heading(title: str) -> None:
    """Print one section heading.

    Args:
        title: The section title.
    """
    print(f"\n{RULE}\n{title}\n{RULE}")


def _kv(label: str, value: object, *, width: int = 34) -> None:
    """Print one aligned ``label: value`` line.

    Args:
        label: The left-hand label.
        value: Anything renderable.
        width: Label column width.
    """
    print(f"  {label:<{width}} {value}")


def _generate(config: DemoConfig) -> dict[str, Any]:
    """Generate the synthetic world and run the domain's own quality gate over it.

    Args:
        config: The demo knobs.

    Returns:
        A JSON-safe dict of what was generated, for the run summary.

    Raises:
        SystemExit: If the domain's quality gate refuses the dataset. A world that fails
            referential integrity or leaves one excursion class empty is not something to
            train on and report numbers from.
    """
    _heading("1 · GENERATE — a seeded pharmaceutical cold-chain world")
    dataset = generate_synthetic_sync(
        GeneratorConfig(seed=config.seed, num_shipments=config.num_shipments)
    )
    metadata = dataset.metadata
    quality = assess_quality(dataset)

    _kv("domain", DOMAIN_ID)
    _kv("schema version", metadata.schema_version)
    _kv("shipments generated", metadata.num_shipments)
    _kv("labelled (received + assayed)", metadata.num_labelled)
    _kv("carriers / facilities", f"{metadata.num_carriers} / {metadata.num_facilities}")
    _kv("data-logger readings", metadata.num_sensor_readings)
    _kv("seed corpus + generated docs", metadata.num_documents)
    _kv("LLM used for prose", metadata.llm_used)
    print()
    _kv("calibrated for oracle R²", metadata.target_r2)
    _kv("i.i.d. noise σ (percentage pts)", metadata.noise_sigma)
    _kv("unobserved confounder σ", metadata.confounder_sigma)
    _kv("excursion share (realised)", metadata.excursion_share)
    _kv("sensor_gap missing (realised)", metadata.missing_sensor_gap_share)
    print()
    _kv("quality gate", "PASS" if quality.ok else "FAIL")
    _kv("  referential integrity", quality.referential_integrity)
    _kv("  product-class coverage", quality.product_coverage)
    _kv("  both excursion classes", quality.excursion_coverage)
    _kv("  temporal consistency", quality.temporal_consistency)
    _kv("  PII-free (scanned, not assumed)", quality.pii_free)
    _kv("  per-product counts", quality.product_counts)
    _kv("  per-excursion counts", quality.excursion_counts)

    if not quality.ok:
        raise SystemExit(
            "The generated dataset failed the domain's own quality gate. Refusing to train "
            "on it: every number downstream would describe a world that is already known to "
            "be malformed."
        )

    events = domain_series_events(num_records=600, seed=config.seed)
    span_days = (max(t for t, _ in events) - min(t for t, _ in events)).days or 1
    _kv("demand series", f"{DOMAIN_SERIES_LABEL} ({DOMAIN_SERIES_UNIT})")
    _kv("  events / span", f"{len(events)} over {span_days} days")

    return {
        "metadata": metadata.model_dump(mode="json"),
        "quality": quality.model_dump(mode="json"),
        "series": {
            "label": DOMAIN_SERIES_LABEL,
            "unit": DOMAIN_SERIES_UNIT,
            "events": len(events),
            "span_days": span_days,
        },
    }


def _realism(config: DemoConfig) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Prove the label is learnable and honestly hard, before anything expensive runs.

    ``assert_learnable`` is the gate: below the floor it raises, because training on a
    target that carries no signal spends the whole budget discovering that and reports it as
    a leaderboard of models that all failed equally — which reads like a hard problem rather
    than a broken generator.

    The realism report is the *evidence*, and it is printed before the pipeline so a reader
    sees the data's honesty before they see any model's score.

    Args:
        config: The demo knobs.

    Returns:
        ``(frame, regression_evidence, classification_evidence)``.
    """
    _heading("2 · REALISM — is the label learnable, and is it honestly hard?")
    frame = ml_spec.training_frame(num_records=config.num_shipments, seed=config.seed)
    floor, ceiling = realism_band_for(PROBLEM)

    score = assert_learnable(frame, PROBLEM, seed=config.seed)
    evidence = realism_report(frame, PROBLEM, LATENT, seed=config.seed)
    achieved = evidence["achieved"]
    noise = evidence["noise"]
    latent = evidence["latent"]

    print("  PRIMARY TARGET — spoilage_risk_pct (regression, unit '%')")
    _kv("  labelled rows", evidence["n_rows"])
    _kv("  held-out R² (measured)", f"{score:.4f}")
    _kv("  realism band", f"[{floor:.2f}, {ceiling:.2f}]")
    _kv("  in band", floor <= score <= ceiling)
    _kv("  oracle R² (knows the function)", f"{noise['oracle_r2']:.4f}")
    _kv("  headroom (achieved / oracle)", f"{achieved['headroom']:.1%}")
    _kv("  analytic R² ceiling", f"{noise['implied_r2_ceiling']:.4f}")
    _kv("  suspiciously easy?", achieved["suspiciously_easy"])
    print()
    print("  WHY IT CANNOT REACH 1.0 — the five declared realism devices")
    _kv("  1. noise σ (calibrated, not guessed)", f"{noise['sigma']:.3f}")
    _kv("     noise-to-signal ratio", f"{noise['noise_to_signal']:.3f}")
    _kv("  2. unobserved confounders", ", ".join(latent["confounders"]))
    _kv("     share of total variance", f"{noise['confounder_share']:.1%}")
    _kv("  3. heteroscedastic on", noise["heteroscedastic_feature"])
    _kv("     residual spread hi/lo quartile", f"{noise['heteroscedasticity_ratio']:.2f}×")
    _kv("  4. MAR missingness (measured)", evidence.get("missingness") or "none")
    _kv("  5. irrelevant features (no driver)", latent["undriven_features"])
    _kv("     interaction terms", latent["n_interactions"])
    _kv("     non-monotone drivers", latent["non_monotone_drivers"] or "none")

    excursion_frame = ml_spec.excursion_frame(
        num_records=config.num_shipments, seed=config.seed
    )
    report = measure_learnability(excursion_frame, EXCURSION_PROBLEM, seed=config.seed)
    excursion_evidence = realism_report(
        excursion_frame, EXCURSION_PROBLEM, EXCURSION_LATENT, seed=config.seed
    )
    acc_floor, acc_ceiling = realism_band_for(EXCURSION_PROBLEM)
    print()
    print("  SECONDARY TARGET — excursion_flag (classification)")
    _kv("  held-out accuracy (measured)", f"{report.metric_value:.4f}")
    _kv("  majority-class rate", f"{report.majority_share:.4f}")
    _kv("  floor (majority + margin)", f"{report.effective_floor:.4f}")
    _kv("  beats a constant predictor by", f"{report.metric_value - report.majority_share:+.4f}")
    _kv("  realism band", f"[{acc_floor:.2f}, {acc_ceiling:.2f}]")
    _kv("  class balance", excursion_evidence.get("class_balance"))
    _kv("  boundary label-flip rate", ml_spec.LABEL_FLIP_RATE)

    return frame, evidence, {
        "accuracy": report.metric_value,
        "majority_share": report.majority_share,
        "effective_floor": report.effective_floor,
        "band": [acc_floor, acc_ceiling],
        "class_balance": excursion_evidence.get("class_balance"),
    }


def _data(frame: pd.DataFrame, config: DemoConfig) -> DataBundle:
    """Run ``data_flow``: contract, profile, learnability, realism, leakage, split, freeze.

    Args:
        frame: The labelled training frame.
        config: The demo knobs.

    Returns:
        The populated ``DataBundle``.
    """
    _heading("3 · DATA_FLOW — admit the frame, or refuse it")
    bundle = data_flow(PROBLEM, frame, latent=LATENT, seed=config.seed)
    train_rows, calib_rows, test_rows = bundle.sizes
    _kv("contract passed", bundle.contract_ok)
    _kv("leakage findings", bundle.leakage or "none")
    _kv("learnability (held-out)", f"{bundle.learnability:.4f}")
    _kv("train / calibration / test", f"{train_rows} / {calib_rows} / {test_rows}")
    _kv("dataset digest", bundle.digest)
    _kv("frozen reference frame", bundle.reference_path)
    return bundle


def _train(frame: pd.DataFrame, config: DemoConfig) -> TrainResult:
    """Run ``train_flow``: AutoML search, HPO, fit, conformal calibration, slice sweep.

    Args:
        frame: The labelled training frame.
        config: The demo knobs.

    Returns:
        The populated ``TrainResult``.
    """
    _heading("4 · TRAIN_FLOW — search, tune, fit, and measure on rows nothing touched")
    result = train_flow(
        PROBLEM,
        frame,
        latent=LATENT,
        seed=config.seed,
        time_budget=config.time_budget,
        do_hpo=config.do_hpo,
        force=True,
    )
    _kv("run id", result.run_id)
    _kv(f"{result.metric_name} (held-out test split)", f"{result.metric_value:.4f}")
    _kv("coverage requested", f"{result.requested_coverage:.0%}")
    empirical = result.empirical_coverage
    _kv(
        "coverage ACHIEVED (measured)",
        f"{empirical:.1%}" if empirical is not None else "not measured",
    )
    if result.recipe is not None:
        members = ", ".join(f"{m.kind}×{m.weight:g}" for m in result.recipe.members)
        _kv("winning tier", result.recipe.tier)
        _kv("ensemble members", members)
    if result.leaderboard is not None:
        print(
            "\n  Leaderboard — the losers are the point; a search with one entry is not "
            "a search:"
        )
        for candidate in result.leaderboard.candidates:
            mark = "*" if candidate.selected else " "
            print(
                f"   {mark} {candidate.name:<28} {candidate.tier:<10} "
                f"{candidate.metric_name}={candidate.metric_value:.4f} "
                f"({candidate.fit_seconds:.1f}s)"
            )
        for tier, reason in result.leaderboard.tiers_skipped.items():
            print(f"     skipped {tier}: {reason}")
    if result.slices:
        worst = min(result.slices, key=lambda s: s.metric_value)
        _kv(
            "\n  worst slice",
            f"{worst.feature}={worst.level} → {worst.metric_name}="
            f"{worst.metric_value:.4f} on {worst.n_rows} rows",
            width=32,
        )
    _kv("artifact", result.artifact_path)
    return result


def _promote(run_id: str) -> GateDecision:
    """Run ``promote_flow``: the five-criterion gate.

    A refusal is a successful demonstration of the gate, not a failed demo. The decision
    carries every number behind it either way, which is the whole point: "promoted" with no
    figures is exactly as opaque as "rejected" with no figures.

    Args:
        run_id: The challenger run.

    Returns:
        The ``GateDecision``.
    """
    _heading("5 · PROMOTE_FLOW — the gate, with the numbers behind it")
    decision = promote_flow(run_id)
    _kv("promoted", decision.promoted)
    _kv("challenger", decision.challenger_run_id)
    _kv("champion", decision.champion_run_id or "none (first model in this domain)")
    print("\n  Criteria:")
    for name, passed in decision.checks.items():
        print(f"   {'PASS' if passed else 'FAIL'}  {name}")
    print("\n  Numbers:")
    for name, value in decision.metrics.items():
        print(f"     {name:<32} {value:.4f}")
    print("\n  Reasons:")
    for reason in decision.reasons:
        print(f"     - {reason}")
    if not decision.promoted:
        print(
            "\n  A refusal here is the gate working. It is reported, not overridden: "
            "`force=True` exists and is deliberately not used by this demo."
        )
    return decision


def _drift(run_id: str, config: DemoConfig) -> DriftReport:
    """Build a deliberately drifted frame and run ``drift_flow`` against the frozen reference.

    The shift is not random noise: it is the way this domain actually degrades. A hot season
    arrives (ambient up), procurement moves volume onto cheaper, longer, more-transferred
    lanes (carrier tier down, transit up, handoffs up), and telemetry cadence worsens with
    it. Every one of those moves a real driver, so the drift detector should see it and the
    performance estimator should say it hurt.

    Args:
        run_id: The registered run whose frozen reference frame is the baseline.
        config: The demo knobs.

    Returns:
        The ``DriftReport``.
    """
    _heading("6 · DRIFT_FLOW — a hot season on cheaper, longer lanes")
    drifted = ml_spec.training_frame(
        num_records=config.drift_shipments, seed=config.seed + 101
    )
    before = {
        "ambient_temp_c": float(drifted["ambient_temp_c"].mean()),
        "transit_hours": float(drifted["transit_hours"].mean()),
        "handoff_count": float(drifted["handoff_count"].mean()),
        "economy_share": float((drifted["carrier_tier"] == "economy").mean()),
    }

    drifted["ambient_temp_c"] = (drifted["ambient_temp_c"] + 9.0).clip(upper=40.0)
    drifted["transit_hours"] = (drifted["transit_hours"] * 1.35).clip(upper=132.0)
    drifted["handoff_count"] = (drifted["handoff_count"] + 1).clip(upper=7)
    drifted["sensor_gap_minutes"] = (drifted["sensor_gap_minutes"] * 1.6).clip(upper=540.0)
    # Two thirds of the book moves to the cheapest tier — the commercial decision that makes
    # the rest of this shift happen at all.
    downgrade = drifted.index[: int(len(drifted) * 0.66)]
    drifted.loc[downgrade, "carrier_tier"] = "economy"
    drifted.loc[downgrade, "route_class"] = "multi_leg"

    after = {
        "ambient_temp_c": float(drifted["ambient_temp_c"].mean()),
        "transit_hours": float(drifted["transit_hours"].mean()),
        "handoff_count": float(drifted["handoff_count"].mean()),
        "economy_share": float((drifted["carrier_tier"] == "economy").mean()),
    }
    print("  Shift applied (mean before → after):")
    for name, value in before.items():
        print(f"     {name:<24} {value:>8.3f} → {after[name]:>8.3f}")
    print()

    report = drift_flow(run_id, drifted)
    _kv("dataset drift detected", report.dataset_drift)
    _kv("verdict", report.verdict)
    _kv("share of features drifted", f"{report.drifted_share:.1%}")
    _kv("drifted features", report.drifted_features or "none")
    _kv("reference / current rows", f"{report.n_reference_rows} / {report.n_current_rows}")
    if report.estimated_metric_name:
        _kv(
            f"ESTIMATED {report.estimated_metric_name} (no labels)",
            f"{report.estimated_metric_value:.4f}"
            if report.estimated_metric_value is not None
            else "not estimated",
        )
        print(
            "     'estimated' is not 'measured': this is the metric inferred from the "
            "model's own confidence under the observed shift, before any ground truth "
            "has arrived."
        )
    _kv("html report", report.html_report_path or "not written")
    print(
        "\n  A drifted model is NOT withdrawn. Aegis serves the model it has and flags it; "
        "what drift blocks is the promotion of anything calibrated on a reference that no "
        "longer describes the world."
    )
    return report


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write the run summary a reviewer can read without re-running anything.

    Args:
        path: Where to write it.
        payload: The collected numbers.
    """
    generated = payload["generated"]["metadata"]
    realism = payload["realism"]
    achieved = realism["achieved"]
    noise = realism["noise"]
    latent = realism["latent"]
    excursion = payload["excursion"]
    train = payload["train"]
    gate = payload["gate"]
    drift = payload["drift"]

    lines = [
        f"# {DOMAIN_ID} — end-to-end run summary",
        "",
        f"Generated {payload['finished_at']} by `scripts/run_demo.py`. Every number below "
        "was measured in that run.",
        "",
        f"> {DOMAIN_DESCRIPTION}",
        "",
        "## Data",
        "",
        f"- shipments generated: **{generated['num_shipments']}**, labelled "
        f"(received and assayed): **{generated['num_labelled']}**",
        f"- calibrated for an oracle R² of **{generated['target_r2']}**; i.i.d. noise σ "
        f"**{generated['noise_sigma']}** percentage points, unobserved-confounder σ "
        f"**{generated['confounder_sigma']}**",
        f"- realised excursion share **{generated['excursion_share']}**, realised "
        f"`sensor_gap_minutes` missingness **{generated['missing_sensor_gap_share']}**",
        f"- domain quality gate: **{'PASS' if payload['generated']['quality']['ok'] else 'FAIL'}**",
        "",
        "## Realism — primary target `spoilage_risk_pct` (regression, `%`)",
        "",
        "| measure | value |",
        "|---|---|",
        f"| held-out R² (measured) | **{achieved['value']:.4f}** |",
        f"| realism band | [{achieved['floor']:.2f}, {achieved['ceiling']:.2f}] |",
        f"| oracle R² (knows the generating function) | {noise['oracle_r2']:.4f} |",
        f"| headroom (achieved ÷ oracle) | {achieved['headroom']:.1%} |",
        f"| analytic R² ceiling | {noise['implied_r2_ceiling']:.4f} |",
        f"| suspiciously easy? | {achieved['suspiciously_easy']} |",
        f"| noise σ (calibrated) | {noise['sigma']:.3f} |",
        f"| noise-to-signal | {noise['noise_to_signal']:.3f} |",
        f"| unobserved confounders | {', '.join(latent['confounders'])} |",
        f"| confounder share of variance | {noise['confounder_share']:.1%} |",
        f"| heteroscedastic on | `{noise['heteroscedastic_feature']}` "
        f"({noise['heteroscedasticity_ratio']:.2f}× spread, top vs bottom quartile) |",
        f"| MAR missingness (measured) | {realism.get('missingness')} |",
        f"| features with NO driver | {', '.join(latent['undriven_features'])} |",
        f"| interaction terms | {latent['n_interactions']} |",
        "",
        "## Realism — secondary target `excursion_flag` (classification)",
        "",
        f"- held-out accuracy: **{excursion['accuracy']:.4f}** "
        f"(band [{excursion['band'][0]:.2f}, {excursion['band'][1]:.2f}])",
        f"- majority-class rate {excursion['majority_share']:.4f}, so the floor is "
        f"{excursion['effective_floor']:.4f} — the classifier clears a constant predictor "
        f"by {excursion['accuracy'] - excursion['majority_share']:+.4f}",
        f"- class balance: {excursion['class_balance']}",
        "",
        "## Train",
        "",
        f"- run id: `{train['run_id']}`",
        f"- {train['metric_name']} on the held-out test split: **{train['metric_value']:.4f}**",
        f"- conformal coverage requested {train['requested_coverage']:.0%}, "
        f"**achieved {train['empirical_coverage']:.1%}**"
        if train["empirical_coverage"] is not None
        else f"- conformal coverage requested {train['requested_coverage']:.0%}",
        f"- winning tier: `{train['tier']}`; rows train/calibration/test: {train['sizes']}",
        f"- worst slice: {train['worst_slice']}",
        "",
        "### Leaderboard",
        "",
        "| model | tier | metric | fit (s) | selected |",
        "|---|---|---|---|---|",
    ]
    for candidate in train["leaderboard"]:
        lines.append(
            f"| {candidate['name']} | {candidate['tier']} | "
            f"{candidate['metric_name']}={candidate['metric_value']:.4f} | "
            f"{candidate['fit_seconds']:.1f} | {'yes' if candidate['selected'] else ''} |"
        )
    lines += [
        "",
        "## Promotion gate",
        "",
        f"- **promoted: {gate['promoted']}** (champion: {gate['champion'] or 'none'})",
        "",
        "| criterion | verdict |",
        "|---|---|",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "Reasons:", ""]
    lines += [f"- {reason}" for reason in gate["reasons"]]
    lines += [
        "",
        "## Drift",
        "",
        "Against a frame shifted the way this domain actually degrades — a hot season, "
        "volume moved onto cheaper and longer multi-leg lanes, telemetry cadence worsening "
        "with it.",
        "",
        f"- dataset drift detected: **{drift['dataset_drift']}** (verdict `{drift['verdict']}`)",
        f"- {drift['drifted_share']:.1%} of features drifted: {drift['drifted_features']}",
        f"- estimated {drift['estimated_metric_name']}: {drift['estimated_metric_value']}"
        if drift["estimated_metric_name"]
        else "- no labelless performance estimate was produced",
        "",
        "## Agent surface",
        "",
        f"- tools registered: {payload['tools']['count']} "
        f"({payload['tools']['domain']} domain + {payload['tools']['ml']} ML)",
        f"- ML tools reaching the agent loop: {', '.join(payload['tools']['ml_names'])}",
        f"- forecast series: {DOMAIN_SERIES_LABEL} ({DOMAIN_SERIES_UNIT})",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    """Run every stage end to end and return the collected, measured numbers.

    Args:
        config: The demo knobs; defaults apply when omitted.

    Returns:
        A JSON-safe dict of everything measured, also written to
        :attr:`DemoConfig.summary_path`.

    Raises:
        SystemExit: If the generated dataset fails the domain's own quality gate.
        Exception: Anything a flow raises is allowed to propagate — a demo that swallows a
            pipeline failure and prints a summary anyway is worse than no demo.
    """
    resolved = config or DemoConfig()

    print(RULE)
    print("aegis_ml · end-to-end demonstration on the worked reference domain")
    print(f"domain: {DOMAIN_ID}    registry: {settings.registry_dir}")
    print(RULE)

    generated = _generate(resolved)
    frame, realism, excursion = _realism(resolved)
    bundle = _data(frame, resolved)
    result = _train(frame, resolved)
    decision = _promote(result.run_id)
    drift = _drift(result.run_id, resolved)

    ml_names = sorted(name for name in TOOL_REGISTRY if name in _ML_TOOL_NAMES)
    payload: dict[str, Any] = {
        "finished_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "domain_id": DOMAIN_ID,
        "generated": generated,
        "realism": realism,
        "excursion": excursion,
        "train": {
            "run_id": result.run_id,
            "metric_name": result.metric_name,
            "metric_value": result.metric_value,
            "requested_coverage": result.requested_coverage,
            "empirical_coverage": result.empirical_coverage,
            "tier": result.recipe.tier if result.recipe else None,
            "sizes": list(bundle.sizes),
            "worst_slice": (
                f"{min(result.slices, key=lambda s: s.metric_value).feature}="
                f"{min(result.slices, key=lambda s: s.metric_value).level} → "
                f"{min(result.slices, key=lambda s: s.metric_value).metric_value:.4f}"
                if result.slices
                else "no slice sweep"
            ),
            "leaderboard": [
                candidate.model_dump(mode="json")
                for candidate in (result.leaderboard.candidates if result.leaderboard else [])
            ],
        },
        "gate": {
            "promoted": decision.promoted,
            "champion": decision.champion_run_id,
            "checks": decision.checks,
            "metrics": decision.metrics,
            "reasons": decision.reasons,
        },
        "drift": {
            "dataset_drift": drift.dataset_drift,
            "verdict": drift.verdict,
            "drifted_share": drift.drifted_share,
            "drifted_features": drift.drifted_features,
            "estimated_metric_name": drift.estimated_metric_name,
            "estimated_metric_value": drift.estimated_metric_value,
        },
        "tools": {
            "count": len(TOOL_REGISTRY),
            "domain": len(TOOL_REGISTRY) - len(ml_names),
            "ml": len(ml_names),
            "ml_names": ml_names,
        },
    }

    _write_summary(resolved.summary_path, payload)

    _heading("7 · DONE")
    _kv("run summary written to", resolved.summary_path)
    _kv("held-out R² (primary)", f"{result.metric_value:.4f}")
    _kv("held-out accuracy (secondary)", f"{excursion['accuracy']:.4f}")
    _kv("promoted", decision.promoted)
    _kv("drift verdict", drift.verdict)
    print(
        "\n  Every figure above was computed in this process. The realism block in "
        "section 2 is the one to read first: it is what says this data is honest."
    )
    return payload


_ML_TOOL_NAMES = frozenset(
    {
        "predict_outcome",
        "explain_prediction",
        "whatif_scenario",
        "forecast_series",
        "check_model_health",
    }
)
"""The five ML tool names, for splitting the registry count in the summary.

Spelled out rather than imported so the summary line stays truthful even if the serving
extra is not installed: it counts what is *in this registry*, which is the thing the
sentence claims.
"""


def main() -> int:
    """Entry point: run the demo, and exit non-zero on any failure.

    Returns:
        ``0`` when every stage completed, ``1`` otherwise. The exit code is the contract
        ``make demo`` and CI read; a demo that prints a traceback and exits 0 is a demo that
        will be believed when it should not be.
    """
    try:
        run_demo()
    except SystemExit as exc:
        print(f"\nDEMO FAILED: {exc}", file=sys.stderr)
        return 1
    except BaseException:
        print("\nDEMO FAILED — traceback follows:\n", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

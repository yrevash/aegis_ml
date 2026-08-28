"""The promotion gate: five criteria, every number recorded, nothing promoted silently.

This is the MLOps heart of the package. Promotion means atomically replacing
``backend/.artifacts/ml_spine.joblib`` — the file ``aegis.ml.get_model()`` loads — so this
decision is the last point at which a worse model can be stopped, and it is the one place
where "it looked better" must be replaced by "here is the number, here is the threshold,
here is the direction".

**All five must hold. Every one is reported with its figure, on a pass as well as a
failure.** A ``GateDecision`` that says ``promoted=True`` with no numbers is exactly as
opaque as one that says ``promoted=False`` with no numbers, and the model card quotes both.

1. **Beats the champion on the primary metric by at least** ``settings.promote_min_gain``.
   Direction comes from :data:`~aegis_ml.evaluate.metrics.HIGHER_IS_BETTER`, never from
   ``>``. The margin exists because on genuinely noisy data (held-out R² in the 0.45–0.80
   band this package targets) fold-to-fold movement of a point or two is normal, and
   promoting on it is promoting noise. A first model with no champion passes this
   criterion trivially — and the reason string *says* it passed trivially, because
   "promoted, beat the champion" written about a run with no champion is a lie the card
   would then repeat.
2. **Measured coverage clears the requested level minus** ``settings.coverage_tolerance``.
   Requested and measured are two separate fields the whole way through. A missing
   empirical coverage FAILS this check by default: unmeasured is not the same as met, and
   defaulting the other way promotes an uncalibrated interval into a system whose entire
   value proposition is the calibrated interval.
3. **All data contracts passed.** The pandera contract is what stands between the model and
   a frame whose columns silently changed meaning.
4. **The worst slice is no worse than the champion's worst slice.** Deliberately the worst
   and not the mean: *a model that improves on average while collapsing on one region is a
   regression for everyone in that region, and an aggregate score is exactly the instrument
   that cannot see it.* The collapsed region's error is diluted by its own small share of
   the rows, so the headline number moves in the right direction while the experience of
   that population gets worse.
5. **No target leakage was flagged.** A leaking feature produces the best held-out score in
   the run and the worst behaviour in production, because the leaked column is not there at
   prediction time. Criterion 1 actively *rewards* leakage, so this criterion has to be able
   to overrule it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from aegis_ml.contracts.errors import PromotionRejectedError
from aegis_ml.contracts.protocols import GateDecision, SliceMetric, TrainResult
from aegis_ml.evaluate.metrics import higher_is_better
from aegis_ml.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "CRITERIA",
    "GateConfig",
    "evaluate_gate",
    "format_decision",
    "promote_or_raise",
]

CRITERIA: tuple[str, ...] = (
    "beats_champion",
    "coverage_meets_request",
    "contracts_pass",
    "worst_slice_not_worse",
    "no_target_leakage",
)
"""The five check keys, in the order they are evaluated and reported.

Named as a constant so the registry, the CLI and the model card iterate the same list and a
criterion can never be quietly dropped from one consumer's view of the decision.
"""


class GateConfig(BaseModel):
    """Thresholds for one gate evaluation, defaulting to the process settings.

    Defaults are resolved at *call* time via ``default_factory`` rather than captured at
    import time, so ``AEGIS_ML_PROMOTE_MIN_GAIN`` set for one pipeline run takes effect for
    that run instead of whichever value happened to exist when the module was first
    imported.

    Attributes:
        min_gain: How much better than the champion the challenger must be, in the primary
            metric's own units, in the metric's improving direction.
        coverage_tolerance: Allowed shortfall of measured against requested coverage. This
            stands for the sampling error of the coverage measurement, not for slack.
        slice_tolerance: How much worse than the champion's worst slice the challenger's
            worst slice may be. Defaults to ``0.0`` — no regression permitted. Widen it
            only with a stated reason, because this is the criterion that protects the
            population the aggregate cannot see.
        require_empirical_coverage: When ``True`` (the default), a challenger with no
            measured coverage fails criterion 2. Unmeasured is not met.
        require_contracts: When ``False``, criterion 3 is reported as skipped rather than
            silently passed — the check key is still present and still ``False``.
    """

    min_gain: float = Field(default_factory=lambda: settings.promote_min_gain, ge=0.0)
    coverage_tolerance: float = Field(
        default_factory=lambda: settings.coverage_tolerance, ge=0.0
    )
    slice_tolerance: float = Field(default=0.0, ge=0.0)
    require_empirical_coverage: bool = True
    require_contracts: bool = True


def _leak_name(item: object) -> str:
    """Render one leakage finding as a name, whatever shape the detector returned.

    Args:
        item: A string feature name, a mapping with a ``feature`` key, or an object with a
            ``.feature`` attribute (e.g. a ``TargetLeakageError``).

    Returns:
        The best available human name for the flagged feature. Never raises: a leakage
        finding that could not be rendered must still block the promotion, so an
        unrecognised shape falls through to ``repr``.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        name = item.get("feature") or item.get("name")
        if name is not None:
            return str(name)
    name = getattr(item, "feature", None)
    if name is not None:
        return str(name)
    return repr(item)


def _worst(slices: Sequence[SliceMetric], metric_name: str) -> SliceMetric | None:
    """Return the worst slice among those measuring ``metric_name``.

    Slices naming a different metric are ignored rather than compared: a run whose slice
    sweep used ``rmse`` and a champion whose sweep used ``r2`` have no comparable worst
    slice, and mixing them would compare 0.6 against 4.2 as if that meant something.

    Args:
        slices: Measured slices from a :class:`~aegis_ml.contracts.protocols.TrainResult`.
        metric_name: The primary metric the comparison is being made in.

    Returns:
        The worst comparable slice, or ``None`` when there is none.
    """
    comparable = [s for s in slices if s.metric_name == metric_name]
    if not comparable:
        return None
    if higher_is_better(metric_name):
        return min(comparable, key=lambda s: s.metric_value)
    return max(comparable, key=lambda s: s.metric_value)


def _check_metric(
    challenger: TrainResult,
    champion: TrainResult | None,
    config: GateConfig,
    checks: dict[str, bool],
    metrics: dict[str, float],
    reasons: list[str],
) -> None:
    """Criterion 1 — the primary metric, with the direction taken from the registry.

    Args:
        challenger: The candidate for promotion.
        champion: The incumbent, or ``None`` for the first model in a domain.
        config: Thresholds for this evaluation.
        checks: Mutated in place with ``beats_champion``.
        metrics: Mutated in place with the challenger/champion values and the margin.
        reasons: Mutated in place with the human sentence, pass or fail.
    """
    name = challenger.metric_name
    metrics["min_gain"] = float(config.min_gain)
    metrics[f"challenger_{name}"] = float(challenger.metric_value)

    try:
        direction = higher_is_better(name)
    except Exception as exc:  # noqa: BLE001 - an unrankable metric must fail, not crash
        checks["beats_champion"] = False
        reasons.append(
            f"FAIL beats_champion: primary metric {name!r} has no declared direction "
            f"({exc}). Nothing here will guess whether larger is better — guessing "
            f"promotes the worse model with a straight face."
        )
        return

    if champion is None:
        checks["beats_champion"] = True
        reasons.append(
            f"PASS beats_champion (trivially): no champion exists for domain "
            f"{challenger.domain_id!r}, so there is nothing to beat. This model is "
            f"promoted as the first baseline at {name}={challenger.metric_value:.4f}, NOT "
            f"because it outperformed anything. Every later challenger is measured "
            f"against this number."
        )
        return

    if champion.metric_name != name:
        checks["beats_champion"] = False
        reasons.append(
            f"FAIL beats_champion: challenger is scored on {name!r} and the champion on "
            f"{champion.metric_name!r}. Two metrics are two scales; comparing them would "
            f"produce a real number that means nothing. Re-evaluate the champion on "
            f"{name!r} before promoting."
        )
        return

    champion_value = float(champion.metric_value)
    metrics[f"champion_{name}"] = champion_value
    gain = (
        challenger.metric_value - champion_value
        if direction
        else champion_value - challenger.metric_value
    )
    metrics["gain"] = float(gain)
    passed = gain >= config.min_gain
    checks["beats_champion"] = passed
    better_word = "higher" if direction else "lower"
    if passed:
        reasons.append(
            f"PASS beats_champion: {name} {challenger.metric_value:.4f} vs champion "
            f"{champion_value:.4f} — a gain of {gain:+.4f} in the improving direction "
            f"({better_word} is better), clearing the required margin of "
            f"{config.min_gain:.4f}."
        )
    else:
        reasons.append(
            f"FAIL beats_champion: {name} {challenger.metric_value:.4f} vs champion "
            f"{champion_value:.4f} — a gain of {gain:+.4f} against a required margin of "
            f"{config.min_gain:.4f} ({better_word} is better). The margin exists because "
            f"run-to-run movement of this size is noise on this data; promoting into it "
            f"replaces a known model with an unknown one for nothing."
        )


def _check_coverage(
    challenger: TrainResult,
    config: GateConfig,
    checks: dict[str, bool],
    metrics: dict[str, float],
    reasons: list[str],
) -> None:
    """Criterion 2 — measured conformal coverage against the level that was requested.

    Args:
        challenger: The candidate for promotion; supplies both coverage fields.
        config: Thresholds for this evaluation.
        checks: Mutated in place with ``coverage_meets_request``.
        metrics: Mutated in place with requested, measured and the floor.
        reasons: Mutated in place with the human sentence, pass or fail.
    """
    requested = float(challenger.requested_coverage)
    floor = requested - config.coverage_tolerance
    metrics["requested_coverage"] = requested
    metrics["coverage_tolerance"] = float(config.coverage_tolerance)
    metrics["coverage_floor"] = float(floor)

    if challenger.empirical_coverage is None:
        if config.require_empirical_coverage:
            checks["coverage_meets_request"] = False
            reasons.append(
                f"FAIL coverage_meets_request: requested {requested:.3f} but NOTHING WAS "
                f"MEASURED — no held-out coverage was computed for this run. Unmeasured is "
                f"not met. The calibrated interval is the product here; promoting one "
                f"nobody has checked ships the claim without the evidence."
            )
        else:
            checks["coverage_meets_request"] = True
            reasons.append(
                f"PASS coverage_meets_request (WAIVED): requested {requested:.3f}, no "
                f"empirical coverage measured, and require_empirical_coverage=False was "
                f"set for this evaluation. The interval this model serves is UNVERIFIED."
            )
        return

    empirical = float(challenger.empirical_coverage)
    metrics["empirical_coverage"] = empirical
    metrics["coverage_gap"] = empirical - requested
    passed = empirical >= floor
    checks["coverage_meets_request"] = passed
    if passed:
        reasons.append(
            f"PASS coverage_meets_request: measured {empirical:.3f} against a requested "
            f"{requested:.3f} (floor {floor:.3f} = requested − tolerance "
            f"{config.coverage_tolerance:.3f}), on {challenger.test_size} held-out rows."
        )
    else:
        reasons.append(
            f"FAIL coverage_meets_request: measured {empirical:.3f} against a requested "
            f"{requested:.3f} (floor {floor:.3f}). The interval is narrower than the data "
            f"supports — under heteroscedastic noise a single calibrated width under-covers "
            f"exactly where the target is hardest to predict, which is where a decision "
            f"taken on it costs the most."
        )


def _check_contracts(
    contract_ok: bool,
    config: GateConfig,
    checks: dict[str, bool],
    metrics: dict[str, float],
    reasons: list[str],
) -> None:
    """Criterion 3 — the data contracts that validated the training frame.

    Args:
        contract_ok: Whether every pandera contract passed for this run.
        config: Thresholds; ``require_contracts=False`` records a skip, not a pass.
        checks: Mutated in place with ``contracts_pass``.
        metrics: Mutated in place with a 0/1 indicator so the value is machine-readable.
        reasons: Mutated in place with the human sentence, pass or fail.
    """
    metrics["contract_ok"] = 1.0 if contract_ok else 0.0
    if not config.require_contracts:
        checks["contracts_pass"] = False
        reasons.append(
            "SKIPPED contracts_pass: contract checking was disabled for this evaluation "
            "(require_contracts=False), so this criterion is recorded as NOT passed rather "
            "than quietly satisfied. A disabled check and a passed check must never look "
            "the same in the decision record."
        )
        return
    checks["contracts_pass"] = bool(contract_ok)
    if contract_ok:
        reasons.append(
            "PASS contracts_pass: every declared data contract validated the training "
            "frame — dtypes, ranges, null policy and categorical level sets."
        )
    else:
        reasons.append(
            "FAIL contracts_pass: at least one data contract failed on the training frame. "
            "The model may be perfectly fitted to a frame whose columns no longer mean what "
            "the spec says they mean, and no accuracy metric can detect that."
        )


def _check_slices(
    challenger: TrainResult,
    champion: TrainResult | None,
    config: GateConfig,
    checks: dict[str, bool],
    metrics: dict[str, float],
    reasons: list[str],
) -> None:
    """Criterion 4 — the WORST slice, compared against the champion's worst slice.

    Why the worst and not the mean: a model that improves on average while collapsing on one
    region is a regression for everyone in that region, and an aggregate score is exactly
    the instrument that cannot see it. The collapsed region contributes only its own share
    of the rows to the headline number, so a large local failure moves the average by very
    little — and the average is the thing being watched.

    Args:
        challenger: The candidate for promotion.
        champion: The incumbent, or ``None``.
        config: Thresholds for this evaluation.
        checks: Mutated in place with ``worst_slice_not_worse``.
        metrics: Mutated in place with both worst-slice values and their delta.
        reasons: Mutated in place with the human sentence, pass or fail.
    """
    name = challenger.metric_name
    challenger_worst = _worst(challenger.slices, name)

    if challenger_worst is None:
        checks["worst_slice_not_worse"] = False
        reasons.append(
            f"FAIL worst_slice_not_worse: the challenger reports no slice measured in "
            f"{name!r}, so there is no worst segment to compare. An unmeasured segment "
            f"distribution is missing evidence, not a pass — run "
            f"`aegis_ml.evaluate.slices.slice_report` on the held-out split."
        )
        return

    metrics["challenger_worst_slice"] = float(challenger_worst.metric_value)
    metrics["challenger_worst_slice_rows"] = float(challenger_worst.n_rows)
    where = f"{challenger_worst.feature}={challenger_worst.level} (n={challenger_worst.n_rows})"

    champion_worst = None if champion is None else _worst(champion.slices, name)
    if champion_worst is None:
        checks["worst_slice_not_worse"] = True
        why = (
            "no champion exists"
            if champion is None
            else f"the champion reports no slice measured in {name!r}"
        )
        reasons.append(
            f"PASS worst_slice_not_worse (no baseline): {why}, so the challenger's worst "
            f"segment cannot have regressed. Its worst segment is {where} at "
            f"{challenger_worst.metric_value:.4f} — this becomes the floor every later "
            f"challenger must hold."
        )
        return

    metrics["champion_worst_slice"] = float(champion_worst.metric_value)
    direction = higher_is_better(name)
    delta = (
        challenger_worst.metric_value - champion_worst.metric_value
        if direction
        else champion_worst.metric_value - challenger_worst.metric_value
    )
    metrics["worst_slice_delta"] = float(delta)
    passed = delta >= -config.slice_tolerance
    checks["worst_slice_not_worse"] = passed
    champion_where = f"{champion_worst.feature}={champion_worst.level}"
    if passed:
        reasons.append(
            f"PASS worst_slice_not_worse: worst segment {where} scores "
            f"{challenger_worst.metric_value:.4f} against the champion's worst "
            f"({champion_where}) at {champion_worst.metric_value:.4f} — {delta:+.4f} in the "
            f"improving direction, within the tolerance of {config.slice_tolerance:.4f}."
        )
    else:
        reasons.append(
            f"FAIL worst_slice_not_worse: worst segment {where} scores "
            f"{challenger_worst.metric_value:.4f} against the champion's worst "
            f"({champion_where}) at {champion_worst.metric_value:.4f} — {delta:+.4f}, a "
            f"regression beyond the tolerance of {config.slice_tolerance:.4f}. The headline "
            f"metric may still have improved; that is precisely the case this criterion "
            f"exists for, because the rows in this segment experience the model as worse "
            f"and the aggregate cannot show it."
        )


def _check_leakage(
    leakage: Sequence[object] | None,
    checks: dict[str, bool],
    metrics: dict[str, float],
    reasons: list[str],
) -> None:
    """Criterion 5 — target leakage flagged by the feature audit.

    Args:
        leakage: Findings from ``aegis_ml.features.leakage``. An empty sequence means the
            audit **ran and found nothing**; ``None`` means it **never ran**, which is not
            the same thing and must not be reported as if it were.
            carrying a ``feature`` attribute are all understood.
        checks: Mutated in place with ``no_target_leakage``.
        metrics: Mutated in place with the finding count.
        reasons: Mutated in place with the human sentence, pass or fail.
    """
    if leakage is None:
        # An audit that never ran is not a clean audit. Reading a missing input as a pass is
        # the exact failure this package exists to prevent: the gate would print "the feature
        # audit flagged nothing" about an audit that did not happen, and a leaking feature —
        # which criterion 1 actively REWARDS, because it produces the best held-out score —
        # would sail through the one criterion written to catch it.
        metrics["leakage_findings"] = float("nan")
        checks["no_target_leakage"] = False
        reasons.append(
            "FAIL no_target_leakage: the leakage audit did not run for this run, so nothing "
            "is known about it. UNPROVEN is not PASS. Re-run `aegis-ml contract` to produce "
            "gate_inputs.json, or pass leakage=[] explicitly to assert the audit ran clean."
        )
        return
    findings = list(leakage)
    metrics["leakage_findings"] = float(len(findings))
    checks["no_target_leakage"] = not findings
    if not findings:
        reasons.append(
            "PASS no_target_leakage: the feature audit flagged nothing. No single feature "
            "predicts the target well enough to be suspected of carrying it."
        )
        return
    names = ", ".join(_leak_name(item) for item in findings)
    reasons.append(
        f"FAIL no_target_leakage: {len(findings)} finding(s) — {names}. A leaking feature "
        f"produces the best held-out score in the run and the worst behaviour in "
        f"production, because the column is not available at prediction time. Criterion 1 "
        f"actively rewards this, which is why it cannot be the only criterion."
    )


def evaluate_gate(
    challenger: TrainResult,
    champion: TrainResult | None,
    *,
    contract_ok: bool,
    leakage: Sequence[object] | None,
    config: GateConfig | None = None,
) -> GateDecision:
    """Run all five criteria and return the full decision, numbers included.

    Every criterion is evaluated even after one has failed. Short-circuiting would produce a
    decision record that names the first problem and hides the rest, and the second problem
    is the one that reappears after the first is fixed.

    Args:
        challenger: The candidate for promotion, carrying its measured metric, its
            requested and empirical coverage, and its slice sweep.
        champion: The incumbent production model, or ``None`` for the first model in a
            domain — which passes criteria 1 and 4 trivially, and says so in ``reasons``.
        contract_ok: Whether every data contract passed for the challenger's frame.
        leakage: Leakage findings; empty means none were flagged.
        config: Thresholds; defaults to the process settings resolved at call time.

    Returns:
        A :class:`~aegis_ml.contracts.protocols.GateDecision` whose ``checks`` carries all
        five criteria, ``metrics`` every number behind them, and ``reasons`` one sentence
        per criterion — on a pass as well as a failure.
    """
    cfg = config or GateConfig()
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    reasons: list[str] = []

    _check_metric(challenger, champion, cfg, checks, metrics, reasons)
    _check_coverage(challenger, cfg, checks, metrics, reasons)
    _check_contracts(contract_ok, cfg, checks, metrics, reasons)
    _check_slices(challenger, champion, cfg, checks, metrics, reasons)
    _check_leakage(leakage, checks, metrics, reasons)

    missing = [key for key in CRITERIA if key not in checks]
    if missing:  # pragma: no cover - a criterion silently skipped is a bug in this module
        raise RuntimeError(
            f"Gate criteria {missing} produced no verdict. A decision that omits a "
            f"criterion is not a decision; refusing to return it."
        )

    promoted = all(checks[key] for key in CRITERIA)
    passed_count = sum(1 for key in CRITERIA if checks[key])
    reasons.append(
        f"{'PROMOTED' if promoted else 'REJECTED'}: {passed_count}/{len(CRITERIA)} criteria "
        f"passed. All five are required — they cover different failure modes and none "
        f"substitutes for another."
    )
    return GateDecision(
        promoted=promoted,
        challenger_run_id=challenger.run_id,
        champion_run_id=None if champion is None else champion.run_id,
        reasons=reasons,
        checks=checks,
        metrics=metrics,
    )


def promote_or_raise(
    challenger: TrainResult,
    champion: TrainResult | None,
    *,
    contract_ok: bool,
    leakage: Sequence[object],
    config: GateConfig | None = None,
) -> GateDecision:
    """Evaluate the gate and raise when the challenger is rejected.

    The two entry points exist for two callers. A pipeline that wants to *record* the
    decision (and publish it on the model card either way) calls :func:`evaluate_gate`; a
    pipeline whose next statement would overwrite the served artifact calls this one, so
    that a rejection cannot be stepped over by a caller who forgot to inspect ``.promoted``.

    Args:
        challenger: The candidate for promotion.
        champion: The incumbent, or ``None``.
        contract_ok: Whether every data contract passed.
        leakage: Leakage findings; empty means none were flagged.
        config: Thresholds; defaults to the process settings.

    Returns:
        The passing :class:`~aegis_ml.contracts.protocols.GateDecision`.

    Raises:
        PromotionRejectedError: When any criterion failed, carrying every failed criterion's
            sentence and its measured number.
    """
    decision = evaluate_gate(
        challenger,
        champion,
        contract_ok=contract_ok,
        leakage=leakage,
        config=config,
    )
    if not decision.promoted:
        failures = [
            reason
            for reason in decision.reasons
            if reason.startswith(("FAIL", "SKIPPED"))
        ]
        raise PromotionRejectedError(failures or list(decision.reasons))
    return decision


def format_decision(decision: GateDecision) -> str:
    """Render a gate decision as the block the CLI prints and the model card embeds.

    One renderer so the terminal, the card and the registry entry say the same thing. Two
    renderers drift, and the day they disagree is the day nobody can reconstruct why a model
    was promoted.

    Args:
        decision: The decision to render.

    Returns:
        A plain-text block: the verdict, one line per criterion, then the numbers.
    """
    head = "PROMOTED" if decision.promoted else "REJECTED"
    champion = decision.champion_run_id or "(none — first model in this domain)"
    lines = [
        f"Promotion gate: {head}",
        f"  challenger: {decision.challenger_run_id}",
        f"  champion:   {champion}",
        "",
        "  Criteria:",
    ]
    for key in CRITERIA:
        mark = "PASS" if decision.checks.get(key) else "FAIL"
        lines.append(f"    [{mark}] {key}")
    lines.extend(["", "  Reasons:"])
    lines.extend(f"    - {reason}" for reason in decision.reasons)
    if decision.metrics:
        lines.extend(["", "  Measured:"])
        lines.extend(
            f"    {name} = {value:.6g}" for name, value in sorted(decision.metrics.items())
        )
    return "\n".join(lines)

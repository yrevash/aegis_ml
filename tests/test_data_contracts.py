"""The pandera data contract: the only thing in the stack that notices a bad frame.

``aegis.ml.model.train`` one-hot-encodes with ``handle_unknown="ignore"``. That flag exists
so *inference* survives an unseen level, and the price is that *training* never learns it
was there: an unseen level encodes to an all-zero block and the row is scored as though the
feature were absent. No error, anywhere. The contract derived from ``MLProblem`` is what
turns that into a refusal at the boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from aegis_ml.contracts.errors import AegisMLError
from aegis_ml.contracts.frames import schema_for, validate


def test_valid_frame_is_accepted_and_returned_coerced(frame, problem) -> None:
    """The COERCED frame comes back — passing the original on discards the correction."""
    validated = validate(frame, problem)
    assert len(validated) == len(frame)
    assert set(problem.feature_names) <= set(validated.columns)
    assert validated[problem.target.name].dtype.kind == "f"
    for feature in problem.features:
        if feature.dtype == "numeric":
            assert validated[feature.name].dtype == np.dtype("float64")


def test_unseen_categorical_level_is_refused(frame, problem) -> None:
    """The failure ``OneHotEncoder(handle_unknown='ignore')`` is contractually silent about."""
    categorical = next(f for f in problem.features if f.dtype == "categorical")
    bad = frame.copy()
    bad.loc[bad.index[0], categorical.name] = "PREMIUM "  # trailing space, as a generator emits

    with pytest.raises(AegisMLError) as excinfo:
        validate(bad, problem)
    message = str(excinfo.value)
    assert categorical.name in message
    assert "PREMIUM " in message
    assert "handle_unknown='ignore'" in message


def test_out_of_range_numeric_is_refused(frame, problem) -> None:
    """A declared bound is enforced, not documented."""
    bounded = next(
        f for f in problem.features if f.dtype == "numeric" and f.minimum is not None
    )
    bad = frame.copy()
    bad.loc[bad.index[0], bounded.name] = float(bounded.minimum) - 1000.0

    with pytest.raises(AegisMLError) as excinfo:
        validate(bad, problem)
    assert bounded.name in str(excinfo.value)


def test_null_in_a_non_nullable_column_is_refused(frame, problem) -> None:
    """A null where the spec says there are none is a generator bug, caught here."""
    strict = next(f for f in problem.features if not f.nullable and f.dtype == "numeric")
    bad = frame.copy()
    bad.loc[bad.index[0], strict.name] = np.nan

    with pytest.raises(AegisMLError) as excinfo:
        validate(bad, problem)
    assert strict.name in str(excinfo.value)


def test_null_in_a_declared_nullable_column_is_accepted(frame, problem) -> None:
    """MAR missingness is a design property here, so a nullable column must survive."""
    nullable = [f for f in problem.features if f.nullable]
    assert nullable, "the reference spec declares at least one nullable feature"
    assert frame[nullable[0].name].isna().any()
    validate(frame, problem)  # must not raise


def test_missing_column_is_refused(frame, problem) -> None:
    """A declared feature that is not in the frame cannot be trained on."""
    dropped = problem.feature_names[0]
    with pytest.raises(AegisMLError) as excinfo:
        validate(frame.drop(columns=[dropped]), problem)
    assert dropped in str(excinfo.value)


def test_null_target_is_refused(frame, problem) -> None:
    """A null label is not a missing measurement — the target is never nullable."""
    bad = frame.copy()
    bad.loc[bad.index[0], problem.target.name] = np.nan
    with pytest.raises(AegisMLError):
        validate(bad, problem)


def test_every_violation_is_reported_at_once(frame, problem) -> None:
    """Lazy validation: three mistakes are three lines, not three rounds of guessing."""
    categorical = next(f for f in problem.features if f.dtype == "categorical")
    bounded = next(f for f in problem.features if f.dtype == "numeric" and f.minimum is not None)

    bad = frame.copy()
    bad.loc[bad.index[0], categorical.name] = "nonsense_level"
    bad.loc[bad.index[1], bounded.name] = float(bounded.minimum) - 500.0

    with pytest.raises(AegisMLError) as excinfo:
        validate(bad, problem)
    message = str(excinfo.value)
    assert categorical.name in message
    assert bounded.name in message


def test_extra_columns_are_allowed_by_default_and_refused_when_strict(frame, problem) -> None:
    """A training frame legitimately carries join keys; a serving payload does not."""
    extra = frame.copy()
    extra["shipment_id"] = [f"ship-{i}" for i in range(len(extra))]

    validate(extra, problem)  # training contract: the spine drops it

    with pytest.raises(AegisMLError):
        validate(extra, problem, strict=True)


def test_inference_contract_does_not_require_the_target(frame, problem) -> None:
    """At prediction time the label does not exist yet."""
    payload = frame.drop(columns=[problem.target.name])
    validated = validate(payload, problem, include_target=False)
    assert problem.target.name not in validated.columns

    with pytest.raises(AegisMLError):
        validate(payload, problem, include_target=True)


def test_schema_is_derived_from_the_spec_not_hand_written(problem) -> None:
    """Every declared feature becomes a column; the categorical levels become a check."""
    schema = schema_for(problem)
    assert set(schema.columns) == {*problem.feature_names, problem.target.name}
    assert problem.domain_id in schema.name

    categorical = next(f for f in problem.features if f.dtype == "categorical")
    checks = schema.columns[categorical.name].checks
    assert any("isin" in str(check) for check in checks), (
        "a categorical column with no isin check cannot notice an unseen level"
    )


def test_inference_schema_omits_the_target(problem) -> None:
    """``include_target=False`` is the serving contract."""
    schema = schema_for(problem, include_target=False)
    assert problem.target.name not in schema.columns
    assert "inference" in schema.name


def test_classification_target_levels_are_enforced(excursion_frame, excursion_problem) -> None:
    """An unknown class label in the target is refused too, not only in the features."""
    validate(excursion_frame, excursion_problem)

    bad = excursion_frame.copy()
    bad.loc[bad.index[0], excursion_problem.target.name] = "maybe"
    with pytest.raises(AegisMLError):
        validate(bad, excursion_problem)


# ── the spec itself refuses what would produce an unenforceable contract ──────


def test_categorical_feature_without_levels_is_refused() -> None:
    """The contract cannot check an open set, so the spec will not accept one."""
    from aegis_ml.contracts.spec import FeatureSpec

    with pytest.raises(ValueError, match="declares no levels"):
        FeatureSpec(name="carrier", dtype="categorical", description="who moved it")


def test_levels_on_a_non_categorical_feature_are_refused() -> None:
    """``levels`` and ``dtype`` disagreeing is how a column gets encoded two ways."""
    from aegis_ml.contracts.spec import FeatureSpec

    with pytest.raises(ValueError, match="declares levels"):
        FeatureSpec(name="hours", dtype="numeric", description="transit", levels=["a", "b"])


def test_feature_name_must_be_a_python_identifier() -> None:
    """The spec is code-generated into a module; a name that cannot survive that is refused."""
    from aegis_ml.contracts.spec import FeatureSpec

    with pytest.raises(ValueError, match="not a valid Python identifier"):
        FeatureSpec(name="transit hours", dtype="numeric", description="transit")


def test_classification_target_needs_at_least_two_levels() -> None:
    """A one-class classification target is not a classification problem."""
    from aegis_ml.contracts.spec import TargetSpec

    with pytest.raises(ValueError, match="<2 levels"):
        TargetSpec(name="flag", task="classification", description="d", levels=["only"])

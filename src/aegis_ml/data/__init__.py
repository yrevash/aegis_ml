"""Data layer: build a target worth predicting, split it honestly, prove it is learnable.

The four jobs here run in this order, and each exists because skipping it produces a
believable number rather than an error:

1. :mod:`~aegis_ml.data.latent` declares the latent function the generator samples around,
   with the realism layer — calibrated noise, unobserved confounders, MAR missingness,
   boundary label noise — that keeps the result inside a credible band instead of at
   R² 0.99. It also owns :func:`~aegis_ml.data.latent.assert_learnable`, the check
   ``SKILL.md`` names and no Aegis conformance check performs.
2. :mod:`~aegis_ml.data.splits` reproduces ``aegis.ml.model.train``'s three-way split
   exactly, guards the calibration size MAPIE needs, and refuses to shuffle a time series.
3. :mod:`~aegis_ml.data.profile` describes a frame for a machine and for a human.
4. :mod:`~aegis_ml.data.contract_check` runs the schema, learnability and structural checks
   together and returns one report.

:mod:`~aegis_ml.data.synth` is the alternative inbound path: fit SDV to a real CSV and
sample more of it. It stays secondary to the procedural + latent generator, for the reasons
its own module docstring gives.

Importing this package costs pydantic and nothing else. Every heavy dependency — pandas,
scikit-learn, skrub, SDV — is imported inside the function that needs it, through
:func:`aegis_ml._require.require`, so a missing one raises naming the install command
rather than degrading into a weaker code path.
"""

from __future__ import annotations

from aegis_ml.data.contract_check import ColumnAudit, ContractReport, check
from aegis_ml.data.latent import (
    ACCURACY_CEILING,
    R2_CEILING,
    Confounder,
    Interaction,
    LatentCalibration,
    LatentDriver,
    LatentModel,
    LearnabilityReport,
    MissingnessRule,
    RealismConfig,
    assert_learnable,
    default_latent_model,
    measure_learnability,
    realism_report,
)
from aegis_ml.data.profile import profile, summarize_column, summarize_columns
from aegis_ml.data.splits import (
    ThreeWaySplit,
    grouped_split,
    min_calibration_rows,
    stratified_split,
    three_way_split,
    time_ordered_split,
)
from aegis_ml.data.synth import (
    FittedSynthesizer,
    fit_synthesizer,
    quality_report,
    sample,
    synthesize,
)

__all__ = [
    "ACCURACY_CEILING",
    "R2_CEILING",
    "ColumnAudit",
    "Confounder",
    "ContractReport",
    "FittedSynthesizer",
    "Interaction",
    "LatentCalibration",
    "LatentDriver",
    "LatentModel",
    "LearnabilityReport",
    "MissingnessRule",
    "RealismConfig",
    "ThreeWaySplit",
    "assert_learnable",
    "check",
    "default_latent_model",
    "fit_synthesizer",
    "grouped_split",
    "measure_learnability",
    "min_calibration_rows",
    "profile",
    "quality_report",
    "realism_report",
    "sample",
    "stratified_split",
    "summarize_column",
    "summarize_columns",
    "synthesize",
    "three_way_split",
    "time_ordered_split",
]

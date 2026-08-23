"""The worked reference domain: pharmaceutical cold-chain logistics.

This package is the proof that ``aegis_ml`` and the Aegis adapter contract work end to end
on a real problem. It is not a fixture and not a sketch — it is a complete, runnable domain:

    reference/
        adapter/        the ten pieces (an Aegis DomainAdapter, satisfied structurally)
        problem.py      the same domain as an aegis_ml MLProblem + LatentModel

    scripts/run_demo.py generates data, runs the four flows, and prints measured numbers

**Why this domain.** Cold-chain logistics is lexically disjoint from Aegis's shipped
service-request domain, so the conformance suite's vocabulary quarantine is genuinely
exercised rather than trivially satisfied. It also carries all three shapes the ML stack
needs to demonstrate at once: a **regression** target (``spoilage_risk_pct``), a
**classification** target (``excursion_flag``), and a **time series** (shipments dispatched
per day) — from one generator, one schema and one declared causal story.

**What is deliberately hard about the data.** The generator does not produce a target a
model can memorise. Held-out R² lands in the 0.62–0.71 range rather than at 0.99, because
the label carries calibrated noise, two unobserved confounders, heteroscedastic spread,
missing-at-random holes and two genuinely irrelevant columns. Every one of those is a
declared property with a named constant, measured by ``aegis_ml.data.latent.realism_report``
and printed by the demo — not an accident of hand-tuned coefficients. See
:mod:`reference.adapter.ml_spec` for the reasoning and ``reference/README.md`` for the
numbers.

Importing this package is cheap: the submodules import on demand, and the adapter itself
carries no hard dependency on an Aegis checkout being present.
"""

from __future__ import annotations

__all__: list[str] = []

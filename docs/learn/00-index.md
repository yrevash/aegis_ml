# Learn `aegis_ml` — start here

This is the **teaching track**. It assumes you can read Python and nothing else: no machine
learning, no MLOps, no knowledge of Aegis. Ten short chapters, read in order, and you will
understand what this package is, why each piece exists, and how to run it.

There is a second, separate track in `docs/01-*.md` … `docs/10-*.md`. Those are **reference**
documents written for someone who already knows the field. Do not start there. This track
links to them when you are ready.

---

## What this package is, in ten lines

* **Aegis** (`/Users/yrevash/aegis/`) is an enterprise platform that runs AI agents which
  take real actions — with human approval, audit trails and guardrails around every step.
* Aegis is **domain-agnostic**. To point it at a new business problem you write one thing: a
  *domain adapter* — ten files describing your problem's data, tools, people and prompts.
* Aegis has a solid but small ML core: it can train one model and put a confidence interval
  around each prediction.
* **`aegis_ml` is the missing ML half**: automatic model search, data-honesty checks,
  explanations, a model registry, a promotion gate, and drift monitoring.
* It also ships **templates** for all ten adapter pieces, and a **complete worked domain**
  (pharmaceutical cold-chain logistics) that runs green end to end today.
* Nothing here replaces Aegis. Everything extends it, and the handoff between the two is a
  single file the Aegis platform already knows how to load.

---

## What you will be able to do at the end

1. Explain what a feature, a target, a held-out split and a conformal interval are — and why
   the last one is the point of the whole exercise.
2. Look at any of the nine charts this package produces and say whether the model is healthy.
3. Run the full pipeline yourself and read every number it prints.
4. Say why synthetic data scoring R² 0.99 is a **bug report, not an achievement**.
5. Describe how a trained model reaches an Aegis agent, and why that needs no changes to the
   Aegis core.

---

## Reading order

| # | Chapter | What it answers | Length |
|---|---|---|---|
| 01 | [What problem does this solve?](01-what-problem-does-this-solve.md) | Why does this package exist at all? | short |
| 02 | [The ML concepts you need](02-ml-concepts-you-need.md) | Features, targets, splits, conformal prediction, SHAP, AutoML | long |
| 03 | [The data problem](03-the-data-problem.md) | Why fake data that is too easy destroys the demo | **most important** |
| 04 | [The pipeline](04-the-pipeline.md) | The seven flows and what each stage does | long |
| 05 | [Reading the charts](05-reading-the-charts.md) | All nine pictures, good vs bad | medium |
| 06 | [MLOps: registry, gate, drift](06-mlops-registry-gate-drift.md) | What happens after training | medium |
| 07 | [How it plugs into Aegis](07-how-it-plugs-into-aegis.md) | The adapter contract and the handoff | medium |
| 08 | [Your first run](08-your-first-run.md) | Hands-on, copy-pasteable | medium |
| 09 | [Glossary](09-glossary.md) | One-line definitions | reference |

Chapters 02 and 03 carry the ideas. Chapter 08 is the one you will come back to.

---

## The map of the repository

```
aegis_ml/
├── src/aegis_ml/          the package
│   ├── contracts/         the shared vocabulary: MLProblem, Recipe, GateDecision …
│   ├── data/              generate data, check it is honest, split it
│   ├── features/          encode columns; detect leakage
│   ├── automl/            search for a model across four tiers
│   ├── evaluate/          metrics, slices, the promotion gate
│   ├── explain/           SHAP, reason codes, the model card
│   ├── registry/          the filesystem model registry; promote and roll back
│   ├── monitor/           drift (Evidently) and label-free estimates (NannyML)
│   ├── report/            the nine charts and the HTML bundle
│   ├── forecast/          time-series wrappers over aegis.forecast
│   ├── serve/             the FastAPI router and the five ML agent tools
│   ├── pipelines/         flows.py — the seven flows that tie it together
│   └── cli.py             every command you will type
├── reference/             the fully worked cold-chain-logistics domain
├── templates/adapter/     the ten adapter pieces as annotated skeletons
├── prompts/               authoring packs, one per adapter piece
├── config/                commented tunables (see the warning below)
├── scripts/run_demo.py    the end-to-end demonstration
└── registry_store/        where runs, models and charts land
```

> **A warning that will save you an hour.** The `.toml` files in `config/` are **documentation
> of the intended settings, not a file the code reads.** Nothing in `src/` parses them. The
> values that actually take effect live in `src/aegis_ml/settings.py` (overridable with
> `AEGIS_ML_*` environment variables) and as module constants such as
> `aegis_ml.pipelines.flows.REALISM_R2_BAND`. Read `config/` for the *reasoning*; change
> behaviour in `settings.py` or the environment.

---

## Conventions used throughout

* **Every number quoted in this track was measured**, either by the committed demo run in
  `registry_store/` or by a command run while writing these pages. Nothing is illustrative.
* Terms are defined the first time they appear, and again in the [glossary](09-glossary.md).
* Where a claim is contested by another document in this repo, the code wins and the
  disagreement is called out.

Next: [01 · What problem does this solve?](01-what-problem-does-this-solve.md)

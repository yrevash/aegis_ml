# 01 · What problem does this solve?

[← Index](00-index.md) · Next: [02 · The ML concepts you need](02-ml-concepts-you-need.md)

---

## 1. Aegis, in one paragraph

Aegis is an enterprise **agentic AI platform**. An "agent" here is a program that takes a
question from a person, decides what to do, and then *does* it — looks records up, writes
notes, changes a booking. The hard part is not making it act. The hard part is making it
safe enough that a company will let it: every action must be explainable, budgeted,
approved by a human when the risk is high, and written to an audit log afterwards.

Aegis's own summary of itself is **"autonomy you can audit."** Six checkpoints stand between
the model and a real action: input guardrails, cited retrieval, a **confidence signal**, a
human approval gate, governance/budget enforcement, and a trace plus an audit row.

Full detail: [`docs/01-what-is-aegis.md`](../01-what-is-aegis.md).

---

## 2. Retargeting Aegis means writing exactly one thing

Aegis is deliberately **domain-agnostic**: the core knows nothing about insurance claims or
cold-chain shipments or IT tickets. All of that lives in a single directory called a
**domain adapter**.

An adapter is ten pieces — eight Python modules plus two content directories:

| # | Piece | What it says |
|---|---|---|
| 1 | `schema.py` | The record types in your world |
| 2 | `ml_spec.py` | The features, the thing you predict, the training data |
| 3 | `generator.py` | How to fabricate a realistic synthetic world |
| 4 | `tools.py` | The actions the agent may take, each with a risk tier |
| 5 | `personas.py` | Who is asking, and what data each may see |
| 6 | `prompts.py` | The system prompt for each persona |
| 7 | `memory_spec.py` | What counts as a durable fact worth remembering |
| 8 | `roster.py` | Which specialists the supervisor routes between |
| 9 | `corpus/` | Seed documents the agent can cite |
| 10 | `skills/` | Procedural playbooks |

Write those ten, and the agent graph, the human gate, memory, retrieval, role-based access
control, tracing, guardrails and the web console all keep working untouched. That is the
whole promise. Chapter [07](07-how-it-plugs-into-aegis.md) covers the contract properly;
the reference version of it is [`docs/02-domain-adapter-contract.md`](../02-domain-adapter-contract.md).

---

## 3. What Aegis already has, and what it lacks

This matters, because the temptation is to rebuild things that already exist.

**Aegis already ships a serious ML spine** (`aegis.ml`, ~966 lines):

* an ensemble of XGBoost and HistGradientBoosting voting together,
* **MAPIE split-conformal** calibration on a separate slice of data (chapter 02 explains
  what that means),
* SHAP explanations averaged across the ensemble,
* a `ModelCard` that keeps the *requested* confidence level and the *measured* one as two
  separate fields,
* SHA-256 fingerprints of the training data,
* and a typed refusal (`MLModelUnavailableError`) instead of quietly returning a guess.

It also ships time-series forecasting (`aegis.forecast`) with conformal intervals and
rolling-origin backtests.

**What Aegis does not have:**

| Missing capability | Consequence without it |
|---|---|
| Automatic model search | You hand-pick one estimator and hope |
| A check that synthetic labels are *learnable* | A target that is pure noise passes every automated check Aegis has |
| A check that they are not *too easy* | R² 0.99 on toy data, and a confidence interval so narrow it says nothing |
| A model registry | No history, no champion, no rollback |
| A promotion gate | A worse model can replace a better one silently |
| Drift monitoring | Nobody notices when the world moves away from the training data |
| Label-free performance estimation | You learn the model degraded only when the labels arrive, days later |
| Per-segment reporting | A model that collapses for one customer group looks fine on average |

`aegis_ml` supplies exactly that list. **It extends the spine; it never replaces it.**

---

## 4. The hackathon framing, stated honestly

This repository was built for a hackathon where **the problem statement is not known until
the day**. That single fact explains most of the design.

You cannot pre-build the solution. What you *can* pre-build is everything that would
otherwise be re-derived under time pressure:

* **The dependency resolution.** Adding a dozen ML libraries to a platform with hard version
  caps is the classic way to lose a morning. It was resolved and locked in advance —
  see [`RESOLUTION.md`](../../RESOLUTION.md) and chapter [04 §6](04-the-pipeline.md#6-two-virtualenvs-one-portable-recipe-decision-d1).
* **The adapter contract**, distilled into templates and one prompt-pack per piece.
* **The ML machinery**, generic over any tabular problem.
* **A complete worked domain** — pharmaceutical cold-chain logistics — proven green, so that
  on the day you pattern-match against working code instead of empty files.

So: **this is a base, not a solution.** It does not know your problem. It knows the shape of
every problem of this kind, and it removes the parts that are the same every time.

---

## 5. Why cold-chain logistics is the worked example

The reference domain in [`reference/`](../../reference/README.md) predicts the **spoilage
risk** (a percentage) of temperature-controlled pharmaceutical shipments. It was chosen for
three concrete reasons, not for flavour:

1. **It shares no vocabulary with the domain Aegis already ships.** Aegis's own example is
   `service_request_management`. One of the fourteen conformance checks scans the entire
   platform for leftover words from the shipped domain. Retargeting to something adjacent —
   tickets, cases, work orders — would satisfy that check by accident. "Consignment",
   "packout", "excursion" and "lane" exercise it for real.
2. **One generator produces three ML shapes**: a regression target (`spoilage_risk_pct`, in
   `%`), a classification target (`excursion_flag`, did the shipment get too warm), and a
   time series ("Shipments dispatched per day"). That shows the breadth of the stack without
   three unrelated fixtures.
3. **The actions have a genuine risk spread.** Looking a consignment up is free. Annotating
   it is cheap and reversible. Rerouting it costs money and the customer sees it.
   Quarantining it strands product a clinic is expecting. That is a real LOW → HIGH ladder,
   not four tools tiered to make a table look full.

---

## 6. What "good" looks like here

A recurring theme, worth internalising before chapter 02: in this codebase **a suspiciously
good number is treated as a defect.**

The committed demo run scores **R² = 0.7199** on data it had never seen. A beginner's
instinct is that 0.99 would be better. It would not — it would mean the synthetic data was
trivial, and every downstream claim about calibrated uncertainty would be worthless. Chapter
[03](03-the-data-problem.md) is entirely about this.

The second recurring theme is Aegis's most important rule, inherited verbatim:

> **No silent fallbacks.** A control that cannot run fails closed and says so — it never
> degrades quietly into something that looks like it worked.

Which is why `aegis_ml.contracts.errors` defines nine named refusals
(`LabelNotLearnableError`, `RecipeNotPortableError`, `TargetLeakageError`,
`PromotionRejectedError`, `DriftThresholdExceededError`, …). Each exists because the
alternative was a plausible-looking number a human would have believed.

---

Next: [02 · The ML concepts you need](02-ml-concepts-you-need.md)

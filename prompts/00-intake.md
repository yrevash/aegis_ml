# PROMPT 00 · Intake — problem statement → Domain Brief

**Run this first. Nothing else runs until it has.**

---

## Role

You are a domain architect. You are given a hackathon problem statement in natural language and you must turn it into a **Domain Brief**: a precise, structured specification from which ten Aegis adapter pieces, a synthetic data generator and an ML pipeline can be written without further interpretation.

You are not writing code in this task. You are removing every remaining decision.

---

## Read first

- `/Users/yrevash/aegis_ml/docs/02-domain-adapter-contract.md` — what the ten pieces are.
- `/Users/yrevash/aegis_ml/docs/04-synthetic-data.md` — why the latent-driver section of the Brief exists and why it is the most important part.

---

## Output

Write **exactly one file**:

```
/Users/yrevash/aegis_ml/DOMAIN_BRIEF.md
```

Use the template in §4 verbatim — same headings, same order, same table columns. Every prompt-pack from `01-schema.md` onward parses it by heading. Do not add sections, do not rename headings, do not reorder.

---

## Rules

1. **Fill every field.** If the problem statement does not say, **decide** and mark the row `[assumed]`. An unfilled field becomes an invented value three prompts later, and inconsistently.
2. **Every identifier must be a valid Python identifier**, `snake_case`, and unique within its section. `MLProblem` rejects anything else.
3. **Every categorical feature must declare its full level set.** `FeatureSpec` refuses a categorical with no `levels`, because the one-hot encoder uses `handle_unknown="ignore"` and an undeclared level silently encodes to all zeros.
4. **The target must not appear in the feature list.** That is perfect leakage and `MLProblem` refuses it.
5. **No feature may be a deterministic function of the target.** Ask of each one: *"at the moment we make this prediction, do we actually know this?"* If not, it is leakage.
6. **8–12 features.** Fewer makes a thin SHAP story; more makes a slow search and a cluttered chart.
7. **At least one HIGH-risk tool.** A domain with no gated action cannot demonstrate the human gate, which is one of Aegis's six trust checkpoints.
8. **Use the client's language for anything a human reads** — `DOMAIN_SERIES_LABEL`, persona display names, tool descriptions, the domain description. These are sentences a judge reads on screen.
9. **Avoid any word from the reference domain's quarantined vocabulary** so that check #14 is exercised for real: `operations_lead`, `dataset.requests`, `Service requests opened per day`, `num_requests`, `queue_depth_at_open`, `agent_tenure_months`, `reopened_count`, `customer_tier`, `description_length`, `resolution_hours`, `ServiceRequest`, `SupportAgent`, `update_request_status`, `assign_request`, `add_case_note`, `closing_requests`, `de_escalation`, `service_request_management`.

---

## 4 · The Domain Brief template

Copy this structure exactly.

````markdown
# Domain Brief — <Domain Title>

Source problem statement: <one-sentence restatement>
Generated: <date>

## 1. Identity

| Field | Value |
|---|---|
| `domain_id` | `<snake_case_id>` |
| Title | <Human Readable Title> |
| One-line pitch | <what this system does, for whom> |

**`DOMAIN_DESCRIPTION`** (one paragraph, 40–80 words). This is wired straight into the
guardrails as `allowed_topics`, so it is a **control input, not metadata**. Name the
entities, the actions and the decisions in scope. A vague description is a loose rail.

> <the paragraph>

## 2. Entities

| Entity | Purpose | Key fields (name: type, constraint) | Notes |
|---|---|---|---|
| `<PascalCaseName>` | | | |

**Enums** (each becomes a `StrEnum`; the `.value` strings become categorical levels):

| Enum | Members → values |
|---|---|
| `<PascalCaseEnum>` | `MEMBER = "value"`, … |

**`SCHEMA_VERSION`**: `1.0.0`
**`SyntheticDataset` holds**: `metadata`, `<list per entity>`
**Lookup helpers**: `<entity>_by_id(...)`, `labelled_<records>()`

## 3. The supervised problem

| Field | Value |
|---|---|
| Target name | `<snake_case>` |
| Task | `regression` \| `classification` |
| Unit (regression) | `<unit string>` |
| Levels (classification) | `["<a>", "<b>"]` |
| Bounds | min `<x>`, max `<y>` |
| Primary metric | `r2` \| `accuracy` \| … |
| Requested coverage | `0.9` |
| Plain-English meaning | <what a number/label means to the client> |
| Decision it supports | <what someone does differently because of it> |

**Secondary target (optional):** `<name>` — `<task>`, for a second chart.

## 4. Features

| # | Name | dtype | Unit | Levels / range | Nullable | What it measures |
|---|---|---|---|---|---|---|
| 1 | `<name>` | numeric \| categorical \| boolean \| datetime | | | no | |

Constraints checked:
- [ ] 8–12 rows
- [ ] every `categorical` row has a full level set
- [ ] the target is not in this table
- [ ] no row is knowable only *after* the target is known
- [ ] 1–2 rows are marked **irrelevant on purpose** (see §5)

## 5. Latent drivers — the ground truth

**This is the most important section in the Brief.** The generator computes
`label = latent_fn(features) + noise`. This table *is* `latent_fn`.

Intercept: `<value>`

| Feature | Sign | Magnitude | Rationale |
|---|---|---|---|
| `<categorical>` | table | `{"level_a": 0.0, "level_b": 6.0, …}` | |
| `<numeric>` | `+` \| `−` | `<coefficient>` per unit | |

**Interaction term** (exactly one):
`<feature_a> × <feature_b>`, coefficient `<c>` — <why this is physically sensible>

**Deliberately irrelevant features** (declared in `FEATURES`, never read by `latent_fn`):
`<name>`, `<name>` — SHAP will show these flat, which proves the explanation is real.

**Unobserved confounder** (affects the label, is NOT a feature):
`<name>` — drawn per `<group>` from the seeded RNG, magnitude `<σ>`.
This is the irreducible error floor and the honest reason the interval has width.

Floor / clamp on the output: `<expression>`

## 6. Realism targets

| Field | Value |
|---|---|
| `target_r2` (or target accuracy) | `<0.50–0.70>` / `<0.70–0.85>` |
| Noise model | Gaussian, heteroscedastic |
| Heteroscedasticity | σ scales with `<feature>`: `σ = base * (<expr>)` |
| MAR missingness | `<feature>` null `<n>%` when `<other_feature> == <value>` |
| Class balance (classification) | `<70/30>` or `<85/15>` |
| Rows for `training_frame` | `1200` |
| Seed | `7` |

## 7. The demand series

| Field | Value |
|---|---|
| `DOMAIN_SERIES_LABEL` | `"<A sentence a client reads>"` |
| `DOMAIN_SERIES_UNIT` | `"<plural noun>"` |
| Event | **arrival** of a `<record>` (not completion) |
| Shape | weekly seasonality, `<trend>`, `<any spike pattern>` |
| Default `num_records` / `seed` | `1400` / `11` |

## 8. Personas

| id | Display name | RBAC roles | Data scope | `prompt_key` | Who they are |
|---|---|---|---|---|---|
| `<staff_id>` | | admin, ai_team, devops | `ALL` | `<staff_id>` | |
| `<user_id>` | | client | `OWN` on `<subject_field>` | `<user_id>` | |

`DEFAULT_PERSONA_ID`: `<staff_id>`

**`PERSONA_BY_ROLE` — every RBAC role must appear:**

| Role | Persona id |
|---|---|
| `admin` | `<staff_id>` |
| `ai_team` | `<staff_id>` |
| `devops` | `<staff_id>` |
| `client` | `<user_id>` |

## 9. Tools

| name | Risk | read_only | destructive | idempotent | Args (name: type) | What it does |
|---|---|---|---|---|---|---|
| `<find_x>` | LOW | ✔ | | ✔ | | |
| `<note_x>` | LOW | | | | | |
| `<assign_x>` | MEDIUM | | | ✔ | | |
| `<commit_x>` | **HIGH** | | ✔ | ✔ | | |

**ALLOWLIST:**

| Persona | Tools |
|---|---|
| `<staff_id>` | all |
| `<user_id>` | `<subset>` |

Plus the five ML tools from `aegis_ml.serve.tools` (all LOW, read-only):
`predict_outcome`, `explain_prediction`, `whatif_scenario`, `forecast_series`, `check_model_health`.

## 10. Roster

**Specialists — the role strings MUST be exactly `qa` and `memory`:**

| role | is_default | Description (re-voiced) | Keywords |
|---|---|---|---|
| `qa` | ✔ | | *(none — it is the fallback)* |
| `memory` | | | `<what-do-you-know-about-me phrasings>` |

**Sub-agents (fan-out team):**

| agent_id | role | label | tool_allowlist | System prompt gist |
|---|---|---|---|---|
| | | | | |

Every name in every `tool_allowlist` must be a key of §9.

## 11. Memory

| Field | Value |
|---|---|
| `FACT_TYPES` | `["preference", "entity_attr", "commitment", "constraint"]` |
| `PROFILE_FIELDS` | `[…]` |
| `PROFILE_ALIASES` | `{…}` or `{}` |
| Memory subject | per `<end user \| account \| case>` → `f"<prefix>:{user_id}"` |
| `IMPORTANCE_HINTS` | <one sentence covering 1-3 / 4-6 / 7-8 / 9-10> |

## 12. Corpus

| Filename | `id` | `kind` | `title` | `category` | `tags` | Content gist |
|---|---|---|---|---|---|---|
| `<name>.md` | `doc-seed-0001` | | | | | |

3–6 documents, each a few hundred words with real headings.

## 13. Skills

| Filename (no `.md`) | Trigger keywords → this file | What the playbook covers |
|---|---|---|
| `<name>` | `<kw>`, `<kw>`, `<kw>` | |

2–4 playbooks. The keyword→filename mapping goes in `select_skills`'s `hints` dict —
**both edits or the playbook is never selected.**

## 14. Quarantine list

The exact strings that go into `aegis/src/aegis/conformance/_vocabulary.py`.
Specific identifiers only, never generic nouns.

```python
SHIPPED_DOMAIN_ID = "<domain_id>"

SHIPPED_VOCABULARY: tuple[str, ...] = (
    "<staff_persona_id>",
    "dataset.<record_collection>",
    "<DOMAIN_SERIES_LABEL verbatim>",
    "num_<records>",
    "<feature_1>", "<feature_2>", "<feature_3>", "<target_name>",
    "<EntityA>", "<EntityB>",
    "<high_risk_tool>", "<medium_risk_tool>", "<low_risk_tool>",
    "<playbook_a>", "<playbook_b>",
    SHIPPED_DOMAIN_ID,
)
```

## 15. Assumptions made

| # | Field | What the statement did not say | What I decided | Why |
|---|---|---|---|---|
````

---

## 5 · Worked example fragment

Problem statement:

> *"Hospitals need to know which scheduled surgeries are at risk of running past their booked slot, so theatre coordinators can re-sequence the list."*

```markdown
## 1. Identity

| Field | Value |
|---|---|
| `domain_id` | `surgical_scheduling` |
| Title | Surgical Theatre Scheduling |
| One-line pitch | Predict slot overrun for booked procedures so coordinators can re-sequence the day's list before it slips. |

> Elective surgical theatre scheduling. Coordinators sequence booked procedures across
> theatres, surgeons and anaesthetists, track how far each theatre is running behind,
> and re-sequence or reassign lists to contain delay. In scope: procedures, theatres,
> surgeons, theatre days, overrun prediction, list re-sequencing and theatre notes.

## 3. The supervised problem

| Field | Value |
|---|---|
| Target name | `slot_overrun_minutes` |
| Task | `regression` |
| Unit | `minutes` |
| Bounds | min `0`, max `240` |
| Primary metric | `r2` |
| Requested coverage | `0.9` |
| Plain-English meaning | How many minutes past its booked finish this procedure is expected to run. |
| Decision it supports | Whether to re-sequence the list, move a case to another theatre, or warn the next patient. |

## 5. Latent drivers — the ground truth

Intercept: `0.0`

| Feature | Sign | Magnitude | Rationale |
|---|---|---|---|
| `procedure_type` | table | `{"hip_replacement": 40, "cataract": 5, "hernia": 20, "arthroscopy": 18, "cholecystectomy": 30}` | Intrinsic complexity |
| `asa_grade` | table | `{"I": 0, "II": 6, "III": 15, "IV": 28}` | Sicker patients take longer |
| `surgeon_seniority` | table | `{"registrar": 0, "consultant": -8, "senior_consultant": -14}` | Experience is faster |
| `slot_position` | `+` | `3.5` per position | Delay accumulates down the list |
| `prior_overrun_mins` | `+` | `0.45` per minute | The day is already behind |
| `equipment_swaps` | `+` | `9.0` per swap | Each changeover costs time |
| `booked_minutes` | `−` | `0.05` per minute | Generous bookings absorb overrun |
| `patient_bmi` | `+` | `0.6` per unit above 25 | Access and positioning |

**Interaction term:** `slot_position × prior_overrun_mins`, coefficient `0.02` —
position only matters once the day has already slipped.

**Deliberately irrelevant features:** `theatre_id`, `booking_channel`.

**Unobserved confounder:** `staffing_pressure` — drawn once per `TheatreDay`,
σ = 6.0 minutes, shifts the whole day's intercept. Never a feature.

Floor: `max(0.0, value)`

## 6. Realism targets

| Field | Value |
|---|---|
| `target_r2` | `0.62` |
| Heteroscedasticity | `σ = base * (0.5 + booked_minutes / 200.0)` |
| MAR missingness | `patient_bmi` null 35% when `asa_grade == "I"` |
| Rows / seed | `1200` / `7` |

## 7. The demand series

| Field | Value |
|---|---|
| `DOMAIN_SERIES_LABEL` | `"Procedures scheduled per day"` |
| `DOMAIN_SERIES_UNIT` | `"procedures"` |
| Event | arrival (booking) of a `Procedure` |
| Shape | strong weekday/weekend seasonality, mild upward trend |
```

---

## 6 · Verify

```bash
test -f /Users/yrevash/aegis_ml/DOMAIN_BRIEF.md && echo "brief present"

grep -c '^## ' /Users/yrevash/aegis_ml/DOMAIN_BRIEF.md    # expect 15
```

Then read it back against this checklist and fix anything that fails:

- [ ] All 15 sections present, in order, with the template's headings.
- [ ] `domain_id` is `snake_case` and appears verbatim in §14.
- [ ] `DOMAIN_DESCRIPTION` is 40–80 words and names entities, actions and decisions.
- [ ] 8–12 features; every categorical has a complete level set.
- [ ] The target is not in the feature table.
- [ ] No feature is knowable only after the target is known.
- [ ] §5 gives a sign and a magnitude for **every** feature except the ones marked irrelevant.
- [ ] Exactly one interaction term, with a physical rationale.
- [ ] 1–2 deliberately irrelevant features are named.
- [ ] Exactly one unobserved confounder is named, with a magnitude.
- [ ] `target_r2` is between 0.50 and 0.70 (or accuracy 0.70–0.85).
- [ ] `DOMAIN_SERIES_LABEL` is a sentence a client would read, not an identifier.
- [ ] `PERSONA_BY_ROLE` covers `admin`, `ai_team`, `devops` and `client`.
- [ ] `DEFAULT_PERSONA_ID` is one of the declared persona ids.
- [ ] At least one tool is HIGH risk.
- [ ] Roster roles are **exactly** `qa` and `memory`; exactly one is default; `team` does not appear.
- [ ] Every `tool_allowlist` name is a tool in §9.
- [ ] Every skill's trigger keywords are listed; the filename has no `.md` in §13's first column.
- [ ] §14 lists specific identifiers only — no generic nouns like "customer", "client", "request".
- [ ] Nothing in the Brief reuses a word from the reference domain's quarantine list (§Rules 9).
- [ ] Every `[assumed]` row also appears in §15.

---

## 7 · Next

`prompts/01-schema.md`.

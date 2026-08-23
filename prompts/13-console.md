# PROMPT 13 · The console — four files, by hand

**These files are outside the adapter AND outside the conformance check, which scans Python only. They will show the old domain's words on screen after an otherwise perfect retarget.**

---

## Role

You are re-voicing the four `web/` files that carry shipped-domain literals. One of them does not merely look wrong — it **breaks the console outright**.

---

## Inputs

- `DOMAIN_BRIEF.md` §1 (identity), §8 (personas), §4 (features), §9 (tools).
- Your finished `app/adapter/personas.py`, `tools.py`, `ml_spec.py`.

---

## The four files

All verified present at the time of writing.

| # | File | What it names | Failure mode |
|---|---|---|---|
| 1 | `/Users/yrevash/aegis/web/src/config/personas.ts` | Two persona ids, the tool names in its prose, and sample queries about the old domain | **Breaks the console.** `POST /query` answers **400 Unknown persona**. |
| 2 | `/Users/yrevash/aegis/web/src/components/ops/opsShared.ts` | `PROMPT_KEY`, and the tool names inside two prompt strings | Wrong prompt key; prompts naming tools that no longer exist |
| 3 | `/Users/yrevash/aegis/web/src/components/sim/SimulationView.tsx` | The two persona ids it drives the scripted demo with, plus `SIM_QUERY` | The headline simulation fails to start, or asks about the old domain |
| 4 | `/Users/yrevash/aegis/web/src/components/ml/MLOpsView.tsx` | **A literal ML feature row** | Every key lands in `unknown_features`, every real feature in `imputed_features` — the panel explains a prediction made entirely from training medians |

> *"The real fix is for the API to serve the persona list and the feature spec so the console reads them like everything else; there is no such endpoint yet, and inventing one mid-retarget is not the moment."* — `SKILL.md`
>
> Re-voice them by hand, and **say in your report that you did.**

---

## File 1 — `web/src/config/personas.ts`

**The most important of the four**, because it is not cosmetic. From the file's own docstring:

> *"Every `id` here must exist in the backend adapter's persona table (`backend/src/app/adapter/personas.py`). `POST /query` resolves the persona id through `get_persona()` and answers **400 Unknown persona** when it cannot, so an invented id does not degrade — it stops the console from running at all."*

### What to change

```ts
export const PERSONAS: Persona[] = [
  {
    id: 'theatre_coordinator',                       // ← must match PERSONAS in personas.py
    name: 'Theatre Coordinator',                     // ← Persona.display_name
    roles: ['ai_team', 'admin', 'devops'],           // ← keep: matches PERSONA_BY_ROLE
    blurb: 'Sequences the day\u2019s theatre lists and contains delay before it spreads.',
    sampleQueries: [
      'What has to be true before a case can be moved to another theatre?',
      'Which lists are at risk today? What does the containment policy require? What does the runbook say? Who approves a re-sequence?',
      'What do you know about me and how I like my lists sequenced?',
    ],
  },
  {
    id: 'surgeon',
    name: 'Surgeon',
    roles: ['client'],
    blurb: 'Operating surgeon tracking delay risk on their own list.',
    sampleQueries: [
      'What is the predicted overrun on my afternoon list?',
      'Add a note to my next case: equipment set was short two trays',
      'What does the policy say about re-taking consent after a rebooking?',
    ],
  },
]
```

`roles` is a **list** because one adapter persona legitimately serves several portals — the admin and devops consoles drive the same staff persona the AI-team console does. Duplicating the entry per portal would mean two registry rows with the same id, which `getPersona` could not tell apart.

### The three sample queries are not decoration

The existing docstring records that they were **measured against a running backend**, one per capability, and that entry 0 is special: *"`QueryBar` seeds its input with `sampleQueries[0]` on three other screens, so entry one has to be the cheap, reliable one."*

Keep that structure:

| # | Demonstrates | Shape that produces it |
|---|---|---|
| 0 | **Grounded retrieval** | ~13 words, **one clause** → the deterministic router returns SINGLE → full recall → retrieve → rerank, with citations |
| 1 | **The fan-out** | **Four `?`/`;` clauses** → `_subquestion_count` = 4 → `depth=team fanout=4`, all sub-agent lanes reporting |
| 2 | **Long-term memory** | Phrasing that matches your `memory` specialist's keywords → `role=memory`, RAG and tools skipped |

Ask about topics your **seed corpus actually covers**, and carry **no hard-coded record id** — the generator makes ids at seed time, so a literal id names a record that does not exist.

### The gate chip — and your chance to beat the reference

The docstring explains why there is no fourth chip reaching the approval gate:

> *"Ten candidate phrasings were run … and **not one of them ever proposed that tool**, so the gate never raised. The cause is structural rather than a matter of wording: `UpdateStatusArgs` requires a `request_id`, the persona's tool roster is write-only (there is no listing or lookup tool), and the … system prompt says never fabricate request ids — so a planner that obeys the prompt has no id it can justify and correctly declines. … Reaching it needs a read-side tool in `backend/src/app/adapter/tools.py` (a LOW-risk `find_requests`), which is an adapter change and not a console one."*

**Your adapter has that read-side tool** (`prompts/04-tools.md` gives every domain a LOW read-only finder, and `aegis_ml.serve.tools` adds `predict_outcome`). So a fourth chip **can** reach the gate:

```
"Find the case most likely to overrun on theatre 2 today and re-sequence the list to protect it"
```

The planner can call the finder, get a real id, then propose the HIGH-risk tool with a justified argument — and the gate fires. **Measure it against the running backend before you ship the chip.** A chip that promises the gate and delivers a paragraph is worse than no chip.

---

## File 2 — `web/src/components/ops/opsShared.ts`

```ts
export const PROMPT_KEY = 'theatre_coordinator'
```

Must equal the `prompt_key` of your staff persona — that is, a key of `SYSTEM_PROMPTS` in `app/adapter/prompts.py`.

Then the two prompt strings that name tools:

```ts
// before
'Only call allowlisted tools. update_request_status is HIGH risk and always needs human approval.',
'Only call allowlisted tools. update_request_status requires human approval.',

// after
'Only call allowlisted tools. resequence_list is HIGH risk and always needs human approval.',
'Only call allowlisted tools. resequence_list requires human approval.',
```

Name **your** HIGH-risk tool and state its tier correctly. Also update the docstring at the top, which lists the real persona keys and the registered tools with their tiers — leave it accurate or delete the claim.

---

## File 3 — `web/src/components/sim/SimulationView.tsx`

Three kinds of occurrence:

```ts
// 1 — the scripted query
const SIM_QUERY = 'Re-sequence my afternoon list to protect the case most likely to overrun'

// 2 — the two lanes it drives (both persona ids)
opsLead.start(SIM_QUERY, 'theatre_coordinator', token)
client.start(SIM_QUERY, 'surgeon', token)

// 3 — the labels and the roleId props
['the theatre coordinator', 'the surgeon'],
{ lane: ops, title: 'Theatre coordinator', roleId: 'theatre_coordinator', icon: UserCog, accent: 'agent' as const },
{ lane: cli, title: 'Surgeon',             roleId: 'surgeon',             icon: UserRound, accent: 'graph' as const },
// ...and a further roleId="theatre_coordinator" further down the file
```

Find every one:

```bash
grep -n "operations_lead\|'client'\|Operations lead\|Client" \
  /Users/yrevash/aegis/web/src/components/sim/SimulationView.tsx
```

**This is the side-by-side simulation** — the same question asked as two personas, showing the data-scope boundary live. Choose a `SIM_QUERY` that **produces visibly different answers for your two personas**: the staff persona sees everything and can act; the end-user persona sees only their own rows and has a smaller allowlist. If both lanes answer identically, the exhibit proves nothing.

---

## File 4 — `web/src/components/ml/MLOpsView.tsx`

**The worst of the four**, and the same defect as the trainer's old sanity probe.

```ts
const EXAMPLE: MLExplainRequest = {
  features: {
    priority: 'high',
    category: 'billing',
    channel: 'email',
    region: 'na',
    customer_tier: 'enterprise',
    agent_tenure_months: 41,
    queue_depth_at_open: 34,
    reopened_count: 0,
    description_length: 420,
  },
}
```

The file's own docstring says why this matters:

> *"Every key is a name in the adapter's `FEATURES` contract (`backend/src/app/adapter/ml_spec.py`). A key that is not in that contract is **not an error** — the spine reports it under `unknown_features` and imputes the training median in its place — so an invented feature set would quietly explain a prediction nobody asked for."*

After a retarget, **every** key here is unknown and **every** real feature is imputed. The panel shows a prediction made entirely from training medians, with a SHAP chart of nothing, and no error anywhere.

### Replace it with your features

```ts
const EXAMPLE: MLExplainRequest = {
  features: {
    procedure_type: 'hip_replacement',
    asa_grade: 'III',
    surgeon_seniority: 'registrar',
    theatre_id: 't2',
    slot_position: 6,
    booked_minutes: 120,
    prior_overrun_mins: 35,
    patient_bmi: 34.5,
    equipment_swaps: 2,
  },
}
```

**Rules:**

1. **Every key must be in `FEATURE_NAMES`.** All of them. No extras.
2. **Every categorical value must be a declared level** — an `.value` from the corresponding `StrEnum`. An undeclared level one-hot-encodes to all zeros without raising.
3. **Choose an interesting row.** A median row predicts the median and the SHAP chart is flat. Pick values that land toward one tail of your latent function — here: a late slot, a day already 35 minutes behind, a registrar, an ASA III patient. The prediction is high, the interval is meaningful, and the drivers are visible.
4. **Do not include the target.**

Generate a good row rather than inventing one:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
import json
from app.adapter import ml_spec
f = ml_spec.training_frame(num_records=400, seed=3).sort_values(ml_spec.TARGET.name)
row = f.iloc[int(len(f) * 0.85)][ml_spec.FEATURE_NAMES].to_dict()
print(json.dumps({k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}, indent=4))
")
```

The 85th percentile gives a high-but-not-extreme row: a large prediction, a real interval, and drivers that separate.

---

## Sweep for anything missed

```bash
cd /Users/yrevash/aegis/web
grep -rn "operations_lead\|update_request_status\|assign_request\|add_case_note\|find_requests\|queue_depth_at_open\|agent_tenure_months\|reopened_count\|description_length\|customer_tier\|resolution_hours\|service_request\|Service requests opened per day\|ServiceRequest\|SupportAgent" \
  src/ tests/ public/ 2>/dev/null
```

```powershell
Set-Location C:\aegis\web
Select-String -Path src\*, tests\*, public\* -Recurse -Pattern "operations_lead|update_request_status|assign_request|add_case_note|find_requests|queue_depth_at_open|agent_tenure_months|reopened_count|description_length|customer_tier|resolution_hours|service_request|Service requests opened per day|ServiceRequest|SupportAgent"
```

Also grep for the domain's *prose* words, which no list can enumerate:

```bash
grep -rniE "service request|support agent|case note|ticket|help ?desk" src/ | grep -v node_modules
```

Anything outside the four files is a fifth file nobody documented. Fix it and **say so in your report** — that is a real finding.

---

## Verify

```bash
cd /Users/yrevash/aegis/web
npx tsc --noEmit
npm test
npx next build
```

```powershell
Set-Location C:\aegis\web
npx tsc --noEmit; npm test; npx next build
```

Then run it and look:

```bash
(cd /Users/yrevash/aegis && ./scripts/dev-native.sh)   # or .\scripts\start.ps1 -Mode full
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m app.seed)
(cd /Users/yrevash/aegis/web && npm run dev)
```

Open **http://localhost:3000**, sign in as `admin` / `demo`, and check with your eyes:

- [ ] The persona picker shows **your** persona names.
- [ ] Clicking sample query 0 produces a **cited** answer about your domain.
- [ ] Sample query 1 fans out — the console shows `depth=team` and every lane label is yours.
- [ ] Sample query 2 routes to `memory`.
- [ ] The **ML-Ops panel** predicts a sensible number in **your** unit, with an interval, and a SHAP chart whose top drivers are the ones you wrote into the latent function.
- [ ] `unknown_features` is **empty** and `imputed_features` is empty or nearly so.
- [ ] The **simulation** starts both lanes and the two answers **visibly differ** in scope.
- [ ] The **forecast** chart title is your `DOMAIN_SERIES_LABEL`, with your unit on the y-axis.
- [ ] Nowhere on any screen do the words "service request", "support agent" or "case note" appear.

---

## Checklist

- [ ] `personas.ts` ids match `PERSONAS` in `app/adapter/personas.py` **exactly**.
- [ ] `roles` on each console persona is consistent with `PERSONA_BY_ROLE`.
- [ ] Sample query 0 is short, single-clause and reliable — it seeds three other screens.
- [ ] Sample query 1 has four `?`/`;` clauses so the fan-out actually fires.
- [ ] Sample query 2 matches your `memory` specialist's keywords.
- [ ] No sample query carries a literal record id.
- [ ] `opsShared.ts` `PROMPT_KEY` is a key of `SYSTEM_PROMPTS`.
- [ ] `opsShared.ts` names **your** HIGH-risk tool, with the right tier.
- [ ] `SimulationView.tsx` uses both of your persona ids everywhere, with your labels.
- [ ] `SIM_QUERY` produces **visibly different** answers for the two personas.
- [ ] `MLOpsView.tsx` `EXAMPLE.features` keys are **exactly** `FEATURE_NAMES`.
- [ ] Every categorical value there is a declared level.
- [ ] The example row is interesting (~85th percentile), not median.
- [ ] `npx tsc --noEmit`, `npm test` and `npx next build` all pass.
- [ ] The sweep grep returns nothing.
- [ ] You checked the running console with your own eyes.
- [ ] **The four files are named in your report.**

---

## Next

`prompts/14-final-gate.md`.

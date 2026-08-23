# PROMPT 10 · Piece 10 — `skills/`

**This piece is two edits in two files. Doing only one is silent and costs an hour in the wrong file.**

---

## Role

You are writing **piece 10 of 10**: the procedural how-to-act playbooks. They are Markdown, discovered from `memory_spec.SKILLS_DIR` at call time, and chosen per query by `memory_spec.select_skills`.

---

## Inputs

- `DOMAIN_BRIEF.md` §13 (skills), §9 (tools), §12 (corpus).
- Piece 7's `SKILLS_DIR` and `select_skills`.
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/skills/`.

## Output files

```
/Users/yrevash/aegis_ml/reference/adapter/skills/<name>.md          ×2–4
```

**and the `hints` table back in**

```
/Users/yrevash/aegis_ml/reference/adapter/memory_spec.py
```

---

## The contract to satisfy

Piece 10 has **no Protocol member of its own** — deliberately. From `aegis/src/aegis/adapter.py`:

> *"`skills/` is the one piece with no member of its own, deliberately: it is a directory of Markdown playbooks discovered at call time, and it is already named by member 7's `SKILLS_DIR`. Giving it a second, top-level spelling would create exactly the kind of 'is it five or six?' ambiguity this module exists to end."*

Two conformance checks cover it:

| Check | Asserts |
|---|---|
| #10 `test_skills_directory_holds_at_least_one_playbook` | `SKILLS_DIR` points at a real directory with at least one `*.md`. |
| #11 `test_every_playbook_is_reachable_from_select_skills` | Every playbook on disk can be selected, and the selector names no playbook that is gone. |

---

## **The trap**

> **Playbooks are selected by filename, through a literal keyword → filename `hints` dict inside `select_skills`.**
>
> Rename `containing_overrun.md` and the entry still reads `"containing_overrun"`, which is then filtered out by `skill in available` — so the renamed playbook can **never be selected again** and the stale entry can **never fire**.
>
> Both halves are silent: `select_skills` returns its other matches, or `None`, the core injects no skill, and the turn proceeds. **You will read that as a prompt problem and spend an hour in the wrong file.**

So **piece 10 is really two edits**: the `*.md` files, and the `hints` dict back in `memory_spec.py`. Do both, in the same commit.

Check #11 reads that table from the selector's **compiled string constants** *and* from its module's **top-level constants**, so it does not matter whether the table sits inside the function or beside it. If it can see no table at all it falls back to a behavioural probe and fails if nothing it is given can ever reach a playbook.

**The other half of check #11** is always-applicable: it probes the selector and asserts it **never returns a name outside the `available` list it was handed**. So `select_skills` must filter on `available`, not just look up `hints`.

**Second trap.** Like `corpus/`, these are data files. A `cp -r` sync leaves `closing_requests.md` and `de_escalation.md` behind. Use `rsync -a --delete`.

---

## What to write

### 1. The playbooks — 2–4 files

A playbook is an **instruction sheet for the agent**, not documentation for a human. It answers: *when does this apply, what do I do, in what order, what do I check first, and what do I escalate?*

```markdown
# Skill: Containing an overrunning theatre list

## When this applies

The user is asking about a list that is running late, a case predicted to overrun, or
what to move. Trigger phrases: "running behind", "overrun", "at risk", "resequence".

## Before you propose anything

1. Call `find_procedures` for the theatre day in question. Do not reason from what the
   user told you — read the list.
2. Call `predict_outcome` for the remaining cases. **Always quote the conformal
   interval, never a bare point estimate.** A predicted overrun of 40 minutes with an
   interval of [-5, 85] does not justify moving a patient; 40 with [22, 58] does.
3. Call `explain_prediction` on the single largest contributor. Name its top two
   drivers when you explain your recommendation.

## The containment ladder

Apply the least disruptive intervention that fixes the problem. Do not skip a rung, and
say which rung you are on.

1. Shorten turnaround — no tool, a note to the coordinator.
2. Re-sequence within the list — `resequence_list`. **HIGH risk: this pauses for human
   approval.** Say so before you propose it, so the coordinator is not surprised.
3. Move a case to another theatre — `reassign_theatre` (MEDIUM).
4. Cancel and rebook — out of scope for you. Escalate to the coordinator and stop.

## Always

- Cite the containment policy document when you recommend a rung.
- Record what you did with `add_theatre_note`.
- If the predicted overrun's interval straddles zero, say the model is not confident
  and recommend watching rather than acting.

## Never

- Never propose moving a case on a point estimate alone.
- Never recommend a cancellation. That decision is not yours.
- Never re-order a list that is already complete.
```

**Requirements:**

- **2–4 playbooks.** Enough to make selection meaningful, few enough to keep the `hints` table honest.
- **Name your tools by their registered names**, so the model connects the procedure to something it can actually call.
- **Say which tools are HIGH risk and pause for approval.** That is how the agent explains the gate to the user instead of appearing to stall.
- **Have a "Never" section.** A playbook that only says what to do gives the model no boundary.
- **`snake_case` filenames, no `.md` in the `hints` table.**
- **At least one playbook should reference the ML tools** and insist on the interval. That is what makes the ML evidence show up in the transcript rather than only in a chart.
- **At least one should be about the risky action**, so the human gate has a narrative around it.

Suggested set for most domains:

| Playbook | Covers |
|---|---|
| `<doing_the_main_thing>` | The core operational procedure, using the ML tools |
| `<handling_the_hard_case>` | The escalation / exception path, where the HIGH-risk tool lives |
| `<saying_no>` | What is out of scope and how to decline (optional but good for guardrails) |

### 2. The `hints` table — back in `memory_spec.py`

```python
def select_skills(query: str, persona_id: str | None, available: list[str]) -> list[str] | None:
    """Select the procedural skills a query needs (a subset of ``available``).

    Playbooks are chosen BY FILENAME through this table. Rename a ``skills/*.md`` file
    without updating it and the playbook can never be selected again — and nothing
    warns you: this function just returns None and the turn proceeds unguided.
    """
    hints = {
        # containing_overrun.md
        "overrun": "containing_overrun",
        "running late": "containing_overrun",
        "running behind": "containing_overrun",
        "behind": "containing_overrun",
        "resequence": "containing_overrun",
        "at risk": "containing_overrun",
        # cancelling_a_case.md
        "cancel": "cancelling_a_case",
        "cancellation": "cancelling_a_case",
        "consent": "cancelling_a_case",
        "rebook": "cancelling_a_case",
        # out_of_scope.md
        "emergency": "out_of_scope",
        "on-call": "out_of_scope",
    }
    lowered = query.lower()
    chosen = [
        skill
        for skill in dict.fromkeys(hints[k] for k in hints if k in lowered)
        if skill in available          # ← check #11 requires this filter
    ]
    return chosen or None
```

- **Group the entries by target file with a comment.** That is what makes the table auditable against `ls skills/`.
- **3–6 trigger phrases per playbook.** Prefer phrases over single words: a bare `"cancel"` fires on "cancel that last instruction".
- **`dict.fromkeys(...)`** de-duplicates while preserving order, so two keywords hitting the same file yield one entry.
- **The `if skill in available` filter is not optional.**

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
from pathlib import Path
import reference.adapter.memory_spec as m

d = Path(m.SKILLS_DIR)
assert d.is_dir(), f'SKILLS_DIR does not exist: {d}'
available = sorted(p.stem for p in d.glob('*.md'))
assert available, 'no playbooks on disk'
print('on disk   :', available)

# 1 — every playbook is reachable from at least one phrase
probes = ['overrun', 'running behind', 'resequence', 'at risk',
          'cancel', 'consent', 'rebook', 'emergency', 'on-call']
reachable = set()
for probe in probes:
    got = m.select_skills(probe, None, available) or []
    assert set(got) <= set(available), f'{probe!r} named something absent: {got}'
    reachable |= set(got)
missing = set(available) - reachable
assert not missing, f'playbooks nothing can select: {sorted(missing)}'
print('reachable :', sorted(reachable))

# 2 — the selector never invents a name
assert m.select_skills('overrun', None, []) in (None, [])
assert m.select_skills('overrun', None, ['not_a_real_one']) in (None, [])

# 3 — an unrelated query selects nothing
print('unrelated :', m.select_skills('what is the weather', None, available))

# 4 — every playbook is substantial
for p in sorted(d.glob('*.md')):
    text = p.read_text(encoding='utf-8')
    assert len(text) > 500, f'{p.name}: too short to be a procedure ({len(text)} chars)'
    assert text.lstrip().startswith('#'), f'{p.name}: no H1'
    print(f'  {p.stem:26s} {len(text):5d} chars')
print('skills consistent')
"
```

Confirm the old playbooks are gone after the sync:

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/skills/*.md
# must NOT contain closing_requests.md or de_escalation.md
```

And check every tool a playbook names actually exists:

```bash
uv run python -c "
import re
from pathlib import Path
import reference.adapter.memory_spec as m
from reference.adapter.tools import TOOL_REGISTRY

named = set()
for p in Path(m.SKILLS_DIR).glob('*.md'):
    named |= set(re.findall(r'\`([a-z_]+)\`', p.read_text(encoding='utf-8')))
unknown = {n for n in named if '_' in n} - set(TOOL_REGISTRY)
print('tools named in playbooks :', sorted(named & set(TOOL_REGISTRY)))
print('backticked non-tools     :', sorted(unknown))
"
```

(The second list is informational — not everything in backticks is a tool — but a name that *looks* like a tool and is not in the registry is worth a second look.)

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k "skill or playbook")
```

### Checklist

- [ ] 2–4 `*.md` playbooks in `skills/`, `snake_case` filenames.
- [ ] Each is > 500 characters, starts with an H1, and has "when this applies", ordered steps, and a "Never" section.
- [ ] Each names your tools by their registered names.
- [ ] At least one names a HIGH-risk tool and says it pauses for approval.
- [ ] At least one uses the ML tools and insists on the conformal interval.
- [ ] **The `hints` table in `memory_spec.py` names every file on disk, and nothing else.**
- [ ] Entries are grouped by target file with a comment.
- [ ] 3–6 trigger phrases per playbook, phrases rather than bare words.
- [ ] `select_skills` filters on `available` and returns `None` (not `[]`) when nothing matches.
- [ ] `closing_requests.md` and `de_escalation.md` are gone from the synced directory.

---

## All ten pieces are now written

Go to `prompts/11-ml-pipeline.md` for the ML half, then `prompts/12-integration.md` for the sync.

Before you do, run the full contract check:

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter as a
from aegis.adapter import DomainAdapter, missing_members
print('missing  :', missing_members(a))
print('satisfies:', isinstance(a, DomainAdapter))
"
uv run aegis-ml contract
```

`missing: []` and `satisfies: True`, or go back to whichever piece is missing.

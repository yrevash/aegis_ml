# PROMPT 07 · Piece 7 — `memory_spec.py`

**Also piece 10's selector lives here. You will come back to this file after writing the playbooks.**

---

## Role

You are writing **piece 7 of 10**: what counts as a durable fact, whose memory it belongs to, and which procedural playbooks a query needs. This is the **only** memory seam — nothing in `app/memory/*` or `aegis/memory/*` changes.

---

## Inputs

- `DOMAIN_BRIEF.md` §11 (memory), §13 (skills), §8 (personas).
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/memory_spec.py`.

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/memory_spec.py
```

---

## The contract to satisfy

```python
@runtime_checkable
class MemorySpecModule(Protocol):
    FACT_TYPES: list[str]
    PROFILE_FIELDS: list[str]
    FACT_EXTRACTION_PROMPT: str
    IMPORTANCE_HINTS: str
    SKILLS_DIR: str
    FactSchema: type[Any]
    FactExtraction: type[Any]

    def memory_subject_for(self, user_id: str | int | None, persona_id: str | None = None) -> str | None: ...
    def render_profile(self, profile: dict[str, Any]) -> str: ...
    def select_skills(self, query: str, persona_id: str | None, available: list[str]) -> list[str] | None: ...
```

Structurally this is `aegis.memory.spec.MemorySpec` — the module a host installs with `set_default_spec(...)` — plus `memory_subject_for`, which the memory package itself does not need but every host does.

---

## **The trap that outranks the others**

> **`memory_spec` is deliberately NOT re-exported by name through `adapter/__init__.py`. Its consumer binds to the *module object*.**
>
> `backend/src/app/memory/__init__.py` calls `set_default_spec(app.adapter.memory_spec)`, and three other places import it directly. So the module must keep **its path and its symbol names**, not just its behaviour.

`from app.adapter.memory_spec import FACT_TYPES, memory_subject_for` and `from app.adapter.memory_spec import FACT_EXTRACTION_PROMPT` are both bound by host code and tests (verified by grep). Rename a constant and the host raises `ImportError` at startup.

Conformance check #9 (`test_memory_spec_satisfies_the_memory_contract`) catches a missing member. Its scar: `set_default_spec` accepts *any* object at startup and the members are read one at a time, deep inside recall and consolidation — so a spec missing `IMPORTANCE_HINTS` or `FactExtraction` **starts cleanly, serves queries, and fails only the first time a conversation is consolidated**, which in a demo is never.

---

## **The second trap — `memory_subject_for` is a data-boundary function**

> *"That function is the domain's answer to 'is memory per end-user, per account, or per case?', and getting it wrong is a **cross-subject data leak** rather than a quality regression."*

Think about it before you write it:

| Domain shape | Subject | Returns |
|---|---|---|
| One human, personal preferences | the end user | `f"user:{user_id}"` |
| A shared account several people use | the account | `f"account:{account_id}"` |
| A long-running case with many participants | the case | `f"case:{case_id}"` |
| An anonymous or unauthenticated principal | nobody | **`None`** |

**Returning `None` means "no memory for this turn"** — nothing is recalled and nothing is written. That is the correct answer for an unauthenticated caller, and returning a shared constant instead would pool every anonymous visitor's facts into one subject that every one of them can then read back.

---

## What to write

### 1. The fact vocabulary

```python
FACT_TYPES: list[str] = ["preference", "entity_attr", "commitment", "constraint"]
"""The typed kinds of durable fact this domain distils from a conversation.

Four is the right number and these four generalise: something the person prefers,
something true about an entity, something we promised, and something we may not do.
Re-voice the descriptions in FACT_EXTRACTION_PROMPT rather than inventing new types —
the memory package's consolidation logic keys on these strings.
"""

PROFILE_FIELDS: list[str] = [
    "display_name", "primary_theatre", "specialty", "seniority",
    "preferred_list_order", "standing_constraints", "notes",
]
"""Ordered structured-profile fields, always injected into the prompt."""

PROFILE_ALIASES: dict[str, str] = {
    "name": "display_name",
    "theatre": "primary_theatre",
    "prefers_order": "preferred_list_order",
    "sub_specialty": "specialty",
}
"""Predicate spellings the extractor emits, mapped onto PROFILE_FIELDS entries.

Optional; an empty dict is a legitimate statement. This table used to live in
``aegis/memory/consolidate.py`` naming the shipped domain's fields, where it quietly
matched nothing after a retarget — which is why it lives here now.
"""
```

### 2. The extraction prompt and the importance scale

```python
IMPORTANCE_HINTS: str = (
    "Rate 1-3 for trivia, 4-6 for useful preferences and attributes, 7-8 for "
    "commitments or scheduling constraints, 9-10 for clinical-safety or consent "
    "constraints that must never be forgotten."
)
"""Domain guidance for the 1..10 importance rating."""

FACT_EXTRACTION_PROMPT: str = (
    "Extract durable facts from this conversation in an elective surgical theatre "
    "service.\n\n"
    f"Fact types: {', '.join(FACT_TYPES)}.\n"
    "- preference: how this person likes lists sequenced or communicated\n"
    "- entity_attr: a lasting attribute of a surgeon, theatre or specialty\n"
    "- commitment: something we said we would do\n"
    "- constraint: something that must not happen\n\n"
    "Do NOT extract: one-off scheduling details, anything about a specific patient, "
    "or anything the person could not reasonably expect us to remember.\n\n"
    f"{IMPORTANCE_HINTS}\n"
)
```

**Embed `IMPORTANCE_HINTS` in the prompt** rather than restating the scale — one source of truth, and the reference does the same.

**The "do NOT extract" clause matters.** Without it, the extractor stores transient details and the profile fills with noise that is then injected into every prompt. Name at least one category of thing that is genuinely sensitive in your domain.

### 3. The models

```python
class FactSchema(BaseModel):
    """One extracted durable fact."""

    fact_type: str = "entity_attr"
    subject: str = "surgeon"
    predicate: str
    object: str
    text: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    importance: int = Field(default=5, ge=1, le=10)
    valid_at: datetime | None = None


class FactExtraction(BaseModel):
    """The container the cheap-model extractor returns."""

    facts: list[FactSchema] = Field(default_factory=list)
```

Keep both names and `facts` as the list field. The memory package constructs `FactExtraction` from the model's JSON output and iterates `.facts`.

### 4. `SKILLS_DIR` — **a `str`, not a `Path`**

```python
SKILLS_DIR: str = str(Path(__file__).parent / "skills")
"""Directory of procedural skill Markdown files — piece 10.

A str, not a Path: the Protocol declares ``SKILLS_DIR: str`` and the skills loader
does string operations on it.
"""
```

Derive it from `__file__`. Hard-coding an absolute path breaks the moment the adapter is synced into the Aegis checkout.

### 5. `memory_subject_for`

```python
def memory_subject_for(user_id: str | int | None, persona_id: str | None = None) -> str | None:
    """Return the memory subject id for a principal, or ``None`` for no memory.

    This domain scopes memory to the individual clinician: their list preferences and
    standing constraints follow them across theatres and days. It is deliberately NOT
    scoped to the theatre or the specialty, which several people share — pooling
    memory across a shared subject is a cross-subject leak, not a quality choice.

    Args:
        user_id: The signed-in principal's id, or ``None`` when unauthenticated.
        persona_id: The persona in force; unused here, but part of the contract for
            domains where the same human has different memory as different personas.

    Returns:
        ``"user:<id>"``, or ``None`` when there is no principal to scope to.
    """
    if user_id is None or user_id == "":
        return None
    return f"user:{user_id}"
```

### 6. `render_profile`

```python
def render_profile(profile: dict[str, Any]) -> str:
    """Render the structured profile as the prompt's human block.

    Iterates PROFILE_FIELDS so the order is stable and an unknown key the extractor
    invented never reaches the prompt.
    """
    lines = [
        f"- {field.replace('_', ' ')}: {profile[field]}"
        for field in PROFILE_FIELDS
        if profile.get(field)
    ]
    return "Known about this person:\n" + "\n".join(lines) if lines else ""
```

Iterate `PROFILE_FIELDS`, not `profile.keys()` — the order must be stable and an invented key must not reach the prompt.

### 7. `select_skills` — **piece 10's other half**

```python
def select_skills(query: str, persona_id: str | None, available: list[str]) -> list[str] | None:
    """Select the procedural skills a query needs (a subset of ``available``).

    Playbooks are chosen BY FILENAME through the keyword table below. Rename a
    ``skills/*.md`` file without updating this table and it can never be selected
    again, and nothing warns you: this function just returns None, the core injects no
    skill, and the agent acts without procedural guidance. You will read that as a
    prompt problem and spend an hour in the wrong file.

    Args:
        query: The user's turn.
        persona_id: The persona in force; unused here.
        available: Playbook names (no ``.md``) actually present on disk.

    Returns:
        The matching playbook names, or ``None`` when none apply.
    """
    hints = {
        "overrun": "containing_overrun",
        "running late": "containing_overrun",
        "behind": "containing_overrun",
        "resequence": "containing_overrun",
        "cancel": "cancelling_a_case",
        "cancellation": "cancelling_a_case",
        "consent": "cancelling_a_case",
    }
    lowered = query.lower()
    chosen = [
        skill for skill in dict.fromkeys(hints[k] for k in hints if k in lowered)
        if skill in available
    ]
    return chosen or None
```

Two invariants:

1. **Never return a name outside `available`.** Conformance check #11 probes this behaviourally.
2. **Every playbook on disk must be reachable** from at least one keyword, and every value in `hints` must be a file that exists.

The table may live inside the function or as a module constant — check #11 reads the selector's compiled string constants **and** its module's top-level constants, so both work.

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
from pathlib import Path
import reference.adapter.memory_spec as m
from aegis.adapter import MemorySpecModule

assert isinstance(m, MemorySpecModule), 'missing a MemorySpecModule member'
print('FACT_TYPES    :', m.FACT_TYPES)
print('PROFILE_FIELDS:', m.PROFILE_FIELDS)
print('SKILLS_DIR    :', m.SKILLS_DIR, type(m.SKILLS_DIR).__name__)

d = Path(m.SKILLS_DIR)
assert d.is_dir(), f'SKILLS_DIR does not exist: {d}'
available = sorted(p.stem for p in d.glob('*.md'))
assert available, 'no playbooks on disk'
print('playbooks     :', available)

# every playbook must be reachable
reachable = set()
for probe in ['overrun', 'running late', 'cancel', 'consent', 'resequence', 'behind']:
    got = m.select_skills(probe, None, available) or []
    assert set(got) <= set(available), f'{probe!r} named a playbook not in available: {got}'
    reachable |= set(got)
unreachable = set(available) - reachable
assert not unreachable, f'playbooks nothing can select: {sorted(unreachable)}'
print('all reachable :', sorted(reachable))

# never names something absent
assert m.select_skills('overrun', None, []) in (None, [])

# memory subject
assert m.memory_subject_for(None) is None
assert m.memory_subject_for('') is None
print('subject       :', m.memory_subject_for(42), m.memory_subject_for('u-9'))

# profile renders in PROFILE_FIELDS order and drops unknowns
print(m.render_profile({'display_name': 'M. Okafor', 'specialty': 'orthopaedics', 'zzz': 'x'}))
assert 'IMPORTANCE' not in m.FACT_EXTRACTION_PROMPT.upper() or m.IMPORTANCE_HINTS in m.FACT_EXTRACTION_PROMPT
print('FactExtraction:', m.FactExtraction(facts=[]).model_dump())
"
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k "memory or skill or playbook")
```

### Checklist

- [ ] All seven required constants are module-level, with the exact names.
- [ ] `FactSchema` and `FactExtraction` exist by those names, with `facts` on the container.
- [ ] `SKILLS_DIR` is a **`str`** derived from `__file__`.
- [ ] `FACT_EXTRACTION_PROMPT` embeds `IMPORTANCE_HINTS` rather than restating it.
- [ ] `FACT_EXTRACTION_PROMPT` names at least one thing **not** to extract.
- [ ] `memory_subject_for(None)` and `memory_subject_for("")` return `None`.
- [ ] `memory_subject_for` scopes to the right subject for this domain — think about the leak.
- [ ] `render_profile` iterates `PROFILE_FIELDS`, not `profile.keys()`.
- [ ] `select_skills` never returns a name outside `available`.
- [ ] **Every playbook on disk is reachable** and every `hints` value is a file that exists.
- [ ] `PROFILE_ALIASES` is present (possibly empty).
- [ ] No fact type, profile field or playbook name from the reference domain survives.

---

## Next

`prompts/08-roster.md`. **Come back here after `prompts/10-skills.md`** to reconcile the `hints` table with the files you actually wrote.

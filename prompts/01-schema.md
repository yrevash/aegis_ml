# PROMPT 01 · Piece 1 — `schema.py`

---

## Role

You are writing **piece 1 of 10** of an Aegis domain adapter: the domain's record types and their version. This is the shared vocabulary every later piece consumes, so every name you choose here is a name you will type eight more times.

---

## Inputs

- `/Users/yrevash/aegis_ml/DOMAIN_BRIEF.md` §2 (Entities) and §1 (Identity).
- Reference implementation to pattern-match: `/Users/yrevash/aegis/backend/src/app/adapter/schema.py`.

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/schema.py
```

(Later synced to `/Users/yrevash/aegis/backend/src/app/adapter/schema.py`.)

---

## The contract to satisfy

`aegis.adapter.SchemaModule`:

```python
@runtime_checkable
class SchemaModule(Protocol):
    SCHEMA_VERSION: str
```

That is the entire *required* surface. The platform passes domain records around **opaquely** — it never introspects a field. `SCHEMA_VERSION` is written onto generated datasets so a corpus or a trained model can be told apart from one produced by a different shape of the same domain.

Everything else in this file exists because *later pieces* need it, not because the Protocol does.

---

## What to write

### 1. Module docstring

State that this is **piece 1 of 10**, name the entities, and say what the domain is. Docstrings in this codebase carry the reasoning, not the signature.

### 2. `StrEnum` per closed vocabulary

```python
from enum import StrEnum

class ProcedureType(StrEnum):
    """Kinds of elective procedure; complexity drives theatre time."""

    HIP_REPLACEMENT = "hip_replacement"
    CATARACT = "cataract"
    HERNIA = "hernia"
    ARTHROSCOPY = "arthroscopy"
    CHOLECYSTECTOMY = "cholecystectomy"
```

**The `.value` strings become your categorical feature levels in piece 2.** Choose them once, `snake_case`, and never re-spell them. `[p.value for p in ProcedureType]` is how piece 2 declares its `levels` — derive, never retype.

### 3. `SCHEMA_VERSION`

```python
SCHEMA_VERSION = "1.0.0"
"""Stamped onto every generated dataset. Bump when a record's shape changes."""
```

### 4. Pydantic v2 models, one per entity

```python
class Procedure(BaseModel):
    """One booked elective procedure on a theatre list."""

    id: str
    theatre_day_id: str
    theatre_id: str
    surgeon_id: str
    procedure_type: ProcedureType
    asa_grade: AsaGrade
    slot_position: int = Field(ge=1, le=8)
    booked_minutes: int = Field(gt=0)
    prior_overrun_mins: float = Field(default=0.0, ge=0.0)
    patient_bmi: float | None = Field(default=None, ge=16.0, le=55.0)
    equipment_swaps: int = Field(default=0, ge=0)
    booked_at: datetime
    scheduled_start: datetime
    actual_finish: datetime | None = None
    slot_overrun_minutes: float | None = Field(default=None, ge=0.0)   # ← the ML target
    notes: list[TheatreNote] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Whether this procedure has a measured outcome."""
        return self.slot_overrun_minutes is not None
```

- **Use field constraints.** `ge`, `le`, `gt`, `min_length` are free validation and they document the ranges your pandera contract will enforce.
- **The target field is `| None`.** Unfinished records have no label; `feature_matrix` only reads the ones that do.
- **A separate note/event model** with `id`, `author`, `body`, `created_at` gives your LOW-risk append tool something to write.

### 5. A dataset-metadata model

```python
class DatasetMetadata(BaseModel):
    """Provenance for one generated world."""

    schema_version: str = SCHEMA_VERSION
    seed: int | None = None
    llm_used: bool
    num_theatres: int
    num_surgeons: int
    num_procedures: int
    num_documents: int
    num_labelled: int
```

### 6. **`SyntheticDataset` — keep this name**

```python
class SyntheticDataset(BaseModel):
    """The container the generator returns and the ML spine reads."""

    metadata: DatasetMetadata
    theatres: list[Theatre]
    surgeons: list[Surgeon]
    theatre_days: list[TheatreDay]
    procedures: list[Procedure]
    documents: list[Document]

    def theatre_by_id(self, theatre_id: str) -> Theatre | None: ...
    def surgeon_by_id(self, surgeon_id: str) -> Surgeon | None: ...
    def labelled_procedures(self) -> list[Procedure]: ...
```

`labelled_*()` returns only rows carrying a target value. `feature_matrix` in piece 2 depends on it, and it must **guarantee** the target is not `None` so piece 2 does not have to re-check.

### 7. A `Document` model

Piece 9's corpus loader returns these. Fields: `id`, `kind` (a `StrEnum`), `title`, `body`, `category` (optional enum), `tags: list[str]`, `source: str = "synthetic"`.

---

## The trap

> **Keep the container names the registry re-exports, even while you change every field inside them.**

`adapter/__init__.py`'s `__all__` names **`SyntheticDataset`** specifically, and `SKILL.md` calls it out: *"`SyntheticDataset` as the container the generator returns and the ML spine reads."* Both the generator and the ML spine bind to it. Rename it and `app.agent.deps`, `app.mcp.server` and `app.demo_graph` break at import.

Your *entity* names are free — `ServiceRequest` becomes `Procedure`, and check #14 requires that it does. `SyntheticDataset` is not an entity name; it is a container name in the contract.

**Second trap.** `StrEnum`, not `Enum`. `str(Priority.HIGH)` on a plain `Enum` gives `"Priority.HIGH"`, which then becomes a categorical level, which then never matches the level set your generator declared, which then one-hot-encodes to all zeros without raising.

---

## Worked example fragment

```python
"""Domain schema — piece 1 of 10 of the adapter.

The elective surgical theatre world: theatres run day lists, surgeons work them,
procedures occupy booked slots, and a procedure that runs past its slot pushes
everything after it. The ML target lives on ``Procedure.slot_overrun_minutes``.

The platform never introspects these models — it passes them around opaquely — so the
only member :class:`aegis.adapter.SchemaModule` requires is ``SCHEMA_VERSION``.
Everything else here exists because pieces 2, 3, 4 and 9 consume it.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"
"""Stamped onto every generated dataset, so a corpus or a model can be told apart
from one produced by a different shape of the same domain."""


class AsaGrade(StrEnum):
    """ASA physical-status classification; higher grades take longer in theatre."""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"


class SurgeonSeniority(StrEnum):
    """Grade of the operating surgeon; experience is speed."""

    REGISTRAR = "registrar"
    CONSULTANT = "consultant"
    SENIOR_CONSULTANT = "senior_consultant"


COMPLETE_STATUSES: frozenset[ProcedureStatus] = frozenset(
    {ProcedureStatus.FINISHED, ProcedureStatus.CANCELLED_IN_THEATRE}
)
"""Statuses that mean an outcome exists. Read by ``labelled_procedures()``."""
```

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter.schema as s
from aegis.adapter import SchemaModule
print('SCHEMA_VERSION:', s.SCHEMA_VERSION)
print('satisfies SchemaModule:', isinstance(s, SchemaModule))
print('SyntheticDataset present:', hasattr(s, 'SyntheticDataset'))
d = s.SyntheticDataset(metadata=s.DatasetMetadata(llm_used=False, num_theatres=0,
      num_surgeons=0, num_procedures=0, num_documents=0, num_labelled=0),
      theatres=[], surgeons=[], theatre_days=[], procedures=[], documents=[])
print('empty dataset constructs:', d.metadata.schema_version)
print('labelled_procedures():', d.labelled_procedures())
"
```

PowerShell: same, with `Set-Location C:\aegis_ml`.

After the sync (`docs/07` §4):

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter/test_schema.py -q)
```

That test file carries **9 shipped-domain literals** and is part of *this* step to rewrite, not a follow-up.

### Checklist

- [ ] `SCHEMA_VERSION` is a module-level `str`.
- [ ] Every closed vocabulary is a `StrEnum` with `snake_case` values.
- [ ] Every entity is a pydantic v2 `BaseModel` with field constraints.
- [ ] `SyntheticDataset` exists **by that exact name**, with `metadata`, one list per entity, `*_by_id` helpers and `labelled_*()`.
- [ ] `DatasetMetadata` carries `schema_version`, `seed`, `llm_used` and a count per entity.
- [ ] A `Document` model exists for piece 9.
- [ ] The target field is `| None` on its entity.
- [ ] No name from the reference domain's quarantine list appears anywhere.
- [ ] `ruff check` is clean; every public symbol has a Google-style docstring.

---

## Next

`prompts/02-ml-spec.md`.

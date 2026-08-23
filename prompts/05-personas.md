# PROMPT 05 · Piece 5 — `personas.py`

**The trap in this piece breaks every login and no test in the repository catches it. Read §"The trap" before you write anything.**

---

## Role

You are writing **piece 5 of 10**: who is asking, and what they are allowed to see. A persona is **an authorisation object, not a personality**.

---

## Inputs

- `DOMAIN_BRIEF.md` §8 (personas and the role mapping).
- Piece 4's `ALLOWLIST` — the keys of that dict must be the ids you declare here.
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/personas.py`.

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/personas.py
```

---

## The contract to satisfy

```python
@runtime_checkable
class PersonasModule(Protocol):
    DEFAULT_PERSONA_ID: str

    @property
    def PERSONAS(self) -> Mapping[str, Any]: ...
    @property
    def PERSONA_BY_ROLE(self) -> Mapping[Any, str]: ...

    def get_persona(self, persona_id: str | None) -> Any: ...
    def persona_for_role(self, role: Any) -> str: ...
```

Each persona object must carry `.id`, `.data_scope` and `.prompt_key` — the platform reads all three.

---

## **The trap**

> **Re-voicing `PERSONAS` without re-pointing `PERSONA_BY_ROLE` makes every login raise `KeyError`.**
>
> Every authenticated principal resolves its persona through `persona_for_role(role)`, which reads that table. An entry naming a persona that no longer exists raises at the **login boundary**.
>
> The adapter suite, the agent suite, ruff and thirteen of the fourteen conformance checks all stay green, because **none of them go through the login path.**

This is a real defect this repository shipped. The host used to decide the mapping itself, with two persona ids written into an `if` in `app/api/routes.py` — which is exactly how the failure was found, and why the mapping moved into the adapter. `aegis/src/aegis/conformance/_vocabulary.py` still quarantines `operations_lead` for this reason.

**Second half of the same trap:** `PERSONAS[DEFAULT_PERSONA_ID]` is evaluated for **every** request that names no persona. A `DEFAULT_PERSONA_ID` that is not a key of `PERSONAS` is not a small mistake — it is every anonymous request 500-ing on the first turn.

**Third half:** the console sends persona ids too. `web/src/config/personas.ts` puts them in `QueryRequest.persona`, and `POST /query` answers **400 Unknown persona** for an id the adapter does not declare. See `prompts/13-console.md`.

Conformance check #7 (`test_every_persona_the_adapter_declares_resolves`) covers all three. Run it after this piece.

---

## What to write

### 1. Scope

```python
class ScopeKind(StrEnum):
    """How much of the record space a persona may see."""

    ALL = "all"
    OWN = "own"


class DataScope(BaseModel):
    """A persona's retrieval and record filter.

    Not a UI hint: ``kind`` becomes a retrieval filter and ``subject_field`` names the
    record column the principal's own id is matched against.
    """

    kind: ScopeKind
    subject_field: str | None = None
```

`ScopeKind` is imported by piece 6 (`prompts.py`). Keep the name.

### 2. `Persona`

```python
class Persona(BaseModel):
    """Who is asking, and what they may see and call."""

    id: str
    role: Role                 # app.api.schemas.Role — the coarse RBAC role
    display_name: str
    description: str
    data_scope: DataScope
    prompt_key: str

    @property
    def tool_names(self) -> frozenset[str]:
        """The tools this persona may call — read from ALLOWLIST, the single source."""
        return ALLOWLIST.get(self.id, frozenset())
```

> **`tool_names` must read `ALLOWLIST`, never hold its own copy.** Two lists of tool names drift, and the one the enforcement path reads is not the one the prompt renders — so the model is told it can do something it cannot, or vice versa. `prompts.py` renders the tool clause from `persona.tool_names`, so this property is what keeps the prompt honest.

### 3. The instances and the tables

```python
THEATRE_COORDINATOR = Persona(
    id="theatre_coordinator",
    role=Role.ADMIN,
    display_name="Theatre Coordinator",
    description="Sequences the day's lists across theatres and contains delay.",
    data_scope=DataScope(kind=ScopeKind.ALL),
    prompt_key="theatre_coordinator",
)

SURGEON = Persona(
    id="surgeon",
    role=Role.CLIENT,
    display_name="Surgeon",
    description="Sees their own list and the delay risk on it.",
    data_scope=DataScope(kind=ScopeKind.OWN, subject_field="surgeon_id"),
    prompt_key="surgeon",
)

PERSONAS: dict[str, Persona] = {p.id: p for p in (THEATRE_COORDINATOR, SURGEON)}

DEFAULT_PERSONA_ID: str = THEATRE_COORDINATOR.id
"""The persona a request that names none resolves to. Evaluated on EVERY anonymous
request, so it must be a key of PERSONAS."""

PERSONA_BY_ROLE: dict[Role, str] = {
    Role.ADMIN:   THEATRE_COORDINATOR.id,
    Role.AI_TEAM: THEATRE_COORDINATOR.id,
    Role.DEVOPS:  THEATRE_COORDINATOR.id,
    Role.CLIENT:  SURGEON.id,
}
"""Coarse RBAC role -> the persona a principal of that role adopts.

EVERY role must appear and EVERY value must be a key of PERSONAS. Every authenticated
principal resolves through this table at the login boundary, and a missing or stale
entry raises KeyError there — which no suite in this repository exercises.
"""
```

**Two personas is the right number for a demo:** one staff/operational persona with `ScopeKind.ALL` and the full allowlist, one end-user persona with a row-scoped `DataScope` and a small allowlist. The contrast is what makes the RLS and scope story visible on screen.

### 4. The resolvers

```python
def get_persona(persona_id: str | None) -> Persona:
    """Return the persona for ``persona_id`` (the default when ``None``).

    Raises:
        KeyError: For an id this domain does not declare. Deliberately loud: an
            unknown persona means an unenforceable data scope, and answering with the
            default would silently widen what the caller may see.
    """
    if persona_id is None:
        return PERSONAS[DEFAULT_PERSONA_ID]
    return PERSONAS[persona_id]


def persona_for_role(role: Role | str) -> str:
    """Return the persona id a principal holding ``role`` adopts.

    Accepts the enum or its string value, because the host resolves roles from a JWT
    claim that is a plain string.

    Raises:
        KeyError: naming the table, so the fix is obvious from the traceback.
    """
    key = role if isinstance(role, Role) else Role(role)
    try:
        return PERSONA_BY_ROLE[key]
    except KeyError as exc:
        raise KeyError(
            f"Role {key!r} has no entry in personas.PERSONA_BY_ROLE — every role the "
            f"platform can issue must map to a persona id that exists in PERSONAS."
        ) from exc
```

Reproduce that error message shape. The reference adapter carries one for exactly this reason.

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter.personas as p
from reference.adapter.tools import ALLOWLIST, TOOL_REGISTRY

assert p.DEFAULT_PERSONA_ID in p.PERSONAS, 'DEFAULT_PERSONA_ID is not a PERSONAS key'
for role, pid in p.PERSONA_BY_ROLE.items():
    assert pid in p.PERSONAS, f'PERSONA_BY_ROLE[{role!r}] -> unknown persona {pid!r}'
for pid in ALLOWLIST:
    assert pid in p.PERSONAS, f'ALLOWLIST names unknown persona {pid!r}'
for pid, persona in p.PERSONAS.items():
    assert persona.tool_names <= set(TOOL_REGISTRY), f'{pid}: unknown tools'
    assert p.get_persona(pid).id == pid
assert p.get_persona(None).id == p.DEFAULT_PERSONA_ID
print('personas:', sorted(p.PERSONAS))
print('by role :', {str(k): v for k, v in p.PERSONA_BY_ROLE.items()})
for pid, persona in sorted(p.PERSONAS.items()):
    print(f'  {pid:24s} scope={persona.data_scope.kind.value:5s} tools={len(persona.tool_names)}')
"
```

Every RBAC role must be present. Confirm you covered all four:

```bash
uv run python -c "
from app.api.schemas import Role
import reference.adapter.personas as p
missing = [r for r in Role if r not in p.PERSONA_BY_ROLE]
assert not missing, f'roles with no persona: {missing}'
print('every RBAC role maps')
" 2>/dev/null || echo "run this one after the sync, from the backend venv"
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k persona)

# and the one that actually goes near the login path
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    tests/api/test_roles_rbac.py -q)
```

Verify pieces 5 and 6 together with `tests/adapter/test_registry.py`.

### Checklist

- [ ] `PERSONAS` is a `dict[str, Persona]` keyed by `.id`.
- [ ] `DEFAULT_PERSONA_ID` is a key of `PERSONAS`.
- [ ] **`PERSONA_BY_ROLE` has an entry for `admin`, `ai_team`, `devops` and `client`.**
- [ ] Every `PERSONA_BY_ROLE` value is a key of `PERSONAS`.
- [ ] Every key of piece 4's `ALLOWLIST` is a key of `PERSONAS`.
- [ ] `Persona.tool_names` reads `ALLOWLIST` — no second copy of the tool list.
- [ ] Every persona's `prompt_key` will exist in piece 6's `SYSTEM_PROMPTS`.
- [ ] At least one persona is row-scoped (`ScopeKind.OWN` with a `subject_field`).
- [ ] `get_persona(None)` returns the default; `get_persona("<unknown>")` raises.
- [ ] `persona_for_role` accepts the enum **and** its string value.
- [ ] `ScopeKind` and `Persona` keep those exact names — piece 6 imports both.
- [ ] No persona id from the reference domain survives.

---

## Next

`prompts/06-prompts.md` — paired with this piece.

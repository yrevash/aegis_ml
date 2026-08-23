# PROMPT 04 · Piece 4 — `tools.py`

---

## Role

You are writing **piece 4 of 10**: what the agent can *do*, at what risk, and who may ask for it. The `risk` field on each tool is the **only** input to the human approval gate.

---

## Inputs

- `DOMAIN_BRIEF.md` §9 (tools, risk tiers, allowlist), §2 (entities).
- Piece 1's models.
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/tools.py`.

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/tools.py
```

---

## The contract to satisfy

```python
@runtime_checkable
class ToolSpecLike(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def risk(self) -> RiskLevel: ...
    def definition(self) -> dict[str, Any]: ...


@runtime_checkable
class ToolsModule(Protocol):
    @property
    def TOOL_REGISTRY(self) -> Mapping[str, ToolSpecLike]: ...
    @property
    def ALLOWLIST(self) -> Mapping[str, frozenset[str]]: ...

    def is_allowed(self, persona_id: str, tool_name: str) -> bool: ...
    def tools_for(self, persona_id: str) -> Sequence[ToolSpecLike]: ...
    def tool_definitions_for(self, persona_id: str) -> list[dict[str, Any]]: ...
    async def run_tool(self, persona_id: str, tool_name: str, args: dict[str, Any], ctx: Any) -> Any: ...
```

**Plus the host-bound symbols** (verified by grep against `backend/src/`; `app/mcp/server.py` and `backend/tests/` import them directly from `app.adapter.tools`):

```
ALLOWLIST  AuditFn  RecordStore  TOOL_REGISTRY  ToolActionResult
ToolContext  ToolNotAllowedError  UnknownToolError  is_allowed
```

Drop any of those and the host raises `ImportError` at startup.

---

## What to write

### 1. Protocols and the store

```python
class RecordStore(Protocol):
    """The record surface a tool handler reads and writes."""

    def get_procedure(self, procedure_id: str) -> Procedure | None: ...
    def list_procedures(self) -> list[Procedure]: ...
    def put_procedure(self, procedure: Procedure) -> None: ...


class AuditFn(Protocol):
    """The append-only audit sink every mutating tool must call."""

    async def __call__(
        self, *, action: str, actor: str | None, model: str | None,
        trace_id: str | None, payload: dict, approved_by: str | None = None,
    ) -> None: ...


class InMemoryRecordStore:
    """A RecordStore over a generated dataset. Bound by app.agent.deps and app.mcp.server."""

    def __init__(self, procedures: list[Procedure] | None = None) -> None: ...

    @classmethod
    def from_dataset(cls, dataset: object) -> InMemoryRecordStore: ...
```

**Keep `from_dataset` as a classmethod taking the dataset object.** `app/agent/deps.py` and `app/mcp/server.py` both call `InMemoryRecordStore.from_dataset(generate_synthetic_sync())`.

### 2. Errors, results, context

```python
class UnknownToolError(KeyError):
    """The model named a tool that is not in the registry."""

class ToolNotAllowedError(PermissionError):
    """The persona is not allowlisted for this tool."""


class InverseAction(BaseModel):
    """How to undo this action, if it can be undone."""
    tool: str
    args: dict


class ToolActionResult(BaseModel):
    """What one tool call did."""
    ok: bool
    changed: bool
    summary: str
    previous_state: dict = Field(default_factory=dict)
    inverse: InverseAction | None = None


@dataclass
class ToolContext:
    """Everything a handler needs that is not an argument."""
    store: RecordStore
    actor: str | None = None
    model: str | None = None
    trace_id: str | None = None
    approved_by: str | None = None
    audit: AuditFn | None = field(default=None)
```

`previous_state` and `inverse` are what make an action **reversible**, which is what lets a HIGH-risk write be approved with confidence.

### 3. `ToolSpec`

```python
@dataclass(frozen=True)
class ToolSpec:
    """One registered action tool."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    risk: RiskLevel
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False

    def definition(self) -> dict:
        """Return the OpenAI/MCP function schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }
```

### 4. One handler per action

```python
async def resequence_list(args: dict, ctx: ToolContext) -> ToolActionResult:
    """Re-order a theatre day's list. HIGH risk: externally visible, patients are told."""
    parsed = ResequenceArgs.model_validate(args)
    day = ctx.store.get_theatre_day(parsed.theatre_day_id)
    if day is None:
        return ToolActionResult(ok=False, changed=False,
                                summary=f"No theatre day {parsed.theatre_day_id!r}.")

    previous = {"order": [p.id for p in day.ordered_procedures()]}
    updated = _apply_order(day, parsed.order)
    ctx.store.put_theatre_day(updated)

    await _emit_audit(ctx, action="resequence_list",
                      payload={"theatre_day_id": parsed.theatre_day_id, **previous})
    return ToolActionResult(
        ok=True, changed=True,
        summary=f"Re-sequenced {parsed.theatre_day_id} to {len(parsed.order)} slots.",
        previous_state=previous,
        inverse=InverseAction(tool="resequence_list",
                              args={"theatre_day_id": parsed.theatre_day_id,
                                    "order": previous["order"]}),
    )
```

Every handler: **typed** (`args_model.model_validate`), **audited** (via `ctx.audit`), **returns `ToolActionResult`**, and **never raises for a business-logic miss** — a missing record is `ok=False`, not an exception.

### 5. The registry

```python
TOOL_REGISTRY: dict[str, ToolSpec] = {
    "find_procedures": ToolSpec(
        name="find_procedures",
        description="Search booked procedures by theatre, day, surgeon or status.",
        args_model=FindProceduresArgs, handler=find_procedures,
        risk=RiskLevel.LOW, read_only=True, idempotent=True,
    ),
    "add_theatre_note": ToolSpec(
        name="add_theatre_note",
        description="Append a note to a procedure's record. " + _ID_RULE,
        args_model=AddNoteArgs, handler=add_theatre_note,
        risk=RiskLevel.LOW, idempotent=False,     # appending twice appends twice
    ),
    "reassign_theatre": ToolSpec(
        name="reassign_theatre",
        description="Move a procedure to a different theatre. " + _ID_RULE,
        args_model=ReassignArgs, handler=reassign_theatre,
        risk=RiskLevel.MEDIUM, idempotent=True,
    ),
    "resequence_list": ToolSpec(
        name="resequence_list",
        description="Re-order a theatre day's list. " + _ID_RULE,
        args_model=ResequenceArgs, handler=resequence_list,
        risk=RiskLevel.HIGH, destructive=True, idempotent=True,
    ),
}
```

Add the ML tools (`docs/07-integration-with-aegis.md` §7·bis):

```python
from aegis_ml.serve.tools import ml_tool_specs

TOOL_REGISTRY.update({spec.name: spec for spec in ml_tool_specs(ToolSpec)})
```

All five (`predict_outcome`, `explain_prediction`, `whatif_scenario`, `forecast_series`, `check_model_health`) are LOW and read-only, because **ML informs, it never gates.**

### 6. `ALLOWLIST` and the functions

```python
ALLOWLIST: dict[str, frozenset[str]] = {
    "theatre_coordinator": frozenset(TOOL_REGISTRY),
    "surgeon": frozenset({"add_theatre_note", "predict_outcome", "explain_prediction"}),
}


def is_allowed(persona_id: str, tool_name: str) -> bool: ...
def tools_for(persona_id: str) -> list[ToolSpec]: ...          # sorted by name
def tool_definitions_for(persona_id: str) -> list[dict]: ...

async def run_tool(persona_id: str, tool_name: str, args: dict, ctx: ToolContext) -> ToolActionResult:
    """Authorise and execute one tool.

    Raises UnknownToolError then ToolNotAllowedError BEFORE any side effect: the
    allowlist is an authorisation boundary, not a filter applied afterwards.
    """
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise UnknownToolError(tool_name)
    if not is_allowed(persona_id, tool_name):
        raise ToolNotAllowedError(f"{persona_id!r} may not call {tool_name!r}")
    return await spec.handler(args, ctx)
```

---

## The risk tier is the whole human gate

> *"A tool at or above `AgentConfig.gate_min_risk` (platform default `HIGH`) pauses for a human. There is no second signal — not model confidence, not the ML prediction."*

Mark a consequential, externally-visible write `HIGH` and the approval gate appears with **no engine change at all**.

**Assign at least one HIGH-risk tool.** A domain with no gated action cannot demonstrate the human gate, which is one of Aegis's six trust checkpoints and the thing the demo is about.

| Question | Tier |
|---|---|
| Reads only, no side effect | **LOW**, `read_only=True` |
| Writes something reversible and internal (a note, a tag) | **LOW** |
| Writes something a colleague would notice (an assignment, a routing change) | **MEDIUM** |
| Writes something a **customer or patient** would notice, or that is expensive to undo | **HIGH** |
| Deletes, sends, charges, publishes | **HIGH** + `destructive=True` |

### `destructive` and `idempotent` are independent of risk

`ToolSpecLike`'s docstring: *"they belong on the tool because risk does not imply idempotency."*

| Tool | risk | destructive | idempotent | Why |
|---|---|---|---|---|
| note-append | LOW | ✗ | **✗** | appending twice appends twice |
| status change | **HIGH** | ✓ | **✓** | setting it twice converges |

Omit both and the conservative reading is published, which is safe but says less than you know. The MCP surface publishes them as advisory hints, and the host's MCP annotations **used to keep them in a table keyed by the shipped domain's tool names** — a retarget renamed every key and silently lost them, which is why they live on the tool now.

---

## The trap, and it fails safe

> **An unregistered tool name resolves to `HIGH`.** So forgetting to register something makes it require approval rather than run unguarded.
>
> **Do not rely on that.** It means a forgotten registration looks like an over-cautious gate rather than a bug, and you will spend twenty minutes wondering why a read is asking for approval.

**Second trap — allowlist typos fail open into silence.** The allowlist is read with `.get(persona_id, frozenset())` and the tool set is filtered by membership. A misspelled **persona key** gives that persona *no tools at all*; a misspelled **tool name** simply never appears in the model's `tools=` payload. Neither raises, and the agent answers the question anyway. Conformance check #6 covers it — run it.

**Third trap — `RiskLevel` must be the real enum.** Import it from where the host does (`app.api.schemas.RiskLevel`, backed by `aegis.core.types.RiskLevel`). A string `"high"` is not a `RiskLevel` and conformance check #5 fails.

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter.tools as t
print('tools:', sorted(t.TOOL_REGISTRY))
for n, s in sorted(t.TOOL_REGISTRY.items()):
    print(f'  {n:24s} risk={s.risk.value:6s} ro={s.read_only!s:5s} destr={s.destructive!s:5s} idem={s.idempotent!s:5s}')
assert any(s.risk.value.lower() == 'high' for s in t.TOOL_REGISTRY.values()), 'no HIGH-risk tool'
for persona, names in t.ALLOWLIST.items():
    bad = names - set(t.TOOL_REGISTRY)
    assert not bad, f'{persona}: unknown tools {bad}'
for s in t.TOOL_REGISTRY.values():
    d = s.definition()
    assert d['type'] == 'function' and d['function']['name'] == s.name
print('registry, allowlist and definitions consistent')
"
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    tests/adapter/test_tools.py tests/adapter/test_allowlist.py -q)
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k "risk or allowlist")
```

`test_tools.py` carries **26** shipped-domain literals and `test_allowlist.py` **19** — the two heaviest files in `backend/tests/adapter/`. Rewriting them is part of this step.

### Checklist

- [ ] Every tool declares a real `RiskLevel`.
- [ ] At least one tool is `HIGH`.
- [ ] `destructive` and `idempotent` are asserted per tool, and are not copies of `risk`.
- [ ] Every handler validates its args with a pydantic model.
- [ ] Every mutating handler calls `ctx.audit`.
- [ ] Every mutating handler returns `previous_state` and, where possible, an `inverse`.
- [ ] `run_tool` raises `UnknownToolError` then `ToolNotAllowedError` **before** any side effect.
- [ ] Every `ALLOWLIST` key is a persona id piece 5 will declare; every value is a subset of `TOOL_REGISTRY`.
- [ ] The five `aegis_ml.serve.tools` specs are registered, LOW and read-only.
- [ ] `ALLOWLIST`, `AuditFn`, `RecordStore`, `TOOL_REGISTRY`, `ToolActionResult`, `ToolContext`, `ToolNotAllowedError`, `UnknownToolError`, `is_allowed`, `InMemoryRecordStore` are all present by those exact names.
- [ ] `InMemoryRecordStore.from_dataset(dataset)` exists as a classmethod.
- [ ] No tool name from the reference domain survives.

---

## Next

`prompts/05-personas.md`.

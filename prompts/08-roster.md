# PROMPT 08 · Piece 8 — `roster.py`

**This is the piece with the real exception to "only the adapter changes". Read §"The core constraint" before you write it.**

---

## Role

You are writing **piece 8 of 10**: which specialists the supervisor may hand a turn to, and which team a wide turn may fan out across. Two rosters, two mechanisms, and they are not interchangeable.

---

## Inputs

- `DOMAIN_BRIEF.md` §10 (roster), §9 (tools).
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/roster.py`.

## Output file

```
/Users/yrevash/aegis_ml/reference/adapter/roster.py
```

---

## The contract to satisfy

```python
@runtime_checkable
class AgentRosterLike(Protocol):
    @property
    def specialists(self) -> Sequence[Any]: ...
    @property
    def default_role(self) -> str: ...
    def roles(self) -> list[str]: ...
    def named(self) -> list[Any]: ...


@runtime_checkable
class RosterModule(Protocol):
    def agent_roster(self) -> AgentRosterLike: ...
    def sub_agent_roster(self) -> Sequence[Any]: ...
```

Host-bound beyond the Protocol: `sub_agent_roster` is imported directly from `app.adapter.roster` by `app/agent/deps.py`, and `AgentRoster` / `RosterSpecialist` / `agent_roster` are imported by `backend/tests/`.

---

## **The core constraint**

`aegis/src/aegis/agent/graph.py`:

```python
SPECIALIST_NODES: dict[str, str] = {
    "qa":     "recall_memory",   # the full retrieve -> plan -> gate -> act pipeline
    "memory": "answer_memory",   # answers from long-term memory, skipping RAG/tools
    "team":   "plan_team",       # router-written fan-out; NOT a roster role
}
```

> **A roster role that is not a key in that map is not routable.** It falls back to the `qa` pipeline and **logs a warning — it does not raise.** The new specialist answers as QA, and the `routing` stream event *still names it*, so the console shows it being chosen.
>
> The build-time warning meant to catch this could not fire either: it iterated `roster.roles` (the bound method, not the list) and the surrounding `except Exception` swallowed the `TypeError`.

So:

| What you want | What to do |
|---|---|
| Domain-specific specialists | **Re-voice `qa` and `memory`.** Change `description` and `keywords` freely, **keep the two role strings.** No core edit. **This is what you should do.** |
| A genuinely new third specialist | Requires a handler node in `graph.py` **plus** a `SPECIALIST_NODES` entry. That is the **one other sanctioned core edit** and it must be **reported**, not done quietly. Avoid it under time pressure. |
| `team` | **Never declare it in a roster.** The router writes it when the depth classifier chooses fan-out. |

Conformance checks #3 and #4 enforce both halves. Check #3 is *"the highest-value check in the suite, because adding a specialist is the most natural first thing to do to an adapter and the failure is completely silent."*

---

## What to write

### 1. `RosterSpecialist` and `AgentRoster`

```python
@dataclass(frozen=True)
class RosterSpecialist:
    """One lane the supervisor may hand a turn to."""

    role: str
    description: str
    keywords: tuple[str, ...] = ()
    is_default: bool = False


@dataclass(frozen=True)
class AgentRoster:
    """The supervisor's hand-off set."""

    specialists: tuple[RosterSpecialist, ...]

    @property
    def default_role(self) -> str:
        """The fall-through role when no specialist matched."""
        for spec in self.specialists:
            if spec.is_default:
                return spec.role
        return self.specialists[0].role      # ← see the trap below

    def roles(self) -> list[str]:
        """Every routable role id, in declaration order."""
        return [s.role for s in self.specialists]

    def named(self) -> list[RosterSpecialist]:
        """The keyword-matchable specialists the classifier may choose between."""
        return [s for s in self.specialists if not s.is_default]
```

> **The `default_role` trap.** That fallback to `specialists[0]` is the reference behaviour and it is why check #4 exists: *"a roster that forgets `is_default` silently promotes whichever specialist happens to be written first, and if that role has no node, every unmatched turn takes the `qa` fallback path."* Mark exactly one specialist `is_default=True` and never rely on declaration order.

### 2. The two specialists — re-voiced, not renamed

```python
_ROSTER = AgentRoster(
    specialists=(
        RosterSpecialist(
            role="qa",
            description=(
                "Answers questions about theatre lists, procedures, delay risk and "
                "scheduling policy, using retrieval, the ML spine and the action "
                "tools. The default lane for anything operational."
            ),
            is_default=True,
        ),
        RosterSpecialist(
            role="memory",
            description=(
                "Answers from what we already know about this clinician — their "
                "standing preferences, constraints and past commitments — without "
                "searching the corpus or calling a tool."
            ),
            keywords=(
                "what do you know about me",
                "what do you remember",
                "do you remember",
                "you remember about me",
                "know about me",
                "remember about me",
                "what have i told you",
                "what did i tell you",
                "my past conversations",
                "our previous discussions",
                "my usual preference",
                "what you know about me",
            ),
        ),
    )
)


def agent_roster() -> AgentRoster:
    """Return the specialists the supervisor may route a turn to."""
    return _ROSTER
```

- **`role="qa"` and `role="memory"` are literal strings you must not change.** Everything else is yours.
- **The default specialist gets no keywords** — it is the fallback; keywords on it would compete with the specialist that should have matched.
- **The `memory` keywords are phrasings, not single words.** Whole phrases are what the classifier matches; a bare `"remember"` fires on "remember to move that case", which is an operational request.

### 3. The fan-out team

```python
def sub_agent_roster() -> tuple[SubAgentSpec, ...]:
    """Return the ``SubAgentSpec`` team a wide turn fans out across (may be empty)."""
    return _SUB_AGENTS


_SUB_AGENTS: tuple[SubAgentSpec, ...] = (
    SubAgentSpec(
        agent_id="risk",
        role="risk",
        label="Delay-risk agent",
        system_prompt=(
            "You assess slot-overrun risk across a theatre day. Use the ML tools to "
            "predict and explain, and report the two or three cases that dominate the "
            "day's risk. Always quote the conformal interval, never a bare number."
        ),
        tool_allowlist=frozenset({"predict_outcome", "explain_prediction", "find_procedures"}),
    ),
    SubAgentSpec(
        agent_id="policy",
        role="policy",
        label="Policy agent",
        system_prompt=(
            "You answer from the theatre service's written policy and runbooks. Cite "
            "the document. If no policy covers the question, say so plainly."
        ),
    ),
    SubAgentSpec(
        agent_id="scheduling",
        role="scheduling",
        label="Scheduling agent",
        system_prompt=(
            "You propose the least disruptive intervention that keeps a list on time: "
            "a single move before a re-sequence, a re-sequence before a cancellation."
        ),
        tool_allowlist=frozenset({"find_procedures", "reassign_theatre", "resequence_list"}),
    ),
)
```

`SubAgentSpec` comes from `aegis.agent`. Fields used: `agent_id`, `role`, `label`, `system_prompt`, `tool_allowlist` (optional), `model_role` (leave at the core default). The sub-agent `role` strings are **not** graph roles — they are labels for the fan-out lanes, so they are free.

---

## The trap on the fan-out team

> **`tool_allowlist` holds literal tool names and is *intersected* with `TOOL_REGISTRY`.** A stale name is **silently dropped**, and the sub-agent runs with fewer tools than you think — or none.
>
> The shipped `data` lane allowlists `{"update_request_status", "add_case_note"}` — both of which your retarget deletes. Nothing raises.

Also: **re-voice every `label` and `system_prompt`.** They are read by the model and **shown on screen**. A "Data agent" lane whose prompt talks about service requests is visible in the console during the demo.

Conformance check #6 (`test_allowlists_name_registered_tools_and_known_personas`) covers every sub-agent allowlist as well as `ALLOWLIST`.

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
import reference.adapter.roster as r
from reference.adapter.tools import TOOL_REGISTRY

SPECIALIST_NODES = {'qa', 'memory', 'team'}    # aegis/src/aegis/agent/graph.py

roster = r.agent_roster()
roles = roster.roles()
print('roles       :', roles)
print('default     :', roster.default_role)
print('named       :', [s.role for s in roster.named()])

unroutable = [x for x in roles if x not in SPECIALIST_NODES]
assert not unroutable, f'roles with no graph node (they will answer as qa): {unroutable}'
assert 'team' not in roles, \"never declare 'team' — the router writes it\"
assert sum(s.is_default for s in roster.specialists) == 1, 'exactly one is_default required'
assert roster.default_role in SPECIALIST_NODES

for spec in r.sub_agent_roster():
    allow = getattr(spec, 'tool_allowlist', None) or frozenset()
    bad = allow - set(TOOL_REGISTRY)
    assert not bad, f'{spec.agent_id}: tool_allowlist names missing tools {sorted(bad)}'
    print(f'  {spec.agent_id:12s} {spec.label:24s} tools={sorted(allow) or \"(all)\"}')
print('roster consistent')
"
```

Then check nothing in the roster still speaks the old domain:

```bash
grep -niE "request|case note|service desk|support agent|ticket" reference/adapter/roster.py
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k "roster or allowlist")
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/agent/test_router.py -q)
```

Then run a query and read the `routing` stream event, or check the build warning.

**Piece 8 is the point at which the whole suite can go green again.** `backend/tests/conftest.py` imports through `app.adapter`, so it has been failing at import since piece 1. Once this lands and every re-export resolves, this is the first command that can pass:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest tests/adapter tests/agent -q)
```

…and only once you have rewritten `tests/adapter/*` for your own domain, which is part of pieces 1–8, not a follow-up.

### Checklist

- [ ] The specialist role strings are **exactly** `"qa"` and `"memory"`.
- [ ] `"team"` does not appear as a roster role.
- [ ] Exactly one specialist has `is_default=True`.
- [ ] The default specialist has no keywords.
- [ ] `memory`'s keywords are whole phrasings, not single words.
- [ ] Both descriptions are re-voiced for this domain.
- [ ] `AgentRoster` exposes `specialists`, `default_role`, `roles()` and `named()`.
- [ ] Every `tool_allowlist` name is a key of `TOOL_REGISTRY`.
- [ ] Every `SubAgentSpec` `label` and `system_prompt` is re-voiced.
- [ ] `sub_agent_roster()` is importable directly from `<adapter>.roster`.
- [ ] No core edit was made. If one was, it is a `SPECIALIST_NODES` addition and it is **in the report**.

---

## Next

`prompts/09-corpus.md`.

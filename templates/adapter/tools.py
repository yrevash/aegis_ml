"""Piece 4 of 10 — what the agent can *do*, at what risk, and who may ask for it.

WHAT YOU WRITE HERE
    One ``async def handler(args, ctx) -> ToolActionResult`` per real action, each:

      * **typed** — arguments validated by a pydantic model, and the MCP/OpenAI
        ``function`` schema derived from that same model, so validation and the
        ``tools=`` payload can never disagree;
      * **idempotent** — re-running with identical arguments converges; the second run
        reports ``changed=False``;
      * **reversible** — the result carries an :class:`InverseAction` (a tool name +
        args) that undoes it, which is what makes a rejected proposal cleanly
        recoverable;
      * **audited** — every invocation appends an audit row through the injected sink;
      * **registered** in :data:`TOOL_REGISTRY` with an honest ``risk`` tier;
      * **allowlisted** in :data:`ALLOWLIST`, persona → tool names.

    Include at least one **read-only lookup** tool. The write tools all take an id, and
    the persona prompt correctly forbids inventing one — so a registry of writes with
    no read is a registry the planner can never legitimately reach. That is not
    hypothetical: it is why ``find_*`` exists in the reference adapter.

THE CONTRACT (aegis.adapter.ToolsModule / ToolSpecLike) — these names must survive
    TOOL_REGISTRY, ALLOWLIST, is_allowed(), tools_for(),
    tool_definitions_for(), run_tool()
    ToolSpec.name / .description / .risk / .definition()

    Host-bound beyond the Protocol (verified by grep against backend/src/app):
    ``ToolContext``, ``ToolActionResult``, ``InMemoryRecordStore``, ``RecordStore``,
    ``AuditFn``, ``UnknownToolError``, ``ToolNotAllowedError``. Keep every one of those
    names; the MCP server and client import them directly from this module.

THE TRAP
    **``risk`` is the ONLY input to the human approval gate.** Not model confidence,
    not the ML prediction, not the persona. A tool at or above
    ``AgentConfig.gate_min_risk`` (platform default ``HIGH``) pauses for a human; one
    below it just runs. Mark a consequential, externally-visible write ``HIGH`` and the
    approval gate appears with no engine change at all.

    An **unregistered** tool name resolves to ``HIGH``, so a forgotten registration
    fails safe — it demands approval rather than running unguarded. Do not lean on
    that: it means a missing registration looks like an over-cautious gate rather than
    a bug, which is a much slower thing to find.

    ``destructive`` and ``idempotent`` are asserted per tool, never derived from risk.
    Risk does not imply idempotency: appending a note is LOW risk and *not* idempotent,
    while a gated stage transition is HIGH risk and *is*. Both default to the cautious
    reading, so a tool registered without thinking is never advertised as safer than it
    is. They used to live in a name-keyed table inside the host's MCP module — core
    naming this domain's tools — so a retarget silently lost every hint.

    ``ALLOWLIST`` is checked **before any side effect**. A tool listed for a persona
    whose data scope does not cover it is not a roster line, it is a scope change: see
    the note on the narrow persona below.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        tests/adapter/test_tools.py tests/adapter/test_allowlist.py -q)
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        --pyargs aegis.conformance --aegis-adapter app.adapter -q -k "tool or allowlist")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, Field

from app.adapter.schema import (
    IntakePath,
    ItemNote,
    UrgencyBand,
    WidgetKind,
    WorkItem,
    WorkItemStage,
)
from app.api.schemas import RiskLevel

# ─────────────────────────────────────────────────────────────────────────────
# Injected data-layer contracts (structural — no hard import of app.data)
#
# The store and the audit sink are injected through ToolContext so every tool is
# unit-testable with no Postgres and no network. In production the core wires the real
# ones in; the lazy fallbacks below keep this module importable before they exist.
# ─────────────────────────────────────────────────────────────────────────────


class RecordStore(Protocol):
    """The minimal record access the tools need (a data-layer view).

    TODO(domain): rename the three methods to your record type, and keep all three
    shapes. ``list_*`` is not optional: ``get_*`` can only answer a question that
    already knows the answer's id, so a store offering nothing else makes an id-taking
    tool the only thing a planner can reach — and a planner that is (correctly)
    forbidden from inventing ids can therefore reach nothing.
    """

    def get_item(self, item_id: str) -> WorkItem | None:
        """Return the record with ``item_id``, or None if absent."""
        ...

    def list_items(self) -> list[WorkItem]:
        """Return every record this store holds, in a stable order."""
        ...

    def put_item(self, item: WorkItem) -> None:
        """Persist ``item`` (insert or replace by id)."""
        ...


class AuditFn(Protocol):
    """Structural view of the host's ``record_audit``. Keep this name — MCP imports it."""

    async def __call__(
        self,
        *,
        action: str,
        actor: str | None,
        model: str | None,
        trace_id: str | None,
        payload: dict,
        approved_by: str | None = None,
    ) -> None:
        """Append one immutable audit record."""
        ...


class InMemoryRecordStore:
    """A dict-backed :class:`RecordStore` for demos, seeding and tests.

    Keep this class name: the host's agent deps and MCP server both import it from
    ``app.adapter`` by name to build the process-wide demo store.
    """

    def __init__(self, items: list[WorkItem] | None = None) -> None:
        """Initialise the store, optionally seeded with ``items``."""
        self._items: dict[str, WorkItem] = {i.id: i for i in (items or [])}

    @classmethod
    def from_dataset(cls, dataset: object) -> InMemoryRecordStore:
        """Build a store from a :class:`~app.adapter.schema.SyntheticDataset`.

        Args:
            dataset: An object exposing an ``items`` iterable of :class:`WorkItem`.
                TODO(domain): point ``getattr`` at your dataset's own collection name.

        Returns:
            A populated in-memory store.
        """
        return cls(list(getattr(dataset, "items", [])))

    def get_item(self, item_id: str) -> WorkItem | None:
        """Return the record with ``item_id``, or None if absent."""
        return self._items.get(item_id)

    def list_items(self) -> list[WorkItem]:
        """Return all stored records (public, ordered by insertion).

        A stable public accessor so callers never reach into the private mapping.
        """
        return list(self._items.values())

    def put_item(self, item: WorkItem) -> None:
        """Persist ``item`` (insert or replace by id)."""
        self._items[item.id] = item


# ─────────────────────────────────────────────────────────────────────────────
# Errors  (keep both names and both base classes — the host catches them by type)
# ─────────────────────────────────────────────────────────────────────────────


class UnknownToolError(KeyError):
    """Raised when a tool name is not in the registry."""


class ToolNotAllowedError(PermissionError):
    """Raised when a persona attempts a tool outside its allowlist."""


# ─────────────────────────────────────────────────────────────────────────────
# Results + execution context
# ─────────────────────────────────────────────────────────────────────────────


class InverseAction(BaseModel):
    """A (tool, args) pair that reverses a completed action."""

    tool: str
    args: dict


class ToolActionResult(BaseModel):
    """The typed outcome of running an action tool.

    ``ok`` and ``summary`` are the two fields the agent loop reads structurally, so
    keep both. ``summary`` is pasted verbatim into the next planning prompt — write it
    for a model *and* for a human watching the trace.
    """

    ok: bool = Field(description="Whether the action succeeded.")
    changed: bool = Field(description="Whether state actually changed (idempotency).")
    summary: str = Field(description="Human-readable, demoable result summary.")
    previous_state: dict = Field(
        default_factory=dict, description="Prior values, for audit / rollback."
    )
    inverse: InverseAction | None = Field(
        default=None, description="Action that undoes this one, if reversible."
    )


@dataclass
class ToolContext:
    """Everything a tool needs to run, injected by the caller.

    Attributes:
        store: The record store the tool reads and writes.
        actor: Persona/agent id performing the action (for the audit trail).
        model: Model id that proposed the action, if any.
        trace_id: Observability correlation id, if any.
        approved_by: Who approved the action at the human gate, if applicable.
        audit: The audit sink; resolved lazily from the host at run time when absent.
    """

    store: RecordStore
    actor: str | None = None
    model: str | None = None
    trace_id: str | None = None
    approved_by: str | None = None
    audit: AuditFn | None = field(default=None)


# ─────────────────────────────────────────────────────────────────────────────
# Argument models (drive both validation and the MCP tool schema)
#
# TODO(domain): one model per tool. Every field needs a `description` — it is what the
# model reads to decide what to pass, and a field described only by its name is a field
# the planner fills in from the words of the question.
# ─────────────────────────────────────────────────────────────────────────────

FIND_ITEMS_DEFAULT_LIMIT = 10
"""Rows :func:`find_work_items` returns when the caller does not ask for a number."""

FIND_ITEMS_MAX_LIMIT = 25
"""Hard ceiling on :func:`find_work_items`, enforced in the args model **and** the body.

A lookup tool exists to hand the planner a *shortlist* to choose from, not a table dump.
Every row is pasted verbatim into the next planning prompt, so an unbounded limit is
simultaneously a token bill, a context-window risk and a larger blast radius for the
tool-result injection rail to screen. The ceiling lives in the pydantic model (so an
over-large ``limit`` is a validation error the planner sees and can correct) and is
re-applied in the body (so a direct caller that bypassed the model still cannot exceed
it).
"""


class FindItemsArgs(BaseModel):
    """Arguments for :func:`find_work_items` — every filter optional, all AND-ed."""

    stage: WorkItemStage | None = Field(
        default=None, description="Only records in this lifecycle stage."
    )
    urgency: UrgencyBand | None = Field(
        default=None, description="Only records at this urgency band."
    )
    kind: WidgetKind | None = Field(
        default=None, description="Only records of this kind."
    )
    intake: IntakePath | None = Field(
        default=None, description="Only records that arrived via this path."
    )
    party_id: str | None = Field(
        default=None, description="Only records raised for this party id."
    )
    assigned_operator_id: str | None = Field(
        default=None,
        description="Only records assigned to this operator id. Use '' for unassigned.",
    )
    text: str | None = Field(
        default=None,
        description=(
            "Case-insensitive substring to match in the record id, title or detail. "
            "Pass a known id here to confirm that record exists."
        ),
    )
    oldest_first: bool = Field(
        default=True,
        description="Order by creation time: oldest first (default), else newest first.",
    )
    limit: int = Field(
        default=FIND_ITEMS_DEFAULT_LIMIT,
        ge=1,
        le=FIND_ITEMS_MAX_LIMIT,
        description=f"Maximum rows to return (1–{FIND_ITEMS_MAX_LIMIT}).",
    )


class AdvanceStageArgs(BaseModel):
    """Arguments for :func:`advance_item_stage`."""

    item_id: str = Field(description="Id of the record to update.")
    stage: WorkItemStage = Field(description="Target lifecycle stage.")


class AssignArgs(BaseModel):
    """Arguments for :func:`assign_work_item`."""

    item_id: str = Field(description="Id of the record to (re)assign.")
    operator_id: str | None = Field(
        default=None, description="Operator id to assign, or null to unassign."
    )


class AddNoteArgs(BaseModel):
    """Arguments for :func:`add_item_note`."""

    item_id: str = Field(
        description=(
            "Id of the record to annotate, as returned by find_work_items. Not a title, "
            "a document name or a phrase from the question."
        )
    )
    body: str = Field(min_length=1, description="Note text to append.")
    author: str | None = Field(default=None, description="Who wrote the note.")
    retract: bool = Field(
        default=False, description="If true, remove the matching note (the inverse)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


async def _emit_audit(ctx: ToolContext, action: str, payload: dict) -> None:
    """Record one audit entry, resolving the sink lazily if it was not injected."""
    audit = ctx.audit
    if audit is None:
        try:  # app.data may not exist yet during parallel builds.
            from app.data import record_audit as audit  # noqa: PLC0415
        except ImportError:
            return  # Best-effort: never let a missing sink block the action.
    await audit(
        action=action,
        actor=ctx.actor,
        model=ctx.model,
        trace_id=ctx.trace_id,
        payload=payload,
        approved_by=ctx.approved_by,
    )


def _age_hours(item: WorkItem, *, now: datetime) -> float:
    """Return how many hours ago ``item`` was opened (0.0 if it is in the future).

    Timestamps are tz-aware in generated data and tz-naive in hand-built fixtures, so a
    naive value is read as UTC rather than being allowed to raise mid-lookup.
    """
    created = item.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return max(0.0, (now - created).total_seconds() / 3600.0)


def _matches(item: WorkItem, parsed: FindItemsArgs) -> bool:
    """Return whether ``item`` satisfies every filter set on ``parsed``."""
    if parsed.stage is not None and item.stage is not parsed.stage:
        return False
    if parsed.urgency is not None and item.urgency is not parsed.urgency:
        return False
    if parsed.kind is not None and item.kind is not parsed.kind:
        return False
    if parsed.intake is not None and item.intake is not parsed.intake:
        return False
    if parsed.party_id is not None and item.party_id != parsed.party_id:
        return False
    if parsed.assigned_operator_id is not None:
        wanted = parsed.assigned_operator_id or None  # '' means "unassigned"
        if item.assigned_operator_id != wanted:
            return False
    if parsed.text:
        # The **id** is in the haystack deliberately. A planner holding an id from an
        # earlier turn types it into the only free-text field the tool has; when that
        # searched prose alone, an id search matched nothing, the lookup reported "no
        # records match", and the planner re-ran the identical call rather than acting
        # on a record that plainly exists. Observed live.
        haystack = f"{item.id}\n{item.title}\n{item.detail}".casefold()
        if parsed.text.casefold() not in haystack:
            return False
    return True


def _describe(item: WorkItem, *, now: datetime) -> str:
    """Render one shortlist row: enough to choose between records, and no more."""
    age = _age_hours(item, now=now)
    assignee = item.assigned_operator_id or "unassigned"
    over = " OVER-TARGET" if age > item.target_cycle_hours else ""
    return (
        f"{item.id} | {item.stage.value} | {item.urgency.value} | {item.kind.value} | "
        f"{assignee} | age {age:.0f}h of {item.target_cycle_hours:.0f}h target{over} | "
        f"{item.title}"
    )


async def find_work_items(args: dict, ctx: ToolContext) -> ToolActionResult:
    """Look up work items matching a filter (**read-only**, LOW risk).

    **Why a read tool is not optional.** Every write tool below takes an ``item_id``,
    and the persona prompt correctly forbids inventing one — so with no read tool there
    is no legitimate way for the planner to *obtain* an id, and the human-approval gate
    on the HIGH-risk write becomes unreachable from a plain question. This tool lowers
    no bar: it neither mutates nor gates, and the write it enables is still HIGH and
    still stops at the human.

    **Read-only, and structurally so.** It calls exactly one store method —
    ``list_items`` — and never ``put_item``, so no code path here can change a record.
    It therefore has no ``previous_state`` and no ``inverse``. It still writes an audit
    row (who looked, with what filter, which ids came back), because "read-only" is a
    statement about state, not about accountability.

    **Bounded.** Filters are AND-ed, results ordered by creation time, and the row count
    truncated to :data:`FIND_ITEMS_MAX_LIMIT` at the latest possible point.

    Args:
        args: Raw arguments, validated against :class:`FindItemsArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult` whose ``summary`` is one line per match. ``ok`` is
        False **only** when nothing matched — an empty shortlist is a round that did not
        advance the goal, and reporting it as such is what lets the graph's bounded
        self-repair loop widen the filter rather than answer "none" from a filter that
        was simply too narrow. ``changed`` is always False.
    """
    parsed = FindItemsArgs.model_validate(args)
    lister = getattr(ctx.store, "list_items", None)
    if lister is None:
        # A store that predates the read side of the protocol. Loud in the summary and
        # never a silent empty result, which would read as "no such records".
        return ToolActionResult(
            ok=False,
            changed=False,
            summary="This record store cannot enumerate work items (no list_items).",
        )

    now = datetime.now(tz=UTC)
    matched = [i for i in lister() if _matches(i, parsed)]
    matched.sort(
        key=lambda i: (
            i.created_at.replace(tzinfo=UTC)
            if i.created_at.tzinfo is None
            else i.created_at
        ),
        reverse=not parsed.oldest_first,
    )
    total = len(matched)
    page = matched[: min(parsed.limit, FIND_ITEMS_MAX_LIMIT)]

    await _emit_audit(
        ctx,
        "find_work_items",
        {
            "filter": parsed.model_dump(exclude_none=True, mode="json"),
            "matched": total,
            "returned": [i.id for i in page],
        },
    )

    if not page:
        return ToolActionResult(
            ok=False,
            changed=False,
            summary="No work items match that filter. Try a broader one.",
        )

    header = (
        f"{len(page)} of {total} matching work item(s) "
        "— id | stage | urgency | kind | assignee | age/target | title:"
    )
    return ToolActionResult(
        ok=True,
        changed=False,
        summary="\n".join([header, *(_describe(i, now=now) for i in page)]),
    )


async def advance_item_stage(args: dict, ctx: ToolContext) -> ToolActionResult:
    """Move a record to a new lifecycle stage (idempotent, reversible, **HIGH risk**).

    TODO(domain): this is the worked example of a gated write. Whatever replaces it
    should be the action a human genuinely wants to confirm — externally visible,
    consequential, hard to unpick socially even when it is technically reversible.

    Args:
        args: Raw arguments, validated against :class:`AdvanceStageArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult`; its inverse sets the stage back.
    """
    parsed = AdvanceStageArgs.model_validate(args)
    item = ctx.store.get_item(parsed.item_id)
    if item is None:
        return ToolActionResult(
            ok=False, changed=False, summary=f"No work item {parsed.item_id!r}."
        )

    previous = item.stage
    changed = previous != parsed.stage
    if changed:
        ctx.store.put_item(
            item.model_copy(
                update={"stage": parsed.stage, "updated_at": datetime.now(tz=UTC)}
            )
        )
    await _emit_audit(
        ctx,
        "advance_item_stage",
        {"item_id": parsed.item_id, "from": previous.value, "to": parsed.stage.value},
    )
    return ToolActionResult(
        ok=True,
        changed=changed,
        summary=(
            f"Stage {previous.value} → {parsed.stage.value}"
            if changed
            else f"Already {parsed.stage.value}"
        ),
        previous_state={"stage": previous.value},
        inverse=InverseAction(
            tool="advance_item_stage",
            args={"item_id": parsed.item_id, "stage": previous.value},
        ),
    )


async def assign_work_item(args: dict, ctx: ToolContext) -> ToolActionResult:
    """Assign (or unassign) a record to an operator (idempotent, reversible).

    Args:
        args: Raw arguments, validated against :class:`AssignArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult`; its inverse restores the prior assignee.
    """
    parsed = AssignArgs.model_validate(args)
    item = ctx.store.get_item(parsed.item_id)
    if item is None:
        return ToolActionResult(
            ok=False, changed=False, summary=f"No work item {parsed.item_id!r}."
        )

    previous = item.assigned_operator_id
    changed = previous != parsed.operator_id
    if changed:
        ctx.store.put_item(
            item.model_copy(
                update={
                    "assigned_operator_id": parsed.operator_id,
                    "updated_at": datetime.now(tz=UTC),
                }
            )
        )
    await _emit_audit(
        ctx,
        "assign_work_item",
        {"item_id": parsed.item_id, "from": previous, "to": parsed.operator_id},
    )
    return ToolActionResult(
        ok=True,
        changed=changed,
        summary=f"Assigned to {parsed.operator_id}" if changed else "Assignee unchanged",
        previous_state={"assigned_operator_id": previous},
        inverse=InverseAction(
            tool="assign_work_item",
            args={"item_id": parsed.item_id, "operator_id": previous},
        ),
    )


def _note_id(item_id: str, author: str, body: str) -> str:
    """Derive a deterministic note id from its content (the dedupe key)."""
    digest = hashlib.sha1(f"{item_id}|{author}|{body}".encode()).hexdigest()  # noqa: S324
    return f"note-{digest[:12]}"


async def add_item_note(args: dict, ctx: ToolContext) -> ToolActionResult:
    """Append (or retract) a note on a record's timeline (idempotent, self-reversible).

    The note id is derived from its content, so adding the same note twice is a no-op.
    Passing ``retract=True`` removes that note — which is exactly the inverse of the
    add, making the tool its own undo.

    Args:
        args: Raw arguments, validated against :class:`AddNoteArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult`; its inverse toggles the ``retract`` flag.
    """
    parsed = AddNoteArgs.model_validate(args)
    item = ctx.store.get_item(parsed.item_id)
    if item is None:
        return ToolActionResult(
            ok=False, changed=False, summary=f"No work item {parsed.item_id!r}."
        )

    author = parsed.author or ctx.actor or "system"
    note_id = _note_id(parsed.item_id, author, parsed.body)
    present = any(n.id == note_id for n in item.notes)

    if parsed.retract:
        changed = present
        notes = [n for n in item.notes if n.id != note_id]
        summary = "Note retracted" if changed else "Note already absent"
    else:
        changed = not present
        notes = list(item.notes)
        if changed:
            notes.append(
                ItemNote(
                    id=note_id,
                    author=author,
                    body=parsed.body,
                    created_at=datetime.now(tz=UTC),
                )
            )
        summary = "Note added" if changed else "Note already present"

    if changed:
        ctx.store.put_item(
            item.model_copy(update={"notes": notes, "updated_at": datetime.now(tz=UTC)})
        )
    await _emit_audit(
        ctx,
        "add_item_note",
        {"item_id": parsed.item_id, "note_id": note_id, "retract": parsed.retract},
    )
    return ToolActionResult(
        ok=True,
        changed=changed,
        summary=summary,
        previous_state={"note_present": present},
        inverse=InverseAction(
            tool="add_item_note",
            args={
                "item_id": parsed.item_id,
                "body": parsed.body,
                "author": author,
                "retract": not parsed.retract,
            },
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry + per-persona allowlist
# ─────────────────────────────────────────────────────────────────────────────


class ToolHandler(Protocol):
    """The async signature every action tool implements."""

    async def __call__(self, args: dict, ctx: ToolContext) -> ToolActionResult:
        """Run the tool against ``args`` in ``ctx``."""
        ...


@dataclass(frozen=True)
class ToolSpec:
    """A registered action tool: its metadata, schema and handler."""

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    risk: RiskLevel
    read_only: bool = False
    """Whether a call cannot modify its environment at all (MCP ``readOnlyHint``).

    Asserted, never inferred from the risk tier: LOW risk means "cheap to get wrong",
    which is not the same claim as "changes nothing". A note-append is LOW and writes;
    a lookup is LOW and does not. The default is the cautious reading, so a tool
    registered without thinking about it is advertised as a writer.
    """
    destructive: bool = False
    """Whether a call overwrites state a reader would miss (MCP ``destructiveHint``)."""
    idempotent: bool = False
    """Whether repeating the identical call converges (MCP ``idempotentHint``)."""

    def definition(self) -> dict:
        """Return the MCP/OpenAI ``function`` tool definition for the LLM.

        Returns:
            ``{"type": "function", "function": {...}}``, whose parameter schema is
            derived from :attr:`args_model` — so the schema the model is shown and the
            schema the handler validates against can never drift apart.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


_ID_RULE = (
    "The item_id must be an id that find_work_items returned — call it first and use "
    "one of its results. Never invent an id or assemble one from words in the question; "
    "an identifier that was not returned by a lookup does not name a real record and "
    "this call will fail against it. If no lookup has been run, run one instead of "
    "guessing."
)
"""Where an id may come from, appended to **every** tool that takes one.

The rule was originally written once, in the lookup tool's own description — which is
precisely where a model reaching for a *write* tool never reads it. Asked a read-only
question, a four-lane fan-out had all four lanes call the note tool with an id
assembled out of the words in the question. Four writes attempted, four "no such
record", nothing learned between them. A constraint that appears only on the tool a
model chose *not* to call is not a constraint.
"""

TOOL_REGISTRY: dict[str, ToolSpec] = {
    "find_work_items": ToolSpec(
        name="find_work_items",
        description=(
            "Look up work items by stage, urgency, kind, intake path, party, assignee "
            "or free text, newest- or oldest-first. Read-only: it changes nothing. "
            "Returns a short list of real ids with the fields needed to choose between "
            "them. Use this to obtain an id — never guess or invent one."
        ),
        args_model=FindItemsArgs,
        handler=find_work_items,
        # LOW, and it is the one tool here for which that tier needs no argument: it
        # reads. It must stay below every deployment's gate_min_risk — a lookup that
        # paused for a human would put the approval dialog in front of the step that
        # merely tells the planner which record the human meant.
        risk=RiskLevel.LOW,
        read_only=True,
        idempotent=True,
    ),
    "advance_item_stage": ToolSpec(
        name="advance_item_stage",
        description="Move a work item to a new lifecycle stage. " + _ID_RULE,
        args_model=AdvanceStageArgs,
        handler=advance_item_stage,
        # HIGH: a consequential, externally-visible state change — it routes to the
        # human-approval gate. This tier IS the gate; there is no second signal.
        risk=RiskLevel.HIGH,
        destructive=True,
        idempotent=True,
    ),
    "assign_work_item": ToolSpec(
        name="assign_work_item",
        description="Assign or unassign a work item to an operator. " + _ID_RULE,
        args_model=AssignArgs,
        handler=assign_work_item,
        risk=RiskLevel.MEDIUM,
        idempotent=True,
    ),
    "add_item_note": ToolSpec(
        name="add_item_note",
        description="Append (or retract) a note on a work item's timeline. " + _ID_RULE,
        args_model=AddNoteArgs,
        handler=add_item_note,
        risk=RiskLevel.LOW,
        # Re-running appends a second note, so explicitly NOT idempotent.
    ),
}
"""Name → :class:`ToolSpec` for every action tool the domain exposes.

TODO(domain): replace all four. The registry is the whole vocabulary — a name absent
from it is treated as HIGH risk by the platform, so a hallucinated tool can never slip
under the gate.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SLOT — where the ML spine reaches the agent loop
#
# `aegis_ml.serve.tools` ships ready-made ToolSpecs over the trained model:
#   predict_outcome (LOW, read-only, idempotent) · explain_prediction (LOW) ·
#   whatif_scenario (LOW) · forecast_series (LOW) · check_model_health (LOW)
#
# They are how a prediction becomes something the agent can *ask for* rather than dead
# code: without them `ml_spec.describe_prediction` has no caller and the ML story never
# reaches a turn. Registering them here — not in the graph — is deliberate: it needs no
# core edit, and it keeps the platform's rule intact. **ML informs; it never gates.**
# The human gate fires on a tool's risk tier, and every ML tool is LOW and read-only,
# so a prediction can never approve or block an action by itself.
#
# `ml_tool_specs` builds them out of THIS module's own `ToolSpec` class — it inspects
# your constructor and passes only the parameters it declares — so the domain keeps its
# native types and `aegis_ml` never imports anything from `app`.
#
# TODO(domain): uncomment once `aegis-ml[serve]` is installed in the backend venv and a
# model has been promoted. THREE edits, and skipping any one of them is silent:
#
#   1. register them — merge into TOOL_REGISTRY at the point of definition:
#
#          from aegis_ml.serve.tools import ML_TOOL_NAMES, ml_tool_specs
#
#          TOOL_REGISTRY: dict[str, ToolSpec] = {
#              ...the domain's own tools above...,
#              **ml_tool_specs(
#                  ToolSpec,
#                  risk_low=RiskLevel.LOW,
#                  result_cls=ToolActionResult,   # keeps the handlers' results native
#                  # problem=PROBLEM,             # optional: an aegis_ml MLProblem, so
#                  #                              # summaries carry the target's unit and
#                  #                              # description instead of bare floats
#              ),
#          }
#
#   2. grant them in ALLOWLIST below — a registered tool no persona may call is
#      invisible to every turn, and nothing warns:
#          "line_supervisor": frozenset(TOOL_REGISTRY)      # already covers them
#          "party_contact":   frozenset({"add_item_note", *ML_TOOL_NAMES})
#
#   3. grant them in the relevant `sub_agent_roster()` `tool_allowlist` in piece 8 —
#      the core INTERSECTS that set with the registry, so a lane that does not name
#      them simply runs without them.
# ══════════════════════════════════════════════════════════════════════════════


ALLOWLIST: dict[str, frozenset[str]] = {
    # The broad persona may perform every action — and, since the registry carries a
    # read tool, enumerate records. Its data scope is already "everything", so listing
    # the lookup grants it nothing its scope did not already say it has.
    "line_supervisor": frozenset(TOOL_REGISTRY),
    # The narrow persona may only annotate records it already owns.
    #
    # **The lookup is deliberately NOT here.** That persona's scope is OWN on
    # ``party_id``, and the narrowing is applied by the retrieval/data layers from the
    # authenticated subject — a value :class:`ToolContext` does not carry, so this
    # module cannot enforce it. A tool taking ``party_id`` as a *filter* would let one
    # party enumerate another's records by passing someone else's id. Listing it here
    # is not a roster line, it is a scope change.
    #
    # TODO(domain): re-derive this per persona from the Brief. Every key must exist in
    # piece 5's PERSONAS and every value must be a key of TOOL_REGISTRY — conformance
    # check #6 fails on either, naming the offender.
    "party_contact": frozenset({"add_item_note"}),
}
"""Persona id → the set of tool names that persona is authorised to call."""


def is_allowed(persona_id: str, tool_name: str) -> bool:
    """Return whether ``persona_id`` may call ``tool_name``."""
    return tool_name in ALLOWLIST.get(persona_id, frozenset())


def tools_for(persona_id: str) -> list[ToolSpec]:
    """Return the :class:`ToolSpec` list a persona is allowed to use."""
    return [TOOL_REGISTRY[name] for name in sorted(ALLOWLIST.get(persona_id, frozenset()))]


def tool_definitions_for(persona_id: str) -> list[dict]:
    """Return the LLM ``tools=`` payload for a persona (allowlist-filtered)."""
    return [spec.definition() for spec in tools_for(persona_id)]


async def run_tool(
    persona_id: str, tool_name: str, args: dict, ctx: ToolContext
) -> ToolActionResult:
    """Authorise and execute one tool for a persona.

    The allowlist is checked **before** any side effect, so an unauthorised call can
    never mutate state or emit an audit record. Keep that ordering: it is the only
    thing standing between a hallucinated tool name and a real write.

    Args:
        persona_id: The calling persona's id.
        tool_name: The tool to run.
        args: Raw tool arguments.
        ctx: The execution context.

    Returns:
        The tool's :class:`ToolActionResult`.

    Raises:
        UnknownToolError: If ``tool_name`` is not registered.
        ToolNotAllowedError: If ``persona_id`` is not allowed to call it.
    """
    if tool_name not in TOOL_REGISTRY:
        raise UnknownToolError(tool_name)
    if not is_allowed(persona_id, tool_name):
        raise ToolNotAllowedError(f"Persona {persona_id!r} may not call tool {tool_name!r}.")
    return await TOOL_REGISTRY[tool_name].handler(args, ctx)


__all__ = [
    "ALLOWLIST",
    "FIND_ITEMS_DEFAULT_LIMIT",
    "FIND_ITEMS_MAX_LIMIT",
    "TOOL_REGISTRY",
    "AddNoteArgs",
    "AdvanceStageArgs",
    "AssignArgs",
    "AuditFn",
    "FindItemsArgs",
    "InMemoryRecordStore",
    "InverseAction",
    "RecordStore",
    "ToolActionResult",
    "ToolContext",
    "ToolHandler",
    "ToolNotAllowedError",
    "ToolSpec",
    "UnknownToolError",
    "add_item_note",
    "advance_item_stage",
    "assign_work_item",
    "find_work_items",
    "is_allowed",
    "run_tool",
    "tool_definitions_for",
    "tools_for",
]

"""Piece 4 of 10 — what the agent can *do*, at what risk, and who may ask for it.

WHAT THIS FILE IS
    One ``async def handler(args, ctx) -> ToolActionResult`` per real cold-chain action,
    each:

      * **typed** — arguments validated by a pydantic model, and the MCP/OpenAI ``function``
        schema derived from that same model, so validation and the ``tools=`` payload can
        never disagree;
      * **idempotent** — re-running with identical arguments converges; the second run
        reports ``changed=False``;
      * **reversible** — the result carries an :class:`InverseAction` (a tool name + args)
        that undoes it, which is what makes a rejected proposal cleanly recoverable;
      * **audited** — every invocation appends an audit row through the injected sink;
      * **registered** in :data:`TOOL_REGISTRY` with an honest ``risk`` tier;
      * **allowlisted** in :data:`ALLOWLIST`, persona → tool names.

    Plus the five ML tools, merged in from :mod:`aegis_ml.serve.tools`. That merge is the
    whole ML integration: prediction reaches the agent through the **tool registry**, not
    through a graph edit.

THE CONTRACT (aegis.adapter.ToolsModule / ToolSpecLike) — these names must survive
    TOOL_REGISTRY, ALLOWLIST, is_allowed(), tools_for(), tool_definitions_for(), run_tool()
    ToolSpec.name / .description / .risk / .definition()

    Host-bound beyond the Protocol: ``ToolContext``, ``ToolActionResult``,
    ``InMemoryRecordStore``, ``RecordStore``, ``AuditFn``, ``UnknownToolError``,
    ``ToolNotAllowedError``. The MCP server and client import those directly from this
    module, so they keep their names.

THE TRAP
    **``risk`` is the ONLY input to the human approval gate.** Not model confidence, not the
    ML prediction, not the persona. A tool at or above the deployment's ``gate_min_risk``
    (platform default ``HIGH``) pauses for a human; one below it just runs. That is why
    :func:`quarantine_shipment` is HIGH: quarantining a consignment strands product a clinic
    is expecting, and it is exactly the decision a human wants to confirm.

    An **unregistered** tool name resolves to ``HIGH``, so a forgotten registration fails
    safe — it demands approval rather than running unguarded. Do not lean on that: a missing
    registration then looks like an over-cautious gate rather than a bug.

    ``read_only``, ``destructive`` and ``idempotent`` are asserted per tool, never derived
    from risk. Risk does not imply idempotency, and idempotency does not imply safety:
    :func:`add_shipment_note` is LOW **and** writes, while :func:`quarantine_shipment` is
    HIGH **and** converges on repetition.

    ``ALLOWLIST`` is checked **before any side effect**, in :func:`run_tool`. Keep that
    ordering: it is the only thing standing between a hallucinated tool name and a real
    write.

WHERE THE ML GOES IN — and why it needs no core change
    :func:`aegis_ml.serve.tools.ml_tool_specs` builds its five specs out of **this module's
    own** :class:`ToolSpec` class: it inspects the constructor and passes only the parameters
    it declares, so the domain keeps its native types and ``aegis_ml`` imports nothing from
    the adapter. All five are LOW, read-only, non-destructive and idempotent, asserted
    per tool, which preserves the platform's rule exactly:

        **ML informs; it never gates.**

    Routing a *prediction* to a human approval dialog would put the gate in front of the
    step that merely tells the planner what the model thinks, while the write it was
    supposed to guard sails past. The gate belongs on the action, and a prediction is not
    one.

    Wiring them is three edits, and skipping any one of them is silent: register them here,
    grant them in :data:`ALLOWLIST`, and grant them in the relevant
    :func:`~reference.adapter.roster.sub_agent_roster` ``tool_allowlist``. All three are
    done — see the ``ML_TOOL_NAMES`` splices below and in piece 8.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from aegis_ml.serve.tools import ML_TOOL_NAMES, ml_tool_specs
from pydantic import BaseModel, Field

from reference.adapter.ml_spec import PROBLEM
from reference.adapter.schema import (
    PackagingType,
    ProductClass,
    RouteClass,
    Shipment,
    ShipmentNote,
    ShipmentStage,
)

__all__ = [
    "ALLOWLIST",
    "FIND_SHIPMENTS_DEFAULT_LIMIT",
    "FIND_SHIPMENTS_MAX_LIMIT",
    "TOOL_REGISTRY",
    "AddNoteArgs",
    "AuditFn",
    "FindShipmentsArgs",
    "InMemoryRecordStore",
    "InverseAction",
    "QuarantineArgs",
    "RecordStore",
    "RerouteArgs",
    "RiskLevel",
    "ToolActionResult",
    "ToolContext",
    "ToolHandler",
    "ToolNotAllowedError",
    "ToolSpec",
    "UnknownToolError",
    "add_shipment_note",
    "find_shipments",
    "is_allowed",
    "quarantine_shipment",
    "reroute_shipment",
    "run_tool",
    "tool_definitions_for",
    "tools_for",
]


class _StandaloneRiskLevel(StrEnum):
    """Risk tiers, for when no Aegis checkout is on the path.

    Member names and values match ``aegis.core.types.RiskLevel`` exactly. Both are
    ``StrEnum``, and a ``StrEnum`` member hashes and compares as its string value, so a
    platform comparing its own member against one of these matches — a tool tiered here is
    tiered identically when the real enum is present.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _resolve_risk_level() -> type[StrEnum]:
    """Return the platform's ``RiskLevel`` enum, or the standalone stand-in.

    The adapter must import with no Aegis checkout present — that is what lets the reference
    domain be run, tested and audited on its own — while still tiering its tools with the
    platform's own enum whenever the platform is there. This is not a swallowed import: the
    failure has exactly one handled cause (no Aegis on the path) and exactly one
    consequence, a value-identical stand-in, which is stated in
    :class:`_StandaloneRiskLevel`'s docstring.

    Returns:
        ``aegis.core.types.RiskLevel`` when importable, else :class:`_StandaloneRiskLevel`.
    """
    try:
        from aegis.core.types import RiskLevel as PlatformRiskLevel
    except ImportError:
        return _StandaloneRiskLevel
    return PlatformRiskLevel


RiskLevel: type[StrEnum] = _resolve_risk_level()
"""The risk enum this registry tiers against — the platform's when it is importable."""


# ─────────────────────────────────────────────────────────────────────────────
# Injected data-layer contracts (structural — no hard import of a host data layer)
#
# The store and the audit sink are injected through ToolContext so every tool is
# unit-testable with no database and no network.
# ─────────────────────────────────────────────────────────────────────────────


class RecordStore(Protocol):
    """The minimal shipment access the tools need (a data-layer view).

    ``list_shipments`` is not optional. ``get_shipment`` can only answer a question that
    already knows the answer's id, so a store offering nothing else makes an id-taking tool
    the only thing a planner can reach — and a planner that is (correctly) forbidden from
    inventing ids can therefore reach nothing.
    """

    def get_shipment(self, shipment_id: str) -> Shipment | None:
        """Return the shipment with ``shipment_id``, or None if absent."""
        ...

    def list_shipments(self) -> list[Shipment]:
        """Return every shipment this store holds, in a stable order."""
        ...

    def put_shipment(self, shipment: Shipment) -> None:
        """Persist ``shipment`` (insert or replace by id)."""
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
        payload: dict[str, Any],
        approved_by: str | None = None,
    ) -> None:
        """Append one immutable audit record."""
        ...


class InMemoryRecordStore:
    """A dict-backed :class:`RecordStore` for demos, seeding and tests.

    Keep this class name: the host's agent deps and MCP server both import it from
    ``reference.adapter`` by name to build the process-wide demo store.
    """

    def __init__(self, shipments: list[Shipment] | None = None) -> None:
        """Initialise the store, optionally seeded with ``shipments``.

        Args:
            shipments: Records to seed, keyed by id in insertion order.
        """
        self._shipments: dict[str, Shipment] = {s.id: s for s in (shipments or [])}

    @classmethod
    def from_dataset(cls, dataset: object) -> InMemoryRecordStore:
        """Build a store from a :class:`~reference.adapter.schema.SyntheticDataset`.

        Args:
            dataset: An object exposing a ``shipments`` iterable.

        Returns:
            A populated in-memory store.
        """
        return cls(list(getattr(dataset, "shipments", [])))

    def get_shipment(self, shipment_id: str) -> Shipment | None:
        """Return the shipment with ``shipment_id``, or None if absent."""
        return self._shipments.get(shipment_id)

    def list_shipments(self) -> list[Shipment]:
        """Return all stored shipments (public, ordered by insertion)."""
        return list(self._shipments.values())

    def put_shipment(self, shipment: Shipment) -> None:
        """Persist ``shipment`` (insert or replace by id)."""
        self._shipments[shipment.id] = shipment


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
    """A ``(tool, args)`` pair that reverses a completed action."""

    tool: str = Field(description="Tool name that undoes the action.")
    args: dict[str, Any] = Field(description="Arguments to pass to that tool.")


class ToolActionResult(BaseModel):
    """The typed outcome of running an action tool.

    ``ok`` and ``summary`` are the two fields the agent loop reads structurally, so both
    stay. ``summary`` is pasted verbatim into the next planning prompt — write it for a
    model *and* for a human watching the trace.
    """

    ok: bool = Field(description="Whether the action succeeded.")
    changed: bool = Field(description="Whether state actually changed (idempotency).")
    summary: str = Field(description="Human-readable, demoable result summary.")
    previous_state: dict[str, Any] = Field(
        default_factory=dict, description="Prior values, for audit / rollback."
    )
    inverse: InverseAction | None = Field(
        default=None, description="Action that undoes this one, if reversible."
    )


@dataclass
class ToolContext:
    """Everything a tool needs to run, injected by the caller.

    Attributes:
        store: The shipment store the tool reads and writes.
        actor: Persona/agent id performing the action (for the audit trail).
        model: Model id that proposed the action, if any.
        trace_id: Observability correlation id, if any.
        approved_by: Who approved the action at the human gate, if applicable.
        audit: The audit sink. ``None`` means no sink is wired in and audit rows are
            dropped — which is legitimate in a test or an offline demo and is never
            allowed to block the action itself.
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
# Every field carries a `description`: it is what the model reads to decide what to pass,
# and a field described only by its name is a field the planner fills in from the words of
# the question.
# ─────────────────────────────────────────────────────────────────────────────

FIND_SHIPMENTS_DEFAULT_LIMIT = 10
"""Rows :func:`find_shipments` returns when the caller does not ask for a number."""

FIND_SHIPMENTS_MAX_LIMIT = 25
"""Hard ceiling on :func:`find_shipments`, enforced in the args model **and** the body.

A lookup tool exists to hand the planner a *shortlist* to choose from, not a table dump.
Every row is pasted verbatim into the next planning prompt, so an unbounded limit is
simultaneously a token bill, a context-window risk and a larger blast radius for the
tool-result injection rail to screen. The ceiling lives in the pydantic model, so an
over-large ``limit`` is a validation error the planner sees and can correct, and is
re-applied in the body, so a direct caller that bypassed the model still cannot exceed it.
"""


class FindShipmentsArgs(BaseModel):
    """Arguments for :func:`find_shipments` — every filter optional, all AND-ed."""

    stage: ShipmentStage | None = Field(
        default=None, description="Only shipments in this lifecycle stage."
    )
    route_class: RouteClass | None = Field(
        default=None, description="Only shipments on this journey shape."
    )
    packaging_type: PackagingType | None = Field(
        default=None, description="Only shipments under this thermal system."
    )
    product_class: ProductClass | None = Field(
        default=None, description="Only shipments carrying this product class."
    )
    carrier_id: str | None = Field(
        default=None, description="Only shipments booked with this carrier id."
    )
    shipper_id: str | None = Field(
        default=None, description="Only shipments owned by this shipper account id."
    )
    excursions_only: bool = Field(
        default=False,
        description="If true, return only shipments whose logger record shows a "
        "temperature excursion.",
    )
    text: str | None = Field(
        default=None,
        description=(
            "Case-insensitive substring to match in the shipment id, consignment "
            "reference, summary or detail. Pass a known id here to confirm that shipment "
            "exists."
        ),
    )
    oldest_first: bool = Field(
        default=True,
        description="Order by dispatch time: oldest first (default), else newest first.",
    )
    limit: int = Field(
        default=FIND_SHIPMENTS_DEFAULT_LIMIT,
        ge=1,
        le=FIND_SHIPMENTS_MAX_LIMIT,
        description=f"Maximum rows to return (1–{FIND_SHIPMENTS_MAX_LIMIT}).",
    )


class AddNoteArgs(BaseModel):
    """Arguments for :func:`add_shipment_note`."""

    shipment_id: str = Field(
        description=(
            "Id of the shipment to annotate, as returned by find_shipments. Not a "
            "consignment reference, a document name or a phrase from the question."
        )
    )
    body: str = Field(min_length=1, description="Note text to append to the timeline.")
    author: str | None = Field(default=None, description="Who wrote the note.")
    retract: bool = Field(
        default=False, description="If true, remove the matching note (the inverse)."
    )


class RerouteArgs(BaseModel):
    """Arguments for :func:`reroute_shipment`."""

    shipment_id: str = Field(description="Id of the shipment to reroute.")
    route_class: RouteClass | None = Field(
        default=None,
        description="New journey shape, or null to leave it unchanged. Moving to 'direct' "
        "removes transfers and is the usual intervention on a lane at risk.",
    )
    carrier_id: str | None = Field(
        default=None,
        description="Id of the carrier to move the consignment to, or null to keep the "
        "current one. Must be a carrier id, not a carrier name.",
    )
    reason: str = Field(
        min_length=1,
        description="Why the lane is being changed. Written to the audit record and to the "
        "shipment timeline, so write it for whoever reads the file six months from now.",
    )


class QuarantineArgs(BaseModel):
    """Arguments for :func:`quarantine_shipment`."""

    shipment_id: str = Field(
        description="Id of the shipment to quarantine, as returned by find_shipments."
    )
    reason: str = Field(
        min_length=1,
        description="Why the consignment is being held. Required: a quarantine with no "
        "recorded reason cannot be reviewed and cannot be lifted by anyone else.",
    )
    release: bool = Field(
        default=False,
        description="If true, lift an existing quarantine instead of applying one — this "
        "is the inverse action.",
    )
    restore_stage: ShipmentStage | None = Field(
        default=None,
        description="Stage to return the shipment to when releasing. Supplied by the "
        "inverse action so a release restores exactly what the quarantine displaced.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _emit_audit(ctx: ToolContext, action: str, payload: dict[str, Any]) -> None:
    """Record one audit entry, if a sink was injected.

    Args:
        ctx: The execution context carrying the sink.
        action: The tool name being recorded.
        payload: The action's own structured detail.
    """
    if ctx.audit is None:
        return
    await ctx.audit(
        action=action,
        actor=ctx.actor,
        model=ctx.model,
        trace_id=ctx.trace_id,
        payload=payload,
        approved_by=ctx.approved_by,
    )


def _aware(moment: datetime) -> datetime:
    """Return ``moment`` as a tz-aware UTC instant.

    Timestamps are tz-naive in generated data and tz-aware in host fixtures, so a naive
    value is read as UTC rather than being allowed to raise mid-lookup.

    Args:
        moment: The instant to normalise.

    Returns:
        The same instant, guaranteed tz-aware.
    """
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


def _age_hours(shipment: Shipment, *, now: datetime) -> float:
    """Return how many hours ago ``shipment`` was dispatched (0.0 if in the future).

    Args:
        shipment: The record to age.
        now: The reference instant.

    Returns:
        Elapsed hours since dispatch, never negative.
    """
    return max(0.0, (now - _aware(shipment.dispatched_at)).total_seconds() / 3600.0)


def _matches(shipment: Shipment, parsed: FindShipmentsArgs) -> bool:
    """Return whether ``shipment`` satisfies every filter set on ``parsed``.

    Args:
        shipment: The candidate record.
        parsed: The validated filter.

    Returns:
        True when every set filter matches.
    """
    if parsed.stage is not None and shipment.stage is not parsed.stage:
        return False
    if parsed.route_class is not None and shipment.route_class is not parsed.route_class:
        return False
    if parsed.packaging_type is not None and shipment.packaging_type is not parsed.packaging_type:
        return False
    if parsed.product_class is not None and shipment.product_class is not parsed.product_class:
        return False
    if parsed.carrier_id is not None and shipment.carrier_id != parsed.carrier_id:
        return False
    if parsed.shipper_id is not None and shipment.shipper_id != parsed.shipper_id:
        return False
    if parsed.excursions_only and shipment.excursion_flag is None:
        return False
    if parsed.excursions_only and shipment.excursion_flag is not None:
        if shipment.excursion_flag.value != "excursion":
            return False
    if parsed.text:
        # The **id** is in the haystack deliberately. A planner holding an id from an
        # earlier turn types it into the only free-text field the tool has; when that
        # searched prose alone, an id search matched nothing, the lookup reported "no
        # shipments match", and the planner re-ran the identical call rather than acting on
        # a record that plainly exists.
        haystack = (
            f"{shipment.id}\n{shipment.reference}\n{shipment.summary}\n{shipment.detail}"
        ).casefold()
        if parsed.text.casefold() not in haystack:
            return False
    return True


def _describe(shipment: Shipment, *, now: datetime) -> str:
    """Render one shortlist row: enough to choose between shipments, and no more.

    Args:
        shipment: The record to render.
        now: The reference instant for the age column.

    Returns:
        A single pipe-delimited line.
    """
    age = _age_hours(shipment, now=now)
    overdue = " OVERDUE" if age > shipment.transit_hours and shipment.delivered_at is None else ""
    risk = (
        f"{shipment.spoilage_risk_pct:.1f}%"
        if shipment.spoilage_risk_pct is not None
        else "unassessed"
    )
    excursion = shipment.excursion_flag.value if shipment.excursion_flag else "unknown"
    return (
        f"{shipment.id} | {shipment.reference} | {shipment.stage.value} | "
        f"{shipment.route_class.value} | {shipment.packaging_type.value} | "
        f"{shipment.product_class.value} | age {age:.0f}h of {shipment.transit_hours:.0f}h "
        f"planned{overdue} | spoilage {risk} | excursion {excursion} | {shipment.summary}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


async def find_shipments(args: dict[str, Any], ctx: ToolContext) -> ToolActionResult:
    """Look up shipments matching a filter (**read-only**, LOW risk).

    **Why a read tool is not optional.** Every write tool below takes a ``shipment_id``, and
    the persona prompt correctly forbids inventing one — so with no read tool there is no
    legitimate way for the planner to *obtain* an id, and the human-approval gate on the
    HIGH-risk quarantine becomes unreachable from a plain question. This tool lowers no bar:
    it neither mutates nor gates, and the write it enables is still HIGH and still stops at
    the human.

    **Read-only, and structurally so.** It calls exactly one store method —
    ``list_shipments`` — and never ``put_shipment``, so no code path here can change a
    record. It therefore has no ``previous_state`` and no ``inverse``. It still writes an
    audit row (who looked, with what filter, which ids came back), because "read-only" is a
    statement about state, not about accountability.

    Args:
        args: Raw arguments, validated against :class:`FindShipmentsArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult` whose ``summary`` is one line per match. ``ok`` is False
        **only** when nothing matched — an empty shortlist is a round that did not advance
        the goal, and reporting it as such is what lets the graph's bounded self-repair loop
        widen the filter rather than answer "none" from a filter that was simply too narrow.
        ``changed`` is always False.
    """
    parsed = FindShipmentsArgs.model_validate(args)
    lister = getattr(ctx.store, "list_shipments", None)
    if lister is None:
        # A store that predates the read side of the protocol. Loud in the summary and
        # never a silent empty result, which would read as "no such shipments".
        return ToolActionResult(
            ok=False,
            changed=False,
            summary="This record store cannot enumerate shipments (no list_shipments).",
        )

    now = datetime.now(tz=UTC)
    matched = [s for s in lister() if _matches(s, parsed)]
    matched.sort(key=lambda s: _aware(s.dispatched_at), reverse=not parsed.oldest_first)
    total = len(matched)
    page = matched[: min(parsed.limit, FIND_SHIPMENTS_MAX_LIMIT)]

    await _emit_audit(
        ctx,
        "find_shipments",
        {
            "filter": parsed.model_dump(exclude_none=True, mode="json"),
            "matched": total,
            "returned": [s.id for s in page],
        },
    )

    if not page:
        return ToolActionResult(
            ok=False,
            changed=False,
            summary="No shipments match that filter. Try a broader one.",
        )

    header = (
        f"{len(page)} of {total} matching shipment(s) — id | reference | stage | route | "
        "packaging | product | age/planned | spoilage | excursion | summary:"
    )
    return ToolActionResult(
        ok=True,
        changed=False,
        summary="\n".join([header, *(_describe(s, now=now) for s in page)]),
    )


def _note_id(shipment_id: str, author: str, body: str) -> str:
    """Derive a deterministic note id from its content (the dedupe key).

    Args:
        shipment_id: The annotated shipment.
        author: Who wrote it.
        body: The note text.

    Returns:
        A stable ``note-<digest>`` id.
    """
    digest = hashlib.sha256(f"{shipment_id}|{author}|{body}".encode()).hexdigest()
    return f"note-{digest[:12]}"


async def add_shipment_note(args: dict[str, Any], ctx: ToolContext) -> ToolActionResult:
    """Append (or retract) a note on a shipment's timeline (LOW risk, self-reversing).

    The note id is derived from its **content**, so adding the same note twice is a no-op
    and the second call honestly reports ``changed=False``. That is why this tool is
    registered ``idempotent=True`` even though it writes: idempotency is a claim about
    convergence, and this converges. Passing ``retract=True`` removes that note — which is
    exactly the inverse of the add, making the tool its own undo.

    Args:
        args: Raw arguments, validated against :class:`AddNoteArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult`; its inverse toggles the ``retract`` flag.
    """
    parsed = AddNoteArgs.model_validate(args)
    shipment = ctx.store.get_shipment(parsed.shipment_id)
    if shipment is None:
        return ToolActionResult(
            ok=False, changed=False, summary=f"No shipment {parsed.shipment_id!r}."
        )

    author = parsed.author or ctx.actor or "system"
    note_id = _note_id(parsed.shipment_id, author, parsed.body)
    present = any(note.id == note_id for note in shipment.notes)

    if parsed.retract:
        changed = present
        notes = [note for note in shipment.notes if note.id != note_id]
        summary = "Note retracted from the shipment timeline." if changed else "Note already absent."
    else:
        changed = not present
        notes = list(shipment.notes)
        if changed:
            notes.append(
                ShipmentNote(
                    id=note_id,
                    author=author,
                    body=parsed.body,
                    created_at=datetime.now(tz=UTC),
                )
            )
        summary = "Note added to the shipment timeline." if changed else "Note already present."

    if changed:
        ctx.store.put_shipment(
            shipment.model_copy(update={"notes": notes, "updated_at": datetime.now(tz=UTC)})
        )
    await _emit_audit(
        ctx,
        "add_shipment_note",
        {"shipment_id": parsed.shipment_id, "note_id": note_id, "retract": parsed.retract},
    )
    return ToolActionResult(
        ok=True,
        changed=changed,
        summary=summary,
        previous_state={"note_present": present},
        inverse=InverseAction(
            tool="add_shipment_note",
            args={
                "shipment_id": parsed.shipment_id,
                "body": parsed.body,
                "author": author,
                "retract": not parsed.retract,
            },
        ),
    )


async def reroute_shipment(args: dict[str, Any], ctx: ToolContext) -> ToolActionResult:
    """Move a shipment to a different journey shape and/or carrier (MEDIUM, idempotent).

    MEDIUM rather than HIGH, and the distinction is worth stating: rerouting changes cost
    and arrival time and is visible to the shipper, but the consignment keeps moving and the
    change is fully reversible by re-running this tool with the previous values — which is
    exactly what the returned inverse does. A quarantine, by contrast, strands product a
    clinic is expecting, which is why that one is HIGH.

    Idempotent: rerouting to the values a shipment already holds changes nothing and says so.

    Args:
        args: Raw arguments, validated against :class:`RerouteArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult`; its inverse restores the prior route and carrier.
    """
    parsed = RerouteArgs.model_validate(args)
    shipment = ctx.store.get_shipment(parsed.shipment_id)
    if shipment is None:
        return ToolActionResult(
            ok=False, changed=False, summary=f"No shipment {parsed.shipment_id!r}."
        )
    if shipment.stage in (ShipmentStage.DELIVERED, ShipmentStage.RELEASED):
        return ToolActionResult(
            ok=False,
            changed=False,
            summary=(
                f"Shipment {shipment.id} is already {shipment.stage.value}; a delivered "
                "consignment cannot be rerouted. Nothing was changed."
            ),
        )

    previous_route = shipment.route_class
    previous_carrier = shipment.carrier_id
    new_route = parsed.route_class or previous_route
    new_carrier = parsed.carrier_id or previous_carrier
    changed = new_route is not previous_route or new_carrier != previous_carrier

    if changed:
        ctx.store.put_shipment(
            shipment.model_copy(
                update={
                    "route_class": new_route,
                    "carrier_id": new_carrier,
                    "updated_at": datetime.now(tz=UTC),
                }
            )
        )
    await _emit_audit(
        ctx,
        "reroute_shipment",
        {
            "shipment_id": parsed.shipment_id,
            "route_from": previous_route.value,
            "route_to": new_route.value,
            "carrier_from": previous_carrier,
            "carrier_to": new_carrier,
            "reason": parsed.reason,
        },
    )
    return ToolActionResult(
        ok=True,
        changed=changed,
        summary=(
            f"Rerouted {shipment.id}: {previous_route.value} → {new_route.value}, "
            f"carrier {previous_carrier} → {new_carrier}. Reason: {parsed.reason}"
            if changed
            else f"Shipment {shipment.id} already runs {new_route.value} on {new_carrier}."
        ),
        previous_state={"route_class": previous_route.value, "carrier_id": previous_carrier},
        inverse=InverseAction(
            tool="reroute_shipment",
            args={
                "shipment_id": parsed.shipment_id,
                "route_class": previous_route.value,
                "carrier_id": previous_carrier,
                "reason": f"Reverting reroute: {parsed.reason}",
            },
        ),
    )


async def quarantine_shipment(args: dict[str, Any], ctx: ToolContext) -> ToolActionResult:
    """Hold a shipment for quality review, or release one (**HIGH risk**, idempotent).

    This is the gated write. Quarantining strands product a clinic or a patient is expecting
    and starts a regulated review that somebody has to close, so it is the action a human
    genuinely wants to confirm — and ``risk=HIGH`` is the *only* thing that routes it to the
    approval gate. There is no second signal.

    Both directions are recorded. A release takes the stage the quarantine displaced, which
    the returned inverse supplies, so lifting a hold puts the consignment back exactly where
    it was rather than into a plausible-looking stage somebody guessed.

    Args:
        args: Raw arguments, validated against :class:`QuarantineArgs`.
        ctx: The execution context (store, actor, audit sink).

    Returns:
        A :class:`ToolActionResult`; its inverse releases (or re-applies) the hold.
    """
    parsed = QuarantineArgs.model_validate(args)
    shipment = ctx.store.get_shipment(parsed.shipment_id)
    if shipment is None:
        return ToolActionResult(
            ok=False, changed=False, summary=f"No shipment {parsed.shipment_id!r}."
        )

    previous = shipment.stage
    held = previous is ShipmentStage.QUARANTINED

    if parsed.release:
        target = parsed.restore_stage or ShipmentStage.HELD_AT_HUB
        changed = held
        summary = (
            f"Quarantine on {shipment.id} lifted; returned to {target.value}. "
            f"Reason: {parsed.reason}"
            if changed
            else f"Shipment {shipment.id} is not under quarantine ({previous.value})."
        )
    else:
        target = ShipmentStage.QUARANTINED
        changed = not held
        summary = (
            f"Shipment {shipment.id} quarantined (was {previous.value}). "
            f"Reason: {parsed.reason}"
            if changed
            else f"Shipment {shipment.id} is already quarantined."
        )

    if changed:
        ctx.store.put_shipment(
            shipment.model_copy(update={"stage": target, "updated_at": datetime.now(tz=UTC)})
        )
    await _emit_audit(
        ctx,
        "quarantine_shipment",
        {
            "shipment_id": parsed.shipment_id,
            "from": previous.value,
            "to": target.value,
            "release": parsed.release,
            "reason": parsed.reason,
        },
    )
    return ToolActionResult(
        ok=True,
        changed=changed,
        summary=summary,
        previous_state={"stage": previous.value},
        inverse=InverseAction(
            tool="quarantine_shipment",
            args={
                "shipment_id": parsed.shipment_id,
                "reason": f"Reverting: {parsed.reason}",
                "release": not parsed.release,
                "restore_stage": previous.value,
            },
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry + per-persona allowlist
# ─────────────────────────────────────────────────────────────────────────────


class ToolHandler(Protocol):
    """The async signature every action tool implements."""

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolActionResult:
        """Run the tool against ``args`` in ``ctx``."""
        ...


@dataclass(frozen=True)
class ToolSpec:
    """A registered action tool: its metadata, schema and handler.

    ``aegis_ml.serve.tools.ml_tool_specs`` constructs instances of *this* class by
    inspecting the signature below, so the ML tools land in the registry as native
    :class:`ToolSpec` objects rather than as a foreign type the host would have to special-
    case. That inspection is why every field here has either a value it can supply
    (``name``, ``description``, ``args_model``, ``handler``, ``risk``) or a default.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    risk: StrEnum
    read_only: bool = False
    """Whether a call cannot modify its environment at all (MCP ``readOnlyHint``).

    Asserted, never inferred from the risk tier: LOW risk means "cheap to get wrong", which
    is not the same claim as "changes nothing". A note append is LOW and writes; a lookup is
    LOW and does not. The default is the cautious reading, so a tool registered without
    thinking about it is advertised as a writer.
    """
    destructive: bool = False
    """Whether a call overwrites state a reader would miss (MCP ``destructiveHint``)."""
    idempotent: bool = False
    """Whether repeating the identical call converges (MCP ``idempotentHint``)."""

    def definition(self) -> dict[str, Any]:
        """Return the MCP/OpenAI ``function`` tool definition for the LLM.

        Returns:
            ``{"type": "function", "function": {...}}``, whose parameter schema is derived
            from :attr:`args_model` — so the schema the model is shown and the schema the
            handler validates against can never drift apart.
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
    " The shipment_id must be an id that find_shipments returned — call it first and use "
    "one of its results. Never invent an id or assemble one from words in the question; an "
    "identifier that was not returned by a lookup does not name a real consignment and this "
    "call will fail against it. If no lookup has been run, run one instead of guessing."
)
"""Where an id may come from, appended to **every** tool that takes one.

The rule was originally written once, in the lookup tool's own description — which is
precisely where a model reaching for a *write* tool never reads it. Asked a read-only
question, a four-lane fan-out had all four lanes call the note tool with an id assembled out
of the words in the question. Four writes attempted, four "no such record", nothing learned
between them. A constraint that appears only on the tool a model chose *not* to call is not
a constraint.
"""

_DOMAIN_TOOLS: dict[str, ToolSpec] = {
    "find_shipments": ToolSpec(
        name="find_shipments",
        description=(
            "Look up temperature-controlled shipments by stage, route class, packaging, "
            "product class, carrier, shipper or free text, newest- or oldest-first, and "
            "optionally only those with a logged temperature excursion. Read-only: it "
            "changes nothing. Returns a short list of real shipment ids with the fields "
            "needed to choose between them. Use this to obtain an id — never guess one."
        ),
        args_model=FindShipmentsArgs,
        handler=find_shipments,
        # LOW, and it is the one tool here for which that tier needs no argument: it reads.
        # It must stay below every deployment's gate_min_risk — a lookup that paused for a
        # human would put the approval dialog in front of the step that merely tells the
        # planner which consignment the human meant.
        risk=RiskLevel.LOW,
        read_only=True,
        idempotent=True,
    ),
    "add_shipment_note": ToolSpec(
        name="add_shipment_note",
        description=(
            "Append (or retract) a note on a shipment's timeline. Notes are deduplicated by "
            "content, so repeating the same note changes nothing." + _ID_RULE
        ),
        args_model=AddNoteArgs,
        handler=add_shipment_note,
        # LOW and a writer. Both facts are stated: a timeline note is cheap to get wrong and
        # trivially retractable, but it is still a write and is advertised as one.
        risk=RiskLevel.LOW,
        idempotent=True,
    ),
    "reroute_shipment": ToolSpec(
        name="reroute_shipment",
        description=(
            "Change a shipment's journey shape and/or carrier while it is still moving — "
            "the standard intervention on a lane whose predicted spoilage risk is too high. "
            "Cannot be applied to a delivered consignment." + _ID_RULE
        ),
        args_model=RerouteArgs,
        handler=reroute_shipment,
        # MEDIUM: commercially consequential and visible to the shipper, but the consignment
        # keeps moving and the returned inverse restores the previous lane exactly.
        risk=RiskLevel.MEDIUM,
        idempotent=True,
    ),
    "quarantine_shipment": ToolSpec(
        name="quarantine_shipment",
        description=(
            "Hold a shipment for quality review, or release one that is already held. "
            "Quarantining strands product a receiving site is expecting and opens a "
            "regulated review, so it routes to a human for approval before it runs."
            + _ID_RULE
        ),
        args_model=QuarantineArgs,
        handler=quarantine_shipment,
        # HIGH: a consequential, externally-visible state change — it routes to the
        # human-approval gate. This tier IS the gate; there is no second signal.
        risk=RiskLevel.HIGH,
        destructive=True,
        idempotent=True,
    ),
}
"""The domain's own action tools, before the ML tools are spliced in."""

TOOL_REGISTRY: dict[str, ToolSpec] = {
    **_DOMAIN_TOOLS,
    # ── The ML spine reaches the agent loop HERE, and nowhere else. ────────────
    # Five ready-made read-only specs — predict_outcome, explain_prediction,
    # whatif_scenario, forecast_series, check_model_health — built out of this module's own
    # ToolSpec class and returning this module's own ToolActionResult, so the registry stays
    # homogeneous and aegis_ml imports nothing from the adapter.
    #
    # `problem=PROBLEM` is what makes the summaries say "48.2 % spoilage risk" instead of
    # "48.2": the tools read the target's unit and description straight off the declarative
    # problem, so the domain's own words reach the planner without a second table to keep in
    # step.
    **ml_tool_specs(
        ToolSpec,
        problem=PROBLEM,
        risk_low=RiskLevel.LOW,
        result_cls=ToolActionResult,
    ),
}
"""Name → :class:`ToolSpec` for every action tool the domain exposes.

The registry is the whole vocabulary. A name absent from it is treated as HIGH risk by the
platform, so a hallucinated tool can never slip under the gate.
"""


ALLOWLIST: dict[str, frozenset[str]] = {
    # The operator-side persona may perform every action — and, since the registry carries a
    # read tool, enumerate shipments. Its data scope is already "everything", so listing the
    # lookup grants it nothing its scope did not already say it has.
    "logistics_lead": frozenset(TOOL_REGISTRY),
    # The quality persona may look, annotate, quarantine and ask the model — but may not
    # reroute. That is not a security boundary, it is an accountability one: rerouting is a
    # commercial decision belonging to the operator, and a quality auditor who could quietly
    # move a consignment to a cheaper lane is an auditor auditing their own work.
    "quality_auditor": frozenset(
        {"find_shipments", "add_shipment_note", "quarantine_shipment", *ML_TOOL_NAMES}
    ),
    # The shipper-side persona may annotate its own consignments and ask the model about
    # them.
    #
    # **The lookup is deliberately NOT here.** That persona's scope is OWN on
    # ``shipper_id``, and the narrowing is applied by the retrieval/data layers from the
    # authenticated subject — a value :class:`ToolContext` does not carry, so this module
    # cannot enforce it. ``find_shipments`` takes ``shipper_id`` as a *filter*, which would
    # let one shipper enumerate another's consignments by passing someone else's id.
    # Listing it here would not be a roster line, it would be a scope change.
    "shipper_client": frozenset({"add_shipment_note", *ML_TOOL_NAMES}),
}
"""Persona id → the set of tool names that persona is authorised to call."""


def is_allowed(persona_id: str, tool_name: str) -> bool:
    """Return whether ``persona_id`` may call ``tool_name``.

    Args:
        persona_id: The calling persona's id.
        tool_name: The tool being reached for.

    Returns:
        True when the tool is in that persona's allowlist.
    """
    return tool_name in ALLOWLIST.get(persona_id, frozenset())


def tools_for(persona_id: str) -> list[ToolSpec]:
    """Return the :class:`ToolSpec` list a persona is allowed to use.

    Args:
        persona_id: The calling persona's id.

    Returns:
        The allowed specs, sorted by name so the ``tools=`` payload is stable across runs.
    """
    return [
        TOOL_REGISTRY[name]
        for name in sorted(ALLOWLIST.get(persona_id, frozenset()))
        if name in TOOL_REGISTRY
    ]


def tool_definitions_for(persona_id: str) -> list[dict[str, Any]]:
    """Return the LLM ``tools=`` payload for a persona (allowlist-filtered).

    Args:
        persona_id: The calling persona's id.

    Returns:
        One OpenAI/MCP function definition per allowed tool.
    """
    return [spec.definition() for spec in tools_for(persona_id)]


async def run_tool(
    persona_id: str, tool_name: str, args: dict[str, Any], ctx: ToolContext
) -> ToolActionResult:
    """Authorise and execute one tool for a persona.

    The allowlist is checked **before** any side effect, so an unauthorised call can never
    mutate state or emit an audit record. Keep that ordering.

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
        raise ToolNotAllowedError(
            f"Persona {persona_id!r} may not call tool {tool_name!r}."
        )
    return await TOOL_REGISTRY[tool_name].handler(args, ctx)

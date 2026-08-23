"""Piece 5 of 10 — who is asking, and what they are allowed to see.

WHAT THIS FILE IS
    A persona is **not** a UI label and not a personality: it is an authorisation object.
    Its ``data_scope`` becomes a retrieval/data filter, and its id is the key into piece 4's
    ``ALLOWLIST``, which becomes its tool set. Writing a persona is therefore a statement
    about who this platform serves and what they may reach.

    Three of them here, and the third earns its place rather than padding the list:

    * :data:`LOGISTICS_LEAD` — the operator. Sees everything, may do everything.
    * :data:`QUALITY_AUDITOR` — also sees everything, and may quarantine, but may **not**
      reroute. That is an accountability boundary rather than a security one, and it is the
      pair that makes the persona model legible: two personas with identical data scope and
      deliberately different tool sets.
    * :data:`SHIPPER_CLIENT` — the customer. Sees only its own consignments.

    Tool scope is NOT duplicated here. It is read straight from
    :data:`~reference.adapter.tools.ALLOWLIST`, so there is exactly one source of truth
    about what a persona may call.

THE CONTRACT (aegis.adapter.PersonasModule) — these names must survive
    PERSONAS, DEFAULT_PERSONA_ID, PERSONA_BY_ROLE, get_persona(), persona_for_role()

    Plus, by convention: ``Persona``, ``DataScope``, ``ScopeKind`` — the registry re-exports
    ``Persona`` and ``ScopeKind``, and piece 6 imports both from here.

THE TRAP — this one bites the moment a human signs in
    **Every RBAC role must map to a persona, or every login raises ``KeyError``.**

    Authenticated principals resolve their persona through :func:`persona_for_role`, which
    reads :data:`PERSONA_BY_ROLE`. Re-voicing :data:`PERSONAS` without re-pointing that
    table makes every sign-in raise — while the adapter suite, the agent suite, the ML suite
    and ruff all stay green, because **no suite in the repository goes through the login
    path**. That is precisely how the failure was originally found: in a rehearsal, by a
    human logging in.

    Every role in the platform's ``Role`` enum needs an entry (``admin``, ``ai_team``,
    ``devops``, ``client``), and every value must be a key of :data:`PERSONAS`. The
    conformance suite asserts both halves — and also that ``get_persona(None)`` returns
    :data:`DEFAULT_PERSONA_ID`'s persona, because that lookup runs on **every request that
    names no persona**.

    Second trap, quieter: :func:`persona_for_role` raises on an unknown role **on purpose**.
    The alternative — falling back to some default — is a request silently running under
    another persona's data scope and tool allowlist, which is an authorisation change
    wearing a convenience's clothes.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from reference.adapter.tools import ALLOWLIST

__all__ = [
    "DEFAULT_PERSONA_ID",
    "LOGISTICS_LEAD",
    "PERSONAS",
    "PERSONA_BY_ROLE",
    "QUALITY_AUDITOR",
    "SHIPPER_CLIENT",
    "DataScope",
    "Persona",
    "Role",
    "ScopeKind",
    "get_persona",
    "persona_for_role",
]


class _StandaloneRole(StrEnum):
    """Coarse RBAC roles, for when no Aegis checkout is on the path.

    Member names and values match ``aegis.governance.types.Role`` exactly, and a
    ``StrEnum`` member hashes and compares as its string value — so a table keyed by these
    members is looked up correctly by the platform's own members, and vice versa. That
    equivalence is what makes the fallback a stand-in rather than a divergence.
    """

    ADMIN = "admin"
    AI_TEAM = "ai_team"
    DEVOPS = "devops"
    CLIENT = "client"


def _resolve_role() -> type[StrEnum]:
    """Return the platform's ``Role`` enum, or the standalone stand-in.

    The adapter must import with no Aegis checkout present, while still keying its RBAC
    table with the platform's own enum whenever the platform is there. The handled failure
    has exactly one cause (no importable ``aegis.governance.types``) and exactly one
    consequence, a value-identical stand-in — it is a documented substitution, not a
    swallowed error.

    Returns:
        ``aegis.governance.types.Role`` when importable, else :class:`_StandaloneRole`.
    """
    try:
        from aegis.governance.types import Role as PlatformRole
    except ImportError:
        return _StandaloneRole
    return PlatformRole


Role: type[StrEnum] = _resolve_role()
"""The RBAC role enum this domain maps onto personas."""


class ScopeKind(StrEnum):
    """How broadly a persona may see cold-chain records.

    Only two kinds, because only two are enforced. A kind added here that nothing knows how
    to apply does not narrow anything — it silently widens to "everything", which is the
    worst possible way for an authorisation vocabulary to fail.
    """

    ALL = "all"
    """Every shipment the deployment holds (operator- and quality-side)."""

    OWN = "own"
    """Only shipments belonging to the authenticated subject."""


class DataScope(BaseModel):
    """Declarative data visibility for a persona.

    The data/retrieval layers translate this into a filter. ``OWN`` scope is bound at
    request time to the authenticated subject id; :attr:`subject_field` names the record
    field that must equal that subject.
    """

    kind: ScopeKind = Field(description="How broadly this persona may see.")
    subject_field: str | None = Field(
        default=None,
        description="Record field constrained for OWN scope, e.g. 'shipper_id'.",
    )


class Persona(BaseModel):
    """A concrete persona the agent can adopt.

    Attributes:
        id: Stable persona id, used in the query request and as the ``ALLOWLIST`` key.
        role: Coarse RBAC role this persona is the default for.
        display_name: Human label for the console.
        description: One-line summary of who this persona is.
        data_scope: What data this persona may see.
        prompt_key: Key into :data:`reference.adapter.prompts.SYSTEM_PROMPTS`.
    """

    id: str
    role: Role
    display_name: str
    description: str
    data_scope: DataScope
    prompt_key: str

    @property
    def tool_names(self) -> frozenset[str]:
        """Return the tool names this persona may call (read from the allowlist)."""
        return ALLOWLIST.get(self.id, frozenset())


# ─────────────────────────────────────────────────────────────────────────────
# The personas
#
# The ids are short, lower_snake and stable — they appear in the ALLOWLIST, in
# SYSTEM_PROMPTS' keys, in the console's persona picker, on the wire in every query request,
# and in the audit trail.
# ─────────────────────────────────────────────────────────────────────────────

LOGISTICS_LEAD = Persona(
    id="logistics_lead",
    role=Role.ADMIN,
    display_name="Cold-Chain Logistics Lead",
    description=(
        "Runs the temperature-controlled network day to day: books lanes, watches "
        "consignments in flight, intervenes on the ones the model flags, and answers for "
        "every write-off. Accountable for both the product that arrives usable and the "
        "freight bill that paid for it, which is why this is the persona that may reroute."
    ),
    data_scope=DataScope(kind=ScopeKind.ALL),
    prompt_key="logistics_lead",
)

QUALITY_AUDITOR = Persona(
    id="quality_auditor",
    role=Role.AI_TEAM,
    display_name="Cold-Chain Quality Auditor",
    description=(
        "Reviews logger records against qualified temperature ranges, decides what is "
        "released and what is quarantined, and writes the assessment that closes a "
        "deviation. Sees every consignment and may hold any of them — and may not reroute "
        "one, because an auditor who could quietly move a consignment to a cheaper lane "
        "would be auditing their own work."
    ),
    data_scope=DataScope(kind=ScopeKind.ALL),
    prompt_key="quality_auditor",
)

SHIPPER_CLIENT = Persona(
    id="shipper_client",
    role=Role.CLIENT,
    display_name="Shipper Contact",
    description=(
        "The pharmaceutical shipper whose product is moving. Asks where a consignment is, "
        "whether it is at risk, and what happens if it is — about their own account's "
        "consignments and nobody else's."
    ),
    data_scope=DataScope(kind=ScopeKind.OWN, subject_field="shipper_id"),
    prompt_key="shipper_client",
)


PERSONAS: dict[str, Persona] = {
    persona.id: persona for persona in (LOGISTICS_LEAD, QUALITY_AUDITOR, SHIPPER_CLIENT)
}
"""Persona id → :class:`Persona` for every persona the domain exposes."""

DEFAULT_PERSONA_ID: str = LOGISTICS_LEAD.id
"""Persona used when a request does not name one.

Must be a key of :data:`PERSONAS`: ``PERSONAS[DEFAULT_PERSONA_ID]`` is evaluated on every
request that omits a persona, so a stale value here is not a small mistake — it is every
anonymous request failing at the first turn.
"""

PERSONA_BY_ROLE: dict[StrEnum, str] = {
    Role.ADMIN: LOGISTICS_LEAD.id,
    Role.AI_TEAM: QUALITY_AUDITOR.id,
    Role.DEVOPS: LOGISTICS_LEAD.id,
    Role.CLIENT: SHIPPER_CLIENT.id,
}
"""Coarse RBAC role → the persona an authenticated principal of that role adopts.

**This table is domain knowledge, and it used to live in the core.** RBAC roles belong to
the platform; *which persona a role adopts* is a statement about who this domain serves, so
the mapping belongs beside the personas it names.

``DEVOPS`` maps to the logistics lead rather than to a persona of its own: a platform
operator debugging this deployment needs the operator's view of the data, and inventing a
fourth persona to hold an identical scope would be a persona nobody could describe.
"""


def persona_for_role(role: StrEnum | str) -> str:
    """Return the persona id a principal holding ``role`` adopts.

    Args:
        role: The coarse RBAC role, as the enum member or its string value.

    Returns:
        The persona id, always a key of :data:`PERSONAS`.

    Raises:
        KeyError: If ``role`` is not a known role, or the domain declares no persona for it.
            Deliberately loud: the alternative is a request that runs under some other
            persona's data scope and tool allowlist, which is a silent authorisation change.
    """
    key = role if isinstance(role, Role) else Role(role)
    persona_id = PERSONA_BY_ROLE.get(key)
    if persona_id is None:
        raise KeyError(
            f"No persona is declared for role {str(key)!r}. Add it to "
            f"reference.adapter.personas.PERSONA_BY_ROLE — every role the platform can "
            f"authenticate must map to one of {sorted(PERSONAS)}."
        )
    return persona_id


def get_persona(persona_id: str | None) -> Persona:
    """Return the persona for ``persona_id`` (or the default if ``None``).

    Args:
        persona_id: The requested persona id, or ``None`` for the default.

    Returns:
        The matching :class:`Persona`.

    Raises:
        KeyError: If ``persona_id`` is given but unknown.
    """
    if persona_id is None:
        return PERSONAS[DEFAULT_PERSONA_ID]
    return PERSONAS[persona_id]

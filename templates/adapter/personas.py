"""Piece 5 of 10 — who is asking, and what they are allowed to see.

WHAT YOU WRITE HERE
    A persona is **not** a UI label and not a personality: it is an authorisation
    object. Its ``data_scope`` becomes a retrieval/data filter, and its id is the key
    into piece 4's ``ALLOWLIST``, which becomes its tool set. Writing a persona is
    therefore a statement about who this platform serves and what they may reach.

    Two is usually the right number to start with — one broad (the operator of the
    system) and one narrow (the party it is operated on behalf of) — because the demo
    that lands is the one where a jury can see the *same question* answered differently
    for two people.

    Tool scope is NOT duplicated here. It is read straight from
    ``app.adapter.tools.ALLOWLIST``, so there is exactly one source of truth about what
    a persona may call.

THE CONTRACT (aegis.adapter.PersonasModule) — these names must survive
    PERSONAS, DEFAULT_PERSONA_ID, PERSONA_BY_ROLE, get_persona(), persona_for_role()

    Plus, by convention: ``Persona``, ``DataScope``, ``ScopeKind`` — the registry
    re-exports ``Persona`` and ``ScopeKind``, and piece 6 imports both from here.

THE TRAP — this one bites the moment a human signs in
    **Every RBAC role must map to a persona, or every login raises ``KeyError``.**

    Authenticated principals resolve their persona through
    :func:`persona_for_role`, which reads :data:`PERSONA_BY_ROLE`. Re-voice
    :data:`PERSONAS` (which is exactly what this step tells you to do) without
    re-pointing that table and every sign-in raises — while the adapter suite, the
    agent suite, the ML suite and ruff all stay green, because **no suite in the
    repository goes through the login path**. The host used to decide this itself with
    two persona ids hardcoded in an ``if`` in its login route; that is precisely how
    the failure was found, in a rehearsal, by a human logging in.

    Every role in ``aegis.governance.types.Role`` needs an entry (today: ``admin``,
    ``ai_team``, ``devops``, ``client``), and every value must be a key of
    :data:`PERSONAS`. Conformance check #7 asserts both halves — and also that
    ``get_persona(None)`` returns ``DEFAULT_PERSONA_ID``'s persona, because that
    lookup runs on **every request that names no persona**.

    Second trap, quieter: :func:`persona_for_role` raises on an unknown role **on
    purpose**. The alternative — falling back to some default — is a request silently
    running under another persona's data scope and tool allowlist, which is an
    authorisation change wearing a convenience's clothes.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        tests/adapter/test_registry.py -q)
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        --pyargs aegis.conformance --aegis-adapter app.adapter -q -k persona)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.adapter.tools import ALLOWLIST
from app.api.schemas import Role


class ScopeKind(StrEnum):
    """How broadly a persona may see domain records.

    TODO(domain): add a kind only if the domain genuinely needs one (a per-site or
    per-tenant scope, say). Every kind added here must be understood by whatever
    applies the filter, or it silently widens to "everything".
    """

    ALL = "all"  # every record the deployment holds (operator-side)
    OWN = "own"  # only records belonging to the authenticated subject


class DataScope(BaseModel):
    """Declarative data visibility for a persona.

    The data/retrieval layers translate this into a filter. ``OWN`` scope is bound at
    request time to the authenticated subject id; :attr:`subject_field` names the
    record field that must equal that subject.
    """

    kind: ScopeKind
    subject_field: str | None = Field(
        default=None,
        description="Record field constrained for OWN scope, e.g. 'party_id'.",
    )


class Persona(BaseModel):
    """A concrete persona the agent can adopt.

    Attributes:
        id: Stable persona id, used in ``QueryRequest.persona`` and as the ALLOWLIST key.
        role: Coarse RBAC role from the locked API schema.
        display_name: Human label for the console.
        description: One-line summary of who this persona is.
        data_scope: What data this persona may see.
        prompt_key: Key into ``app.adapter.prompts.SYSTEM_PROMPTS``.
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
# TODO(domain): replace both. Keep the ids short, lower_snake and stable — they appear
# in the ALLOWLIST, in SYSTEM_PROMPTS' keys, in the console's persona picker, on the
# wire in every QueryRequest, and in the audit trail.
# ─────────────────────────────────────────────────────────────────────────────

LINE_SUPERVISOR = Persona(
    id="line_supervisor",
    role=Role.ADMIN,
    display_name="Line Supervisor",
    description=(
        "TODO(domain): the operator-side persona — who runs this process day to day, "
        "and what they are accountable for."
    ),
    data_scope=DataScope(kind=ScopeKind.ALL),
    prompt_key="line_supervisor",
)

PARTY_CONTACT = Persona(
    id="party_contact",
    role=Role.CLIENT,
    display_name="Party Contact",
    description=(
        "TODO(domain): the party-side persona — the external person the process is run "
        "for, who may see only their own records."
    ),
    data_scope=DataScope(kind=ScopeKind.OWN, subject_field="party_id"),
    prompt_key="party_contact",
)


PERSONAS: dict[str, Persona] = {p.id: p for p in (LINE_SUPERVISOR, PARTY_CONTACT)}
"""Persona id → :class:`Persona` for every persona the domain exposes."""

DEFAULT_PERSONA_ID: str = LINE_SUPERVISOR.id
"""Persona used when a request does not name one.

Must be a key of :data:`PERSONAS`: ``PERSONAS[DEFAULT_PERSONA_ID]`` is evaluated on
every request that omits a persona, so a stale value here is not a small mistake — it
is every anonymous request failing at the first turn.
"""

PERSONA_BY_ROLE: dict[Role, str] = {
    Role.ADMIN: LINE_SUPERVISOR.id,
    Role.AI_TEAM: LINE_SUPERVISOR.id,
    Role.DEVOPS: LINE_SUPERVISOR.id,
    Role.CLIENT: PARTY_CONTACT.id,
}
"""Coarse RBAC role → the persona an authenticated principal of that role adopts.

**This table is domain knowledge, and it used to live in the core.** RBAC roles belong
to the platform; *which persona a role adopts* is a statement about who this domain
serves, so the mapping belongs beside the personas it names.

TODO(domain): re-point every value. Every :class:`~aegis.governance.types.Role` must
appear as a key — a role with no persona cannot sign in — and every value must be a key
of :data:`PERSONAS`. If a new ``Role`` member exists in your Aegis checkout, add it
here too; the conformance check enumerates the enum, not this dict.
"""


def persona_for_role(role: Role | str) -> str:
    """Return the persona id a principal holding ``role`` adopts.

    Args:
        role: The coarse RBAC role, as the enum or its string value.

    Returns:
        The persona id, always a key of :data:`PERSONAS`.

    Raises:
        KeyError: If ``role`` is not a known role, or the domain declares no persona
            for it. Deliberately loud: the alternative is a request that runs under
            some other persona's data scope and tool allowlist, which is a silent
            authorisation change.
    """
    key = role if isinstance(role, Role) else Role(role)
    persona_id = PERSONA_BY_ROLE.get(key)
    if persona_id is None:
        raise KeyError(
            f"No persona is declared for role {key.value!r}. Add it to "
            f"app.adapter.personas.PERSONA_BY_ROLE — every role the platform can "
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


__all__ = [
    "DEFAULT_PERSONA_ID",
    "PERSONAS",
    "PERSONA_BY_ROLE",
    "DataScope",
    "Persona",
    "ScopeKind",
    "get_persona",
    "persona_for_role",
]

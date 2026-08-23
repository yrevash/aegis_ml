"""Piece 7 of 10 — what counts as a durable fact, plus ``SKILLS_DIR`` (piece 10).

WHAT YOU WRITE HERE
    The core memory subsystem is domain-agnostic: *how* to persist, score, recall,
    budget and consolidate is core. *What counts as a durable fact*, *how to extract
    it*, *who memory is scoped to*, and *how the profile reads back* are domain meaning
    and live here. On the day this file (plus ``skills/*.md``) is rewritten and nothing
    in ``app/memory/*`` or ``aegis/memory/*`` changes.

      * :data:`FACT_TYPES` / :data:`PROFILE_FIELDS` — what a durable fact is in this
        world, and the small always-injected structured block.
      * :data:`PROFILE_ALIASES` — the predicate spellings your extractor actually emits,
        mapped onto :data:`PROFILE_FIELDS`. Absent is a legitimate statement ("no
        aliases"); wrong is a profile that never fills in.
      * :data:`FACT_EXTRACTION_PROMPT` / :data:`IMPORTANCE_HINTS` — how facts are pulled
        out of a conversation, and how they are rated 1..10.
      * :func:`memory_subject_for` — **whose** memory a turn reads and writes.
      * :func:`render_profile` — how the profile reads back into a prompt.
      * :func:`select_skills` — which playbooks a query needs.
      * :data:`SKILLS_DIR` — where piece 10 lives.

THE CONTRACT (aegis.adapter.MemorySpecModule) — every one of these must survive
    FACT_TYPES, PROFILE_FIELDS, FACT_EXTRACTION_PROMPT, IMPORTANCE_HINTS, SKILLS_DIR,
    FactSchema, FactExtraction,
    memory_subject_for(), render_profile(), select_skills()

    **The module object itself is the contract.** ``app.memory`` installs it with
    ``set_default_spec(app.adapter.memory_spec)``, and other host modules import
    ``FACT_TYPES`` and ``memory_subject_for`` from it by path. So this file must keep
    its **path and its symbol names**, not merely its behaviour — and
    ``adapter/__init__.py`` must import it as a *module* or it is not a member of the
    package at all (see the registry's own note; this bit the reference adapter).

THE TRAP — the one that is a data leak rather than a quality regression
    :func:`memory_subject_for` decides the app-level isolation key every memory query
    filters on (``WHERE subject_id = …``, the primary NULL-safe isolator, with tenant
    RLS underneath it as the belt). Getting it wrong does not degrade an answer — it
    reads and writes **another subject's memory**. Nothing outside this function may
    compose that key, which is exactly why the whole platform goes through it: the
    query route, the read surfaces' "is this the caller's own subject?" checks, and the
    control plane's write/correct/erase scoping all derive from its output, and the
    browser never composes it at all.

THE OTHER TRAP — silent, and you will read it as a prompt problem
    Playbooks are selected **by filename**, through the literal keyword→filename table
    in :func:`select_skills` / :data:`SKILL_HINTS`. Add or rename a playbook without
    updating that table and it can never be chosen: ``select_skills`` returns ``None``,
    the core injects no skill, the agent answers anyway without its procedure, and
    **nothing warns**. You will spend an hour in the prompt file.

    So piece 10 is really two edits — the ``*.md`` files and this table. Conformance
    check #11 reads the table (whether it sits inside the function or beside it as a
    module constant — both are read) and fails naming any playbook on disk that no
    literal can reach.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        --pyargs aegis.conformance --aegis-adapter app.adapter -q -k "memory or skill")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

#: The typed kinds of durable fact this domain distils (guides the extractor).
#:
#: TODO(domain): four is a good number. Each should name something that is still true
#: next month — a fact type describing the current turn is a fact type that fills the
#: store with noise.
FACT_TYPES: list[str] = [
    "preference",   # how this subject likes to be handled (channel, language, cadence)
    "entity_attr",  # a stable attribute (tier, site, timezone, configuration)
    "commitment",   # something the organisation promised this subject
    "constraint",   # a standing limitation (contractual, compliance, do-not-contact)
]

#: Structured profile fields (the always-injected "human block"). Small + high-value.
#:
#: TODO(domain): these are injected into EVERY turn, so the cost of a low-value field
#: is paid on every request. Order matters — :func:`render_profile` renders in this
#: order, so put the field that changes the answer first.
PROFILE_FIELDS: list[str] = [
    "display_name",
    "tier",
    "zone",
    "timezone",
    "preferred_channel",
    "preferred_language",
    "open_commitments",
    "notes",
]

PROFILE_ALIASES: dict[str, str] = {
    # TODO(domain): the LEFT side is how your extractor's model actually phrases the
    # attribute; the RIGHT side is your field name. Both halves are domain vocabulary.
    "prefers_channel": "preferred_channel",
    "channel": "preferred_channel",
    "prefers_language": "preferred_language",
    "language": "preferred_language",
    "name": "display_name",
    "party_tier": "tier",
    "site": "zone",
}
"""Predicate spellings the extractor emits → the :data:`PROFILE_FIELDS` they write.

Consolidation writes a fact into the structured profile when its predicate *is* a
profile field, or when this table maps it onto one. It lived in the core's consolidate
module as a constant until a retarget rehearsal, where it became an alias set matching
nothing in a domain that had never heard of the shipped domain's fields — so the
profile silently stopped filling in while every fact still landed in the store.
"""

SKILLS_DIR: str = str(Path(__file__).parent / "skills")
"""Where procedural skill playbooks live — **this is piece 10**.

Resolved relative to this file, so it survives a checkout, a wheel and an rsync. It has
no separate Protocol member of its own precisely because it is named here.
"""

IMPORTANCE_HINTS: str = (
    "TODO(domain): map your fact types onto the 1-10 scale. Rate 1-3 for trivia, 4-6 "
    "for useful preferences and attributes, 7-8 for commitments and contractual "
    "constraints, 9-10 for safety, legal or do-not-contact constraints."
)
"""Domain guidance for the 1..10 importance (poignancy) rating during extraction.

Concatenated into :data:`FACT_EXTRACTION_PROMPT`, so it is guidance the extractor model
actually reads — the scale is what decides which facts survive the recall budget.
"""


class FactSchema(BaseModel):
    """One durable fact the cheap-model extractor must emit (the typed target).

    TODO(domain): re-default ``subject`` to your domain's subject noun and re-voice the
    field descriptions. The *shape* — a subject/predicate/object triple plus a
    natural-language rendering plus confidence and importance — is what the memory
    subsystem indexes, so keep it.

    Attributes:
        fact_type: One of :data:`FACT_TYPES`.
        subject: Canonical entity the fact is about (a noun, or an id).
        predicate: The relation (``"prefers_channel"``, ``"tier"``, ``"timezone"``).
        object: The value (``"email"``, ``"prime"``, ``"Europe/Berlin"``).
        text: Natural-language rendering — what gets injected and embedded.
        confidence: Extractor confidence in [0, 1].
        importance: Poignancy 1..10 (see :data:`IMPORTANCE_HINTS`).
        valid_at: World-time the fact became true, if the turn states it (else null).
    """

    fact_type: str = Field(default="entity_attr")
    subject: str = Field(default="party")
    predicate: str
    object: str
    text: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    importance: int = Field(default=5, ge=1, le=10)
    valid_at: datetime | None = Field(default=None)


class FactExtraction(BaseModel):
    """Container the extractor returns (a JSON object with a ``facts`` list)."""

    facts: list[FactSchema] = Field(default_factory=list)


FACT_EXTRACTION_PROMPT: str = (
    "TODO(domain): 'You maintain the long-term memory of a <domain> assistant.' From "
    "the conversation turns, extract only DURABLE facts about the subject that will "
    "still matter in future conversations — stable preferences, attributes, "
    "commitments made to them, and standing constraints. Do NOT extract transient "
    "detail about the current record, one-off questions, or the assistant's own "
    "statements. Merge duplicates. For each fact give a short predicate and object, a "
    "one-sentence natural-language `text`, a `confidence` in [0,1], and an "
    "`importance` 1-10. " + IMPORTANCE_HINTS + " Return JSON of the form "
    '{"facts": [{"fact_type":..., "subject":..., "predicate":..., "object":..., '
    '"text":..., "confidence":..., "importance":...}, ...]}. Return an empty list if '
    "nothing durable was said."
)
"""The system prompt driving the cheap-model fact extractor.

TODO(domain): the two clauses doing the real work are "durable, not transient" and
"return an empty list if nothing durable was said". Keep both — without the second, a
model asked for facts will always find some.
"""


def memory_subject_for(
    user_id: str | int | None, persona_id: str | None = None
) -> str | None:
    """Resolve the memory subject a run is scoped to (the app-level isolation key).

    **This function is THE adapter seam for memory scoping, and it is load-bearing
    twice.** It decides the key every memory query filters on, and it decides *what a
    memory is about* — which is a domain statement, not an infrastructure one.

    The default below scopes memory to the authenticated **user**. A domain that scopes
    memory to a **business entity** instead — the account, the case, the vehicle, the
    party a persona is acting on behalf of — changes this one function and nothing
    else. That is the whole reason nothing outside it may compose the key itself.

    ``None`` means "no subject" (anonymous, single-shot): memory stays inert, which is a
    real answer rather than a failure.

    TODO(domain): decide the scope deliberately and write the reason down here. Getting
    it wrong is a cross-subject data leak, not a quality regression, and the wrong
    choice looks completely normal in every test.

    Args:
        user_id: The authenticated user's id (or ``None``).
        persona_id: The active persona — unused in the default, available for
            entity-scoped domains where the persona names which entity is in view.

    Returns:
        A stable subject key, or ``None`` to disable memory for this run.
    """
    if user_id is None or user_id == "":
        return None
    return f"user:{user_id}"


def render_profile(profile: dict[str, Any]) -> str:
    """Render the structured profile JSON as a compact prompt "human block".

    Args:
        profile: The stored profile data (a subset of :data:`PROFILE_FIELDS`).

    Returns:
        A short multi-line block, or an empty string when the profile is empty — an
        empty string rather than a header with nothing under it, so a subject the
        platform knows nothing about costs no tokens and makes no claim.
    """
    if not profile:
        return ""
    # TODO(domain): re-voice this header in your domain's noun.
    lines = ["Known about this subject:"]
    for field in PROFILE_FIELDS:
        value = profile.get(field)
        if value in (None, "", [], {}):
            continue
        label = field.replace("_", " ")
        rendered = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines) if len(lines) > 1 else ""


SKILL_HINTS: dict[str, str] = {
    # keyword found in the query  →  playbook filename WITHOUT the .md
    #
    # TODO(domain): rewrite this table and ``skills/`` together, in one edit. Every
    # ``*.md`` in SKILLS_DIR must appear as a value here or it can never be selected,
    # and conformance check #11 will name it. A value naming a file that does not exist
    # is harmless (it is filtered by ``skill in available``) but is dead weight.
    #
    # These entries point at ``skills/EXAMPLE.md``, which ships as the format guide.
    # Delete it once you have written real playbooks, and delete these rows with it.
    "example": "EXAMPLE",
    "escalate": "EXAMPLE",
    "close": "EXAMPLE",
}
"""Literal keyword → playbook filename (no extension).

Hoisted to a module constant on purpose: conformance check #11 reads the selector's
compiled constants **and** its module's top-level strings, so either placement works —
but a constant is easier to keep in step with the directory, and the check's own scar
is that reading only the function's constants made a tidying refactor silently empty
the check.
"""


def select_skills(
    query: str, persona_id: str | None, available: list[str]
) -> list[str] | None:
    """Deterministically select the procedural skills a query needs.

    Returns a subset of ``available`` (filenames without ``.md``) by simple keyword
    match — cheap, offline, and reshapeable in seconds on the day. Returning ``None``
    or an empty list means "no skill for this turn" and the core injects none.

    **Only ever return names from ``available``.** The check probes this function with
    every playbook name, every word inside those names, and every string it carries,
    and fails if it invents one.

    Args:
        query: The user query.
        persona_id: The active persona (for per-persona allowlisting, if a domain
            needs it — the default ignores it).
        available: Skill names discovered under :data:`SKILLS_DIR`.

    Returns:
        The selected skill names, or ``None`` when nothing applies.
    """
    lowered = query.lower()
    chosen: list[str] = []
    for keyword, skill in SKILL_HINTS.items():
        if keyword in lowered and skill in available and skill not in chosen:
            chosen.append(skill)
    return chosen or None


__all__ = [
    "FACT_EXTRACTION_PROMPT",
    "FACT_TYPES",
    "IMPORTANCE_HINTS",
    "PROFILE_ALIASES",
    "PROFILE_FIELDS",
    "SKILLS_DIR",
    "SKILL_HINTS",
    "FactExtraction",
    "FactSchema",
    "memory_subject_for",
    "render_profile",
    "select_skills",
]

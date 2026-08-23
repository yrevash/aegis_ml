"""Piece 7 of 10 — what counts as a durable fact, plus ``SKILLS_DIR`` (piece 10).

WHAT THIS FILE IS
    The core memory subsystem is domain-agnostic: *how* to persist, score, recall, budget
    and consolidate is core. *What counts as a durable fact*, *how to extract it*, *who
    memory is scoped to*, and *how the profile reads back* are domain meaning and live here.

      * :data:`FACT_TYPES` / :data:`PROFILE_FIELDS` — what a durable fact is in a cold-chain
        world, and the small always-injected structured block.
      * :data:`PROFILE_ALIASES` — the predicate spellings the extractor actually emits,
        mapped onto :data:`PROFILE_FIELDS`.
      * :data:`FACT_EXTRACTION_PROMPT` / :data:`IMPORTANCE_HINTS` — how facts are pulled out
        of a conversation, and how they are rated 1..10.
      * :func:`memory_subject_for` — **whose** memory a turn reads and writes.
      * :func:`render_profile` — how the profile reads back into a prompt.
      * :func:`select_skills` + :data:`SKILL_HINTS` — which playbooks a query needs.
      * :data:`SKILLS_DIR` — where piece 10 lives.

THE CONTRACT (aegis.adapter.MemorySpecModule) — every one of these must survive
    FACT_TYPES, PROFILE_FIELDS, FACT_EXTRACTION_PROMPT, IMPORTANCE_HINTS, SKILLS_DIR,
    FactSchema, FactExtraction,
    memory_subject_for(), render_profile(), select_skills()

    **The module object itself is the contract.** A host installs it with
    ``set_default_spec(reference.adapter.memory_spec)``, and other host modules import
    ``FACT_TYPES`` and ``memory_subject_for`` from it by path. So this file keeps its **path
    and its symbol names**, not merely its behaviour — and ``adapter/__init__.py`` must
    import it as a *module* or it is not a member of the package at all.

THE TRAP — the one that is a data leak rather than a quality regression
    :func:`memory_subject_for` decides the app-level isolation key every memory query
    filters on. Getting it wrong does not degrade an answer — it reads and writes **another
    subject's memory**. Nothing outside this function may compose that key, which is exactly
    why the whole platform goes through it.

THE OTHER TRAP — silent, and it reads as a prompt problem
    Playbooks are selected **by filename**, through the literal keyword→filename table in
    :data:`SKILL_HINTS`. Add or rename a playbook without updating that table and it can
    never be chosen: :func:`select_skills` returns ``None``, the core injects no skill, the
    agent answers anyway without its procedure, and **nothing warns**. You will spend an hour
    in the prompt file.

    So piece 10 is really two edits — the ``*.md`` files and this table — and the conformance
    suite reads the table and fails naming any playbook on disk that no literal can reach.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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

FACT_TYPES: list[str] = [
    "preference",  # how this subject likes to be handled (channel, cadence, packout)
    "entity_attr",  # a stable attribute (tier, region, cold-room capability, timezone)
    "commitment",  # something the distributor promised this subject
    "constraint",  # a standing limitation (regulatory, contractual, do-not-ship)
]
"""The typed kinds of durable fact this domain distils (guides the extractor).

Four, and each names something that is still true next month. A fact type describing the
current consignment is a fact type that fills the store with noise: "ship-000412 is at the
hub" is a lookup, not a memory.
"""

PROFILE_FIELDS: list[str] = [
    "display_name",
    "shipper_tier",
    "qualified_temperature_range",
    "preferred_packaging",
    "receiving_window",
    "region",
    "timezone",
    "preferred_channel",
    "open_commitments",
    "standing_constraints",
]
"""Structured profile fields — the always-injected "human block". Small and high-value.

Injected into **every** turn, so the cost of a low-value field is paid on every request.
Order matters: :func:`render_profile` renders in this order, and the two fields that most
often change the answer — the qualified range and the preferred packout — sit near the top
because a coordinator who reads only the first three lines still reads the ones that matter.
"""

PROFILE_ALIASES: dict[str, str] = {
    # LEFT: how the extractor's model actually phrases the attribute.
    # RIGHT: the :data:`PROFILE_FIELDS` name it writes.
    "name": "display_name",
    "account_tier": "shipper_tier",
    "tier": "shipper_tier",
    "temperature_range": "qualified_temperature_range",
    "qualified_range": "qualified_temperature_range",
    "packaging": "preferred_packaging",
    "prefers_packaging": "preferred_packaging",
    "delivery_window": "receiving_window",
    "goods_in_hours": "receiving_window",
    "prefers_channel": "preferred_channel",
    "channel": "preferred_channel",
    "site_region": "region",
}
"""Predicate spellings the extractor emits → the :data:`PROFILE_FIELDS` they write.

Consolidation writes a fact into the structured profile when its predicate *is* a profile
field, or when this table maps it onto one. It lived in the core's consolidate module as a
constant until a retarget rehearsal, where it became an alias set matching nothing in a
domain that had never heard of the shipped domain's fields — so the profile silently stopped
filling in while every fact still landed in the store.
"""

SKILLS_DIR: str = str(Path(__file__).parent / "skills")
"""Where procedural skill playbooks live — **this is piece 10**.

Resolved relative to this file, so it survives a checkout, a wheel and an rsync. It has no
separate Protocol member of its own precisely because it is named here.
"""

IMPORTANCE_HINTS: str = (
    "Rate 1-3 for trivia and passing remarks. Rate 4-6 for handling preferences and stable "
    "account attributes — a preferred packout, a goods-in window, a usual contact channel. "
    "Rate 7-8 for commitments the distributor has made to this subject and for contractual "
    "constraints such as a qualified temperature range or a carrier the account has "
    "excluded. Rate 9-10 for anything with a patient-safety, regulatory or licensing "
    "consequence: a product that must never be frozen, a site that may not receive "
    "controlled stock, a standing recall or hold."
)
"""Domain guidance for the 1..10 importance (poignancy) rating during extraction.

Concatenated into :data:`FACT_EXTRACTION_PROMPT`, so it is guidance the extractor model
actually reads — the scale is what decides which facts survive the recall budget.
"""


class FactSchema(BaseModel):
    """One durable fact the cheap-model extractor must emit (the typed target).

    The *shape* — a subject/predicate/object triple plus a natural-language rendering plus
    confidence and importance — is what the memory subsystem indexes, so it is kept as the
    platform expects. The defaults and the field meanings are this domain's.

    Attributes:
        fact_type: One of :data:`FACT_TYPES`.
        subject: Canonical entity the fact is about (a shipper account, a receiving site).
        predicate: The relation (``"qualified_range"``, ``"prefers_packaging"``).
        object: The value (``"2-8C"``, ``"passive_pcm"``, ``"Europe/Lisbon"``).
        text: Natural-language rendering — what gets injected and embedded.
        confidence: Extractor confidence in [0, 1].
        importance: Poignancy 1..10 (see :data:`IMPORTANCE_HINTS`).
        valid_at: World-time the fact became true, if the turn states it (else null).
    """

    fact_type: str = Field(default="entity_attr", description="One of FACT_TYPES.")
    subject: str = Field(default="shipper", description="Who or what the fact is about.")
    predicate: str = Field(description="The relation, in lower_snake.")
    object: str = Field(description="The value the relation holds.")
    text: str = Field(description="One-sentence natural-language rendering.")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Extractor confidence.")
    importance: int = Field(default=5, ge=1, le=10, description="Poignancy 1..10.")
    valid_at: datetime | None = Field(
        default=None, description="When the fact became true, if the turn says."
    )


class FactExtraction(BaseModel):
    """Container the extractor returns (a JSON object with a ``facts`` list)."""

    facts: list[FactSchema] = Field(default_factory=list, description="Extracted facts.")


FACT_EXTRACTION_PROMPT: str = (
    "You maintain the long-term memory of a pharmaceutical cold-chain logistics assistant. "
    "From the conversation turns, extract only DURABLE facts about the subject that will "
    "still matter in future conversations — stable handling preferences, account and site "
    "attributes, commitments the distributor has made to them, and standing regulatory or "
    "contractual constraints. Do NOT extract transient detail about the consignment "
    "currently in flight, one-off questions, or the assistant's own statements. A shipment "
    "id, a current stage and a today-only delay are lookups, not memories. Merge duplicates. "
    "For each fact give a short predicate and object, a one-sentence natural-language "
    "`text`, a `confidence` in [0,1], and an `importance` 1-10. " + IMPORTANCE_HINTS + " "
    'Return JSON of the form {"facts": [{"fact_type":..., "subject":..., "predicate":..., '
    '"object":..., "text":..., "confidence":..., "importance":...}, ...]}. Return an empty '
    "list if nothing durable was said."
)
"""The system prompt driving the cheap-model fact extractor.

The two clauses doing the real work are "durable, not transient" and "return an empty list
if nothing durable was said". Without the second, a model asked for facts will always find
some.
"""


def memory_subject_for(
    user_id: str | int | None, persona_id: str | None = None
) -> str | None:
    """Resolve the memory subject a run is scoped to (the app-level isolation key).

    **This function is THE adapter seam for memory scoping, and it is load-bearing twice.**
    It decides the key every memory query filters on, and it decides *what a memory is
    about* — which is a domain statement, not an infrastructure one.

    The decision here, deliberately: memory is scoped to the **shipper account** for the
    client-side persona and to the **individual user** for operator-side personas.

    The reason is that a durable cold-chain fact is almost always an *account* fact rather
    than a person fact. "Our Lisbon site cannot take dry ice", "this product must never be
    frozen", "goods-in closes at 15:00" are true of the account no matter which of its
    coordinators is on the call, and re-learning them from each new coordinator is exactly
    the failure long-term memory exists to prevent. Operator-side turns, by contrast, are
    personal working context — which lanes this lead watches, how they like a summary — and
    scoping those to an account would pool three leads' preferences into one incoherent
    profile.

    The two namespaces carry **different prefixes**, and that is not cosmetic. It guarantees
    a shipper key and an operator key can never collide even when the underlying ids do, so
    an operator's memory cannot be served to a client that happens to share their id.

    ``None`` means "no subject" (anonymous, single-shot): memory stays inert, which is a real
    answer rather than a failure.

    Args:
        user_id: The authenticated principal's id, or ``None``.
        persona_id: The active persona. Load-bearing here — it is what selects between the
            account namespace and the personal one.

    Returns:
        A stable subject key, or ``None`` to disable memory for this run.
    """
    if user_id is None or user_id == "":
        return None
    if persona_id == "shipper_client":
        return f"shipper:{user_id}"
    return f"user:{user_id}"


def render_profile(profile: dict[str, Any]) -> str:
    """Render the structured profile JSON as a compact prompt "human block".

    Args:
        profile: The stored profile data (a subset of :data:`PROFILE_FIELDS`).

    Returns:
        A short multi-line block, or an empty string when the profile is empty — an empty
        string rather than a header with nothing under it, so a subject the platform knows
        nothing about costs no tokens and makes no claim.
    """
    if not profile:
        return ""
    lines = ["Known about this shipper account:"]
    for name in PROFILE_FIELDS:
        value = profile.get(name)
        if value in (None, "", [], {}):
            continue
        label = name.replace("_", " ")
        rendered = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines) if len(lines) > 1 else ""


SKILL_HINTS: dict[str, str] = {
    # keyword found in the query  →  playbook filename WITHOUT the .md
    #
    # Every ``*.md`` in SKILLS_DIR appears as a value here. A playbook no literal can reach
    # is a playbook that can never be selected, and the conformance suite names it.
    "excursion": "handling_excursions",
    "excursions": "handling_excursions",
    "temperature": "handling_excursions",
    "out of range": "handling_excursions",
    "quarantine": "handling_excursions",
    "logger": "handling_excursions",
    "handling": "handling_excursions",
    "expedite": "expediting_shipments",
    "expediting": "expediting_shipments",
    "reroute": "expediting_shipments",
    "rerouting": "expediting_shipments",
    "late": "expediting_shipments",
    "delayed": "expediting_shipments",
    "at risk": "expediting_shipments",
    "shipments": "expediting_shipments",
}
"""Literal keyword → playbook filename (no extension).

Hoisted to a module constant on purpose: the conformance check reads the selector's compiled
constants **and** its module's top-level strings, so either placement works — but a constant
is easier to keep in step with the directory, and the check's own scar is that reading only
the function's constants made a tidying refactor silently empty the check.
"""


def select_skills(
    query: str, persona_id: str | None, available: list[str]
) -> list[str] | None:
    """Deterministically select the procedural skills a query needs.

    Returns a subset of ``available`` (filenames without ``.md``) by simple keyword match —
    cheap, offline, and reshapeable in seconds. Returning ``None`` means "no skill for this
    turn" and the core injects none.

    **Only ever returns names from ``available``.** The conformance check probes this
    function with every playbook name, every word inside those names, and every string it
    carries, and fails if it invents one.

    Args:
        query: The user query.
        persona_id: The active persona. Unused: both playbooks describe procedure that every
            persona here should follow, and a per-persona filter would mean the shipper
            contact is told a different procedure than the one actually being run.
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

"""Piece 8 of 10 — which specialists a turn may be handed to, and which team it may fan out across.

WHAT THIS FILE IS
    Two rosters, two mechanisms, and they are **not** interchangeable:

      * :func:`agent_roster` — the **supervisor's hand-off set**. Every role it names must
        have a handler node in the core graph, or a turn routes to a specialist that cannot
        answer it.
      * :func:`sub_agent_roster` — the **fan-out team** a wide turn is split across.
        Deliberately empty-able: an absent team means every turn runs single-lane rather than
        fanning out to agents the domain never declared.

    The core owns both mechanisms (the classifier, the hand-off protocol, the fan-out and the
    synthesis); the domain declares *which* specialists exist, how each is recognised, and
    what each one does.

THE CONTRACT (aegis.adapter.RosterModule / AgentRosterLike) — these must survive
    agent_roster() -> AgentRosterLike, sub_agent_roster() -> Sequence[SubAgentSpec]
    AgentRoster.specialists / .default_role / .roles() / .named()

    ``default_role`` is the load-bearing member: it is where a turn goes when no specialist
    matched, so a roster without one is a turn with nowhere to land.

╔══════════════════════════════════════════════════════════════════════════════╗
║ THE TRAP — the one real exception to "only the adapter changes"              ║
║                                                                              ║
║ The graph dispatches on a FIXED map in the core:                             ║
║                                                                              ║
║     SPECIALIST_NODES = {"qa": ..., "memory": ..., "team": ...}                ║
║                                                                              ║
║ A roster role that is not a key in that map **is not routable**. It falls     ║
║ back to `qa` and logs a warning — it does NOT raise — so an unroutable        ║
║ specialist looks exactly like a working one until you read the `routing`      ║
║ stream event and notice every turn landed in the same lane.                   ║
║                                                                              ║
║ This roster therefore declares exactly `qa` and `memory`, re-voiced for cold  ║
║ chain, and **no core edit was made**. Adding a genuinely new specialist would ║
║ require a handler node in the graph plus a `SPECIALIST_NODES` entry; that is  ║
║ a sanctioned exception and it must be reported, not done quietly.             ║
║                                                                               ║
║ `team` is never declared here. The router writes it itself when the depth     ║
║ classifier chooses fan-out.                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE OTHER TRAP — the half checklists used to omit
    Each sub-agent spec carries a ``tool_allowlist`` of **literal tool names**. The core
    intersects it with ``TOOL_REGISTRY``, so a name piece 4 renamed or deleted is **silently
    dropped** and the lane runs with fewer tools than you think — or none. Every name below
    is read out of :data:`~reference.adapter.tools.TOOL_REGISTRY` rather than typed, which
    makes that particular failure impossible here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aegis_ml.serve.tools import ML_TOOL_NAMES

from reference.adapter.tools import TOOL_REGISTRY

__all__ = [
    "AgentRoster",
    "RosterSpecialist",
    "SubAgentSpec",
    "agent_roster",
    "sub_agent_roster",
]


@dataclass(frozen=True)
class _StandaloneSubAgentSpec:
    """One fan-out lane, for when no Aegis checkout is on the path.

    Field names and defaults mirror ``aegis.agent.SubAgentSpec``, so a roster built from
    this class is read identically by a host that has the real one. ``model_role`` is a
    plain string here rather than the platform's ``ModelRole`` enum, which is
    value-compatible because that enum is a ``StrEnum``.

    Attributes:
        agent_id: Stable id stamped on every event this lane emits. Unique within a run.
        role: The lane's kind (``research`` | ``knowledge`` | ``data`` | ``policy``).
        label: Human label for the lane's card in the console.
        system_prompt: The **floor** prompt — the adapter's shipped string. The lane sends
            the prompt registry's active version when one exists, and this when it does not.
        prompt_key: The LLM-Ops registry key this lane's prompt is versioned under. Empty
            means the derived default ``subagent:<role>``.
        tool_allowlist: Tool names this lane may reach. Intersected with the persona's
            allowlist — never a widening of it.
        model_role: Which model tier the lane runs on.
        max_steps: Hard cap on loop iterations — the guarantee it terminates.
        timeout_s: The lane's wall clock.
    """

    agent_id: str
    role: str
    label: str
    system_prompt: str
    prompt_key: str = ""
    tool_allowlist: frozenset[str] = field(default_factory=frozenset)
    model_role: str = "cheap"
    max_steps: int = 4
    timeout_s: float = 45.0

    @property
    def registry_key(self) -> str:
        """Return the prompt key this lane's system prompt is versioned under."""
        return self.prompt_key or f"subagent:{self.role}"


def _resolve_subagent_spec() -> type[Any]:
    """Return the platform's ``SubAgentSpec`` class, or the standalone stand-in.

    The adapter must import with no Aegis checkout present, while still building native
    platform objects whenever the platform is there. The handled failure has exactly one
    cause (no importable ``aegis.agent``) and exactly one consequence, a field-compatible
    stand-in — a documented substitution, not a swallowed error.

    Returns:
        ``aegis.agent.SubAgentSpec`` when importable, else :class:`_StandaloneSubAgentSpec`.
    """
    try:
        from aegis.agent import SubAgentSpec as PlatformSubAgentSpec
    except ImportError:
        return _StandaloneSubAgentSpec
    return PlatformSubAgentSpec


SubAgentSpec: type[Any] = _resolve_subagent_spec()
"""The fan-out lane class this roster builds — the platform's when it is importable."""


@dataclass(frozen=True)
class RosterSpecialist:
    """One routable specialist the supervisor may hand a turn to.

    Attributes:
        role: Stable role id. It must be a key of the core graph's ``SPECIALIST_NODES``
            (``qa`` or ``memory``) unless a core edit has been made — and reported.
        description: One-line summary of what this specialist answers. Used both as the
            glass-box hand-off reason and as the menu handed to the cheap-model tiebreak when
            two specialists tie, so it is written to be *chosen between*, not to be pretty.
        keywords: Lower-case phrase hints the deterministic classifier looks for in the
            query. Longer, more specific phrases are the honest signal; a single common word
            captures turns that were never meant for this lane.
        is_default: Whether this is the fall-through when no keyword matched. Exactly one
            specialist carries it.
    """

    role: str
    description: str
    keywords: tuple[str, ...] = ()
    is_default: bool = False


@dataclass(frozen=True)
class AgentRoster:
    """The set of specialists the supervisor may route between."""

    specialists: tuple[RosterSpecialist, ...]

    @property
    def default_role(self) -> str:
        """Return the fall-through role id (the first ``is_default`` specialist)."""
        for spec in self.specialists:
            if spec.is_default:
                return spec.role
        # Defensive: an unmarked roster falls through to its first entry rather than to
        # nothing, because a turn with nowhere to land is worse than a turn in the wrong lane.
        return self.specialists[0].role if self.specialists else "qa"

    def roles(self) -> list[str]:
        """Return every routable role id in declaration order."""
        return [spec.role for spec in self.specialists]

    def named(self) -> list[RosterSpecialist]:
        """Return the non-default (keyword-matchable) specialists."""
        return [spec for spec in self.specialists if not spec.is_default]


_ROSTER = AgentRoster(
    specialists=(
        RosterSpecialist(
            role="qa",
            description=(
                "General question answering over the cold-chain SOPs, excursion policies "
                "and the shipment records themselves, with tools, ML decision-support and "
                "the human-approval gate. The default specialist — this is the full "
                "pipeline and where most turns should land."
            ),
            is_default=True,
        ),
        RosterSpecialist(
            role="memory",
            description=(
                "Answers questions about the subject themselves — what the assistant knows "
                "or remembers about this shipper account, its standing constraints and "
                "handling preferences, and their past interactions — directly from "
                "long-term memory, skipping retrieval and tools entirely."
            ),
            keywords=(
                # The generic phrases work in any domain and are worth keeping; the
                # domain-specific ones are added rather than substituted, because a user who
                # asks the generic question in a cold-chain deployment still means it.
                "what do you know about me",
                "what do you remember",
                "do you remember",
                "know about me",
                "remember about me",
                "what have i told you",
                "what did i tell you",
                "my past interactions",
                "our past conversations",
                "what do you know about our account",
                "what did i tell you about our sites",
                "our standing instructions",
                "our usual packout",
            ),
        ),
    )
)


def agent_roster() -> AgentRoster:
    """Return the domain's :class:`AgentRoster` — the specialists the core may route to.

    The single adapter contract the supervisor consumes.

    Returns:
        The roster: a default ``qa`` lane and a keyword-matched ``memory`` lane.
    """
    return _ROSTER


# ─────────────────────────────────────────────────────────────────────────────
# The fan-out team
#
# Four lanes, because the platform default for ``max_parallel_agents`` is 4 and a roster
# shorter than the cap IS the cap. Roster ORDER is priority order: a narrow team takes the
# first ``width`` entries, so the two lanes that most often decide a cold-chain answer —
# what the corpus says, and what the records and the model say — come first.
#
# Three of the four carry no write tool at all, on purpose. A research, knowledge or policy
# lane reads and reports; giving a read-only remit a write tool is how a fan-out acquires
# four routes to a consequential action instead of one.
#
# ``model_role`` is left at the core default (cheap) for every lane. That is most of why
# fanning out four ways is affordable at all — the expensive model is spent once, on the
# synthesis.
# ─────────────────────────────────────────────────────────────────────────────


def _lane_tools(*names: str) -> frozenset[str]:
    """Return the subset of ``names`` that actually exists in the tool registry.

    The core intersects a lane's allowlist with ``TOOL_REGISTRY`` and drops unknown names in
    silence, so a stale allowlist is a lane with fewer tools and no error. Filtering here
    does not prevent that — it makes it *visible*, because a name that survives this filter
    is a name the registry really has, and a name that does not is one this function can be
    made to shout about the moment anyone wants it to.

    Args:
        names: Candidate tool names.

    Returns:
        The names present in :data:`~reference.adapter.tools.TOOL_REGISTRY`.
    """
    return frozenset(name for name in names if name in TOOL_REGISTRY)


_SUB_AGENTS: tuple[Any, ...] = (
    SubAgentSpec(
        agent_id="knowledge",
        role="knowledge",
        label="Knowledge agent",
        system_prompt=(
            "You are the knowledge agent on a pharmaceutical cold-chain team. Answer your "
            "sub-task from the shared retrieved context you were given: quote or paraphrase "
            "what the SOP, policy or runbook actually says, name the document id it came "
            "from, and say explicitly when the corpus does not cover the question rather "
            "than filling the gap from general knowledge. Thresholds matter here — if a "
            "document states a qualified range or a review cadence, give the number."
        ),
    ),
    SubAgentSpec(
        agent_id="data",
        role="data",
        label="Shipment data agent",
        system_prompt=(
            "You are the shipment data agent on a pharmaceutical cold-chain team. Answer "
            "your sub-task from the concrete consignment records, using your tools to look "
            "them up, to ask the model for a spoilage-risk prediction with its interval, "
            "and to annotate a timeline. Always look a shipment up before you name it; "
            "never work from an id that appeared only in the question. Any consequential "
            "change is a proposal: request it as a tool call and it will be routed to a "
            "human for approval — you never take one yourself."
        ),
        # The only lane with a write remit, and the writes it may REQUEST are still gated:
        # the quarantine is HIGH-risk, so this lane can propose it and nothing else. The note
        # is LOW and is the one thing it may just do. The ML tools are LOW and read-only, so
        # the lane can bring evidence to its proposal instead of asserting one.
        tool_allowlist=_lane_tools(
            "find_shipments", "add_shipment_note", "quarantine_shipment", *ML_TOOL_NAMES
        ),
    ),
    SubAgentSpec(
        agent_id="policy",
        role="policy",
        label="Quality policy agent",
        system_prompt=(
            "You are the quality policy agent on a pharmaceutical cold-chain team. Judge "
            "your sub-task against the organisation's excursion-handling, release and "
            "escalation rules: state which rule applies, what it requires, and where the "
            "situation in front of you departs from it. Be explicit about the difference "
            "between a measured excursion and a predicted risk — only the first triggers a "
            "deviation. Recommend; never act."
        ),
    ),
    SubAgentSpec(
        agent_id="research",
        role="research",
        label="Research agent",
        system_prompt=(
            "You are the research agent on a pharmaceutical cold-chain team. Establish what "
            "is externally true about your sub-task — current GDP guidance, carrier or "
            "airport disruption, seasonal ambient conditions on a lane, anything the "
            "internal corpus would not know. Report what you found and, just as plainly, "
            "what you could not establish. Never present an assumption as a finding."
        ),
    ),
)


def sub_agent_roster() -> tuple[Any, ...]:
    """Return the domain's sub-agent roster — the team a wide turn fans out across.

    The second adapter contract the supervisor consumes. The core reads it defensively (an
    absent hook, a raising hook, or an empty roster all mean "no team, run single-pass"),
    clamps every entry's bounds down to the tenant's agent config, and takes the first
    ``width`` entries — so roster order is the priority order for a narrow team.

    Returns:
        Four lanes: knowledge, shipment data, quality policy, research.
    """
    return _SUB_AGENTS

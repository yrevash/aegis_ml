"""Piece 8 of 10 — which specialists a turn may be handed to, and which team it may fan out across.

WHAT YOU WRITE HERE
    Two rosters, two mechanisms, and they are **not** interchangeable:

      * :func:`agent_roster` — the **supervisor's hand-off set**. Every role it names
        must have a handler node in the core graph, or a turn routes to a specialist
        that cannot answer it.
      * :func:`sub_agent_roster` — the **fan-out team** a wide turn is split across.
        Deliberately empty-able: an absent team means every turn runs single-lane
        rather than fanning out to agents the domain never declared. The core ships no
        roster of its own precisely so that a host cannot accidentally inherit one.

    The core owns both mechanisms (the classifier, the hand-off protocol, the fan-out
    and the synthesis); the domain declares *which* specialists exist, how each is
    recognised, and what each one does.

THE CONTRACT (aegis.adapter.RosterModule / AgentRosterLike) — these must survive
    agent_roster() -> AgentRosterLike, sub_agent_roster() -> Sequence[SubAgentSpec]
    AgentRoster.specialists / .default_role / .roles() / .named()

    ``default_role`` is the load-bearing member: it is where a turn goes when no
    specialist matched, so a roster without one is a turn with nowhere to land.

╔══════════════════════════════════════════════════════════════════════════════╗
║ THE TRAP — the one real exception to "only the adapter changes"              ║
║                                                                              ║
║ The graph dispatches on a FIXED map in `aegis/src/aegis/agent/graph.py`:      ║
║                                                                              ║
║     SPECIALIST_NODES: dict[str, str] = {                                      ║
║         "qa":     "recall_memory",  # the full retrieve→plan→gate→act pipeline ║
║         "memory": "answer_memory",  # answers from long-term memory only       ║
║         "team":   "plan_team",      # router-written fan-out; NOT a roster role║
║     }                                                                         ║
║                                                                              ║
║ A roster role that is not a key in that map **is not routable**. It falls     ║
║ back to `qa` and logs a warning — **it does NOT raise**, so an unroutable     ║
║ specialist looks exactly like a working one until you read the `routing`      ║
║ stream event and notice every turn landed in the same lane.                   ║
║                                                                              ║
║   * Re-voice `qa` and `memory` for your domain: change descriptions and       ║
║     keywords freely, KEEP the two role strings. No core edit. This is what    ║
║     you should do under time pressure.                                        ║
║   * Adding a genuinely new specialist requires a **core edit** — a handler     ║
║     node in the graph plus a `SPECIALIST_NODES` entry. That is a sanctioned    ║
║     exception and it MUST BE REPORTED, not done quietly.                      ║
║   * Never declare `team` in a roster. The router writes it itself when the    ║
║     depth classifier chooses fan-out.                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE OTHER TRAP — the half checklists used to omit
    Each :class:`~aegis.agent.SubAgentSpec` carries a ``tool_allowlist`` of **literal
    tool names**. The core intersects it with ``TOOL_REGISTRY``, so a name piece 4
    renamed or deleted is **silently dropped** and the lane runs with fewer tools than
    you think — or none. Re-point every allowlist at names that exist in your registry,
    and re-voice each spec's ``label`` and ``system_prompt`` while you are there: both
    are read by the model and shown on screen.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        --pyargs aegis.conformance --aegis-adapter app.adapter -q -k roster)
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        tests/agent/test_router.py -q)
    Then run a query and read the ``routing`` stream event.
"""

from __future__ import annotations

from dataclasses import dataclass

from aegis.agent import SubAgentSpec


@dataclass(frozen=True)
class RosterSpecialist:
    """One routable specialist the supervisor may hand a turn to.

    Attributes:
        role: Stable role id. It must be a key of the core graph's ``SPECIALIST_NODES``
            (``qa`` or ``memory``) unless you have made — and reported — the core edit.
        description: One-line summary of what this specialist answers. Used both as the
            glass-box hand-off reason and as the menu handed to the cheap-LLM tiebreak
            when two specialists tie, so write it to be *chosen between*, not to be
            pretty.
        keywords: Lower-case phrase hints the deterministic classifier looks for in the
            query. Longer, more specific phrases are the honest signal; a single common
            word captures turns that were never meant for this lane.
        is_default: Whether this is the fall-through when no keyword matched. Exactly
            one specialist must carry it.
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
        # Defensive: an unmarked roster falls through to its first entry.
        return self.specialists[0].role if self.specialists else "qa"

    def roles(self) -> list[str]:
        """Return every routable role id in declaration order."""
        return [spec.role for spec in self.specialists]

    def named(self) -> list[RosterSpecialist]:
        """Return the non-default (keyword-matchable) specialists."""
        return [spec for spec in self.specialists if not spec.is_default]


# ─────────────────────────────────────────────────────────────────────────────
# The supervisor's roster
#
# TODO(domain): re-voice both entries. KEEP the role strings "qa" and "memory".
# ─────────────────────────────────────────────────────────────────────────────

_ROSTER = AgentRoster(
    specialists=(
        RosterSpecialist(
            role="qa",
            description=(
                "TODO(domain): general question answering over <your corpus> and "
                "<your records>, with tools and the human-approval gate. The default "
                "specialist — this is the full pipeline and where most turns should "
                "land."
            ),
            is_default=True,
        ),
        RosterSpecialist(
            role="memory",
            description=(
                "Answers questions about the user themselves — what the assistant "
                "knows or remembers about them, their stated preferences, and their "
                "past interactions — directly from long-term memory, skipping "
                "retrieval and tools entirely."
            ),
            keywords=(
                # TODO(domain): these generic phrases work in any domain and are worth
                # keeping. Add domain-specific ones ("what did I tell you about my
                # <entity>") rather than replacing them.
                "what do you know about me",
                "what do you remember",
                "do you remember",
                "know about me",
                "remember about me",
                "what have i told you",
                "what did i tell you",
                "my past interactions",
                "our past conversations",
            ),
        ),
    )
)


def agent_roster() -> AgentRoster:
    """Return the domain's :class:`AgentRoster` — the specialists the core may route to.

    The single adapter contract the supervisor consumes. Swapping the domain means
    editing the roster above and keeping this function's shape stable.
    """
    return _ROSTER


# ─────────────────────────────────────────────────────────────────────────────
# The fan-out team
#
# TODO(domain): four lanes, because the platform default for ``max_parallel_agents``
# is 4 and a roster shorter than the cap IS the cap. Roster ORDER is priority order:
# a narrow team takes the first ``width`` entries.
#
# Three of the four carry no tools at all, on purpose. A research, knowledge or policy
# lane reads and reports; giving a read-only remit a write tool is how a fan-out
# acquires four routes to a consequential action instead of one.
#
# ``model_role`` is left at the core default (CHEAP) for every lane. That is most of
# why fanning out four ways is affordable at all — the expensive model is spent once,
# on the synthesis.
# ─────────────────────────────────────────────────────────────────────────────

_SUB_AGENTS: tuple[SubAgentSpec, ...] = (
    SubAgentSpec(
        agent_id="research",
        role="research",
        label="Research agent",
        system_prompt=(
            "TODO(domain): 'You are the research agent on a <domain> team.' Establish "
            "what is externally true about your sub-task — current policy, recent "
            "regulatory or vendor changes, anything the internal corpus would not "
            "know. Report what you found and, just as plainly, what you could not "
            "establish. Never present an assumption as a finding."
        ),
    ),
    SubAgentSpec(
        agent_id="knowledge",
        role="knowledge",
        label="Knowledge agent",
        system_prompt=(
            "TODO(domain): 'You are the knowledge agent on a <domain> team.' Answer "
            "your sub-task from the shared retrieved context you were given: quote or "
            "paraphrase what the corpus actually says, name the passage it came from, "
            "and say explicitly when the corpus does not cover the question rather "
            "than filling the gap from general knowledge."
        ),
    ),
    SubAgentSpec(
        agent_id="data",
        role="data",
        label="Data agent",
        system_prompt=(
            "TODO(domain): 'You are the data agent on a <domain> team.' Answer your "
            "sub-task from the concrete records, using your tools to read and annotate "
            "them. Any consequential change is a proposal: request it as a tool call "
            "and it will be routed to a human for approval — you never take one "
            "yourself."
        ),
        # The only lane with a write remit, and the writes it may REQUEST are still
        # gated: the stage change is HIGH-risk, so the lane can propose it and nothing
        # else. The note is LOW and is the one thing it may just do.
        #
        # TODO(domain): every name here must be a key of piece 4's TOOL_REGISTRY. The
        # core intersects this set with the registry and drops unknown names in
        # silence — a stale allowlist is a lane with no tools and no error.
        tool_allowlist=frozenset({"advance_item_stage", "add_item_note"}),
    ),
    SubAgentSpec(
        agent_id="policy",
        role="policy",
        label="Policy agent",
        system_prompt=(
            "TODO(domain): 'You are the policy agent on a <domain> team.' Judge your "
            "sub-task against the organisation's escalation, target and approval "
            "rules: state which rule applies, what it requires, and where the "
            "situation in front of you departs from it. Recommend; never act."
        ),
    ),
)


def sub_agent_roster() -> tuple[SubAgentSpec, ...]:
    """Return the domain's sub-agent roster — the team a wide turn fans out across.

    The second adapter contract the supervisor consumes. The core reads it defensively
    (an absent hook, a raising hook, or an empty roster all mean "no team, run
    single-pass"), clamps every entry's bounds down to the tenant's ``AgentConfig``, and
    takes the first ``width`` entries — so roster order is the priority order for a
    narrow team.

    Swapping the domain means rewriting the entries above and keeping this function's
    shape stable.
    """
    return _SUB_AGENTS


__all__ = [
    "AgentRoster",
    "RosterSpecialist",
    "agent_roster",
    "sub_agent_roster",
]

"""The adapter registry — the single clean surface the core imports.

**This file is not one of the ten pieces.** It is the interface contract: the core
(agent, retrieval, ml, memory, guardrails, forecast, mcp, api) depends on the domain
*only* through the names bound here. Swapping a domain means rewriting the ten pieces
underneath and keeping this surface resolvable.

    | # | Piece | File |
    |---|-------------------------|-----------------|
    | 1 | Data schema             | `schema.py`     |
    | 2 | ML features + target    | `ml_spec.py`    |
    | 3 | Synthetic generator     | `generator.py`  |
    | 4 | Tool definitions        | `tools.py`      |
    | 5 | Personas                | `personas.py`   |
    | 6 | Prompts                 | `prompts.py`    |
    | 7 | Memory contract         | `memory_spec.py`|
    | 8 | Agent roster            | `roster.py`     |
    | 9 | Domain corpus           | `corpus/`       |
    | 10| Procedural skills       | `skills/`       |

WHAT YOU EDIT HERE
    **You will edit this file and its ``__all__``, and that is correct.** Piece 1
    replaces the record types it re-exports and piece 2 renames the latent function it
    re-exports, so leaving ``__all__`` untouched is impossible. What must stay stable is
    not the list of names, it is **the contract**: ``aegis.adapter.DomainAdapter`` and
    its sub-Protocols, plus the host-bound names listed below.

THE CONTRACT (aegis.adapter.DomainAdapter) — eleven members
    Nine module members — ``schema``, ``ml_spec``, ``generator``, ``tools``,
    ``personas``, ``prompts``, ``memory_spec``, ``roster``, ``corpus`` — plus
    :data:`DOMAIN_ID` and :data:`DOMAIN_DESCRIPTION`.

    Check it, do not count files::

        (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -c "
        import app.adapter
        from aegis.adapter import DomainAdapter, missing_members
        print('missing:', missing_members(app.adapter))
        print('satisfies:', isinstance(app.adapter, DomainAdapter))
        ")

THE TRAP — a member can be on disk and still be missing
    **Protocol members are attributes of the package, and a submodule becomes an
    attribute only once something imports it.** An adapter whose ``__init__.py`` never
    touches ``memory_spec`` does not *have* a ``memory_spec`` member, however present
    the file is on disk — and ``missing_members`` will say so while every test that
    imports the module by path passes. This bit the reference adapter itself: it
    satisfied nine of the ten pieces with the tenth sitting unreachable beside them.

    Hence the explicit ``from app.adapter import memory_spec, ml_spec, schema`` below.
    The other six submodules become attributes as a side effect of the ``from
    app.adapter.<piece> import ...`` lines — but that is an accident of what those
    lines happen to need, so if you ever stop importing a name from one of them, import
    the module itself instead of deleting the line.

THE SECOND TRAP — the description is a control input, not metadata
    :data:`DOMAIN_DESCRIPTION` is wired straight into the guardrails as the topical
    rail's ``allowed_topics``. **A vague description is a loose rail.** "A system for
    managing things" admits every off-topic question ever asked; a description naming
    the entities, the actions and the audience is a rail that actually holds.

VERIFY
    (cd backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \\
        --pyargs aegis.conformance --aegis-adapter app.adapter -q)
"""

from __future__ import annotations

# The three modules imported AS MODULES — see "THE TRAP" above. ``memory_spec`` is the
# load-bearing one: ``app.memory`` installs the module object itself as the process-wide
# default spec (``set_default_spec(app.adapter.memory_spec)``), so the module — not any
# name inside it — is what that consumer binds to.
from app.adapter import memory_spec, ml_spec, schema
from app.adapter.corpus import load_seed_corpus
from app.adapter.generator import (
    DOMAIN_SERIES_LABEL,
    DOMAIN_SERIES_UNIT,
    GeneratorConfig,
    assess_quality,
    domain_series_events,
    generate_synthetic,
    generate_synthetic_sync,
)
from app.adapter.memory_spec import FACT_TYPES, memory_subject_for
from app.adapter.ml_spec import (
    FEATURE_NAMES,
    FEATURES,
    TARGET,
    describe_prediction,
    feature_matrix,
    features_for_item,
    latent_cycle_time_hours,
    training_frame,
)
from app.adapter.personas import (
    DEFAULT_PERSONA_ID,
    PERSONA_BY_ROLE,
    PERSONAS,
    Persona,
    ScopeKind,
    get_persona,
    persona_for_role,
)
from app.adapter.prompts import (
    PLATFORM_FLOOR,
    SYSTEM_PROMPTS,
    render_platform_floor,
    render_system_prompt,
)
from app.adapter.roster import (
    AgentRoster,
    RosterSpecialist,
    agent_roster,
    sub_agent_roster,
)
from app.adapter.schema import (
    Document,
    Operator,
    Party,
    SyntheticDataset,
    WorkItem,
)
from app.adapter.tools import (
    ALLOWLIST,
    TOOL_REGISTRY,
    InMemoryRecordStore,
    ToolActionResult,
    ToolContext,
    ToolNotAllowedError,
    UnknownToolError,
    is_allowed,
    run_tool,
    tool_definitions_for,
    tools_for,
)

DOMAIN_ID = "TODO_domain_id"
"""Stable machine id of the loaded domain.

TODO(domain): lower_snake, specific, and **not** the shipped ``service_request_
management``. It is written into artifacts, model cards and conformance output, so it
is how a trained model is told apart from one trained against a different problem.
The ``TODO_`` prefix is here so a forgotten edit is visible in every one of those.
"""

DOMAIN_DESCRIPTION = (
    "TODO(domain): one paragraph describing what this domain is about — the entities, "
    "the actions taken on them, and who is served. THIS IS THE GUARDRAILS' TOPICAL "
    "RAIL: it is passed straight through as `allowed_topics`, so write it as the set "
    "of questions this deployment should answer. Name the nouns ('shipments, carriers "
    "and cold-chain excursions'), name the verbs ('rerouting, holding and releasing'), "
    "and name the audience. A vague description is a loose rail — 'a system for "
    "managing things' admits every off-topic question ever asked."
)
"""One-paragraph description of the domain — and the guardrails' ``allowed_topics``.

Not metadata. A **control input**. See the trap in the module docstring.
"""

# ─────────────────────────────────────────────────────────────────────────────
# The host-bound surface.
#
# Verified by grep against `backend/src/app/` — these are the names the CORE actually
# imports, and the reason this file exists. Renaming any of them without re-pointing
# its consumer is a runtime ImportError at host start-up, not a lint error.
#
#   from app.adapter          → DEFAULT_PERSONA_ID, DOMAIN_DESCRIPTION,
#                               DOMAIN_SERIES_LABEL, DOMAIN_SERIES_UNIT,
#                               GeneratorConfig, InMemoryRecordStore, TARGET,
#                               TOOL_REGISTRY, ToolContext, agent_roster,
#                               domain_series_events, generate_synthetic_sync,
#                               get_persona, is_allowed, load_seed_corpus,
#                               ml_spec, memory_spec, persona_for_role,
#                               render_platform_floor, render_system_prompt,
#                               run_tool, sub_agent_roster, tool_definitions_for,
#                               training_frame
#   from app.adapter.tools    → ALLOWLIST, AuditFn, RecordStore, TOOL_REGISTRY,
#                               ToolActionResult, ToolContext, ToolNotAllowedError,
#                               UnknownToolError, is_allowed
#   from app.adapter.memory_spec → FACT_TYPES, memory_subject_for
#   from app.adapter.roster      → sub_agent_roster
#   importlib.resources.files("app.adapter.corpus")  ← the corpus package PATH
#
# The names imported from a *submodule* path are re-exported here as well, so a
# consumer may reach them either way. `AuditFn` and `RecordStore` are the exception:
# they are Protocols the MCP server imports from `app.adapter.tools` directly, and they
# stay there rather than widening this surface.
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ALLOWLIST",
    "DEFAULT_PERSONA_ID",
    "DOMAIN_DESCRIPTION",
    "DOMAIN_ID",
    "DOMAIN_SERIES_LABEL",
    "DOMAIN_SERIES_UNIT",
    "FACT_TYPES",
    "FEATURES",
    "FEATURE_NAMES",
    "PERSONAS",
    "PERSONA_BY_ROLE",
    "PLATFORM_FLOOR",
    "SYSTEM_PROMPTS",
    "TARGET",
    "TOOL_REGISTRY",
    "AgentRoster",
    "Document",
    "GeneratorConfig",
    "InMemoryRecordStore",
    "Operator",
    "Party",
    "Persona",
    "RosterSpecialist",
    "ScopeKind",
    "SyntheticDataset",
    "ToolActionResult",
    "ToolContext",
    "ToolNotAllowedError",
    "UnknownToolError",
    "WorkItem",
    "agent_roster",
    "assess_quality",
    "describe_prediction",
    "domain_series_events",
    "feature_matrix",
    "features_for_item",
    "generate_synthetic",
    "generate_synthetic_sync",
    "get_persona",
    "is_allowed",
    "latent_cycle_time_hours",
    "load_seed_corpus",
    "memory_spec",
    "memory_subject_for",
    "ml_spec",
    "persona_for_role",
    "render_platform_floor",
    "render_system_prompt",
    "run_tool",
    "schema",
    "sub_agent_roster",
    "tool_definitions_for",
    "tools_for",
    "training_frame",
]

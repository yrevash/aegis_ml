"""The adapter registry — the single clean surface the platform imports.

**This file is not one of the ten pieces.** It is the interface contract: the core (agent,
retrieval, ml, memory, guardrails, forecast, mcp, api) depends on the domain *only* through
the names bound here. Swapping a domain means rewriting the ten pieces underneath and
keeping this surface resolvable.

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

THE CONTRACT (aegis.adapter.DomainAdapter) — eleven members
    Nine module members — ``schema``, ``ml_spec``, ``generator``, ``tools``, ``personas``,
    ``prompts``, ``memory_spec``, ``roster``, ``corpus`` — plus :data:`DOMAIN_ID` and
    :data:`DOMAIN_DESCRIPTION`.

    Check it, do not count files::

        PYTHONPATH=/Users/yrevash/aegis/aegis/src .venv/bin/python -c "
        import reference.adapter as adapter
        from aegis.adapter import DomainAdapter, missing_members
        print('missing:', missing_members(adapter))
        print('satisfies:', isinstance(adapter, DomainAdapter))
        "

THE TRAP — a member can be on disk and still be missing
    **Protocol members are attributes of the package, and a submodule becomes an attribute
    only once something imports it.** An adapter whose ``__init__.py`` never touches
    ``memory_spec`` does not *have* a ``memory_spec`` member, however present the file is on
    disk — and ``missing_members`` will say so while every test that imports the module by
    path passes. This bit the shipped reference adapter itself: it satisfied nine of the ten
    pieces with the tenth sitting unreachable beside them.

    Hence the explicit ``from reference.adapter import memory_spec, ml_spec, schema`` below.
    The other six submodules become attributes as a side effect of the
    ``from reference.adapter.<piece> import ...`` lines — but that is an accident of what
    those lines happen to need, so if a name ever stops being imported from one of them,
    import the module itself rather than deleting the line.

THE SECOND TRAP — the description is a control input, not metadata
    :data:`DOMAIN_DESCRIPTION` is wired straight into the guardrails as the topical rail's
    ``allowed_topics``. **A vague description is a loose rail.** "A system for managing
    things" admits every off-topic question ever asked; a description naming the entities,
    the actions and the audience is a rail that actually holds.
"""

from __future__ import annotations

# The three modules imported AS MODULES — see "THE TRAP" above. ``memory_spec`` is the
# load-bearing one: a host installs the module object itself as the process-wide default
# memory spec, so the module — not any name inside it — is what that consumer binds to.
from reference.adapter import memory_spec, ml_spec, schema
from reference.adapter.corpus import load_seed_corpus
from reference.adapter.generator import (
    DOMAIN_SERIES_LABEL,
    DOMAIN_SERIES_UNIT,
    DatasetQualityReport,
    GeneratorConfig,
    assess_quality,
    domain_series_events,
    generate_synthetic,
    generate_synthetic_sync,
)
from reference.adapter.memory_spec import FACT_TYPES, memory_subject_for
from reference.adapter.ml_spec import (
    EXCURSION_PROBLEM,
    FEATURE_NAMES,
    FEATURES,
    PROBLEM,
    SECONDARY_TARGET,
    TARGET,
    describe_prediction,
    excursion_frame,
    feature_matrix,
    features_for_shipment,
    latent_spoilage_risk,
    training_frame,
)
from reference.adapter.personas import (
    DEFAULT_PERSONA_ID,
    PERSONA_BY_ROLE,
    PERSONAS,
    Persona,
    ScopeKind,
    get_persona,
    persona_for_role,
)
from reference.adapter.prompts import (
    PLATFORM_FLOOR,
    SYSTEM_PROMPTS,
    render_platform_floor,
    render_system_prompt,
)
from reference.adapter.roster import (
    AgentRoster,
    RosterSpecialist,
    agent_roster,
    sub_agent_roster,
)
from reference.adapter.schema import (
    SCHEMA_VERSION,
    Carrier,
    Document,
    Facility,
    SensorReading,
    Shipment,
    SyntheticDataset,
)
from reference.adapter.tools import (
    ALLOWLIST,
    TOOL_REGISTRY,
    AuditFn,
    InMemoryRecordStore,
    RecordStore,
    ToolActionResult,
    ToolContext,
    ToolNotAllowedError,
    ToolSpec,
    UnknownToolError,
    is_allowed,
    run_tool,
    tool_definitions_for,
    tools_for,
)

DOMAIN_ID = "cold_chain_logistics"
"""Stable machine id of the loaded domain.

Written into artifacts, model cards and conformance output, so it is how a model trained
against this problem is told apart from one trained against a different one. It matches
``ml_spec.PROBLEM.domain_id`` exactly, and the demo asserts that rather than trusting it.
"""

DOMAIN_DESCRIPTION = (
    "Pharmaceutical cold-chain logistics. This deployment answers questions about "
    "temperature-controlled shipments of vaccines, biologics, small-molecule product and "
    "diagnostic kits moving from origin depots through transfer hubs to clinics, cold "
    "stores and hospital pharmacies — the carriers that move them, the packouts that "
    "protect them, the data loggers that evidence them, and the facilities that receive "
    "them. It predicts a consignment's spoilage risk and whether it suffered a temperature "
    "excursion, explains those predictions from the lane's booked characteristics, and "
    "supports four actions on a shipment: finding it, annotating its timeline, rerouting it "
    "to a different journey shape or carrier, and quarantining or releasing it for quality "
    "review. It serves cold-chain logistics leads, cold-chain quality auditors, and the "
    "pharmaceutical shippers whose product is moving. Questions about anything other than "
    "these shipments, their lanes, their temperature records and the SOPs and policies "
    "governing them are out of scope."
)
"""One-paragraph description of the domain — and the guardrails' ``allowed_topics``.

Not metadata. A **control input**. It names the nouns (shipments, carriers, packouts,
loggers, facilities), the verbs (find, annotate, reroute, quarantine, release) and the
audience (logistics leads, quality auditors, shippers), and it closes the set explicitly,
which is what makes it a rail that holds rather than a paragraph that sounds tidy.
"""

# ─────────────────────────────────────────────────────────────────────────────
# The host-bound surface.
#
# These are the names a deployed Aegis host actually imports, and the reason this file
# exists. Renaming any of them without re-pointing its consumer is a runtime ImportError at
# host start-up, not a lint error.
#
#   from <adapter>          → DEFAULT_PERSONA_ID, DOMAIN_DESCRIPTION, DOMAIN_SERIES_LABEL,
#                             DOMAIN_SERIES_UNIT, GeneratorConfig, InMemoryRecordStore,
#                             TARGET, TOOL_REGISTRY, ToolContext, agent_roster,
#                             domain_series_events, generate_synthetic_sync, get_persona,
#                             is_allowed, load_seed_corpus, ml_spec, memory_spec,
#                             persona_for_role, render_platform_floor, render_system_prompt,
#                             run_tool, sub_agent_roster, tool_definitions_for,
#                             training_frame
#   from <adapter>.tools    → ALLOWLIST, AuditFn, RecordStore, TOOL_REGISTRY,
#                             ToolActionResult, ToolContext, ToolNotAllowedError,
#                             UnknownToolError, is_allowed
#   from <adapter>.memory_spec → FACT_TYPES, memory_subject_for
#   from <adapter>.roster      → sub_agent_roster
#   importlib.resources.files("<adapter>.corpus")  ← the corpus package PATH
#
# ``AuditFn`` and ``RecordStore`` are re-exported here as well as living on ``tools``: the
# MCP server imports them from the submodule directly, and a consumer that reaches for them
# on the package should find them rather than discovering the distinction the hard way.
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ALLOWLIST",
    "DEFAULT_PERSONA_ID",
    "DOMAIN_DESCRIPTION",
    "DOMAIN_ID",
    "DOMAIN_SERIES_LABEL",
    "DOMAIN_SERIES_UNIT",
    "EXCURSION_PROBLEM",
    "FACT_TYPES",
    "FEATURES",
    "FEATURE_NAMES",
    "PERSONAS",
    "PERSONA_BY_ROLE",
    "PLATFORM_FLOOR",
    "PROBLEM",
    "SCHEMA_VERSION",
    "SECONDARY_TARGET",
    "SYSTEM_PROMPTS",
    "TARGET",
    "TOOL_REGISTRY",
    "AgentRoster",
    "AuditFn",
    "Carrier",
    "DatasetQualityReport",
    "Document",
    "Facility",
    "GeneratorConfig",
    "InMemoryRecordStore",
    "Persona",
    "RecordStore",
    "RosterSpecialist",
    "ScopeKind",
    "SensorReading",
    "Shipment",
    "SyntheticDataset",
    "ToolActionResult",
    "ToolContext",
    "ToolNotAllowedError",
    "ToolSpec",
    "UnknownToolError",
    "agent_roster",
    "assess_quality",
    "describe_prediction",
    "domain_series_events",
    "excursion_frame",
    "feature_matrix",
    "features_for_shipment",
    "generate_synthetic",
    "generate_synthetic_sync",
    "get_persona",
    "is_allowed",
    "latent_spoilage_risk",
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

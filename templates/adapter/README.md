# `templates/adapter/` — the ten pieces, as annotated skeletons

This directory is a **fill-in-the-blanks copy of a working Aegis domain adapter**. On
the day, you rewrite these files from the Domain Brief and `rsync` the result into
`backend/src/app/adapter/`. Everything else in `app/` is the stable core and does not
change.

The authoritative procedure is [`SKILL.md`](../../../aegis/SKILL.md) at the Aegis
repository root (the `retarget-aegis` skill). This README is the local **map**; the
day-of running order with its verify commands is [`_CHECKLIST.md`](_CHECKLIST.md).
Neither is a second procedure — where any of the three disagree, `SKILL.md` is right.

## The ten pieces

Eight Python modules plus two content directories. `__init__.py` is **not** one of
them — it is the registry, the interface the core imports.

| # | Piece | What it defines | Lands in |
|---|-------|-----------------|----------|
| 1 | **Data schema** | The entities and enums of the new world — the vocabulary every other piece shares | `schema.py` |
| 2 | **ML features + target** | What the spine predicts, on which features, and the latent ground-truth signal | `ml_spec.py` |
| 3 | **Synthetic generator** | Seeded records + LLM prose + a templated fallback, labelled from piece 2's latent function | `generator.py` |
| 4 | **Tool definitions** | The real actions, typed, MCP-shaped, risk-tiered and allowlisted per persona | `tools.py` |
| 5 | **Personas** | Who is served; persona → data scope + tool allowlist + RBAC role mapping | `personas.py` |
| 6 | **Prompts** | Who the agent is, per persona — plus the platform floor no tenant may edit | `prompts.py` |
| 7 | **Memory contract** | What a durable fact is, whose memory a turn touches, how the profile reads | `memory_spec.py` |
| 8 | **Agent roster** | Which specialists the supervisor may route to, and the fan-out team | `roster.py` |
| 9 | **Domain corpus** | Seed Markdown documents ingested into the graph/vector store | `corpus/` |
| 10 | **Procedural skills** | How-to-act playbooks, selected per query by `memory_spec.select_skills` | `skills/` |

Ten pieces map to **nine** `DomainAdapter` members: `skills/` has none of its own
because it is already named by `memory_spec.SKILLS_DIR`. The other two members are
`DOMAIN_ID` and `DOMAIN_DESCRIPTION`, both in `__init__.py`.

## How to read a template

Every module opens with the same four-part header, and it is worth reading before
touching the code underneath:

- **WHAT YOU WRITE HERE** — the piece's job, and what a filled-in version contains.
- **THE CONTRACT** — the exact names that must survive, and which `aegis.adapter`
  sub-Protocol requires them.
- **THE TRAP** — the failure this piece is known to produce. Every one of them is a
  defect that actually shipped, and every one is **silent**: the suites stay green, the
  demo looks fine, and the symptom arrives on stage.
- **VERIFY** — the command that proves this piece.

In the body, `TODO(domain):` marks every blank. The placeholder world (a widget line:
`WorkItem`, `Party`, `Operator`, kinds `alpha`/`beta`/`gamma`/`delta`) is deliberately
generic — it exists so the files parse, so the shapes are visible, and so nothing here
can be mistaken for the shipped `service_request_management` domain. **Delete all of
it.** A `TODO(domain):` string surviving into a demo is the intended failure mode: loud
and on screen, rather than plausible and wrong.

## These files import `app.*`, on purpose

They are written for their **destination**, `backend/src/app/adapter/`, so
`from app.adapter.schema import ...` and `from app.api.schemas import RiskLevel`
resolve once they land there. They therefore do **not** import standalone from
`aegis_ml/` — `python -c "import templates.adapter"` will fail, and that is expected.

What is checked here is that they parse and lint:

```bash
cd /Users/yrevash/aegis_ml && python3 -c \
  "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('templates/adapter/**/*.py', recursive=True)]"
ruff check templates
```

`templates/*` carries `ANN`, `D` and `F821` ruff ignores (see `pyproject.toml`) because
a skeleton legitimately references names that do not exist yet. Write them well anyway
— the filled-in version inherits whatever discipline the skeleton had.

## The three things that must survive by name

Everything else is yours to re-voice. These are not:

1. **`SyntheticDataset`** — the container the generator returns and the ML spine reads.
2. **The sub-Protocol member names** — `FEATURE_NAMES`, `TARGET`, `training_frame`,
   `describe_prediction`, `TOOL_REGISTRY`, `ALLOWLIST`, `run_tool`, `PERSONAS`,
   `DEFAULT_PERSONA_ID`, `PERSONA_BY_ROLE`, `persona_for_role`,
   `DOMAIN_SERIES_LABEL`, `DOMAIN_SERIES_UNIT`, `domain_series_events`,
   `agent_roster`, `sub_agent_roster`, `load_seed_corpus`, and every `memory_spec`
   member.
3. **The host-bound extras** — `ToolContext`, `ToolActionResult`, `InMemoryRecordStore`,
   `RecordStore`, `AuditFn`, `UnknownToolError`, `ToolNotAllowedError`,
   `GeneratorConfig`, `Persona`, `ScopeKind`, `TARGET`, `training_frame`. The full
   verified list, with the module that imports each one, is in `__init__.py`'s own
   comment block.

## Where the ML goes in

`tools.py` carries a marked **SLOT** showing where `aegis_ml.serve.tools` registers its
`predict_outcome` / `explain_prediction` / `whatif_scenario` / `forecast_series` /
`check_model_health` specs. That is the whole integration: ML reaches the agent through
the **tool registry**, not through a graph edit. It needs no core change, and it keeps
the platform's rule intact — **ML informs; it never gates.** The human gate fires on a
tool's risk tier, and every ML tool is LOW and read-only.

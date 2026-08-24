"""The reference domain — a working adapter, checked against the platform's own contract.

``reference/`` exists so every other test in this suite runs against a realistic domain
rather than an invented one. That is only worth anything if the domain is genuinely
complete, so this module checks it the way the platform does: through
``aegis.adapter.missing_members`` and ``aegis.ml.spec.resolve_spec``, not by counting files.

The trap ``resolve_spec`` sets is the reason for the ``FALLBACK_SPEC`` test. Its own code is:

    if not features or not target:
        return FALLBACK_SPEC          # four columns of generated noise

A misspelled ``FEATURE_NAMES`` does not raise. It trains the trustworthy spine on noise and
serves the result as domain evidence, and the only native symptom is ``distinct=False`` on
the last line of ``python -m app.ml``.

Tests that need the Aegis platform skip when it is not importable; the rest run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import reference.adapter as adapter

from aegis_ml.serve.tools import ML_TOOL_NAMES

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


# ── the platform contract ─────────────────────────────────────────────────────


@pytest.mark.aegis
def test_adapter_has_no_missing_members() -> None:
    """All eleven Protocol members are reachable ON THE PACKAGE, not merely on disk.

    A submodule becomes an attribute only once something imports it, so an adapter whose
    ``__init__`` never touches ``memory_spec`` does not *have* that member however present
    the file is.
    """
    pytest.importorskip("aegis", reason="needs PYTHONPATH=/Users/yrevash/aegis/aegis/src")
    from aegis.adapter import missing_members

    assert missing_members(adapter) == []


@pytest.mark.aegis
def test_adapter_satisfies_the_domain_adapter_protocol() -> None:
    """``isinstance(adapter, DomainAdapter)`` — the check the host performs at start-up."""
    pytest.importorskip("aegis", reason="needs PYTHONPATH=/Users/yrevash/aegis/aegis/src")
    from aegis.adapter import DomainAdapter

    assert isinstance(adapter, DomainAdapter)


@pytest.mark.aegis
def test_resolve_spec_does_not_fall_back_to_generated_noise() -> None:
    """``resolve_spec`` must resolve to THIS domain, never to ``FALLBACK_SPEC``."""
    pytest.importorskip("aegis", reason="needs PYTHONPATH=/Users/yrevash/aegis/aegis/src")
    from aegis.ml.spec import FALLBACK_SPEC, resolve_spec

    resolved = resolve_spec(adapter.ml_spec)

    assert resolved is not FALLBACK_SPEC
    assert resolved.target == adapter.TARGET.name == "spoilage_risk_pct"
    assert list(resolved.features) == list(adapter.FEATURE_NAMES)
    assert set(resolved.features) != set(FALLBACK_SPEC.features)
    assert len(resolved.features) == 10


# ── the spec the pipelines read ───────────────────────────────────────────────


def test_domain_id_matches_the_problem(problem) -> None:
    """One id, asserted rather than trusted: it is written into every artifact."""
    assert adapter.DOMAIN_ID == problem.domain_id == "cold_chain_logistics"


def test_problem_is_re_exported_not_rebuilt(problem) -> None:
    """``reference.problem.PROBLEM`` IS ``ml_spec.PROBLEM`` — one object, three consumers."""
    assert problem is adapter.ml_spec.PROBLEM
    assert problem is adapter.PROBLEM


def test_feature_names_preserve_declaration_order(problem) -> None:
    """Order is preserved into ``FEATURE_NAMES``, which the one-hot subset is derived from."""
    assert list(adapter.FEATURE_NAMES) == [f.name for f in problem.features]


def test_every_categorical_feature_declares_its_levels(problem) -> None:
    """The contract cannot check an open set; the spec refuses one, and this proves it held."""
    for feature in problem.features:
        if feature.dtype == "categorical":
            assert feature.levels, f"{feature.name} is categorical with no levels"


def test_regression_target_carries_a_unit(problem) -> None:
    """``describe_prediction`` renders the conformal interval with it; bare floats are useless."""
    assert problem.target.unit


def test_domain_description_is_a_usable_guardrail(problem) -> None:
    """``DOMAIN_DESCRIPTION`` is wired in as the topical rail's ``allowed_topics``.

    A vague description is a loose rail, so it has to name the nouns, the verbs and the
    audience, and close the set.
    """
    description = adapter.DOMAIN_DESCRIPTION.lower()
    assert len(description) > 400, "a one-line description admits every off-topic question"
    for noun in ("shipment", "carrier", "packout", "logger", "facilit"):
        assert noun in description, f"the rail does not name {noun!r}"
    for verb in ("rerout", "quarantin", "annotat"):
        assert verb in description, f"the rail does not name the action {verb!r}"
    assert "out of scope" in description, "the rail never closes the set"


# ── piece 10: procedural skills ───────────────────────────────────────────────


def test_every_skill_file_on_disk_is_reachable_from_select_skills() -> None:
    """THE SILENT TRAP: a playbook no literal can reach is never injected and nothing warns.

    Playbooks are selected by filename through ``SKILL_HINTS``. Add or rename a ``.md``
    without updating that table and ``select_skills`` returns ``None`` for it forever, the
    core injects no skill, and the agent answers without its procedure.
    """
    from reference.adapter.memory_spec import SKILL_HINTS, SKILLS_DIR, select_skills

    skills_dir = Path(SKILLS_DIR)
    on_disk = sorted(p.stem for p in skills_dir.glob("*.md"))
    assert on_disk, f"no skill playbooks found under {skills_dir}"

    reachable_by_table = set(SKILL_HINTS.values())
    unreachable = set(on_disk) - reachable_by_table
    assert not unreachable, (
        f"these playbooks exist on disk but no SKILL_HINTS literal can reach them: "
        f"{sorted(unreachable)}"
    )

    reached: set[str] = set()
    for keyword, expected in SKILL_HINTS.items():
        selected = select_skills(f"please help with {keyword}", None, on_disk)
        assert selected is not None, f"keyword {keyword!r} selected nothing"
        assert expected in selected, f"keyword {keyword!r} did not select {expected!r}"
        reached.update(selected)

    assert reached == set(on_disk), f"unreached playbooks: {sorted(set(on_disk) - reached)}"


def test_skill_hints_never_name_a_playbook_that_does_not_exist() -> None:
    """The other direction: a renamed file leaves a table entry pointing at nothing."""
    from reference.adapter.memory_spec import SKILL_HINTS, SKILLS_DIR

    on_disk = {p.stem for p in Path(SKILLS_DIR).glob("*.md")}
    dangling = set(SKILL_HINTS.values()) - on_disk
    assert not dangling, f"SKILL_HINTS names playbooks that are not on disk: {sorted(dangling)}"


def test_select_skills_returns_none_for_an_unrelated_query() -> None:
    """No keyword matched means no playbook, not an arbitrary one."""
    from reference.adapter.memory_spec import SKILLS_DIR, select_skills

    available = [p.stem for p in Path(SKILLS_DIR).glob("*.md")]
    assert select_skills("what is the capital of France", None, available) is None


# ── piece 4: tools. ML informs, never gates ───────────────────────────────────


def test_every_ml_tool_is_low_risk_and_read_only() -> None:
    """The house rule: an ML tool may inform a decision, never take one.

    A prediction that can quarantine a consignment is a model making an operational
    decision without a human in the loop. Every ML tool must therefore be LOW risk,
    read-only, non-destructive and idempotent — and the *write* tools next to them must not
    be, or the classification means nothing.
    """
    registry = adapter.TOOL_REGISTRY
    ml_tools = [name for name in ML_TOOL_NAMES if name in registry]
    assert ml_tools, f"no ML tool from {list(ML_TOOL_NAMES)} is registered in this domain"

    for name in ml_tools:
        spec = registry[name]
        assert str(spec.risk.value if hasattr(spec.risk, "value") else spec.risk) == "low", (
            f"ML tool {name!r} is {spec.risk!r} risk"
        )
        assert spec.read_only is True, f"ML tool {name!r} is not read-only"
        assert spec.destructive is False, f"ML tool {name!r} is marked destructive"
        assert spec.idempotent is True, f"ML tool {name!r} is not idempotent"


def test_the_domains_write_tools_are_not_low_risk_read_only() -> None:
    """The contrast that gives the previous test meaning."""
    registry = adapter.TOOL_REGISTRY
    write_tools = [name for name, spec in registry.items() if not spec.read_only]
    assert write_tools, "a domain with no write action cannot demonstrate an approval gate"

    quarantine = registry["quarantine_shipment"]
    assert quarantine.read_only is False
    assert quarantine.destructive is True
    risk = getattr(quarantine.risk, "value", quarantine.risk)
    assert str(risk) == "high"


def test_every_ml_tool_is_on_the_allowlist() -> None:
    """A registered tool no persona may call is dead weight the roster cannot reach."""
    allowlisted = {name for names in adapter.ALLOWLIST.values() for name in names}
    for name in ML_TOOL_NAMES:
        if name in adapter.TOOL_REGISTRY:
            assert name in allowlisted, f"{name} is registered but on no persona's allowlist"


def test_allowlist_never_names_an_unregistered_tool() -> None:
    """An allowlist entry for a tool that does not exist fails at call time, in front of a user."""
    for persona, names in adapter.ALLOWLIST.items():
        unknown = set(names) - set(adapter.TOOL_REGISTRY)
        assert not unknown, f"persona {persona!r} is allowed unknown tools {sorted(unknown)}"


# ── piece 9: corpus ───────────────────────────────────────────────────────────


def test_seed_corpus_loads_and_is_non_trivial() -> None:
    """Retrieval with an empty corpus answers from the model's own priors and sounds fine."""
    documents = adapter.load_seed_corpus()
    assert len(documents) >= 3
    for document in documents:
        text = getattr(document, "text", None) or getattr(document, "content", str(document))
        assert len(text) > 200, "a corpus document too short to ground an answer"


# ── the generator ─────────────────────────────────────────────────────────────


def test_training_frame_matches_its_own_declared_contract(frame, problem) -> None:
    """The domain's generator satisfies the pandera contract derived from its own spec."""
    from aegis_ml.contracts.frames import validate

    validated = validate(frame, problem)
    assert len(validated) == len(frame)


def test_excursion_frame_matches_its_own_declared_contract(
    excursion_frame, excursion_problem
) -> None:
    """The secondary classification problem is complete too, not a stub beside the first."""
    from aegis_ml.contracts.frames import validate

    validate(excursion_frame, excursion_problem)
    assert set(excursion_frame[excursion_problem.target.name].unique()) <= set(
        excursion_problem.target.levels
    )

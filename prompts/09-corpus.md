# PROMPT 09 · Piece 9 — `corpus/`

---

## Role

You are writing **piece 9 of 10**: the seed knowledge the platform retrieves over before anything is ingested. A loader module plus 3–6 Markdown documents.

---

## Inputs

- `DOMAIN_BRIEF.md` §12 (corpus), §2 (entities and enums).
- Piece 1's `Document` model and its `kind` / `category` enums.
- Reference: `/Users/yrevash/aegis/backend/src/app/adapter/corpus/`.

## Output files

```
/Users/yrevash/aegis_ml/reference/adapter/corpus/__init__.py
/Users/yrevash/aegis_ml/reference/adapter/corpus/<doc>.md      ×3–6
```

---

## The contract to satisfy

```python
@runtime_checkable
class CorpusModule(Protocol):
    def load_seed_corpus(self) -> list[Any]: ...
```

> A domain with **no** corpus is legal and honest — the loader may return an empty list, and retrieval then returns no candidates instead of reaching for someone else's documents. **What is not legal is *absent*:** a missing loader is the difference between "this domain ships no seed knowledge" and "the seed knowledge silently failed to load."

---

## The trap

> **These are `*.md` data files, not Python.** A `cp -r` sync overwrites the modules and **leaves the reference domain's three documents in place**: `kb_request_closure.md`, `policy_escalation.md`, `runbook_login_failures.md`.
>
> Retrieval will ingest and serve them alongside yours, and the agent will cite the wrong domain's policy on stage. Nothing raises.

Use `rsync -a --delete` (or `robocopy /MIR`), and verify afterwards:

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/*.md
```

**And if retrieval already ingested them**, deleting the files is not enough — reindex, or drop the collection and re-seed.

**Second trap.** Conformance check #13 (`test_seed_corpus_records_carry_identity_and_chunk`) requires **a stable unique id** and **a body that actually produces chunks**. Its scar: *"the ingestion chunker shipped producing chunks with no tenant and a `doc_id` that did not join `documents.id`. Retrieval still returned passages; every citation on them resolved to nothing, and the answer looked fully sourced."* A two-line document is text in the index that nothing can be traced back to.

---

## What to write

### 1. `corpus/__init__.py` — the loader

```python
"""Seed corpus — piece 9 of 10 of the adapter.

The hand-written documents retrieval reads before anything real is ingested. They are
Markdown with YAML-style frontmatter, discovered from this package directory at call
time, so adding a document is dropping a file — no registry, no import.

Every record is stamped ``source="seed"`` so a seeded passage can be told apart from an
ingested one in a citation.
"""

from __future__ import annotations

from importlib import resources

from reference.adapter.schema import Category, Document, DocumentKind

__all__ = ["load_seed_corpus"]

_FRONTMATTER_DELIM = "---"


def _parse_tags(raw: str) -> list[str]:
    """Parse a ``[a, b, c]`` tag list."""
    return [t.strip() for t in raw.strip().strip("[]").split(",") if t.strip()]


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``---``-delimited frontmatter from the body.

    Returns:
        ``({key: value}, body)``. A file with no frontmatter yields ``({}, text)``
        rather than raising, so a document is never lost to a formatting slip.
    """
    ...


def _to_document(fields: dict[str, str], body: str, fallback_id: str) -> Document:
    """Build one Document, falling back to the filename for id and title."""
    category = Category(fields["category"]) if fields.get("category") else None
    return Document(
        id=fields.get("id") or fallback_id,
        kind=DocumentKind(fields.get("kind", DocumentKind.KB_ARTICLE.value)),
        title=fields.get("title") or fallback_id,
        body=body.strip(),
        category=category,
        tags=_parse_tags(fields.get("tags", "")),
        source="seed",
    )


def load_seed_corpus() -> list[Document]:
    """Load the seed documents as typed domain records (possibly empty).

    Reads every ``*.md`` beside this module through ``importlib.resources`` rather than
    ``Path(__file__).parent``, so the corpus survives being packaged into a wheel or a
    zipimport. Returns them sorted by id, so ingestion order is deterministic and a
    re-seed produces the same chunk ids.
    """
    docs: list[Document] = []
    for entry in resources.files(__package__).iterdir():
        if not entry.name.endswith(".md"):
            continue
        fields, body = _parse_frontmatter(entry.read_text(encoding="utf-8"))
        docs.append(_to_document(fields, body, entry.name.removesuffix(".md")))
    return sorted(docs, key=lambda d: d.id)
```

**Keep `importlib.resources`**, not `Path(__file__)`. And **keep the sort** — deterministic ingestion order means a re-seed produces the same chunk ids, which is what makes a citation stable across restarts.

### 2. The documents

**Frontmatter keys** — use exactly these, `---`-delimited, `key: value`:

| Key | Required | Fallback |
|---|---|---|
| `id` | yes in practice | the filename |
| `kind` | no | `kb_article` |
| `title` | yes in practice | the filename |
| `category` | no | `None`; must be a value of your `Category` enum |
| `tags` | no | `[]`; `[a, b, c]` form |

```markdown
---
id: doc-seed-0002
kind: policy
title: Theatre overrun and list-containment policy
category: scheduling
tags: [overrun, escalation, containment, consent]
---

# Theatre overrun and list-containment policy

## Scope

This policy applies to all elective theatre lists across the four main theatres. It does
not cover emergency lists, which run under the on-call escalation procedure.

## When a list is declared "at risk"

A list is at risk when the cumulative overrun across completed cases exceeds 45 minutes,
or when the delay-risk model predicts an overrun above 30 minutes for a case in the
second half of the list with the lower bound of its 90% interval above zero.

The lower bound matters. A predicted overrun of 40 minutes with an interval of
[-5, 85] is a different operational statement from 40 minutes with [22, 58]: the first
does not justify moving a patient, and the second does.

## Containment ladder

Interventions are applied in order of least disruption. Do not skip a rung.

1. **Shorten the turnaround.** ...
2. **Re-sequence within the list.** ...
3. **Move a case to another theatre.** ...
4. **Cancel and rebook.** Requires the coordinator's approval and a consultant's
   agreement. Consent must be re-taken if the rebooked date is more than 28 days out.

## What must be recorded

Every intervention is recorded against the theatre day with the reason, the predicted
overrun that triggered it, and who approved it.
```

**Content requirements:**

- **3–6 documents.** Fewer gives retrieval nothing to rank; more is work you do not need.
- **Several hundred words each, with real `##` headings.** The chunker splits on structure. A short document produces one chunk and retrieval cannot rank within it.
- **Cover different `kind`s** — a policy, a runbook, a knowledge article, an FAQ — so `DocumentKind` is exercised and the console's kind filter shows more than one value.
- **Write the rules your tools enforce.** A `resequence_list` tool that is HIGH-risk should have a policy explaining *why* it needs approval. That is the demo: the agent cites the policy, then the gate fires, then a human approves.
- **Name at least one thing the model must not do.** Guardrail stories need a document to point at.
- **Reference the ML target by name at least once**, as the overrun policy above does. It ties the retrieval story and the ML story together in one citation.
- **Use unique, stable ids.** `doc-seed-0001`, `doc-seed-0002`, … Do not reuse the reference domain's ids if you have changed what they mean.

---

## Verify

```bash
cd /Users/yrevash/aegis_ml
uv run python -c "
from reference.adapter.corpus import load_seed_corpus

docs = load_seed_corpus()
assert docs, 'no seed documents loaded'
ids = [d.id for d in docs]
assert len(ids) == len(set(ids)), f'duplicate ids: {ids}'
for d in docs:
    assert d.id and d.id.strip(), 'a document has no stable id'
    assert len(d.body) > 400, f'{d.id}: body too short to chunk ({len(d.body)} chars)'
    assert d.source == 'seed'
    print(f'{d.id:16s} {d.kind.value:12s} {str(d.category):14s} {len(d.body):5d} chars  {d.title}')
print('kinds:', sorted({d.kind.value for d in docs}))
print('tags :', sorted({t for d in docs for t in d.tags}))
"
```

And confirm nothing from the old domain survived the sync:

```bash
ls /Users/yrevash/aegis/backend/src/app/adapter/corpus/*.md
# must NOT contain kb_request_closure.md, policy_escalation.md, runbook_login_failures.md

grep -rniE "service request|support agent|case note|resolution hours" \
  /Users/yrevash/aegis/backend/src/app/adapter/corpus/
```

After the sync:

```bash
(cd /Users/yrevash/aegis/backend && PYTHONPATH=src:../aegis/src .venv/bin/python -m pytest \
    --pyargs aegis.conformance --aegis-adapter app.adapter -q -k corpus)
```

### Checklist

- [ ] `load_seed_corpus()` exists, takes no arguments, returns `list[Document]`.
- [ ] It reads via `importlib.resources`, not `Path(__file__)`.
- [ ] It returns documents **sorted by id**, each stamped `source="seed"`.
- [ ] 3–6 `*.md` files, each several hundred words with real `##` headings.
- [ ] Every document has a unique, stable `id`.
- [ ] Frontmatter uses `id`, `kind`, `title`, `category`, `tags` and nothing else.
- [ ] Every `category` value is a member of your `Category` enum; every `kind` a member of `DocumentKind`.
- [ ] At least two different `kind`s appear.
- [ ] At least one document explains why a HIGH-risk tool needs approval.
- [ ] At least one document names the ML target.
- [ ] The reference domain's three documents are **gone** from the synced directory.

---

## Next

`prompts/10-skills.md` — then go back to `prompts/07-memory-spec.md` and reconcile the `hints` table.

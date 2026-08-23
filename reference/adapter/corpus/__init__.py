"""Piece 9 of 10 — the seed knowledge the platform retrieves over before anything is ingested.

WHAT THIS FILE IS
    Three small Markdown documents with flat YAML-style frontmatter, plus the loader that
    parses them into typed :class:`~reference.adapter.schema.Document` records. The retrieval
    layer ingests these alongside the generated corpus from piece 3, so they are what the
    agent can cite on turn one, before a single real document exists.

    Files are read through :mod:`importlib.resources`, so the loader is portable — no
    absolute paths, works from a wheel or a checkout. **Adding a document needs no code
    change**: drop a new ``*.md`` in this directory with the same frontmatter keys.

    A domain with no corpus is legal and honest — return an empty list and retrieval returns
    no candidates instead of reaching for someone else's documents. What is not legal is an
    *absent* loader: that is the difference between "this domain ships no seed knowledge" and
    "the seed knowledge silently failed to load".

THE CONTRACT (aegis.adapter.CorpusModule) — this name must survive
    load_seed_corpus() -> list[Document]

THE TRAP
    The conformance suite requires every returned record to carry a **unique, non-empty id**
    and a **body the chunker can actually split**. Both failures are invisible at runtime:
    chunks are written with the record's id as their ``doc_id``, so a record with no id — or
    two records sharing one — produces chunks whose citation resolves to the wrong document
    or to nothing at all. Retrieval keeps returning passages, and a retrieved passage with a
    dead citation reads on screen exactly like a sourced one.

    Second trap, on a retarget day: **``rsync -a --delete``, never ``cp -r``.** A plain copy
    leaves the previous domain's ``*.md`` files sitting in this directory, and this loader
    will happily ingest them — so retrieval serves the old domain's policies under the new
    domain's name, cited and confident.
"""

from __future__ import annotations

import importlib.resources as resources

from reference.adapter.schema import Document, DocumentKind, ProductClass

__all__ = ["load_seed_corpus"]


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``---``-delimited frontmatter block from the Markdown body.

    Deliberately a hand-rolled parser rather than a YAML dependency: the frontmatter is a
    flat ``key: value`` block by design, and an adapter should not add a dependency to read
    six lines.

    Args:
        text: The raw file contents.

    Returns:
        ``(fields, body)`` — frontmatter keys mapped to raw string values, and everything
        after the frontmatter.
    """
    if not text.lstrip().startswith("---"):
        return {}, text
    stripped = text.lstrip()
    _, _, rest = stripped.partition("---")
    block, _, body = rest.partition("---")
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields, body.strip()


def _parse_tags(raw: str) -> list[str]:
    """Parse a ``[a, b, c]`` frontmatter list into a list of strings.

    Args:
        raw: The raw frontmatter value.

    Returns:
        The tags, whitespace-trimmed, empty entries dropped.
    """
    inner = raw.strip().strip("[]")
    return [tag.strip() for tag in inner.split(",") if tag.strip()]


def _to_document(fields: dict[str, str], body: str, fallback_id: str) -> Document:
    """Build a :class:`~reference.adapter.schema.Document` from frontmatter and body.

    Args:
        fields: Parsed frontmatter keys.
        body: The Markdown body.
        fallback_id: The filename, used when the frontmatter omits ``id`` — so a document
            still gets a unique, stable id rather than an empty string that fails the
            conformance check.

    Returns:
        The typed document record.

    Raises:
        ValueError: When ``kind`` or ``product_scope`` names a value the schema's enums do
            not carry. Loud on purpose: a silently coerced document is one that will be
            retrieved under the wrong scope for the rest of the deployment's life.
    """
    scope_raw = fields.get("product_scope")
    return Document(
        id=fields.get("id", fallback_id),
        kind=DocumentKind(fields.get("kind", DocumentKind.SOP.value)),
        title=fields.get("title", fallback_id),
        body=body,
        product_scope=ProductClass(scope_raw) if scope_raw else None,
        tags=_parse_tags(fields.get("tags", "")),
        source="seed",
    )


def load_seed_corpus() -> list[Document]:
    """Load every seed ``*.md`` document in this package as a typed record.

    Returns:
        The seed documents, sorted by id so ingest order is deterministic across processes
        and platforms (``iterdir`` order is not).
    """
    documents: list[Document] = []
    package = resources.files(__package__)
    for entry in package.iterdir():
        if entry.name.endswith(".md"):
            fields, body = _parse_frontmatter(entry.read_text(encoding="utf-8"))
            documents.append(_to_document(fields, body, fallback_id=entry.name))
    return sorted(documents, key=lambda document: document.id)

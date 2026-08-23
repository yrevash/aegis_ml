---
id: doc-seed-0001
kind: guide
kind_scope: alpha
tags: [example, format, replace-me]
title: EXAMPLE — the seed-document format
---

# EXAMPLE — the seed-document format

**Delete this file.** It exists to show the frontmatter contract that
`corpus/__init__.py` parses, and it will be ingested into retrieval and cited by the
agent if it is still here on the day.

## The frontmatter contract

Five keys, all flat `key: value`, delimited by `---` above and below. The loader is a
hand-rolled parser, not YAML — nesting, quotes and multi-line values are not supported
and will be read literally.

| Key | Required | Notes |
|---|---|---|
| `id` | yes, in practice | Unique and stable. Conformance check #13 fails on a missing or duplicated id, because chunks are written with it as their `doc_id` and a broken citation reads on screen exactly like a sourced one. Defaults to the filename if omitted, which is unique but not stable across a rename. |
| `kind` | yes | Must be a value of the schema's `DocumentKind` enum — `guide`, `policy`, `faq`, `runbook` in the template. An unknown value raises at load time, loudly, which is correct. |
| `kind_scope` | no | A value of the domain's subject-area enum, or omitted for a document that spans all of them. |
| `tags` | no | A flat `[a, b, c]` list. Retrieval tags — lower-case, no spaces inside a tag. |
| `title` | yes | The heading a citation shows. Write it as the thing a reader would search for. |

## What belongs in a seed document

Body text, in Markdown, long enough that the chunker produces at least one chunk — a
one-line body is a record check #13 rejects. Three to eight short paragraphs is the
useful range.

Write the documents the agent needs to answer correctly on turn one, before anything
real has been ingested: the policy it must not contradict, the procedure it must
follow, the thresholds and escalation rules it will otherwise invent. These are the
documents a demo cites, so they are also the documents a reviewer reads.

Prefer specifics over prose. A policy that states "escalate above three failed
attempts" is checkable; one that says "escalate when appropriate" gives the model
permission to decide, which is exactly what a grounded answer is supposed to avoid.

## Naming

`<kind>_<subject>.md` — `policy_escalation.md`, `runbook_intake_failures.md`,
`guide_item_closure.md`. The filename is not read by the loader (the `id` is), but it
is what a human scans when the corpus grows past five files.

## On the day

Replace every file in this directory with your own, using `rsync -a --delete` rather
than `cp -r`. A plain copy leaves the previous domain's documents behind and this
loader will ingest them without complaint — retrieval then serves the old domain's
policies under the new domain's name.

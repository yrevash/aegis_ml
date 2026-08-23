# Skill: EXAMPLE — the procedural-playbook format

**Use when** the query mentions the word `example`, `escalate` or `close` — because
those are the three keys pointing at this file in `memory_spec.SKILL_HINTS`. Delete
this playbook and those three rows together once you have written real ones.

A playbook is *how to act*, not *what is true*. Facts belong in `corpus/` and are
retrieved; procedure belongs here and is selected. If you find yourself writing
"the SLA is 24 hours", that is a corpus document. If you are writing "confirm the
record before acting, then propose the change rather than asserting it", that is here.

## The rules of the format

- **One playbook, one situation.** The selector is a keyword match; a playbook covering
  three situations is selected for all three and is wrong for two of them.
- **Filename is the identity.** `select_skills` returns filenames without `.md`, and
  the core injects the file whose stem it returns. Renaming this file without editing
  `SKILL_HINTS` makes it permanently unselectable, in silence — the selector just
  returns `None`, the turn answers without its procedure, and nothing appears in the
  trace to say a skill was wanted and missed.
- **Lower_snake_case names**, matching what the hints table spells:
  `closing_items.md`, `de_escalation.md`, `handling_disputes.md`.
- **Short.** Five to ten bullets. This text is injected into the prompt on every turn
  it is selected for, so every line is paid for repeatedly.
- **Imperative and checkable.** "Confirm X before Y" beats "be careful about Y".

## The structure

```
# Skill: <what this is>

Use when <the situation, in the words a user would actually type>.

- <step or rule>
- <step or rule>
- <the gate rule — see below>
- <what to do when the precondition fails>
```

## The one bullet every playbook should carry

Whatever else it says, a playbook that touches an action needs the gate rule, because
the model reads the playbook more attentively than the system prompt:

> Any consequential change is a **write action** — propose it as a tool call and let
> the approval gate route it to a human. Never assert that it is done.

## Worked shape (replace all of it)

```
# Skill: Closing a work item

Use when the party asks for their item to be closed, completed, or marked done.

- Confirm the specific record — id, current stage, opened date — with the lookup tool
  before acting. Never work from an id that appeared only in the question.
- State the closure rule plainly: an item closes once the party confirms the outcome,
  or after 30 days with no reply while it is held.
- Closing is a write action — propose it, do not assert it as done.
- If the party is on the top tier, note that their account contact should be copied on
  the outcome.
- Close by confirming the expected timeline for the approval itself.
```

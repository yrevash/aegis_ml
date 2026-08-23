# Skill: Handling a temperature excursion

Use when the query mentions an excursion, a temperature out of range, a logger reading, a
quarantine, or a consignment that may have gone out of its qualified band.

- Confirm the specific consignment first — id, stage, product class, packaging, carrier —
  with `find_shipments`. Never work from an id that appeared only in the question.
- Separate what was **measured** from what was **predicted**, out loud. A logged excursion
  opens a deviation; a high predicted spoilage risk does not. Say which one you are quoting.
- State the qualified range that applies to this product class before saying whether it was
  breached: 2–8 °C refrigerated, −25 to −15 °C frozen, 15–25 °C controlled room temperature,
  2–30 °C and never frozen for diagnostic kits (POL-CC-204).
- Report cumulative time out of range, not just the peak. The stability assessment is run
  against cumulative time.
- Quarantine is a **write action** — propose it as a tool call with a recorded reason and let
  the approval gate route it to a human. Never assert that a consignment has been held.
- If any part of the journey has no logger coverage, say so plainly: an unassessable gap is a
  reject under POL-CC-204, not a judgement call.
- Escalate to the quality lead — not just the logistics lead — when the consignment is
  patient-specific, when vaccine or biologic product has excursed for more than 6 hours
  cumulative, or when this is the second excursion on the same lane within 30 days.
- Close by naming who must decide next and what evidence they will need.

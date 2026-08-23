---
id: doc-seed-0002
kind: policy
tags: [excursion, quarantine, release, deviation, escalation]
title: POL-CC-204 — Temperature excursion handling, quarantine and release
---

# POL-CC-204 — Temperature excursion handling, quarantine and release

## What counts as an excursion

An excursion is **any recorded time outside the product's qualified range**, however brief.
The qualified ranges this network ships against are:

- 2 °C to 8 °C for refrigerated vaccine and most biologics;
- −25 °C to −15 °C for frozen biologics on dry ice;
- 15 °C to 25 °C for controlled-room-temperature small molecules;
- 2 °C to 30 °C for diagnostic kits, which additionally **must never be allowed to freeze**.

A predicted spoilage risk is not an excursion. A model output, however high and however well
calibrated, is a forecast about a consignment; an excursion is a measurement taken from a
logger. Only the second opens a deviation. Confusing the two either buries real deviations
under model noise or lets a real one go unrecorded because the model happened to be
optimistic, and both have been observed.

## Immediate actions on a detected excursion

1. **Quarantine the consignment.** It is held at its current location and does not enter
   picking, dispensing or onward shipment. Quarantine requires a recorded reason; a hold
   nobody can explain is a hold nobody else can lift.
2. **Preserve the logger.** Download the record before the shipper is broken down. A logger
   separated from its packout can no longer evidence anything, and the consignment then has
   to be written off on the presumption of loss.
3. **Record the excursion window** — start, end, peak deviation and cumulative time out of
   range. Cumulative time is what the stability assessment is run against, not the peak.
4. **Notify the shipper within 4 hours** of detection, with the window and the current hold
   status. Notification is not a decision; it does not commit either party to an outcome.

## Release criteria

A quarantined consignment may only be released by a quality auditor, against a written
assessment naming the stability data relied on. Three outcomes:

- **Release** — the cumulative time out of range is inside the product's documented
  stability budget and the peak deviation is inside its excursion limit.
- **Release with reduced shelf life** — inside the excursion limit but consuming a
  documented part of the stability budget. The revised expiry must be written onto the
  consignment before release.
- **Reject** — outside the excursion limit, or the logger record is incomplete for any part
  of the journey. An incomplete record is a reject, not a judgement call: an unmonitored gap
  cannot be assessed and must be assumed to be the worst case.

## Unmonitored lanes

Where a carrier publishes no telemetry interval, the lane is **unmonitored**. Unmonitored
lanes are permitted only for diagnostic kits and small molecules, never for vaccine or
biologic product, and they must be reviewed against the planned schedule every four hours
rather than on telemetry. A consignment that arrives on an unmonitored lane with no logger
record is a reject under the criteria above.

## Escalation

Escalate to the quality lead, not merely to the logistics lead, when any of these hold: the
consignment is patient-specific; the product is vaccine or biologic and the excursion exceeds
6 hours cumulative; two or more consignments on the same lane and carrier have excursed
within 30 days; or a release-with-reduced-shelf-life decision would take remaining shelf life
below 60 days.

Repeat excursions on one lane are a lane problem, not a consignment problem. Two in 30 days
requires the lane to be re-qualified under SOP-CC-101 before further product is booked on it.

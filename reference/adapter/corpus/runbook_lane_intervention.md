---
id: doc-seed-0003
kind: runbook
tags: [reroute, intervention, risk, carrier, monitoring]
title: RB-CC-311 — Intervening on a lane the model has flagged
---

# RB-CC-311 — Intervening on a lane the model has flagged

## When this runbook applies

A consignment is in flight, or booked and not yet dispatched, and the spoilage-risk model has
returned a prediction the logistics lead considers actionable. This runbook is about what to
do with that number. It is not about excursions: a *recorded* excursion goes straight to
POL-CC-204 and this document does not apply.

## Read the interval, not the point estimate

Every prediction arrives with a conformal interval at the requested coverage. Act on the
interval.

- A prediction of 42% with an interval of [28%, 56%] and a prediction of 42% with an interval
  of [39%, 45%] are different findings, and only the second supports a confident
  intervention.
- Long lanes carry systematically wider intervals, because more of the journey is outside
  anyone's direct control. A wide interval on a 120-hour multi-leg lane is the model being
  honest, not the model being broken.
- If the prediction lists imputed features, read which ones. A consignment whose telemetry
  interval was imputed is being scored partly on the average lane rather than on itself, and
  the right first action is often to obtain the missing value rather than to intervene.

## Intervention ladder

Work down this list and stop at the first action that brings the predicted risk inside
tolerance. Each step costs more than the one above it.

1. **Obtain the missing input.** Ask the carrier for the telemetry interval, or the depot for
   the actual packout used. Roughly one flagged consignment in five is flagged partly because
   nobody recorded something knowable.
2. **Shorten the review cadence.** Cheap, reversible, and it converts a prediction into an
   early measurement.
3. **Reroute to a shorter journey shape.** Moving from `multi_leg` to `single_transfer` or
   `direct` removes custody transfers, which is the intervention with the best ratio of risk
   removed to cost added. Record why on the shipment.
4. **Move to a stronger packout at the next hub.** Only where the hub actually holds
   conditioned stock of the target system; re-packing into unconditioned coolant is worse
   than doing nothing.
5. **Move to a validated carrier.** The largest single reduction available, and the most
   expensive. Reserve it for patient-specific and campaign-critical consignments.

## What not to do

- Do not quarantine on a prediction. Quarantine is for measured excursions and for lanes with
  an unassessable gap in the record. Holding product because a forecast was high strands a
  clinic on the strength of a number nobody has verified.
- Do not reroute a delivered consignment. It does nothing, it confuses the audit trail, and
  the tooling will refuse it.
- Do not re-run the prediction hoping for a different number. It is deterministic for the same
  inputs; a different answer means an input changed, and the changed input is the finding.

## Recording the intervention

Every intervention is written onto the shipment timeline with: the predicted risk and its
interval at the time of the decision, which rung of the ladder was taken, and what the
predicted risk became afterwards. That triple is what makes the next quarter's review of
these decisions possible at all — without the "before" figure, an intervention that worked
and one that was never needed look identical.

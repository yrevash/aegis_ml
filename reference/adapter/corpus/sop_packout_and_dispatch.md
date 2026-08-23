---
id: doc-seed-0001
kind: sop
tags: [packout, dispatch, packaging, qualification]
title: SOP-CC-101 — Packout selection and pre-dispatch verification
---

# SOP-CC-101 — Packout selection and pre-dispatch verification

## Scope

Applies to every temperature-controlled consignment leaving an origin depot, for all four
product classes. It governs which thermal system may be booked against a given lane
duration, and what must be verified on the dock before the consignment is released to the
carrier.

## Qualified duration by packaging type

A packout may only be booked where its qualified duration exceeds the planned door-to-door
transit time with the stated margin. These are the durations this network qualifies:

| Packaging | Qualified duration | Required margin | Typical use |
|---|---|---|---|
| `passive_gel` | 36 hours | 8 hours | short direct lanes, tolerant product |
| `passive_pcm` | 96 hours | 12 hours | the default for multi-leg lanes |
| `active_electric` | 240 hours | 24 hours | high-value or very long lanes |
| `dry_ice` | 72 hours | 12 hours | frozen product only |

A planned transit of 60 hours therefore may not be booked on `passive_gel` (36 + 8 = 44
hours of cover) and must move to `passive_pcm` or `active_electric`. This is the single most
common booking error in the network, and it is why lane duration and packaging together
dominate the spoilage-risk model rather than either one alone.

## Ambient allowance

The qualified durations above assume a mean ambient of 25 °C or below along the lane. Above
25 °C, reduce the qualified duration of both passive systems by 20% and re-check the margin.
Below 0 °C, `passive_gel` and `passive_pcm` are at risk of *freezing* the payload rather than
warming it, which destroys diagnostic kits and most biologics outright. Cold-weather lanes
carrying freeze-sensitive product must be booked on `active_electric`.

## Pre-dispatch verification

Before the consignment is handed to the carrier, the dispatching depot must confirm and
record all four of the following. A consignment released without them is a deviation.

1. The physical packout matches the packaging type on the booking. A substitution made on the
   dock — usually gel packs standing in for an unavailable PCM shipper — is the most common
   root cause of an unexplained excursion, because nothing in the record shows the cover
   changed.
2. The data logger is inside the payload envelope, not taped to the outside of the carton,
   and its serial is written onto the consignment.
3. The contracted telemetry interval has been recorded on the shipment. Where the carrier
   publishes no interval, record that fact explicitly and treat the lane as unmonitored — see
   POL-CC-204.
4. Pre-conditioning of the coolant is complete and the packout has been closed for at least
   30 minutes before dispatch. A packout sealed straight from the conditioning room carries
   heat that no logger will show for several hours.

## Handoffs

Each planned custody transfer must be named on the booking. An unplanned transfer discovered
in transit is treated as a lane change and requires the consignment to be re-assessed against
this SOP at the point it is discovered, not on arrival.

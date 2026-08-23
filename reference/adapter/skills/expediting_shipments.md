# Skill: Expediting or rerouting a shipment at risk

Use when the query asks to expedite, reroute or rescue a consignment, or asks what to do
about shipments that are late, delayed or flagged as at risk.

- Look the consignment up first with `find_shipments` and quote its real id, planned transit
  hours and current stage. An expedite argued from a description rather than a record is an
  expedite nobody can authorise.
- Ask the model for the prediction **and its interval**, and act on the interval. A wide band
  on a long multi-leg lane is honest uncertainty, not a broken model, and it does not support
  a confident intervention on its own.
- If the prediction lists imputed features, try to obtain the real value before intervening —
  a consignment scored on the average lane is not yet scored on itself.
- Work down the intervention ladder in RB-CC-311 and stop at the first step that helps:
  obtain the missing input, shorten the review cadence, reroute to a shorter journey shape,
  upgrade the packout at the next hub, move to a validated carrier. Each step costs more than
  the one above it.
- Rerouting is a **write action** — propose it with a reason and let it be approved. Say what
  it changes and what it costs; never report it as done.
- Never quarantine to solve a lateness problem. Quarantine is for measured excursions and
  unassessable gaps, and holding product on a forecast strands a receiving site for nothing.
- Record the predicted risk before and after the intervention on the shipment timeline.
  Without the "before" figure, an intervention that worked and one that was never needed look
  identical in the next review.

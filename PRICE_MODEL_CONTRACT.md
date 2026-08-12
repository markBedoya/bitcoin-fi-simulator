# Price Model Contract

This file captures the shared rules for future Price Model changes so individual fixes do not break the model's overall economic meaning.

1. **The structural/geometric centerline is Bitcoin's long-term fair-value reference.** It is fitted from the selected training data and must remain visible, continuous, and unmodified by cycle guardrails.
2. **Future turning points remain connected to fair value.** A turning-point price is derived from the centerline at that date multiplied by a learned cycle valuation multiple.
3. **Peaks and troughs are learned independently.** Peak valuation uses `peak / centerline`; trough valuation uses `trough / centerline`. No forced symmetric envelope is required.
4. **Maturity means compression toward fair value unless held-out evidence rejects it.** Peak excess and trough discount can shrink independently. With insufficient independent same-type anchors, the latest observed valuation distance is carried forward rather than extrapolating an unvalidated trend.
5. **The long-term cycle structure should not drift downward while fair value rises.** Successive projected peaks and successive projected troughs may not form a secular downward staircase. Same-type historical/projected anchors provide a structural floor.
6. **Sequential bull gains and bear losses are validation, not the primary endpoint generator.** They check phase direction and show whether the fair-value forecast implies historically plausible transitions.
7. **No manually chosen future Bitcoin price targets or arbitrary drawdown percentages.** Prices, valuation multiples, trends, and guardrails must be consequences of observed data or structural invariants.
8. **Historical backtests cannot use future prices.** When training ends at an old anchor, prices after that cutoff may be shown only for diagnostics after the forecast has been produced.
9. **Cycle timing remains fixed for now.** The active schedule remains 1428 days: 1064 bull days + 364 bear days. Price behavior is the current research focus.
10. **Projection-boundary continuity is exact.** Historical fit, latest actual Bitcoin price, and the future path meet at the selected training end.
11. **Empirical bull/bear phase shapes remain responsible for the daily path between turning points.** Endpoint changes must not silently replace phase-shape learning.
12. **Anchor start/end controls are the research laboratory.** The model must support rapid fake-today testing at historical anchors plus the latest imported Bitcoin price as an end option.

## Current endpoint equation

For a future peak:

`Projected peak = structural centerline at peak date × learned peak valuation multiple`

For a future trough:

`Projected trough = structural centerline at trough date × learned trough valuation multiple`

Peak and trough valuation maturity are fitted separately from historical anchors available inside the selected training window.

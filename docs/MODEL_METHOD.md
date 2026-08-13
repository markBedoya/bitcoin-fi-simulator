# Bottom-Anchored Dynamic-Settling Method

## Purpose

The model asks whether Bitcoin's comparatively stable bear-market bottom regions can provide a useful structural foundation while still allowing an incomplete cycle to revise today's estimate. Observed evidence, learned estimates, and user scenarios remain separate.

## Turning regions

Turning dates identify broad market-regime transitions, not exact extrema. The primary bottom definition searches a 241-day window centered on each anchor and represents the region with the median of its seven lowest daily closes. Peak regions use the median of the seven highest closes in a 181-day window.

The exact extreme and its date remain visible. Sensitivity tests repeat the calculation using half-windows of 60, 90, 120, and 180 days; clusters of 3, 7, 14, and 30 closes; and both median and geometric-mean region prices. Unavailable combinations are recorded rather than silently filled.

## Dynamic settling

The current turning region is not assumed to be complete. Before observing it, four internal models estimate its eventual level from earlier completed bottoms. The pre-observation ensemble is blended with the forming observed region in log space:

```text
evidence = elapsed_fraction_of_turning_window
dynamic_bottom = exp((1 - evidence) × log(forecast) + evidence × log(observed_region))
```

Evidence progresses from zero at the start of the turning window to one at its end. This makes the estimated bottom—and therefore the historical fair-value curve—settle gradually as later prices reveal the full region.

For completed historical cycles, the application performs a fake-today test. It fixes a reference date before the turn, reveals later data one month at a time, recomputes the dynamic bottom and fixed-date fair value, and compares them with the values obtained after the full turning window is known. The forming cycle has no final answer yet.

## Internal bottom models

Four transparent methods estimate a future bottom region:

1. an expanding-history power law fitted only to observed bottom regions;
2. decay of bottom-to-bottom growth above `1×`, fitted across all available transitions;
3. the same excess-growth decay fitted to the two most recent transitions;
4. a recency-weighted local power-law exponent derived from consecutive bottom regions.

The excess-growth methods fit:

```text
log(growth_multiple - 1) = intercept + slope × cycle_index
```

Each later observable bottom is hidden in turn. Models fit only earlier regions and forecast the hidden target. Ensemble influence is based on weighted absolute log error; the forming cycle supplies partial rather than full evidence. The central result is a validation-weighted geometric mean. The low/high values are model disagreement, not a statistical confidence interval.

## Bottom foundation

Completed bottom regions and the dynamically settling current region are joined in log-price space. Beyond the current anchor, the foundation extends to the validation-weighted next-bottom estimate. Projection segments are marked separately from observed history.

## Fair value

The application computes four internally derived cycle-neutral multiples above each cycle's bottom foundation:

1. peak/bottom log midpoint;
2. time-weighted cycle median;
3. time-weighted geometric mean;
4. midpoint of the central 50% of price-to-foundation observations in log space.

Each method is walked forward into later cycles. Completed-cycle median error and time-above/time-below neutrality determine its validation weight. The incomplete current cycle is diagnostic only and does not determine these weights. Today's fair-value multiple is the validation-weighted geometric mean of the four current estimates.

## All-price diagnostic

A cycle-balanced regression over all daily prices remains optional inside the Research Lab. It is labeled as a diagnostic rather than fair value because high bull-market prices can pull it above the cycle-neutral level. It does not change any bottom or fair-value estimate.

## Research boundaries

- Bitcoin supplies very few independent market cycles.
- The current bottom region remains incomplete.
- Turning dates, windows, and cluster statistics are transparent research choices rather than facts of nature.
- Walk-forward validation has only a few held-out cycles.
- Future candidate ranges are structural disagreement, not probability intervals.
- User peak and bottom inputs are scenarios only and never become model evidence.
- The model is `RESEARCH_ONLY` and does not provide investment advice or a guaranteed floor.

The next priority is to collect later forming-cycle observations and watch whether the current estimate settles in the same way the historical fake-today tests did.

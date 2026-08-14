# Bottom-Anchored Dynamic-Settling Method

## Purpose

The model asks whether Bitcoin's comparatively stable bear-market bottom regions can provide a useful structural foundation while still allowing an incomplete cycle to revise today's estimate. Observed evidence, learned estimates, and user scenarios remain separate.

## Turning regions

Turning dates identify broad market-regime transitions, not exact extrema. The primary bottom definition searches a 241-day window centered on each anchor and represents the region with the median of its seven lowest daily closes. Peak regions use the median of the seven highest closes in a 181-day window.

The exact extreme and its date remain visible. Sensitivity tests repeat the calculation using half-windows of 60, 90, 120, and 180 days; clusters of 3, 7, 14, and 30 closes; and both median and geometric-mean region prices. Unavailable combinations are recorded rather than silently filled.

## Dynamic settling

The current turning region is not assumed to be complete. Before observing it, four internal models estimate its eventual level from earlier completed bottoms. The pre-observation ensemble is blended with the forming observed region in log space:

```text
evidence = calibrated_historical_settling_weight
dynamic_bottom = exp((1 - evidence) × log(forecast) + evidence × log(observed_region))
```

The evidence schedule is calibrated by replaying the 2011, 2015, 2018, and 2022 bottom windows in 5% increments. At every increment, the partial region's remaining absolute log error is compared with its error at the beginning of the window:

```text
settling_progress = 1 - current_partial_log_error / initial_partial_log_error
```

The median progress across completed cycles is made monotonic. Because only four cycles are available, the raw empirical curve is shrunk toward the original linear-time schedule using two linear-prior cycle equivalents. The resulting weight begins at zero and reaches one at the end of the window.

Evidence weight is the observed region's influence in the blend. It is not the probability that the market bottom has occurred.

### Dependence on individual cycles

The calibration is rebuilt four times, omitting one completed bottom region in each run. Every reduced-history curve is regularized using the same two linear-prior cycle equivalents, then propagated through today's settling bottom, bottom foundation, fair value, and mature-cycle next-bottom calculation.

The minimum and maximum across those four runs form the leave-one-cycle-out cycle-dependence range. This answers whether one unusual historical cycle is carrying the current result. It is not a bootstrap distribution, probability interval, or 95% confidence interval.

For completed historical cycles, the application performs a fake-today test. It fixes a reference date before the turn, reveals later data one month at a time, recomputes the dynamic bottom and fixed-date fair value, and compares them with the values obtained after the full turning window is known. Each historical target uses a calibration curve built only from earlier completed bottoms. The forming cycle has no final answer yet.

## Mature-cycle next-bottom projection

The public projection uses a decay fit beginning with the 2015 bottom. It therefore measures the three later bottom-to-bottom transitions: 2015→2018, 2018→2022, and the dynamically settling 2022→2026 transition. For each transition, growth above `1×` is fitted in log space:

```text
log(growth_multiple - 1) = intercept + slope × mature_transition_index
```

The next growth multiple is applied to the dynamic current bottom. The central definition range is the 10th–90th percentile of this result across every currently available window-width, cluster-size, and region-statistic definition. The full minimum and maximum remain available as definition stress values. These ranges measure sensitivity to model definitions; they are not statistical confidence intervals.

The early 2011→2015 transition is excluded from the core projection because Bitcoin's micro-cap growth regime is not treated as representative of mature-cycle growth. This is a structural research choice and is stated explicitly rather than learned from enough independent cycles.

## Pre-observation bottom prior

Before a forming bottom is observed, four transparent methods create a prior estimate:

1. an expanding-history power law fitted only to observed bottom regions;
2. decay of bottom-to-bottom growth above `1×`, fitted across all available transitions;
3. the same excess-growth decay fitted to the two most recent transitions;
4. a recency-weighted local power-law exponent derived from consecutive bottom regions.

The excess-growth methods fit:

```text
log(growth_multiple - 1) = intercept + slope × cycle_index
```

Each later observable bottom is hidden in turn. Models fit only earlier regions and forecast the hidden target. Ensemble influence is based on weighted absolute log error; the forming cycle supplies partial rather than full evidence. This ensemble supplies the pre-observation starting point for dynamic settling. It is not used as four competing paths for the 2030 public projection.

## Bottom foundation

Completed bottom regions and the dynamically settling current region are joined in log-price space. Beyond the current anchor, the foundation extends to the mature-cycle next-bottom estimate. Projection segments are marked separately from observed history.

## Fair value

The application computes four internally derived cycle-neutral multiples above each cycle's bottom foundation:

1. peak/bottom log midpoint;
2. time-weighted cycle median;
3. time-weighted geometric mean;
4. midpoint of the central 50% of price-to-foundation observations in log space.

Each method is walked forward into later cycles. Completed-cycle median error and time-above/time-below neutrality determine its validation weight. The incomplete current cycle is diagnostic only and does not determine these weights. Today's fair-value multiple is the validation-weighted geometric mean of the four current estimates.

Fair value is not projected beyond the latest observed date. A future fair-value curve would require a separately validated model of peak compression; until then, only the bottom foundation is extended to the next region.

## All-price diagnostic

A cycle-balanced regression over all daily prices remains optional inside the Research Lab. It is labeled as a diagnostic rather than fair value because high bull-market prices can pull it above the cycle-neutral level. It does not change any bottom or fair-value estimate.

## Research boundaries

- Bitcoin supplies very few independent market cycles.
- The current bottom region remains incomplete.
- Turning dates, windows, and cluster statistics are transparent research choices rather than facts of nature.
- Walk-forward validation has only a few held-out cycles.
- The mature-cycle projection has only three transitions and includes a forming current endpoint.
- The empirical settling curve has only four completed regions and is deliberately regularized toward linear time.
- The leave-one-cycle-out spread measures single-cycle dependence, not total model uncertainty.
- Definition ranges are not probability intervals.
- User peak and bottom inputs are scenarios only and never become model evidence.
- The model is `RESEARCH_ONLY` and does not provide investment advice or a guaranteed floor.

The next priority is to collect later forming-cycle observations and watch whether the current estimate settles in the same way the historical fake-today tests did.

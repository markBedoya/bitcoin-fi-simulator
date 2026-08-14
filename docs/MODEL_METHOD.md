# Bottom-Anchored Dynamic-Settling Method

## Purpose

The model asks whether Bitcoin's comparatively stable bear-market bottom regions can provide a useful structural foundation while still allowing an incomplete cycle to revise today's estimate. Observed evidence, learned estimates, and user scenarios remain separate.

## Turning regions

Turning dates identify broad market-regime transitions, not exact extrema. The primary bottom definition searches a 241-day window centered on each anchor and represents the region with the median of its seven lowest daily closes. Peak regions use the median of the seven highest closes in a 181-day window.

The exact extreme and its date remain visible. Sensitivity tests repeat the calculation using half-windows of 60, 90, 120, and 180 days; clusters of 3, 7, 14, and 30 closes; and both median and geometric-mean region prices. Unavailable combinations are recorded rather than silently filled.

## Current anchor timing

The original October 7, 2026 date was a rough estimate based on an earlier cycle-start convention. The bottom-region model instead uses November 21, 2022 as its preceding completed anchor. The two completed mature intervals were 1,431 days from 2015 to 2018 and 1,437 days from 2018 to 2022. Their midpoint gives a 1,434-day central interval and an October 25, 2026 anchor, with October 22–28 as the directly observed mature-cycle timing range.

Only two independent mature timing intervals exist, so the narrow six-day range must not be mistaken for strong statistical certainty. The application also retains October 7 as an early stress date and adds an equally distant November 12 late stress date. Every timing variant re-extracts the forming region and recomputes its window progress, evidence weight, dynamic bottom, current fair value, and next-bottom projection.

The primary forming endpoint does not select one exact date from the October 22–28 empirical range. It runs the October 22, 25, and 28 anchor models separately, gives each one equal influence, and combines price-valued outputs with a geometric mean. Window progress and evidence weights use their arithmetic mean. This marginalization occurs before the bottom foundation, fair-value calibration, and mature-cycle projection are rebuilt, preventing one anchor window boundary from deciding the public result.

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

Completed bottom regions and the active bottom estimate are joined in log-price space. While its observation window is forming, that endpoint blends prior and observed-region evidence; before the window begins, it is entirely prior-based. Beyond the active anchor, the foundation extends to the mature-cycle next-bottom estimate. Projection segments are marked separately from observed history.

## Fair value

The application computes four internally derived cycle-neutral multiples above each cycle's bottom foundation:

1. peak/bottom log midpoint;
2. time-weighted cycle median;
3. time-weighted geometric mean;
4. midpoint of the central 50% of price-to-foundation observations in log space.

Each method is walked forward into later cycles. Completed-cycle median error and time-above/time-below neutrality determine its validation weight. The incomplete current cycle is diagnostic only and does not determine these weights. Today's fair-value multiple is the validation-weighted geometric mean of the four current estimates.

Fair value is not projected beyond the latest observed date. A future fair-value curve would require a separately validated model of peak compression; until then, only the bottom foundation is extended to the next region.

## Projection horizon

The public bottom foundation extends one region beyond the active rolling target. In August 2026 that endpoint is the expected 2030 region; after the 2026 window settles and rolls forward, the active target becomes the next cycle and the single additional projection advances with it. Extending the graph to a fixed ten-year horizon would require recursively applying mature-cycle decay and validating that recursive procedure on historical fake-today forecasts. That work has not yet been completed.

A ten-year fair-value curve requires the recursive bottom foundation plus a separate forward model for the compression or stabilization of cycle-neutral fair-value multiples. Holding today's multiple constant would be a scenario assumption, not an empirically validated result, so it is not drawn as model output.

## Operational lifecycle

Every active target has one of three states:

1. `pre_window`: the model uses only the walk-forward prior and assigns zero observed-region evidence;
2. `forming`: daily prices inside the target window update the region, empirical evidence weight, bottom foundation, fair value, and sensitivities;
3. `settled`: the complete fixed window is promoted into the historical catalog.

Promotion is deterministic rather than stored as mutable application state. Once a window has closed, its extraction always stops at the recorded window end, so prices arriving later cannot change that settled region. The empirical region date becomes the timing origin for the next bottom target. Completed mature intervals are recalculated, and their median generates the new central anchor with early and late timing variants. The same eligibility rule creates, observes, and settles later peak regions for fair-value calibration.

The first automatic boundary is the 2026 window ending February 22, 2027. With that day's data available, cycle 4 is promoted and cycle 5 becomes the active target. Offline time-travel tests cover the forming-to-settled boundary, a prior-only next target, rolling peak formation and settlement, immutability of a closed bottom, and a second bottom rollover.

The runtime price cache expires after 24 hours. A manual refresh remains available. Therefore a continuously hosted app can incorporate new daily observations and move between lifecycle states without a code edit, subject to the data provider and host remaining operational.

## All-price diagnostic

A cycle-balanced regression over all daily prices remains optional inside the Research Lab. It is labeled as a diagnostic rather than fair value because high bull-market prices can pull it above the cycle-neutral level. It does not change any bottom or fair-value estimate.

## Research boundaries

- Bitcoin supplies very few independent market cycles.
- The active bottom region can be incomplete or pre-window; its lifecycle state is always reported.
- The current anchor is based on only two completed mature timing intervals.
- Equal weighting across the three empirical anchor dates is a transparent smoothing choice, not a learned probability distribution.
- Turning dates, windows, and cluster statistics are transparent research choices rather than facts of nature.
- Walk-forward validation has only a few held-out cycles.
- The mature-cycle projection has very few transitions and includes an active endpoint that may be forming or prior-only.
- The empirical settling curve has only four completed regions and is deliberately regularized toward linear time.
- The leave-one-cycle-out spread measures single-cycle dependence, not total model uncertainty.
- Definition ranges are not probability intervals.
- User peak and bottom inputs are scenarios only and never become model evidence.
- The model is `RESEARCH_ONLY` and does not provide investment advice or a guaranteed floor.

The next projection priority is to validate recursive mature-cycle bottom forecasts before extending the public foundation to a fixed ten-year horizon. Future fair value remains a separate later step.

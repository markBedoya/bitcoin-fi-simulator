# Bottom-Anchored Fair-Value Method

## Purpose

The model asks whether Bitcoin's comparatively stable bear-market bottom regions provide a cleaner structural foundation than an all-price regression that is strongly influenced by early bull-market peaks.

It deliberately separates observed evidence, learned estimates, external benchmarks, and user scenarios.

## Turning regions

Turning dates identify broad market-regime transitions, not exact extrema. For each bottom anchor, the model searches a 241-day window centered on the turning date and represents the region with the median of its seven lowest daily closes. Peak regions use the median of the seven highest closes in a 181-day window.

The region method reduces dependence on a single anomalous daily wick. The exact extreme and its date remain visible in diagnostics.

## Bottom foundation

Observed bottom regions are joined in log-price space. Four transparent methods estimate the next bottom region:

1. a fixed power law learned from observed bottom regions;
2. the published `0.42 × 1.0117e-17 × days^5.82` benchmark;
3. decay of bottom-to-bottom growth in excess of `1×`, fitted across all cycles;
4. the same excess-growth formulation fitted to the two most recent transitions.

The decay candidates use:

```text
log(growth_multiple - 1) = intercept + slope × cycle_index
```

This guarantees a positive bottom-to-bottom increase without imposing a hand-picked future price.

## Walk-forward validation

Each observable later bottom is hidden in turn. Candidate models fit only earlier bottom regions and forecast the hidden target. Model influence is based on weighted absolute log error. The still-forming 2026 region receives half the evidence weight of a completed region.

The ensemble central estimate is the validation-weighted geometric mean of candidate forecasts. The displayed low/high range is the full candidate range. It is not a statistical confidence interval.

## Experimental fair value

For each cycle, the observed peak region is measured relative to the bottom foundation at the peak date. The cycle-neutral multiplier is the log-space midpoint between the bottom foundation (`1×`) and the peak multiple:

```text
cycle_neutral_multiplier = sqrt(peak_price / bottom_foundation_at_peak)
fair_value = bottom_foundation × cycle_neutral_multiplier
```

This makes bottom stability foundational while allowing peaks to compress independently as Bitcoin matures.

## Research comparisons

Two older structures remain visible only for comparison:

- the published `5.82 × 0.71` fair-price formula;
- a cycle-balanced all-price power-law backbone.

Neither comparison silently changes the bottom-derived estimate.

## Current limitations

- Bitcoin supplies very few independent market cycles.
- The current bottom region remains incomplete.
- Turning windows and seven-close clusters are transparent research choices that require sensitivity testing.
- Walk-forward validation contains only a few held-out bottoms.
- The current fair-value midpoint is descriptive and has not yet earned production status.
- Future peak modeling is intentionally deferred; user peak and bottom ranges are scenario inputs only.

The next research priority is sensitivity testing of region width and cluster size, followed by historical walk-forward testing of the fair-value layer itself.

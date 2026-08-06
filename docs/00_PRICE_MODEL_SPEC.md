# Bitcoin Price Model v2.0.1

## Purpose
Generate a reproducible daily Bitcoin price path using adjustable Coin Metrics history.

## Default behavior
Use all available Coin Metrics daily PriceUSD data.

## User controls
- Training start date
- Training end date
- Use latest available data
- Projection horizon from 1 to 40 years
- Overlay excluded actual prices
- Logarithmic or linear price scale

## Model
Projected price = structural centerline × exp(empirical cycle deviation)

The implementation does not use a constant annual-return assumption.

## CSV fields
- date
- row_type
- actual_price_usd
- included_in_training
- structural_centerline_usd
- cycle_progress
- cycle_phase
- cycle_shape_value
- cycle_amplitude
- fitted_or_projected_price_usd
- model_version
- training_start_date
- training_end_date


## v1.1 change
The CAGR diagnostics table lists every horizon year from 1 through the selected projection horizon.

## v1.2 change

The training range slider now directly controls both the training start and end.
The previous latest-date override was removed. A reset button restores all
available Coin Metrics data.


## v1.5 conservative monthly start search

The application tests one candidate start per calendar month, using that
month's latest available Coin Metrics observation.

Candidate rules:

- Training end remains fixed.
- Every candidate must leave at least eight years of training data.
- Candidates span the full available history through the newest eligible month.
- The complete price model is refitted for each candidate.
- The selected start is the candidate with the lowest implied CAGR at the
  currently selected projection horizon.
- The training-range start handle updates automatically.


- Renamed the existing search button to indicate the 8-year minimum.
- Added a second identical search using a 4-year minimum training history.


## v1.5 leaderboard

Each conservative monthly-start search now retains the full ranked result set.

The leaderboard displays:

- rank,
- candidate training start,
- training length,
- projection horizon,
- implied CAGR,
- projected ending price.

The user may select any ranked row and apply its training start to the main
price model without rerunning the search.


## v1.5.1 fix

- Both the 8-year and 4-year conservative searches now save their complete
  ranked result tables.
- Removed a duplicated leaderboard assignment in the 8-year search.
- Resetting the training range clears the prior leaderboard.
- A visible confirmation appears when leaderboard results are ready.


## v1.6 change

The projection horizon now supports 1 through 80 years. The annual CAGR
diagnostics and daily CSV projection expand to the selected horizon.


## v1.7 change

Added conservative monthly-start searches for 4-, 5-, 6-, 7-, and 8-year
minimum training histories. All options use the same ranking logic,
leaderboard, projection horizon, and one-click row application.


## v1.7.1 fix

Applying a leaderboard row now queues the selected training range, reruns the
page, and applies the range before the Streamlit slider is instantiated. This
avoids modifying a widget-bound session-state key after widget creation.


## v2.0 defaults and BTC Financial Independence

Default Price Model settings:
- Training start: 2018-12-31
- Training end: latest available Coin Metrics observation
- Projection horizon: 80 years

The BTC Financial Independence tab compares the structural-centerline and cycle-adjusted Bitcoin paths, combines either with an other-investments account, inflates monthly spending, and solves for the earliest retirement age that remains solvent through the selected ending age.


## v2.0.1 change

BTC Financial Independence inputs now use a Streamlit form. Editing fields no longer triggers the retirement solver. The simulation runs only after the user presses **Calculate**, and the last submitted result remains visible until recalculated.


## v2.0.2 withdrawal source

BTC Financial Independence supports three monthly withdrawal methods:

- Proportional — default; withdrawals follow current portfolio weights.
- BTC first.
- Other investments first.

With the proportional option, an 80% BTC / 20% other-investments portfolio
funds each monthly withdrawal using the same 80 / 20 split.


## v2.0.3 separate other-investment returns

The BTC Financial Independence calculator has two annual-return assumptions for
all other investments:

- Annual return while contributing — default 10%.
- Annual return while drawing — default 5%.

The accumulation return applies before retirement. The drawing return begins
automatically on the retirement date and continues while withdrawals are made.
This supports modeling a move from growth assets to a more conservative
retirement allocation.


## v2.1 BTC Financial Independence redesign

- Fixed stale `other_return` references.
- Reorganized the page into Personal Plan, Retirement Spending, Bitcoin, and
  All Other Investments sections.
- Replaced Calculate with Run Simulation.
- Preserved submit-only calculation behavior.
- Added side-by-side results for structural-centerline and cycle-adjusted BTC
  scenarios.
- Correctly supports separate other-investment returns before and after
  retirement.
- Keeps the last submitted results visible until the next simulation.


## v2.1.1 default changes

- Compound interest defaults to 12 times annually.
- Contributions default to the end of each compounding period.


## v2.1.2 portfolio component charts

Each BTC scenario now has its own portfolio-path tab showing:

- Bitcoin value over time.
- All other investments over time.
- Combined total portfolio.
- Financial-independence date.
- Bitcoin versus other-investment portfolio share over time.

This makes it possible to see which asset group contributes more to future
portfolio growth and how the allocation changes during accumulation and
withdrawal.


## v2.1.3 fix

Added explicit unique Streamlit keys to each scenario's portfolio-path and
portfolio-composition Plotly charts. This prevents duplicate auto-generated
element IDs when scenario charts share the same structure.


## v2.1.4

FI results now store the exact Price Model training range and projection
horizon. Changing those settings clears stale FI results and requires a fresh
simulation.

New defaults:
- Ending age 100
- BTC starting value $265,000
- BTC weekly contribution $944
- Other investments starting value $200
- Other weekly contribution $900
- Other-investment return while contributing 12%
- Other-investment return while drawing 7%


## v2.1.5 active Price Model synchronization

The Price Model page now saves its exact active configuration to:

`config/active_price_model.json`

The saved configuration includes:

- training start,
- training end,
- projection horizon,
- latest Coin Metrics data date,
- a fingerprint derived from the calculated model path.

BTC Financial Independence reads this persisted configuration and recalculates
the same model before running retirement simulations. Stored FI results include
the fingerprint, so any change to the range, horizon, data, or calculated price
path invalidates old results.

The default other-investment starting principal is corrected to $200,000.


## v2.1.6 no-Bitcoin counterfactual

BTC Financial Independence now calculates a third scenario:

**No BTC — all funds in other investments**

This counterfactual:

- adds the Bitcoin starting value to the other-investment starting principal,
- adds the weekly Bitcoin contribution to the weekly other-investment contribution,
- uses the same accumulation return, retirement return, compounding frequency,
  contribution timing, spending, and inflation assumptions,
- and contains no Bitcoin exposure.

Its total portfolio path is drawn as a dashed comparison line on both Bitcoin
scenario charts, making Bitcoin's incremental contribution visible over time.


## v2.2 FI target and coast planning

BTC Financial Independence now supports three planning modes:

1. Find earliest FI age using entered weekly contributions.
2. Target FI age — solve the minimum weekly contribution required.
3. Coast to FI age — solve the minimum weekly contribution required only until
   the selected coast age, followed by zero contributions until FI.

Bitcoin conviction defines the percentage of future weekly contributions going
to Bitcoin. The remainder goes to all other investments. Existing balances are
not rebalanced by the conviction input.

The comparison set now includes:

- BTC structural centerline,
- BTC cycle-adjusted path,
- 100% other investments,
- 100% Bitcoin.

All user-facing retirement terminology was replaced with financial-independence
terminology.


## v2.3 true monthly contributions

All FI contribution inputs, simulations, and target/coast solvers now operate directly monthly. Defaults are $3,776 BTC and $3,600 other investments monthly. Portfolio Paths adds Total Portfolio, Portfolio Components, and Portfolio Comparison views.

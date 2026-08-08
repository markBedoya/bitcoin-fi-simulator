import numpy as np

from src.price_model import (
    _combine_phase_curves,
    _monotone_cubic_eval,
    _phase_curve_diagnostics,
)

# Synthetic bull phases that accelerate late. The learned median should preserve
# that late acceleration without any hand-set '75%' parameter.
grid = np.linspace(0.0, 1.0, 401)
bull_curves = [grid ** 2.8, grid ** 3.0, grid ** 3.2]
bull_template = _combine_phase_curves(bull_curves, grid)
bull_diag = _phase_curve_diagnostics(grid, bull_template)

assert abs(bull_template[0]) < 1e-12
assert abs(bull_template[-1] - 1.0) < 1e-12
assert np.all(np.diff(bull_template) >= -1e-12)
assert bull_diag["half_move_progress"] > 0.70
assert bull_diag["max_velocity_progress"] > 0.70

# Synthetic bear phases that lose value quickly early and then bottom slowly.
bear_curves = [
    1.0 - (1.0 - grid) ** 2.4,
    1.0 - (1.0 - grid) ** 2.6,
]
bear_template = _combine_phase_curves(bear_curves, grid)
bear_diag = _phase_curve_diagnostics(grid, bear_template)

assert abs(bear_template[0]) < 1e-12
assert abs(bear_template[-1] - 1.0) < 1e-12
assert np.all(np.diff(bear_template) >= -1e-12)
assert bear_diag["half_move_progress"] < 0.35
assert bear_diag["max_velocity_progress"] < 0.30

# Shape-preserving cubic interpolation must keep endpoints and stay in range.
for t in np.linspace(0.0, 1.0, 101):
    b = _monotone_cubic_eval(grid, bull_template, float(t))
    r = _monotone_cubic_eval(grid, bear_template, float(t))
    assert -1e-9 <= b <= 1.0 + 1e-9
    assert -1e-9 <= r <= 1.0 + 1e-9

print("Empirical phase-shape learning checks passed.")

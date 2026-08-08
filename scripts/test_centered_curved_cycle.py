import numpy as np

from src.price_model import _bull_ease, _bear_ease

# Exact endpoints.
assert _bull_ease(0.0) == 0.0
assert _bull_ease(1.0) == 1.0
assert _bear_ease(0.0) == 0.0
assert _bear_ease(1.0) == 1.0

# Bull phase is deliberately gradual early and accelerates later.
assert _bull_ease(0.25) < 0.25
assert _bull_ease(0.50) == 0.5
assert _bull_ease(0.75) > 0.75

# Bear phase moves faster early and slows into the bottom.
assert _bear_ease(0.25) > 0.25
assert abs(_bear_ease(0.50) - 0.5) < 1e-12
assert _bear_ease(0.75) < 0.75

# Both phase mappings retain midpoint symmetry, preventing persistent drift.
for t in np.linspace(0.0, 1.0, 101):
    assert abs((_bull_ease(t) + _bull_ease(1.0 - t)) - 1.0) < 1e-12
    assert abs((_bear_ease(t) + _bear_ease(1.0 - t)) - 1.0) < 1e-12

print("Centered curved-cycle shape checks passed.")

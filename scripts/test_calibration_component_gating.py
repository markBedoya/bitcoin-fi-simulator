from src.walk_forward_calibration import _component_status

assert _component_status(0.70, 0.80)[0] == "REJECTED"
assert _component_status(0.70, 0.68)[0] == "MODEST"
assert _component_status(0.70, 0.55)[0] == "PASS"
print("Independent structural/envelope gating checks passed.")

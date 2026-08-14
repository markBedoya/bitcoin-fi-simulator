from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
forbidden_names = {
    "app_pages",
    "price_model_v2.py",
    "walk_forward_calibration.py",
    "financial_independence.py",
    "active_model_config.py",
    ".git",
    ".venv",
    ".vs",
}

found = {
    path.name
    for path in ROOT.rglob("*")
    if path.name in forbidden_names
}
assert not found, f"Obsolete or generated project content remains: {sorted(found)}"

python_files = sorted(
    path.relative_to(ROOT).as_posix()
    for path in ROOT.rglob("*.py")
)
expected_python = [
    "scripts/test_lifecycle.py",
    "scripts/test_price_model.py",
    "scripts/test_project_manifest.py",
    "scripts/test_ui_contract.py",
    "src/__init__.py",
    "src/data_pipeline.py",
    "src/price_model.py",
    "streamlit_app.py",
]
assert python_files == expected_python, (python_files, expected_python)
print("Minimal project manifest checks passed.")

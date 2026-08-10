from __future__ import annotations

import ast
from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
EXPECTED_ROUTES = {
    "data-management",
    "price-model",
    "calibrated-price-model",
    "btc-financial-independence",
}


def _page_url_paths() -> list[str]:
    tree = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
    routes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "Page"):
            continue
        for kw in node.keywords:
            if kw.arg != "url_path":
                continue
            if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, str):
                raise AssertionError("All non-default navigation url_path values must be string literals")
            routes.append(kw.value.value)
    return routes


def main() -> None:
    routes = _page_url_paths()
    assert set(routes) == EXPECTED_ROUTES, (routes, EXPECTED_ROUTES)
    assert len(routes) == len(set(routes)), f"Duplicate routes: {routes}"
    assert all("/" not in route for route in routes), routes
    print("Explicit Streamlit navigation routes are stable and unique.")


if __name__ == "__main__":
    main()

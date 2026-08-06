from pathlib import Path
import py_compile

root = Path(__file__).resolve().parents[1]
files = [root / "streamlit_app.py", *root.glob("pages/*.py"), *root.glob("src/*.py")]

for file in files:
    py_compile.compile(str(file), doraise=True)

print(f"Compiled {len(files)} Python files successfully.")

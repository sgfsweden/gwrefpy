import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NOTEBOOKS = sorted(DOCS.rglob("*.ipynb"))
NOTEBOOK_IDS = [path.relative_to(DOCS).as_posix() for path in NOTEBOOKS]


def test_notebooks_exist():
    assert NOTEBOOKS, f"No notebooks found in {DOCS}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebooks_are_committed_without_outputs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"{path} has committed outputs"
            assert cell["execution_count"] is None, (
                f"{path} has committed execution count"
            )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=NOTEBOOK_IDS)
def test_notebooks_are_committed_without_python_version(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    language_info = notebook.get("metadata", {}).get("language_info", {})
    assert "version" not in language_info, (
        f"{path} has committed Python version metadata"
    )

"""Source check: Chainlit prepare_initial_state must pass memory db_path.

Do not import app (CI has no chainlit). Do not open inferops_memory.db.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_PY = _REPO_ROOT / "app.py"


def _prepare_initial_state_via_to_thread(tree: ast.AST) -> list[ast.Call]:
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "to_thread"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Name) and first.id == "prepare_initial_state":
            found.append(node)
    return found


def test_chainlit_prepare_initial_state_passes_db_path_keyword():
    source = _APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app.py")
    calls = _prepare_initial_state_via_to_thread(tree)
    assert calls, "expected asyncio.to_thread(prepare_initial_state, ...)"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords if kw.arg}
        assert "db_path" in keywords, (
            "prepare_initial_state must get db_path as a keyword so "
            "memory_db_path is set for compatible history"
        )
        # fn + workload + prefix + budget + task; db_path is not a 5th positional
        assert len(call.args) == 5


def test_source_check_does_not_create_memory_db():
    assert not (_REPO_ROOT / "inferops_memory.db").exists()

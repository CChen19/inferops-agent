"""Source check: Chainlit prepare_initial_state must pass memory db_path.

Do not import app (CI has no chainlit). Do not open inferops_memory.db.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_PY = _REPO_ROOT / "app.py"
_MEMORY_DB = _REPO_ROOT / "inferops_memory.db"
_DEFAULT_DB_PATH = "inferops_memory.db"


def _snapshot_memory_db() -> tuple[bool, float | None]:
    existed = _MEMORY_DB.exists()
    mtime = _MEMORY_DB.stat().st_mtime if existed else None
    return existed, mtime


def _assert_memory_db_untouched(existed: bool, mtime: float | None) -> None:
    if existed:
        assert _MEMORY_DB.exists(), "test must not delete inferops_memory.db"
        assert _MEMORY_DB.stat().st_mtime == mtime, (
            "test must not modify inferops_memory.db"
        )
    else:
        assert not _MEMORY_DB.exists(), "test must not create inferops_memory.db"


def _parse_app_py() -> ast.AST:
    existed, mtime = _snapshot_memory_db()
    source = _APP_PY.read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app.py")
    _assert_memory_db_untouched(existed, mtime)
    return tree


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


def _db_path_keyword(call: ast.Call) -> ast.keyword:
    matches = [kw for kw in call.keywords if kw.arg == "db_path"]
    assert matches, (
        "prepare_initial_state must get db_path as a keyword so "
        "memory_db_path is set for compatible history"
    )
    return matches[0]


def test_chainlit_prepare_initial_state_passes_db_path_keyword():
    tree = _parse_app_py()
    calls = _prepare_initial_state_via_to_thread(tree)
    assert calls, "expected asyncio.to_thread(prepare_initial_state, ...)"
    for call in calls:
        # fn + workload + prefix + budget + task; db_path is not a 5th positional
        assert len(call.args) == 5
        kw = _db_path_keyword(call)
        assert isinstance(kw.value, ast.Constant), (
            "db_path must be a string Constant, not None or a computed value"
        )
        assert kw.value.value == _DEFAULT_DB_PATH, (
            f"db_path must be {_DEFAULT_DB_PATH!r}, not {kw.value.value!r}"
        )


def test_source_check_does_not_create_or_touch_memory_db():
    existed, mtime = _snapshot_memory_db()
    _parse_app_py()
    _assert_memory_db_untouched(existed, mtime)

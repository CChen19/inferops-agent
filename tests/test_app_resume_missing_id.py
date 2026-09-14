"""Source check: missing resume id uses format_resume_failure (validation).

Do not import app (CI has no chainlit). Do not open inferops_memory.db.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_APP_PY = _REPO_ROOT / "app.py"
_MEMORY_DB = _REPO_ROOT / "inferops_memory.db"


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


def _on_message_fn(tree: ast.AST) -> ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_message":
            return node
    raise AssertionError("expected async def on_message in app.py")


def _is_resume_command_if(fn: ast.AsyncFunctionDef) -> ast.If:
    """The branch taken when parse_resume_task_id is None but text is a resume cmd."""
    for node in fn.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and test.func.id == "is_resume_command"
        ):
            return node
    raise AssertionError(
        "expected `if is_resume_command(...):` in on_message for missing-id path"
    )


def _format_resume_failure_calls(branch: ast.AST) -> list[ast.Call]:
    found: list[ast.Call] = []
    for node in ast.walk(branch):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "format_resume_failure":
            found.append(node)
    return found


def test_missing_resume_id_uses_format_resume_failure_validation():
    tree = _parse_app_py()
    branch = _is_resume_command_if(_on_message_fn(tree))
    calls = _format_resume_failure_calls(branch)
    assert calls, (
        "missing-id resume path must call format_resume_failure — "
        "do not hardcode a third Cannot-resume / No-GPU copy family in app.py"
    )
    call = calls[0]
    assert call.args, "format_resume_failure needs a kind argument"
    kind = call.args[0]
    assert isinstance(kind, ast.Constant) and kind.value == "validation", (
        "missing-id path must use format_resume_failure('validation', ...) "
        "so honesty about GPU spend stays in resume.py"
    )


def test_missing_resume_id_branch_does_not_inline_no_gpu_copy():
    """Honesty line belongs in format_resume_failure, not duplicated in app.py."""
    tree = _parse_app_py()
    branch = _is_resume_command_if(_on_message_fn(tree))
    for node in ast.walk(branch):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "No GPU budget was spent" not in node.value, (
                "do not hardcode 'No GPU budget was spent' in the missing-id "
                "branch — use format_resume_failure('validation', detail)"
            )


def test_source_check_does_not_create_or_touch_memory_db():
    existed, mtime = _snapshot_memory_db()
    _parse_app_py()
    _assert_memory_db_untouched(existed, mtime)

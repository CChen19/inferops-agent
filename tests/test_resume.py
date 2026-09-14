"""CPU tests for resume parsing, reflector status, and resume failure honesty.

Do not import app (CI has no chainlit).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferops.agent.graph import run_agent
from inferops.resume import (
    ResumeValidationError,
    format_reflector_update,
    format_resume_failure,
    format_resume_help,
    is_resume_command,
    parse_resume_task_id,
)
from inferops.memory.db import save_task
from inferops.task import confirm_task, default_task_for_workload


def test_parse_resume_prefix_forms():
    assert parse_resume_task_id("resume abcdabcdabcd") == "abcdabcdabcd"
    assert parse_resume_task_id("resume-task DEADBEEF0001") == "deadbeef0001"
    assert parse_resume_task_id("  RESUME  0123456789ab  ") == "0123456789ab"


def test_parse_bare_twelve_hex_whole_message():
    assert parse_resume_task_id("cafebabeface") == "cafebabeface"
    assert parse_resume_task_id("CafeBabeFace") == "cafebabeface"


def test_parse_rejects_ordinary_and_partial_messages():
    assert parse_resume_task_id("") is None
    assert parse_resume_task_id("resume") is None
    assert parse_resume_task_id("resume-task") is None
    assert parse_resume_task_id("resume not-an-id") is None
    assert parse_resume_task_id("resume abcdabcdabc") is None  # 11 hex
    assert parse_resume_task_id("please resume abcdabcdabcd later") is None
    assert parse_resume_task_id("I have Qwen2.5 on RTX 3060") is None
    assert parse_resume_task_id("abcdabcdabcd extra") is None


def test_is_resume_command_without_valid_id():
    assert is_resume_command("resume") is True
    assert is_resume_command("resume-task") is True
    assert is_resume_command("resume abcdabcdabcd") is True
    assert is_resume_command("cafebabeface") is True
    # Non-id extra tokens are ordinary chat (draft a task), not resume.
    assert is_resume_command("resume nope") is False
    assert is_resume_command("resume not-an-id") is False
    assert is_resume_command("resume abcdabcdabc") is False  # 11 hex
    assert parse_resume_task_id("resume nope") is None
    assert parse_resume_task_id("resume not-an-id") is None
    assert parse_resume_task_id("resume abcdabcdabc") is None
    assert is_resume_command("please resume later") is False
    assert is_resume_command("chat_short maximize throughput") is False


def test_format_resume_help_names_cli_and_chat():
    text = format_resume_help("abcdabcdabcd")
    assert "resume abcdabcdabcd" in text
    assert "--resume-task abcdabcdabcd" in text
    assert "inferops agent" in text


def test_format_resume_failure_validation_vs_runtime():
    validation = format_resume_failure("validation", "No persisted task found")
    runtime = format_resume_failure("runtime", "boom during invoke")
    assert "No GPU budget was spent" in validation
    assert "No GPU budget was spent" not in runtime
    assert "may already have executed" in runtime
    assert "task record" in runtime.lower()


def test_format_reflector_update_distinguishes_actions():
    cont = format_reflector_update("continue", streak=2)
    rem = format_reflector_update("remeasure", streak=1)
    back = format_reflector_update("rollback", streak=0)
    stop = format_reflector_update("stop", stop_reason="no_reliable_improvement")

    assert "continue" in cont.lower()
    assert "streak 2" in cont
    assert "remeasure" in rem.lower()
    assert "rollback" in back.lower()
    assert "Done" in stop
    assert "no_reliable_improvement" in stop
    assert len({cont, rem, back, stop}) == 4


def test_format_reflector_update_does_not_collapse_to_continuing():
    """The UI used to say only 'continuing' vs 'Done'."""
    for action in ("continue", "remeasure", "rollback"):
        text = format_reflector_update(action, streak=1)
        assert action in text.lower()
        if action != "continue":
            assert "continuing" not in text.lower()


def test_run_agent_missing_resume_id_raises_resume_validation(tmp_path):
    llm = MagicMock()
    with pytest.raises(ResumeValidationError, match="No persisted task"):
        run_agent(
            None,
            llm,
            resume_task_id="abcdabcdabcd",
            db_path=tmp_path / "memory.db",
        )


def test_run_agent_post_start_valueerror_is_not_resume_validation(tmp_path, monkeypatch):
    """After checkpointer / invoke, ValueError must not become ResumeValidationError."""
    db = tmp_path / "memory.db"
    task = confirm_task(default_task_for_workload("chat_short", 2))
    save_task(
        task_id=task.task_id,
        session_prefix="resume_",
        thread_id="resume",
        confirmed_task=task.model_dump(mode="json"),
        status="confirmed",
        db_path=db,
    )

    class _FakeGraph:
        def get_state(self, config):
            snap = MagicMock()
            snap.values = {"already": True}  # skip prepare_initial_state
            snap.next = ()
            return snap

        def invoke(self, state, config):
            raise ValueError("simulated post-start failure")

    monkeypatch.setattr(
        "inferops.agent.graph.build_graph",
        lambda *a, **k: _FakeGraph(),
    )
    monkeypatch.setattr(
        "inferops.agent.graph.production_checkpointer",
        lambda *a, **k: _NullCtx(),
    )

    llm = MagicMock()
    with pytest.raises(ValueError, match="simulated post-start") as ei:
        run_agent(None, llm, resume_task_id=task.task_id, db_path=db)
    assert not isinstance(ei.value, ResumeValidationError)
    msg = format_resume_failure("runtime", str(ei.value))
    assert "No GPU budget was spent" not in msg
    assert "may already have executed" in msg


class _NullCtx:
    def __enter__(self):
        return MagicMock()

    def __exit__(self, *exc):
        return False

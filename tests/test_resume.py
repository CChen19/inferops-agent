"""CPU tests for resume parsing and reflector status text. Do not import app."""

from __future__ import annotations

from inferops.resume import (
    format_reflector_update,
    format_resume_help,
    is_resume_command,
    parse_resume_task_id,
)


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
    assert is_resume_command("resume nope") is True
    assert is_resume_command("resume abcdabcdabcd") is True
    assert is_resume_command("please resume later") is False
    assert is_resume_command("chat_short maximize throughput") is False


def test_format_resume_help_names_cli_and_chat():
    text = format_resume_help("abcdabcdabcd")
    assert "resume abcdabcdabcd" in text
    assert "--resume-task abcdabcdabcd" in text
    assert "inferops agent" in text


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

"""User-visible resume and reflector-status helpers (no Chainlit import)."""

from __future__ import annotations

import re

_TASK_ID = r"[0-9a-fA-F]{12}"
_BARE_ID = re.compile(rf"^{_TASK_ID}$")
_RESUME_WITH_ID = re.compile(
    rf"^(?:resume(?:-task)?)\s+({_TASK_ID})\s*$",
    re.IGNORECASE,
)
_RESUME_COMMAND = re.compile(
    r"^(?:resume(?:-task)?)(?:\s+\S+)?\s*$",
    re.IGNORECASE,
)


class ResumeValidationError(ValueError):
    """Resume failed before any baseline / graph work — no GPU budget spent.

    Raised only for pre-execution checks (missing task, unconfirmed dump).
    Failures after ``prepare_initial_state`` or ``graph.invoke`` must stay
    ordinary exceptions so the UI does not claim "no GPU was spent".
    """


def parse_resume_task_id(text: str) -> str | None:
    """Return a 12-hex task id from a resume chat line, or None.

    Accepts ``resume <id>``, ``resume-task <id>``, or a bare 12-hex id when
    that is the whole message. Does not treat ordinary sentences as resume.
    """
    s = (text or "").strip()
    if not s:
        return None
    if _BARE_ID.fullmatch(s):
        return s.lower()
    m = _RESUME_WITH_ID.fullmatch(s)
    if m:
        return m.group(1).lower()
    return None


def is_resume_command(text: str) -> bool:
    """True when the whole message is a resume request, with or without an id."""
    s = (text or "").strip()
    if parse_resume_task_id(s):
        return True
    return bool(_RESUME_COMMAND.fullmatch(s))


def format_resume_help(task_id: str) -> str:
    """How to continue this task from CLI ``--resume-task`` or chat ``resume``."""
    tid = (task_id or "").strip()
    return (
        f"Resume later with `resume {tid}` in chat, "
        f"or `inferops agent --resume-task {tid}` in the CLI."
    )


def format_resume_failure(kind: str, detail: str) -> str:
    """Honest resume-failure copy. Only ``validation`` may claim no GPU spend."""
    text = (detail or "").strip() or "unknown error"
    if kind == "validation":
        return (
            f"**Cannot resume.** {text}\n\n"
            "No GPU budget was spent."
        )
    return (
        f"**Resume failed.** {text}\n\n"
        "The run may already have executed (baseline or graph work may have "
        "started). Check the task record — do not assume no GPU was spent."
    )


def format_reflector_update(
    next_action: str,
    *,
    stop_reason: str = "",
    streak: int = 0,
) -> str:
    """User-visible reflector status. Distinguishes continue/remeasure/rollback/stop."""
    action = (next_action or "").strip().lower()
    reason = (stop_reason or "").strip()
    if action == "stop":
        extra = f" {reason}" if reason else ""
        return f"⏹ **Done.**{extra}".rstrip()
    if action == "remeasure":
        return (
            f"🔁 **Reflector:** remeasure — repeating this candidate "
            f"(no-improvement streak {streak})."
        )
    if action == "rollback":
        return (
            f"↩️ **Reflector:** rollback — restoring the previous config "
            f"(no-improvement streak {streak})."
        )
    if action == "continue":
        return (
            f"🔄 **Reflector:** continue — no-improvement streak {streak}, "
            "trying next hypothesis…"
        )
    unknown = action or "unknown"
    extra = f" {reason}" if reason else ""
    return (
        f"**Reflector:** `{unknown}` — no-improvement streak {streak}.{extra}"
    )

"""Tool: write_report_section — append a section to the agent's running report."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from inferops.observability import span

_DEFAULT_REPORT = "reports/agent_report.md"


class WriteReportInput(BaseModel):
    section_title: str = Field(description="Markdown H2 heading for this section, e.g. 'Iteration 3 Results'.")
    content: str = Field(description="Markdown body text for the section.")
    report_path: str = Field(
        default=_DEFAULT_REPORT,
        description="Path to the report file. Created if it doesn't exist.",
    )


class WriteReportOutput(BaseModel):
    report_path: str
    section_title: str
    bytes_written: int
    total_sections: int


def write_report_section(inp: WriteReportInput) -> WriteReportOutput:
    """
    Append a section to the agent's running Markdown report.

    Creates the report file if it doesn't exist. Each call adds one H2 section.
    Designed for incremental write — the agent can call this after each experiment
    to build up the report without holding the full document in context.
    """
    path = Path(inp.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    section = f"\n## {inp.section_title}\n\n{inp.content}\n"

    with span("tool.write_report_section", {"section": inp.section_title, "path": inp.report_path}):
        with open(path, "a", encoding="utf-8") as f:
            if path.stat().st_size == 0 if path.exists() else True:
                # New file — write header
                if not path.exists() or path.stat().st_size == 0:
                    f.write("# InferOps Agent Report\n")
            f.write(section)
        bytes_written = len(section.encode())

        # Count existing H2 sections
        text = path.read_text(encoding="utf-8")
        total_sections = text.count("\n## ")

    return WriteReportOutput(
        report_path=str(path),
        section_title=inp.section_title,
        bytes_written=bytes_written,
        total_sections=total_sections,
    )

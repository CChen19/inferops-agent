"""Unit tests for write_report_section tool."""

from __future__ import annotations

from inferops.tools.write_report import WriteReportInput, write_report_section


def test_creates_file_with_header(tmp_report):
    out = write_report_section(WriteReportInput(
        section_title="Test Section",
        content="Some findings here.",
        report_path=str(tmp_report),
    ))

    assert tmp_report.exists()
    text = tmp_report.read_text()
    assert "# InferOps Agent Report" in text
    assert "## Test Section" in text
    assert "Some findings here." in text
    assert out.total_sections == 1


def test_appends_multiple_sections(tmp_report):
    for i in range(3):
        write_report_section(WriteReportInput(
            section_title=f"Section {i}",
            content=f"Content {i}",
            report_path=str(tmp_report),
        ))

    text = tmp_report.read_text()
    assert text.count("## Section") == 3


def test_returns_correct_section_count(tmp_report):
    write_report_section(WriteReportInput(section_title="A", content="x", report_path=str(tmp_report)))
    out = write_report_section(WriteReportInput(section_title="B", content="y", report_path=str(tmp_report)))
    assert out.total_sections == 2


def test_creates_parent_dir(tmp_path):
    nested = tmp_path / "sub" / "report.md"
    write_report_section(WriteReportInput(
        section_title="Hello",
        content="World",
        report_path=str(nested),
    ))
    assert nested.exists()

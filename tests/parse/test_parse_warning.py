"""ParseWarning is constructible and carries the documented fields."""

from __future__ import annotations

from molbuilder.parse import ParseWarning


def test_parsewarning_constructible_with_documented_fields():
    """ParseWarning is a frozen dataclass per model/parse.md § 2."""
    w = ParseWarning(
        source="/path/to/file.out",
        line_no=42,
        snippet="suspect line",
        error="parse failure",
        category="scf-format",
    )
    assert w.source == "/path/to/file.out"
    assert w.line_no == 42
    assert w.snippet == "suspect line"
    assert w.error == "parse failure"
    assert w.category == "scf-format"


def test_parsewarning_allows_none_for_line_no_and_snippet():
    """line_no and snippet are Optional per the dataclass — None
    must be accepted (e.g., parser-level errors without a specific
    source line)."""
    w = ParseWarning(source="<text>", line_no=None, snippet=None,
                     error="malformed payload", category="json")
    assert w.line_no is None
    assert w.snippet is None

"""A deck loaded where an output belongs is told so.

`model/parse.md` § 1: when no parser claims a file the registry asks each
parser's `footgun_hint_for`, and its answer joins the refusal.  The refusal
itself, and the formats it lists, are `tests/parse/test_registry.py`'s.
"""

from __future__ import annotations

from molbuilder.parse import UnknownFormatError, detect as detect_parser


def test_unknown_format_fdf_suggests_out_file(tmp_path):
    """A .fdf filename means the user loaded the SIESTA INPUT, not
    the output.  The error must call this out explicitly and point
    at the corresponding .out and .molwatch.log files."""
    p = tmp_path / "siesta.fdf"
    p.write_text("SystemName test\nNumberOfAtoms 2\n")  # FDF-shaped, not output
    try:
        detect_parser(str(p))
    except UnknownFormatError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected UnknownFormatError")
    assert "INPUT" in msg or "input" in msg
    assert "siesta.out" in msg
    assert ".molwatch.log" in msg

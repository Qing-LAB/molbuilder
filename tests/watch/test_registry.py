"""Parser-registry tests.

Verifies that ``detect_parser`` picks the right parser based on file
content, and that ``UnknownFormatError`` fires for unrecognised input.
"""

from __future__ import annotations

import pytest

from molbuilder.parse import UnknownFormatError, detect as detect_parser
from molbuilder.parse.registry import _registered_file_parsers as _list_file_parsers


def test_detect_unknown_format(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some random text\n")
    with pytest.raises(UnknownFormatError):
        detect_parser(str(p))


def test_unknown_format_error_lists_supported(tmp_path):
    """Error message must enumerate every registered format with its
    hint -- that's how users learn which file to grab."""
    p = tmp_path / "garbage.txt"
    p.write_text("just some random text\n")
    try:
        detect_parser(str(p))
    except UnknownFormatError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected UnknownFormatError")
    assert "SIESTA" in msg
    assert "PySCF" in msg
    # Hints should appear too.
    assert "_optim.xyz" in msg


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
    assert "siesta-run<N>.out" in msg
    assert ".molwatch.log" in msg


def test_unknown_format_generic_hint_points_at_docs(tmp_path):
    """For files that don't match either of the targeted hints, the
    error message must still steer the user somewhere useful -- the
    README and the spec doc both have a debug section."""
    p = tmp_path / "mystery.dat"
    p.write_text("not a recognised format\n")
    try:
        detect_parser(str(p))
    except UnknownFormatError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected UnknownFormatError")
    assert "README" in msg or "docs/model/parse.md" in msg


def test_registry_lists_all_parsers():
    names = [c.name for c in _list_file_parsers()]
    assert "siesta" in names
    assert "pyscf" in names

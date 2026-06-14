"""L1 unit tests for the SIESTA .fdf Geometry.Constraints parser.

``molbuilder/parsers/_sidecar.py::read_frozen_atoms_from_siesta_fdf``
recovers per-atom frozen indices from a SIESTA ``.fdf`` input file
paired with a trajectory output.  Fallback path for runs that have
no ``.molstruct.json`` sidecar — without it, the trajectory
inspector's "Hide frozen atoms" toggle stays hidden because no
indices are exposed to the JS even when the parser detected
constrained atoms via the SIESTA output.

Per docs/protocols/code-audit.md § 3.3 (cross-source-of-truth gap):
when a UI feature requires data X, trace every code path that
could provide X; verify each is honored.  This file pins that the
fdf path IS honored when the sidecar path comes back empty.

Tests cover:

  * Single-atom enumeration (`position 5`).
  * Multi-atom enumeration (`position 5 7 12`).
  * Range form (`position from 1 to 5`).
  * Range with step (`position from 1 to 9 step 2`).
  * Mixed enumeration + range across multiple lines.
  * 1-based to 0-based index conversion.
  * Block-marker case insensitivity.
  * Commented-out lines are ignored.
  * No ``.fdf`` paired → empty set (not an error).
  * No Constraints block → empty set.
  * Malformed token in enumeration → silent skip of that line.
  * Sibling-file discovery (same-stem + single-fdf-in-dir fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.parsers._sidecar import (
    read_frozen_atoms_from_siesta_fdf,
    _siesta_fdf_path_for,
)


# --------------------------------------------------------------------- #
#  Pairing logic (sidecar-file discovery)                                #
# --------------------------------------------------------------------- #


def test_same_stem_fdf_is_picked(tmp_path):
    out = tmp_path / "run.out"
    out.write_text("dummy\n")
    fdf = tmp_path / "run.fdf"
    fdf.write_text("dummy\n")
    assert _siesta_fdf_path_for(str(out)) == str(fdf)


def test_molwatch_log_strips_double_suffix(tmp_path):
    log = tmp_path / "run.molwatch.log"
    log.write_text("dummy\n")
    fdf = tmp_path / "run.fdf"
    fdf.write_text("dummy\n")
    assert _siesta_fdf_path_for(str(log)) == str(fdf)


def test_no_fdf_paired_returns_none(tmp_path):
    out = tmp_path / "run.out"
    out.write_text("dummy\n")
    assert _siesta_fdf_path_for(str(out)) is None


def test_single_other_fdf_in_dir_falls_back(tmp_path):
    """A common layout: one-run folder with mismatched basenames.
    When there's exactly one ``.fdf`` in the directory, use it."""
    (tmp_path / "siesta.out").write_text("dummy\n")
    (tmp_path / "input.fdf").write_text("dummy\n")
    assert _siesta_fdf_path_for(
        str(tmp_path / "siesta.out")) == str(tmp_path / "input.fdf")


def test_multiple_fdfs_in_dir_returns_none(tmp_path):
    """Bail rather than guess wrong when the stem doesn't match
    and multiple ``.fdf`` files share the directory."""
    (tmp_path / "run.out").write_text("dummy\n")
    (tmp_path / "a.fdf").write_text("dummy\n")
    (tmp_path / "b.fdf").write_text("dummy\n")
    assert _siesta_fdf_path_for(str(tmp_path / "run.out")) is None


# --------------------------------------------------------------------- #
#  Block parsing                                                         #
# --------------------------------------------------------------------- #


def _write_fdf(tmp_path: Path, body: str) -> str:
    """Helper: write a minimal ``run.fdf`` with the given block body,
    return the path of the paired ``run.out`` for the parser call."""
    fdf = tmp_path / "run.fdf"
    fdf.write_text(body)
    out = tmp_path / "run.out"
    out.write_text("dummy\n")
    return str(out)


def test_single_atom_enumeration(tmp_path):
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position 5
%endblock Geometry.Constraints
""")
    # 1-based 5 → 0-based 4.
    assert read_frozen_atoms_from_siesta_fdf(out) == {4}


def test_multi_atom_enumeration(tmp_path):
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position 5 7 12
%endblock Geometry.Constraints
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {4, 6, 11}


def test_range_form(tmp_path):
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position from 1 to 5
%endblock Geometry.Constraints
""")
    # 1..5 inclusive → 0..4 inclusive.
    assert read_frozen_atoms_from_siesta_fdf(out) == {0, 1, 2, 3, 4}


def test_range_with_step(tmp_path):
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position from 1 to 9 step 2
%endblock Geometry.Constraints
""")
    # 1, 3, 5, 7, 9 → 0, 2, 4, 6, 8.
    assert read_frozen_atoms_from_siesta_fdf(out) == {0, 2, 4, 6, 8}


def test_mixed_lines_across_block(tmp_path):
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position from 1 to 3
position 7
position 10 11
%endblock Geometry.Constraints
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {0, 1, 2, 6, 9, 10}


def test_block_marker_case_insensitive(tmp_path):
    out = _write_fdf(tmp_path, """\
%BLOCK GEOMETRY.CONSTRAINTS
POSITION from 1 to 3
%ENDBLOCK GEOMETRY.CONSTRAINTS
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {0, 1, 2}


def test_commented_lines_ignored(tmp_path):
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
# comment-out: position 99
position 5 # inline comment
%endblock Geometry.Constraints
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {4}


def test_non_position_keywords_skipped(tmp_path):
    """``stress`` and ``cellangle`` are valid Constraints entries
    but don't translate to per-atom freezes — silently ignore."""
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
stress 1.0 0 0 0
position 3
cellangle 1 0
%endblock Geometry.Constraints
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {2}


def test_no_block_returns_empty(tmp_path):
    out = _write_fdf(tmp_path, """\
SystemName trial
SystemLabel run
%block Wibble
position 5
%endblock Wibble
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == set()


def test_no_fdf_paired_returns_empty(tmp_path):
    """No error, just empty — frozen-atom data is optional."""
    out = tmp_path / "run.out"
    out.write_text("dummy\n")
    assert read_frozen_atoms_from_siesta_fdf(str(out)) == set()


def test_malformed_enumeration_token_skipped(tmp_path):
    """A bad token bails out the WHOLE line (don't half-freeze) but
    other lines still contribute."""
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position 3 xxx 7
position 11
%endblock Geometry.Constraints
""")
    # The "3 xxx 7" line is skipped wholesale; "position 11" still
    # registers atom index 10.
    assert read_frozen_atoms_from_siesta_fdf(out) == {10}


def test_reverse_range_skipped(tmp_path):
    """``from 5 to 1`` is malformed; skip the line rather than
    enumerate negatively."""
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position from 5 to 1
position 8
%endblock Geometry.Constraints
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {7}


def test_zero_step_skipped(tmp_path):
    """``step 0`` would loop infinitely; bail."""
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position from 1 to 5 step 0
position 8
%endblock Geometry.Constraints
""")
    assert read_frozen_atoms_from_siesta_fdf(out) == {7}


def test_unreadable_fdf_returns_empty(tmp_path, monkeypatch):
    """An I/O error on the fdf is silent — frozen-atom data is
    optional and must not break trajectory loading."""
    out = _write_fdf(tmp_path, """\
%block Geometry.Constraints
position 5
%endblock Geometry.Constraints
""")

    # Replace open with one that raises OSError on the fdf path.
    real_open = open

    def patched_open(path, *args, **kwargs):
        if str(path).endswith(".fdf"):
            raise OSError("simulated read failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)
    assert read_frozen_atoms_from_siesta_fdf(out) == set()

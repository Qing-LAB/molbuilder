"""Regression pin for the 2026-06-14 hide-frozen-toggle disappearance.

ROOT-CAUSE LESSON: a heuristic that ``looks like a contract`` is a
bug waiting to happen.  The pre-fix parser searched for a paired
.fdf by filename stem to find the frozen-atom indices.  That worked
on the happy-path fixture (``foo.out`` ↔ ``foo.fdf``, one .fdf per
directory) and silently broke on EVERY real wrapper-launched run
(``foo-stage1-run3.out`` doesn't pair with ``foo-stage1.fdf``
because the strip didn't know about ``-run<N>``; staged projects
have multiple sibling .fdf files which busted the multi-fdf
fallback too).  The "Hide frozen atoms" toggle was hidden on every
single Results-tab view that used a wrapper-launched .out.

The fix: SIESTA ALREADY echoes the resolved constraints into the
.out itself via::

    siesta: Constraints applied in the following order:
    siesta: Constraint (N): pos
      [ start -- end, start -- end ]

So the parser reads them directly from the .out -- ZERO filename
heuristics.  The data lives in the same file the Results-tab UI
reads.  These tests pin:

  1. The .out-direct extractor returns the right indices for the
     real BDT-stage-1 format (multi-range comma-separated).
  2. The full SiestaParser populates ``runtime_info.frozen_atoms``
     when the .out has the echo -- regardless of what the .fdf
     filename is or whether a paired .fdf exists at all.
  3. An empty / absent constraints section returns the empty set
     (toggle correctly hidden).
  4. The .out-direct path is preferred over both the sidecar and
     the .fdf-pairing fallback.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from molbuilder.parsers._sidecar import (
    read_frozen_atoms_from_siesta_out,
)
from molbuilder.parsers.siesta import SiestaParser


# Real SIESTA v5 echo, verified verbatim against the BDT-stage-1
# .out (lines 1104-1110 of
# projects/BDT/optimization/BDT-withAuJunction/
# siesta-BDT-withAuJunction-stage1-run0.out).
_REAL_BDT_ECHO = """\
diag: some other stuff before the section

siesta: Constraints applied in the following order:
siesta: Constraint (20): pos
  [ 88 -- 107 ]
siesta: Constraint (20): pos
  [ 108 -- 112, 188 -- 202 ]
siesta: Constraint (10): pos
  [ 203 -- 212 ]


ts: irrelevant trailing stuff
"""

# Expected 0-based set from the BDT echo.  Computed by hand:
#   1-based 88..107 -> 0-based 87..106
#   1-based 108..112 + 188..202 -> 0-based 107..111 + 187..201
#   1-based 203..212 -> 0-based 202..211
_BDT_EXPECTED = set(range(87, 112)) | set(range(187, 202)) | set(range(202, 212))


class TestExtractFromOut:
    """The .out-direct extractor: zero filename heuristics."""

    def test_real_bdt_echo_extracts_50_indices(self, tmp_path):
        out = tmp_path / "any-name-the-wrapper-picked.out"
        out.write_text(_REAL_BDT_ECHO)
        result = read_frozen_atoms_from_siesta_out(str(out))
        assert len(result) == 50, (
            f"expected 50 frozen atoms from the BDT echo; "
            f"got {len(result)}"
        )
        assert result == _BDT_EXPECTED

    def test_no_constraints_section_returns_empty(self, tmp_path):
        """A .out that doesn't have the constraints echo (an
        unconstrained run) returns the empty set -- and the UI
        correctly hides the toggle."""
        out = tmp_path / "free.out"
        out.write_text("siesta: some other text\nsiesta: more text\n")
        assert read_frozen_atoms_from_siesta_out(str(out)) == set()

    def test_empty_constraint_section(self, tmp_path):
        """Section header present but no Constraint lines follow."""
        out = tmp_path / "empty-section.out"
        out.write_text(dedent("""\
            siesta: Constraints applied in the following order:

            unrelated next section starts
        """))
        assert read_frozen_atoms_from_siesta_out(str(out)) == set()

    def test_missing_file_returns_empty(self, tmp_path):
        """OSError on read -> empty set (parser shouldn't crash)."""
        result = read_frozen_atoms_from_siesta_out(
            str(tmp_path / "does-not-exist.out")
        )
        assert result == set()

    def test_single_range_no_comma(self, tmp_path):
        """A single ``[ N -- M ]`` line (no comma-separated parts)."""
        out = tmp_path / "single.out"
        out.write_text(dedent("""\
            siesta: Constraints applied in the following order:
            siesta: Constraint (3): pos
              [ 5 -- 7 ]
        """))
        result = read_frozen_atoms_from_siesta_out(str(out))
        # 1-based 5,6,7 -> 0-based 4,5,6
        assert result == {4, 5, 6}

    def test_multiple_ranges_comma_separated(self, tmp_path):
        out = tmp_path / "multi.out"
        out.write_text(dedent("""\
            siesta: Constraints applied in the following order:
            siesta: Constraint (5): pos
              [ 1 -- 2, 5 -- 7 ]
        """))
        result = read_frozen_atoms_from_siesta_out(str(out))
        # 1-based 1,2,5,6,7 -> 0-based 0,1,4,5,6
        assert result == {0, 1, 4, 5, 6}


class TestFullParserNoFilenameHeuristic:
    """End-to-end: the full SiestaParser populates
    ``runtime_info.frozen_atoms`` from the .out's echo, regardless
    of what the file is named.  No paired .fdf required."""

    def _make_minimal_out(self, tmp_path: Path, name: str) -> Path:
        # Synthesize the absolute minimum a SIESTA .out needs for
        # SiestaParser to run + the constraints echo it should pick
        # up.  We don't need real frames -- just the runtime_info
        # population path.
        out = tmp_path / name
        out.write_text(dedent("""\
            * Running on    1 nodes in parallel.
            siesta: Constraints applied in the following order:
            siesta: Constraint (3): pos
              [ 5 -- 7 ]

            other content
        """))
        return out

    def test_run_index_suffix_in_name_does_not_matter(self, tmp_path):
        """The pre-fix bug: ``foo-stage1-run3.out`` failed to pair
        with ``foo-stage1.fdf``.  Now the parser doesn't care --
        the data comes from the .out itself."""
        out = self._make_minimal_out(
            tmp_path,
            "foo-stage1-run3.out"
        )
        traj = SiestaParser.parse(str(out))
        frozen = traj.runtime_info.get("frozen_atoms")
        assert frozen is not None, (
            "runtime_info[frozen_atoms] must be populated from the "
            ".out echo even when the filename has the wrapper's "
            "-run<N> suffix"
        )
        assert sorted(frozen) == [4, 5, 6]

    def test_no_fdf_in_directory_at_all(self, tmp_path):
        """The pre-fix bug also failed when there was NO .fdf
        anywhere.  .out-direct path is unaffected."""
        out = self._make_minimal_out(tmp_path, "lonely.out")
        traj = SiestaParser.parse(str(out))
        frozen = traj.runtime_info.get("frozen_atoms")
        assert frozen is not None
        assert sorted(frozen) == [4, 5, 6]

    def test_multi_fdf_in_directory(self, tmp_path):
        """Three .fdf siblings (staged-relaxation layout) used to
        break the multi-fdf-fallback heuristic.  .out-direct path
        is unaffected."""
        for s in (1, 2, 3):
            (tmp_path / f"foo-stage{s}.fdf").write_text("")
        out = self._make_minimal_out(
            tmp_path,
            "foo-stage1-run0.out"
        )
        traj = SiestaParser.parse(str(out))
        frozen = traj.runtime_info.get("frozen_atoms")
        assert frozen is not None
        assert sorted(frozen) == [4, 5, 6]


class TestArchitecturalContract:
    """Documents the source-of-truth ordering for the parser.

    These tests are part of the regression pin's *contract* layer:
    they pin not just behaviour but the rule ``the .out's echo is
    authoritative, fallbacks are emergency-only.``"""

    def test_out_echo_wins_over_paired_fdf_disagreement(self, tmp_path):
        """If the .out and the .fdf disagree on which atoms are
        frozen (e.g. user edited the .fdf after the run finished),
        the .out's data is the truth."""
        # .out says [5, 6, 7] are frozen (1-based -> 0-based 4,5,6).
        out = tmp_path / "foo-stage1-run0.out"
        out.write_text(dedent("""\
            * Running on    1 nodes in parallel.
            siesta: Constraints applied in the following order:
            siesta: Constraint (3): pos
              [ 5 -- 7 ]
        """))
        # .fdf disagrees -- says [10, 11, 12] are frozen.
        (tmp_path / "foo-stage1.fdf").write_text(dedent("""\
            NumberOfAtoms 20
            %block Geometry.Constraints
            position 10 11 12
            %endblock Geometry.Constraints
        """))
        traj = SiestaParser.parse(str(out))
        frozen = traj.runtime_info.get("frozen_atoms")
        # .out wins -> 0-based 4,5,6 (NOT 9,10,11).
        assert sorted(frozen) == [4, 5, 6], (
            "The .out's constraints echo MUST win over a disagreeing "
            ".fdf; .fdf is a fallback for when the .out lacks the echo, "
            "not a co-equal source.  If you got the .fdf's [9,10,11], "
            "the source-of-truth ordering regressed."
        )

    def test_fdf_fallback_when_out_has_no_echo(self, tmp_path):
        """Defensive: when the .out somehow lacks the constraints
        echo (truncated, mid-run flush), the .fdf fallback still
        works.  Belt-and-suspenders."""
        out = tmp_path / "truncated-foo.out"
        out.write_text("* Running on 1 nodes\nsome content but no echo\n")
        (tmp_path / "truncated-foo.fdf").write_text(dedent("""\
            NumberOfAtoms 20
            %block Geometry.Constraints
            position 10 11 12
            %endblock Geometry.Constraints
        """))
        traj = SiestaParser.parse(str(out))
        frozen = traj.runtime_info.get("frozen_atoms")
        # .fdf path -> 1-based 10,11,12 -> 0-based 9,10,11.
        assert sorted(frozen) == [9, 10, 11]

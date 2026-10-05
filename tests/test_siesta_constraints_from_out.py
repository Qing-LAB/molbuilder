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
heuristics, and since 2026-10-04 nothing else: the sidecar and the
paired-.fdf fallbacks went with plan B12 (W56 4c), the .out being the
run's own statement of what the engine held (`model/parse.md` § 5.3).
These tests pin the echo's grammar: the real BDT-stage-1 format
(multi-range, comma-separated), and an empty or absent section read as
the empty set (toggle correctly hidden).  The parser filling
``runtime_info.frozen_atoms`` from it is pinned on a measured run
(`test_structure_info_bridge.py`).
"""
from __future__ import annotations

from textwrap import dedent

from molbuilder.parse.engines.siesta import (
    read_frozen_atoms_from_siesta_out,
)


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

    def test_streaming_stops_at_section_end_not_eof(self, tmp_path,
                                                    monkeypatch):
        """2026-06-14 robustness pass: the reader streams the file
        line-by-line and STOPS as soon as the constraints section ends
        -- a 100 MB .out used to OOM the pre-streaming implementation.

        Pinned by COUNTING BYTES READ through the parser's own ``open``
        (B-6, 2026-08-13).  The RSS probe this replaces read
        ``ru_maxrss``, a process-lifetime HIGH-WATER: whenever any
        earlier test in the process had already peaked higher, "growth"
        read 0 and the slurp regression it guarded sailed through --
        false-passing in exactly the dangerous direction -- while a
        160 MB fixture was written on every run to feed it."""
        from molbuilder.parse.engines import siesta as _siesta
        body_head = dedent("""\
            siesta: Constraints applied in the following order:
            siesta: Constraint (3): pos
              [ 5 -- 7 ]

            siesta: program continues with normal SCF output...
        """)
        out = tmp_path / "huge.out"
        with open(out, "w") as fh:
            fh.write(body_head)
            line_pat = "   scf:    {0}  -1234.567890  -1234.567890  0.0001\n"
            for i in range(20_000):
                fh.write(line_pat.format(i))
        total = out.stat().st_size
        assert total > 500_000, "fixture too small to prove early exit"

        consumed = {"bytes": 0}
        real_open = open

        class _Counting:
            def __init__(self, f):
                self._f = f

            def __iter__(self):
                for line in self._f:
                    consumed["bytes"] += len(line)
                    yield line

            def readline(self, *a, **k):
                s = self._f.readline(*a, **k)
                consumed["bytes"] += len(s)
                return s

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

            def __getattr__(self, name):
                return getattr(self._f, name)

        def counting_open(file, *a, **k):
            f = real_open(file, *a, **k)
            return _Counting(f) if str(file) == str(out) else f

        monkeypatch.setattr(_siesta, "open", counting_open, raising=False)
        result = read_frozen_atoms_from_siesta_out(str(out))
        # 0-based 4, 5, 6.
        assert result == {4, 5, 6}, f"expected {{4,5,6}}; got {result}"
        assert 0 < consumed["bytes"] < total // 10, (
            f"read {consumed['bytes']} of {total} bytes -- the parser no "
            f"longer stops at the section end (or the counter never saw "
            f"the open)")

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


# `TestFullParserNoFilenameHeuristic`, `TestArchitecturalContract` and
# `TestTheSidecarPathLearnsTheSameLesson` retired 2026-10-04 (W56 4c, plan
# B12): they pinned the precedence among the .out echo, a sidecar beside
# the output and the paired .fdf, and the sidecar lookup's own guards -- on
# outputs, decks and sidecars a test wrote.  The .out's echo is now the
# one source, its grammar pinned above and the parser's reading of it on a
# measured run (`test_structure_info_bridge.py`); a PySCF run states its
# own in its progress log (`tests/data/held_atoms.toml`).

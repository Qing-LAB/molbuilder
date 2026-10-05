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


from molbuilder.parse.engines.siesta import (
    read_frozen_atoms_from_siesta_out,
)


class TestExtractFromOut:
    """The .out-direct extractor: zero filename heuristics."""

    # Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
    # 6 tests here read a constraint echo invented as text, or one
    # pasted out of a real output (`process/testing.md` § 6).

    def test_missing_file_returns_empty(self, tmp_path):
        """OSError on read -> empty set (parser shouldn't crash)."""
        result = read_frozen_atoms_from_siesta_out(
            str(tmp_path / "does-not-exist.out")
        )
        assert result == set()


# `TestFullParserNoFilenameHeuristic`, `TestArchitecturalContract` and
# `TestTheSidecarPathLearnsTheSameLesson` retired 2026-10-04 (W56 4c, plan
# B12): they pinned the precedence among the .out echo, a sidecar beside
# the output and the paired .fdf, and the sidecar lookup's own guards -- on
# outputs, decks and sidecars a test wrote.  The .out's echo is now the
# one source, its grammar pinned above and the parser's reading of it on a
# measured run (`test_structure_info_bridge.py`); a PySCF run states its
# own in its progress log (`tests/data/held_atoms.toml`).

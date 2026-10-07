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
heuristics, the .out being the run's own statement of what the engine
held (`model/parse.md` § 5.3).  The parser filling
``runtime_info.frozen_atoms`` from it is pinned on a measured run
(`test_siesta_flat_run_e2e.py`).
"""
from __future__ import annotations


from molbuilder.parse.engines.siesta import (
    read_frozen_atoms_from_siesta_out,
)


class TestExtractFromOut:
    """The .out-direct extractor: zero filename heuristics."""


    def test_missing_file_returns_empty(self, tmp_path):
        """OSError on read -> empty set (parser shouldn't crash)."""
        result = read_frozen_atoms_from_siesta_out(
            str(tmp_path / "does-not-exist.out")
        )
        assert result == set()

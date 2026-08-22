"""Validates the archived spec's (docs/archive/old_docs/tabs/spectra/spec.md) § 5.1 — the spectra atom-index contract.

Invariants 1–2 (the free_atom_idxs partition + eigenvector length) are enforced
by ``SpectraResults.__post_init__``; these pin that enforcement, including the
out-of-range gap that a count-only check used to miss.  Invariant 3 (the frontend
``free_atom_idxs`` scatter) is a JS test (``test_spectra_scatter_js.py``);
invariant 4 (1-based display via ``atomIndexModel``) is the shared _atom-index
rule and is not spectra-specific.
"""
import dataclasses

import numpy as np
import pytest

from molbuilder.spectra import ModeData
from tests.spectra._helpers import _make_mode, _make_results


def _mode_with_rows(n_rows: int) -> ModeData:
    ev = np.zeros((n_rows, 3))
    return dataclasses.replace(_make_mode(),
                               eigenvector_canonical=ev, eigenvector_display=ev)


def _results(*, n_atoms_total, free, frozen, n_free_rows=None):
    """A valid-shaped SpectraResults with the partition fields overridden;
    ``dataclasses.replace`` re-runs ``__post_init__`` so violations raise."""
    if n_free_rows is None:
        n_free_rows = len(free)
    return dataclasses.replace(
        _make_results(),
        n_atoms_total=n_atoms_total,
        free_atom_idxs=list(free),
        frozen_atom_idxs=list(frozen),
        modes=[_mode_with_rows(n_free_rows)],
    )


# --- Invariant 1: free ⊎ frozen partition range(n_atoms_total) --------------

def test_valid_partition_with_frozen_atoms_constructs():
    # n=3, free=[0,2], frozen=[1] -> partitions {0,1,2}; 2 free eigenvector rows.
    r = _results(n_atoms_total=3, free=[0, 2], frozen=[1])
    assert r.free_atom_idxs == [0, 2] and r.frozen_atom_idxs == [1]


def test_out_of_range_free_index_raises():
    # free=[0,1,5], frozen=[], n=3: count 3==3 + no overlap (the OLD check
    # passed), but 5 is out of range -> the true partition check must catch it.
    with pytest.raises(ValueError, match="partition"):
        _results(n_atoms_total=3, free=[0, 1, 5], frozen=[], n_free_rows=3)


def test_uncovered_index_raises():
    # free=[0], frozen=[], n=2 -> atom 1 is covered by neither.
    with pytest.raises(ValueError, match="partition"):
        _results(n_atoms_total=2, free=[0], frozen=[], n_free_rows=1)


def test_free_frozen_overlap_raises():
    # atom 1 in both free and frozen.
    with pytest.raises(ValueError, match="overlap"):
        _results(n_atoms_total=2, free=[0, 1], frozen=[1], n_free_rows=2)


# --- Invariant 2: len(eigenvector_*) == len(free_atom_idxs) -----------------

def test_eigenvector_rows_must_match_n_free():
    # valid partition (free=[0,1], n=2) but the eigenvector carries 3 rows.
    with pytest.raises(ValueError):
        _results(n_atoms_total=2, free=[0, 1], frozen=[], n_free_rows=3)

"""SMILES builder + siesta module smoke tests."""

from __future__ import annotations


import pytest

from molbuilder.siesta import SiestaConfig
from molbuilder.smiles import build_from_smiles


# --------------------------------------------------------------------- #
#  SMILES builder (RDKit-dependent)                                     #
# --------------------------------------------------------------------- #


def _smiles_or_skip(smiles: str, **kw):
    try:
        return build_from_smiles(smiles, **kw)
    except ImportError as e:
        pytest.skip(f"RDKit not installed: {e}")


def test_smiles_benzene_planarity():
    s = _smiles_or_skip("c1ccccc1", title="benzene")
    assert s.n_atoms == 12
    assert s.elements.count("C") == 6
    assert s.elements.count("H") == 6
    z_spread = s.positions[:, 2].max() - s.positions[:, 2].min()
    assert z_spread < 0.1, z_spread


def test_smiles_xyz_header_count():
    s = _smiles_or_skip("c1ccccc1")
    xyz = s.to_xyz()
    assert int(xyz.splitlines()[0]) == 12


def test_smiles_bdt_has_two_sulphurs():
    s2 = _smiles_or_skip("Sc1ccc(S)cc1", title="bdt")
    assert "S" in s2.elements
    assert s2.elements.count("S") == 2


# --------------------------------------------------------------------- #
#  the SIESTA deck's block size                                         #
# --------------------------------------------------------------------- #


def test_block_size_auto_pick_rule():
    """The single-rank baseline (``mpi_np = 1``): the per-rank orbital
    derivation (job-contracts.md § 3.2/§ 3.3) has one rank to divide by, so
    the conservative ladder capped at 8 applies (a single rank ignores
    BlockSize anyway)."""
    from molbuilder.siesta.input import _auto_block_size
    # Known thresholds: each step at a power-of-2 boundary.
    assert _auto_block_size(2, 1)  == 1
    assert _auto_block_size(3, 1)  == 1
    assert _auto_block_size(4, 1)  == 2
    assert _auto_block_size(7, 1)  == 2
    assert _auto_block_size(8, 1)  == 4
    assert _auto_block_size(15, 1) == 4
    assert _auto_block_size(16, 1) == 8
    assert _auto_block_size(50, 1) == 8
    assert _auto_block_size(500, 1) == 8
    assert _auto_block_size(81, 1) == 8
    # Ladder invariant: the single-rank baseline never exceeds n_atoms
    # (each ladder step sits at/below its threshold).  Note propor IMAX=0
    # is NOT a BlockSize trigger -- it is a matel_table/mpi_np issue
    # (2026-05-28 empirical sweep).
    for n in range(1, 64):
        assert _auto_block_size(n, 1) <= n, n


def test_block_size_honours_orbital_rank_constraint():
    """With mpi_np set, the cap is stated in ORBITALS: SIESTA
    distributes ORBITALS across ranks, so ``BlockSize <= min(256,
    floor(n_orbitals_est / mpi_np))`` with n_orbitals_est = 10 *
    n_atoms -- the SAME estimate the deck's BENCH-MARKS block
    records (job-contracts.md § 3.2/§ 3.3).  The 256 ceiling is the top of the
    BENCH-MARKS legal override window (range=[16,256])."""
    from molbuilder.siesta.input import _auto_block_size

    # The 2026-05-28 hemeC-dithiol geometry: 81 atoms x 15 ranks ->
    # 810 orb-est / 15 = 54, largest pow2 <= 54 is 32.  (Its propor
    # IMAX=0 crash was a matel_table/mpi_np issue -- BlockSize 1, 2,
    # 4 all crashed identically -- so no BlockSize pick "fixes" it.)
    assert _auto_block_size(81, mpi_np=15) == 32

    # Sweeping the orbital rank constraint.
    # 200 atoms / 16 ranks -> 2000 orb / 16 = 125, pow2 = 64.
    assert _auto_block_size(200, mpi_np=16) == 64
    # 2000 atoms / 64 ranks -> 20000 orb / 64 = 312 -> ceiling 256.
    assert _auto_block_size(2000, mpi_np=64) == 256
    # 10000 atoms / 32 ranks -> 100000 orb / 32 = 3125 -> ceiling
    # 256 (top of the BENCH-MARKS window; beyond it lies the
    # load-imbalance regime and, past 1024, the ELPA kernel limit).
    assert _auto_block_size(10000, mpi_np=32) == 256
    # 100 atoms / 16 ranks -> 1000 orb / 16 = 62, pow2 = 32.
    assert _auto_block_size(100, mpi_np=16) == 32
    # 80 atoms / 32 ranks -> 800 orb / 32 = 25, pow2 = 16.
    assert _auto_block_size(80, mpi_np=32) == 16
    # 20 atoms / 32 ranks (more ranks than ATOMS, but not than
    # orbitals) -> 200 orb / 32 = 6, pow2 = 4.  Every rank still
    # gets an orbital block.
    assert _auto_block_size(20, mpi_np=32) == 4
    # 17 atoms / 4 ranks -> 170 orb / 4 = 42, pow2 = 32.
    assert _auto_block_size(17, mpi_np=4) == 32
    # Universal invariants: BlockSize >= 1 always; with ranks not
    # exceeding the orbital estimate, BlockSize * mpi_np <= 10 *
    # n_atoms (every rank gets >= 1 ORBITAL block); and never above
    # the 256 window top.
    for n in (5, 7, 11, 17, 19, 31, 47, 81, 199, 250):
        for r in (1, 2, 4, 7, 15, 16, 32, 64):
            bs = _auto_block_size(n, mpi_np=r)
            assert bs >= 1, f"BlockSize must be >=1, got {bs}"
            assert bs <= 256, f"BlockSize={bs} above the window top"
            if 2 <= r <= 10 * n:
                assert bs * r <= 10 * n, (
                    f"BlockSize={bs} x mpi_np={r} > n_orbitals_est="
                    f"{10 * n} -- trailing ranks would get no "
                    f"orbital block"
                )



def _h2_in_a_box():
    import numpy as np
    from molbuilder.structure import Structure
    return Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2", vacuum=(12.0, 12.0, 12.0))


def test_a_shift_on_a_single_k_point_axis_is_warned_about():
    """The one case a person cannot see going wrong.

    Shifting an axis sampled at ONE k-point moves that point to the zone
    boundary; SIESTA runs it happily and the number is just wrong.  Warn,
    do not refuse -- it is a legal input and the user may mean it.
    """
    from molbuilder.validation import validate
    issues = validate(_h2_in_a_box(), SiestaConfig(
        system_label="h2", kgrid=(4, 4, 1),
        kgrid_displacement=(0.5, 0.5, 0.5)))
    mine = [i for i in issues if i.where == "config.kgrid_displacement"]
    assert len(mine) == 1, [i.message for i in mine]
    assert mine[0].severity == "warn"
    assert "kgrid[2] = 1" in mine[0].message
    # The two shifted axes with 4 points each are a legitimate choice.
    ok = validate(_h2_in_a_box(), SiestaConfig(
        system_label="h2", kgrid=(4, 4, 1),
        kgrid_displacement=(0.5, 0.5, 0.0)))
    assert not [i for i in ok if i.where == "config.kgrid_displacement"]


# --------------------------------------------------------------------- #
#  kgrid vs the axes — the 2026-08-20 rule (user):                      #
#  k>1 is the user's explicit statement; k=1 states nothing and is      #
#  validated not at all.                                                #
# --------------------------------------------------------------------- #


def _kgrid_struct(axis_kind):
    from molbuilder.structure import Structure
    meta = {"cell": [[5, 0, 0], [0, 5, 0], [0, 0, 30]]}
    if axis_kind:
        meta["axis_kind"] = axis_kind
    return Structure.from_dict({
        "elements": ["Au"] * 4 + ["S", "C"],
        "positions": [[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0],
                      [1, 1, 5], [1, 1, 7]],
        "metadata": meta,
    })


def _kgrid_findings(struct, kgrid):
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    return [i for i in validate(struct, SiestaConfig(kgrid=kgrid))
            if "kgrid[" in i.message]


def test_k_equal_one_is_never_validated():
    """k=1 states nothing: correct for an isolated axis, a legitimate
    Gamma-only choice for a periodic one.  Whatever the axes say, k=1 is
    silent."""
    for kinds in (["periodic"] * 3, ["isolated"] * 3,
                  ["periodic", "periodic", "transport"]):
        assert _kgrid_findings(_kgrid_struct(kinds), (1, 1, 1)) == [], kinds


def test_k_above_one_on_an_isolated_axis_warns_by_axis():
    """Sampling a direction the user said does not repeat is a
    contradiction between two of their own statements — said per axis
    (`engines/siesta.md` § 6.1).  A `transport` axis in a calculation that
    is not a transport one is periodic in the deck and sampled like one
    (user, 2026-09-30): here its images sit far apart, so it earns the gap's
    hint, not the open boundary's rule, which is a transport rung's alone.

    MUTATION THIS MUST FAIL AGAINST: the plain run's transport axis given
    the open role (a "fake periodicity" finding, or none)."""
    found = _kgrid_findings(
        _kgrid_struct(["isolated", "isolated", "isolated"]), (2, 2, 1))
    assert len(found) == 2, [i.message for i in found]
    assert all("on an isolated axis" in i.message for i in found)
    found = _kgrid_findings(
        _kgrid_struct(["periodic", "periodic", "transport"]), (2, 2, 4))
    assert len(found) == 1, [i.message for i in found]
    assert "kgrid[2] = 4" in found[0].message
    assert "images sit ~" in found[0].message, found[0].message


def test_k_above_one_across_a_wide_gap_is_a_hint_not_silence():
    """A periodic axis whose images sit >= 5 A apart, sampled with k>1:
    the geometric gap (cell minus atom span) is the real vacuum whether or
    not the field was set, and a weak-image-interaction setup can be
    deliberate — so a HINT that names the gap and carries on."""
    found = _kgrid_findings(_kgrid_struct(["periodic"] * 3), (2, 2, 2))
    assert len(found) == 1, [i.message for i in found]
    assert "images sit ~" in found[0].message
    assert "carry on" in found[0].message
    # tight packing on the sampled axes: fully consistent, silent
    assert _kgrid_findings(_kgrid_struct(["periodic"] * 3), (2, 2, 1)) == []


def test_intended_transport_setup_is_silent():
    """The BDT-Au shape that raised this: x,y periodic sampled 2x2, the
    transport z at k=1 — every statement consistent, nothing said."""
    assert _kgrid_findings(
        _kgrid_struct(["periodic", "periodic", "transport"]), (2, 2, 1)) == []

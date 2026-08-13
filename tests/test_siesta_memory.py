"""Unit tests for molbuilder.siesta.memory (peak-memory estimator)."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest

from molbuilder.siesta.memory import (
    FdfMemInputs,
    MemEstimate,
    MemModel,
    estimate_norb,
    estimate_siesta_memory,
    parse_fdf_mem_inputs,
)


# --------------------------------------------------------------------- #
#  fdf fixtures                                                         #
# --------------------------------------------------------------------- #


def _write(tmp_path: Path, text: str, name: str = "in.fdf") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


# A small but complete fdf: 4 atoms (2 C, 1 O, 1 H), cubic 10 Ang cell
# (volume 1000 A^3 exactly), 4x4x1 kgrid -> 16 k-points, TZP, mesh
# 400 Ry.  NumberOfAtoms intentionally set to 444 to prove the field is
# read straight from the key (independent of the coords-block length).
FULL_FDF = """\
    # a deliberately mixed-style fdf
    SystemLabel       demo
    NumberOfAtoms     444
    MeshCutoff        400 Ry
    PAO.BasisSize     TZP
    LatticeConstant   1.0 Ang

    %block LatticeVectors
    10.0 0.0 0.0
    0.0 10.0 0.0
    0.0 0.0 10.0
    %endblock LatticeVectors

    %block kgrid_Monkhorst_Pack
    4 0 0 0.0
    0 4 0 0.0
    0 0 1 0.0
    %endblock kgrid_Monkhorst_Pack

    %block ChemicalSpeciesLabel
    1 6 C
    2 8 O
    3 1 H
    %endblock ChemicalSpeciesLabel

    %block AtomicCoordinatesAndAtomicSpecies
    0.0 0.0 0.0 1
    1.0 0.0 0.0 1
    2.0 0.0 0.0 2
    3.0 0.0 0.0 3
    %endblock AtomicCoordinatesAndAtomicSpecies
"""


def test_parse_full_fdf(tmp_path):
    inp = parse_fdf_mem_inputs(_write(tmp_path, FULL_FDF))
    assert inp.n_atoms == 444
    assert inp.mesh_cutoff_ry == pytest.approx(400.0)
    assert inp.basis_size == "TZP"
    assert inp.cell_volume_ang3 == pytest.approx(1000.0)
    assert inp.n_kpoints == 16
    assert inp.elements == {"C": 2, "O": 1, "H": 1}
    assert inp.parse_warnings == []  # everything present


def test_meshcutoff_hartree_doubles(tmp_path):
    fdf = """\
        NumberOfAtoms 1
        MeshCutoff 200 Ha
    """
    inp = parse_fdf_mem_inputs(_write(tmp_path, fdf))
    assert inp.mesh_cutoff_ry == pytest.approx(400.0)  # 200 Ha -> 400 Ry


def test_meshcutoff_hartree_word(tmp_path):
    inp = parse_fdf_mem_inputs(_write(tmp_path, "MeshCutoff 150 Hartree\n"))
    assert inp.mesh_cutoff_ry == pytest.approx(300.0)


def test_key_separator_insensitivity(tmp_path):
    # PAO-Basis_Size and Mesh.Cutoff use the alternate separators.
    fdf = """\
        Number_Of_Atoms 3
        Mesh.Cutoff 250 Ry
        PAO-BasisSize DZP
    """
    inp = parse_fdf_mem_inputs(_write(tmp_path, fdf))
    assert inp.n_atoms == 3
    assert inp.mesh_cutoff_ry == pytest.approx(250.0)
    assert inp.basis_size == "DZP"


def test_kgrid_dotted_block_name(tmp_path):
    fdf = """\
        NumberOfAtoms 1
        %block kgrid.MonkhorstPack
        2 0 0 0.0
        0 3 0 0.0
        0 0 5 0.0
        %endblock kgrid.MonkhorstPack
    """
    inp = parse_fdf_mem_inputs(_write(tmp_path, fdf))
    assert inp.n_kpoints == 30  # 2*3*5


def test_bohr_lattice_constant(tmp_path):
    # 10 Bohr cubic cell -> edge = 10 * 0.529177 Ang.
    fdf = """\
        NumberOfAtoms 1
        LatticeConstant 10.0 Bohr
        %block LatticeVectors
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
        %endblock LatticeVectors
    """
    inp = parse_fdf_mem_inputs(_write(tmp_path, fdf))
    edge = 10.0 * 0.529177
    assert inp.cell_volume_ang3 == pytest.approx(edge ** 3)


def test_missing_keys_defaults_and_warnings(tmp_path):
    inp = parse_fdf_mem_inputs(_write(tmp_path, "SystemLabel empty\n"))
    # Defaults, no exception.
    assert inp.n_atoms == 0
    assert inp.mesh_cutoff_ry == 0.0
    assert inp.basis_size == "DZP"
    assert inp.n_kpoints == 1
    assert inp.elements == {}
    assert inp.parse_warnings  # non-empty
    # Specifically: the coords block absence is reported (NOT split evenly).
    assert any("AtomicCoordinates" in w for w in inp.parse_warnings)


def test_coords_block_absent_leaves_elements_empty(tmp_path):
    # species labelled but no coordinates -> empty + warning, never a
    # NumberOfAtoms even-split guess.
    fdf = """\
        NumberOfAtoms 100
        %block ChemicalSpeciesLabel
        1 6 C
        %endblock ChemicalSpeciesLabel
    """
    inp = parse_fdf_mem_inputs(_write(tmp_path, fdf))
    assert inp.elements == {}
    assert any("AtomicCoordinates" in w for w in inp.parse_warnings)


def test_nonexistent_file_is_tolerant(tmp_path):
    inp = parse_fdf_mem_inputs(tmp_path / "does_not_exist.fdf")
    assert inp.n_atoms == 0
    assert inp.parse_warnings


# --------------------------------------------------------------------- #
#  estimate_norb                                                        #
# --------------------------------------------------------------------- #


def test_norb_pure_carbon_tzp_exact():
    # Carbon valence l-shells (fallback) = {s, p} = {0, 1}, l_max = 1.
    # TZP -> nzeta = 3, polarized, n_pol = 1.
    #   sum_l (2l+1)*nzeta = (1*3) + (3*3) = 3 + 9 = 12
    #   polarization shell at l_max+1 = 2 (d): (2*2+1)*1 = 5
    #   per atom = 12 + 5 = 17
    # 10 carbons -> 170 orbitals.
    inp = FdfMemInputs(
        n_atoms=10, mesh_cutoff_ry=400.0, cell_volume_ang3=1000.0,
        basis_size="TZP", n_kpoints=1, elements={"C": 10},
    )
    n_orb, warnings = estimate_norb(inp)
    assert n_orb == 170
    assert isinstance(warnings, list)


def _carbon_inputs(basis: str) -> FdfMemInputs:
    return FdfMemInputs(
        n_atoms=1, mesh_cutoff_ry=400.0, cell_volume_ang3=1000.0,
        basis_size=basis, n_kpoints=1, elements={"C": 1},
    )


def test_norb_zeta_ordering():
    sz = estimate_norb(_carbon_inputs("SZ"))[0]
    dz = estimate_norb(_carbon_inputs("DZ"))[0]
    tz = estimate_norb(_carbon_inputs("TZ"))[0]
    assert sz < dz < tz
    # Carbon {s,p}: SZ = 1+3 = 4, DZ = 8, TZ = 12.
    assert (sz, dz, tz) == (4, 8, 12)


def test_norb_polarization_adds_orbitals():
    sz = estimate_norb(_carbon_inputs("SZ"))[0]
    szp = estimate_norb(_carbon_inputs("SZP"))[0]
    assert szp > sz
    assert szp == sz + 5  # one d shell (2*2+1)


def test_norb_unrecognized_basis_defaults_dzp():
    weird = estimate_norb(_carbon_inputs("GARBAGE"))[0]
    dzp = estimate_norb(_carbon_inputs("DZP"))[0]
    assert weird == dzp


def test_norb_empty_elements_warns():
    inp = FdfMemInputs(
        n_atoms=0, mesh_cutoff_ry=0.0, cell_volume_ang3=1.0,
        basis_size="DZP", n_kpoints=1, elements={},
    )
    n_orb, warnings = estimate_norb(inp)
    assert n_orb == 0
    assert warnings


def test_norb_au_uses_d_shell():
    # Au fallback = {s,p,d}, l_max=2.  DZP: sum (1+3+5)*2 = 18, pol shell
    # at l=3 (f): (2*3+1)*1 = 7 -> 25 per atom.
    inp = FdfMemInputs(
        n_atoms=1, mesh_cutoff_ry=400.0, cell_volume_ang3=1000.0,
        basis_size="DZP", n_kpoints=1, elements={"Au": 1},
    )
    assert estimate_norb(inp)[0] == 25


# --------------------------------------------------------------------- #
#  MemModel.from_config                                                 #
# --------------------------------------------------------------------- #


def test_memmodel_from_config_defaults_and_unknowns():
    assert MemModel.from_config(None) == MemModel()
    m = MemModel.from_config({"base_gb": 8.0, "bogus": 123, "safety": 1.5})
    assert m.base_gb == 8.0
    assert m.safety == 1.5
    assert m.c_dense == MemModel().c_dense  # untouched default


# --------------------------------------------------------------------- #
#  estimate_siesta_memory                                              #
# --------------------------------------------------------------------- #


def _carbon_fdf(tmp_path, n_atoms=20, mesh="400 Ry", basis="DZP",
                kgrid=(1, 1, 1), edge=20.0, name="c.fdf") -> Path:
    """Build an fdf with ``n_atoms`` carbons in a cubic ``edge`` cell."""
    coords = "\n".join(f"{i*0.1:.4f} 0.0 0.0 1" for i in range(n_atoms))
    kx, ky, kz = kgrid
    fdf = f"""\
NumberOfAtoms {n_atoms}
MeshCutoff {mesh}
PAO.BasisSize {basis}
LatticeConstant 1.0 Ang
%block LatticeVectors
{edge} 0.0 0.0
0.0 {edge} 0.0
0.0 0.0 {edge}
%endblock LatticeVectors
%block kgrid_Monkhorst_Pack
{kx} 0 0 0.0
0 {ky} 0 0.0
0 0 {kz} 0.0
%endblock kgrid_Monkhorst_Pack
%block ChemicalSpeciesLabel
1 6 C
%endblock ChemicalSpeciesLabel
%block AtomicCoordinatesAndAtomicSpecies
{coords}
%endblock AtomicCoordinatesAndAtomicSpecies
"""
    return _write(tmp_path, fdf, name=name)


def test_estimate_monotonic_in_ntasks(tmp_path):
    fdf = _carbon_fdf(tmp_path)
    lo = estimate_siesta_memory(fdf, ntasks=1)
    hi = estimate_siesta_memory(fdf, ntasks=64)
    assert hi.repl_gb > lo.repl_gb
    assert hi.request_gb >= lo.request_gb


def test_estimate_kpoints_increase_memory(tmp_path):
    # Big enough that dense_gb doesn't round to 0.0 at 1 decimal.
    gamma = _carbon_fdf(tmp_path, n_atoms=800, basis="TZP",
                        kgrid=(1, 1, 1), name="g.fdf")
    kmesh = _carbon_fdf(tmp_path, n_atoms=800, basis="TZP",
                        kgrid=(4, 4, 1), name="k.fdf")
    g = estimate_siesta_memory(gamma, ntasks=4)
    k = estimate_siesta_memory(kmesh, ntasks=4)
    # Gamma: bpc=8, n_kpoints=1.  4x4x1 mesh: bpc=16, n_kpoints=16.
    # Dense term scales bpc * n_kpoints -> (16*16)/(8*1) = 32x.
    assert k.dense_gb == pytest.approx(32.0 * g.dense_gb, rel=0.02)
    assert k.request_gb > g.request_gb


def test_estimate_node_mem_cap(tmp_path):
    # A big system that would request far more than a tiny node cap.
    fdf = _carbon_fdf(tmp_path, n_atoms=400, basis="TZP", kgrid=(4, 4, 1))
    model = MemModel.from_config({"node_mem_gb": 16.0})
    est = estimate_siesta_memory(fdf, ntasks=64, model=model)
    assert est.capped is True
    assert est.request_gb == 16


def test_estimate_floor_respected_for_tiny_system(tmp_path):
    """The floor must LIFT the answer, so the probe puts it ABOVE the raw
    estimate (B-5, 2026-08-13: with the default 4 GB floor under a ~8 GB
    one-atom estimate, deleting the floor's max() stayed green -- the
    assert was satisfied by the estimate itself)."""
    fdf = _carbon_fdf(tmp_path, n_atoms=1, mesh="100 Ry", edge=8.0)
    est = estimate_siesta_memory(fdf, ntasks=1)
    assert est.request_gb >= math.ceil(MemModel().floor_gb)
    assert est.capped is False
    lifted = estimate_siesta_memory(
        fdf, ntasks=1,
        model=MemModel.from_config({"floor_gb": est.request_gb + 48.0}))
    assert lifted.request_gb >= est.request_gb + 48, (
        "a floor above the raw estimate did not lift the request -- "
        "the floor max() is gone")


def test_estimate_extra_gb_adds(tmp_path):
    fdf = _carbon_fdf(tmp_path, n_atoms=50)
    base = estimate_siesta_memory(fdf, ntasks=4)
    bumped = estimate_siesta_memory(
        fdf, ntasks=4, model=MemModel.from_config({"extra_gb": 20.0}))
    assert bumped.request_gb >= base.request_gb + 20


def test_breakdown_str_format(tmp_path):
    fdf = _carbon_fdf(tmp_path, n_atoms=50, basis="TZP")
    est = estimate_siesta_memory(fdf, ntasks=8)
    s = est.breakdown_str()
    assert s.startswith("dense=")
    assert "mesh=" in s and "repl=" in s and "base=" in s
    assert "x1.3" in s            # the default safety multiplier
    assert s.endswith(f"-> {est.request_gb}G")


def test_estimate_realistic_444_tzp(tmp_path):
    # ~444-atom metal-junction-ish system, TZP, 4x4x1, 64 ranks.
    fdf = _carbon_fdf(tmp_path, n_atoms=444, basis="TZP",
                      kgrid=(4, 4, 1), edge=25.0, name="big.fdf")
    est = estimate_siesta_memory(fdf, ntasks=64)
    assert isinstance(est, MemEstimate)
    assert 10 <= est.request_gb <= 400
    assert est.n_orb > 0
    assert est.capped is False


def test_mesh_calibration_against_real_siesta(tmp_path):
    # Real SIESTA run: 17.64 x 17.64 x 58.41 A cell at MeshCutoff 400 Ry
    # produced a 192 x 192 x 720 = 26.5 Mpts real-space grid.  Our
    # isotropic cube-root-edge proxy should land within ~30% of that.
    fdf = """\
        NumberOfAtoms 1
        MeshCutoff 400 Ry
        LatticeConstant 1.0 Ang
        %block LatticeVectors
        17.64 0.0 0.0
        0.0 17.64 0.0
        0.0 0.0 58.41
        %endblock LatticeVectors
        %block ChemicalSpeciesLabel
        1 6 C
        %endblock ChemicalSpeciesLabel
        %block AtomicCoordinatesAndAtomicSpecies
        0.0 0.0 0.0 1
        %endblock AtomicCoordinatesAndAtomicSpecies
    """
    est = estimate_siesta_memory(_write(tmp_path, fdf), ntasks=4)
    real_mpts = 192 * 192 * 720 / 1e6   # 26.54 Mpts
    assert abs(est.mesh_mpts - real_mpts) / real_mpts < 0.30


def _bdt_au444_tzp_fdf(tmp_path) -> Path:
    """The real BDT-Au(111) junction the model is calibrated against:
    432 Au + 6 C + 2 S + 4 H = 444 atoms, TZP, 4x4x1 (=16) k-grid, in
    the 17.64 x 17.64 x 58.41 A cell.  N_orb ~= 14.8k."""
    lines = (["0.0 0.0 0.0 1"] * 432 + ["0.0 0.0 0.0 2"] * 6
             + ["0.0 0.0 0.0 3"] * 2 + ["0.0 0.0 0.0 4"] * 4)
    coords = "\n".join(lines)
    fdf = f"""\
NumberOfAtoms 444
MeshCutoff 400 Ry
PAO.BasisSize TZP
LatticeConstant 1.0 Ang
%block LatticeVectors
17.64 0.0 0.0
0.0 17.64 0.0
0.0 0.0 58.41
%endblock LatticeVectors
%block kgrid_Monkhorst_Pack
4 0 0 0.0
0 4 0 0.0
0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
%block ChemicalSpeciesLabel
1 79 Au
2 6 C
3 16 S
4 1 H
%endblock ChemicalSpeciesLabel
%block AtomicCoordinatesAndAtomicSpecies
{coords}
%endblock AtomicCoordinatesAndAtomicSpecies
"""
    return _write(tmp_path, fdf, name="bdtau.fdf")


def test_calibration_against_observed_anchors(tmp_path):
    # Real anchors on the BDT-Au junction (N_orb ~= 14.8k, 16 k-pts):
    #   * np=4  -> ~162 GB total RSS (observed, did not OOM).
    #   * np=64 -> OOM'd at --mem=320G; SURVIVED 480G; and the unthrottled
    #     480G survivor reported a peak cgroup RSS of 433.15 GB
    #     (`sacct TRESUsageInMax mem=`, Sol jobs 2026-06-27) -- the
    #     GROUND-TRUTH np=64 peak.  Our estimate must sit ABOVE 433 (so it
    #     doesn't OOM) yet under the 480 that survived / the node cap.
    fdf = _bdt_au444_tzp_fdf(tmp_path)
    norb, _ = estimate_norb(parse_fdf_mem_inputs(fdf))
    assert 14000 <= norb <= 15500

    model = MemModel.from_config({"node_mem_gb": 500.0})
    np4 = estimate_siesta_memory(fdf, ntasks=4, model=model)
    np64 = estimate_siesta_memory(fdf, ntasks=64, model=model)

    assert 170 <= np4.request_gb <= 230      # brackets observed ~162 GB
    assert np64.request_gb > 433             # must exceed the REAL peak
    assert 440 <= np64.request_gb <= 500     # at/under the 480 survival
    assert not np4.capped and not np64.capped

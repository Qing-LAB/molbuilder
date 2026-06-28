"""Tests for the TranSIESTA consistency preflight
(molbuilder/transport/preflight.py)."""
from __future__ import annotations

import pytest

from molbuilder.transport.preflight import (
    parse_fdf_params, preflight, preflight_files,
)

# A minimal but realistic pair: same lateral cell + contract, device
# kz=1 / electrode kz=80 -> should PASS.
_LATTICE = """\
LatticeConstant 1.0 Ang
%block LatticeVectors
17.64 0.0 0.0
0.0 17.64 0.0
0.0 0.0 {zlen}
%endblock LatticeVectors
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
0.0 0.0 {z0} 1
0.0 0.0 {z1} 1
%endblock AtomicCoordinatesAndAtomicSpecies
"""


def _device(kx=4, ky=4, kz=1, mesh=400, eshift=0.01, xc=("GGA", "PBE"),
            basis="DZP", zlen=58.0):
    return f"""\
SolutionMethod transiesta
MeshCutoff {mesh} Ry
PAO.EnergyShift {eshift} Ry
XC.Functional {xc[0]}
XC.Authors {xc[1]}
PAO.BasisSize {basis}
%block kgrid_Monkhorst_Pack
{kx} 0 0 0.0
0 {ky} 0 0.0
0 0 {kz} 0.0
%endblock kgrid_Monkhorst_Pack
""" + _LATTICE.format(zlen=zlen, z0=2.0, z1=55.0)


def _electrode(kx=4, ky=4, kz=80, mesh=400, eshift=0.01, xc=("GGA", "PBE"),
               basis="DZP", zlen=14.0, savehs=True):
    head = "TS.HS.Save T\n" if savehs else ""
    return head + f"""\
SolutionMethod diagon
MeshCutoff {mesh} Ry
PAO.EnergyShift {eshift} Ry
XC.Functional {xc[0]}
XC.Authors {xc[1]}
PAO.BasisSize {basis}
%block kgrid_Monkhorst_Pack
{kx} 0 0 0.0
0 {ky} 0 0.0
0 0 {kz} 0.0
%endblock kgrid_Monkhorst_Pack
""" + _LATTICE.format(zlen=zlen, z0=0.0, z1=13.0)


def _ids(report, severity=None):
    return {c.id for c in report.checks
            if severity is None or c.severity == severity}


# --------------------------------------------------------------------- #
#  parsing                                                             #
# --------------------------------------------------------------------- #


def test_parse_fdf_params():
    p = parse_fdf_params(_device())
    assert p.kgrid == (4, 4, 1)
    assert p.mesh_cutoff_ry == 400.0
    assert p.energy_shift_ry == 0.01
    assert p.xc == "GGA/PBE"
    assert p.basis_size == "DZP"
    assert p.solution_method == "transiesta"
    assert p.z_len_ang == pytest.approx(58.0)


def test_parse_energy_shift_hartree_to_ry():
    p = parse_fdf_params("PAO.EnergyShift 0.005 Ha\n")
    assert p.energy_shift_ry == pytest.approx(0.010)


def test_parse_saves_ts_hs():
    assert parse_fdf_params("TS.HS.Save T\n").saves_ts_hs is True
    assert parse_fdf_params("SaveHS .true.\n").saves_ts_hs is True
    assert parse_fdf_params("MeshCutoff 400 Ry\n").saves_ts_hs is False


# --------------------------------------------------------------------- #
#  the gates                                                            #
# --------------------------------------------------------------------- #


def test_consistent_pair_passes(tmp_path):
    dev = tmp_path / "dev.fdf"; dev.write_text(_device())
    ele = tmp_path / "ele.fdf"; ele.write_text(_electrode())
    r = preflight_files(dev, ele)
    assert r.ok()                                  # no ERROR
    assert "kgrid.device_kz" in _ids(r, "ok")
    assert "kgrid.electrode_kz" in _ids(r, "ok")
    assert "contract.meshcutoff" in _ids(r, "ok")


def test_device_kz_not_one_is_error():
    r = preflight(parse_fdf_params(_device(kz=4)),
                  parse_fdf_params(_electrode()))
    assert not r.ok()
    assert "kgrid.device_kz" in _ids(r, "error")


def test_electrode_kz_one_is_error():
    # the final-discussion bug: electrode with kz=1 (under-sampled bulk)
    r = preflight(parse_fdf_params(_device()),
                  parse_fdf_params(_electrode(kz=1)))
    assert not r.ok()
    assert "kgrid.electrode_kz" in _ids(r, "error")


def test_electrode_kz_low_warns():
    r = preflight(parse_fdf_params(_device()),
                  parse_fdf_params(_electrode(kz=8)))
    assert r.ok()                                  # warn, not error
    assert "kgrid.electrode_kz" in _ids(r, "warn")


def test_numerical_contract_mismatch_is_error():
    r = preflight(parse_fdf_params(_device(mesh=400)),
                  parse_fdf_params(_electrode(mesh=300)))
    assert not r.ok()
    assert "contract.meshcutoff" in _ids(r, "error")
    r2 = preflight(parse_fdf_params(_device(basis="DZP")),
                   parse_fdf_params(_electrode(basis="SZP")))
    assert "contract.basis" in _ids(r2, "error")
    r3 = preflight(parse_fdf_params(_device(xc=("GGA", "PBE"))),
                   parse_fdf_params(_electrode(xc=("LDA", "CA"))))
    assert "contract.xc" in _ids(r3, "error")


def test_transverse_k_mismatch_is_error():
    r = preflight(parse_fdf_params(_device(kx=4, ky=4)),
                  parse_fdf_params(_electrode(kx=2, ky=2)))
    assert not r.ok()
    assert "kgrid.transverse" in _ids(r, "error")


def test_lateral_cell_mismatch_is_error():
    dev = parse_fdf_params(_device())
    ele = parse_fdf_params(_electrode())
    ele.cell_ang[0][0] = 12.0                      # different a vector
    r = preflight(dev, ele)
    assert not r.ok()
    assert "cell.transverse" in _ids(r, "error")


def test_electrode_too_thin_warns():
    r = preflight(parse_fdf_params(_device()),
                  parse_fdf_params(_electrode(zlen=5.0)))   # ~3 layers
    assert r.ok()                                  # warn, not error
    assert "electrode.thickness" in _ids(r, "warn")


def test_missing_save_hs_warns():
    r = preflight(parse_fdf_params(_device()),
                  parse_fdf_params(_electrode(savehs=False)))
    assert "electrode.saveHS" in _ids(r, "warn")


def test_device_z_vacuum_warns():
    # cell z 58 Ang, atoms span only ~13 Ang -> ~45 Ang of z-vacuum
    dev = _device(zlen=58.0).replace("0.0 0.0 55.0 1", "0.0 0.0 13.0 1")
    r = preflight(parse_fdf_params(dev), parse_fdf_params(_electrode()))
    assert "device.z_vacuum" in _ids(r, "warn")


def test_transport_help_carries_worked_examples():
    # get-started examples must stay in --help (no doc-flipping needed)
    from click.testing import CliRunner
    from molbuilder.transport._cli import transport_group
    r = CliRunner().invoke(transport_group, ["-h"])
    assert r.exit_code == 0
    assert "QUICKSTART" in r.output
    assert "molbuilder transport bundle --device" in r.output
    assert "run-transport.sh" in r.output
    for sub in ("preflight", "electrode", "bundle"):
        rs = CliRunner().invoke(transport_group, [sub, "-h"])
        assert "EXAMPLE" in rs.output and "molbuilder transport" in rs.output


def test_parser_matches_transiesta_emitter_keys():
    # The fixtures mirror the ACTUAL keys molbuilder's transiesta emitter
    # writes (PAO.BasisSize/EnergyShift, XC.Functional/Authors, MeshCutoff,
    # kgrid block, SolutionMethod transiesta) -- a parse must recover them.
    p = parse_fdf_params(_device())
    assert None not in (p.kgrid, p.mesh_cutoff_ry, p.energy_shift_ry,
                        p.xc, p.basis_size, p.solution_method)

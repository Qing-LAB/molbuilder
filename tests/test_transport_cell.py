"""Cell/boundary preservation in the transport emitter + sidecar
round-trip (the hex-Au(111) fix)."""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import TransportConfig
from molbuilder.sidecars import molstruct as msj
from molbuilder.structure import Structure
from molbuilder.transport.preflight import parse_fdf_params, preflight
from molbuilder.transport.transiesta import (
    TransiestaEngine, axis_vacuum, _emit_geometry,
)
from molbuilder.transport.wizard import electrode_wizard, extract_electrode_model

# A hexagonal Au(111)-like cell: a,b at 60 deg ~17.3 A; c (transport) 40 A.
HEX = np.array([[17.30, 0.0, 0.0],
                [8.65, 14.98, 0.0],
                [0.0, 0.0, 40.0]])


def _hex_device():
    d = 2.40
    elems, pos, L, B, R = [], [], [], [], []
    i = 0
    for k in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, k * d]); L.append(i); i += 1
    z0 = 4 * d
    for j in range(2):
        elems.append("S"); pos.append([1.0, 1.0, z0 + 1.5 + j * 1.8]); B.append(i); i += 1
    z1 = z0 + 1.5 + 2 * 1.8 + 1.5
    for k in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, z1 + k * d]); R.append(i); i += 1
    return Structure(elements=elems, positions=np.asarray(pos, float), cell=HEX,
                     pbc=(True, True, True),
                     regions={"L-electrode": L, "bridge": B, "R-electrode": R})


# ------------------------------------------------------------------ #
#  Structure + sidecar carry the cell                                #
# ------------------------------------------------------------------ #


def test_structure_cell_and_pbc_defaults():
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    assert s.cell is None and s.pbc == (False, False, False)
    s2 = Structure(elements=["C"], positions=np.zeros((1, 3)), cell=HEX)
    assert s2.cell.shape == (3, 3) and s2.pbc == (True, True, True)


def test_structure_cell_validation():
    with pytest.raises(ValueError):
        Structure(elements=["C"], positions=np.zeros((1, 3)),
                  cell=np.zeros((2, 3)))


def test_sidecar_cell_round_trip(tmp_path):
    d = msj.to_dict(n_atoms_total=1, structure_hash="0" * 32,
                    cell=HEX, pbc=(True, True, False))
    assert d["pbc"] == [True, True, False]
    p = tmp_path / "x.molstruct.json"
    msj.save(p, d)
    loaded = msj.load(p)
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    msj.apply_to_structure(s, loaded)
    assert np.allclose(s.cell, HEX) and s.pbc == (True, True, False)


def test_sidecar_without_cell_is_nonperiodic():
    d = msj.to_dict(n_atoms_total=1, structure_hash="0" * 32)
    del d["cell"]; del d["pbc"]                       # simulate an old v3 file
    s = Structure(elements=["C"], positions=np.zeros((1, 3)))
    msj.apply_to_structure(s, d)
    assert s.cell is None


# ------------------------------------------------------------------ #
#  Emitter preserves the explicit cell verbatim                      #
# ------------------------------------------------------------------ #


def test_emitter_preserves_hex_cell_verbatim():
    fdf = "\n".join(_emit_geometry(_hex_device()))
    p = parse_fdf_params(fdf)
    assert np.allclose(p.cell_ang[0], [17.30, 0.0, 0.0])
    assert np.allclose(p.cell_ang[1], [8.65, 14.98, 0.0])   # NOT squared off


def test_emitter_fallback_warns_isolated_cluster():
    # No cell -> orthorhombic vacuum box, flagged loudly.
    dev = _hex_device()
    dev.cell = None
    fdf = "\n".join(_emit_geometry(dev))
    assert "ISOLATED CLUSTER" in fdf
    assert "hexagonal cell" in fdf


def test_axis_vacuum_flags_transport_axis_gap():
    # atoms span z 0..~21 in a 40 A c-axis -> ~19 A z-vacuum
    dev = _hex_device()
    vac = axis_vacuum(dev.cell, dev.positions)
    assert vac[2] > 5.0
    fdf = "\n".join(_emit_geometry(dev))
    assert "transport axis (c) has vacuum" in fdf


# ------------------------------------------------------------------ #
#  Wizard preserves the hex lateral vectors                          #
# ------------------------------------------------------------------ #


def test_wizard_uses_device_hex_lateral_vectors():
    m = extract_electrode_model(_hex_device(), "L-electrode")
    assert np.allclose(m.lat_a, [17.30, 0.0, 0.0])
    assert np.allclose(m.lat_b, [8.65, 14.98, 0.0])


def test_hex_device_and_electrode_pass_preflight():
    dev = _hex_device()
    cfg = TransportConfig(job_name="hex", siesta_mesh_cutoff_ry=400,
                          k_mesh_transverse=(2, 2, 1))
    device_fdf = TransiestaEngine.render_script(dev, cfg)
    _job, ele_fdf, _m = electrode_wizard(dev, cfg, which="L-electrode")[0]
    rep = preflight(parse_fdf_params(device_fdf), parse_fdf_params(ele_fdf))
    errs = {c.id for c in rep.checks if c.severity == "error"}
    assert "cell.transverse" not in errs        # hex lateral vectors match

"""Tests for the TranSIESTA electrode wizard
(molbuilder/transport/wizard.py)."""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import TransportConfig
from molbuilder.structure import Structure
from molbuilder.transport.preflight import parse_fdf_params, preflight
from molbuilder.transport.transiesta import TransiestaEngine
from molbuilder.transport.wizard import (
    bulk_z_period, detect_layers, electrode_wizard, extract_electrode_model,
    render_electrode_fdf,
)


def _au_device():
    """A toy [L-electrode][bridge][R-electrode] junction along z.

    Each electrode = 4 Au layers spaced 2.35 Å (one atom/layer for the
    test); a 2-atom S-S bridge sits between them.  Contiguous ordering
    so the device emitter is happy.
    """
    d = 2.35
    elems, pos, left, bridge, right = [], [], [], [], []
    i = 0
    for layer in range(4):                       # left electrode
        elems.append("Au"); pos.append([1.0, 1.0, layer * d]); left.append(i); i += 1
    z0 = 4 * d
    for j in range(2):                           # bridge
        elems.append("S"); pos.append([1.0, 1.0, z0 + 1.5 + j * 1.8]); bridge.append(i); i += 1
    z1 = z0 + 1.5 + 2 * 1.8 + 1.5
    for layer in range(4):                       # right electrode
        elems.append("Au"); pos.append([1.0, 1.0, z1 + layer * d]); right.append(i); i += 1
    return Structure(
        elements=elems, positions=np.asarray(pos, dtype=float),
        regions={"L-electrode": left, "bridge": bridge, "R-electrode": right})


# --------------------------------------------------------------------- #
#  layer detection + bulk period                                        #
# --------------------------------------------------------------------- #


def test_detect_layers_groups_by_z():
    z = np.array([0.0, 0.02, 2.35, 2.36, 4.70])   # 3 layers, jitter within
    assert detect_layers(z) == pytest.approx([0.01, 2.355, 4.70], abs=1e-3)


def test_bulk_z_period_adds_one_spacing():
    # 4 layers at 2.35 spacing -> span 7.05, +d => period 9.40
    zper, d, n = bulk_z_period([0.0, 2.35, 4.70, 7.05])
    assert d == pytest.approx(2.35)
    assert n == 4
    assert zper == pytest.approx(9.40)


def test_bulk_z_period_single_layer_raises():
    with pytest.raises(ValueError):
        bulk_z_period([0.0])


# --------------------------------------------------------------------- #
#  extraction: the clone guarantees                                     #
# --------------------------------------------------------------------- #


def test_extract_clones_atoms_and_uses_device_lateral_cell():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode")
    assert m.n_atoms == 4
    assert set(m.elements) == {"Au"}
    assert m.n_layers == 4
    assert m.positions[:, 2].min() == pytest.approx(0.0)   # shifted to 0
    # lateral cell is the DEVICE cell, not the electrode's own (1.0) extent
    from molbuilder.transport.transiesta import _compute_cell_from_extents
    a, b, _c = _compute_cell_from_extents(dev)
    assert (m.cell_a, m.cell_b) == pytest.approx((a, b))


def test_extract_explicit_z_period_overrides():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode", z_period=8.5)
    assert m.z_period == pytest.approx(8.5)


def test_extract_thin_electrode_notes_warning():
    dev = _au_device()
    m = extract_electrode_model(dev, "L-electrode", min_thickness_ang=12.0)
    # 4 layers * 2.35 span ~7 Å < 12 -> a note is emitted
    assert any("principal layer" in n for n in m.notes)


# --------------------------------------------------------------------- #
#  the invariants hold by construction (device <-> electrode preflight)  #
# --------------------------------------------------------------------- #


def test_emitted_electrode_passes_preflight_against_device():
    dev = _au_device()
    cfg = TransportConfig(job_name="junc", siesta_mesh_cutoff_ry=400,
                          k_mesh_transverse=(4, 4, 1))
    device_fdf = TransiestaEngine.render_script(dev, cfg)
    models = electrode_wizard(dev, cfg, which="L-electrode", electrode_kz=40)
    assert len(models) == 1
    _job, ele_fdf, _m = models[0]

    rep = preflight(parse_fdf_params(device_fdf), parse_fdf_params(ele_fdf))
    ids = {c.id for c in rep.checks if c.severity == "error"}
    # The contract + cell + transverse-k + kz invariants must NOT error.
    assert "contract.meshcutoff" not in ids
    assert "contract.xc" not in ids
    assert "contract.basis" not in ids
    assert "cell.transverse" not in ids
    assert "kgrid.transverse" not in ids
    assert "kgrid.device_kz" not in ids
    assert "kgrid.electrode_kz" not in ids


def test_emitted_electrode_kz_is_dense_and_transverse_matches_device():
    dev = _au_device()
    cfg = TransportConfig(job_name="junc", k_mesh_transverse=(6, 6, 1))
    _job, ele_fdf, _m = electrode_wizard(
        dev, cfg, which="L", electrode_kz=48)[0]
    p = parse_fdf_params(ele_fdf)
    assert p.kgrid == (6, 6, 48)            # transverse matches device, kz dense
    assert p.saves_ts_hs is True
    assert p.solution_method == "diagon"


def test_both_emits_one_fdf_per_electrode():
    dev = _au_device()
    cfg = TransportConfig(job_name="junc")
    models = electrode_wizard(dev, cfg, which="both")
    jobs = {job for job, _f, _m in models}
    assert jobs == {"junc_L-electrode", "junc_R-electrode"}


def test_no_electrode_regions_raises():
    plain = Structure(elements=["Au"], positions=np.zeros((1, 3)))
    with pytest.raises(ValueError):
        electrode_wizard(plain, TransportConfig())

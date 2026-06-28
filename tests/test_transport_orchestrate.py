"""Tests for the TranSIESTA 3-run orchestration
(molbuilder/transport/orchestrate.py)."""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import TransportConfig
from molbuilder.structure import Structure
from molbuilder.transport.orchestrate import (
    build_transport_bundle, render_relax_fdf, render_driver,
    _device_fdf_with_relaxed_geometry, _frozen_for_relax,
)
from molbuilder.transport.preflight import parse_fdf_params, preflight


def _au_device():
    d = 2.35
    elems, pos, left, bridge, right = [], [], [], [], []
    i = 0
    for layer in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, layer * d]); left.append(i); i += 1
    z0 = 4 * d
    for j in range(2):
        elems.append("S"); pos.append([1.0, 1.0, z0 + 1.5 + j * 1.8]); bridge.append(i); i += 1
    z1 = z0 + 1.5 + 2 * 1.8 + 1.5
    for layer in range(4):
        elems.append("Au"); pos.append([1.0, 1.0, z1 + layer * d]); right.append(i); i += 1
    return Structure(
        elements=elems, positions=np.asarray(pos, dtype=float),
        regions={"L-electrode": left, "bridge": bridge, "R-electrode": right})


def _cfg():
    return TransportConfig(job_name="junc", siesta_mesh_cutoff_ry=400,
                           k_mesh_transverse=(4, 4, 1))


# --------------------------------------------------------------------- #
#  frozen-lead selection                                                #
# --------------------------------------------------------------------- #


def test_frozen_defaults_to_electrode_atoms():
    dev = _au_device()
    frozen = _frozen_for_relax(dev)
    assert frozen == [0, 1, 2, 3, 6, 7, 8, 9]      # both electrodes, not bridge


def test_frozen_unions_explicit_with_electrodes():
    # The leads MUST always be frozen (clone invariant); explicit
    # frozen_atoms are unioned in, never substituted for the electrodes.
    dev = _au_device()
    dev.frozen_atoms = [4]                          # a bridge atom
    frozen = _frozen_for_relax(dev)
    assert set(frozen) == {0, 1, 2, 3, 6, 7, 8, 9, 4}   # electrodes + the extra


# --------------------------------------------------------------------- #
#  relax fdf                                                            #
# --------------------------------------------------------------------- #


def test_relax_fdf_has_cg_and_frozen_constraints():
    fdf = render_relax_fdf(_au_device(), _cfg(), relax_steps=50)
    assert "MD.TypeOfRun           CG" in fdf
    assert "MD.NumCGsteps          50" in fdf
    assert "%block Geometry.Constraints" in fdf
    # 1-based electrode indices appear in a position line
    assert "position 1 2 3 4 7 8 9 10" in fdf
    assert "SolutionMethod" not in fdf            # NOT a TranSIESTA run
    assert "TS.Elecs" not in fdf


def test_relax_fdf_shares_contract_with_device():
    dev, cfg = _au_device(), _cfg()
    relax = parse_fdf_params(render_relax_fdf(dev, cfg))
    from molbuilder.transport.transiesta import TransiestaEngine
    device = parse_fdf_params(TransiestaEngine.render_script(dev, cfg))
    assert relax.mesh_cutoff_ry == device.mesh_cutoff_ry
    assert relax.energy_shift_ry == device.energy_shift_ry
    assert relax.xc == device.xc
    assert relax.basis_size == device.basis_size
    assert relax.kgrid == device.kgrid            # same transverse k + kz=1


# --------------------------------------------------------------------- #
#  device hand-off                                                      #
# --------------------------------------------------------------------- #


def test_device_fdf_injects_usesavexv():
    fdf = _device_fdf_with_relaxed_geometry(_au_device(), _cfg())
    assert "MD.UseSaveXV           .true." in fdf
    # flag sits after the coordinate block
    assert (fdf.index("MD.UseSaveXV")
            > fdf.index("%endblock AtomicCoordinatesAndAtomicSpecies"))


# --------------------------------------------------------------------- #
#  driver script                                                        #
# --------------------------------------------------------------------- #


def test_driver_orders_runs_and_handoffs():
    drv = render_driver("junc", ["junc_L-electrode", "junc_R-electrode"])
    # electrodes + relax appear before the device run
    i_elec = drv.index("junc_L-electrode")
    i_relax = drv.index('run_siesta "junc_relax"')
    i_handoff = drv.index("cp \"junc_relax.XV\" \"junc.XV\"")
    i_device = drv.index('run_siesta "junc"    ')
    i_tbt = drv.index("=== tbtrans:")
    assert i_elec < i_device
    assert i_relax < i_handoff < i_device < i_tbt
    assert "set -euo pipefail" in drv


def test_driver_is_valid_bash():
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash not available")
    drv = render_driver("junc", ["junc_L-electrode", "junc_R-electrode"])
    r = subprocess.run([bash, "-n"], input=drv, text=True,
                       capture_output=True)
    assert r.returncode == 0, r.stderr


# --------------------------------------------------------------------- #
#  the full bundle                                                      #
# --------------------------------------------------------------------- #


def test_bundle_has_all_four_fdfs_plus_driver():
    bundle = build_transport_bundle(_au_device(), _cfg())
    names = set(bundle.files)
    assert "junc_relax.fdf" in names
    assert "junc_L-electrode.fdf" in names
    assert "junc_R-electrode.fdf" in names
    assert "junc.fdf" in names
    assert "run-transport.sh" in names
    assert "README.txt" in names


def test_bundle_electrode_passes_preflight_against_device():
    bundle = build_transport_bundle(_au_device(), _cfg())
    device = parse_fdf_params(bundle.files["junc.fdf"])
    ele = parse_fdf_params(bundle.files["junc_L-electrode.fdf"])
    rep = preflight(device, ele)
    errs = {c.id for c in rep.checks if c.severity == "error"}
    assert not (errs & {"contract.meshcutoff", "contract.xc", "contract.basis",
                        "cell.transverse", "kgrid.transverse",
                        "kgrid.device_kz", "kgrid.electrode_kz"})


def test_bundle_electrode_tshs_name_matches_device_reference():
    # the win over the manual flow: no rename needed.
    bundle = build_transport_bundle(_au_device(), _cfg())
    device_fdf = bundle.files["junc.fdf"]
    # device references HS  junc_L-electrode.TSHS; the driver produces it
    assert "junc_L-electrode.TSHS" in device_fdf
    assert 'run_siesta "junc_L-electrode"          "junc_L-electrode.TSHS"' \
        in bundle.files["run-transport.sh"]

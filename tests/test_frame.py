"""Frame and Trajectory dataclass smoke tests.

These tests exercise the parsers' new primary API (Iterator[Frame]
via Trajectory) directly, without going through the legacy-dict
adapter.  Most of the existing parser tests in tests/watch/ assert
on the legacy dict shape via trajectory_to_legacy_dict; this file is
the explicit Frame-shape coverage for Phase 2.

Spec: docs/design.md "Frame and Trajectory (parser output type)".
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.frame import Frame, Trajectory
from molbuilder.parsers.molwatch_log import MolwatchLogParser
from molbuilder.parsers.siesta import SiestaParser
from molbuilder.structure import Structure


_MW_SAMPLE = """\
# molwatch trajectory log v1
# engine: pyscf
# job: water_relax
# units: energy=eV, force=eV/Ang, coords=Ang

==== molwatch step 0 begin ====
step_index: 0
n_atoms: 3
coordinates (Ang):
   O      0.00000000      0.00000000      0.00000000
   H      0.95700000      0.00000000      0.00000000
   H     -0.23900000      0.92700000      0.00000000
energy (eV): -76.12345600
forces (eV/Ang):
   O     -0.00100000     -0.00200000      0.00000000
   H      0.00050000      0.00100000      0.00000000
   H      0.00050000      0.00100000      0.00000000
max_force (eV/Ang): 0.00240000
scf_history begin
#  cycle  energy(eV)  delta_E(eV)  gnorm  ddm
       1   -76.0      0.00         0.05   0.1
       2   -76.1     -0.10         0.005  0.01
scf_history end
==== molwatch step 0 end ====
"""


def test_trajectory_is_a_frame_container(tmp_path):
    p = tmp_path / "x.molwatch.log"
    p.write_text(_MW_SAMPLE)
    traj = MolwatchLogParser.parse(str(p))
    assert isinstance(traj, Trajectory)
    assert traj.source_format == "pyscf"   # from "# engine: pyscf" header
    assert traj.lattice is None
    # Trajectory supports len(), iteration, and indexing.
    assert len(traj) == 1
    assert traj[0] is traj.frames[0]
    assert list(traj) == traj.frames


def test_frame_carries_structure_and_physics(tmp_path):
    p = tmp_path / "x.molwatch.log"
    p.write_text(_MW_SAMPLE)
    traj = MolwatchLogParser.parse(str(p))
    f = traj[0]
    assert isinstance(f, Frame)
    assert isinstance(f.structure, Structure)
    assert f.structure.elements == ["O", "H", "H"]
    np.testing.assert_allclose(f.structure.positions[0], [0.0, 0.0, 0.0])
    assert f.step_index == 0
    assert f.energy == pytest.approx(-76.123456)
    assert f.max_force == pytest.approx(0.0024)
    # Forces survive as an (N, 3) ndarray.
    assert isinstance(f.forces, np.ndarray)
    assert f.forces.shape == (3, 3)
    # SCF history is a list of dicts with the unified key set.
    assert f.scf_history is not None
    assert len(f.scf_history) == 2
    assert {"cycle", "energy", "delta_E", "gnorm", "ddm"} <= set(
        f.scf_history[0].keys()
    )


def test_siesta_trajectory_lattice(tmp_path):
    """SIESTA puts the cell on Trajectory.lattice (3x3 ndarray), not
    on per-Frame lattice -- it's constant across frames."""
    sample = (
        "Welcome to SIESTA\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        "outcell: Unit cell vectors (Ang):\n"
        "       10.0    0.0    0.0\n"
        "        0.0   10.0    0.0\n"
        "        0.0    0.0   10.0\n"
        "\n"
        "siesta: E_KS(eV) =          -100.0\n"
    )
    p = tmp_path / "run.out"
    p.write_text(sample)
    traj = SiestaParser.parse(str(p))
    assert traj.lattice is not None
    assert isinstance(traj.lattice, np.ndarray)
    assert traj.lattice.shape == (3, 3)
    np.testing.assert_allclose(np.diag(traj.lattice), [10.0, 10.0, 10.0])
    # All frames inherit the trajectory-level lattice; per-frame
    # Frame.lattice is reserved for variable-cell trajectories.
    assert all(f.lattice is None for f in traj.frames)


def test_frame_post_init_coerces_list_inputs():
    """Frame accepts forces / lattice as plain lists; __post_init__
    upgrades them to ndarrays so downstream code sees consistent
    types."""
    s = Structure(elements=["H"],
                  positions=np.array([[0.0, 0.0, 0.0]]))
    f = Frame(
        structure  = s,
        step_index = 0,
        forces     = [[0.1, 0.2, 0.3]],
        lattice    = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert isinstance(f.forces, np.ndarray)
    assert f.forces.shape == (1, 3)
    assert isinstance(f.lattice, np.ndarray)
    assert f.lattice.shape == (3, 3)

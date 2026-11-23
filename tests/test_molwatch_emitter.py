"""Unit tests for molbuilder.trajectory_log.emitter.MolwatchEmitter.

The class is the source-of-truth for the streaming ``<JOB>.molwatch.log``
emitter, and the PySCF script imports it (from ``mb_pyscf.pyz``).

These tests exercise the class directly with a minimal fake-mol stub
and a hand-built ``envs`` dict, so format invariants (header lines,
preview-block shape, opt-step block shape, scf_history table) are
pinned at unit-test time without needing PySCF.

WHY ITS INPUT IS TYPED, NOT RUN: the writer's inputs are what PySCF hands
it -- a ``mol`` and geomeTRIC's callback values -- and PySCF lives only in
the ``molbuilder-pySCF`` env, never in the suite's.  The values here are
chosen so each converted number is known by hand: a gradient of -0.05
Ha/Bohr is a force of +0.05 x 51.42 eV/Ang, the sign and the unit the
science rule pins (`process/testing.md` § 3b, the scientific exception).
No run makes a number whose conversion a test knows in advance; the log a
real PySCF run writes through this class is read end to end by
`test_pyscf_relaxation_outcome_e2e.py` (the run, `support.real_runs`).

This file pins the writer's lines -- and, for its SCF table, what the
reader reads back from them (the one table both take their order from,
`trajectory_log.keys.SCF_TABLE`)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
from molbuilder.trajectory_log import MolwatchEmitter


class _FakeMol:
    """Minimal stand-in for a PySCF ``mol`` object.  Implements only
    the three methods the emitter calls: ``natm``, ``atom_coords``,
    ``atom_symbol``."""

    def __init__(self, elements, positions_ang):
        self._elements = list(elements)
        self._positions = np.asarray(positions_ang, dtype=float)
        self.natm = len(self._elements)

    def atom_coords(self, unit="Bohr"):
        if unit == "Ang":
            return self._positions
        # Bohr conversion -- emitter only ever asks for Ang, so any
        # non-Ang call is a test-bug indicator.
        raise AssertionError(f"unexpected unit {unit!r}")

    def atom_symbol(self, i):
        return self._elements[i]


@pytest.fixture
def h2_mol():
    return _FakeMol(["H", "H"], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])


def test_constructor_writes_header_and_preview_block(tmp_path, h2_mol):
    """Constructor writes (a) the v1 header lines and (b) the step-0
    preview block, in that order, to the given path.  The preview is
    written before any SCF would run -- so molwatch can render the
    structure the moment a user loads the file."""
    p = tmp_path / "test.molwatch.log"
    em = MolwatchEmitter(str(p), "h2", h2_mol)

    txt = p.read_text()
    # Header lines
    assert "# molwatch trajectory log v1" in txt
    assert "# generator: molbuilder/pyscf_input" in txt
    assert "# engine: pyscf" in txt
    assert "# job: h2" in txt
    assert "# units: energy=eV, force=eV/Ang, coords=Ang" in txt
    assert re.search(r"^# created: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$",
                     txt, re.M), "missing ISO 8601 # created: line"
    # Step 0 preview block
    block = txt.split("==== molwatch step 0 begin ====", 1)[1]
    block = block.split("==== molwatch step 0 end ====", 1)[0]
    assert "step_index: 0" in block
    assert "kind: initial_preview" in block
    assert re.search(r"wall_time: \d+\.\d{3}", block)
    assert "n_atoms: 2" in block
    # Both H atoms appear in the coordinates table.  Layout is
    # `   {el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}` -- match
    # tolerantly so the test pins data, not whitespace tuning.
    assert re.search(r"H\s+0\.00000000\s+0\.00000000\s+0\.00000000", block)
    assert re.search(r"H\s+0\.74000000\s+0\.00000000\s+0\.00000000", block)
    # Energy / forces are null in the preview
    assert "energy (eV): None" in block
    assert "max_force (eV/Ang): None" in block
    # SCF history empty (begin immediately followed by end)
    assert "scf_history begin\nscf_history end" in block
    # Step counter advanced past the preview
    assert em._step == 1


def test_scf_cycle_hook_buffers_per_cycle_data(tmp_path, h2_mol):
    """scf_cycle_hook accumulates one entry per SCF cycle into
    self._scf_buf.  The next opt_step_hook call drains the buffer
    into the step's scf_history table.  cycle=0 resets the buffer
    (a new SCF run is starting)."""
    p = tmp_path / "test.molwatch.log"
    em = MolwatchEmitter(str(p), "h2", h2_mol)

    # Two cycles of a fresh SCF run:
    em.scf_cycle_hook({"cycle": 0, "e_tot": -1.10, "last_hf_e": None,
                       "norm_gorb": 1.0e-2, "norm_ddm": 1.0e-3})
    em.scf_cycle_hook({"cycle": 1, "e_tot": -1.15, "last_hf_e": -1.10,
                       "norm_gorb": 1.0e-4, "norm_ddm": 1.0e-5})
    assert len(em._scf_buf) == 2
    assert em._scf_buf[0]["cycle"] == 1     # 1-indexed in our log
    assert em._scf_buf[1]["cycle"] == 2
    # Hartree -> eV conversion for energy
    assert em._scf_buf[0]["energy"] == pytest.approx(-1.10 * HARTREE_EV)
    # Delta-E uses last_hf_e where available
    assert em._scf_buf[1]["delta_E"] == pytest.approx(
        (-1.15 - -1.10) * HARTREE_EV
    )
    # the orbital-gradient norm is an ENERGY (Hartree over dimensionless
    # orbital rotations): it converts as one, never as a force
    assert em._scf_buf[0]["gnorm"] == pytest.approx(1.0e-2 * HARTREE_EV)

    # New SCF run starts: cycle=0 should reset the buffer.
    em.scf_cycle_hook({"cycle": 0, "e_tot": -1.16, "last_hf_e": None,
                       "norm_gorb": 5.0e-3, "norm_ddm": 5.0e-4})
    assert len(em._scf_buf) == 1, "cycle=0 must clear prior cycles"

    # Missing cycle key is a no-op (geomeTRIC pre-callback edge case).
    em.scf_cycle_hook({"e_tot": -1.16})
    assert len(em._scf_buf) == 1
    # Missing e_tot is also a no-op (cycle without convergence info).
    em.scf_cycle_hook({"cycle": 1})
    assert len(em._scf_buf) == 1


def test_opt_step_hook_writes_block_with_forces_and_scf_history(tmp_path,
                                                                h2_mol):
    """opt_step_hook drains the SCF buffer into a marker-delimited
    block, with forces in eV/Å (sign-flipped from the gradient input
    which is in Ha/Bohr) and a scf_history table of buffered cycles."""
    p = tmp_path / "test.molwatch.log"
    em = MolwatchEmitter(str(p), "h2", h2_mol)

    # Buffer one SCF cycle so the opt-step block has scf_history.
    em.scf_cycle_hook({"cycle": 0, "e_tot": -1.15, "last_hf_e": -1.10,
                       "norm_gorb": 1.0e-4, "norm_ddm": 1.0e-5})

    # Construct an envs dict the way pyscf.geomopt.geometric_solver
    # would: gradient as a (natm*3) array in Ha/Bohr.
    grad_ha_bohr = np.array([
        [-0.05, 0.0, 0.0],   # force on H1 pulls it left
        [+0.05, 0.0, 0.0],   # force on H2 pulls it right
    ])
    em.opt_step_hook({
        "mol":       h2_mol,
        "energy":    -1.15,
        "gradients": grad_ha_bohr,    # NOTE plural -- geomeTRIC's key
    })

    txt = p.read_text()
    block = txt.split("==== molwatch step 1 begin ====", 1)[1]
    block = block.split("==== molwatch step 1 end ====", 1)[0]
    # Energy in eV (Hartree * 27.211...)
    assert re.search(r"energy \(eV\): -3?\d\.\d{8}", block)
    # Forces are -gradient * (Hartree/Bohr -> eV/Ang) = -grad * 51.4221
    # H1's gradient was -0.05 Ha/Bohr, so its force is +0.05 * 51.42... eV/Ang
    expected_F = 0.05 * HARTREE_BOHR_EV_ANGSTROM_ASE
    assert re.search(rf"H\s+{expected_F:.8f}\s+", block), (
        f"expected force {expected_F:.8f} eV/Ang for H1; block was:\n{block}"
    )
    # max_force
    assert re.search(rf"max_force \(eV/Ang\): {expected_F:.8f}", block)
    # SCF history table populated from the buffer
    assert "scf_history begin" in block
    assert "scf_history end" in block
    assert "energy(eV)" in block      # column header
    # Step counter advanced
    assert em._step == 2


def test_opt_step_hook_missing_envs_is_noop(tmp_path, h2_mol):
    """A geomeTRIC callback that fires before mol/energy/gradients are
    populated is a no-op -- the emitter must not crash and must not
    write a stale block.  One example, as the emitter has one rule for
    it: the gradients not yet there."""
    p = tmp_path / "test.molwatch.log"
    em = MolwatchEmitter(str(p), "h2", h2_mol)

    before = p.read_text()
    em.opt_step_hook({"mol": h2_mol, "energy": -1.0})
    after = p.read_text()
    assert before == after, "no-op envs must not write to the log"
    # Step counter must NOT have advanced.
    assert em._step == 1   # only the preview block was written


def test_emitted_lines_exact_for_zero_force_step(tmp_path, h2_mol):
    """Pin the formatting of a clean opt step (zero forces, no SCF
    history): every line of the block must match an exact substring,
    so a parser change downstream is forced to be deliberate."""
    p = tmp_path / "test.molwatch.log"
    em = MolwatchEmitter(str(p), "h2", h2_mol)
    em.opt_step_hook({
        "mol":       h2_mol,
        "energy":    0.0,
        "gradients": np.zeros((2, 3)),
    })
    txt = p.read_text()
    block = txt.split("==== molwatch step 1 begin ====", 1)[1]
    block = block.split("==== molwatch step 1 end ====", 1)[0]
    for needle in [
        "step_index: 1",
        "n_atoms: 2",
        "coordinates (Ang):",
        "energy (eV): 0.00000000",
        "forces (eV/Ang):",
        "max_force (eV/Ang): 0.00000000",
        "scf_history begin",
        "scf_history end",
    ]:
        assert needle in block, f"missing {needle!r}; block:\n{block}"


def test_each_scf_value_is_read_back_from_the_column_it_was_written_in(
        tmp_path, h2_mol):
    """`engines/pyscf.md` § 4: the SCF table's columns are one table, in the
    order its rows write them (`trajectory_log.keys.SCF_TABLE`), which the
    writer renders and the reader reads by -- so each cycle's energy, its
    change, the orbital gradient's norm, the density's change and the clock
    come back from the registered reader as the deck wrote them: a norm the
    deck did not have read as none, and the first cycle's change 0, the
    change from no energy before it as the columns' declaration states it
    (`molwatch_grammar.SCF_COLUMNS`).  THE SILENT FAILURE: a column inserted
    in the writer alone, and the reader taking the density's change for the
    gradient's norm, with no error."""
    import time
    from molbuilder.parse import parse
    p = tmp_path / "h2.molwatch.log"
    em = MolwatchEmitter(str(p), "h2", h2_mol)
    before = time.time()
    em.scf_cycle_hook({"cycle": 0, "e_tot": -1.10, "last_hf_e": None,
                       "norm_gorb": 1.0e-2, "norm_ddm": 1.0e-3})
    em.scf_cycle_hook({"cycle": 1, "e_tot": -1.15, "last_hf_e": -1.10,
                       "norm_gorb": None, "norm_ddm": 1.0e-5})
    after = time.time()
    em.opt_step_hook({"mol": h2_mol, "energy": -1.15,
                      "gradients": np.zeros((2, 3))})
    rows = parse(str(p)).frames[1].scf_history
    assert [(r["cycle"], r["energy"], r["delta_E"], r["gnorm"], r["ddm"])
            for r in rows] == [
        (1, pytest.approx(-1.10 * HARTREE_EV), 0.0,
         pytest.approx(1.0e-2 * HARTREE_EV), pytest.approx(1.0e-3)),
        (2, pytest.approx(-1.15 * HARTREE_EV),
         pytest.approx(-0.05 * HARTREE_EV), None, pytest.approx(1.0e-5)),
    ], rows
    assert all(before - 1e-3 <= r["wall_clock_s"] <= after + 1e-3
               for r in rows), rows

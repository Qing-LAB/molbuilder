"""Streaming emitter for the unified ``<JOB>.molwatch.log`` format.

This module is the **single source of truth** for the
``MolwatchEmitter`` class.  The PySCF script generator at
``molbuilder/pyscf/input.py:_emit_molwatch_emitter`` inlines this
class via :func:`inspect.getsource` into the user-runnable script,
so the generated script stays self-contained -- the user's runtime
environment doesn't need molbuilder installed.

Why a real module instead of an inline list-of-strings:

* The class can be **unit-tested** directly: instantiate, call the
  hooks with stub envs dicts, assert the on-disk file contents.
  Pre-fix, behaviour could only be verified by subprocess-running a
  generated script (review-fix D's smoke test).
* The class can be **type-checked, linted, and read** as normal Python.
  The inline list-of-strings was opaque to every tool.
* A single class definition feeds **both** the test fixtures AND the
  emitted script -- no risk of drift.

Format: see :mod:`molbuilder.trajectory_log.format` for the spec
docstring (the same format the standalone ``write_initial_preview``
helper writes).  The emitter writes a step-0 preview block on
construction (so molwatch can render the molecule from second one,
even before SCF starts) plus one block per accepted opt step.

The class is named without a leading underscore here because it's
the public surface of this module; the generated script also uses
the same name (the inlined source is verbatim).  Treat it as a
public spec contract -- changing the format breaks the molwatch
parser at :mod:`molbuilder.parsers.molwatch_log`.
"""

from __future__ import annotations

import time as _mw_time

import numpy as _mw_np


class MolwatchEmitter:
    """Streams ``<JOB>.molwatch.log`` with one marker-delimited
    block per opt step.  See molbuilder spec for the format.
    """
    HARTREE_TO_EV          = 27.211386245988
    HARTREE_BOHR_TO_EV_ANG = 51.42208619

    def __init__(self, path, job, mol):
        self.path = path
        self.job  = job
        self._scf_buf   = []   # per-cycle dicts; reset each new SCF
        self._step      = 0    # log block counter; step 0 reserved for preview
        with open(self.path, 'w') as fh:
            fh.write("# molwatch trajectory log v1\n")
            fh.write("# generator: molbuilder/pyscf_input\n")
            fh.write("# engine: pyscf\n")
            fh.write(f"# job: {self.job}\n")
            fh.write("# units: energy=eV, force=eV/Ang, coords=Ang\n")
            fh.write(f"# created: {_mw_time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            fh.write("\n")
        # Step 0: initial-state preview, written BEFORE any SCF runs.
        # Carries coordinates only; energy / forces / scf_history are
        # null because none have been computed yet.  This guarantees
        # molwatch can render the molecule the moment a user loads the
        # log -- they don't have to wait for the first SCF to finish.
        self._write_initial_preview(mol)

    def _write_initial_preview(self, mol):
        coords_A = mol.atom_coords(unit='Ang')
        elements = [mol.atom_symbol(i) for i in range(mol.natm)]
        idx = self._step
        with open(self.path, 'a') as fh:
            fh.write(f"==== molwatch step {idx} begin ====\n")
            fh.write(f"step_index: {idx}\n")
            fh.write("kind: initial_preview\n")
            fh.write(f"wall_time: {_mw_time.time():.3f}\n")
            fh.write(f"n_atoms: {mol.natm}\n")
            fh.write("coordinates (Ang):\n")
            for i, el in enumerate(elements):
                x, y, z = coords_A[i]
                fh.write(f"   {el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")
            fh.write("energy (eV): None\n")
            fh.write("forces (eV/Ang):\n")
            fh.write("max_force (eV/Ang): None\n")
            fh.write("scf_history begin\n")
            fh.write("scf_history end\n")
            fh.write(f"==== molwatch step {idx} end ====\n")
            fh.write("\n")
            fh.flush()
        self._step += 1

    # ----- SCF cycle hook (wired to mf.callback) -----
    def scf_cycle_hook(self, envs):
        cycle = envs.get('cycle', None)        # 0-indexed in PySCF
        if cycle is None:
            return
        if cycle == 0:
            # New SCF run starts: clear cycle buffer
            self._scf_buf = []
        e_tot     = envs.get('e_tot', None)
        last_e    = envs.get('last_hf_e', None)
        norm_gorb = envs.get('norm_gorb', None)
        norm_ddm  = envs.get('norm_ddm', None)
        if e_tot is None:
            return
        e_eV    = float(e_tot)  * self.HARTREE_TO_EV
        dE_eV   = (float(e_tot) - float(last_e)) * self.HARTREE_TO_EV \
                  if last_e is not None else 0.0
        g_eV_A  = (float(norm_gorb) * self.HARTREE_BOHR_TO_EV_ANG) \
                  if norm_gorb is not None else None
        ddm     = float(norm_ddm) if norm_ddm is not None else None
        self._scf_buf.append({
            'cycle':   int(cycle) + 1,        # 1-indexed in our log
            'energy':  e_eV,
            'delta_E': dE_eV,
            'gnorm':   g_eV_A,
            'ddm':     ddm,
        })

    # ----- opt step hook (wired to optimize(callback=...)) -----
    def opt_step_hook(self, envs):
        mol      = envs.get('mol')
        energy   = envs.get('energy')
        gradient = envs.get('gradients')
        if mol is None or energy is None or gradient is None:
            return
        coords_A = mol.atom_coords(unit='Ang')
        elements = [mol.atom_symbol(i) for i in range(mol.natm)]
        e_eV     = float(energy) * self.HARTREE_TO_EV
        F        = -_mw_np.asarray(gradient).reshape(-1, 3) \
                      * self.HARTREE_BOHR_TO_EV_ANG  # eV/Ang
        f_mag    = _mw_np.sqrt((F * F).sum(axis=1))
        max_f    = float(f_mag.max()) if f_mag.size else 0.0
        scf      = list(self._scf_buf)
        idx      = self._step
        with open(self.path, 'a') as fh:
            fh.write(f"==== molwatch step {idx} begin ====\n")
            fh.write(f"step_index: {idx}\n")
            fh.write(f"wall_time: {_mw_time.time():.3f}\n")
            fh.write(f"n_atoms: {mol.natm}\n")
            fh.write("coordinates (Ang):\n")
            for i, el in enumerate(elements):
                x, y, z = coords_A[i]
                fh.write(f"   {el:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}\n")
            fh.write(f"energy (eV): {e_eV:.8f}\n")
            fh.write("forces (eV/Ang):\n")
            for i, el in enumerate(elements):
                fx, fy, fz = F[i]
                fh.write(f"   {el:<2s}  {fx:14.8f}  {fy:14.8f}  {fz:14.8f}\n")
            fh.write(f"max_force (eV/Ang): {max_f:.8f}\n")
            fh.write("scf_history begin\n")
            fh.write("#  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)            ddm\n")
            for c in scf:
                g_str = (f"{c['gnorm']:.8e}" if c['gnorm'] is not None
                         else 'None')
                d_str = (f"{c['ddm']:.8e}" if c['ddm'] is not None
                         else 'None')
                fh.write(
                    f"   {c['cycle']:5d}   {c['energy']:18.8f}"
                    f"  {c['delta_E']:18.8f}  {g_str:>20s}  {d_str:>16s}\n"
                )
            fh.write("scf_history end\n")
            fh.write(f"==== molwatch step {idx} end ====\n")
            fh.write("\n")
            fh.flush()
        self._step += 1


__all__ = ["MolwatchEmitter"]

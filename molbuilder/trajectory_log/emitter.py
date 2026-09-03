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
parser at :mod:`molbuilder.parse.engines.molwatch`.
"""

from __future__ import annotations

import time as _mw_time

import numpy as _mw_np


class MolwatchEmitter:
    """Streams ``<JOB>.molwatch.log`` with one marker-delimited
    block per opt step.  See molbuilder spec for the format.
    """
    # Unit-conversion constants used at write time so ``.molwatch.log``
    # is unit-self-consistent and the parser does zero conversion.
    #
    # A LITERAL, AND IT HAS TO BE.  This class body is copied VERBATIM into
    # the standalone script the user runs on the cluster, where molbuilder is
    # not importable -- so `from molbuilder.constants import ...` here becomes
    # a NameError out there, several machines away from the edit.  (Tried it,
    # 2026-08-30: the generated preview script died on exactly that line.)
    # It is the same CODATA-2018 value `molbuilder/constants.py` holds; when
    # one changes, change both.
    #
    # HARTREE_TO_EV is the CODATA-2018 value 27.211386245988 eV/Hartree.
    # HARTREE_BOHR_TO_EV_ANG = 51.42208619 is the value used by ASE,
    # NIST historical tables, and most quantum-chemistry packages; the
    # CODATA-2018-derived value (27.211386245988 / 0.529177210903 =
    # 51.422067476) differs by ~4 ppm.  We pick the ASE/literature
    # convention so forces emitted here line up with what users see in
    # ASE / VASP / QE log files; the difference is well below the SCF
    # noise floor.
    HARTREE_TO_EV          = 27.211386245988
    HARTREE_BOHR_TO_EV_ANG = 51.42208619

    def __init__(self, path, job, mol, runtime_info=None,
                 convergence_targets=None):
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
            # Runtime-info header: one ``# runtime.<key>: <value>`` line
            # for EVERY key the caller handed us, in the order they
            # populated it.  The trajectory-log parser reads these back
            # into Trajectory.runtime_info; the /results trajectory
            # inspector renders them as the same CPU/GPU/Host rows the
            # spectra inspector uses.  Skipped when runtime_info is None
            # (callers without the shared threading-setup block).
            #
            # NO WHITELIST HERE, deliberately.  This loop used to carry a
            # literal tuple of eleven key names, and it had already
            # drifted from molbuilder.runtime_info.RUNTIME_INFO_KEYS --
            # ``max_memory_mb`` is IN the canonical list, was written by
            # every script, and was silently dropped on the way to disk.
            # Nothing could have caught it: the canonical tuple is
            # imported by nobody, and the reader
            # (parse/engines/molwatch.py) accepts any ``runtime.<key>``
            # it finds, so the writer was the only closed door in an
            # otherwise open pipe.
            #
            # A whitelist cannot live here anyway: this class is inlined
            # into the generated script verbatim via inspect.getsource,
            # which copies the class body ONLY -- the script runs in an
            # env where ``molbuilder`` is not importable, so the class
            # can never reference the canonical tuple at runtime.  Write
            # what you are given; let the caller decide what is worth
            # recording.  RUNTIME_INFO_KEYS stays as documentation of the
            # keys every engine SHOULD populate, which is a different job
            # from filtering.
            if runtime_info:
                for k, v in runtime_info.items():
                    # Strip newlines so a misbehaving value can't
                    # break the line-oriented parse.
                    v_str = str(v).replace("\n", " ").replace("\r", " ")
                    fh.write(f"# runtime.{k}: {v_str}\n")
            # Convergence-target header: one ``# convergence.<key>:
            # <value>`` line per known target.  The trajectory-log
            # parser reads these into ``runtime_info["convergence_
            # targets"]``; the /results inspector renders the threshold
            # line + "current vs target" readout from them.  Skipped
            # when convergence_targets is None (older scripts) -- the
            # parser tolerates absence and the inspector falls back to
            # a "targets not found in source" hint.
            if convergence_targets:
                # Two header shapes share this format:
                #
                #   FLAT  (legacy, single-stage runs):
                #     # convergence.max_force_tol_eV_per_A: 0.0231
                #
                #   NESTED  (staged runs, task #534):
                #     # convergence.01_coarse.max_force_tol_eV_per_A: 0.103
                #     # convergence.02_tight.max_force_tol_eV_per_A: 0.0231
                # (the first segment is the stage's artifact TOKEN --
                # digit-first, `job-contracts.md` 6.3 -- which is what the
                # reader's key grammar must accept)
                #
                # Detection: any top-level value that's a dict tags
                # the payload as nested-shape; otherwise flat.  Empty
                # input falls through.  Stage names are constrained
                # by the StageSpec validator to [A-Za-z0-9_]+ so they
                # round-trip cleanly through the parser regex.
                # Per-stage / per-engine convergence-target leaves.
                # Names are stable cross-engine (PySCF + SIESTA both
                # populate the subset that's meaningful for their
                # own convergence-criteria scheme).  Post #534 7a:
                # geomeTRIC's full 5-criteria set is plumbed end-to-end
                # for PySCF staged-opt: max + RMS grad, max + RMS
                # displ, energy-step + the legacy SCF / iter caps.
                _LEAF_KEYS = (
                    # Force / gradient thresholds (eV/Å convention)
                    "max_force_tol_eV_per_A",
                    "rms_force_tol_eV_per_A",
                    # Displacement thresholds (Å convention)
                    "max_displ_ang",
                    "rms_displ_ang",
                    # Energy-step + SCF self-consistency
                    "energy_step_tol_eV",
                    "scf_energy_tol",
                    # Iter caps (integers, surfaced for the UI)
                    "max_scf_iter",
                    "max_geom_iter",
                    # Legacy alias (SIESTA-side parser writes
                    # ``scf_grad_tol`` for the gradient norm; keep
                    # accepting it on the emitter for round-trip
                    # symmetry with the parser, even though
                    # ``rms_force_tol_eV_per_A`` is the canonical key).
                    "scf_grad_tol",
                )

                def _is_nested(ct):
                    return any(isinstance(v, dict)
                               for v in ct.values())

                def _write_flat(prefix, leaf_dict):
                    for k in _LEAF_KEYS:
                        v = leaf_dict.get(k)
                        if v is None:
                            continue
                        v_str = str(v).replace("\n", " ").replace("\r", " ")
                        fh.write(f"# convergence.{prefix}{k}: {v_str}\n")

                if _is_nested(convergence_targets):
                    for stage_name, leaf in convergence_targets.items():
                        if not isinstance(leaf, dict):
                            continue
                        _write_flat(f"{stage_name}.", leaf)
                else:
                    _write_flat("", convergence_targets)
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
        # Snapshot wall-clock at the moment this SCF cycle finished.
        # Surfaced as a 6th column on the per-cycle row so per-cycle
        # time is a plain difference of neighbours, with no client-side
        # stitching.  Same epoch-second format as the per-step
        # ``wall_time:`` line above.
        #
        # ``wall_time`` is this FILE FORMAT's column name and stays as
        # written -- logs already on disk must keep parsing.  It is an
        # absolute epoch, so the reader surfaces it under the name that
        # says so, ``wall_clock_s`` (docs/model/parse.md § 2a); the
        # translation is the parser's job, not this writer's.
        wt      = _mw_time.time()
        self._scf_buf.append({
            'cycle':     int(cycle) + 1,      # 1-indexed in our log
            'energy':    e_eV,
            'delta_E':   dE_eV,
            'gnorm':     g_eV_A,
            'ddm':       ddm,
            'wall_time': wt,
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
        # `_mw_`-PREFIXED LIKE EVERYTHING ELSE THIS CLASS BINDS.  The
        # deck is a PROGRAM: it does `from pyscf import gto, scf, dft`,
        # and a bare `scf` here rebinds that module to a list for the
        # rest of this function.  Nothing in this function reads the
        # module today, so nothing breaks today -- and that is exactly
        # the state in which the next edit inside it reaches for
        # `scf.RHF` and silently gets a list of dicts.  The convention
        # this class already follows (`_mw_np`, `_mw_time`) exists so
        # molbuilder's machinery cannot take a name the engine owns.
        _mw_scf  = list(self._scf_buf)
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
            fh.write("#  cycle      energy(eV)         delta_E(eV)        gnorm(eV/Ang)            ddm        wall_time(s)\n")
            for c in _mw_scf:
                g_str = (f"{c['gnorm']:.8e}" if c['gnorm'] is not None
                         else 'None')
                d_str = (f"{c['ddm']:.8e}" if c['ddm'] is not None
                         else 'None')
                # wall_time is the 6th column (epoch seconds, .3f).
                # Older logs without this column round-trip fine
                # through the parser (it skips beyond the 5th token).
                wt    = c.get('wall_time')
                w_str = (f"{wt:.3f}" if wt is not None else 'None')
                fh.write(
                    f"   {c['cycle']:5d}   {c['energy']:18.8f}"
                    f"  {c['delta_E']:18.8f}  {g_str:>20s}  {d_str:>16s}"
                    f"  {w_str:>16s}\n"
                )
            fh.write("scf_history end\n")
            fh.write(f"==== molwatch step {idx} end ====\n")
            fh.write("\n")
            fh.flush()
        self._step += 1


__all__ = ["MolwatchEmitter"]

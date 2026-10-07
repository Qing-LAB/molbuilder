"""Streaming emitter for the unified ``<JOB>.molwatch.log`` format.

This module is the **single source of truth** for the
``MolwatchEmitter`` class, and the PySCF script imports it: the file
travels beside every PySCF job inside ``mb_pyscf.pyz``
(``runwrap.PYSCF_COMPANIONS``, ``engines/pyscf.md`` § 3), where molbuilder
is not installed -- so it imports only the standard library, numpy and
``constants``, the last the two ways a travelling module does.

Why a real module:

* The class can be **unit-tested** directly: instantiate, call the
  hooks with stub envs dicts, assert the on-disk file contents.
* The class can be **type-checked, linted, and read** as normal Python.
* The class the tests exercise is the class the run imports -- no
  copy to drift.

Format: the spec is in ``docs/engines/pyscf.md`` (the same format the
standalone ``write_initial_preview`` helper writes).  The emitter writes a step-0 preview block on
construction (so molwatch can render the molecule from second one,
even before SCF starts) plus one block per accepted opt step.

The class is named without a leading underscore here because it's
the public surface of this module; the script imports it under this
name.  Treat it as a public spec contract -- changing the format
breaks the molwatch parser at :mod:`molbuilder.parse.engines.molwatch`.
"""

from __future__ import annotations

import time

import numpy as np

# TWO WAYS, because this module travels: the PySCF script imports it from
# `mb_pyscf.pyz` beside the job (`runwrap.PYSCF_COMPANIONS`), where the
# package is not installed.  The force factor is the ASE convention, as the
# log's reader converts with (`constants` says why there are two).
try:                                        # inside molbuilder
    from ..constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    from ..pyscf.end_lines import FOOTER_CONCLUDED, FOOTER_ERROR
except ImportError:                         # beside a job, in mb_pyscf.pyz
    from constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    from end_lines import FOOTER_CONCLUDED, FOOTER_ERROR


#: The convergence targets a log's header states, one ``# convergence.
#: [<stage>.]<key>: <value>`` line each -- the keys the reader asks for
#: (`molwatch_grammar.parse_convergence_line`, `trajectory/core.js`), whose
#: threshold lines the Results tab draws.  Force and displacement in eV/A and
#: A, the energy step in eV, the two caps as counts.
_LEAF_KEYS = (
    "max_force_tol_eV_per_A",
    "rms_force_tol_eV_per_A",
    "max_displ_ang",
    "rms_displ_ang",
    "energy_step_tol_eV",
    "max_scf_iter",
    "max_geom_iter",
)


def _convergence_lines(targets) -> list:
    """``targets`` flat (``{key: value}``) or nested by the stage's artifact
    TOKEN (``{"01_coarse": {key: value}}`` -- digit-first, `job-contracts.md`
    6.3, which the reader's key grammar accepts), as header lines."""
    def flat(prefix, leaf):
        return [f"# convergence.{prefix}{k}: "
                + str(leaf[k]).replace("\n", " ").replace("\r", " ")
                for k in _LEAF_KEYS if leaf.get(k) is not None]
    if any(isinstance(v, dict) for v in targets.values()):
        return [line for stage, leaf in targets.items()
                if isinstance(leaf, dict) for line in flat(f"{stage}.", leaf)]
    return flat("", targets)


def header_and_preview(*, job, engine, generator, elements, coords_ang,
                       frozen_atoms=(), runtime_info=None,
                       convergence_targets=None) -> str:
    """The text a ``.molwatch.log`` starts with: its header and the step-0
    preview -- the structure, nothing computed yet.  THE ONE WRITER of both:
    prep's seed writes it before a run starts (`format.write_initial_preview`)
    and the PySCF script's :class:`MolwatchEmitter` when its run does.

    ``frozen_atoms`` -- the atoms the run holds, 0-based -- is the line a
    SIESTA run's ``.out`` states too, read as the log's own content
    (`model/parse.md` § 5.3).  ``runtime_info`` writes one
    ``# runtime.<key>: <value>`` line per key handed in, in its order, with
    no whitelist: the reader accepts any key.
    """
    lines = ["# molwatch trajectory log v1",
             f"# generator: {generator}",
             f"# engine: {engine}",
             f"# job: {job}",
             "# units: energy=eV, force=eV/Ang, coords=Ang",
             f"# created: {time.strftime('%Y-%m-%dT%H:%M:%S')}"]
    if frozen_atoms:
        lines.append("# frozen_atoms: " + " ".join(
            str(int(i)) for i in sorted(frozen_atoms)))
    for k, v in (runtime_info or {}).items():
        # a value cannot break the line-oriented parse
        lines.append(f"# runtime.{k}: "
                     + str(v).replace("\n", " ").replace("\r", " "))
    if convergence_targets:
        lines += _convergence_lines(convergence_targets)
    lines += ["",
              "==== molwatch step 0 begin ====",
              "step_index: 0",
              "kind: initial_preview",
              f"wall_time: {time.time():.3f}",
              f"n_atoms: {len(elements)}",
              "coordinates (Ang):"]
    for el, (x, y, z) in zip(elements, coords_ang):
        lines.append(f"   {el:<2s}  {float(x):14.8f}  {float(y):14.8f}  "
                     f"{float(z):14.8f}")
    lines += ["energy (eV): None",
              "forces (eV/Ang):",
              "max_force (eV/Ang): None",
              "scf_history begin",
              "scf_history end",
              "==== molwatch step 0 end ====",
              ""]
    return "\n".join(lines) + "\n"


class MolwatchEmitter:
    """Streams ``<JOB>.molwatch.log`` with one marker-delimited
    block per opt step.  See molbuilder spec for the format.
    """

    def __init__(self, path, job, mol, runtime_info=None,
                 convergence_targets=None, frozen_atoms=()):
        self.path = path
        self.job  = job
        self._scf_buf   = []   # per-cycle dicts; reset each new SCF
        # The log opens with its header and the step-0 preview, written
        # BEFORE any SCF runs -- coordinates only, energy / forces /
        # scf_history null -- so molwatch can render the molecule the moment
        # a person loads the log.  Through the one writer prep's seed uses
        # too (:func:`header_and_preview`).
        with open(self.path, 'w') as fh:
            fh.write(header_and_preview(
                job=job, engine="pyscf", generator="molbuilder/pyscf_input",
                elements=[mol.atom_symbol(i) for i in range(mol.natm)],
                coords_ang=mol.atom_coords(unit='Ang'),
                frozen_atoms=frozen_atoms, runtime_info=runtime_info,
                convergence_targets=convergence_targets))
        self._step      = 1    # log block counter; step 0 was the preview

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
        e_eV    = float(e_tot)  * HARTREE_EV
        dE_eV   = (float(e_tot) - float(last_e)) * HARTREE_EV \
                  if last_e is not None else 0.0
        # THE ORBITAL-GRADIENT NORM IS AN ENERGY: dE/d(kappa) over
        # dimensionless orbital rotations, in Hartree -- what PySCF compares
        # with `conv_tol_grad` -- so it converts as an energy, to eV.  It is
        # not a force.
        g_eV    = (float(norm_gorb) * HARTREE_EV) \
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
        wt      = time.time()
        self._scf_buf.append({
            'cycle':     int(cycle) + 1,      # 1-indexed in our log
            'energy':    e_eV,
            'delta_E':   dE_eV,
            'gnorm':     g_eV,
            'ddm':       ddm,
            'wall_time': wt,
        })

    # ----- opt step hook (wired to the relaxation's callback=) -----
    def opt_step_hook(self, envs):
        mol      = envs.get('mol')
        energy   = envs.get('energy')
        gradient = envs.get('gradients')
        if mol is None or energy is None or gradient is None:
            return
        coords_A = mol.atom_coords(unit='Ang')
        elements = [mol.atom_symbol(i) for i in range(mol.natm)]
        e_eV     = float(energy) * HARTREE_EV
        F        = -np.asarray(gradient).reshape(-1, 3) \
                      * HARTREE_BOHR_EV_ANGSTROM_ASE  # eV/Ang
        f_mag    = np.sqrt((F * F).sum(axis=1))
        max_f    = float(f_mag.max()) if f_mag.size else 0.0
        cycles   = list(self._scf_buf)
        idx      = self._step
        with open(self.path, 'a') as fh:
            fh.write(f"==== molwatch step {idx} begin ====\n")
            fh.write(f"step_index: {idx}\n")
            fh.write(f"wall_time: {time.time():.3f}\n")
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
            fh.write("#  cycle      energy(eV)         delta_E(eV)        gnorm(eV)                ddm        wall_time(s)\n")
            for c in cycles:
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


    # ----- the end lines, at exit -----
    def conclude_at_exit(self):
        """Write the log's end lines when the process exits
        (`engines/pyscf.md` § 4): ``# error: <the uncaught exception>`` when
        one ended it, then ``# concluded: <time>`` -- on a clean exit, an
        exception or Ctrl-C alike.  A process killed outright (SIGKILL, a
        power loss) writes neither, which reads correctly as a run not
        finished: not finished, whether slow or dead, which no file can tell
        (`model/parse.md` § 2b, P-S1).  The words are the reader's own
        (`end_lines`).  Installs an excepthook that remembers the exception
        and hands it to Python's own, and an atexit hook; a failure to
        write never breaks the exit."""
        import atexit
        import sys
        error: list = []

        def _remember(exc_type, exc_value, exc_tb):
            error.append(f"{exc_type.__name__}: {exc_value}")
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        def _finalize():
            try:
                with open(self.path, 'a') as fh:
                    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
                    if error:
                        fh.write(f"{FOOTER_ERROR} "
                                 f"{error[-1].replace(chr(10), ' ')}\n")
                    fh.write(f"{FOOTER_CONCLUDED} {ts}\n")
            except Exception:
                pass    # never break the person's exit on a logging issue

        sys.excepthook = _remember
        atexit.register(_finalize)


__all__ = ["MolwatchEmitter", "header_and_preview"]

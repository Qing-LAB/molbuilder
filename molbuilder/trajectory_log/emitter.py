"""Streaming emitter for the unified ``<JOB>.molwatch.log`` format.

This module is the **single source of truth** for the
``MolwatchEmitter`` class, and the PySCF script imports it: the file
travels beside every PySCF job inside ``mb_pyscf.pyz``
(``runwrap.members``, ``engines/pyscf.md`` § 3), where molbuilder
is not installed -- so it imports the standard library, numpy and the
members its `TRAVELS_WITH` names, those the two ways a travelling module
does.  It NEVER
travels with the monitor: the log's vocabulary the monitor's grammar reads
is `keys` (standard library only), and this writer imports it from there
*(the day the writer travelled with the monitor, every monitor on a
cluster's bare python died at import -- 2026-10-19)*.

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
# `mb_pyscf.pyz` beside the job (`runwrap.members`), where the
# package is not installed.  The force factor is the ASE convention, as the
# log's reader converts with (`constants` says why there are two).
# THE VOCABULARY IS `keys` (stdlib, travels with the monitor).
try:                                        # inside molbuilder
    from ..constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    from . import keys as _k
except ImportError:                         # beside a job, in mb_pyscf.pyz
    from constants import HARTREE_BOHR_EV_ANGSTROM_ASE, HARTREE_EV
    import keys as _k


def _convergence_lines(targets) -> list:
    """``targets`` flat (``{key: value}``) or nested by the stage's artifact
    TOKEN (``{"01_coarse": {key: value}}`` -- digit-first, `job-contracts.md`
    6.3, which the reader's key grammar accepts), as header lines."""
    def flat(prefix, leaf):
        return [f"{_k.CONVERGENCE_LINE}{prefix}{k}: "
                + str(leaf[k]).replace("\n", " ").replace("\r", " ")
                for k in _k.LEAF_KEYS if leaf.get(k) is not None]
    if any(isinstance(v, dict) for v in targets.values()):
        return [line for stage, leaf in targets.items()
                if isinstance(leaf, dict) for line in flat(f"{stage}.", leaf)]
    return flat("", targets)


def _atom_row(element, xyz) -> str:
    """One atom's row of a block's coordinates or forces table: its element,
    then its three numbers."""
    x, y, z = (float(v) for v in xyz)
    return f"   {element:<2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}"


def _optional(value, spec: str) -> str:
    return "None" if value is None else format(value, spec)


#: HOW EACH SCF COLUMN IS WRITTEN, by its key in the log's table
#: (`keys.SCF_TABLE`, whose order the row takes); a value not stated is
#: ``None``.
_SCF_CELLS = {
    "cycle":     lambda v: f"{v:5d}",
    "energy":    lambda v: f"{v:18.8f}",
    "delta_E":   lambda v: f"{v:18.8f}",
    "gnorm":     lambda v: f"{_optional(v, '.8e'):>20s}",
    "ddm":       lambda v: f"{_optional(v, '.8e'):>16s}",
    "wall_time": lambda v: f"{_optional(v, '.3f'):>16s}",
}
if set(_SCF_CELLS) != set(_k.SCF_TABLE_KEYS):
    # AT IMPORT, so a column the table gains stops the first thing that
    # loads the writer, not a run's first SCF row -- as a ValueError: an
    # ImportError would be caught by an importer's two-way import and said
    # as a module not found.
    raise ValueError(
        f"trajectory_log.emitter writes SCF columns "
        f"{sorted(_SCF_CELLS)} where the log's table is "
        f"{list(_k.SCF_TABLE_KEYS)}: give each column its cell here")


def _scf_row(cycle_record) -> str:
    """One SCF cycle's row, its columns in the table's order."""
    return "   " + "  ".join(_SCF_CELLS[key](cycle_record.get(key))
                             for key in _k.SCF_TABLE_KEYS)


def header_and_preview(*, job, engine, generator, elements, coords_ang,
                       frozen_atoms=(), runtime_info=None,
                       convergence_targets=None, atom_index_map=None) -> str:
    """The text a ``.molwatch.log`` starts with: its header and the step-0
    preview -- the structure, nothing computed yet.  THE ONE WRITER of both:
    prep's seed writes it before a run starts (`format.write_initial_preview`)
    and the PySCF script's :class:`MolwatchEmitter` when its run does.

    ``frozen_atoms`` -- the atoms the run holds, 0-based -- is the line a
    SIESTA run's ``.out`` states too, read as the log's own content
    (`model/parse.md` § 5.3).  ``runtime_info`` writes one
    ``# runtime.<key>: <value>`` line per key handed in, in its order, with
    no whitelist: the reader accepts any key.  ``atom_index_map`` -- the
    map's own block (`model/overview.md` § 2.2b) -- one
    ``# atom_index_map: {…}`` line, the log's atoms' indices said.
    """
    lines = [f"{_k.LOG_TITLE} {_k.LOG_VERSION}",
             f"# generator: {generator}",
             f"{_k.ENGINE_LINE} {engine}",
             f"# job: {job}",
             "# units: energy=eV, force=eV/Ang, coords=Ang",
             f"# created: {time.strftime('%Y-%m-%dT%H:%M:%S')}"]
    if frozen_atoms:
        lines.append(f"{_k.FROZEN_ATOMS_LINE} " + " ".join(
            str(int(i)) for i in sorted(frozen_atoms)))
    if atom_index_map is not None:
        import json
        lines.append(f"{_k.ATOM_INDEX_MAP_LINE} "
                     + json.dumps(atom_index_map, separators=(",", ":")))
    for k, v in (runtime_info or {}).items():
        # a value cannot break the line-oriented parse
        lines.append(f"{_k.RUNTIME_LINE}{k}: "
                     + str(v).replace("\n", " ").replace("\r", " "))
    if convergence_targets:
        lines += _convergence_lines(convergence_targets)
    lines += ["",
              _k.STEP_BEGIN.format(0),
              "step_index: 0",
              f"{_k.KIND} {_k.PREVIEW_KIND}",
              f"{_k.WALL_TIME} {time.time():.3f}",
              f"n_atoms: {len(elements)}",
              _k.COORDINATES_LINE]
    lines += [_atom_row(el, xyz) for el, xyz in zip(elements, coords_ang)]
    lines += [f"{_k.ENERGY} None",
              _k.FORCES_LINE,
              f"{_k.MAX_FORCE} None",
              _k.SCF_HISTORY_BEGIN,
              _k.SCF_HISTORY_END,
              _k.STEP_END.format(0),
              ""]
    return "\n".join(lines) + "\n"


class MolwatchEmitter:
    """Streams ``<JOB>.molwatch.log`` with one marker-delimited
    block per opt step.  See molbuilder spec for the format.
    """

    def __init__(self, path, job, mol, *, engine, generator,
                 runtime_info=None, convergence_targets=None,
                 frozen_atoms=(), atom_index_map=None):
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
                # THE ENGINE AND THE WRITER the header names are the deck's,
                # told by the line that builds this emitter
                # (`engines/pyscf.md` § 4), as prep's seed is told them.
                job=job, engine=engine, generator=generator,
                elements=[mol.atom_symbol(i) for i in range(mol.natm)],
                coords_ang=mol.atom_coords(unit='Ang'),
                frozen_atoms=frozen_atoms, runtime_info=runtime_info,
                convergence_targets=convergence_targets,
                atom_index_map=atom_index_map))
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
            fh.write(f"{_k.STEP_BEGIN.format(idx)}\n")
            fh.write(f"step_index: {idx}\n")
            fh.write(f"{_k.WALL_TIME} {time.time():.3f}\n")
            fh.write(f"n_atoms: {mol.natm}\n")
            fh.write(f"{_k.COORDINATES_LINE}\n")
            for el, xyz in zip(elements, coords_A):
                fh.write(_atom_row(el, xyz) + "\n")
            fh.write(f"{_k.ENERGY} {e_eV:.8f}\n")
            fh.write(f"{_k.FORCES_LINE}\n")
            for el, force in zip(elements, F):
                fh.write(_atom_row(el, force) + "\n")
            fh.write(f"{_k.MAX_FORCE} {max_f:.8f}\n")
            fh.write(f"{_k.SCF_HISTORY_BEGIN}\n")
            fh.write(f"{_k.SCF_TABLE_HEADER}\n")
            for c in cycles:
                fh.write(_scf_row(c) + "\n")
            fh.write(f"{_k.SCF_HISTORY_END}\n")
            fh.write(f"{_k.STEP_END.format(idx)}\n")
            fh.write("\n")
            fh.flush()
        self._step += 1


    # ----- the end lines -----
    def relaxed(self, converged: bool) -> None:
        """Write ``# relaxed: yes|no`` when the relaxation returns -- whether
        it converged, the deck's own word (`_GEOM_CONVERGED`), which the
        log's ending reads as the run's geometry verdict."""
        try:
            with open(self.path, 'a') as fh:
                fh.write(f"{_k.FOOTER_RELAXED} "
                         f"{'yes' if converged else 'no'}\n")
        except Exception:
            pass    # never break the run on a logging issue

    def conclude_at_exit(self):
        """Write the log's end lines when the process exits
        (`engines/pyscf.md` § 4): ``# error: <the uncaught exception>`` when
        one ended it, then ``# concluded: <time>`` -- on a clean exit, an
        exception or Ctrl-C alike.  A process killed outright (SIGKILL, a
        power loss) writes neither, which reads correctly as a run not
        finished: not finished, whether slow or dead, which no file can tell
        (`model/parse.md` § 2b, P-S1).  The words are the log's own
        vocabulary's (`keys`).  Installs an excepthook that remembers the exception
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
                        fh.write(f"{_k.FOOTER_ERROR} "
                                 f"{error[-1].replace(chr(10), ' ')}\n")
                    fh.write(f"{_k.FOOTER_CONCLUDED} {ts}\n")
            except Exception:
                pass    # never break the person's exit on a logging issue

        sys.excepthook = _remember
        atexit.register(_finalize)


__all__ = ["MolwatchEmitter", "header_and_preview"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.constants",
                "molbuilder.trajectory_log.keys")

#: ITS NAME BESIDE A JOB (`runwrap.bundle_members`): the name the PySCF
#: script has always imported it by -- a rename would change the script's
#: text and refuse a launched stage's prep as another calculation.
TRAVELS_AS = "molwatch_emitter"

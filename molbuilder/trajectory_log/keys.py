"""The trajectory log's VOCABULARY -- the convergence-target keys a
``.molwatch.log`` header states, their rows as a page says them, and the one
reader of a parsed block -- standard library only.

MODULE  trajectory_log.keys (L1; stdlib)
ROLE    the keys, named once; every producer and reader imports them
        (`pyscf/input.py`, `jobset/prep.py`, `siesta_grammar.TARGET_LINES`,
        `parse/contract.py`, the Results tab's rows)
TRAVELS beside every job's monitor (`runwrap.MONITOR_COMPANIONS`: the
        SIESTA grammar reads its echo of these targets with it), beside a
        PySCF deck and a vibration finish (the emitter imports it there) --
        so nothing but the standard library, on any python 3
        (`run-reports.md` § 2.3).  The log's WRITER, which needs numpy, is
        `emitter.py`, which never travels with the monitor.
"""
from __future__ import annotations

from typing import Optional

#: The convergence targets a log's header states, one ``# convergence.
#: [<stage>.]<key>: <value>`` line each -- the keys the reader asks for
#: (`molwatch_grammar.parse_convergence_line`, `trajectory/core.js`), whose
#: threshold lines the Results tab draws.  Force and displacement in eV/A and
#: A, the energy step in eV, the two caps as counts.
#: Each key, named once -- the PySCF deck (`pyscf/input.py`), prep's seed
#: (`jobset/prep.py`), SIESTA's echo (`siesta_grammar.TARGET_LINES`) and the
#: citation reader (`parse/contract.py`) import these; a spelling anywhere
#: else is a copy (2026-10-18: four producers had spelled them).
MAX_FORCE_TOL   = "max_force_tol_eV_per_A"
RMS_FORCE_TOL   = "rms_force_tol_eV_per_A"
MAX_DISPL       = "max_displ_ang"
RMS_DISPL       = "rms_displ_ang"
ENERGY_STEP_TOL = "energy_step_tol_eV"
MAX_SCF_ITER    = "max_scf_iter"
MAX_GEOM_ITER   = "max_geom_iter"

#: ...IN THE FILE'S ORDER, with the words a page says each in and its unit
#: (`web/trajectory.md` § 3): ``(key, label, unit)``.
LEAF_ROWS = (
    (MAX_FORCE_TOL,   "max |F|",        "eV/Å"),
    (RMS_FORCE_TOL,   "rms |F|",        "eV/Å"),
    (MAX_DISPL,       "max displ",      "Å"),
    (RMS_DISPL,       "rms displ",      "Å"),
    (ENERGY_STEP_TOL, "energy step",    "eV"),
    (MAX_SCF_ITER,    "MaxSCFIterations", ""),
    (MAX_GEOM_ITER,   "geom steps cap", ""),
)
LEAF_KEYS = tuple(r[0] for r in LEAF_ROWS)


def targets_of(convergence_targets) -> "Optional[dict]":
    """THE RUN'S OWN LEAVES out of a parsed ``convergence_targets`` block --
    nested one level by the stage's token, as every writer writes it
    (:func:`_convergence_lines`; `engines/pyscf.md` § 4): one stage nested
    is the run's; several is a header this run did not write alone, and
    the answer is ``None`` rather than a pick.  A FLAT block -- SIESTA's
    echo of its own limits (`siesta_reader`, ``source: siesta_input_echo``)
    and a progress log seeded before 2026-10-18 -- is the run's own as it
    stands.  The one rule, asked by the citation reader (`parse/contract.py`)
    and by the page's rows (:func:`targets_said`)."""
    if not isinstance(convergence_targets, dict):
        return None
    nested = [v for k, v in convergence_targets.items()
              if k != "source" and isinstance(v, dict)]
    if nested:
        return dict(nested[0]) if len(nested) == 1 else None
    flat = {k: v for k, v in convergence_targets.items() if k in LEAF_KEYS}
    return flat or None


def targets_said(convergence_targets) -> "list":
    """The run's targets as a page draws them -- ``[[label, text], ...]``
    in :data:`LEAF_ROWS`' order, each key the block states -- so the page
    spells no key and no unit (`web/trajectory.md` § 3)."""
    leaf = targets_of(convergence_targets) or {}
    out = []
    for key, label, unit in LEAF_ROWS:
        v = leaf.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        text = (f"{v:.4f}" if key in (MAX_FORCE_TOL, RMS_FORCE_TOL)
                else str(v)) + (f" {unit}" if unit else "")
        out.append([label, text])
    return out


#: The progress log's FOOTER, which a PySCF deck's exit hook appends to its
#: `.molwatch.log` (`pyscf/input.py`) and `molwatch_grammar` reads: the error
#: line when the run raised, then the concluded line.  Declared here, in the
#: log's vocabulary, so the writer and the reader share one spelling.
FOOTER_ERROR = "# error:"
FOOTER_CONCLUDED = "# concluded:"
#: ``# relaxed: yes|no`` -- whether the relaxation converged, written by the
#: deck when its relaxation returns (`MolwatchEmitter.relaxed`), read by the
#: progress log's ending (`_run_ending._molwatch_ending`) so a PySCF
#: relaxation's *geometry yes / NO* is stated like SIESTA's
#: (`web/results.md` § 2.4; 2026-10-18).
FOOTER_RELAXED = "# relaxed:"


__all__ = ["MAX_FORCE_TOL", "RMS_FORCE_TOL", "MAX_DISPL", "RMS_DISPL",
           "ENERGY_STEP_TOL", "MAX_SCF_ITER", "MAX_GEOM_ITER",
           "LEAF_ROWS", "LEAF_KEYS", "targets_of", "targets_said",
           "FOOTER_ERROR", "FOOTER_CONCLUDED", "FOOTER_RELAXED"]

"""What a run report may carry -- the ONE declaration (`engines/stages.md`
§ 6.9, `execution/run-reports.md` § 4.1a).

Each field in the report's own wire name, with the words the Task-setup card
offers it by, the chat card's label and unit, and WHICH RUNS CAN STATE IT: a
field is offered for a calculation only when a run of its engine and kind
writes what the field is read from (`run-reports.md` § 2.3).  A description
(`task.py`), the wrapper, the monitor, the listener and the Task-setup card
all read this table; none keeps a list.

**Stdlib only, and it travels beside every job** (`runwrap.MONITOR_COMPANIONS`):
the monitor reads it there, as `config_dir` travels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ReportField:
    """One field a report may carry."""
    #: The report's own name for it -- on the wire, in `task.json`'s
    #: ``notify.report``, in the listener's record: one word everywhere.
    name: str
    #: What the Task-setup card offers it as.
    offered_as: str
    #: The chat card's label for it, and its unit.
    card: str
    unit: str = ""
    #: The engines whose runs state it; empty is every engine.
    engines: Tuple[str, ...] = ()
    #: The calculation kinds that have it; empty is every kind.
    calculations: Tuple[str, ...] = ()

    def applies(self, engine: Optional[str],
                calculation: Optional[str]) -> bool:
        """Can a run of this engine and calculation state it?  ``None`` for
        either narrows nothing."""
        return ((not self.engines or engine is None
                 or engine in self.engines)
                and (not self.calculations or calculation is None
                     or calculation in self.calculations))


#: THE FIELDS, in the order a report shows them.  Each is read by the
#: framework reader `run-reports.md` § 2.3 names, and ``engines`` /
#: ``calculations`` say which runs write what it reads.
FIELDS: Tuple[ReportField, ...] = (
    ReportField("elapsed_s", "How long it has been running", "elapsed", " s"),
    ReportField("n_iters", "SCF iterations", "SCF iters"),
    ReportField("energy", "The last energy", "energy", " eV"),
    # The step the ENGINE began, in its own numbering: a relaxation's move, a
    # force-constant run's displacement.  A transport rung is a single point
    # and states none.
    ReportField("geom_step", "Which step it is on", "step",
                calculations=("optimization", "vibration")),
    # The largest force on an atom, as the run states it -- what a
    # relaxation converges on; a force-constant run's displaced geometries
    # and a transport rung's single points state one too.
    ReportField("max_force", "The largest force", "max force", " eV/Ang"),
    # Each engine's stamped SCF rows, timed by one rule (`run-reports.md`
    # § 2.3, `scf_timing_rows.timing_of`): the SIESTA wrapper's tee, a PySCF
    # deck's progress log.
    ReportField("per_iter_s", "Seconds per SCF iteration", "per iter", " s"),
)

#: Every field's name, in order -- the vocabulary a name is checked against.
NAMES: Tuple[str, ...] = tuple(f.name for f in FIELDS)


def for_run(engine: Optional[str] = None,
            calculation: Optional[str] = None) -> Tuple[ReportField, ...]:
    """The fields a run of this engine and calculation can state, in order."""
    return tuple(f for f in FIELDS if f.applies(engine, calculation))


def refusal(name: str, engine: Optional[str] = None,
            calculation: Optional[str] = None) -> Optional[str]:
    """Why ``name`` cannot be asked of such a run, in words -- or ``None``
    when it can.  The one sentence a description, a flag and a route give."""
    field = next((f for f in FIELDS if f.name == name), None)
    if field is None:
        return (f"{name!r} is not a report field. The fields are "
                f"{', '.join(NAMES)} -- the report's own names "
                f"(run-reports.md 4.1a), not labels. The calculation's name "
                f"and job id are always sent and are not on the list")
    if not field.applies(engine, calculation):
        where = " ".join(w for w in (engine, calculation) if w)
        can = ", ".join(f.name for f in for_run(engine, calculation))
        return (f"{name!r} is never reported for a {where} run -- nothing "
                f"such a run writes states it (stages.md 6.9). It can carry "
                f"{can}")
    return None

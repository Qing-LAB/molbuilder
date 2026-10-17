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
    #: A length of time, in seconds -- said in words (:func:`span_words`),
    #: never with a unit.
    span: bool = False
    #: A share in per cent, said as the engine prints it (:func:`pct_words`).
    percent: bool = False
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

    def said(self, value) -> str:
        """``value`` as a card shows it: a length of time by the span rule,
        a share as the engine prints it, else the number and its unit."""
        if self.span:
            return span_words(value)
        if self.percent:
            return pct_words(value) + self.unit
        return f"{value}{self.unit}"


#: THE FIELDS, in the order a report shows them.  Each is read by the
#: framework reader `run-reports.md` § 2.3 names, and ``engines`` /
#: ``calculations`` say which runs write what it reads.
FIELDS: Tuple[ReportField, ...] = (
    ReportField("elapsed_s", "How long it has been running", "elapsed",
                span=True),
    ReportField("n_iters", "SCF iterations", "SCF iters"),
    ReportField("energy", "The last energy", "energy", " eV"),
    # The step the ENGINE began, in its own numbering: a relaxation's move, a
    # force-constant run's displacement.  A transport rung is a single point
    # and states none.
    ReportField("geom_step", "Which step it is on", "step",
                calculations=("optimization", "vibration")),
    # The largest force on a free atom -- one the run does not hold -- as the
    # run states it --
    # what a relaxation converges on; a force-constant run's displaced
    # geometries and a transport rung's single points state one too, but a
    # TranSIESTA device, which computes none (`run-reports.md` § 2.7).
    ReportField("max_force", "The largest force", "max force", " eV/Ang"),
    # Every atom's, only when it is the larger -- a held atom carries it.
    ReportField("max_force_held", "The largest force on a held atom",
                "held atoms' force", " eV/Ang", engines=("siesta",)),
    # Each engine's stamped SCF rows, timed by one rule (`run-reports.md`
    # § 2.3, `scf_timing_rows.timing_of`): the SIESTA wrapper's tee, a PySCF
    # deck's progress log.
    ReportField("per_iter_s", "Seconds per SCF iteration", "per iter", " s"),
    # THE ENGINE'S OWN ESTIMATE (`run-reports.md` § 4.1a): of the spin pass
    # underway, the share TBtrans says is done and the time it says is left,
    # read by the output's reading pass -- the transport calculation's
    # transmission; which pass it is, the summary line says.
    ReportField("done_pct", "How much is done", "done", " %", percent=True,
                engines=("siesta",), calculations=("transport",)),
    ReportField("eta_s", "How long it says is left", "left", span=True,
                engines=("siesta",), calculations=("transport",)),
)


def span_words(seconds: float) -> str:
    """A length of time as every page says one (`web/ui-contract.md` § 4,
    `lib/time-ago.js`'s ``timeSpan``): each unit until one and a half of the
    next -- 90 s, 90 min, 36 h -- then days; ``12s``, ``5m``, ``6h``,
    ``6d``.  The same rule in Python: the monitor's, beside the job where
    the page's helper cannot reach, and every sentence the server composes
    for a page (`tbtrans.loop_words`)."""
    s = max(0.0, float(seconds))
    for size, unit, below in ((1.0, "s", 90.0), (60.0, "m", 5400.0),
                              (3600.0, "h", 129600.0)):
        if s < below:
            return f"{int(s / size + 0.5)}{unit}"
    return f"{int(s / 86400.0 + 0.5)}d"


def pct_words(value: float) -> str:
    """A share in per cent as an engine prints it -- TBtrans to two
    decimals (``f7.2``) -- its trailing zeros dropped: ``40``, ``10.01``."""
    return f"{round(float(value), 2):g}"

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

"""ParseResult hierarchy — frozen dataclasses returned by every parser.

Per ``docs/model/parse.md`` § 2.  Every concrete
result subclass sets a fixed ``result_kind`` discriminator that
consumers use to type-narrow.  Adding a new subclass requires a
new discriminator value + a doc update.

Forbidden by the doc:
* ``ParseResult`` subclasses are frozen; mutate via
  ``dataclasses.replace(result, ...)`` to make new copies.
* Adding a curated key list (engine_body_summary etc.) requires
  a doc + test update — silent additions break Results consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Forward-declared types — Frame and Structure live elsewhere; we
# only need the references for typing.  The actual types are
# imported lazily where needed to avoid circularity at package
# import.
from molbuilder.frame import Frame
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Common envelope                                                      #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParseWarning:
    """Level-3 fail-soft warning emitted by any parser.

    Mirrors the legacy ``parsers.base.ParseWarning`` shape so the
    existing renderer in Results can consume both during the
    migration window.
    """
    source:   str
    line_no:  Optional[int]
    snippet:  Optional[str]
    error:    str
    category: str


#: Every result carries this.  One number, not one per sub-package.
SCHEMA_VERSION = 1


def _source_str(source) -> str:
    """An absolute path where one can be had, the raw string otherwise.

    A parser that already read the file must not then fail on describing
    WHERE it came from -- the answer is in hand and the envelope is
    decoration.

    **Catches ``RuntimeError`` as well as ``OSError``, because the case
    the guard is for raises the first.**  The four builders this replaced
    all wrote ``except OSError``, and this docstring named "a broken
    symlink loop or an unreadable parent" as the reason.  Measured
    2026-09-05 on python 3.12: a symlink loop makes ``resolve()`` raise
    ``RuntimeError`` ("Symlink loop"), which is not an ``OSError``, so
    the guard never fired on its own example; and an unreadable parent
    raises nothing at all, because ``resolve()`` defaults to
    ``strict=False`` and does not stat.  The guard was written for a
    failure it could not catch and justified by one that does not
    happen.
    """
    from pathlib import Path as _P
    try:
        return str(_P(source).resolve())
    except (OSError, RuntimeError):
        return str(source)


@dataclass(frozen=True)
class ParseResult:
    """Base envelope for every parse output.

    Concrete subclasses set ``result_kind`` to a fixed string for
    consumer type-narrowing (``match result.result_kind:``).
    """
    schema_version: int
    parsed_at:      str               # ISO-8601 UTC string
    parser_name:    str               # name of the parser class that produced this
    source:         str               # path str OR "<text>" for TextParsers
    result_kind:    str = "abstract"  # discriminator; subclasses override

    @staticmethod
    def envelope(parser_name: str, source=None) -> Dict[str, Any]:
        """The four fields every result carries, spread by each builder.

        ``source`` is resolved to an absolute path; ``None`` means a
        ``TextParser``, which has no file and says ``"<text>"``.

        **One home, because there were five.**  Each sub-package's
        ``_helpers.py`` filled these four by hand and carried its own
        copy of the timestamp helper -- four identical ``_iso_z``
        definitions, and `parse/instruments/` added a fifth on
        2026-09-04 without noticing the other four.  They agreed, which
        is the only reason nothing had broken: four copies of a format
        string is four chances for one of them to drift, and the version
        in `dirs/job.py` already takes a different argument.
        """
        from datetime import datetime, timezone
        return {
            "schema_version": SCHEMA_VERSION,
            "parsed_at": datetime.now(timezone.utc)
                         .isoformat(timespec="milliseconds")
                         .replace("+00:00", "Z"),
            "parser_name": parser_name,
            "source": "<text>" if source is None else _source_str(source),
        }


# --------------------------------------------------------------------- #
#  Concrete result types                                                #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrajectoryResult(ParseResult):
    """Per-step physics from an engine .out / .log.

    Mirrors the legacy ``Trajectory`` dataclass in ``frame.py``
    so the migration in Phase C is a thin wrapper.
    """
    frames:        List[Frame] = field(default_factory=list)
    lattice:       Optional[np.ndarray] = None
    source_format: str = "unknown"
    #: HOW THE RUN ENDED -- `model/parse.md` § 2b, P-S1.  A fact about
    #: the process: "running"|"ended"|"stopped"|"out_of_memory"|"unknown".
    run_state:     str = "unknown"
    #: P-S2: whether the SCF met its criterion.  A REPORTED FACT, never an
    #: input to `run_state` -- not converging is normal and often
    #: deliberate (a capped benchmark, a relaxation step mid-flight).
    scf_converged: Optional[bool] = None
    error_message: Optional[str] = None
    runtime_info:  Dict[str, Any] = field(default_factory=dict)
    parse_warnings: List[ParseWarning] = field(default_factory=list)
    result_kind:   str = "trajectory"


@dataclass(frozen=True)
class StructureResult(ParseResult):
    """Geometry from .XV / .STRUCT_OUT / .xyz / .fdf coords block /
    PySCF final geometry.

    Carries ``cell`` separately because :class:`Structure` is
    geometry-only by historical design.  Phase E (migrating
    siesta_struct) populates it from the file's lattice block.
    """
    structure:      Optional[Structure] = None
    cell:           Optional[np.ndarray] = None
    source_format:  str = "unknown"
    parse_warnings: List[ParseWarning] = field(default_factory=list)
    result_kind:    str = "structure"


@dataclass(frozen=True)
class SidecarResult(ParseResult):
    """Generic payload + schema tag for molbuilder JSON sidecars
    (molstruct, spectra, transport, etc.).

    ``payload`` is the validated JSON body.  ``schema`` carries
    the discriminator (e.g. ``"molstruct/v3"``) so consumers can
    type-narrow further.
    """
    payload: Dict[str, Any] = field(default_factory=dict)
    schema:  str = "unknown/v0"
    result_kind: str = "sidecar"


@dataclass(frozen=True)
class RunDirResult(ParseResult):
    """One RUN DIRECTORY -> the four questions anything asks about one.

    **ZERO OF THESE SEVEN FIELDS HAS A PRODUCTION READER (2026-09-18).**  The
    table below names one for each, and none of them reads this TYPE: they
    read the underlying functions -- `contract.engine_of`, `job.run_status`,
    `rundir.openable_in` -- directly.  `parse_dir` has no production caller,
    so no `RunDirResult` is built outside the tests.  The consumer this was
    made for is the Results file picker, which is a browser; if that surface
    is never built, these fields go with it.

    `model/parse.md` § 5.0.  **No field is added without naming its reader**
    -- that is the rule the deleted predecessor broke: it answered eleven
    fields, ten of which had no reader anywhere in the tree, and reached the
    eleventh by parsing every result file to build plots and then discarding
    them.

    ============  ==========================================  =================
    field         the question                                who reads it
    ============  ==========================================  =================
    ``engine``    which engine ran                            `/api/watch/*`'s
                                                              ``format``
    ``status``    how is it doing                             `jobset/runstatus`
    ``files``     what is here                                `jobset/summarize`
    ``active``    which file speaks for the run's STATUS       the status
                                                              combiner
    ``openable``  what a VIEWER should load                    `web/watch`
    ``attempts``  what was tried, for the refusal              `web/watch`
    ============  ==========================================  =================

    **``active`` and ``openable`` are different questions** (§ 5.1), and
    conflating them is the trap that section exists to mark.  ``active``
    considers RESULT files only -- every ``.out`` plus each molwatch log whose
    footer concludes the run -- because letting an unconcluded log vote would
    let a prep-time seed outrank a real ``.out``.  ``openable`` PREFERS an
    unconcluded molwatch log: that is exactly the run in progress somebody
    wants to watch.  A directory mid-run therefore has an ``openable`` and no
    ``active``, and that is correct in both directions.

    **``active`` is a bare FILENAME and ``openable`` is a PATH**, which is the
    same distinction showing up in the types.  ``active`` is
    ``RunStatus.active_source`` unchanged, and that value is serialized into
    the status envelope the browser reads, where a server-side absolute path
    has no business; the directory it is relative to is ``run_dir``, right
    beside it.  ``openable`` is handed to a reader that opens it.  A caller
    that wants the path composes ``run_dir`` with ``active``.
    """
    run_dir:  str = ""
    engine:   str = "unknown"
    files:    Dict[str, List[Path]] = field(default_factory=dict)
    active:   Optional[str] = None
    openable: Optional[str] = None
    attempts: List[str] = field(default_factory=list)
    status:   Dict[str, Any] = field(default_factory=dict)
    result_kind: str = "rundir"


# `ScriptResult` stood here until 2026-09-05.
#
# One field per reserved block, produced by six `TextParser` classes that
# each filled ONE of them and left the rest None.  Nothing ever read the
# type: `result_kind == "script"` was never checked anywhere, and the
# blocks are now read by `script_emit`'s `_extract_*_dict` extractors
# -- beside the emitters that write them.  `plans/plan.md` § 5d.


@dataclass(frozen=True)
class InstrumentResult(ParseResult):
    """What the WRAPPER measured about a run — `parse.md` § 5c.

    ``metrics`` is a flat dict of measured numbers, one entry per figure
    the file states.  Not a :class:`SidecarResult`: that carries a JSON
    payload plus a schema discriminator, and these are plain-text logs
    and a CSV the wrapper writes beside the deck.
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    parse_warnings: List["ParseWarning"] = field(default_factory=list)
    result_kind: str = "instrument"


# ``JobResult`` stood here until 2026-09-04 -- the directory decoder's
# eleven-field summary.  Ten fields had no reader anywhere in the tree;
# the eleventh, ``status``, is now ``parse.dirs.job.run_status`` and is
# built from the parsers' own ``run_state`` instead of from plot data
# that was thrown away.  See ``parse/dirs/__init__.py``.


# (BundleResult stood here until 2026-08-29 -- the run-dir -> next-stage
#  handoff shape retired with the bundle parser; the composite CITES a
#  finished attempt and `transport/compose.py` fuses at prep.)


def answers_a_trajectory(parser_cls) -> bool:
    """Whether this parser's own declared ``output`` is a trajectory.

    Every route that reads ``.frames`` after detection has to ask this:
    ``/api/watch/*`` and the three trajectory CLI verbs.  Both hand-rolled
    it, and the CLI's copy asked ``is_dir()`` instead -- so a single-geometry
    file the parser reads perfectly still reached trajectory code and raised
    ``AttributeError: 'StructureResult' object has no attribute 'frames'``.

    A parser declares its answer, so this asks rather than guesses.
    """
    out = getattr(parser_cls, "output", None)
    return out is not None and issubclass(out, TrajectoryResult)

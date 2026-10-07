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
from typing import Any, Dict, List, Optional

import numpy as np

# Frame and Structure live elsewhere; the result types reference them.
from molbuilder.frame import Frame
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Common envelope                                                      #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParseWarning:
    """Level-3 fail-soft warning emitted by any parser."""
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

    **Catches ``RuntimeError`` as well as ``OSError``**: measured on
    python 3.12, a symlink loop makes ``resolve()`` raise ``RuntimeError``
    ("Symlink loop"), which is not an ``OSError``.
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
    source:         str               # path str, or "<text>" with no file
    result_kind:    str = "abstract"  # discriminator; subclasses override

    @staticmethod
    def envelope(parser_name: str, source=None) -> Dict[str, Any]:
        """The four fields every result carries, spread by each builder.

        ``source`` is resolved to an absolute path; ``None`` means no
        file, and says ``"<text>"``.
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

    Mirrors the ``Trajectory`` dataclass in ``frame.py``.
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
    #: WHAT STOPPED IT, as the ending reader names it (`model/parse.md`
    #: § 2b): the first fatal line's marker, or the SCF's required
    #: non-convergence; None when nothing did or the format cannot say.
    cause:         Optional[str] = None
    runtime_info:  Dict[str, Any] = field(default_factory=dict)
    parse_warnings: List[ParseWarning] = field(default_factory=list)
    result_kind:   str = "trajectory"


@dataclass(frozen=True)
class StructureResult(ParseResult):
    """Geometry from .XV / .pdb / PySCF final geometry.

    Carries ``cell`` beside the structure, from the file's lattice block
    where it has one.
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


@dataclass(frozen=True)
class EngineParamsResult(ParseResult):
    """What the ENGINE says it used -- `parse.md` § 5d.3.

    The run record's third column: never the deck molbuilder wrote, always
    the engine's own account.  ``params`` holds one entry per key the engine
    read, under the key normalised the way fdf matches labels (case, ``.``,
    ``_`` and ``-`` ignored): ``key`` as spelled, ``value`` as stated,
    ``number`` and ``unit`` when it is a quantity, ``default`` when nobody set
    it, and ``original`` when the engine converted the spelling it was given.
    ``blocks`` holds each ``%block`` verbatim.
    """
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    blocks: Dict[str, List[str]] = field(default_factory=dict)
    parse_warnings: List["ParseWarning"] = field(default_factory=list)
    result_kind: str = "engine-params"


def answers_a_trajectory(parser_cls) -> bool:
    """Whether this parser's own declared ``output`` is a trajectory.

    Every route that reads ``.frames`` after detection has to ask this:
    ``/api/watch/*`` and the three trajectory CLI verbs.

    A parser declares its answer, so this asks rather than guesses.
    """
    out = getattr(parser_cls, "output", None)
    return out is not None and issubclass(out, TrajectoryResult)

"""ParseResult hierarchy — frozen dataclasses returned by every parser.

Per ``docs/model/parse.md`` § 3.  Every concrete
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
class ScriptResult(ParseResult):
    """The 6 reserved blocks of script-contract.md (HEADER /
    PROVENANCE / BENCH-MARKS / ATOM-METADATA / USER-CUSTOM)
    extracted from a single ``.fdf`` / ``.py`` text body.

    Each sub-block carries a presence semantics: ``None`` ==
    "block absent"; empty collection / empty string == "block
    present but empty" (which matters for the ATOM-METADATA
    emission rule in script-contract.md § 4.4).
    """
    header:        Optional[str] = None
    provenance:    Optional[Dict[str, str]] = None
    bench_marks:   Optional[Dict[str, Any]] = None
    atom_metadata: Optional[Dict[str, Any]] = None
    user_custom:   Optional[List[str]] = None
    block_schema_versions: Dict[str, int] = field(default_factory=dict)
    result_kind:   str = "script"


@dataclass(frozen=True)
class JobResult(ParseResult):
    """Directory-level decoded job — what JobMonitor + Results
    tab consume.

    Schema pinned by ``docs/execution/running-a-job.md § 4``.  Field
    semantics match the dict shape ``decode_run_dir`` returned
    pre-migration so consumer code converts via ``asdict(result)``
    when needed.
    """
    job_type:              str = "unknown"
    engine:                str = "unknown"
    system_label:          Optional[str] = None
    run_dir:               str = ""
    status:                Dict[str, Any] = field(default_factory=dict)
    progress:              Dict[str, Any] = field(default_factory=dict)
    geometry:              Dict[str, Any] = field(default_factory=dict)
    plots:                 Dict[str, Dict[str, List[List[float]]]] = field(default_factory=dict)
    source_files:          List[Dict[str, Any]] = field(default_factory=list)
    engine_input_by_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parse_warnings:        List[ParseWarning] = field(default_factory=list)
    diagnostics:           Dict[str, Any] = field(default_factory=dict)
    result_kind:           str = "job"


# (BundleResult stood here until 2026-08-29 -- the run-dir -> next-stage
#  handoff shape retired with the bundle parser; the composite CITES a
#  finished attempt and `transport/compose.py` fuses at prep.)

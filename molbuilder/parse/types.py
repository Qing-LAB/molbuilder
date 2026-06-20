"""ParseResult hierarchy — frozen dataclasses returned by every parser.

Per ``docs/protocols/parse-module.md`` § 3.  Every concrete
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
    run_state:     str = "unknown"
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

    Schema pinned by ``docs/protocols/job-decoder.md``.  Field
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


@dataclass(frozen=True)
class BundleResult(ParseResult):
    """Run-dir → next-stage handoff bundle.

    Mirrors the legacy ``RunBundle`` shape from ``script_bundle.py``
    so Phase G is a thin wrapper.  Phase G also surfaces ``cell``
    explicitly (the legacy returned cell embedded in Structure).

    ``source_engine`` + ``final_coords_from`` were added in H2 so
    :func:`molbuilder.bundle_writer.write_bundle_as_handoff` can
    write the same "bundled from <source> (<engine>/<coords-source>)"
    XYZ comment the legacy materializer wrote.  Optional so older
    BundleResult constructions (and the audit-gap public-surface
    coverage test) stay valid.
    """
    structure:    Optional[Structure] = None
    cell:         Optional[np.ndarray] = None
    regions:      Dict[str, List[int]] = field(default_factory=dict)
    frozen_atoms: List[int] = field(default_factory=list)
    notes:        List[str] = field(default_factory=list)
    block_schema_versions: Dict[str, int] = field(default_factory=dict)
    source_engine:     Optional[str] = None   # "siesta" | "pyscf"
    final_coords_from: Optional[str] = None   # "xv"/"fdf-initial"/"py-opt"/"py-initial"
    result_kind:  str = "bundle"

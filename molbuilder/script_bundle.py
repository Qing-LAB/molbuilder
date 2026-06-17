"""molbuilder.script_bundle -- workflow-handoff object.

Fuses a finished run's final structure with the labels and
annotations extracted from its originating .fdf / .py.  A bundle
materialized as ``<stem>.xyz`` + ``<stem>.molstruct.json`` lets the
next workflow stage pick the work up without dirging back to the
original design directory.

Contract: ``docs/protocols/bundle-contract.md``.

PR-A defines the dataclass + error type + API signatures.  The
final-coords readers and assembler implementations land in PR-B
(SIESTA `.XV`), PR-C (PySCF), and PR-D (materializer).  Today the
stubs raise :class:`NotImplementedError` with a pointer to the
follow-on task so callers can wire the typed seat now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# Python 3.8+ doesn't have typing.Literal until 3.8 but we target 3.10+.
from typing import Literal


class BundleError(Exception):
    """Raised by ``assemble_from_run_dir`` / ``write_bundle_as_handoff``
    on irrecoverable state.

    Bundle-contract.md § 8 enumerates the states that raise.  In
    short: missing-script, both-engines-present, atom-count
    mismatch, and (for the materializer) target-exists with
    ``overwrite=False``.
    """


@dataclass(frozen=True)
class RunBundle:
    """Portable workflow-handoff object.

    Field semantics per ``docs/protocols/bundle-contract.md § 3``.
    Frozen so callers can pass it around without defensive copies;
    construction-time validation lives in ``assemble_from_run_dir``.

    Notes on each field:

    * ``structure``: final coords + elements from the converged-run
      output (preferred) or initial coords (fallback).  Source
      recorded in ``final_coords_from``.
    * ``regions`` / ``frozen_atoms``: from the source script's
      ATOM-METADATA block.  Empty (``{}`` / ``[]``) when the script
      had no labels at generation time.
    * ``user_custom_lines``: inner lines of the USER-CUSTOM block
      verbatim (no ``#`` prefix stripping).  Empty list when the
      block was absent or empty.
    * ``provenance``: flat key/value extracted from PROVENANCE.
      Empty dict when the block was absent.
    * ``source_script``: absolute path to the .fdf / .py that fed
      extraction.
    * ``source_engine``: ``"siesta"`` for .fdf, ``"pyscf"`` for .py.
    * ``final_coords_from``: ``"xv"`` (SIESTA .XV), ``"stdout"``
      (SIESTA .out parsed), ``"fdf-initial"`` (.fdf initial coords
      fallback), ``"py-log"`` (PySCF optimized geometry source),
      ``"py-initial"`` (.py initial mol.atom fallback).
    * ``notes``: non-fatal diagnostics.  Always a (possibly empty)
      list; never ``None``.
    """
    # ``Structure`` typed via TYPE_CHECKING below to avoid a
    # heavyweight import at module-load time (molbuilder.structure
    # pulls in element tables + parsers).
    structure:         "Structure"  # type: ignore[name-defined]
    regions:           Dict[str, List[int]]
    frozen_atoms:      List[int]
    user_custom_lines: List[str]
    provenance:        Dict[str, str]
    source_script:     Path
    source_engine:     Literal["siesta", "pyscf"]
    final_coords_from: Literal["xv", "stdout", "fdf-initial",
                               "py-log", "py-initial"]
    notes:             List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
#  Assembler + materializer -- typed seats (PR-A) / impls (PR-B/C/D)    #
# --------------------------------------------------------------------- #


def assemble_from_run_dir(run_dir: Path) -> RunBundle:
    """Walk ``run_dir``, pick engine + final-coords source per the
    bundle-contract, fuse with ``ScriptSource`` from the originating
    script.  Raises :class:`BundleError` on irrecoverable state.

    See ``docs/protocols/bundle-contract.md § 4`` for source
    priority and § 8 for the error model.

    Not yet implemented -- the SIESTA .XV reader (PR-B) and PySCF
    final-geom reader (PR-C) land separately; the API signature is
    reserved here so callers can wire to ``script_bundle`` today.
    """
    raise NotImplementedError(
        "assemble_from_run_dir lands in tasks #489 (SIESTA, PR-B) + "
        "#490 (PySCF, PR-C).  Today the API surface is reserved by "
        "the bundle-contract; the typed seat exists so call sites "
        "can wire to script_bundle without churning when the "
        "implementation arrives."
    )


def write_bundle_as_handoff(bundle: RunBundle,
                            target_dir: Path,
                            *,
                            stem: str,
                            overwrite: bool = False
                            ) -> Tuple[Path, Path]:
    """Materialize ``bundle`` as ``<target_dir>/<stem>.xyz`` +
    ``<target_dir>/<stem>.molstruct.json``.

    Atomic via :func:`molstruct_json.save`.  ``overwrite=False``
    raises :class:`BundleError` when either destination file exists.

    Returns ``(xyz_path, sidecar_path)``.

    Not yet implemented -- lands in PR-D (task #491) alongside the
    L3 round-trip test.
    """
    raise NotImplementedError(
        "write_bundle_as_handoff lands in task #491 (PR-D)."
    )


__all__ = [
    "BundleError",
    "RunBundle",
    "assemble_from_run_dir",
    "write_bundle_as_handoff",
]

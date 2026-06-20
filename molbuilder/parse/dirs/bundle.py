"""Run-bundle DirParser — directory → handoff-ready BundleResult.

Phase G of parse-module.md migration: wraps the legacy
``molbuilder.script_bundle.assemble_from_run_dir`` as a typed
:class:`DirParser` returning :class:`BundleResult`.  The legacy
function stays intact because ``script_bundle.write_bundle_as_handoff``
(the materializer half of the round-trip) still consumes its
``RunBundle`` directly; absorbing both into ``parse/dirs/bundle.py``
is a Phase H scope decision.

Dispatch design — NOT auto-registered with the DirParser
registry.  Two reasons:

  1. ``JobDirParser`` already auto-claims any directory with a
     ``.fdf`` for JobMonitor decoding.  Bundle assembly is a
     DIFFERENT user intent — "extract the handoff bundle from
     this run dir" — and shouldn't fight for the dispatch slot.
  2. Bundle assembly raises ``BundleError`` on irrecoverable state
     (e.g. both ``.fdf`` and ``.py`` in the same dir).  That belongs
     at the explicit-call boundary, not in a heuristic
     ``can_parse``.

Callers use the class directly:

    from molbuilder.parse.dirs.bundle import BundleDirParser
    result = BundleDirParser.parse(run_dir)
    # result: BundleResult (structure + regions + frozen_atoms + ...)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from molbuilder.parse.base import DirParser
from molbuilder.parse.types import BundleResult
from molbuilder.script_bundle import (
    BundleError,
    assemble_from_run_dir as _legacy_assemble,
)


def _iso_z() -> str:
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


class BundleDirParser(DirParser):
    """Assemble a run directory into a portable handoff bundle.

    Returns :class:`BundleResult` carrying the final-coords
    structure + the script-contract ATOM-METADATA labels
    (regions + frozen_atoms) + the user-custom block + the
    diagnostic notes the legacy assembler emits.

    NOT auto-registered (see module docstring); call explicitly.
    """

    name   = "bundle-dir"
    label  = "molbuilder run-bundle handoff (script_bundle.assemble_from_run_dir)"
    output = BundleResult

    @classmethod
    def can_parse(cls, run_dir: Path) -> bool:
        """A directory bundleable iff it has at least one ``.fdf`` or
        ``.py`` engine script.  Mirrors the legacy assembler's
        precondition exactly; the assembler's ambiguity check (both
        engines present → BundleError) only fires in ``parse()``,
        not here.
        """
        try:
            for child in run_dir.iterdir():
                if child.is_file() and (
                    child.name.endswith(".fdf")
                    or child.name.endswith(".py")
                ):
                    return True
        except OSError:
            return False
        return False

    @classmethod
    def parse(cls, run_dir: Path) -> BundleResult:
        """Run the legacy assembler + wrap the RunBundle in a typed
        :class:`BundleResult`.  Propagates :exc:`BundleError`
        upward unchanged — bundle assembly errors are caller
        concerns (ambiguous engines, atom-count mismatch, missing
        .XV that the strict path required).
        """
        bundle = _legacy_assemble(Path(run_dir))   # may raise BundleError
        # Bundle source path is resolved absolute, matching the
        # envelope convention used by engines/sidecars/coords.
        try:
            source_str = str(Path(run_dir).resolve())
        except OSError:
            source_str = str(run_dir)
        # Cell — the legacy Structure dataclass is geometry-only.
        # Phase E's StructureResult.cell solved this at the leaf
        # parser layer, but the legacy RunBundle still doesn't
        # carry cell.  When (Phase H) the assembler is rewritten
        # to use coords/SiestaXVFileParser, cell will flow through
        # naturally.  For now BundleResult.cell stays None.
        cell: Optional[np.ndarray] = None
        # block_schema_versions surfaces just the script-contract
        # schema versions that the assembler observed in its
        # source script (atom-metadata is the only one the legacy
        # RunBundle persists via its `schema_version` attr on the
        # ScriptSource it pulls in; provenance has no version).
        block_schema_versions = {}
        # The legacy ScriptSource carries schema_version on the
        # umbrella; we mirror that into BundleResult for downstream
        # round-trip checks.
        sv = getattr(bundle, "schema_version", None)
        if isinstance(sv, int):
            block_schema_versions["atom-metadata"] = sv
        return BundleResult(
            schema_version=1,
            parsed_at=_iso_z(),
            parser_name=cls.name,
            source=source_str,
            structure=bundle.structure,
            cell=cell,
            regions=dict(bundle.regions or {}),
            frozen_atoms=list(bundle.frozen_atoms or []),
            notes=list(bundle.notes or []),
            block_schema_versions=block_schema_versions,
        )


__all__ = ["BundleDirParser", "BundleError"]

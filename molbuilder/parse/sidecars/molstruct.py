"""``.molstruct.json`` sidecar FileParser.

H1 of parse-module.md migration (was Phase D wrapper around
``molbuilder.parsers.molstruct_json.load``): absorbed the read-side
``load`` + supporting validators directly so this module no longer
imports from ``molbuilder.parsers``.  The legacy module stays in
place until H4; consumers (web blueprints' /api/selection/save and
similar) still use it for the write-side helpers (``save``,
``with_lock``, ``sidecar_path_for``, ``to_dict``,
``apply_to_structure``, ``sha256_of_file``) which H2 rehomes.

The sidecar carries per-atom region + frozen-atom metadata that
rides next to a structure file (``<stem>.molstruct.json``).
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
# Canonical home for both ``SCHEMA_VERSION`` (the on-disk version
# constant) and ``MolstructJsonError`` is the write-side module so
# read and write surfaces stay in lock-step.
from molbuilder.sidecars.molstruct import (
    SCHEMA_VERSION,
    MolstructJsonError,
    normalise_selection_rules,
    structure_fields_via_dataclass,
)

from ._helpers import build_sidecar_result


# Readable schema versions.  v1/v2 sidecars (older ``fixed_atoms`` key)
# must be re-exported from /modify.  v3 has no ``annotations`` (a v3
# reader/consumer sees only regions + frozen); v4 adds the extensible
# annotation channels (atom-annotations.md § 3) additively -- v3 files
# still load (annotations absent -> empty).  v5 drops the ``kgrid`` field
# (k-grid moved off the geometry onto SiestaConfig / TransportConfig);
# a ``kgrid`` key in an older v3/v4 file loads fine -- it's ignored.
# v7 moves the reserved ``frozen`` label into ``regions`` with every other
# label and stops writing a top-level ``frozen_atoms`` key; v3-v6 files still
# load, because ``apply_metadata_dict`` folds that key into the label store.
_READABLE_SCHEMA_VERSIONS = (3, 4, 5, 6, 7)


def _normalised_dict(
    *,
    n_atoms_total: int,
    structure_hash: str,
    regions: Optional[Dict[str, List[int]]] = None,
    frozen_atoms: Optional[List[int]] = None,
    selection_rules: Optional[Dict[str, Any]] = None,
    cell: Optional[Any] = None,
    cell_origin: Optional[Any] = None,
    pbc: Optional[Any] = None,
    axis_kind: Optional[Any] = None,
    vacuum: Optional[Any] = None,
    annotations: Optional[Dict[str, Any]] = None,
    created_by: str = "molbuilder",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-validate + canonicalise sidecar fields.  Private read-side
    helper; mirrors the legacy ``to_dict`` validation logic so an on-
    disk sidecar that snuck past with hand-edited indices fails loudly
    at load time (with the file path in the error message) rather than
    at engine-load time.

    Region membership is NOT mutually exclusive (an atom may carry
    multiple labels); engines that need a disjoint partition validate
    that separately at engine-load time.
    """
    if not isinstance(n_atoms_total, int) or n_atoms_total < 0:
        raise MolstructJsonError(
            f"n_atoms_total must be a non-negative int; got "
            f"{n_atoms_total!r}"
        )
    if not isinstance(structure_hash, str) or len(structure_hash) < 16:
        raise MolstructJsonError(
            f"structure_hash must be a hex string (got "
            f"{structure_hash!r})"
        )

    # The Structure FIELDS -> validated + canonicalised through the ONE dataclass
    # authority, SHARED byte-for-byte with the write validator (to_dict).  This
    # is what closes the read/write drift that silently dropped cell_origin: a
    # field cannot exist on one side and not the other, because there is only one
    # side.  ``selection_rules`` (a sidecar-only pass-through) shares its one
    # validator too.
    fields = structure_fields_via_dataclass(n_atoms_total, {
        "regions":      regions,
        "frozen_atoms": frozen_atoms,
        "cell":         cell,
        "cell_origin":  cell_origin,
        "pbc":          pbc,
        "axis_kind":    axis_kind,
        "vacuum":       vacuum,
        "annotations":  annotations,
    })
    normed_rules = normalise_selection_rules(
        selection_rules, set(fields["regions"]))

    return {
        "schema_version":  SCHEMA_VERSION,
        "n_atoms_total":   n_atoms_total,
        "structure_hash":  structure_hash,
        # SPREAD, not re-listed -- the same rule the write side follows, and for
        # the same reason: a field added to (or removed from) the dataclass must
        # ride onto both sides with no edit here.  This block used to name each
        # field, which is why it had to be touched at all when the reserved
        # label stopped being one.
        **fields,
        "selection_rules": normed_rules,
        "created_by":      str(created_by),
        "created_at":      created_at,
    }


def load_text(text: str, *, source: str = "<sidecar>") -> Dict[str, Any]:
    """Parse + validate a sidecar JSON **string** (the same strict checks
    as :func:`_load`, minus the file read).  Returns the normalised dict.

    Used by ``/api/build/load``: the browser reads the ``.molstruct.json``
    file's bytes through the concealed projects file package
    (``projects.readFile``) and hands the CONTENT to the parse seam, which
    validates from a string rather than re-reading a path.  ``source`` names
    the origin for error messages (a path when called from :func:`_load`, a
    placeholder for in-body content).

    Validation is strict: missing required fields, wrong types, out-of-range
    indices, and unknown schema version all raise :class:`MolstructJsonError`.
    Region overlap (one atom in two regions) is INTENTIONALLY permitted --
    the model is multi-label freeform tagging.  ``regions`` / ``frozen_atoms``
    are permissive (missing == empty); everything else is required.
    """
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError as exc:
        raise MolstructJsonError(
            f"sidecar {source} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise MolstructJsonError(
            f"sidecar {source}: top-level value must be an "
            f"object; got {type(data).__name__}"
        )

    sv = data.get("schema_version")
    if sv not in _READABLE_SCHEMA_VERSIONS:
        raise MolstructJsonError(
            f"sidecar {source}: schema_version is {sv!r}; "
            f"this molbuilder build reads versions "
            f"{list(_READABLE_SCHEMA_VERSIONS)!r}"
        )

    for key in ("n_atoms_total", "structure_hash"):
        if key not in data:
            raise MolstructJsonError(
                f"sidecar {source} missing required field {key!r}"
            )

    n = data["n_atoms_total"]
    if not isinstance(n, int) or n < 0:
        raise MolstructJsonError(
            f"sidecar {source}: n_atoms_total must be a non-"
            f"negative int; got {n!r}"
        )

    sh = data["structure_hash"]
    if not isinstance(sh, str) or len(sh) < 16:
        raise MolstructJsonError(
            f"sidecar {source}: structure_hash must be a hex "
            f"string of >= 16 chars; got {sh!r}"
        )

    # Re-validate regions + frozen_atoms via _normalised_dict.  This
    # catches malformed user-edited JSON BEFORE any consumer tries to
    # apply the data to a Structure (where the same checks would run
    # but with a less specific error message that doesn't mention
    # which file is at fault).
    try:
        return _normalised_dict(
            n_atoms_total   = n,
            structure_hash  = sh,
            regions         = data.get("regions"),
            frozen_atoms    = data.get("frozen_atoms"),
            selection_rules = data.get("selection_rules"),
            cell            = data.get("cell"),
            cell_origin     = data.get("cell_origin"),
            pbc             = data.get("pbc"),
            axis_kind       = data.get("axis_kind"),
            vacuum          = data.get("vacuum"),
            annotations     = data.get("annotations"),
            created_by      = data.get("created_by", "unknown"),
            created_at      = data.get("created_at"),
        )
    except MolstructJsonError as exc:
        raise MolstructJsonError(f"sidecar {source}: {exc}") from exc


def _load(sidecar_path: Union[str, Path]) -> Dict[str, Any]:
    """Read + validate a sidecar JSON from a PATH.  Reads the bytes, then
    delegates to :func:`load_text` for the parse + strict validation."""
    sidecar_path = Path(sidecar_path)
    try:
        # ``encoding="utf-8-sig"`` accepts an optional BOM (some
        # Windows editors insert one) and decodes as UTF-8.
        text = sidecar_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MolstructJsonError(
            f"failed to read sidecar {sidecar_path}: {exc}"
        ) from exc
    return load_text(text, source=str(sidecar_path))


class MolstructSidecarFileParser(FileParser):
    """Parse a ``<stem>.molstruct.json`` sidecar — the per-atom
    region + frozen_atom metadata that rides next to a structure
    file.  Returns a :class:`SidecarResult` whose ``payload`` is
    the raw schema-validated dict and ``schema`` is the version-
    qualified discriminator (``"molstruct/v3"``)."""

    name   = "molstruct-json"
    label  = "molbuilder .molstruct.json sidecar"
    hint   = "files ending in .molstruct.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return path.name.endswith(".molstruct.json") and path.is_file()

    @classmethod
    def parse(cls, path: Path) -> SidecarResult:
        payload = _load(path)
        # _load() has already validated schema_version against the
        # readable set; read it strictly rather than defaulting.
        sv = payload["schema_version"]
        return build_sidecar_result(
            payload=payload,
            schema=f"molstruct/v{sv}",
            parser_name=cls.name,
            source=path,
        )


__all__ = ["MolstructJsonError", "MolstructSidecarFileParser"]

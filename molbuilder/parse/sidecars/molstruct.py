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
)

from ._helpers import build_sidecar_result


# Only the current schema version loads.  v1/v2 sidecars (which used
# the older ``fixed_atoms`` key name) must be re-exported from /modify
# to produce a v3 file.
_READABLE_SCHEMA_VERSIONS = (3,)


def _normalised_dict(
    *,
    n_atoms_total: int,
    structure_hash: str,
    regions: Optional[Dict[str, List[int]]] = None,
    frozen_atoms: Optional[List[int]] = None,
    selection_rules: Optional[Dict[str, Any]] = None,
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

    normed_regions: Dict[str, List[int]] = {}
    if regions:
        for name, idxs in regions.items():
            if not isinstance(name, str) or not name:
                raise MolstructJsonError(
                    f"region label must be non-empty string; got "
                    f"{name!r}"
                )
            unique = sorted({int(i) for i in idxs})
            for idx in unique:
                if not 0 <= idx < n_atoms_total:
                    raise MolstructJsonError(
                        f"regions[{name!r}]: atom index {idx} out of "
                        f"range [0, {n_atoms_total})"
                    )
            normed_regions[name] = unique

    normed_frozen: List[int] = []
    if frozen_atoms:
        unique = sorted({int(i) for i in frozen_atoms})
        for idx in unique:
            if not 0 <= idx < n_atoms_total:
                raise MolstructJsonError(
                    f"frozen_atoms: atom index {idx} out of range "
                    f"[0, {n_atoms_total})"
                )
        normed_frozen = unique

    normed_rules: Dict[str, Any] = {}
    if selection_rules:
        # Import locally to keep the import graph narrow -- selection.py
        # is a leaf utility module and we only need it here if the
        # caller is using selection rules at all.
        from molbuilder.selection import from_json as _rule_from_json
        from molbuilder.selection import to_json as _rule_to_json
        from molbuilder.selection import SelectionError
        valid_targets = set(normed_regions) | {"frozen_atoms"}
        for target, rule_payload in selection_rules.items():
            if not isinstance(target, str) or not target:
                raise MolstructJsonError(
                    f"selection_rules: target label must be non-empty "
                    f"string; got {target!r}"
                )
            if target not in valid_targets:
                raise MolstructJsonError(
                    f"selection_rules: target {target!r} doesn't match "
                    f"any region or 'frozen_atoms' (known: "
                    f"{sorted(valid_targets)!r})"
                )
            try:
                rule = _rule_from_json(rule_payload)
            except SelectionError as exc:
                raise MolstructJsonError(
                    f"selection_rules[{target!r}]: invalid rule: {exc}"
                ) from exc
            # Re-serialise so the stored form is normalised.
            normed_rules[target] = _rule_to_json(rule)

    return {
        "schema_version":  SCHEMA_VERSION,
        "n_atoms_total":   n_atoms_total,
        "structure_hash":  structure_hash,
        "regions":         normed_regions,
        "frozen_atoms":    normed_frozen,
        "selection_rules": normed_rules,
        "created_by":      str(created_by),
        "created_at":      created_at,
    }


def _load(sidecar_path: Union[str, Path]) -> Dict[str, Any]:
    """Read + validate a sidecar JSON.  Returns the parsed dict.

    Validation is strict: missing required fields, wrong types, out-
    of-range indices, and unknown schema version all raise
    :class:`MolstructJsonError` with a message that points at the
    problem.

    Region overlap (one atom in two regions) is INTENTIONALLY
    permitted -- the data model is multi-label freeform tagging.
    The on-disk schema is permissive about ``regions`` and
    ``frozen_atoms`` (missing == empty); everything else is required.
    """
    sidecar_path = Path(sidecar_path)
    try:
        # ``encoding="utf-8-sig"`` accepts an optional BOM (some
        # Windows editors insert one) and decodes as UTF-8.
        text = sidecar_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MolstructJsonError(
            f"failed to read sidecar {sidecar_path}: {exc}"
        ) from exc
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError as exc:
        raise MolstructJsonError(
            f"sidecar {sidecar_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: top-level value must be an "
            f"object; got {type(data).__name__}"
        )

    sv = data.get("schema_version")
    if sv not in _READABLE_SCHEMA_VERSIONS:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: schema_version is {sv!r}; "
            f"this molbuilder build reads versions "
            f"{list(_READABLE_SCHEMA_VERSIONS)!r}"
        )

    for key in ("n_atoms_total", "structure_hash"):
        if key not in data:
            raise MolstructJsonError(
                f"sidecar {sidecar_path} missing required field "
                f"{key!r}"
            )

    n = data["n_atoms_total"]
    if not isinstance(n, int) or n < 0:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: n_atoms_total must be a non-"
            f"negative int; got {n!r}"
        )

    sh = data["structure_hash"]
    if not isinstance(sh, str) or len(sh) < 16:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: structure_hash must be a hex "
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
            created_by      = data.get("created_by", "unknown"),
            created_at      = data.get("created_at"),
        )
    except MolstructJsonError as exc:
        raise MolstructJsonError(
            f"sidecar {sidecar_path}: {exc}"
        ) from exc


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
        sv = payload.get("schema_version", SCHEMA_VERSION)
        return build_sidecar_result(
            payload=payload,
            schema=f"molstruct/v{sv}",
            parser_name=cls.name,
            source=path,
        )


__all__ = ["MolstructJsonError", "MolstructSidecarFileParser"]

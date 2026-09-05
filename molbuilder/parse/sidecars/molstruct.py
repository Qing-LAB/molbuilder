"""``.molstruct.json`` sidecar FileParser.

The READ side. Absorbed from the legacy
``molbuilder.parsers.molstruct_json.load`` (deleted 2026-06-21) with its
validators; the write side -- ``save``, ``with_lock``,
``sidecar_path_for``, ``to_dict``, ``apply_to_structure``,
``sha256_of_file`` -- is :mod:`molbuilder.sidecars.molstruct`
(provenance: `docs/archive/old_docs/protocols/parse-module.md` § 8).

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
    READABLE_VERSIONS,
    SCHEMA_VERSION,
    ENVELOPE_KEYS,
    MolstructJsonError,
    normalise_selection_rules,
    structure_fields_via_dataclass,
)

from ._helpers import build_sidecar_result


# The readable-version gate is `sidecars.molstruct.READABLE_VERSIONS`,
# imported above and used by `load_text` below.
#
# A private `_READABLE_SCHEMA_VERSIONS = (SCHEMA_VERSION,)` stood here
# until 2026-09-05 with eighteen lines arguing for a STRICT one-version
# gate -- while the gate that runs accepts three.  It was read by
# nothing, so the argument was decoration and the disagreement was
# invisible.  The reasoning it carried was not wasted and now lives
# with the live constant, where a person changing the set will meet it.


def _normalised_dict(
    *,
    n_atoms_total: int,
    structure_hash: str,
    regions: Optional[Dict[str, List[int]]] = None,
    selection_rules: Optional[Dict[str, Any]] = None,
    cell: Optional[Any] = None,
    cell_origin: Optional[Any] = None,
    pbc: Optional[Any] = None,
    axis_kind: Optional[Any] = None,
    vacuum: Optional[Any] = None,
    annotations: Optional[Dict[str, Any]] = None,
    identity: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None,
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
        "cell":         cell,
        "cell_origin":  cell_origin,
        "pbc":          pbc,
        "axis_kind":    axis_kind,
        "vacuum":       vacuum,
        "annotations":  annotations,
    })
    normed_rules = normalise_selection_rules(
        selection_rules, set(fields["regions"]))

    # The identity columns (schema 8) -- validated through the SAME dataclass
    # authority as everything else: a scratch Structure carrying them re-runs
    # __post_init__'s own length/type checks, so a hand-edited sidecar whose
    # residue_ids no longer match the atom count fails HERE, with the file
    # named, not at engine-load time.
    identity = dict(identity or {})
    if identity:
        from molbuilder.structure import Structure as _Structure
        try:
            _scratch = _Structure(
                elements=["X"] * n_atoms_total,
                positions=[[0.0, 0.0, 0.0]] * n_atoms_total,
                title=identity.get("title", ""),
                atom_names=identity.get("atom_names"),
                residue_ids=identity.get("residue_ids"),
                residue_names=identity.get("residue_names"),
                chain_ids=identity.get("chain_ids"),
            )
        except (ValueError, TypeError) as exc:
            raise MolstructJsonError(str(exc)) from exc

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
        **identity,
        # The `info` block (schema 9): carried whole when the file holds
        # one -- the store is open by design, so nothing here enumerates
        # its keys (archive/2026-09-01-structure-info-plan.md).
        **({"info": dict(info)} if isinstance(info, dict) and info else {}),
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
    if sv not in READABLE_VERSIONS:
        # REFUSED, NOT READ PARTIALLY.  An older sidecar does not store the same
        # facts in the same places, so reading it here would not recover them --
        # it would return a payload that LOOKS complete and quietly is not.  The
        # v3 case is why this is strict: its frozen atoms live under a top-level
        # key this reader does not name, so the file loaded, the atoms did not,
        # and the omission first became visible as a missing
        # ``Geometry.Constraints`` block in a generated SIESTA input.
        raise MolstructJsonError(
            f"sidecar {source}: schema_version is {sv!r}, but this molbuilder "
            f"build reads versions {sorted(READABLE_VERSIONS)} only (v8 "
            f"added only the OPTIONAL identity columns, so a v7 file reads "
            f"whole; older versions do not).\n"
            f"Older sidecars are NOT read: they store the same facts in "
            f"different places (before v{SCHEMA_VERSION}, frozen atoms sat in a "
            f"top-level 'frozen_atoms' key rather than in 'regions'), so reading "
            f"one would silently drop what it cannot map.\n"
            f"Re-export it: open the structure in Modify and save it, which "
            f"writes a v{SCHEMA_VERSION} pair. Check the labels afterwards -- "
            f"anything the old format kept elsewhere has to be applied again."
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

    # A KEY NOBODY READS IS METADATA THE WRITER THINKS IT SAVED.
    #
    # ``apply_to_structure`` already refuses stray keys with exactly that
    # reasoning -- but it never got the chance, because this function reads the
    # keys it NAMES and drops the rest on the floor one layer earlier.  That is
    # the hole the v3 frozen-atom loss went through: the guard was real, correct,
    # and downstream of the leak.  It is checked HERE now, where the payload is
    # still whole.
    from molbuilder.structure import IDENTITY_FIELDS, METADATA_FIELDS
    # `info` (schema 9): the free-form NON-structural store -- known by
    # NAME here (the block is open by design, so its keys are not
    # enumerated; archive/2026-09-01-structure-info-plan.md).
    known = (set(METADATA_FIELDS) | set(IDENTITY_FIELDS)
             | set(ENVELOPE_KEYS) | {"info"})
    stray = sorted(k for k in data if k not in known)
    if stray:
        raise MolstructJsonError(
            f"sidecar {source} carries {stray!r}, which this version does not "
            f"read. Refused rather than ignored -- a key nobody reads is "
            f"metadata the writer thinks it saved. Known keys: "
            f"{sorted(known)!r}"
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
            selection_rules = data.get("selection_rules"),
            cell            = data.get("cell"),
            cell_origin     = data.get("cell_origin"),
            pbc             = data.get("pbc"),
            axis_kind       = data.get("axis_kind"),
            vacuum          = data.get("vacuum"),
            annotations     = data.get("annotations"),
            identity        = {k: data[k] for k in IDENTITY_FIELDS
                               if k in data},
            info            = data.get("info"),
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

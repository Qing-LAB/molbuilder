"""``.molstruct.json`` sidecar FileParser.

The READ side; the write side -- ``save``, ``with_lock``,
``sidecar_path_for``, ``to_dict``, ``apply_to_structure``,
``geometry_hash`` -- is :mod:`molbuilder.sidecars.molstruct`.

The sidecar carries per-atom region + frozen-atom metadata that
rides next to a structure file (``<stem>.molstruct.json``).
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
# Canonical home for both ``SCHEMA_VERSION`` (the on-disk version
# constant) and ``MolstructJsonError`` is the write-side module so
# read and write surfaces stay in lock-step.
from molbuilder.sidecars.molstruct import (
    READABLE_VERSIONS,
    SCHEMA_VERSION,
    ENVELOPE_KEYS,
    RETIRED_ENVELOPE_KEYS,
    MolstructJsonError,
    is_geometry_hash,
    is_sidecar,
    normalise_selection_rules,
    structure_fields_via_dataclass,
)

from ._helpers import build_sidecar_result


# The readable-version gate is `sidecars.molstruct.READABLE_VERSIONS`,
# imported above and used by `load_text` below.


def _normalised_dict(
    *,
    n_atoms_total: int,
    n_frames_total: Optional[int],
    geometry_hash: Optional[str],
    fields: Dict[str, Any],
    selection_rules: Optional[Dict[str, Any]] = None,
    identity: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None,
    created_by: str = "molbuilder",
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-validate + canonicalise sidecar fields.  Private read-side
    helper; mirrors the write side's ``to_dict`` validation so an on-
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
    if geometry_hash is not None and not is_geometry_hash(geometry_hash):
        raise MolstructJsonError(
            f"geometry_hash must be a sha256 hex string (got "
            f"{geometry_hash!r})")

    if n_frames_total is not None and (
            isinstance(n_frames_total, bool)
            or not isinstance(n_frames_total, int) or n_frames_total < 1):
        raise MolstructJsonError(
            f"n_frames_total must be a positive int; got {n_frames_total!r}")

    # The Structure FIELDS -> validated + canonicalised through the ONE dataclass
    # authority, SHARED byte-for-byte with the write validator (to_dict).  This
    # is what closes the read/write drift: a field cannot exist on one side and
    # not the other, because there is only one side.  ``selection_rules`` (a
    # sidecar-only pass-through) shares its one validator too.  A file that
    # states no frame count (v7-v10) holds nothing indexed by frame, so its
    # fields are checked against one frame.
    fields = structure_fields_via_dataclass(n_atoms_total, fields,
                                            n_frames_total or 1)
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
            _Structure(                  # built for its checks alone
                elements=["X"] * n_atoms_total,
                positions=[[0.0, 0.0, 0.0]] * n_atoms_total,
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
        **({"n_frames_total": n_frames_total}
           if n_frames_total is not None else {}),
        # THE PIN, when the file states one (v13): a v7-v12 file has none,
        # and its pair opens unchecked (`structure-molstruct.md` § 3).
        **({"geometry_hash": geometry_hash}
           if geometry_hash is not None else {}),
        # SPREAD, not re-listed -- the same rule the write side follows, and for
        # the same reason: a field added to (or removed from) the dataclass must
        # ride onto both sides with no edit here.
        **fields,
        **identity,
        # The `info` block (schema 9): carried whole when the file holds
        # one -- the store is open by design, so nothing here enumerates
        # its keys.
        **({"info": dict(info)} if isinstance(info, dict) and info else {}),
        "selection_rules": normed_rules,
        "created_by":      str(created_by),
        "created_at":      created_at,
    }


def load_text(text: str, *, source: str = "<sidecar>",
              retired_out: "Dict[str, Any] | None" = None) -> Dict[str, Any]:
    """Parse + validate a sidecar JSON **string** (the same strict checks
    as :func:`_load`, minus the file read).  Returns the normalised dict.
    ``source`` names the origin for error messages (a path when called from
    :func:`_load`, a placeholder for in-body content).

    ``retired_out``, when a dict is passed, receives each retired key the file
    carried WITH A VALUE (``structure.RETIRED_METADATA_KEYS``): read and
    ignored, never applied -- and a door that opens the file for a person can
    say so (the load door names a dropped ``cell_origin``, plan § 5q D14).

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
        # it would return a payload that LOOKS complete and quietly is not.
        raise MolstructJsonError(
            f"sidecar {source}: schema_version is {sv!r}, but this molbuilder "
            f"build reads versions {sorted(READABLE_VERSIONS)} only (v8 "
            f"added only the OPTIONAL identity columns, so a v7 file reads "
            f"whole; older versions do not).\n"
            f"Older sidecars are NOT read: they store the same facts in "
            f"different places (before v7, frozen atoms sat in a "
            f"top-level 'frozen_atoms' key rather than in 'regions'), so reading "
            f"one would silently drop what it cannot map.\n"
            f"Re-export it: open the structure in Modify and save it, which "
            f"writes a v{SCHEMA_VERSION} pair. Check the labels afterwards -- "
            f"anything the old format kept elsewhere has to be applied again."
        )

    # `n_frames_total` is v11's, `geometry_hash` v13's: every writer of
    # that version states it (§ 1, § 3).
    required = (("n_atoms_total",)
                + (("n_frames_total",) if sv >= 11 else ())
                + (("geometry_hash",) if sv >= 13 else ()))
    for key in required:
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


    # A KEY NOBODY READS IS METADATA THE WRITER THINKS IT SAVED.  Checked
    # HERE, where the payload is still whole: this function reads the keys it
    # NAMES, so a stray one would be dropped before ``apply_to_structure``
    # could refuse it.
    from molbuilder.structure import (IDENTITY_FIELDS, METADATA_FIELDS,
                                      RETIRED_METADATA_KEYS,
                                      RETIRED_IDENTITY_KEYS)
    # `info` (schema 9): the free-form NON-structural store -- known by
    # NAME here (the block is open by design, so its keys are not
    # enumerated).
    # AND THE RETIRED ONES, which are accepted and ignored rather than
    # refused -- a key this project used to write is not a key nobody reads,
    # and the files carrying it are the user's (`structure.RETIRED_METADATA_KEYS`).
    known = (set(METADATA_FIELDS) | set(IDENTITY_FIELDS)
             | set(ENVELOPE_KEYS) | {"info"} | set(RETIRED_METADATA_KEYS)
             | set(RETIRED_IDENTITY_KEYS) | set(RETIRED_ENVELOPE_KEYS))
    stray = sorted(k for k in data if k not in known)
    if stray:
        raise MolstructJsonError(
            f"sidecar {source} carries {stray!r}, which this version does not "
            f"read. Refused rather than ignored -- a key nobody reads is "
            f"metadata the writer thinks it saved. Known keys: "
            f"{sorted(known)!r}"
        )
    if retired_out is not None:
        retired_out.update({k: data[k] for k in RETIRED_METADATA_KEYS
                            if data.get(k) is not None})

    # Re-validate the fields via _normalised_dict.  This
    # catches malformed user-edited JSON BEFORE any consumer tries to
    # apply the data to a Structure (where the same checks would run
    # but with a less specific error message that doesn't mention
    # which file is at fault).
    try:
        return _normalised_dict(
            n_atoms_total   = n,
            n_frames_total  = data.get("n_frames_total"),
            geometry_hash   = data.get("geometry_hash"),
            # SPREAD, never listed: every field the Structure names, read by
            # the same tuple the writer spreads (`model/structure.md` § 2.2).
            fields          = {k: data.get(k) for k in METADATA_FIELDS},
            selection_rules = data.get("selection_rules"),
            identity        = {k: data[k] for k in IDENTITY_FIELDS
                               if k in data},
            info            = data.get("info"),
            created_by      = data.get("created_by", "unknown"),
            created_at      = data.get("created_at"),
        )
    except MolstructJsonError as exc:
        raise MolstructJsonError(f"sidecar {source}: {exc}") from exc


def _load(sidecar_path: Union[str, Path], *,
          retired_out: "Dict[str, Any] | None" = None) -> Dict[str, Any]:
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
    return load_text(text, source=str(sidecar_path), retired_out=retired_out)


class MolstructSidecarFileParser(FileParser):
    """Parse a ``<stem>.molstruct.json`` sidecar — the per-atom
    region + frozen_atom metadata that rides next to a structure
    file.  Returns a :class:`SidecarResult` whose ``payload`` is
    the raw schema-validated dict and ``schema`` is the version-
    qualified discriminator (``"molstruct/v<N>"``)."""

    name   = "molstruct-json"
    label  = "molbuilder .molstruct.json sidecar"
    hint   = "files ending in .molstruct.json"
    output = SidecarResult

    @classmethod
    def can_parse(cls, path: Path) -> bool:
        return is_sidecar(path) and path.is_file()

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

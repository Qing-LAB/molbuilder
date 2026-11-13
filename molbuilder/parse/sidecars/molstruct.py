"""``.molstruct.json`` sidecar FileParser, and the sidecar's text reader.

The READ side: :func:`load_text`, the one reader of a sidecar's text,
which the read door -- ``load``, beside the write side (``dumps``,
``sidecar_path_for``, ``to_dict``, ``apply_to_structure``,
``geometry_hash``) in :mod:`molbuilder.sidecars.molstruct` -- asks with the
file's text, and the registry's FileParser, which asks that door.

The sidecar carries per-atom region + frozen-atom metadata that
rides next to a structure file (``<stem>.molstruct.json``).
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Dict

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
# ``SCHEMA_VERSION`` (the on-disk version constant) and the payload's one
# builder (``to_dict``) live in the write-side module, so read and write
# surfaces stay in lock-step; the pair's refusals are `structure_pair`'s.
from molbuilder.sidecars.molstruct import (
    READABLE_VERSIONS,
    SCHEMA_VERSION,
    is_sidecar,
    load,
    to_dict,
    unread_keys,
)
from molbuilder.atom_permutation import MAP_KEY
from molbuilder.structure_pair import MolstructJsonError

from ._helpers import build_sidecar_result


# The readable-version gate is `sidecars.molstruct.READABLE_VERSIONS`,
# imported above and used by `load_text` below.


def load_text(text: str, *, source: str = "<sidecar>",
              retired_out: "Dict[str, Any] | None" = None,
              stated_out: "set | None" = None) -> Dict[str, Any]:
    """Parse + validate a sidecar JSON **string** -- the text the read
    door (``sidecars.molstruct.load``) read from the file.  Returns the
    normalised dict, built by the payload's one builder
    (``sidecars.molstruct.to_dict``) with the version the file STATES,
    never this build's (`model/overview.md` § 2.2b, F4), so an on-disk
    sidecar that snuck past with hand-edited indices fails loudly here, with
    the file named, rather than at engine-load time.  ``source`` names the
    origin for error messages (the path the read door read).

    ``retired_out``, when a dict is passed, receives each retired key the file
    carried WITH A VALUE (``structure.RETIRED_METADATA_KEYS``): read and
    ignored, never applied -- and a door that opens the file for a person can
    say so (the load door names a dropped ``cell_origin``, plan § 5q D14).

    ``stated_out``, when a set is passed, receives every metadata field the
    file STATES, a null one among them -- what a reader asks to know whether
    a file was written with a field at all, by its format, never its version
    (`model/overview.md` § 2.2a: a junction whose sidecar states no
    ``atom_order``).

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

    # A KEY NOBODY READS IS METADATA THE WRITER THINKS IT SAVED.  Checked
    # HERE, where the payload is still whole: this function reads the keys it
    # NAMES, so a stray one would be dropped before ``apply_to_structure``
    # could refuse it.
    # Refused by the list's owner (`sidecars.molstruct.unread_keys`): the
    # retired keys are accepted and ignored rather than refused -- a key
    # this project used to write is not a key nobody reads, and the files
    # carrying it are the user's (`structure.RETIRED_METADATA_KEYS`).
    from molbuilder.structure import (ATOM_COLUMNS, FILE_FIELDS,
                                      RETIRED_METADATA_KEYS)
    said = unread_keys(data)
    if said:
        raise MolstructJsonError(f"sidecar {source}: {said}.")
    if retired_out is not None:
        retired_out.update({k: data[k] for k in RETIRED_METADATA_KEYS
                            if data.get(k) is not None})
    if stated_out is not None:
        stated_out.update(k for k in data if k in FILE_FIELDS)

    # The payload through its ONE BUILDER, with the file's version and its
    # envelope as it states it -- the builder holds which envelope keys a
    # version states and every check of them and of the fields.  This
    # catches malformed user-edited JSON BEFORE any consumer tries to
    # apply the data to a Structure (where the same checks would run
    # but with a less specific error message that doesn't mention
    # which file is at fault).
    try:
        return to_dict(
            # SPREAD, never listed: every field the Structure names, read by
            # the same tuple the writer spreads (`model/structure.md` § 2.2).
            {k: data.get(k) for k in FILE_FIELDS},
            identity       = {k: data[k] for k in ATOM_COLUMNS if k in data},
            n_atoms_total  = data.get("n_atoms_total"),
            n_frames_total = data.get("n_frames_total"),
            geometry_hash  = data.get("geometry_hash"),
            info           = data.get("info"),
            created_by     = data.get("created_by", "unknown"),
            created_at     = data.get("created_at"),
            atom_index_map = data.get(MAP_KEY),
            schema_version = sv,
        )
    except MolstructJsonError as exc:
        raise MolstructJsonError(f"sidecar {source}: {exc}") from exc


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
        payload = load(path)
        return build_sidecar_result(
            payload=payload,
            kind="molstruct",
            parser_name=cls.name,
            source=path,
        )


__all__ = ["MolstructJsonError", "MolstructSidecarFileParser"]

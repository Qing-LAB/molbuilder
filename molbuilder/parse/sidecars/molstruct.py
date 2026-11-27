"""``.molstruct.json`` sidecar FileParser, and the sidecar's reader.

The READ side: :func:`load_record`, the one reader of a sidecar's bytes
(through the one record reader, `persist.object_of`), which the read door
-- ``load``, beside the write side (``dumps``, ``sidecar_path_for``,
``to_dict``, ``apply_to_structure``, ``geometry_hash``) in
:mod:`molbuilder.sidecars.molstruct` -- asks with the file's bytes, and the
registry's FileParser, which asks that door.

The sidecar carries per-atom region + frozen-atom metadata that
rides next to a structure file (``<stem>.molstruct.json``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from molbuilder.parse.base import FileParser
from molbuilder.parse.types import SidecarResult
# ``SCHEMA_VERSION`` (the on-disk version constant) and the payload's one
# builder (``to_dict``) live in the write-side module, so read and write
# surfaces stay in lock-step; the pair's refusals are `structure_pair`'s.
from molbuilder.sidecars.molstruct import (
    READABLE_VERSIONS,
    SCHEMA_SINCE,
    SCHEMA_VERSION,
    is_sidecar,
    load,
    to_dict,
    unread_keys,
)
from molbuilder.atom_permutation import MAP_KEY
from molbuilder.persist import LATER_WAY, object_of
from molbuilder.structure_pair import (MolstructJsonError, MolstructUnreadable,
                                       MolstructVersionError)

from ._helpers import build_sidecar_result


# The readable-version gate is `sidecars.molstruct.READABLE_VERSIONS`,
# imported above and used by `load_record` below.


def load_record(data: bytes, *, source: str = "<sidecar>") -> Dict[str, Any]:
    """Parse + validate a sidecar's **bytes** -- what the read door
    (``sidecars.molstruct.load``) read from the file -- through the one
    record reader (`persist.object_of`: UTF-8 text, one JSON object).  Returns the
    normalised dict, built by the payload's one builder
    (``sidecars.molstruct.to_dict``), so an on-disk sidecar that snuck past
    with hand-edited indices fails loudly here, with the file named, rather
    than at engine-load time.  ``source`` names the origin for error messages
    (the path the read door read).

    Validation is strict: a version this build does not read -- every
    sidecar saved before 2026-11-04 (the user, 2026-11-26: "Drop all as
    asked"), and one a later molbuilder wrote -- missing required fields, wrong types, out-of-range indices and
    a key the version does not carry all raise :class:`MolstructJsonError`.
    Region overlap (one atom in two regions) is INTENTIONALLY permitted --
    the model is multi-label freeform tagging.  ``regions`` / ``frozen_atoms``
    are permissive (missing == empty); everything else is required.
    """
    try:
        data = object_of(data)
    except ValueError as exc:
        raise MolstructUnreadable(f"sidecar {source}: {exc}") from exc

    sv = data.get("schema_version")
    if (isinstance(sv, int) and not isinstance(sv, bool)
            and sv > max(READABLE_VERSIONS)):
        # A LATER MOLBUILDER'S PAIR: its way on is that molbuilder, never
        # this one's for a pair saved before the version it reads
        # (`job-contracts.md` § 6.1, *The versions' words*).
        fact = (f"its sidecar is version {sv!r}, written by a later "
                f"molbuilder than this one, and this molbuilder reads "
                f"version {SCHEMA_VERSION}")
        raise MolstructVersionError(
            f"sidecar {source}: schema_version is {sv!r}, written by a "
            f"later molbuilder than this one -- this molbuilder reads "
            f"version {SCHEMA_VERSION} -- {LATER_WAY}.",
            fact=fact, later=True)
    if sv not in READABLE_VERSIONS:
        # REFUSED, NOT READ PARTIALLY -- one version read (`structure-
        # molstruct.md` § 2), the way on said: the pair made again, or its
        # geometry opened alone (metadata is never required).
        raise MolstructVersionError(
            f"sidecar {source}: schema_version is {sv!r} -- a pair saved "
            f"before {SCHEMA_SINCE} -- and this molbuilder reads version "
            f"{SCHEMA_VERSION} alone.  Make the pair again: save the "
            f"structure from the calculation it came from, or build it again; "
            f"its .xyz opens alone with this .molstruct.json moved aside, its "
            f"labels, cell and rows gone.",
            fact=(f"its sidecar is version {sv!r}, saved before "
                  f"{SCHEMA_SINCE}, and this molbuilder reads version "
                  f"{SCHEMA_VERSION} alone"))

    # A KEY NOBODY READS IS METADATA THE WRITER THINKS IT SAVED.  Checked
    # HERE, where the payload is still whole: this function reads the keys it
    # NAMES, so a stray one would be dropped before ``apply_to_structure``
    # could refuse it (`sidecars.molstruct.unread_keys`, the list's owner).
    from molbuilder.structure import ATOM_COLUMNS, FILE_FIELDS
    said = unread_keys(data)
    if said:
        raise MolstructUnreadable(f"sidecar {source}: {said}.")

    # The payload through its ONE BUILDER, its envelope as the file states
    # it -- the builder holds every check of the envelope and the fields.
    # This catches malformed user-edited JSON BEFORE any consumer tries to
    # apply the data to a Structure (where the same checks would run but
    # with a less specific error message that doesn't mention which file is
    # at fault).
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
        )
    except MolstructJsonError as exc:
        # THE FILE'S CONTENT its one builder refuses: the file does not read.
        raise MolstructUnreadable(f"sidecar {source}: {exc}") from exc


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

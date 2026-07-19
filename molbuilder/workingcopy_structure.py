"""Structure + sidecar codec (``StructureCodec``).

MODULE: a standalone L2 codec for the ``<stem>.xyz`` (coordinates) +
``<stem>.molstruct.json`` (labels/annotations) file pair.  ``load`` parses the
``.xyz``/``.pdb`` source and applies the companion sidecar into a
:class:`~molbuilder.structure.Structure`; ``files`` serialises one back to the
pair; ``scratch_blob`` / ``from_scratch`` round-trip it through an in-memory
``{xyz, sidecar}`` blob.  It never learns what an atom means beyond structure +
sidecar.

USED BY: ``/api/structure/resolve-cell`` (web/blueprints/build.py) — rebuilds a
Structure from a scratch blob via ``from_scratch`` to resolve the effective cell.
(This codec also *used* to back the ``molbuilder.workingcopy`` working-copy core +
the ``/api/workingcopy/*`` structure-editor door; both were retired — the codec is
the survivor.)

Layer: L2 — reuses `structure` (L1) + the `sidecars.molstruct` write/read stack.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List, Tuple

from .structure import Structure, annotations_to_json
from .sidecars import molstruct


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class StructureCodec:
    """``.xyz`` + ``.molstruct.json`` ⇄ :class:`~molbuilder.structure.Structure`
    (+ in-memory scratch round-trip)."""

    # ---- load durable -> working Structure --------------------------- #
    def load(self, source_path) -> Structure:
        src = Path(source_path)
        # Parse the SOURCE in ITS OWN format (dispatch on the extension) -- the
        # file picker accepts .xyz AND .pdb, and each needs its own parser: a
        # .pdb read as XYZ chokes on its "HEADER ..." first line.  An unknown
        # extension is an EXPLICIT error, not a silent from_xyz attempt.  (The
        # working copy is then maintained as .xyz + sidecar via files().)
        suffix = src.suffix.lower()
        if suffix == ".pdb":
            struct = Structure.from_pdb(src)
        elif suffix == ".xyz":
            struct = Structure.from_xyz(src)
        else:
            raise ValueError(
                f"StructureCodec.load: unsupported structure format "
                f"{src.suffix!r} for {src.name!r}; expected .xyz or .pdb")
        sidecar_path = molstruct.sidecar_path_for(src)
        if sidecar_path.exists():
            molstruct.apply_to_structure(struct, molstruct.load(sidecar_path))
        return struct

    # ---- the durable files: <stem>.xyz + <stem>.molstruct.json ------- #
    def files(self, struct: Structure, target) -> List[Tuple[Path, bytes]]:
        target = Path(target)
        xyz_bytes = struct.to_xyz().encode("utf-8")
        sidecar_bytes = (json.dumps(self._sidecar_dict(struct, xyz_bytes),
                                    indent=2) + "\n").encode("utf-8")
        return [(target, xyz_bytes),
                (molstruct.sidecar_path_for(target), sidecar_bytes)]

    # ---- scratch round-trip ------------------------------------------ #
    def scratch_blob(self, struct: Structure) -> Any:
        xyz_text = struct.to_xyz()
        return {"xyz": xyz_text,
                "sidecar": self._sidecar_dict(struct, xyz_text.encode("utf-8"))}

    def from_scratch(self, blob: Any) -> Structure:
        struct = Structure.from_xyz(blob["xyz"])
        molstruct.apply_to_structure(struct, blob["sidecar"])
        return struct

    # ---- internal ---------------------------------------------------- #
    def _sidecar_dict(self, struct: Structure, xyz_bytes: bytes) -> dict:
        # structure_hash is the sha256 of the .xyz we are about to write, so the
        # committed sidecar's hash matches the committed .xyz (the atom-identity
        # invariant the generation gate relies on).
        return molstruct.to_dict(
            n_atoms_total  = struct.n_atoms,
            structure_hash = _sha256_bytes(xyz_bytes),
            regions        = dict(struct.regions or {}),
            frozen_atoms   = list(struct.frozen_atoms or []),
            annotations    = annotations_to_json(struct.annotations),
            cell           = struct.cell.tolist() if struct.cell is not None else None,
            cell_origin    = (struct.cell_origin.tolist()
                              if struct.cell_origin is not None else None),
            pbc            = [bool(x) for x in struct.pbc] if struct.pbc is not None else None,
            axis_kind      = list(struct.axis_kind) if struct.axis_kind is not None else None,
            vacuum         = list(struct.vacuum),
        )

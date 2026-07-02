"""Structure + sidecar codec — the first application of the working-copy core
(working-copy-persistence.md § 11).

Loads/saves a structure as the pair ``<stem>.xyz`` (coordinates) +
``<stem>.molstruct.json`` (labels/annotations).  The working ``data`` is a
:class:`~molbuilder.structure.Structure`.  Plugging this into
:mod:`molbuilder.workingcopy` gives the structure editor a draft (survives
reload/crash) + an explicit save (overwrite or save-as).

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
    """Codec plugging structure + sidecar into :mod:`molbuilder.workingcopy`."""

    # ---- load durable -> working Structure --------------------------- #
    def load(self, source_path) -> Structure:
        src = Path(source_path)
        struct = Structure.from_xyz(src)
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
            pbc            = [bool(x) for x in struct.pbc] if struct.pbc is not None else None,
        )

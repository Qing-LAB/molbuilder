"""Structure + sidecar codec (``StructureCodec``).

MODULE: a standalone L2 codec for the ``<stem>.xyz`` (coordinates) +
``<stem>.molstruct.json`` (labels/annotations) file pair.  ``load`` parses the
``.xyz``/``.pdb`` source and applies the companion sidecar into a
:class:`~molbuilder.structure.Structure`; ``files`` serialises one back to the
pair; ``scratch_blob`` / ``from_scratch`` round-trip it through an in-memory
``{xyz, sidecar}`` blob.  It never learns what an atom means beyond structure +
sidecar.

USED BY: ``/api/structure/periodicity`` + ``/api/structure/save`` +
``/api/build/load`` (web/blueprints/build.py) — the gated pair/blob seams.
(This codec also *used* to back the ``molbuilder.workingcopy`` working-copy core +
the ``/api/workingcopy/*`` structure-editor door; both were retired — the codec is
the survivor.)

Layer: L2 — reuses `structure` (L1) + the `sidecars.molstruct` write/read stack.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Tuple

from .structure import Structure
from .sidecars import molstruct


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _metadata_is_default(meta: dict) -> bool:
    """True when a metadata dict (``Structure.metadata_to_dict`` output) carries
    nothing worth a sidecar -- a plain molecule (no cell / origin / pbc /
    regions / frozen / annotations / vacuum / non-isolated axis).  Decides
    whether the ``.molstruct.json`` half of the pair exists at all
    (``no .json == empty metadata``)."""
    if meta.get("cell") is not None or meta.get("cell_origin") is not None:
        return False
    if meta.get("regions") or meta.get("frozen_atoms") or meta.get("annotations"):
        return False
    if meta.get("pbc") and any(meta["pbc"]):
        return False
    if any(float(v) != 0.0 for v in (meta.get("vacuum") or ())):
        return False
    ak = meta.get("axis_kind")
    if ak and any(k != "isolated" for k in ak):
        return False
    return True


class StructureCodec:
    """``.xyz`` + ``.molstruct.json`` ⇄ :class:`~molbuilder.structure.Structure`
    (+ in-memory scratch round-trip)."""

    # ---- load durable -> working Structure --------------------------- #
    def load(self, source_path, *,
             notices_out: "list | None" = None) -> Structure:
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
        # The frame-contract gate on the READ seam too (§ 6.1 clause 1-2:
        # defaulting/healing is gated by the LOADER and saver of the pair).
        # Without this, /api/build/load served a corrupted pair unhealed and
        # MolView drew the box from the world origin -- the live symptom on
        # projects/hemeC-dithiol (explicit cell, dropped origin).
        from .periodicity_gate import validate_and_heal
        struct, notices = validate_and_heal(struct)
        if notices_out is not None:
            notices_out.extend(notices)
        return struct

    # ---- the durable files: <stem>.xyz + <stem>.molstruct.json ------- #
    def files(self, struct: Structure, target) -> List[Tuple[Path, bytes]]:
        target = Path(target)
        xyz_bytes = struct.to_xyz().encode("utf-8")
        sidecar_bytes = (json.dumps(self._sidecar_dict(struct, xyz_bytes),
                                    indent=2) + "\n").encode("utf-8")
        return [(target, xyz_bytes),
                (molstruct.sidecar_path_for(target), sidecar_bytes)]

    # ---- write the pair to disk, atomically -------------------------- #
    def write(self, struct: Structure, target, *, atomic: bool = True) -> Path:
        """Write ``struct`` to the ``<stem>.xyz`` + ``<stem>.molstruct.json``
        pair on disk and return the geometry path.  THE paired-file door
        (structure-authority.md § 3.3): owns the pairing rule + the
        both-or-neither atomicity so no caller re-derives the sidecar path or
        re-implements the write order.

        Atomicity: each half is staged to a temp sibling and ``os.replace``-d
        (per-file atomic).  The geometry is swapped first, then the sidecar, so
        the only visible interleaving is OLD-sidecar + NEW-geometry for a tiny
        window -- never a torn file.  When ``struct`` carries no metadata worth
        persisting AND a stale sidecar exists, it is removed so the pair can't
        disagree (``no .json == empty metadata``, matching :meth:`load`)."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        xyz_text = struct.to_xyz()
        meta = struct.metadata_to_dict()
        sidecar_path = molstruct.sidecar_path_for(target)
        want_sidecar = not _metadata_is_default(meta)

        if atomic:
            tmp = target.with_suffix(target.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(xyz_text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        else:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(xyz_text)

        if want_sidecar:
            payload = molstruct.to_dict(
                meta,
                n_atoms_total  = struct.n_atoms,
                structure_hash = _sha256_bytes(xyz_text.encode("utf-8")),
            )
            molstruct.save(sidecar_path, payload)   # tempfile + os.replace atomic
        elif sidecar_path.exists():
            sidecar_path.unlink()
        return target

    # ---- read the pair from disk (alias of load, symmetric name) ----- #
    def read(self, source_path, *,
             notices_out: "list | None" = None) -> Structure:
        """Symmetric read-side name for :meth:`load` -- parse the geometry +
        apply its paired sidecar into a Structure (missing sidecar => empty
        metadata, not an error).  ``notices_out`` collects the frame-contract
        gate's heal notices (structure-periodicity.md 6.1) so the load door
        can SURFACE a heal instead of silently rewriting the user's state."""
        return self.load(source_path, notices_out=notices_out)

    # ---- scratch round-trip ------------------------------------------ #
    def scratch_blob(self, struct: Structure) -> Any:
        xyz_text = struct.to_xyz()
        return {"xyz": xyz_text,
                "sidecar": self._sidecar_dict(struct, xyz_text.encode("utf-8"))}

    def from_scratch(self, blob: Any, *,
                     notices_out: "list | None" = None) -> Structure:
        struct = Structure.from_xyz(blob["xyz"])
        molstruct.apply_to_structure(struct, blob["sidecar"])
        # The frame-contract gate (structure-periodicity.md § 6.1): ALL
        # heal/validation of periodicity state happens at this seam.  A
        # stored explicit cell that does not contain its atoms (the 2026-07
        # hemeC corruption) is healed here -- origin to the expected corner
        # -- and the notice surfaces when the caller passes notices_out.
        from .periodicity_gate import validate_and_heal
        struct, notices = validate_and_heal(struct)
        if notices_out is not None:
            notices_out.extend(notices)
        return struct

    # ---- internal ---------------------------------------------------- #
    def _sidecar_dict(self, struct: Structure, xyz_bytes: bytes) -> dict:
        # structure_hash is the sha256 of the .xyz we are about to write, so the
        # committed sidecar's hash matches the committed .xyz (the atom-identity
        # invariant the generation gate relies on).
        # ONE metadata authority: the struct serialises its own field set
        # (metadata_to_dict), and to_dict layers the envelope on -- no field is
        # hand-listed here, so this can't drift from the dataclass.
        return molstruct.to_dict(
            struct.metadata_to_dict(),
            n_atoms_total  = struct.n_atoms,
            structure_hash = _sha256_bytes(xyz_bytes),
        )

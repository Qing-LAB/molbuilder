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
import os
from pathlib import Path
from typing import Any, List, NamedTuple, Tuple

from .structure import Structure
from .sidecars import molstruct


class StructurePair(NamedTuple):
    """What a Structure looks like outside memory: the coordinate document, the
    sidecar payload beside it, and whether that payload is worth keeping.

    ONE shape for every consumer -- disk, bytes, blob, wire -- so "what does this
    structure look like when it leaves" has one answer instead of one per caller.
    """
    document: str
    sidecar: dict
    keep_sidecar: bool


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _metadata_is_default(meta: dict) -> bool:
    """True when a metadata dict (``Structure.metadata_to_dict`` output) carries
    nothing worth a sidecar -- a plain molecule (no cell / origin / pbc /
    labels / annotations / vacuum / non-isolated axis).  `regions` covers the
    reserved labels too; there is no second store to check.  Decides
    whether the ``.molstruct.json`` half of the pair exists at all
    (``no .json == empty metadata``)."""
    if meta.get("cell") is not None or meta.get("cell_origin") is not None:
        return False
    if meta.get("regions") or meta.get("annotations"):
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

    # ---- THE ONE GENERATOR: a Structure -> the pair --------------------- #
    def pair(self, struct: Structure) -> "StructurePair":
        """A Structure as the two things that represent it: the coordinate
        document, and the sidecar payload beside it.

        THE ONE PLACE either is produced.  :meth:`write` puts this on disk,
        :meth:`files` hands it over as bytes, :meth:`scratch_blob` hands it over
        as a round-trip blob, and the export route returns it -- so a structure
        saved to a project and the same structure downloaded cannot differ.
        They used to be three code paths computing the same three calls, agreeing
        by coincidence rather than by construction; ``files`` even serialised the
        JSON with different settings from ``save``, so a non-ASCII region label
        came out escaped on one path and literal on the other.

        ``keep_sidecar`` is False when the metadata is all default -- a plain
        molecule with no cell, labels, frozen atoms or annotations.  Then the
        pair is the document alone and a stale sidecar beside it is removed, so
        "no .json" always means "no metadata" (:meth:`load` reads it that way).
        The payload is still built, because a blob is a round trip rather than a
        file and its reader expects one.
        """
        document = struct.to_xyz()
        meta = struct.metadata_to_dict()
        payload = molstruct.to_dict(
            meta,
            n_atoms_total  = struct.n_atoms,
            structure_hash = _sha256_bytes(document.encode("utf-8")),
        )
        return StructurePair(document=document, sidecar=payload,
                             keep_sidecar=not _metadata_is_default(meta))

    # ---- the durable files: <stem>.xyz + <stem>.molstruct.json ------- #
    def files(self, struct: Structure, target) -> List[Tuple[Path, bytes]]:
        """The pair as bytes, with the paths they belong at -- what :meth:`write`
        writes, without writing it."""
        target = Path(target)
        made = self.pair(struct)
        out = [(target, made.document.encode("utf-8"))]
        if made.keep_sidecar:
            out.append((molstruct.sidecar_path_for(target),
                        molstruct.dumps(made.sidecar).encode("utf-8")))
        return out

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
        made = self.pair(struct)                 # the ONE generator
        xyz_text = made.document
        sidecar_path = molstruct.sidecar_path_for(target)

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

        if made.keep_sidecar:
            molstruct.save(sidecar_path, made.sidecar)  # tempfile + os.replace
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
        made = self.pair(struct)
        return {"xyz": made.document, "sidecar": made.sidecar}

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
        made = self.pair(struct)
        return {"xyz": made.document, "sidecar": made.sidecar}

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

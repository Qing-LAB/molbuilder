"""Structure + sidecar codec (``StructureCodec``).

MODULE: the standalone L2 codec for the ``<stem>.xyz`` (coordinates) +
``<stem>.molstruct.json`` (labels/annotations) file pair.  It owns FOUR things
nothing else in the system may hold a second copy of:

  1. the PAIRING RULE -- how the sidecar's name follows the geometry's;
  2. the FORMAT CHOICE -- a plain ``.xyz`` for one frame, extended XYZ for many,
     decided by the count and never asked as a separate question;
  3. the SIDECAR ENVELOPE -- ``schema_version``, the ``structure_hash`` pinning
     it to its geometry, and the one serialisation (``molstruct.dumps``);
  4. the INVARIANTS -- ``no .json == empty metadata`` in both directions,
     both-or-neither atomicity on write, the periodicity gate on read.

SHAPE: one generator, and one adapter per destination.  :meth:`StructureCodec.pair`
is the generator; :meth:`~StructureCodec.write` puts it on disk,
:meth:`~StructureCodec.files` hands it over as bytes WITH THE NAMES THEY BELONG
UNDER, and :meth:`~StructureCodec.read` brings it back.

THE RULE (model/structure.md § 2.4): *every structure-to-bytes translation goes
through this codec, and every adapter has exactly one door.*  An adapter with no
door is either RETIRED or UNBUILT, and those have opposite fixes -- which is why
the question gets asked rather than answered by counting callers.

USED BY: ``/api/structure/save`` -> ``write`` · ``/api/structure/export`` ->
``files`` · ``/api/build/load`` -> ``read`` (web/blueprints/build.py) · and
``bundle_writer.write_bundle_as_handoff`` -> ``write``.  NOT yet by the CLI,
which still writes geometry alone -- the last surface not obeying the rule
(task #73).

RETIRED 2026-07-31: ``scratch_blob`` / ``from_scratch``, which round-tripped a
structure through an in-memory ``{xyz, sidecar}`` TEXT blob.  Their last caller
was ``/api/structure/periodicity`` before it took the envelope; a blob means a
coordinate document is written in order to ask a question about coordinates,
which is what web/molview.md § 11.7 forbids.  (This codec also used to back the
retired ``molbuilder.workingcopy`` core + the ``/api/workingcopy/*`` door; it is
the survivor of both.)

Layer: L2 — reuses `structure` (L1) + the `sidecars.molstruct` write/read stack.
"""
from __future__ import annotations


import hashlib
import os
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

from .structure import Structure
from .sidecars import molstruct


class StructurePair(NamedTuple):
    """What a Structure looks like outside memory: the coordinate document, the
    sidecar payload beside it, whether that payload is worth keeping, and the
    extension it belongs under.

    ONE shape for every consumer -- disk, bytes, wire -- so "what does this
    structure look like when it leaves" has one answer instead of one per caller.

    ``suffix`` is always ``.xyz``: extended XYZ is a strict superset of plain
    XYZ, so the format can follow the frame count while the NAME does not have
    to.  It is carried here rather than assumed by each caller because the
    pairing rule is the codec's -- a caller that appends its own extension is
    keeping a second copy of a rule it does not own, which is how the sidecar's
    name came to be derived in two places.
    """
    document: str
    sidecar: dict
    keep_sidecar: bool
    suffix: str


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
    """``.xyz`` + ``.molstruct.json`` ⇄ :class:`~molbuilder.structure.Structure`."""

    #: THE ONE EXTENSION THIS CODEC WRITES.  One frame or four hundred, plain
    #: XYZ or extended -- the file is ``.xyz``, because extended XYZ is a strict
    #: superset of plain XYZ and shares its extension by convention.
    GEOMETRY_SUFFIX = ".xyz"

    #: Extensions RECOGNISED on a target so they are replaced rather than
    #: appended to.  ``.extxyz`` is here because it is a name a caller might
    #: hand us, NOT one we produce: ``files(struct, "run.extxyz")`` answers
    #: ``run.xyz``.  Anything else keeps its whole name and gets the suffix
    #: appended, so ``run.v2`` becomes ``run.v2.xyz`` rather than losing the
    #: ``.v2`` a bare ``with_suffix`` would have eaten.
    _REPLACEABLE_SUFFIXES = (".xyz", ".extxyz")

    # ---- load durable -> working Structure --------------------------- #
    def load(self, source_path, *,
             frames_out: "list | None" = None) -> Structure:
        """Read the pair back into a Structure.

        ``frames_out`` closes the round trip this codec can now write: a range
        goes out as one extended-XYZ document (:meth:`pair`), and a caller that
        passes a list here gets EVERY frame of it back, in file order.  Without
        it a trajectory reopens as its first frame -- which is the right default
        for a Structure (one geometry) and the wrong answer for whoever wrote
        the range.
        """
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
            struct = Structure.from_xyz(src, frames_out=frames_out)
        else:
            raise ValueError(
                f"StructureCodec.load: unsupported structure format "
                f"{src.suffix!r} for {src.name!r}; expected .xyz or .pdb")
        sidecar_path = molstruct.sidecar_path_for(src)
        if sidecar_path.exists():
            molstruct.apply_to_structure(struct, molstruct.load(sidecar_path))
        # READING DOES NOT JUDGE (structure-periodicity.md § 8.2, decided
        # 2026-08-03).  A file whose sidecar holds an unusable box -- a
        # left-handed cell, or one too small for any origin -- OPENS, and what
        # is wrong with it is reported by whoever hands the structure on.
        #
        # It used to raise here, and that made such a file unopenable and
        # therefore UNFIXABLE: the Cell page is the one place the box can be
        # corrected, and it cannot be reached without the structure on screen.
        # The load door answered "could not load <file>" and the only way out
        # was to hand-edit the .molstruct.json outside molbuilder, or delete it
        # and lose the labels with it.
        #
        # NOTHING IS LEFT UNGUARDED BY THIS.  What must not happen is a
        # CALCULATION built on an impossible box, and that is refused where it
        # belongs: `validate()` reports a left-handed cell as an ERROR, and both
        # emitters run `report(validate(...))` before writing anything
        # (siesta/input.py, and the PySCF renderer).  The web's emitting doors
        # refuse it a second time at the request seam.  So the box is stopped at
        # every door that would ACT on it, and at none that would merely show it.
        return struct

    # ---- THE ONE GENERATOR: a Structure -> the pair --------------------- #
    def pair(self, struct: Structure, *,
             frames: "Sequence | None" = None) -> "StructurePair":
        """A Structure as the two things that represent it: the coordinate
        document, and the sidecar payload beside it.

        THE ONE PLACE either is produced.  :meth:`write` puts this on disk and
        :meth:`files` hands it over as named bytes (which is what the export
        route returns) -- so a structure saved to a project and the same
        structure downloaded cannot differ.  They used to be three code paths
        computing the same three calls, agreeing by coincidence rather than by
        construction; ``files`` even serialised the JSON with different settings
        from ``save``, so a non-ASCII region label came out escaped on one path
        and literal on the other.

        ``keep_sidecar`` is False when the metadata is all default -- a plain
        molecule with no cell, labels, frozen atoms or annotations.  Then the
        pair is the document alone and a stale sidecar beside it is removed, so
        "no .json" always means "no metadata" (:meth:`load` reads it that way).
        """
        # ONE FRAME OR MANY, decided by what was handed over and by nothing
        # else.  A trajectory needs extended XYZ, because a plain .xyz has
        # nowhere to put a cell and would lose the box on every frame; a single
        # structure keeps the plain .xyz every code reads.  The caller says
        # WHICH frames (molview.md § 11.3's range); the format follows from how
        # many there are, and is never a second question.
        #
        # THE SUFFIX IS DECIDED HERE, WITH THE FORMAT, and travels with the
        # pair.  Deriving it anywhere else is deriving it a second time, and a
        # second derivation is a chance to disagree with the bytes.
        #
        # THE SIDECAR IS BUILT ONCE EITHER WAY.  The labels and the cell are the
        # structure's shared identity -- the same for frame 0 and frame 400 --
        # so there is one .json beside a trajectory, not one per frame.  Its
        # hash pins it to the document actually written, whichever that is.
        # THE FORMAT follows the count; THE NAME does not follow the format.
        # Extended XYZ is a strict SUPERSET of plain XYZ -- the extra facts ride
        # in the comment line, which a plain reader skips -- so both are written
        # under ``.xyz``.  That is the ordinary convention (ASE, where the
        # format's modern use comes from, writes extended XYZ to ``.xyz`` by
        # default), and it is the only extension :meth:`load` accepts: a range
        # named ``.extxyz`` was a file THIS CODEC COULD NOT REOPEN, so a
        # trajectory saved into a project could never be loaded again.
        document = (struct.to_extxyz(frames=frames) if frames
                    else struct.to_xyz())
        meta = struct.metadata_to_dict()
        payload = molstruct.to_dict(
            meta,
            n_atoms_total  = struct.n_atoms,
            structure_hash = _sha256_bytes(document.encode("utf-8")),
        )
        return StructurePair(document=document, sidecar=payload,
                             keep_sidecar=not _metadata_is_default(meta),
                             suffix=self.GEOMETRY_SUFFIX)

    # ---- the pair as NAMED bytes: <stem>.xyz + <stem>.molstruct.json -- #
    def files(self, struct: Structure, target, *,
              frames: "Sequence | None" = None) -> List[Tuple[Path, bytes]]:
        """The pair as bytes, WITH THE NAMES THEY BELONG UNDER -- what
        :meth:`write` writes, without writing it, and what the export door
        answers with.

        THE SUFFIX IS THE ONE :meth:`pair` CHOSE, not the one the caller
        guessed.  A range produces extended XYZ, so a caller appending ``.xyz``
        names a file after a format it does not contain -- at the extension
        every trajectory reader dispatches on.  Hand this a bare stem and it
        comes back named correctly; hand it a full ``<stem>.xyz`` and the suffix
        is corrected in place, which is why comparing this against :meth:`write`
        still compares the same paths.

        Contrast :meth:`write`, which does NOT correct the name: a project save
        was given an exact path through a picker with an overwrite gate on it,
        so the bytes go exactly there.  An export was given a stem and nothing
        else.  Different questions (model/structure.md § 2.4).
        """
        target = Path(target)
        made = self.pair(struct, frames=frames)
        if target.suffix.lower() in self._REPLACEABLE_SUFFIXES:
            target = target.with_suffix(made.suffix)
        else:
            target = target.with_name(target.name + made.suffix)
        out = [(target, made.document.encode("utf-8"))]
        if made.keep_sidecar:
            out.append((molstruct.sidecar_path_for(target),
                        molstruct.dumps(made.sidecar).encode("utf-8")))
        return out

    # ---- write the pair to disk, atomically -------------------------- #
    def write(self, struct: Structure, target, *, atomic: bool = True,
              frames: "Sequence | None" = None) -> Path:
        """Write ``struct`` to the ``<stem>.xyz`` + ``<stem>.molstruct.json``
        pair on disk and return the geometry path.  THE paired-file door
        (structure-authority.md § 3.3): owns the pairing rule + the
        both-or-neither atomicity so no caller re-derives the sidecar path or
        re-implements the write order.

        The target is written VERBATIM -- unlike :meth:`files`, this does not
        correct the suffix, because the caller did not guess it: a save names an
        exact path, chosen through a picker and cleared by an overwrite gate,
        and silently writing somewhere else would make that gate a lie.  (Known
        consequence: saving a frame RANGE to a ``.xyz`` path puts extended-XYZ
        bytes under an ``.xyz`` name.  Naming that file is the caller's job and
        the export door is where the codec does it.)

        Atomicity: each half is staged to a temp sibling and ``os.replace``-d
        (per-file atomic).  The geometry is swapped first, then the sidecar, so
        the only visible interleaving is OLD-sidecar + NEW-geometry for a tiny
        window -- never a torn file.  When ``struct`` carries no metadata worth
        persisting AND a stale sidecar exists, it is removed so the pair can't
        disagree (``no .json == empty metadata``, matching :meth:`load`)."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        made = self.pair(struct, frames=frames)  # the ONE generator
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
             frames_out: "list | None" = None) -> Structure:
        """Symmetric read-side name for :meth:`load` -- parse the geometry +
        apply its paired sidecar into a Structure (missing sidecar => empty
        metadata, not an error).  ``frames_out`` collects every frame of a
        multi-frame document."""
        return self.load(source_path, frames_out=frames_out)

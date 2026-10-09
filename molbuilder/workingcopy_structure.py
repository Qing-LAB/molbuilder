"""Structure + sidecar codec (``StructureCodec``).

MODULE: the standalone L2 codec for the ``<stem>.xyz`` (coordinates) +
``<stem>.molstruct.json`` (labels/annotations) file pair.  It owns FOUR things
nothing else in the system may hold a second copy of:

  1. the PAIRING RULE -- how the sidecar's name follows the geometry's;
  2. the FORMAT CHOICE -- plain ``.xyz``, a block per frame the structure holds,
     decided by the count and never asked as a separate question;
  3. the SIDECAR ENVELOPE -- ``schema_version``, the ``structure_hash`` pinning
     it to its geometry, and the one serialisation (``molstruct.dumps``);
  4. the INVARIANTS -- ``no .json == empty metadata`` in both directions,
     both-or-neither atomicity on write.  NOT a periodicity gate on read:
     reading does not judge (structure-periodicity.md § 8.2, 2026-08-03).

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
the task hand-over doors -> ``files``/``write`` · the PySCF script, from
inside its run -> ``write_moved`` (imported from ``mb_pyscf.pyz``).

Layer: L2 — reuses `structure` (L1) + the `sidecars.molstruct` write/read stack.
"""
from __future__ import annotations


import hashlib
import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

# TWO WAYS, because this module travels: the PySCF script writes every
# geometry it saves through it (`write_moved`), imported from `mb_pyscf.pyz`
# beside the job (`runwrap.PYSCF_COMPANIONS`), where the package is not
# installed.  What that call reaches imports only the standard library and
# numpy; the rest of the codec reads through the package.
try:                                        # inside molbuilder
    from .structure import Structure
    from .sidecars import molstruct
except ImportError:                         # beside a job, in mb_pyscf.pyz
    from structure import Structure
    import molstruct


class StructurePair(NamedTuple):
    """What a Structure looks like outside memory: the coordinate document, the
    sidecar payload beside it, whether that payload is worth keeping, and the
    extension it belongs under.

    ONE shape for every consumer -- disk, bytes, wire -- so "what does this
    structure look like when it leaves" has one answer instead of one per caller.

    ``suffix`` is ``.xyz`` unless the destination named ``.pdb``: one frame or
    many, the document is plain XYZ, a block per frame.  The CONTAINER is the
    one axis the caller names -- ``write`` reads it off the target's suffix,
    which is the same suffix ``read`` dispatches on.

    Either way it is carried here rather than assumed by each caller, because
    the pairing rule is the codec's -- a caller that appends its own extension
    is keeping a second copy of a rule it does not own.
    """
    document: str
    sidecar: dict
    keep_sidecar: bool
    suffix: str


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


#: What a structure with nothing stated serialises to.  Built once from the
#: authority itself rather than typed out, and n-independent (every value is
#: a store-level default, none of them per-atom).
_DEFAULT_METADATA: Optional[dict] = None


def _metadata_is_default(meta: dict) -> bool:
    """True when a metadata dict (``Structure.metadata_to_dict`` output)
    carries nothing worth a sidecar.  Decides whether the
    ``.molstruct.json`` half of the pair exists at all
    (``no .json == empty metadata``).

    ASKED OF THE AUTHORITY, not re-enumerated.  Comparing against the authority's own empty output cannot drift, and a
    field added to ``METADATA_FIELDS`` is covered without touching this.
    """
    global _DEFAULT_METADATA
    if _DEFAULT_METADATA is None:
        from .structure import Structure
        _DEFAULT_METADATA = Structure(
            elements=[], positions=[]).metadata_to_dict()
    return meta == _DEFAULT_METADATA


class StructureCodec:
    """``.xyz`` + ``.molstruct.json`` ⇄ :class:`~molbuilder.structure.Structure`."""

    #: THE ONE EXTENSION THIS CODEC WRITES.  One frame or four hundred, the
    #: file is plain XYZ under ``.xyz``, a block per frame.
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
             frame: "int | None" = None,
             frames: bool = False,
             retired_out: "dict | None" = None,
             said_out: "list | None" = None) -> Structure:
        """Read the pair back into a Structure.

        WHICH FRAME (``model/structure.md`` § 2.3, *Which frame*): the pair is
        read whole and its sidecar applied to the whole set, then the answer is
        frame 0 -- what every caller has always had -- or frame ``frame``, or,
        with ``frames=True``, the whole set (:meth:`frame_choice`).  A
        one-frame file answers the same for all three.

        ``retired_out``: a caller that passes a dict gets each retired key the
        sidecar carried with a value -- read and ignored, and the load door
        says so (plan § 5q D14).

        ``said_out``: a caller that passes a list gets what the read says to a
        person about this file -- a LONE ``.xyz`` (no sidecar, not an engine's
        own file in one of our runs) is atoms and coordinates only, and its
        comment line is quoted, never read as metadata (``model/structure.md``
        § 2.3).  Every door that opens a file for a person says it.
        """
        src = Path(source_path)
        # Parse the SOURCE in ITS OWN format (dispatch on the extension) -- the
        # file picker accepts .xyz AND .pdb, and each needs its own parser: a
        # .pdb read as XYZ chokes on its "HEADER ..." first line.  An unknown
        # extension is an EXPLICIT error, not a silent from_xyz attempt.
        suffix = src.suffix.lower()
        if suffix not in (".xyz", ".pdb"):
            raise ValueError(
                f"StructureCodec.load: unsupported structure format "
                f"{src.suffix!r} for {src.name!r}; expected .xyz or .pdb")
        # THE FILE IS READ HERE, because this is the door that takes a path.
        # The readers below take text and nothing else (`structure.py`
        # ``_require_text``), so there is one place a structure file is opened
        # and it is the same place the sidecar is picked up.
        #
        # ``utf-8-sig`` accepts an optional UTF-8 BOM (some Windows editors
        # emit one) -- without an explicit encoding Python falls back to the
        # platform locale (cp1252 / latin-1 on some installs), which mojibakes
        # any non-ASCII in the XYZ comment line or PDB residue names.  Same
        # hardening molstruct_json / spectra_json / transport_json carry.
        text = src.read_text(encoding="utf-8-sig")
        if suffix == ".pdb":
            struct = Structure.from_pdb(text)
        else:
            struct = Structure.from_xyz(text)
        sidecar_path = molstruct.sidecar_path_for(src)
        if sidecar_path.exists():
            molstruct.apply_to_structure(
                struct, molstruct.load(sidecar_path, retired_out=retired_out))
        elif self._engine_own(src):
            # AN ENGINE'S OWN STRUCTURE FILE -- SIESTA's `<label>.xyz` in a run
            # folder -- has no sidecar, because molbuilder writes every
            # structure as a pair and the engine writes none.  Its frame is the
            # run's: the cell and axis kinds THAT RUN'S OWN DECK recorded
            # (`runs.declared`), and the engine's origin, a stated 0
            # (`model/structure-periodicity.md` § 6.0: *"Every door that makes
            # a structure from an engine's output states it"*) -- a source
            # molbuilder knows.
            from .runs import declared, run_of
            run_frame = declared(run_of(src)).frame()
            if run_frame and run_frame.get("cell") is not None:
                changes = {"cell": run_frame["cell"],
                           "engine_offset": run_frame["engine_offset"]}
                if run_frame.get("axis_kind"):
                    changes["axis_kind"] = tuple(run_frame["axis_kind"])
                struct = struct.replace(**changes)
        elif suffix == ".xyz" and said_out is not None:
            # A LONE FILE: nothing molbuilder knows states its metadata, so it
            # is what `from_xyz` read -- atoms and coordinates -- and the read
            # says so, quoting the comment line it did not read as metadata
            # (`model/structure.md` § 2.3).
            lines = text.splitlines()
            comment = lines[1].strip() if len(lines) > 1 else ""
            said_out.append(
                f"{src.name} has no .molstruct.json beside it, so only its "
                f"atoms and coordinates were read -- no cell, no labels"
                + (f"; its comment line was not read as metadata: «{comment}»"
                   if comment else "")
                + ".  Set what you meant on the Molbuilder tab, its Cell and "
                "Metadata pages; saving there writes it to the .molstruct.json.")
        # READING DOES NOT JUDGE (structure-periodicity.md § 8.2, decided
        # 2026-08-03).  A file whose sidecar holds an unusable box -- a
        # left-handed cell, or one too small for any origin -- OPENS, and what
        # is wrong with it is reported by whoever hands the structure on.
        #
        # Raising here would make such a file unopenable and therefore
        # UNFIXABLE: the Cell page is the one place the box can be corrected,
        # and it cannot be reached without the structure on screen.
        #
        # NOTHING IS LEFT UNGUARDED BY THIS.  What must not happen is a
        # CALCULATION built on an impossible box, and that is refused where it
        # belongs: `validate()` reports a left-handed cell as an ERROR, and
        # every deck runs `report(validate(...))` before writing anything
        # (`script_emit.render_deck`).  So the box is stopped at
        # every door that would ACT on it, and at none that would merely show it.
        return self.frame_choice(struct, frame=frame, frames=frames)

    @staticmethod
    def frame_choice(struct: Structure, *, frame: "int | None" = None,
                     frames: bool = False) -> Structure:
        """THE rule for which frame a read answers with, for every door
        that reads a structure -- this codec's :meth:`load` and
        ``/api/build/load``: frame 0 unless asked (*"keep the default
        reading, just take frame zero"*, user 2026-10-09), frame ``frame`` --
        0-based, refused outside the set naming its count -- or with
        ``frames`` the whole set."""
        if frames and frame is not None:
            raise ValueError("ask for one frame or for the whole set, not "
                             "both")
        if frames:
            return struct
        return struct.frame_at(0 if frame is None else frame)

    @staticmethod
    def _engine_own(src: Path) -> bool:
        """Is ``src`` an engine's own structure file in one of our runs --
        the one sidecar-less file whose frame has a known source, that run's
        own deck (``model/structure.md`` § 2.4)?"""
        from .runs import about, run_of
        return run_of(src) is not None and not about(src)["ours"]

    # ---- THE ONE GENERATOR: a Structure -> the pair --------------------- #
    def pair(self, struct: Structure, *,
             fmt: str = "xyz") -> "StructurePair":
        """A Structure as the two things that represent it: the coordinate
        document, and the sidecar payload beside it.

        THE ONE PLACE either is produced.  :meth:`write` puts this on disk and
        :meth:`files` hands it over as named bytes (which is what the export
        route returns) -- so a structure saved to a project and the same
        structure downloaded cannot differ.

        ``keep_sidecar`` is False when the metadata is all default -- a plain
        molecule with no cell, labels, frozen atoms or annotations.  Then the
        pair is the document alone and a stale sidecar beside it is removed, so
        "no .json" always means "no metadata" (:meth:`load` reads it that way).
        """
        # PLAIN XYZ, A BLOCK PER FRAME the structure holds (`model/structure.md`
        # § 2.2e), each comment line the title and nothing else: the comment
        # line is never metadata (§ 2.3), so the cell has one home, the
        # sidecar, and nothing here writes a second copy of it.
        #
        # THE SUFFIX IS DECIDED HERE, WITH THE FORMAT, and travels with the
        # pair.  Deriving it anywhere else is deriving it a second time, and a
        # second derivation is a chance to disagree with the bytes.
        #
        # THE SIDECAR IS BUILT ONCE.  The labels and the cell are the
        # structure's shared identity -- the same for frame 0 and frame 400 --
        # so there is one .json beside a frame set, not one per frame.  Its
        # hash pins it to the document actually written.
        # WHICH CONTAINER.  `fmt` is the format the DESTINATION names, not a
        # preference: `write` reads it off the target's suffix and `read`
        # dispatches the same way.
        if fmt not in ("xyz", "pdb"):
            raise ValueError(
                f"StructureCodec.pair: unsupported format {fmt!r}; "
                f"expected 'xyz' or 'pdb'")
        if fmt == "pdb":
            if struct.n_frames > 1:
                raise ValueError(
                    "StructureCodec.pair: PDB holds one geometry, and this is "
                    "a frame set. Write the set to a .xyz.")
            document = struct.to_pdb()
        else:
            document = struct.to_xyz()
        meta = struct.metadata_to_dict()
        # The REAL identity columns ride the sidecar (schema 8, 2026-08-20):
        # additive "extra" -- an xyz-born structure's synthesized placeholders
        # come back empty here and the sidecar is unchanged; a PDB-born
        # residue identity stops being erased by a save.  And real identity is
        # by itself a reason for the sidecar to EXIST: it is exactly the
        # "something to say" the keep rule asks about.
        identity = struct.identity_to_dict()
        payload = molstruct.to_dict(
            meta,
            identity       = identity,
            n_atoms_total  = struct.n_atoms,
            n_frames_total = struct.n_frames,
            structure_hash = _sha256_bytes(document.encode("utf-8")),
            info           = dict(struct.info) if struct.info else None,
        )
        return StructurePair(document=document, sidecar=payload,
                             keep_sidecar=(not _metadata_is_default(meta)
                                           or bool(identity)
                                           or bool(struct.info)),
                             suffix=(".pdb" if fmt == "pdb"
                                     else self.GEOMETRY_SUFFIX))

    # ---- the pair as NAMED bytes: <stem>.xyz + <stem>.molstruct.json -- #
    def files(self, struct: Structure, target) -> List[Tuple[Path, bytes]]:
        """The pair as bytes, WITH THE NAMES THEY BELONG UNDER -- what
        :meth:`write` writes, without writing it, and what the export door
        answers with.

        THE SUFFIX IS THE ONE :meth:`pair` CHOSE, not the one the caller
        guessed -- the pairing rule is the codec's.  Hand this a bare stem and it
        comes back named correctly; hand it a full ``<stem>.xyz`` and the suffix
        is corrected in place, which is why comparing this against :meth:`write`
        still compares the same paths.

        Contrast :meth:`write`, which does NOT correct the name: a project save
        was given an exact path through a picker with an overwrite gate on it,
        so the bytes go exactly there.  An export was given a stem and nothing
        else.  Different questions (model/structure.md § 2.4).
        """
        target = Path(target)
        made = self.pair(struct)
        if target.suffix.lower() in self._REPLACEABLE_SUFFIXES:
            target = target.with_suffix(made.suffix)
        else:
            target = target.with_name(target.name + made.suffix)
        out = [(target, made.document.encode("utf-8"))]
        if made.keep_sidecar:
            out.append((molstruct.sidecar_path_for(target),
                        molstruct.dumps(made.sidecar).encode("utf-8")))
        return out

    def source_files(self, struct: Structure,
                     label: str) -> List[Tuple[Path, bytes]]:
        """The structure a calculation is OF, as named bytes:
        ``<label>.source.xyz`` and its sidecar -- the catalogue's name
        (`runfiles.WRITTEN`, `job-contracts.md` § 6.3), for :meth:`files` to
        give the format's suffix.  Both writers of a calculation's structure
        ask it: the hand-over and `jobset init`."""
        from .runfiles import compose
        return self.files(struct, compose(label, ".source.xyz"))

    # ---- write the pair to disk, atomically -------------------------- #
    def write(self, struct: Structure, target, *, atomic: bool = True,
              fmt: "str | None" = None) -> Path:
        """Write ``struct`` to the ``<stem>.xyz`` + ``<stem>.molstruct.json``
        pair on disk and return the geometry path.  THE paired-file door
        (``model/structure.md`` § 2.4): owns the pairing rule + the
        both-or-neither atomicity so no caller re-derives the sidecar path or
        re-implements the write order.

        The target is written VERBATIM -- unlike :meth:`files`, this does not
        correct the suffix, because the caller did not guess it: a save names an
        exact path, chosen through a picker and cleared by an overwrite gate,
        and silently writing somewhere else would make that gate a lie.

        Atomicity: each half is staged to a temp sibling and ``os.replace``-d
        (per-file atomic).  The geometry is swapped first, then the sidecar, so
        the only visible interleaving is OLD-sidecar + NEW-geometry for a tiny
        window -- never a torn file.  When ``struct`` carries no metadata worth
        persisting AND a stale sidecar exists, it is removed so the pair can't
        disagree (``no .json == empty metadata``, matching :meth:`load`)."""
        target = Path(target)
        # `fmt` names the container.  Default: read it off the name the caller
        # chose, because `read` dispatches on that same suffix -- which is what
        # makes write->read a round trip rather than a coincidence.  A caller
        # with its own answer (the CLI's `--output-format`, which may name a
        # format the extension does not) passes it and is obeyed.
        made = self.pair(struct,
                         fmt=(fmt or ("pdb" if target.suffix.lower() == ".pdb"
                                      else "xyz")))   # the ONE generator
        return self._write_pair(target, made.document,
                                made.sidecar if made.keep_sidecar else None,
                                atomic=atomic)

    def write_moved(self, target, elements: Sequence[str], positions,
                    sidecar: dict, *, comment: str) -> Path:
        """The pair for a structure whose atoms MOVED, written where only the
        new coordinates are at hand -- inside a PySCF run, whose script
        imports this codec from ``mb_pyscf.pyz`` (`engines/pyscf.md` § 3) and
        calls this for every geometry it saves.

        ``elements`` and ``positions`` (Å) become the document through
        :meth:`Structure.to_xyz`, the codec's own text, with ``comment`` as
        its comment line.  ``sidecar`` is the payload :meth:`pair` made for
        the structure before it moved -- its labels, cell and periodicity
        are the ones the run was given, never derived a second time -- and
        only its envelope is renewed: ``structure_hash`` pinned to the new
        document, ``created_at`` the moment it is written.
        It is always written: it states the engine's origin
        (``engine_offset`` 0), which a document alone cannot.  Written
        through :meth:`write`'s own path: same order, same atomicity.
        """
        document = Structure(elements=list(elements),
                             positions=positions).to_xyz(comment=comment)
        # THE ENVELOPE IS OF THIS WRITE: the hash pinned to the document
        # written, and the time it was written -- `pair` stamped the moment
        # the payload was made, which for a run's geometry is render time.
        payload = dict(sidecar,
                       structure_hash=_sha256_bytes(document.encode("utf-8")),
                       created_at=molstruct._now_iso_z())
        return self._write_pair(Path(target), document, payload)

    @staticmethod
    def _write_pair(target: Path, document: str, sidecar: Optional[dict], *,
                    atomic: bool = True) -> Path:
        """Both halves to disk, in :meth:`write`'s order: the document first,
        then the sidecar, or -- with ``sidecar`` None -- the removal of a
        stale one, so the pair cannot disagree."""
        target.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path = molstruct.sidecar_path_for(target)

        if atomic:
            # THE SIDECAR'S RULE (`molstruct.save`), for both halves: fsync
            # where the filesystem takes it -- tmpfs on some kernels refuses,
            # and the bytes still land before the rename -- and no temp file
            # left behind when the write fails.  This path runs inside jobs
            # too (`write_moved`, the PySCF script's every saved geometry).
            tmp = target.with_suffix(target.suffix + ".tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(document)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                os.replace(tmp, target)
            finally:
                if tmp.exists():
                    tmp.unlink()
        else:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write(document)

        if sidecar is not None:
            molstruct.save(sidecar_path, sidecar)  # tempfile + os.replace
        elif sidecar_path.exists():
            sidecar_path.unlink()
        return target

    # ---- read the pair from disk (alias of load, symmetric name) ----- #
    def read(self, source_path, *,
             frame: "int | None" = None,
             frames: bool = False,
             retired_out: "dict | None" = None,
             said_out: "list | None" = None) -> Structure:
        """Symmetric read-side name for :meth:`load` -- parse the geometry +
        apply its paired sidecar into a Structure (missing sidecar => empty
        metadata, not an error -- except an engine's own file where its
        run is recorded, which reads with that run's frame; see `load`).
        ``frame`` / ``frames`` choose which frame (:meth:`frame_choice`);
        ``retired_out`` collects the retired keys it carried, ``said_out``
        what it says about a lone file."""
        return self.load(source_path, frame=frame, frames=frames,
                         retired_out=retired_out, said_out=said_out)

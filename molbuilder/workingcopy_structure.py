"""Structure + sidecar codec (``StructureCodec``).

MODULE: the standalone L2 codec for the ``<stem>.xyz`` (coordinates) +
``<stem>.molstruct.json`` (labels/annotations) file pair.  It owns FOUR things
nothing else in the system may hold a second copy of:

  1. the PAIRING RULE -- how the sidecar's name follows the geometry's;
  2. the FORMAT CHOICE -- plain ``.xyz``, a block per frame the structure holds,
     decided by the count and never asked as a separate question;
  3. the SIDECAR ENVELOPE -- ``schema_version``, the ``geometry_hash`` pinning
     it to its geometry (made on every write, compared on every read --
     structure-molstruct.md § 3), and the one serialisation
     (``molstruct.dumps``);
  4. the INVARIANTS -- every pair it writes has its sidecar, both-or-neither
     atomicity on write, and is in your order, an engine's record alone
     excepted (``write(..., engine_record=True)``, model/overview.md § 2.2a).
     NOT a periodicity gate on read: reading does not judge
     (structure-periodicity.md § 8.2, 2026-08-03).

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


import os
from pathlib import Path
from typing import List, NamedTuple, Sequence, Tuple

# TWO WAYS, because this module travels: the PySCF script writes every
# geometry it saves through it (`write_moved`), imported from `mb_pyscf.pyz`
# beside the job (`runwrap.PYSCF_COMPANIONS`), where the package is not
# installed.  What that call reaches imports only the standard library and
# numpy; the rest of the codec reads through the package.
try:                                        # inside molbuilder
    from .structure import REMAPPED_NOT_EDITED, Structure
    from .sidecars import molstruct
    from .structure_pair import GeometryChangedError
except ImportError:                         # beside a job, in mb_pyscf.pyz
    from structure import REMAPPED_NOT_EDITED, Structure
    import molstruct
    from structure_pair import GeometryChangedError


class StructurePair(NamedTuple):
    """What a Structure looks like outside memory: the coordinate document, the
    sidecar payload beside it, and the extension it belongs under.

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
    suffix: str


def lone_xyz_said(name: str, text: str) -> str:
    """What reading a LONE ``.xyz`` says to a person (``model/structure.md``
    § 2.3): atoms and coordinates only, and the comment line it did not read as
    metadata, quoted whole.  ``name`` is what the person knows it by -- the
    file's name, or the filename a pasted text came with -- and ``text`` its
    content.  One sentence for every door that opens one: the codec's read,
    and the load route's text."""
    lines = text.splitlines()
    comment = lines[1].strip() if len(lines) > 1 else ""
    return (f"{name} has no .molstruct.json beside it, so only its atoms and "
            f"coordinates were read -- no cell, no labels"
            + (f"; its comment line was not read as metadata: «{comment}»"
               if comment else "")
            + ".  Set what you meant on the Molbuilder tab, its Cell and "
            "Metadata pages; saving there writes it to the .molstruct.json.")


def geometry_changed_said(name: str) -> str:
    """What the read says of a pair whose sidecar's geometry hash is not the
    file's (``structure-molstruct.md`` § 3) -- the refusal of a door that
    builds from it, and the notice of the browser's load door, one sentence."""
    return (f"{name}'s atoms or cell are not the ones its .molstruct.json was "
            f"saved with -- the file changed after the pair was written, so "
            f"its labels and cell may sit on the wrong atoms.  Open it on the "
            f"Molbuilder tab, check its labels and cell, and save it there: "
            f"that writes the pair anew.")


#: WHAT A READ SAYS TO A PERSON, each by its name and at its level -- the
#: notices a read gives in ``said_out`` (:meth:`StructureCodec.load`), and
#: every door that says one of them asks :func:`read_notice`: a lone file,
#: atoms and coordinates only (``model/structure.md`` § 2.3); a pair whose
#: geometry changed (``structure-molstruct.md`` § 3); a run whose order
#: nothing states (``model/overview.md`` § 2.2a, I9); a structure put back
#: in your order (I1, I8).
LONE_FILE = "structure.lone_file"
GEOMETRY_CHANGED = "structure.geometry_changed"
ORDER_UNSTATED = "structure.order_unstated"
RESTORED = "structure.restored"
READ_NOTICES = {LONE_FILE: "info", GEOMETRY_CHANGED: "warn",
                ORDER_UNSTATED: "warn", RESTORED: "info"}


def read_notice(where: str, message: str):
    """The notice ``where`` (one of :data:`READ_NOTICES`) saying
    ``message``, at its level -- an ``issues.Issue``."""
    from .issues import Issue
    return Issue(READ_NOTICES[where], message, where)


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
             stated_out: "set | None" = None,
             said_out: "list | None" = None,
             opened: "tuple | frozenset" = (),
             in_your_order: bool = False) -> Structure:
        """Read the pair back into a Structure.

        WHICH FRAME (``model/structure.md`` § 2.3, *Which frame*): the pair is
        read whole and its sidecar applied to the whole set, then the answer is
        frame 0 -- what every caller has always had -- or frame ``frame``, or,
        with ``frames=True``, the whole set (:meth:`frame_choice`).  A
        one-frame file answers the same for all three.

        ``retired_out``: a caller that passes a dict gets each retired key the
        sidecar carried with a value -- read and ignored, and the load door
        says so (plan § 5q D14).

        ``stated_out``: a caller that passes a set gets every metadata field
        the sidecar states -- a null one among them -- so a reader can tell a
        field stated null from one written before the field existed
        (``model/overview.md`` § 2.2a, I9: the composed junction's order
        record, `transport.compose.recorded_junction`).

        ``said_out``: a caller that passes a list gets what the read says to a
        person about this file, each a notice at its level
        (:func:`read_notice`) -- a LONE ``.xyz`` (no sidecar, not an engine's
        own file in one of our runs) is atoms and coordinates only, and its
        comment line is quoted, never read as metadata (``model/structure.md``
        § 2.3), :data:`LONE_FILE`.  Every door that opens a file for a person
        says it.

        TWO STATES ARE REFUSED, and a door that only shows the structure
        names them in ``opened`` to open it instead, said in ``said_out``:
        THE PIN COMPARED (``structure-molstruct.md`` § 3) -- a sidecar whose
        geometry hash is not the file's, its atoms or cell changed after the
        pair was written, :class:`structure_pair.GeometryChangedError`,
        :data:`GEOMETRY_CHANGED` (a sidecar written before v13 states no
        hash, and its pair opens unchecked); and AN ENGINE'S OWN FILE OF A
        RUN WHOSE ORDER NOTHING STATES -- its deck written before the order
        was recorded, for a kind that reorders (``model/overview.md`` § 2.2a,
        I9) -- :class:`runs.OrderUnstated`, :data:`ORDER_UNSTATED`.

        ``in_your_order``: the read of an editor, a calculation's structure
        or a citation (I1, I3, I8) -- a structure in an engine's order is
        put back in yours (``Structure.in_your_order``; an engine's own file
        through its run's view, ``runs.RunView.in_your_order``), said
        (:data:`RESTORED`) to a caller that passes ``said_out``; one whose
        order nothing states cannot be, and is refused -- and asking to
        open it as well is refused, the two contradicting each other.
        """
        src = Path(source_path)
        # Parse the SOURCE in ITS OWN format (dispatch on the extension) -- the
        # file picker accepts .xyz AND .pdb, and each needs its own parser: a
        # .pdb read as XYZ chokes on its "HEADER ..." first line.  An unknown
        # extension is an EXPLICIT error, not a silent from_xyz attempt.
        suffix = src.suffix.lower()
        if suffix not in molstruct.GEOMETRY_SUFFIXES:
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
        fmt = self.format_named(src)
        if fmt == "pdb":
            struct = Structure.from_pdb(text)
        else:
            struct = Structure.from_xyz(text)
        if opened and said_out is None:
            raise ValueError("a read that opens what it would refuse says "
                             "so: pass said_out")
        if in_your_order and ORDER_UNSTATED in opened:
            raise ValueError("a read in your order cannot open a run whose "
                             "order nothing states: nothing can put it in "
                             "yours (I9)")
        restored_said = None
        sidecar_path = molstruct.sidecar_path_for(src)
        if sidecar_path.exists():
            payload = molstruct.load(sidecar_path, retired_out=retired_out,
                                     stated_out=stated_out)
            pinned = payload.get("geometry_hash")
            if pinned is not None and pinned != molstruct.geometry_hash(
                    text, fmt, payload["cell"], payload["axis_kind"]):
                if GEOMETRY_CHANGED not in opened:
                    raise GeometryChangedError(
                        geometry_changed_said(src.name))
                said_out.append(read_notice(GEOMETRY_CHANGED,
                                            geometry_changed_said(src.name)))
            molstruct.apply_to_structure(struct, payload)
        elif self._engine_own(src):
            # AN ENGINE'S OWN STRUCTURE FILE -- SIESTA's `<label>.xyz` in a run
            # folder -- has no sidecar, because molbuilder writes every
            # structure as a pair and the engine writes none.  It is read
            # through ITS RUN'S VIEW (`runs.view_of`), as every door that
            # builds a structure from a run reads it: the cell and axis kinds
            # that run's own deck recorded, and the engine's origin, a stated
            # 0 (`model/structure-periodicity.md` § 6.0: *"Every door that
            # makes a structure from an engine's output states it"*); the
            # labels of its deck, and the order record of a reordered run --
            # the file is in its engine's order (`model/overview.md` § 2.2a,
            # I5); what the run says about itself.
            from .runs import run_of, view_of
            view = view_of(run_of(src), n_atoms=struct.n_atoms)
            if in_your_order:
                # AN EDITOR'S OR A BUILDER'S READ crosses back through the
                # run's view, refused when nothing states its order (I9).
                struct, restored_said = view.in_your_order(struct, src.name)
            else:
                if view.order_unstated:
                    if ORDER_UNSTATED not in opened:
                        view.declared.require_order()
                    said_out.append(read_notice(ORDER_UNSTATED,
                                                view.order_unstated))
                struct = view.onto(struct)
        elif fmt == "xyz" and said_out is not None:
            # A LONE FILE: nothing molbuilder knows states its metadata, so it
            # is what `from_xyz` read -- atoms and coordinates -- and the read
            # says so, quoting the comment line it did not read as metadata
            # (`model/structure.md` § 2.3).
            said_out.append(read_notice(LONE_FILE,
                                        lone_xyz_said(src.name, text)))
        if in_your_order and restored_said is None:
            # A PAIR IN AN ENGINE'S ORDER -- an engine's record -- crosses
            # back too (`model/overview.md` § 2.2a, I1, I8).
            struct, restored_said = struct.in_your_order(src.name)
        if restored_said is not None and said_out is not None:
            # SAID to a door that tells a person what it read; a builder
            # reading a structure that is in your order by design (I1)
            # passes no list, and its restore is a guard.
            said_out.append(read_notice(RESTORED, restored_said))
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

        A PAIR IS IN YOUR ORDER (`model/overview.md` § 2.2a, I1): a remapped
        structure is refused, naming :meth:`Structure.restored`.  An engine's
        own record in its order has one door, :meth:`write`'s
        ``engine_record``.

        THE ONE PLACE either is produced.  :meth:`write` puts this on disk and
        :meth:`files` hands it over as named bytes (which is what the export
        route returns) -- so a structure saved to a project and the same
        structure downloaded cannot differ.

        THE SIDECAR IS ALWAYS PART OF IT, a plain molecule's with nothing to
        say included: it carries the pin, the geometry hash made over this
        document (``structure-molstruct.md`` § 3), which :meth:`load` compares.
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
        # geometry hash pins it to the document actually written.
        # WHICH CONTAINER.  `fmt` is the format the DESTINATION names, not a
        # preference: `write` reads it off the target's suffix and `read`
        # dispatches the same way.
        if struct.atom_order is not None:
            raise ValueError(f"a structure file is written in your order: "
                             f"{REMAPPED_NOT_EDITED}")
        return self._pair(struct, fmt=fmt)

    def _pair(self, struct: Structure, *, fmt: str) -> "StructurePair":
        """:meth:`pair`'s generator, in whatever order the structure is in --
        reached past the order's refusal by :meth:`write`'s ``engine_record``
        alone."""
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
        # residue identity stops being erased by a save.
        identity = struct.identity_to_dict()
        payload = molstruct.to_dict(
            meta,
            identity       = identity,
            n_atoms_total  = struct.n_atoms,
            n_frames_total = struct.n_frames,
            geometry_hash  = molstruct.geometry_hash(
                document, fmt, meta["cell"], meta["axis_kind"]),
            info           = dict(struct.info) if struct.info else None,
        )
        return StructurePair(document=document, sidecar=payload,
                             suffix=(".pdb" if fmt == "pdb"
                                     else self.GEOMETRY_SUFFIX))

    @staticmethod
    def format_named(target) -> str:
        """The container a destination names: ``pdb`` for a ``.pdb``, ``xyz``
        for every other name (``model/structure.md`` § 2.4, *the format
        choice*).  :meth:`load` dispatches on that same suffix -- which is
        what makes write->read a round trip rather than a coincidence."""
        return "pdb" if Path(target).suffix.lower() == ".pdb" else "xyz"

    # ---- the pair as NAMED bytes: <stem>.xyz + <stem>.molstruct.json -- #
    def files(self, struct: Structure, target, *,
              fmt: "str | None" = None) -> List[Tuple[Path, bytes]]:
        """The pair as bytes, WITH THE NAMES THEY BELONG UNDER -- what
        :meth:`write` writes, without writing it, and what the export door
        answers with.

        THE SUFFIX IS THE ONE :meth:`pair` CHOSE, not the one the caller
        guessed -- the pairing rule is the codec's.  Hand this a bare stem and it
        comes back named correctly; hand it a full ``<stem>.xyz`` and the suffix
        is corrected in place, which is why comparing this against :meth:`write`
        still compares the same paths.  ``fmt`` as :meth:`write` takes it: by
        default the format the target names (:meth:`format_named`), so a
        ``.pdb`` is a PDB, named so; a caller with its own answer passes it.

        Contrast :meth:`write`, which does NOT correct the name: a project save
        was given an exact path through a picker with an overwrite gate on it,
        so the bytes go exactly there.  An export was given a stem and nothing
        else.  Different questions (model/structure.md § 2.4).
        """
        target = Path(target)
        made = self.pair(struct, fmt=(fmt or self.format_named(target)))
        if (target.suffix.lower() in self._REPLACEABLE_SUFFIXES
                or target.suffix.lower() == made.suffix):
            target = target.with_suffix(made.suffix)
        else:
            target = target.with_name(target.name + made.suffix)
        return [(target, made.document.encode("utf-8")),
                (molstruct.sidecar_path_for(target),
                 molstruct.dumps(made.sidecar).encode("utf-8"))]

    @staticmethod
    def fingerprint(files: Sequence[Tuple[Path, bytes]]) -> str:
        """THE PAIR'S FINGERPRINT: a SHA-256 over its two files' bytes, in
        order -- :meth:`files`' answer, or the pair as it lies on disk --
        their stamps masked (`deck_record.without_stamps`, the sidecar's
        ``created_at``).  Every atom, coordinate, label, channel, named value
        and the box are in it, so any edit to the structure changes it.  The
        description records it of the structure a calculation is OF
        (:meth:`source_files`), and every prep compares it
        (`engines/stages.md` § 6.3)."""
        import hashlib

        from .deck_record import without_stamps
        digest = hashlib.sha256()
        for _path, data in files:
            digest.update(without_stamps(data))
        return digest.hexdigest()

    @classmethod
    def fingerprint_on_disk(cls, path) -> str:
        """:meth:`fingerprint` of the pair at ``path`` -- its geometry file
        and the ``.molstruct.json`` beside it, as they lie on disk."""
        geometry = Path(path)
        return cls.fingerprint([(p, p.read_bytes()) for p in (
            geometry, molstruct.sidecar_path_for(geometry))])

    def source_files(self, struct: Structure,
                     label: str) -> List[Tuple[Path, bytes]]:
        """The structure a calculation is OF, as named bytes:
        ``<label>.source.xyz`` and its sidecar -- the catalogue's name
        (`runfiles.WRITTEN`, `job-contracts.md` § 6.3), for :meth:`files` to
        give the format's suffix.  Both writers of a calculation's structure
        ask it: the hand-over and `molbuilder describe`.  Written AS GIVEN, its
        origin's record with it: prep decides how the calculation takes it
        (`cell.entering`, `jobset.prep._structure_for`)."""
        from .runfiles import compose
        return self.files(struct, compose(label, ".source.xyz"))

    # ---- write the pair to disk, atomically -------------------------- #
    def write(self, struct: Structure, target, *,
              fmt: "str | None" = None,
              engine_record: bool = False) -> Path:
        """Write ``struct`` to the ``<stem>.xyz`` + ``<stem>.molstruct.json``
        pair on disk and return the geometry path.  THE paired-file door
        (``model/structure.md`` § 2.4): owns the pairing rule + the
        both-or-neither atomicity so no caller re-derives the sidecar path or
        re-implements the write order.

        The target is written VERBATIM -- unlike :meth:`files`, this does not
        correct the suffix, because the caller did not guess it: a save names an
        exact path, chosen through a picker and cleared by an overwrite gate,
        and silently writing somewhere else would make that gate a lie.

        Both or neither (:meth:`_write_pair`): both halves rendered, then
        both staged, then both put in place -- a sidecar that cannot be
        written stops the write before anything is on disk.

        ``engine_record``: THE ONE DOOR for an engine's record in its own
        order -- the transport calculation's composed junction, written by
        the composer alone, its order record in the sidecar beside it
        (`model/overview.md` § 2.2a).  Every other write is in your order."""
        target = Path(target)
        # `fmt` names the container.  Default: the one the name the caller
        # chose names (`format_named`).  A caller with its own answer (the
        # CLI's `--output-format`, which may name a format the extension does
        # not) passes it and is obeyed.
        fmt = fmt or self.format_named(target)
        made = (self._pair(struct, fmt=fmt) if engine_record
                else self.pair(struct, fmt=fmt))
        return self._write_pair(target, made.document, made.sidecar)

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
        only its envelope is renewed: ``geometry_hash`` made over the new
        document, ``created_at`` the moment it is written.
        It is always written: it states the engine's origin
        (``engine_offset`` 0), which a document alone cannot.  Written
        through :meth:`write`'s own path: same order, same atomicity.
        """
        document = Structure(elements=list(elements),
                             positions=positions).to_xyz(comment=comment)
        # THE ENVELOPE IS OF THIS WRITE: the geometry hash made over the
        # document written, with the cell the payload states, and the time it
        # was written -- `pair` stamped the moment the payload was made,
        # which for a run's geometry is render time.
        payload = dict(sidecar,
                       geometry_hash=molstruct.geometry_hash(
                           document, "xyz", sidecar["cell"],
                           sidecar["axis_kind"]),
                       created_at=molstruct._now_iso_z())
        return self._write_pair(Path(target), document, payload)

    @staticmethod
    def _write_pair(target: Path, document: str, sidecar: dict) -> Path:
        """Both halves to disk as one unit (`model/structure.md` § 2.4).

        BOTH RENDERED FIRST: the sidecar is serialised by its one
        serialiser (`molstruct.dumps`) before anything is written, so one
        that cannot be -- a NaN in ``info`` -- stops the write with the pair
        on disk as it was.  THEN BOTH STAGED, each to a temp sibling, synced
        where the filesystem takes it -- tmpfs on some kernels refuses, and
        the bytes still land before the rename -- and only then both put in
        place, the document first, then the sidecar: a failure staging
        either leaves the old pair, and no temp file is left behind.  This
        path runs inside jobs too (`write_moved`, the PySCF script's every
        saved geometry)."""
        target.parent.mkdir(parents=True, exist_ok=True)
        halves = [(target, document),
                  (molstruct.sidecar_path_for(target), molstruct.dumps(sidecar))]
        staged = [(path.with_suffix(path.suffix + ".tmp"), path, text)
                  for path, text in halves]
        try:
            for tmp, _path, text in staged:
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(text)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
            for tmp, path, _text in staged:
                os.replace(tmp, path)
        finally:
            for tmp, _path, _text in staged:
                if tmp.exists():
                    tmp.unlink()
        return target

    # ---- read the pair from disk (alias of load, symmetric name) ----- #
    def read(self, source_path, *,
             frame: "int | None" = None,
             frames: bool = False,
             retired_out: "dict | None" = None,
             stated_out: "set | None" = None,
             said_out: "list | None" = None,
             opened: "tuple | frozenset" = (),
             in_your_order: bool = False) -> Structure:
        """Symmetric read-side name for :meth:`load` -- parse the geometry +
        apply its paired sidecar into a Structure (missing sidecar => empty
        metadata, not an error -- except an engine's own file where its
        run is recorded, which reads with that run's frame; see `load`).
        ``frame`` / ``frames`` choose which frame (:meth:`frame_choice`);
        ``retired_out`` collects the retired keys it carried, ``stated_out``
        the metadata fields it states, ``said_out`` what the read says,
        ``opened`` what a viewer opens rather than refuses,
        ``in_your_order`` the read of an editor or a builder (see `load`)."""
        return self.load(source_path, frame=frame, frames=frames,
                         retired_out=retired_out, stated_out=stated_out,
                         said_out=said_out, opened=opened,
                         in_your_order=in_your_order)

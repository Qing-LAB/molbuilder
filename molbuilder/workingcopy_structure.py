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
     atomicity on write, and is in the user order, an engine's record alone
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
from typing import List, NamedTuple, Optional, Sequence, Tuple

# TWO WAYS, because this module travels: the PySCF script writes every
# geometry it saves through it (`write_moved`), imported from `mb_pyscf.pyz`
# beside the job (`runwrap.PYSCF_COMPANIONS`), where the package is not
# installed.  What that call reaches imports only the standard library and
# numpy; the rest of the codec reads through the package.
try:                                        # inside molbuilder
    from .atom_permutation import (LISTED_NOT_STATED, MAP_KEY, AtomIndexMap,
                                   Permutation)
    from .structure import REMAPPED_NOT_EDITED, Structure
    from .sidecars import molstruct
    from .structure_pair import GeometryChangedError
except ImportError:                         # beside a job, in mb_pyscf.pyz
    from atom_permutation import (  # type: ignore[no-redef]
        LISTED_NOT_STATED, MAP_KEY, AtomIndexMap, Permutation)
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
#: in the user order (I1, I8); a pair that states no atom order, read as
#: listed by user index (§ 2.2b, F4) -- which refuses nothing.
LONE_FILE = "structure.lone_file"
GEOMETRY_CHANGED = "structure.geometry_changed"
ORDER_UNSTATED = "structure.order_unstated"
RESTORED = "structure.restored"
ORDER_NOT_STATED = "structure.order_not_stated"
READ_NOTICES = {LONE_FILE: "info", GEOMETRY_CHANGED: "warn",
                ORDER_UNSTATED: "warn", RESTORED: "info",
                ORDER_NOT_STATED: "warn"}


def order_not_stated_said(name: str, why: str = "") -> str:
    """What a read says of a pair that states no atom order
    (`model/overview.md` § 2.2b, F4) -- one saved before 2026-10-14, or one
    whose map states ``not_stated`` and ``why``."""
    known = (f"{name} states its atom order is not known -- {why}: read "
             f"as listed by user index." if why else
             f"{name} states no atom order (saved before 2026-10-14): read "
             f"as listed by user index. A file saved then from a view of a "
             f"sorted run may be in the engine's order.")
    return (f"{known} If you know it is listed by user index -- the order "
            f"you built it in -- saving it again on the Molbuilder tab "
            f"records that, and a calculation described from it is then "
            f"described again from the saved pair.")


#: WHY A PAIR SAVED FROM A VIEW OF A RUN WHOSE ORDER NOTHING STATES IS
#: `not_stated` (`model/overview.md` § 2.2b, F4) -- its map's own sentence.
VIEW_ORDER_UNSTATED_WHY = (
    "saved from a view of a structure whose order nothing states -- a run's "
    "deck or a composed junction's sidecar written before 2026-10-14: its "
    "atoms may be listed by engine index")


def pair_order_unstated_said(sidecar_name: str) -> str:
    """What every door says of a pair the catalogue lists by engine index --
    a calculation's composed junction -- whose sidecar states no order
    record (`model/overview.md` § 2.2a, I9): one sentence, the junction's
    one read (`transport.compose.recorded_junction`) and `atom_index`
    alike."""
    return (f"{sidecar_name} states no order record -- it was written "
            f"before the field -- so its atoms stand in the engine's order "
            f"with nothing stating the way back to the user order")


def pair_atom_index_map(geometry: Path, payload: dict, *,
                        listing: "Optional[bool]", states_order: bool
                        ) -> "Optional[AtomIndexMap]":
    """WHAT A PAIR STATES OF ITS ATOMS' ORDER (`model/overview.md` § 2.2b,
    F4) -- the one answer for the codec's read, `atom_index` and prep: its
    sidecar's own map (v15, read and checked by the sidecar's reader);
    else, a sidecar that ``states_order`` (v14: its ``atom_order`` field,
    known by what the file states), made from its order record -- listed by
    engine index where it carries one or the catalogue lists the pair by
    engine index (``listing`` not ``None``, :func:`pair_engine_listing`: an
    engine's record whose sort moved nothing); else, a v7-v13 sidecar
    stating no order: the user's pair read as listed by user index with
    F4's warning, an engine's record of a kind that sorts nothing
    (``listing`` ``False`` -- a PySCF run's) the identity, and one of a
    kind that may sort ``None``: nothing states its order (I9,
    :func:`pair_order_unstated_said`)."""
    n = int(payload["n_atoms_total"])
    if MAP_KEY in payload:
        atoms = AtomIndexMap.from_dict(payload[MAP_KEY], n_atoms=n)
        if atoms.listed_by == LISTED_NOT_STATED:
            atoms = AtomIndexMap.assumed_user(
                n, order_not_stated_said(geometry.name, atoms.said))
        return atoms
    if states_order:
        record = payload.get("atom_order")
        record = (Permutation.from_dict(record, n_atoms=n) if record
                  else None)
        if record is not None or listing is not None:
            return AtomIndexMap.of_engine_record(record, n)
        return AtomIndexMap.of_user_structure(n)
    if listing is None:
        return AtomIndexMap.assumed_user(
            n, order_not_stated_said(geometry.name))
    return None if listing else AtomIndexMap.of_engine_record(None, n)


def pair_engine_listing(sidecar: Path) -> "Optional[bool]":
    """WHETHER THE CATALOGUE LISTS THIS PAIR BY ENGINE INDEX -- a
    calculation's composed junction, a PySCF run's ``_initial`` and
    ``_optimized`` pairs (`runfiles.listed_by_of`, F3) -- and if so whether
    its kind may be sorted: ``None`` for the user's pair (a pair in no
    calculation of ours is the user's, whatever its name); ``True`` where
    the one sort table sorts the stage it belongs to
    (`transport.sort.deck_sort_key` -- the composed junction), and where the
    calculation's description does not read, so nothing says; ``False`` for
    a kind that sorts nothing (a PySCF run's)."""
    try:                                    # inside molbuilder
        from .atom_permutation import LISTED_BY_ENGINE
        from .runfiles import listed_by_of
        from .runs import place_of, run_of
        from .transport.sort import deck_sort_key
    except ImportError:                     # beside a job: no calculation
        return None
    if listed_by_of(sidecar.name) != LISTED_BY_ENGINE:
        return None
    place = place_of(sidecar.parent)
    if place.root is None:
        return None
    if place.task is None:
        return True
    run = run_of(sidecar.parent)
    return deck_sort_key(place.task.engine, place.task.calculation,
                         run.stage if run is not None else None) is not None


#: THE ENGINE RECORD'S WORDS in its `.xyz` comment line, after the title
#: (§ 2.2b's files table): said for a tool that reads the `.xyz` alone, and
#: never part of the title -- a structure restored from the record carries
#: its title, not these words, into a file in the user order.
_ENGINE_RECORD_SAYS = " atoms listed by engine index ("


def engine_record_comment(title: str, words: str, sidecar_name: str) -> str:
    """An engine record's `.xyz` comment line: its ``title``, then which
    index its atoms are listed by, in ``words``, and where each atom's user
    index is -- the one spelling :func:`title_of_comment` takes back off."""
    title = (title or "").strip()
    return ((title + " --") if title else "--") + (
        f"{_ENGINE_RECORD_SAYS}{words}); each atom's user index is in "
        f"{sidecar_name}")


def title_of_comment(line: str) -> str:
    """A `.xyz` comment line's title -- the line without an engine record's
    words (:func:`engine_record_comment`)."""
    at = (line or "").find(" --" + _ENGINE_RECORD_SAYS)
    if at < 0 and (line or "").startswith("--" + _ENGINE_RECORD_SAYS):
        at = 0
    return line if at < 0 else line[:at]


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
             in_user_order: bool = False,
             a_copy: bool = False) -> Structure:
        """Read the pair back into a Structure.

        ``a_copy``: the read of a door that copies the file for a
        calculation without the person's save (`describe`'s source pair) --
        a pair that states no atom order comes back marked
        ``order_unstated``, so the copy says ``not_stated`` too; a person's
        own save from an editor records the user index (`model/overview.md`
        § 2.2b, F4: nothing molbuilder writes claims what it does not
        know).

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

        ``in_user_order``: the read of an editor, a calculation's structure
        or a citation (I1, I3, I8) -- a structure in an engine's order is
        put back in the user order (``Structure.in_user_order``; an engine's own file
        through its run's view, ``runs.RunView.in_user_order``), said
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
            # THE TITLE, never an engine record's words about its order.
            struct.title = title_of_comment(struct.title)
        if opened and said_out is None:
            raise ValueError("a read that opens what it would refuse says "
                             "so: pass said_out")
        if in_user_order and ORDER_UNSTATED in opened:
            raise ValueError("a read in the user order cannot open a run whose "
                             "order nothing states: nothing can put it in "
                             "the user order (I9)")
        restored_said = None
        sidecar_path = molstruct.sidecar_path_for(src)
        if sidecar_path.exists():
            # WHAT THE FILE STATES, by its fields -- the order record's
            # among them (`model/overview.md` § 2.2a, I9: known by what the
            # file states, never its version).
            stated = stated_out if stated_out is not None else set()
            payload = molstruct.load(sidecar_path, retired_out=retired_out,
                                     stated_out=stated)
            pinned = payload.get("geometry_hash")
            if pinned is not None and pinned != molstruct.geometry_hash(
                    text, fmt, payload["cell"], payload["axis_kind"]):
                if GEOMETRY_CHANGED not in opened:
                    raise GeometryChangedError(
                        geometry_changed_said(src.name))
                said_out.append(read_notice(GEOMETRY_CHANGED,
                                            geometry_changed_said(src.name)))
            try:
                molstruct.apply_to_structure(struct, payload)
            except molstruct.MolstructJsonError as exc:
                # THE FILE THAT SAID IT, named (a map against its record) --
                # the refusal's own kind kept (a pairing refusal among them).
                raise type(exc)(f"{sidecar_path.name}: {exc}") from exc
            # WHAT THE PAIR STATES OF ITS ORDER (§ 2.2b, F4): one that states
            # none -- saved before 2026-10-14, or from a view of a run whose
            # order nothing states -- is read as listed by user index, and
            # said; an engine's record that states none of a kind that may
            # sort (I9) is refused to an editor and a builder, and opened
            # for a viewer marked, and said.
            atoms = pair_atom_index_map(
                src, payload, listing=pair_engine_listing(sidecar_path),
                states_order="atom_order" in stated)
            if atoms is None:
                said = pair_order_unstated_said(sidecar_path.name)
                if in_user_order or ORDER_UNSTATED not in opened:
                    raise ValueError(said)
                struct.order_unstated = True
                said_out.append(read_notice(ORDER_UNSTATED, said))
            elif atoms.listed_by == LISTED_NOT_STATED:
                if said_out is not None:
                    said_out.append(read_notice(ORDER_NOT_STATED, atoms.said))
                if a_copy:
                    struct.order_unstated = True
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
            if in_user_order:
                # AN EDITOR'S OR A BUILDER'S READ crosses back through the
                # run's view, refused when nothing states its order (I9).
                struct, restored_said = view.in_user_order(struct, src.name)
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
        if in_user_order and restored_said is None:
            # A PAIR IN AN ENGINE'S ORDER -- an engine's record -- crosses
            # back too (`model/overview.md` § 2.2a, I1, I8).
            struct, restored_said = struct.in_user_order(src.name)
        if restored_said is not None and said_out is not None:
            # SAID to a door that tells a person what it read; a builder
            # reading a structure that is in the user order by design (I1)
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

        A PAIR IS IN THE USER ORDER (`model/overview.md` § 2.2a, I1): a remapped
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
            raise ValueError(f"a structure file is written in the user order: "
                             f"{REMAPPED_NOT_EDITED}")
        # THE USER'S STRUCTURE, given to no engine (`model/overview.md`
        # § 2.2b, F2): listed by user index, the user column alone -- or,
        # from a view of a run whose order nothing states, `not_stated`
        # (F4: nothing molbuilder writes claims what it does not know).
        atoms = (AtomIndexMap.assumed_user(struct.n_atoms,
                                           VIEW_ORDER_UNSTATED_WHY)
                 if struct.order_unstated
                 else AtomIndexMap.of_user_structure(struct.n_atoms))
        return self._pair(struct, fmt=fmt, atoms=atoms)

    def _pair(self, struct: Structure, *, fmt: str, atoms,
              comment: str = "") -> "StructurePair":
        """:meth:`pair`'s generator, in whatever order the structure is in --
        reached past the order's refusal by :meth:`write`'s ``engine_record``
        alone; ``atoms``, the atom index map the sidecar states, the door's
        (`model/overview.md` § 2.2b, F2), and ``comment`` the xyz's comment
        line where the door says one."""
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
            document = struct.to_xyz(comment=comment)
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
            atom_index_map = atoms.to_dict(),
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
        (`model/overview.md` § 2.2a).  Every other write is in the user order."""
        target = Path(target)
        # `fmt` names the container.  Default: the one the name the caller
        # chose names (`format_named`).  A caller with its own answer (the
        # CLI's `--output-format`, which may name a format the extension does
        # not) passes it and is obeyed.
        fmt = fmt or self.format_named(target)
        if engine_record:
            # AN ENGINE'S RECORD (`model/overview.md` § 2.2b, F2): listed by
            # engine index, both columns from its order record -- and its
            # `.xyz` says so in words, for a tool that reads it alone.
            words = (struct.atom_order.words if struct.atom_order is not None
                     else "the sort moved nothing")
            made = self._pair(
                struct, fmt=fmt,
                atoms=AtomIndexMap.of_engine_record(struct.atom_order,
                                                    struct.n_atoms),
                comment=engine_record_comment(
                    struct.title, words,
                    molstruct.sidecar_path_for(target).name))
        else:
            made = self.pair(struct, fmt=fmt)
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
        # A RUN'S OWN PAIR (`model/overview.md` § 2.2b, F2): listed by
        # engine index -- the user order, which PySCF keeps.
        payload = dict(sidecar,
                       geometry_hash=molstruct.geometry_hash(
                           document, "xyz", sidecar["cell"],
                           sidecar["axis_kind"]),
                       created_at=molstruct._now_iso_z(),
                       **{MAP_KEY: AtomIndexMap.of_engine_record(
                           None, len(elements)).to_dict()})
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
             in_user_order: bool = False) -> Structure:
        """Symmetric read-side name for :meth:`load` -- parse the geometry +
        apply its paired sidecar into a Structure (missing sidecar => empty
        metadata, not an error -- except an engine's own file where its
        run is recorded, which reads with that run's frame; see `load`).
        ``frame`` / ``frames`` choose which frame (:meth:`frame_choice`);
        ``retired_out`` collects the retired keys it carried, ``stated_out``
        the metadata fields it states, ``said_out`` what the read says,
        ``opened`` what a viewer opens rather than refuses,
        ``in_user_order`` the read of an editor or a builder (see `load`)."""
        return self.load(source_path, frame=frame, frames=frames,
                         retired_out=retired_out, stated_out=stated_out,
                         said_out=said_out, opened=opened,
                         in_user_order=in_user_order)

"""WHICH INDEX A FILE'S ATOMS ARE LISTED BY -- the one reader of the atom
index map, for any file or run folder (`model/overview.md` § 2.2b).

MODULE  atom_index (floor 2: it reads a run's deck through the run door)
ROLE    :func:`atom_index_map` -- the map a file states of itself (F1), its
        sidecar's (F4), its deck's, or its run's deck's (F3), with
        ``stated_by`` and ``said``; molbuilder's own doors and a person's
        notebook alike (`molbuilder.postprocess`), and ``molbuilder
        atom-index``
USED-BY postprocess, cli (``atom-index``)

**Never a guess**: a file is answered by what states its atoms' indices --
the file itself, the sidecar beside it, a deck's own block, the deck of the
run that holds it, the catalogue's row for a kind written before the file
said so -- or refused by name (:class:`AtomIndexUnknown`).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .atom_permutation import (FIELD_ENGINE, FIELD_LISTED_BY, FIELD_USER,
                               LISTED_BY_USER, MAP_KEY, AtomIndexError,
                               AtomIndexMap)


class AtomIndexUnknown(ValueError):
    """Nothing states a file's atoms' indices -- the message says what was
    looked for, ready to surface."""


#: How many run folders a refusal of a folder names before it stops.
_RUNS_NAMED = 8


def atom_index_map(path) -> AtomIndexMap:
    """THE ATOM INDEX MAP of ``path`` (`model/overview.md` § 2.2b): a file
    molbuilder wrote (its own fields, F1), a structure pair (its sidecar,
    F4), a deck (its own block), an engine's file or a run folder (its run's
    deck, F3).  A folder that is no run is refused, naming the run folders
    it holds; a lead's file in another rung's run, naming its rung (I6).
    An engine's input says its own text counts as its engine does (F5)."""
    p = Path(path)
    if p.is_dir():
        return _of_folder(p)
    if not p.is_file():
        raise FileNotFoundError(f"{p} is not there")
    atoms = _stated(p)
    if atoms is None:
        atoms = _of_run(p, p.parent)
    return _counted(p, atoms)


def map_of_block(block, source: str) -> AtomIndexMap:
    """A JSON record's map block (``atom_index_map``), read by the map's one
    reader -- its atom count its columns'."""
    n = len(block.get(FIELD_USER) or block.get(FIELD_ENGINE) or ())
    return AtomIndexMap.from_dict(block, n_atoms=n, source=source)


def _counted(p: Path, atoms: AtomIndexMap) -> AtomIndexMap:
    """``atoms``, with what an engine's input's own text counts by said
    beside it (`runfiles.counted_by_of`, F5) -- the map counts from 0."""
    from .engine_atom_index import convert_to_engine_numbering
    from .runfiles import counted_by_of
    engines = counted_by_of(p.name)
    if not engines:
        return atoms
    words = (f"{p.name} is an engine's input: its own text numbers the atoms "
             + " and ".join(f"as {e} counts, from "
                            f"{convert_to_engine_numbering(0, e)}"
                            for e in engines)
             + " (model/overview.md 2.2b, F5)")
    return replace(atoms, said=f"{atoms.said}; {words}" if atoms.said
                   else words)


def _point_folder(folder: Path) -> Path:
    """The folder a file is read with: a point's ``shared/`` folder -- at
    any depth, a share of a share -- is the point's (F3)."""
    from .transport.energies import SHARED_DIR
    while folder.name == SHARED_DIR:
        folder = folder.parent
    return folder


def _of_folder(folder: Path) -> AtomIndexMap:
    """A run folder's map -- its deck's; a folder that is no run is
    refused, naming the run folders it holds; a point's ``shared/`` folder
    is the point's."""
    from .runs import atom_index_map_of, run_of
    run = run_of(_point_folder(folder))
    if run is not None:
        return atom_index_map_of(run)
    runs = _runs_under(folder)
    named = ", ".join(str(r.relative_to(folder)) for r in runs[:_RUNS_NAMED])
    more = (f" and {len(runs) - _RUNS_NAMED} more"
            if len(runs) > _RUNS_NAMED else "")
    raise AtomIndexUnknown(
        f"{folder.resolve().name} is no run's folder: each run lists its atoms by its "
        f"own deck (an electrode rung's cell is a structure of its own) -- "
        + (f"name one of its runs: {named}{more}" if runs
           else "name a run folder or a file"))


def _runs_under(folder: Path) -> List[Path]:
    """The run folders below ``folder``, in name order -- what a refusal of a
    folder that is no run names."""
    from .runs import run_of
    found, stack = [], [folder]
    while stack:
        here = stack.pop(0)
        for d in sorted(c for c in here.iterdir() if c.is_dir()):
            if run_of(d) is not None:
                found.append(d)
            else:
                stack.append(d)
    return found


def _stage_name(stage: Optional[str]) -> Optional[str]:
    """A stage's name from its folder's token (``02_electrode_L`` →
    ``electrode_L``), a name as it is."""
    from .identity import parse_token
    got = parse_token(stage) if stage else None
    return got[1] if got else stage


def _of_run(p: Path, folder: Path) -> AtomIndexMap:
    """The map of the run that holds ``p`` -- its deck's (F3): the file's
    own run (`runs.run_of`), a point's ``shared/`` folder the point's.  A lead's file gathered into another rung's run is that lead's
    own (I6), refused naming the electrode run that answers it; an input a
    run of points keeps once, in its attempt, refused naming its points.  A
    file whose catalogue row lists it by user index -- a result written
    before it said so -- is that map listed by user index."""
    from .kind_table import kind, rung_label
    from .runfiles import listed_by_of
    from .runs import atom_index_map_of, place_of, run_of
    point = _point_folder(folder)
    run = (run_of(p) if point == folder else None) or run_of(point)
    task = (run.place if run is not None else place_of(point)).task
    if task is not None:
        for stage in kind(task.calculation).leads:
            if (p.name.startswith(rung_label(task.label, stage) + ".")
                    and (run is None or _stage_name(run.stage) != stage)):
                raise AtomIndexUnknown(
                    f"{p.name} is the {stage} rung's file, gathered here: "
                    f"its electrode cell is a structure of its own "
                    f"(model/overview.md 2.2a, I6) -- its atoms are "
                    f"answered by {_lead_run_words(point, stage)}")
    if run is None:
        raise AtomIndexUnknown(_no_run_words(p, point, task))
    atoms = atom_index_map_of(run)
    if listed_by_of(p.name) == LISTED_BY_USER:
        atoms = atoms.as_listed_by_user()
    if run.deck is not None and p.resolve() == run.deck.resolve():
        return atoms
    return replace(atoms, said=atoms.said or (
        f"{p.name} states no atom index map: its run's deck does"))


def _no_run_words(p: Path, folder: Path, task) -> str:
    """Why no run answers ``p``, by where it lies: a lead's surface Green's
    function kept as a run's scratch -- its electrode's orbitals, a
    structure of its own (I6); an input a run of points keeps once, in its
    attempt -- its points named; a calculation's or a stage's own file --
    the runs that answer named; a file no calculation of ours claims."""
    from .calcdirs import ATTEMPT
    from .runs import place_of
    from .warmfiles import is_scratch
    if task is None:
        return f"{p.name} states no atom index map, and no run of ours holds it"
    if is_scratch(p.name):
        return (f"{p.name} is a lead's surface Green's function, its run's "
                f"scratch: its electrode's orbitals, a structure of its own "
                f"(model/overview.md 2.2a, I6) -- it names no atom of the "
                f"junction")
    runs = _runs_under(folder)
    named = ", ".join(str(r.relative_to(folder)) for r in runs[:_RUNS_NAMED])
    if place_of(folder).where.what == ATTEMPT and runs:
        return (f"{p.name} is an input this attempt keeps once for its "
                f"points, each a run of its own -- ask one of them: {named}")
    return (f"{p.name} is no run's file -- its calculation's or its stage's "
            f"own, listing no atom of a run" + (
                f"; a run's file or folder answers: {named}" if runs else ""))


def _lead_run_words(folder: Path, stage: str) -> str:
    """The electrode rung's run that answers its own files -- its newest
    run's folder, relative to the calculation, or its rung's name when it
    has run nowhere yet."""
    from .jobset.materialize import latest_attempt
    from .jobset.naming import stage_home
    from .runs import place_of
    place = place_of(folder)
    try:
        home = stage_home(place.root, place.task, stage).dir
        run = latest_attempt(home)
    except (OSError, ValueError, KeyError):
        run = None
    return (f"its run folder, {run.relative_to(place.root)}" if run is not None
            else f"the {stage} rung's run")


def _stated(p: Path) -> Optional[AtomIndexMap]:
    """What ``p`` states of itself, by its kind -- ``None`` where it states
    nothing (an engine's file, a file written before the rule, a file in a
    run folder its run's deck answers)."""
    from .runfiles import ROLE_MOLWATCH, deck_roles, role_of
    from .sidecars import molstruct
    name = p.name
    if molstruct.is_sidecar(p):
        geometry = molstruct.geometry_for(p)
        return _of_pair(geometry or p, p)
    if p.suffix.lower() in molstruct.GEOMETRY_SUFFIXES:
        return _of_pair(p, molstruct.sidecar_path_for(p))
    if p.suffix == ".nc":
        return _of_nc(p)
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8")) if p.stat().st_size \
            else None
        block = d.get(MAP_KEY) if isinstance(d, dict) else None
        if isinstance(block, dict):
            return replace(map_of_block(block, name), stated_by=str(p))
        return _of_older_result(
            p, d.get("n_atoms_total") if isinstance(d, dict) else None)
    if role_of(name) == ROLE_MOLWATCH:
        return _of_log(p)
    if p.suffix in deck_roles():
        return _of_deck(p)
    return None


def _of_older_result(p: Path, n) -> Optional[AtomIndexMap]:
    """A file of ours listed by user index that states no map -- a result
    written before 2.32.0, its ``.json`` or its ``.nc`` (F3): its rows are
    the user order (I1), so it states its user column itself; its engine
    column is its run's deck's where the deck answers -- and where it does
    not (no run of ours holds it, its deck states no atom count or no order,
    I9) not known, said.  ``None`` for any other file."""
    from .runfiles import listed_by_of
    from .runs import atom_index_map_of, run_of
    if listed_by_of(p.name) != LISTED_BY_USER or not isinstance(n, int):
        return None
    run = run_of(_point_folder(p.parent))
    why = ("written before 2.32.0, and no run of ours holds it: its engine "
           "index is not known")
    if run is not None:
        try:
            atoms = atom_index_map_of(run)
            if atoms.n_atoms == n and atoms.user is not None:
                return replace(atoms.as_listed_by_user(), stated_by=str(p))
            why = (f"written before 2.32.0: its run's deck is over "
                   f"{atoms.n_atoms} atoms, so its engine index is not known")
        except (AtomIndexError, ValueError) as exc:
            why = f"written before 2.32.0: its engine index is not known -- {exc}"
    return replace(AtomIndexMap.of_user_structure(n, why), stated_by=str(p))


def _of_pair(geometry: Path, sidecar: Path) -> Optional[AtomIndexMap]:
    """A structure pair's map (F4), as the codec's read answers it
    (`workingcopy_structure.pair_atom_index_map`), stated by its sidecar;
    a pair the catalogue lists by engine index whose sidecar states no
    order is refused (I9).  A geometry file with no sidecar is the user's
    own, unless a run of ours holds it (``None``: its run answers)."""
    from .runs import run_of
    from .sidecars import molstruct
    from .workingcopy_structure import (pair_atom_index_map,
                                        pair_engine_listing,
                                        pair_order_unstated_said)
    if not sidecar.is_file():
        if run_of(_point_folder(geometry.parent)) is not None:
            return None
        from .structure import Structure
        text = geometry.read_text(encoding="utf-8-sig")
        n = (Structure.from_pdb(text) if geometry.suffix.lower() == ".pdb"
             else Structure.from_xyz(text)).n_atoms
        return replace(AtomIndexMap.of_user_structure(
            n, "the user's own file: its order is the user order"),
            stated_by=str(geometry))
    # THE PAIR AS THE CODEC'S READ TAKES IT -- one reader: a map against its
    # order record, a geometry changed since its pin, refused here as there
    # (`model/overview.md` § 2.2b; `structure-molstruct.md` § 1, § 3).
    from .workingcopy_structure import ORDER_UNSTATED, StructureCodec
    if (geometry.is_file()
            and geometry.suffix.lower() in molstruct.GEOMETRY_SUFFIXES):
        StructureCodec().load(geometry, frames=True, said_out=[],
                              opened=(ORDER_UNSTATED,))
    stated: set = set()
    payload = molstruct.load(sidecar, stated_out=stated)
    atoms = pair_atom_index_map(geometry, payload,
                                listing=pair_engine_listing(sidecar),
                                states_order="atom_order" in stated)
    if atoms is None:
        raise AtomIndexUnknown(pair_order_unstated_said(sidecar.name))
    return replace(atoms, stated_by=str(sidecar))


def _of_deck(deck: Path) -> Optional[AtomIndexMap]:
    """A deck's own map -- its ATOM-METADATA block, read by the run door's
    deck reader for the stage it was prepared for (`runs.declared_of_deck`,
    `runs.deck_atom_index_map`): a stage's prepared deck, the cited deck at
    a transport calculation's root.  ``None`` for a deck in a run folder
    (its run answers) or in no calculation of ours.  The cited deck is a
    relaxation's, as it was: its order is judged by a relaxation's sort
    rule, never its citing calculation's."""
    from .runfiles import JUNCTION_CITED_FILE
    from .runs import (OrderUnstated, declared_of_deck, deck_atom_index_map,
                       place_of, run_of)
    if run_of(deck) is not None:
        return None
    place = place_of(deck.parent)
    if not place.ours:
        return None
    stage = _stage_name(place.where.stage)
    kind = "optimization" if deck.name == JUNCTION_CITED_FILE else None
    try:
        atoms = deck_atom_index_map(
            declared_of_deck(deck, place.task, stage, kind=kind),
            place.task, stage)
    except OrderUnstated as exc:
        raise AtomIndexUnknown(str(exc)) from exc
    return replace(atoms, stated_by=str(deck))


def _of_nc(p: Path) -> Optional[AtomIndexMap]:
    """A NetCDF file's map -- the group of it that states one (a vibration
    result's ``/structure``; the transport data file's and a digest's
    root); ``None`` where none does (an engine's ``.TBT.nc``, an older
    file)."""
    from .ncdata import ATOM_AXIS, read_atom_index_map, reading
    n = None
    with reading(p) as ds:
        groups = [ds]
        while groups:
            g = groups.pop(0)
            if FIELD_LISTED_BY in g.ncattrs():
                return replace(read_atom_index_map(g, source=p.name),
                               stated_by=str(p))
            if n is None and ATOM_AXIS in g.dimensions:
                n = len(g.dimensions[ATOM_AXIS])
            groups.extend(g.groups.values())
    # A FILE OF OURS WRITTEN BEFORE IT SAID SO, by its catalogue row (F3).
    return _of_older_result(p, n)


def _of_log(p: Path) -> Optional[AtomIndexMap]:
    """A trajectory log's map, from its header line (``# atom_index_map:``)
    -- read through the format's grammar; ``None`` where the header states
    none."""
    from .parse.engines import molwatch_grammar as _MG
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if _MG.BLOCK_BEGIN.search(line):
                break
            m = _MG.ATOM_INDEX_MAP.match(line.strip())
            if m:
                try:
                    block = json.loads(m.group(1))
                except ValueError as exc:
                    raise AtomIndexError(f"{p.name}: its atom_index_map line "
                                         f"does not read: {exc}") from exc
                return replace(map_of_block(block, p.name),
                               stated_by=str(p))
    return None


__all__ = ["AtomIndexUnknown", "atom_index_map", "map_of_block"]

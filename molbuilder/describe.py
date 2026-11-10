"""``describe`` — write the portable description and its job request.

**Module:** L2. Imports ``task``, ``template``, ``identity`` and
``validation.task``; imported by ``jobset/_cli``, the hand-over
(``web/blueprints/build``)
and a kind's own describe (``transport/describe``, ``transport/reuse``).
It is **pure up to one transaction**: everything is built and checked in
memory, and :func:`publish` -- which :func:`write_description` and the
hand-over write through -- is the only part that touches a disk.

**Contract:** [`execution/architecture.md`](?doc=execution/architecture.md) § 4
— the `describe` route: *"**write** the portable description and its job
request — the template, ``task.json``, ``job-request.json``, the data files ·
ask → check → write · **2, asking 3 and 4 about the description itself**"* ·
[`execution/job-request.md`](?doc=execution/job-request.md) § 9 ·
[`execution/job-system.md`](?doc=execution/job-system.md) § 5.1 (the command) ·
[`execution/project-layout.md`](?doc=execution/project-layout.md) § 2.1 (what
makes the folder portable) · [`engines/stages.md`](?doc=engines/stages.md) § 6.6
(the split preflight).

WHAT IT WRITES, AND WHAT IT MUST NOT.  Three files: ``<label>.template.toml``,
``task.json`` and ``job-request.json`` -- each stage's group, nothing else
asked yet -- plus whatever data files travel (pseudopotentials).  **It writes
no deck and reads no machine**: rendering moved to ``prep`` step 3 (`script-preparation.md` § 3.0) because a deck carries values that
depend on how it will be launched (`project-layout.md` § 2.2), and a
description that named a machine would stop
meaning the same thing when you copied it.

THE STRUCTURE IS REFERENCED, NEVER COPIED (`stages.md` § 6.3).  ``task.json``
records where the structure was and a **witness** — its formula and atom count —
so a description opened against a structure that has since changed can *say so*
instead of silently building a different calculation under the same id.

ASK → CHECK → WRITE, AND THE ORDER IS THE POINT.  Every refusal happens while
nothing has been written: :class:`~molbuilder.task.Task` runs the codec's four
checks in its constructor, and :func:`~molbuilder.validation.task.preflight`
runs § 6.6's other four — the ones that need the engine's field schema.  Only
then does anything reach a disk, and it reaches it through a staging directory
that is published in one pass or removed entirely.  *"Describing a calculation
writes every file or none."*

WHY A DATA OBJECT AND THEN A WRITER, rather than one function that writes.  Two
surfaces have to produce byte-identical descriptions — the terminal and the
browser's doors — and both write on the server through the one transaction,
:func:`publish`: one producer, one writer.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import dataclasses as _dc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .job_request import JobRequest
from .task import FILENAME as TASK_FILENAME
from .task import Stage, StructureRef, Task, derive_run, varies_for
from .template_file import template_path
from .template import template_with_values
from .workingcopy_structure import StructureCodec


class DescribeError(Exception):
    """A calculation could not be described. Nothing was written."""


@dataclass(frozen=True)
class Description:
    """A calculation's portable description and its job request, as
    **data** — no filesystem.

    ``task`` is the mission and ``template`` the parameter catalogue;
    ``request`` how it runs -- with them, floor 2 entire. ``pseudo_species`` names the species labels whose
    ``<label>.psml`` files travel with it — the only *data* files a SIESTA calculation
    needs beyond the structure.  The structure is a REFERENCE in this object
    (``task.structure.source`` names it; no bytes here) — but
    :func:`write_description` does lay a travelling copy down beside the
    description (the ``.source`` pair), so the folder runs anywhere without
    the original tree.
    """
    task: Task
    template: str
    pseudo_species: Tuple[str, ...] = ()
    request: Optional[JobRequest] = None
    #: what its own check said beside a refusal it did not make (a value
    #: outside its recommended range), for the road to say
    notes: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """EVERY STAGE STATES ITS GROUP (`execution/job-request.md` § 3.3):
        a new calculation's job request, its ladder numbered here when the
        producer states none -- `molbuilder describe`'s, the Transport
        tab's -- the defaults repeating the number above wherever the
        ladder's links allow it."""
        if self.request is None:
            from .job_request import fresh
            from .jobset.group import default_groups
            groups = default_groups(self.task, None, self.template)
            object.__setattr__(self, "request", fresh(self.task, {
                st.name: g for st, g in zip(self.task.stages, groups)}))

    @property
    def label(self) -> str:
        """The ``SystemLabel``, and the stem of every file. Derived by
        :class:`~molbuilder.task.Task` through the one normaliser."""
        return self.task.label


# --------------------------------------------------------------------- #
#  Ask -> check                                                         #
# --------------------------------------------------------------------- #

def build_description(
    struct,
    cfg,
    stages: Sequence[Stage],
    *,
    engine: str,
    name: str,
    source: str,
    calculation: str,
    created: str = "",
    pseudo_species: Sequence[str] = (),
    generators: Optional[Dict[str, Any]] = None,
    value_sources: Optional[Dict[str, str]] = None,
) -> Description:
    """Build and **check** a description. Nothing is written.

    ``cfg`` carries the values the template records — the science that holds
    unless a stage says otherwise. ``stages`` is the ladder, and it needs at
    least one entry: § 6.5 (2026-08-16) makes ONE stage the ordinary starting
    point rather than a special shape, so there is no stage-less form to pass
    an empty sequence for. An empty one is refused, naming the fix.

    ``source`` is where the structure lives, recorded as a **reference**
    (§ 6.3). ``name`` is what the user called this calculation; the label and
    the run id are derived from it and never retyped -- the label written
    into ``cfg``'s identity setting from the task's own (`task.label_of`),
    so the template and the description cannot disagree about it.

    Raises :class:`DescribeError` if any § 6.6 check refuses — with the
    offending key, the stage it was in, and the bound it broke, because this is
    a file people edit by hand.
    """
    ladder = tuple(stages)
    if not ladder:
        raise DescribeError(
            "no stages. A job has at least one stage -- one is the ordinary "
            "case, not a special shape. Pass a single "
            "Stage carrying no overrides for a calculation that is just the "
            "template.")
    stage_names = tuple(s.name for s in ladder)

    task = Task(
        engine=engine,
        run=derive_run(name, struct.formula, created=created,
                       stage_names=stage_names),
        structure=StructureRef(source=source,
                               formula=struct.formula,
                               atoms=struct.n_atoms),
        # § 6.5: a job always has at least one stage, so these travel
        # together and both are always present.  ``varies`` may be empty --
        # several stages differing in nothing but their name is a real state.
        varies=varies_for(s.overrides for s in ladder),
        stages=ladder,
        calculation=calculation,
    )

    from .engine_table import relabelled
    cfg = relabelled(engine, cfg, task.label)
    # § 4.3: this calculation's template is the CATALOGUE, narrowed to this
    # engine, carrying the values `cfg` holds.  The questions were asked by the
    # catalogue; `describe` supplies the answers, exactly as a surface does.
    # ...and where each value came from (§ 6.6 obligation 2): what the
    # caller says (``value_sources``), every other item nobody's choice.
    text = template_with_values(cfg, engine=engine,
                                calculation=calculation,
                                sources=value_sources)
    desc = Description(task=task, template=text,
                       pseudo_species=tuple(pseudo_species))
    return _dc.replace(desc, notes=tuple(_check(desc, generators=generators)))


def _check(desc: Description, *, generators=None) -> List[Any]:
    """§ 6.6's schema-dependent half: every error refuses, named; the
    warnings are returned, for the road to say.

    The codec's own four checks have already run — ``Task`` refuses in its
    constructor, so there is no object to get here without them. This is the
    rest of § 6.6, with the template and the job request this describe
    writes (the job request's own check among them), and it is **the
    existing preflight** rather than a second implementation:
    `validation/task.py` already walks them in the contract's order and
    already names what it refused.
    """
    from .validation.task import preflight, refuse_on_error
    issues = preflight(desc.task, template_text=desc.template,
                       request=desc.request,
                       **({"generators": generators} if generators else {}))
    try:
        return refuse_on_error(issues)
    except Exception as exc:
        raise DescribeError(str(exc)) from exc


# --------------------------------------------------------------------- #
#  ... -> write                                                         #
# --------------------------------------------------------------------- #

def _place(dest):
    """``(folder, tree, segments)`` of a calculation's folder ``dest``: the
    tree it lies in, found as a calculation finds it
    (`projects.find_projects_root`) -- the folder itself only when no tree
    lies above it and it is the tree's own root -- and its segments from
    there; ``tree`` ``None`` for a folder in none."""
    from .projects import PROJECTS_ROOT_NAME, find_projects_root, segments
    folder = Path(dest).resolve()
    tree = find_projects_root(folder)
    if tree is None:
        if folder.name != PROJECTS_ROOT_NAME:
            return folder, None, None
        tree = folder
    return folder, tree, segments(folder, tree)


def refuse_place(dest) -> None:
    """A CALCULATION'S PLACE (`job-contracts.md` § 2.5): under a topic --
    project / topic / its own folder, three levels or more inside the tree
    it lies in -- and every folder describing makes on the way named as one
    made in the sidebar is (`projects.validate_folder_name`), a project not
    there yet by a project's name, for it is made whole
    (:func:`project_to_make`).  A folder above that place, a file where a
    folder goes, or a name a folder to be made cannot take, is refused
    naming it, before anything is written.  Asked by every door that describes, beside
    :func:`refuse_described`; the fence, which each door asks first, keeps
    a folder outside the tree out."""
    from .projects import (InvalidName, address, validate_folder_name,
                           validate_name)
    folder, tree, parts = _place(dest)
    if tree is None:
        return
    if len(parts) < 3:
        raise DescribeError(
            f"{'the projects root' if not parts else address(folder, tree)} "
            f"is too high in the tree -- a calculation lives under a topic, "
            f"<project>/<topic>/<its folder>.  Describe it in a folder of "
            f"its own further down.")
    for depth, name in enumerate(parts):
        here = tree.joinpath(*parts[:depth + 1])
        if here.is_dir():
            continue
        if here.exists():
            raise DescribeError(
                f"{address(here, tree)} is a file, not a folder -- a "
                f"calculation's files go in a folder of its own.")
        try:
            if depth == 0:
                validate_name(name, kind="project")
            else:
                validate_folder_name(name, depth, new=True)
        except InvalidName as exc:
            raise DescribeError(
                f"{address(here, tree)} cannot be made: {exc}") from None


def project_to_make(dest):
    """The project a calculation's folder ``dest`` lies in, when it is not
    there yet -- made whole by :func:`publish`, as New project makes one
    (`projects.populate_project_skeleton`) -- else ``None``."""
    _folder, tree, parts = _place(dest)
    if tree is None or not parts:
        return None
    project = tree / parts[0]
    return None if project.exists() else project


def refuse_described(dest) -> None:
    """ONE CALCULATION PER FOLDER (user, 2026-10-01: refuse): a folder that
    already holds a description is changed in Task setup or its
    ``task.json``, never described again.  NEVER ONE INSIDE ANOTHER (user,
    2026-11-07: *"calculations can continue from another, not within
    another"*; `project-layout.md` § 7, invariant 2a): a folder inside a
    calculation's folder, or one holding a calculation below it, is
    refused, naming that calculation.  Asked by every door that
    describes -- `molbuilder describe`, the hand-over and a kind's own
    describe (`transport.describe`, `engines/transport.md` § 3.8.2a) --
    before anything is written.  Each folder is named by its address from
    the projects root (`projects.address`)."""
    from .calcdirs import calculation_above, calculation_below, described
    from .projects import address
    if described(dest):
        raise DescribeError(
            f"{address(dest)} already holds a calculation ({TASK_FILENAME}) -- one "
            f"calculation per folder.  Change it in Task setup or its "
            f"{TASK_FILENAME}, or describe the new one in a folder of its "
            f"own.")
    above = calculation_above(dest)
    other = above or calculation_below(dest)
    if other is None:
        return
    raise DescribeError(
        f"{address(dest)} {'is inside' if above else 'holds'} the "
        f"calculation {address(other)} -- a calculation is never inside another "
        f"calculation's folder.  Describe it in a folder of its own beside "
        f"{other.name}; a calculation that builds on another's result cites "
        f"it.")


def write_description(desc: Description, dest, *,
                      psml_lib=None, struct=None) -> List[Path]:
    """Write *desc* into *dest*, publishing every file or none.

    *struct*, when given, is the structure AS DESCRIBED -- including any
    modification describe itself applied (``--vacuum``); ``None`` reads the
    source file through the codec.  Either way it travels as the
    calculation's own pair, named for the label (`StructureCodec.source_files`,
    the call the hand-over makes).

    Written through :func:`publish`, every file or none.

    *psml_lib* is where the pseudopotentials are read from. It is a **path on
    this machine** and does not enter the description; what travels is the
    files themselves, which are the same everywhere (`project-layout.md`
    § 2.1).

    A transport calculation's junction is never written here: its first
    prep composes it from its citation, or takes the one a finished
    calculation ran (`engines/transport.md` § 2a.16).
    """
    out_dir = Path(dest)

    # THE STRUCTURE TRAVELS AS THE CALCULATION'S OWN PAIR,
    # ``<label>.source.xyz`` + its sidecar, through the one call the hand-over
    # makes (`StructureCodec.source_files`), and the written ``task.json``
    # records the name that call gave (`job-contracts.md` § 6.3): identities
    # are validated dot-free, so no engine output -- which stems every file
    # on an identity -- can ever take a dotted name.  Without the mark, a run
    # whose label matched the structure's stem would overwrite its own input
    # (WriteCoorXmol writes ``<SystemLabel>.xyz``).
    #
    # NAMED FOR THE LABEL, never for the structure file *(plan D20,
    # 2026-10-04)*, as the hand-over, the catalogue and the Task setup card
    # name it.  And WHICH BYTES travel is the codec's, always: describe can
    # MODIFY the structure it was handed (``--vacuum``), whose facts live in
    # metadata a bare copy has nowhere to put.  The original path stays the locator; what lands in the folder is the
    # description's own, self-contained reference.
    # A TRANSPORT DESCRIPTION NAMES NO STRUCTURE: its structure is the
    # junction citation, composed at prep (`engines/transport.md` § 3.1).
    src = (Path(desc.task.structure.source).expanduser()
           if desc.task.structure is not None and desc.task.structure.source
           else None)
    pair: List[Tuple[Path, bytes]] = []
    if src is not None and src.is_file():
        codec = StructureCodec()
        pair = codec.source_files(
            struct if struct is not None else codec.load(src, a_copy=True),
            desc.task.label)
        desc = _dc.replace(desc, task=_dc.replace(
            desc.task, structure=_dc.replace(
                desc.task.structure, source=pair[0][0].name,
                fingerprint=StructureCodec.fingerprint(pair))))

    def _fill(staging: Path) -> None:
        # THE TWO RECORDS THROUGH THEIR OWNERS' WRITERS (`architecture.md`
        # A16), into the staging folder -- the writers Save takes too, so
        # the two surfaces write the same bytes; the template has no writer
        # of its own and is written as its text.
        from .job_request import write as _write_request
        from .task import write_task
        from .persist import write_bytes
        write_bytes(template_path(staging, desc.label),
                    desc.template.encode("utf-8"))
        write_task(staging, desc.task)
        _write_request(staging, desc.request)
        # The STRUCTURE travels with the calculation: the description records
        # a reference plus a witness (stages.md § 6.3), and `prep` looks
        # "beside the calculation FIRST" -- a relative source recorded from
        # another cwd would be unresolvable from inside the folder.  Written
        # like the pseudos: the file is the calculation's
        # data, the PATH stays this machine's.
        for path, data in pair:
            write_bytes(staging / path.name, data)
        if psml_lib and desc.pseudo_species:
            from .pseudos import (copy_pseudopotentials, describe_psml_anchor,
                                  resolve_psml_lib)
            # THE SAME ANCHOR RULE AS EVERY OTHER SURFACE (job-contracts.md
            # § 2.5a): the bare name that means the tree everywhere else means
            # it here too.
            #
            # The anchor is `out_dir`, not `staging`: the calculation is the
            # folder being described, and staging is a transaction detail
            # that gets renamed away.  Anchoring on it would make a `./`
            # spelling point at a directory that ceases to exist.
            from .pseudos import PsmlLibError
            try:
                lib = resolve_psml_lib(str(psml_lib), dest_dir=out_dir)
            except PsmlLibError as exc:
                raise DescribeError(str(exc))
            if not lib.is_dir():
                raise DescribeError(
                    f"--psml-lib {psml_lib!r} is not a directory.  "
                    + describe_psml_anchor(str(psml_lib), dest_dir=out_dir)
                    + "  The pseudopotentials travel with the calculation, "
                      "so they have to be somewhere readable now.")
            copy_pseudopotentials(list(desc.pseudo_species), lib, staging)

    return publish(out_dir, _fill)


def publish(dest, fill) -> List[Path]:
    """EVERY FILE OR NONE, into the folder *dest* -- the one transaction a
    calculation's files are written through: `molbuilder describe`, the
    transport describe door and the hand-over (`web/handover-procedure.md`
    § 2).  ``fill(staging)`` writes every file into a staging directory
    **beside the target**; once it returns, each is put in place with
    :func:`os.replace`, and if it raises the staging directory is removed
    entirely and the target is exactly as it was.  Beside rather than in
    ``/tmp`` so the publish is a rename within one filesystem -- across
    devices it would be a copy, and a copy is not atomic.  THE STRUCTURE
    FIRST, its geometry before its sidecar (`model/structure.md` § 2.4), for
    the records name it; then every other file by name.  A project the
    folder lies in that is not there yet is made whole first
    (:func:`project_to_make`; `projects.populate_project_skeleton`), and
    removed again if the files are not published.  Returns the paths
    written, in that order."""
    out_dir = Path(dest)
    project = project_to_make(out_dir)
    if project is not None:
        from .projects import populate_project_skeleton
        populate_project_skeleton(project, project_name=project.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.describe-",
                                    dir=out_dir.parent))
    try:
        fill(staging)
    except BaseException:
        # Nothing is published, so the target is exactly as it was -- and
        # a project made for it is taken back.
        shutil.rmtree(staging, ignore_errors=True)
        if project is not None:
            shutil.rmtree(project, ignore_errors=True)
        raise
    from .sidecars.molstruct import is_geometry, is_sidecar
    written: List[Path] = []
    for src in sorted(staging.iterdir(), key=lambda p: (
            not (is_geometry(p) or is_sidecar(p)), is_sidecar(p), p.name)):
        dst = out_dir / src.name
        os.replace(src, dst)
        written.append(dst)
    staging.rmdir()
    return written


__all__ = ["Description", "DescribeError",
           "build_description", "project_to_make", "publish", "refuse_place",
           "refuse_described", "write_description"]

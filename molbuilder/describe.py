"""``describe`` — write the portable description, and nothing else.

**Module:** L2. Imports ``task``, ``template``, ``identity`` and
``validation.task``; imported by ``jobset/_cli`` and (later) by the web's
describe route. It is **pure up to one function**: everything is built and
checked in memory, and :func:`write_description` is the only part that touches
a disk.

**Contract:** [`execution/architecture.md`](?doc=execution/architecture.md) § 4
— the `describe` route: *"**write** the portable description — the template,
``task.json``, the data files · ask → check → write · **floor 2 only**"* ·
[`execution/job-system.md`](?doc=execution/job-system.md) § 5.1 (the command) ·
[`execution/project-layout.md`](?doc=execution/project-layout.md) § 2.1 (what
makes the folder portable) · [`engines/stages.md`](?doc=engines/stages.md) § 6.6
(the split preflight).

WHAT IT WRITES, AND WHAT IT MUST NOT.  Two files: ``<label>.template.toml`` and
``task.json``, plus whatever data files travel (pseudopotentials).  **It renders
no deck.**  That is the whole of *floor 2 only*: rendering moved to ``prep``
step 3 because a deck carries values that depend on how it will be launched
(`project-layout.md` § 2.3.1), and a description that named a machine would stop
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
browser — and the browser writes through its own concealed file layer rather
than through raw paths.  One producer, two writers, is the shape
``job-system.md`` § 4.1 calls Promotion A.  (``StageBundle`` was its other
instance; that producer lost its last caller when the old surface went,
2026-08-11, and folds away with `bench` — plan step 6.)
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .task import FILENAME as TASK_FILENAME
from .task import Stage, StructureRef, Task, derive_run, varies_for
from .template import SUFFIX as TEMPLATE_SUFFIX
from .template import template_with_values
from .workingcopy_structure import StructureCodec


class DescribeError(Exception):
    """A calculation could not be described. Nothing was written."""


@dataclass(frozen=True)
class Description:
    """A calculation's portable description, as **data** — no filesystem.

    ``task`` is the mission and ``template`` the parameter catalogue; together
    they are floor 2 entire. ``pseudo_species`` names the elements whose
    ``.psml`` files travel with it — the only *data* files a SIESTA calculation
    needs beyond the structure, which is referenced rather than copied.
    """
    task: Task
    template: str
    pseudo_species: Tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The ``SystemLabel``, and the stem of every file. Derived by
        :class:`~molbuilder.task.Task` through the one normaliser."""
        return self.task.label

    def files(self) -> Dict[str, str]:
        """``{filename: text}`` — the two files a description *is*.

        The template's name comes from the label and one constant; the
        description's from the codec that owns it. Neither is spelled out
        here, because exactly one module owns each name.
        """
        return {
            f"{self.label}{TEMPLATE_SUFFIX}": self.template,
            TASK_FILENAME: _task_json_text(self.task),
        }


def _task_json_text(task: Task) -> str:
    """``task.json``'s bytes, through the codec that owns them.

    Written via the codec rather than ``json.dumps`` here so that key order,
    which keys are omitted, and the schema string all stay in the one module
    ``test_only_one_module_reads_or_writes_task_json`` guards.
    """
    import json
    return json.dumps(task.to_dict(), indent=2) + "\n"


# --------------------------------------------------------------------- #
#  Ask -> check                                                         #
# --------------------------------------------------------------------- #

def build_description(
    struct,
    cfg,
    stages: Sequence[Stage],
    *,
    engine: str,
    shape: str,
    name: str,
    source: str,
    created: str = "",
    calculation: str = "optimization",
    pseudo_species: Sequence[str] = (),
    generators: Optional[Dict[str, Any]] = None,
) -> Description:
    """Build and **check** a description. Nothing is written.

    ``cfg`` carries the values the template records — the science that holds
    unless a stage says otherwise. ``stages`` is the ladder; pass an empty
    sequence for a calculation with a single parameter set, which
    `stages.md` § 6.5 spells by omitting both keys rather than by an empty
    list.

    ``source`` is where the structure lives, recorded as a **reference**
    (§ 6.3). ``name`` is what the user called this calculation; the label and
    the run id are derived from it and never retyped.

    Raises :class:`DescribeError` if any § 6.6 check refuses — with the
    offending key, the stage it was in, and the bound it broke, because this is
    a file people edit by hand.
    """
    ladder = tuple(stages)
    stage_names = tuple(s.name for s in ladder)

    task = Task(
        engine=engine,
        shape=shape,
        run=derive_run(name, struct.formula, created=created,
                       stage_names=stage_names),
        structure=StructureRef(source=source,
                               formula=struct.formula,
                               atoms=struct.n_atoms),
        # § 6.5 -- absent together. An empty ``stages`` would be a second way
        # to spell "one parameter set", so the no-ladder case passes neither.
        varies=(varies_for(s.overrides for s in ladder) if ladder else None),
        stages=(ladder or None),
        calculation=calculation,
    )

    _check(task, cfg, generators=generators)

    # § 4.3: this calculation's template is the CATALOGUE, narrowed to this
    # engine, carrying the values `cfg` holds.  The questions were asked by the
    # catalogue; `describe` supplies the answers, exactly as a surface does.
    text = template_with_values(cfg, engine=engine)
    return Description(task=task, template=text,
                       pseudo_species=tuple(pseudo_species))


def _check(task: Task, cfg, *, generators=None) -> None:
    """§ 6.6's schema-dependent half, refusing on the first error.

    The codec's own four checks have already run — ``Task`` refuses in its
    constructor, so there is no object to get here without them. This is the
    other four, and it is **the existing preflight** rather than a second
    implementation: `validation/task.py` already walks them in the contract's
    order and already names what it refused.
    """
    from .validation.task import preflight, refuse_on_error
    issues = preflight(task, type(cfg),
                       **({"generators": generators} if generators else {}))
    try:
        refuse_on_error(issues)
    except Exception as exc:
        raise DescribeError(str(exc)) from exc


# --------------------------------------------------------------------- #
#  ... -> write                                                         #
# --------------------------------------------------------------------- #

def write_description(desc: Description, dest, *,
                      psml_lib=None, struct=None) -> List[Path]:
    """Write *desc* into *dest*, publishing every file or none.

    *struct*, when given, is the structure AS DESCRIBED -- including any
    modification describe itself applied (``--vacuum``).  It decides how the
    structure travels: with metadata, as the codec pair; without, as a raw
    copy of the source (see the comment at the copy below).  ``None`` keeps
    the raw-copy behaviour for callers that never modify.

    The transaction is a staging directory **beside the target**, published
    with :func:`os.replace` once every artifact exists, and removed entirely if
    anything raises. Beside rather than in ``/tmp`` so the publish is a rename
    within one filesystem — across devices it would be a copy, and a copy is
    not atomic.

    *psml_lib* is where the pseudopotentials are read from. It is a **path on
    this machine** and does not enter the description; what travels is the
    files themselves, which are the same everywhere (`project-layout.md`
    § 2.1).
    """
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.describe-",
                                    dir=out_dir.parent))
    try:
        for filename, text in desc.files().items():
            (staging / filename).write_text(text, encoding="utf-8")
        # The STRUCTURE travels with the calculation (found by M9's walk,
        # 2026-08-12): the description records a reference plus a witness
        # (stages.md § 6.3), and `prep` looks "beside the calculation
        # FIRST" -- but nothing made that true.  A relative source recorded
        # from another cwd was unresolvable the moment you stood inside the
        # folder.  Copied like the pseudos: the file is the calculation's
        # data, the PATH stays this machine's.
        #
        # WHICH bytes travel is the codec's call (2026-08-12): describe can
        # MODIFY the structure it was handed (--vacuum), and those facts
        # live in metadata a bare .xyz has nowhere to put -- so a raw copy
        # silently dropped them, and prep rendered the 3 A-default cell
        # over an explicit scientific choice.  A structure with metadata
        # travels as the codec pair (document + .molstruct.json, the pair
        # prep's loader already reads); one without travels as the raw
        # copy, byte-identical provenance.
        src = (Path(desc.task.structure.source).expanduser()
               if desc.task.structure.source else None)
        if src is not None and src.is_file():
            pair = ([] if struct is None else
                    StructureCodec().files(struct, staging / src.name))
            if len(pair) > 1:      # keep_sidecar: metadata worth carrying
                for path, data in pair:
                    path.write_bytes(data)
            else:
                shutil.copy2(src, staging / src.name)
        if psml_lib and desc.pseudo_species:
            from .siesta.input import copy_pseudopotentials
            lib = Path(psml_lib).expanduser()
            if not lib.is_dir():
                raise DescribeError(
                    f"--psml-lib {psml_lib!r} is not a directory. The "
                    f"pseudopotentials travel with the calculation, so they "
                    f"have to be somewhere readable now.")
            copy_pseudopotentials(list(desc.pseudo_species), lib, staging)
    except BaseException:
        # Nothing is published, so the target is exactly as it was.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    written: List[Path] = []
    for src in sorted(staging.iterdir()):
        dst = out_dir / src.name
        os.replace(src, dst)
        written.append(dst)
    staging.rmdir()
    return written


__all__ = ["Description", "DescribeError",
           "build_description", "write_description"]

"""The engine seam -- what an engine supplies for `prep` to run its steps
over it (`execution/script-preparation.md` § 4), and the hooks SIESTA
answers with: its sibling artifacts, its data files (the pseudopotentials,
screened and copied into ``pseudos/``), its shared package.  Floor 3
(`execution/architecture.md` § 2.1).

A module of its own since 2026-10-03 (W55 B7): it lived in the conductor,
so what a stage continues from imported the conductor to learn an
engine's deck suffix and config class.  Nothing here prepares anything --
it describes an engine, and the conductor asks it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from ..pseudos import PSEUDO_DIRNAME
from .errors import PrepError


@dataclass(frozen=True)
class EngineSeam:
    """What an engine supplies for `prep` to run the steps over it —
    `script-preparation.md` § 4's seam, stated as data.

    **That document indexes the questions by the STEP that asks them**, which is
    the ordering to read them in: a bag of callables cannot answer *"what does
    this engine still owe?"*, and against the steps a gap is a blank row.

    **Fifteen questions, ten members.**  One is answered by shared code
    (`validation.validate`), and four arrive together through ``spec_for`` --
    the layout, the syntax, the record's values and the check rules are all the
    engine describing its deck, so they ride on one ``DeckSpec`` rather than on
    four seam members.  A member that answers ``None`` is answering *nothing*,
    which is a real answer and a recorded one (§ 4, W5): PySCF gives it to
    ``provide_data``, ``shared_package`` and ``sibling_artifacts``, and to
    ``bench_marks`` on the spec.

    Everything engine-specific that the loop below needs lives HERE, so the
    loop itself never asks which engine it is in.  ``_job_for`` branched on
    ``task.engine == "siesta"`` until 2026-08-12, which was § 7's forbidden
    ``if`` one floor down from where it was deleted.
    """
    #: The config class the template rebuilds into.
    config_cls: type
    #: ``(structure, config, stage_token=) -> DeckSpec`` — the engine
    #: DESCRIBES its deck; the framework renders, writes and checks it
    #: (`script-preparation.md` § 4.3).  The token is a RENDER ARGUMENT (step
    #: 7, C7): the emitter never learns the word, the deck's filename carries
    #: it.
    #:
    #: **It handed back finished TEXT until 2026-08-18**, and that one fact was
    #: what kept the framework's step-3 runner unreachable: given text, the
    #: conductor had no form to pass on, so it performed the write and the
    #: check itself and the ORDER of step 3 was stated in two places.  Given a
    #: form, the framework can also re-derive what the deck was supposed to
    #: contain, so nothing has to be carried alongside the text to make the
    #: check possible.
    spec_for: Callable
    #: The deck's type suffix (``.fdf``).
    suffix: str
    #: ``config -> the engine's identity literal`` (``SystemLabel`` / ``JOB``).
    label_of: Callable
    #: ``(config, label) -> config`` — the identity WRITTEN, for a trial's
    #: relabelling.  Filename relabelling alone is not the § 2.3.2
    #: protection: the deck's own ``SystemLabel`` line is what keys the warm
    #: files, and until 2026-08-12 it kept the run's label (found by the
    #: first sweep that ever rendered a deck).
    relabel: Callable
    #: ``(label, config, calculation, base_dir) -> warm-file declaration``
    #: for the Job -- the engine reads its § 4.2a rules file for the TYPE
    #: (U2), and ``base_dir`` lets a calculation's own fine-tuned copy win
    #: (U6a; the comment said 3 args while the call passed 4 -- C-d).
    warm_for: Callable
    #: ``config -> traits`` the launcher routes on (GPU solver, …).
    traits_for: Callable
    #: ``(struct, config, deck_path) -> None`` — the sibling files this
    #: engine's deck TEXT promises (E6: a charged SIESTA deck instructs
    #: running a script; the promise must be kept on every route that
    #: renders the deck).  ``None`` for an engine whose decks promise
    #: nothing.
    sibling_artifacts: Optional[Callable] = None
    #: ``(base_dir) -> [filename]`` — which files in the calculation are the
    #: SHARED PACKAGE every job links to.  The engine that put them there is
    #: the one that can name them: this was a ``*.psml`` glob in shared code,
    #: a SIESTA fact stated a floor below where SIESTA may speak, so a second
    #: engine with data files of its own would have shipped none of them.
    #: ``None`` for an engine that puts nothing in — the package is then empty,
    #: which is the honest answer rather than an accident of a glob.
    shared_package: Optional[Callable] = None
    #: ``(struct, config, base_dir) -> None`` — the DATA FILES this engine's
    #: deck cannot run without, put into the calculation.
    #:
    #: *Named ``stage_data`` for about a minute: "stage" is this project's
    #: core noun and here it was being borrowed as a verb -- the collision
    #: `submit._staged_for_launch` was renamed for.*  Distinct from
    #: ``sibling_artifacts``, which is about what a deck's own text promises;
    #: this is about what the ENGINE will open.  ``None`` for an engine that
    #: needs none (PySCF's basis sets ship inside PySCF).
    #:
    #: It belongs to `prep` because `project-layout.md` § 2.6 puts the copy on
    #: the machine that runs the job — where the library lives is a fact about
    #: that machine — and because `prep` is already what decides the shared
    #: package.  Added 2026-08-18: the rule *"a calculation copies the
    #: pseudopotentials it needs into its own shared package"* was written and
    #: unowned, so `jobset init` performed it and the browser's hand-over
    #: did not, and a calculation described in the browser prepped, laid out
    #: its directories and reported success with no pseudopotentials in it.
    provide_data: Optional[Callable] = None


def _siesta_sibling_artifacts(struct, cfg, deck_path: Path, *,
                              kind: str) -> None:
    """The sibling files a SIESTA deck's own text PROMISES.

    A charged deck instructs ``python3 makov_payne_correction.py`` in its
    header -- a promise only ``convert`` kept until E6 (redo 2026-08-12):
    the described route rendered the same header and never wrote the
    script, so `prep` shipped an instruction to run a file that did not
    exist.  Same writer both routes, so they cannot drift.

    The charge is the electronic state's (`science/chemistry-correctness.md`
    § 2a) -- the one the deck beside it was written from.  The deck has just
    been written from that state, so a label naming no element cannot reach
    here.  ONLY FOR A FINITE SYSTEM (§ 2b): the script's formula is a
    molecule's in a vacuum box, and a charged slab or crystal -- whose deck
    says it gets no formula -- got the script too until the M6 review."""
    from ..electronic_state import electronic_state
    from ..siesta.makov_payne import emit_correction_script
    state = electronic_state(struct, cfg, kind=kind)
    q = state.net_charge.value
    if q != 0 and state.finite:
        emit_correction_script(fdf_path=deck_path,
                               system_label=cfg.system_label, q=q)


def _pseudo_dir(base: Path) -> Path:
    """THE PARENT'S DATA IS GROUPED (roadmap 7.10 M6): the calculation's
    pseudopotential copies live in ``pseudos/``, one folder, instead of
    N ``<El>.psml`` entries loose at the root.  Root strays -- put there
    by `init`, an earlier prep, or a travelled bundle -- are ADOPTED,
    the same move-in the deck adoption uses.  The run directories are
    untouched by this: each still receives ``<El>.psml`` beside the
    deck (that is SIESTA's own contract; it has no search path).

    ONE rule, two providers: the SIESTA arm below and the transport
    composite's (which fetches from the citation instead of a library).
    """
    pdir = base / PSEUDO_DIRNAME
    pdir.mkdir(exist_ok=True)
    # A container, and it says so (`project-layout.md` § 1.4a): the shared
    # package holds files, never a run.  Left unstamped it was the directory
    # that reported a calculation *running* because it had no result file in
    # it -- which is the shape of answer § 1.4a exists to stop.  It reads back
    # as *support* rather than a stage, because the naming authority maps no
    # job to it; that is derived, not stored.
    from .. import calcdirs
    calcdirs.write(pdir, role=calcdirs.CONTAINER, root=base)
    for stray in base.glob("*.psml"):
        target = pdir / stray.name
        if not target.exists():
            stray.replace(target)
        else:
            stray.unlink()
    return pdir


def _siesta_provide_pseudos(struct, cfg, base: Path) -> None:
    """Put the pseudopotentials this deck needs into the calculation.

    SIESTA opens ``<element>.psml`` in the directory it runs from and has no
    search path, so a missing file is not a preference — it is a run that
    cannot start, after a queue wait and however long MPI takes to come up.

    **Idempotent, and the folder wins.**  Anything already here was put here by
    an earlier prep, by `jobset init`, or by travelling with the folder;
    `copy_pseudopotentials` leaves it alone.  Only what is missing is fetched,
    from the library named by ``psml_lib``.

    **The species come from the STRUCTURE**, which `prep` has just loaded and
    checked against the description's witness — not from a list in the
    description.  A recorded list would be a second answer to *which elements
    is this calculation of*, and the structure is the first.

    A species in neither place stops `prep` **by name**, before a deck is
    written.

    **And then the science protocol runs on what is actually there.**
    `science/pseudopotentials.md` exists because a defective `S.psml` with a
    dead p-channel shipped into a real run on 2026-06-26: wrong sulfur bonding,
    and `propor: ERROR: IMAX=0` — but only at high rank counts, so a small run
    would have reported plausible, wrong numbers instead of crashing. The check
    that catches that class reads the pseudopotentials themselves.

    It has always run against ``psml_lib`` — the LIBRARY — and it is gated on
    that field being set, so a calculation whose files are already beside it and
    whose ``psml_lib`` is empty had **nothing checked at all**: the only thing
    said was *"psml_lib is not set … once set, this preflight will check
    coverage"*, while three real pseudopotentials sat in the folder the run
    would open them from. This step makes that state the normal one, so it runs
    the protocol here, against the **calculation** — which is where the files
    the run reads actually are — and refuses on the same ERROR statuses the
    preflight and `molbuilder pseudo check` refuse on, from the same shared
    constant.
    """
    from ..pseudos import psml_sources, resolve_psml_lib
    from ..siesta.input import copy_pseudopotentials
    from ..chemistry import species_order

    species = species_order(struct.elements)
    if not species:
        return
    pdir = _pseudo_dir(base)
    # THE FOLDER WINS, by the one rule the settings gate asks too.
    want = [s for s, d in psml_sources(species, dest_dir=base).items()
            if d is None]
    if not want:
        _screen_pseudos(species, cfg, pdir)
        return

    lib_raw = getattr(cfg, "psml_lib", None)
    if not lib_raw:
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)} and none are in {base.name}/, but no "
            f"pseudopotential directory is set.  Set `psml_lib` in the "
            f"template to the library they live in -- the convention is the "
            f"bare name `pseudopotential`, which means the projects tree "
            f"this calculation lives in (project-layout.md § 2.6, "
            f"job-contracts.md § 2.5a).")
    from ..pseudos import PsmlLibError
    try:
        lib = resolve_psml_lib(str(lib_raw), dest_dir=base)
    except PsmlLibError as exc:
        raise PrepError(str(exc))
    if not lib.is_dir():
        # Name the anchor the SPELLING asked for, in the rule's own words.
        # This used to print only the resolved path, which under the old
        # cascade was whichever candidate was tried last -- on Sol that was
        # `<calc>/projects/pseudopotential`, a folder assembled from the
        # user's working directory that nobody had chosen (2026-08-21).
        from ..pseudos import describe_psml_anchor
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)}, and the library they should come from is "
            f"not a directory.  "
            + describe_psml_anchor(str(lib_raw), dest_dir=base)
            + "  Put the .psml files there, or set `psml_lib` to a "
              "directory that has them.")
    missing = copy_pseudopotentials(want, lib, pdir)
    if missing:
        raise PrepError(
            f"this calculation needs {', '.join(f'{m}.psml' for m in missing)}"
            f" and there is none in {base.name}/ or in {lib}.  SIESTA opens "
            f"<element>.psml in the directory it runs from and has no search "
            f"path, so it would refuse at startup.  Put the file in the "
            f"library, or point `psml_lib` at one that has it.")
    _screen_pseudos(species, cfg, pdir)


def _screen_pseudos(species, cfg, base: Path) -> None:
    """`science/pseudopotentials.md` § 1, run on the calculation's own files.

    Same engine (`pseudos.check_coverage`), same severity set
    (`pseudos.ERROR_STATUSES`) and same XC-family table
    (`pseudos.expected_xc_family`) as the render-time preflight and the
    `molbuilder pseudo check` CLI, so the three surfaces cannot disagree about
    what blocks.  What differs is only WHICH DIRECTORY is read: this one asks
    about the files the run will open.

    The three blocking statuses are `missing`, `dead_projector` and
    `xc_family_mismatch` — a file absent, a valence channel physically absent,
    or the wrong XC family.  The rest are advisory, and the settings gate
    reports them in the deck's report: it reads these same files
    (`pseudos.psml_sources`), so printing them here too said each one twice.
    """
    from ..pseudos import ERROR_STATUSES, check_coverage, expected_xc_family
    entries = check_coverage(
        species, base,
        expected_xc_family=expected_xc_family(
            getattr(cfg, "xc_authors", "") or ""),
        expected_xc_authors=(getattr(cfg, "xc_authors", "") or "") or None,
    )
    blocking = [e for e in entries if e.status in ERROR_STATUSES]
    if blocking:
        raise PrepError(
            "the pseudopotentials in this calculation do not pass the "
            "screening (science/pseudopotentials.md § 1):\n  - "
            + "\n  - ".join(f"{e.element}: {e.message}" for e in blocking)
            + "\n  These are the checks that exist because a dead-channel "
              "S.psml once shipped into a real run -- wrong bonding, and a "
              "propor IMAX=0 crash that only appeared at high rank counts.")


def engine_seam(engine: str) -> EngineSeam:
    if engine == "siesta":
        from ..config.siesta import SiestaConfig
        from ..siesta.input import spec_for as _siesta_spec
        from ..siesta.stages import _traits, _warm_declaration
        return EngineSeam(config_cls=SiestaConfig, spec_for=_siesta_spec,
                          suffix=".fdf",
                          label_of=lambda cfg: cfg.system_label,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, system_label=label),
                          warm_for=_warm_declaration, traits_for=_traits,
                          sibling_artifacts=_siesta_sibling_artifacts,
                          provide_data=_siesta_provide_pseudos,
                          shared_package=_siesta_shared_package)
    if engine == "pyscf":
        from ..config.pyscf import PySCFConfig
        from ..pyscf.input import spec_for as _pyscf_spec
        from ..pyscf.stages import _traits as _pyscf_traits
        from ..pyscf.stages import _warm_declaration as _pyscf_warm
        # NO ``provide_data`` and NO ``sibling_artifacts``, and both absences
        # are ANSWERS rather than omissions (`script-preparation.md` § 4, W5):
        # PySCF's basis sets ship inside PySCF, so there is no file to put in
        # the calculation; and its script's own text instructs nothing to be
        # run beside it, so there is no promise to keep.
        return EngineSeam(config_cls=PySCFConfig, spec_for=_pyscf_spec,
                          suffix=".py",
                          label_of=lambda cfg: cfg.job_name,
                          relabel=lambda cfg, label: dataclasses.replace(
                              cfg, job_name=label),
                          warm_for=_pyscf_warm, traits_for=_pyscf_traits)
        # NO ``shared_package``: PySCF's basis sets ship inside PySCF, so
        # there is nothing in the calculation for every job to link to.
    raise PrepError(
        f"no deck writer for engine {engine!r}. An engine supplies its "
        f"catalogue rows and an answer at each preparation step "
        f"(script-preparation.md § 4); this backend has neither for that "
        f"name.")


def _siesta_shared_package(base: Path) -> List[str]:
    """SIESTA's shared package: the pseudopotentials it put in the folder,
    and the atom-permutation record when its decks are written from a sorted
    copy.

    The same files ``_siesta_provide_pseudos`` stages, named by the engine
    that staged them (`script-preparation.md` § 4, the data-files step).
    Under ``pseudos/`` since the layout repair (roadmap 7.10 M6); the bare
    root glob stays as the fallback for a bundle prepped before it, so a
    travelled calculation still names its package.

    THE PERMUTATION TRAVELS WITH THE RUNS, because a run of a sorted copy
    speaks the sorted order in every file it writes and the record is the
    one way back (`atom_permutation`, I7): every attempt holds its copy, so
    a SIESTA force-constant job's finish reads it beside the run it finishes
    (`engines/vibration.md` § 5.5).
    """
    from ..atom_permutation import PERMUTATION_FILE
    grouped = sorted(f"{PSEUDO_DIRNAME}/{p.name}"
                     for p in (base / PSEUDO_DIRNAME).glob("*.psml"))
    pseudos = grouped or sorted(p.name for p in base.glob("*.psml"))
    return pseudos + ([PERMUTATION_FILE]
                      if (base / PERMUTATION_FILE).is_file() else [])


__all__ = ["EngineSeam", "engine_seam"]

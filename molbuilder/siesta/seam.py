"""SIESTA's seam -- what SIESTA supplies for `prep` to run its steps over it
(`execution/script-preparation.md` § 4): its deck's spec, its sibling
artifacts, its data files (the pseudopotentials, screened and copied into
``pseudos/``), its shared package and its describe findings.  The part its
record in the engine table names (``seam``), resolved by
`jobset.engines.engine_seam`.  Floor 3 (`execution/architecture.md` § 2.1):
what it takes from `jobset/` it takes when it is called, so this module
loads beside the conductor that imports it.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from ..pseudos import PSEUDO_DIRNAME
from ..runfiles import PSEUDO_FILE, deck_role, fixed_name

if TYPE_CHECKING:
    from ..jobset.engines import EngineSeam


def _siesta_sibling_artifacts(struct, cfg, deck_path: Path, *,
                              kind: str, names, plan=None) -> None:
    """The sibling files a SIESTA deck's own text PROMISES.

    A charged deck instructs ``python3 makov_payne_correction.py`` in its
    header -- a promise this keeps.

    The charge is the electronic state's (`science/chemistry-correctness.md`
    § 2a) -- the one the deck beside it was written from.  The deck has just
    been written from that state, so a label naming no element cannot reach
    here.  ONLY FOR A FINITE SYSTEM (§ 2b): the script's formula is a
    molecule's in a vacuum box, and a charged slab or crystal's deck says it
    gets no formula."""
    from ..electronic_state import electronic_state
    from .makov_payne import emit_correction_script
    state = electronic_state(struct, cfg, kind=kind)
    q = state.net_charge.value
    if q != 0 and state.finite:
        emit_correction_script(fdf_path=deck_path,
                               system_label=cfg.system_label, q=q,
                               names=names, plan=plan)


def _siesta_describe_findings(struct, cfg, *, calculation: str,
                              dest_dir=None):
    """SIESTA's describe findings: its pseudopotentials, settled before a
    calculation is written (`web/handover-procedure.md` § 2.2) -- the check
    `prep` makes, asked of the folder the files go into -- and the refusal
    while a species has none."""
    from ..validation.siesta import pseudopotential_findings
    found = pseudopotential_findings(struct, cfg, calculation=calculation,
                                     dest_dir=dest_dir)
    if not any(i.severity == "error" for i in found):
        return found, None
    return found, ("the pseudopotentials are not settled: SIESTA needs a "
                   ".psml file for every species, named for it, so this "
                   "would write a "
                   "calculation that cannot start -- the findings are "
                   "beside the pseudopotential field "
                   "(handover-procedure.md 2.2)")


def pseudo_dir(base: Path, plan=None) -> Path:
    """THE PARENT'S DATA IS GROUPED (roadmap 7.10 M6): the calculation's
    pseudopotential copies live in ``pseudos/``, one folder, instead of
    N ``<label>.psml`` entries loose at the root.  Root strays -- put there
    by `molbuilder describe`, an earlier prep, or a travelled bundle -- are ADOPTED,
    the same move-in the deck adoption uses.  The run directories are
    untouched by this: each still receives ``<label>.psml`` beside the
    deck (that is SIESTA's own contract; it has no search path).

    ONE rule, two providers: the SIESTA arm below and the transport
    composite's (which fetches from the citation instead of a library).

    ``plan`` (`jobset.planned.Plan`) receives the folder, its stamp and the
    adoptions instead of the disk; without one they happen now.
    """
    from ..jobset.planned import Plan
    own = plan is None
    plan = Plan() if own else plan
    # A container, and it says so (`project-layout.md` § 1.4a): the shared
    # package holds files, never a run.  Unstamped, it would report a
    # calculation *running* because it has no result file in it -- the
    # answer § 1.4a exists to stop.  It reads back
    # as *support* rather than a stage, because its place is no stage's
    # (`calcdirs.where`); that is read off its name, not stored.
    from ..jobset.naming import open_container
    pdir = open_container(base, base / PSEUDO_DIRNAME, plan)
    for stray in plan.glob(base, "*.psml"):
        target = pdir / stray.name
        if not plan.is_file(target):
            plan.move(stray, target)
        else:
            plan.remove(stray)
    if own:
        plan.carry_out()
    return pdir


def siesta_provide_pseudos(struct, cfg, base: Path, plan=None) -> None:
    """Put the pseudopotentials this deck needs into the calculation.

    SIESTA opens ``<label>.psml`` in the directory it runs from and has no
    search path, so a missing file is not a preference — it is a run that
    cannot start, after a queue wait and however long MPI takes to come up.

    **Idempotent, and the folder wins.**  Anything already here was put here by
    an earlier prep, by `molbuilder describe`, or by travelling with the folder;
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

    This step runs the protocol against the **calculation** — which is where
    the files the run reads actually are — and refuses on the same ERROR statuses the
    preflight and `molbuilder pseudo check` refuse on, from the same shared
    constant.

    ``plan`` (`jobset.planned.Plan`) receives the copies; the screening
    reads the folder as the plan will leave it (`pseudos.PLANNED`): `prep`
    decides everything before it writes (`job-system.md` § 5.0).
    """
    from ..jobset.errors import PrepError
    from ..pseudos import psml_sources, resolve_psml_lib
    from .input import copy_pseudopotentials
    from ..chemistry import species_order

    species = species_order(struct.elements)
    if not species:
        return
    pdir = pseudo_dir(base, plan)
    # THE FOLDER WINS, by the one rule the settings gate asks too.
    want = [s for s, d in psml_sources(species, dest_dir=base).items()
            if d is None]
    if not want:
        screen_pseudos(species, cfg, pdir)
        return

    lib_raw = getattr(cfg, "psml_lib", None)
    if not lib_raw:
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)} and none are in {base.name}/, but no "
            f"pseudopotential directory is set.  Set `psml_lib` in the "
            f"template to the library they live in -- the convention is the "
            f"bare name `pseudopotential`, which means that folder in the "
            f"projects tree this calculation lives in (job-contracts.md "
            f"§ 2.5a).")
    from ..pseudos import PsmlLibError
    try:
        lib = resolve_psml_lib(str(lib_raw), dest_dir=base)
    except PsmlLibError as exc:
        raise PrepError(str(exc))
    if not lib.is_dir():
        # Name the anchor the SPELLING asked for, in the rule's own words.
        from ..pseudos import describe_psml_anchor
        raise PrepError(
            f"this calculation needs pseudopotentials for "
            f"{', '.join(want)}, and the library they should come from is "
            f"not a directory.  "
            + describe_psml_anchor(str(lib_raw), dest_dir=base)
            + "  Put the .psml files there, or set `psml_lib` to a "
              "directory that has them.")
    missing = copy_pseudopotentials(want, lib, pdir, plan=plan)
    if missing:
        raise PrepError(
            f"this calculation needs "
            f"{', '.join(fixed_name(PSEUDO_FILE, species=m) for m in missing)}"
            f" and there is none in {base.name}/ or in {lib}.  SIESTA opens "
            f"<label>.psml in the directory it runs from and has no search "
            f"path, so it would refuse at startup.  Put the file in the "
            f"library, or point `psml_lib` at one that has it.")
    screen_pseudos(species, cfg, pdir)


def screen_pseudos(species, cfg, base: Path) -> None:
    """`science/pseudopotentials.md` § 1, run on the calculation's own files.

    Same engine (`pseudos.check_coverage`), same severity set
    (`pseudos.ERROR_STATUSES`) and same XC-family table
    (`pseudos.expected_xc_family`) as the render-time preflight and the
    `molbuilder pseudo check` CLI, so the three surfaces cannot disagree about
    what blocks.  What differs is only WHICH DIRECTORY is read: this one asks
    about the files the run will open.

    What blocks is `pseudos.ERROR_STATUSES` — a file absent or under a name
    SIESTA does not open, a valence channel physically absent or claiming
    zero strength, or the wrong XC family.  The rest are advisory, and the settings gate
    reports them in the deck's report: it reads these same files
    (`pseudos.psml_sources`), so printing them here too said each one twice.
    While a prep is planned it reads the folder as the plan will leave it
    (`pseudos.PLANNED`).
    """
    from ..jobset.errors import PrepError
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


def _siesta_shared_package(base: Path, plan=None) -> List[str]:
    """SIESTA's shared package: the pseudopotentials it put in the folder --
    the same files ``siesta_provide_pseudos`` stages, named by the engine
    that staged them (`script-preparation.md` § 4, the data-files step),
    under ``pseudos/``.

    ``plan`` -- the folder as a prep will leave it (`jobset.planned`).
    """
    from ..jobset.planned import Plan
    view = plan if plan is not None else Plan()
    return sorted(f"{PSEUDO_DIRNAME}/{p.name}"
                  for p in view.glob(base / PSEUDO_DIRNAME, "*.psml"))


def siesta_seam() -> EngineSeam:
    """SIESTA's seam -- the part its record in the engine table names
    (``seam``)."""
    from ..jobset.engines import EngineSeam, relabel
    from .input import spec_for as _siesta_spec
    from .stages import traits_for, warm_for
    return EngineSeam(spec_for=_siesta_spec,
                      suffix=deck_role("siesta"),
                      relabel=relabel("siesta"),
                      warm_for=warm_for, traits_for=traits_for,
                      sibling_artifacts=_siesta_sibling_artifacts,
                      provide_data=siesta_provide_pseudos,
                      shared_package=_siesta_shared_package,
                      describe_findings=_siesta_describe_findings)


__all__ = ["siesta_seam", "pseudo_dir", "screen_pseudos",
           "siesta_provide_pseudos"]

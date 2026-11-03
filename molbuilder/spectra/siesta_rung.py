"""A SIESTA VIBRATION'S RUNG -- the vibration kind's own module choosing by
engine (`execution/architecture.md` § 3.3): its builder,
:func:`siesta_vibration_rung_of`, is the pair's ``rung`` part, which the
conductor resolves (`jobset.prep`); one run of points in phases
(`engines/vibration.md` § 5.11), its force-constant deck's `vibration` block
(:func:`_vibration_block`) built where the stage's resolved config is held.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from .. import calcdirs
from ..jobset.rung import Rung, structure_for


def _vibration_block(stage: str, cfg, *, criterion, criterion_from: str,
                     struct) -> dict:
    """A SIESTA force-constant deck's `vibration` block, format v5: the
    facts the run's finish reads and no SIESTA keyword states
    (`engines/vibration.md` § 5.3, § 5.11 § 4), built here because only
    `prep` holds all of them -- the stage, its resolved config, the
    molbuilder rendering the deck -- and *Raman for*, the modes the optical
    phase applies (`select_modes`), the one fact of that phase no keyword
    states.  ``criterion`` is the relaxation phase's own ``relax_force_tol``
    -- the force the reference geometry is relaxed to, and judged by.
    ``struct`` is the structure the deck is written from, whose atoms'
    masses the block states (`chemistry.atomic_mass`, I3).  The relaxation
    record is the `relax` call's, read from the run's index at the finish
    (§ 5.11 § 3), never written here."""
    from .. import __version__ as _mb_version
    from ..chemistry import atomic_mass
    from ..siesta.vibration_phases import OPTICAL, phases_on
    from .siesta_vibration import vibration_record
    from ..structure import IDENTITY_FIELDS
    from ..template import inactive_settings
    # WHAT ENTERS, asked of its declarations: the optical phase of the phase
    # table's switch (`phases_on`), each of Raman for's ends of its
    # `active_when` -- a setting that enters nothing is not stated as if it
    # did (`template.inactive_settings`).
    off = inactive_settings(cfg, engine="siesta")

    def _entering(name: str):
        return None if name in off else getattr(cfg, name)
    return vibration_record(
        stage=stage,
        force_criterion_ev_ang=criterion,
        force_criterion_from=criterion_from,
        already_relaxed=bool(getattr(cfg, "already_relaxed", False)),
        temperature_K=float(cfg.temperature_K),
        masses_amu=[atomic_mass(el) for el in struct.elements],
        molbuilder_version=str(_mb_version),
        # RAMAN FOR, as the optical phase applies it -- none when the
        # stage runs no optical phase.
        raman=({"modes": str(cfg.raman_modes),
                "freq_min_cm1": _entering("raman_freq_min_cm1"),
                "freq_max_cm1": _entering("raman_freq_max_cm1")}
               if any(p.name == OPTICAL for p in phases_on(cfg)) else None),
        # WHAT NO OTHER BLOCK OF THE DECK CARRIES of the structure the result
        # states (engines/vibration.md 6.10.1).
        title=str(struct.title or ""),
        identity={k: getattr(struct, k) for k in IDENTITY_FIELDS})


def siesta_vibration_rung_of(calc, pset, *, seam, sweep, log, plan,
                             reuse=None) -> Rung:
    """A SIESTA vibration's rung -- ONE RUN OF POINTS (`engines/vibration.md`
    § 5.11): the relaxation, the force constants and the optical calls,
    each phase a point of the run with its own deck, written for its part
    by the one deck writer (§ 5.11 § 4) from the structure as given -- the
    atoms in the order the force-constant run needs, the held ones first,
    on a COPY whose order record its deck states and the job's finish
    inverts (`model/overview.md` § 2.2a, I5); the user order never reaches
    the engine and the sorted order never reaches a person.  Every phase's
    deck is written at the structure's coordinates; what a call computes
    at is its own `coordinates.fdf`, written by the run's plan step
    (§ 5.11 § 3).  Which phases run is the stage's config's: each `PHASES`
    row's switch (`already_relaxed` turns the relaxation off,
    `compute_raman` off the optical calls).  Each phase's run script
    is rendered with its own `resumes` (:attr:`Rung.wrapper_at`); its
    report steps and the kind's own steps are every run of points'
    (`PointOfRun`, the kind table's ``steps``)."""
    from ..siesta.vibration_phases import (FC, RELAX, Call, phase, phases_on,
                                           run_size_words)
    from ..kind_table import atom_order
    from ..transport.sort import sort_by
    from ..warmfiles import warm_list
    from ..jobset.naming import stage_home
    base, task = calc.base, calc.task
    # THE GEOMETRY IN THE USER ORDER, then sorted as every prep sorts
    # (`model/overview.md` § 2.2a, I3-I5): the structure as given, every
    # phase's deck at it.  THE ORDER RECORD RIDES THE COPY into its deck's
    # ATOM-METADATA block (I5): the deck states the way back, beside the job.
    struct = structure_for(task, base, pset)
    key = atom_order(task.calculation, task.engine, pset.stage)
    struct = sort_by(struct, key).structure
    log.step(f"the atom order the engine needs -- {key}, {struct.n_atoms} "
             f"atoms; the deck states its order record")
    cfg0 = pset[0].render_config()
    # THE PHASES THIS STAGE RUNS, by the table's switches (§ 5.11 § 1) --
    # each a point of the run: its folder below the stage, holding the
    # phase's deck and run script every call of it copies (§ 5.11 § 2).
    on = phases_on(cfg0)
    stage_dir = stage_home(base, task, pset.stage).dir
    points = tuple((calcdirs.prepared_folder(stage_dir, c.prepared), c)
                   for c in (Call(p.name, p.name) for p in on))
    log.step("the run's phases: " + ", ".join(p.name for p in on))
    # THE FINISH'S FORCE CRITERION is the relaxation phase's own tolerance
    # (`engines/vibration.md` § 5.5): the one yardstick the run relaxes to
    # and judges the reference geometry by -- said with where it came from.
    criterion = getattr(cfg0, "relax_force_tol", None)
    criterion_from = "the stage's relax_force_tol"

    def spec_extra(element, cfg, point):
        # THE PART THE DECK IS -- the point's phase; a benchmark trial's
        # the force constants' (a trial measures the one call that costs).
        part = point.phase if point is not None else FC
        return {"part": part,
                # The finish's facts ride the force-constant deck alone --
                # never a benchmark trial's: a spectrum of capped SCFs is a
                # spectrum of nothing (§ 5.5).
                **({"vibration": _vibration_block(
                        pset.stage, cfg, criterion=criterion,
                        criterion_from=criterion_from, struct=struct)}
                   if part == FC and not element.is_trial else {})}

    def wrapper_at(point):
        # EACH PHASE'S RUN SCRIPT (§ 5.11 § 6): the phase's `resumes` from
        # its warm-files section -- what its Mode line says of the restart
        # files it finds; what a launch again does with the point is its
        # declared relaunch (`PointOfRun.relaunch`).
        return {"resumes": warm_list(task.engine, phase(point.phase).warm_kind,
                                     base).resumes}

    def role_at(point):
        # THE ROLE A PHASE PLAYS (`PHASES`' role): its deck states the rows
        # that role reads, and its run script's record of what the engine
        # will read lists them -- never another phase's settings, nor the
        # relaxation's `restart` in a call that fixes its own start (§ 5.11
        # § 4).  A benchmark trial's deck, at no point, plays every role.
        return phase(point.phase).role if point is not None else None

    def beside(point, spec, folder, plan):
        # THE STRUCTURE'S COORDINATES, beside every phase's deck (§ 5.11
        # § 4): the rows the deck writer derived, as the deck reads them.
        from ..runfiles import COORDINATES_FILE
        plan.text(Path(folder) / COORDINATES_FILE,
                  spec.derived["coordinates_file"])

    # THE COUNT AND THE SIZES, beside the wall (§ 5.11 § 9): the free atoms
    # are those the held label leaves; the modes the run reports, the rank
    # rule's -- the one count prep's check states too.
    from ..validation.spectra import modes_of
    n_free, _n_rigid, n_modes = modes_of(struct, task.engine)

    def job_facts(job):
        # THE RUN'S PHASES, on its job (`Job.phases`): what launch opens and
        # walks, what the run's own python follows.  AND WHAT A LAUNCH AGAIN
        # CARRIES (§ 5.11 § 6): the relaxation call's files, by its row's
        # section (`warm_kind`) -- none without a relaxation, nor where the
        # stage starts clean (`_job_for` gave it none).
        from ..jobset.model import warm_declaration
        relaxes = any(p.name == RELAX for p in on)
        return dataclasses.replace(
            job, phases=[p.name for p in on],
            warm=(warm_declaration(task.engine, task.label,
                                   phase(RELAX).warm_kind, base)
                  if relaxes and job.warm else []))
    return Rung(struct=struct, render_kind=task.calculation,
                said=(run_size_words(cfg0, n_free, n_modes),),
                spec_extra=spec_extra, points=points, wrapper_at=wrapper_at,
                beside=beside, role_at=role_at,
                relaxes_at=lambda pt: pt is not None and pt.phase == RELAX,
                job_facts=job_facts)


__all__ = ["siesta_vibration_rung_of"]

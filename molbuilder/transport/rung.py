"""A TRANSPORT RUNG, built from the description as it reads now -- the
transport kind's own module (`execution/architecture.md` § 3.3): its
builder, :func:`transport_rung_of`, is the pair's ``rung`` part, which the
conductor resolves (`jobset.prep`); its junction read from its record
(:func:`junction_record`; composed from the citation by prep,
`gather.composed_for_prep`), the template ⊕ the rung's overrides ⊕ its run
card resolved (:func:`resolve_transport`), its structure and electronic
state (:func:`transport_parts`), each deck's spec at its point
(:func:`rung_spec`, :func:`at_frame`, :func:`at_point`) and its
pseudopotentials (:func:`provide_pseudos`).

Asked by the conductor through the builder, and by the gather, which renders
an upstream rung NOW to compare it with the deck its run ran
(`gather.rung_deck_now`; `engines/transport.md` § 6.1) -- one builder, so
the two cannot render a rung two ways.
"""
from __future__ import annotations

import dataclasses
from typing import Optional
from pathlib import Path

from ..pseudos import PSEUDO_DIRNAME
from ..runfiles import PSEUDO_FILE, fixed_name
from ..siesta.seam import pseudo_dir, screen_pseudos
from ..jobset.errors import PrepError, user_error_as_prep
from ..jobset.naming import stage_home
from ..jobset.rung import Rung, citation_of


def provide_pseudos(struct, cfg, base: Path,
                    citation: str, *, plan=None) -> None:
    """A cited run's pseudopotentials arrive FROM THE RUN
    (engines/transport.md § 3.1: the files the junction relaxed with travel
    with it — one template governs).

    Idempotent, and the folder wins, exactly like the SIESTA arm: what
    ``pseudos/`` already holds (an earlier prep, or the travelled
    folder) is left alone; only missing species are fetched, from the
    cited calculation's own ``pseudos/`` — the files the junction
    actually relaxed with.  Then the science screening runs on what is
    actually here, same protocol, same blocking statuses.
    """
    from ..projects import find_projects_root
    from ..pseudos import psml_sources
    from ..siesta.input import copy_pseudopotentials
    from ..chemistry import species_order

    species = species_order(struct.elements)
    if not species:
        return
    pdir = pseudo_dir(base, plan)
    # THE FOLDER WINS, by the one rule the settings gate asks too.
    want = [s for s, d in psml_sources(species, dest_dir=base).items()
            if d is None]
    if want:
        root = find_projects_root(base)
        lib = None
        if root is not None:
            cited = Path(root) / citation
            if (cited / PSEUDO_DIRNAME).is_dir():
                lib = cited / PSEUDO_DIRNAME
            elif any(cited.glob("*.psml")):
                lib = cited
        missing = (copy_pseudopotentials(want, lib, pdir, plan=plan)
                   if lib is not None else list(want))
        if missing:
            raise PrepError(
                f"this transport calculation needs "
                f"{', '.join(fixed_name(PSEUDO_FILE, species=m) for m in missing)}"
                f" and there is none in {base.name}/{PSEUDO_DIRNAME}/ or in "
                f"the cited "
                f"directory ({citation}).  The pseudopotentials travel "
                f"with the citation (engines/transport.md § 3.1): a "
                f"relaxation run of molbuilder's own holds the .psml it "
                f"ran with, so relax the junction through `molbuilder describe` -> "
                f"`prep task` -> `launch task` and cite that run.")
    screen_pseudos(species, cfg, pdir)


def resolve_transport(task, stage: str, allocation, *, template_text,
                      run_settings=None, log=None):
    """Floor 3's step 2 for a transport rung — the template ⊕ its overrides
    ⊕ its run card's settings (``run_settings``, `job-request.md` § 4) --
    from the
    template's text as the prep read it, once (`script-preparation.md`
    § 3.0).

    **The same `resolve` every other kind uses**: a transport calculation
    has a template, so it decides through the one conductor that exists.

    Returns the rung's :class:`~molbuilder.resolve.ResolvedConfig` — **the
    whole element, not just its values**. What is gained over assembling it by
    hand is **provenance**: every value says whether the template, the stage or
    a pin set it, which is the whole of what the pipeline log had nothing to
    print.

    THE RESOURCES RIDE ON THE ELEMENT: `resolve` folds ``wall_slack``
    and ``use_gpu`` onto each element's resources, both config answers that
    the WRAPPER reads.
    """
    from ..resolve import ResolveError, resolve

    # A SHARED VALUE IS NOT A PER-STAGE OVERRIDE, NOR ONE THE RUNG FIXES --
    # refused by `resolve`, the one door every kind's prep goes through
    # (`template.shared_by_every_stage` + `why_shared`; `fixed_by_role` +
    # `why_role`), which also lays each rung's own answers on its config.
    # AND THE RUNG THAT READS A VALUE (`stages`, the same § 6.4) is asked at
    # that one door too, for every kind (`template.unread_overrides`, plan
    # § 5w K4): a transmission window on the seed's rung is refused there
    # by name.

    if log is not None:
        log.phase("STEP 2 · RESOLVE — the description becomes a ParameterSet")
        log.received("stage", stage)
    try:
        ps = resolve(template_text, task,
                     allocation=allocation, stage=stage,
                     run_settings=run_settings)
    except ResolveError as exc:
        # `resolve` translates the template's and the overrides' refusals
        # (ValueError) into its own.
        raise PrepError(str(exc)) from exc
    # ONE ELEMENT.  A transport rung is a production run; its one axis is the
    # bias, and that is the device's own directory level rather than a sweep
    # (engines/transport.md § 2a.11).  `resolve` still returns a list, because
    # the length is the whole of the difference between a run and a sweep and
    # no reader below floor 7 may branch on which.
    element = ps.elements[0]
    if log is not None:
        log.produced("elements", f"{len(ps)} (a run, not a sweep)")
        # THE KIND'S VALUES, the ones `resolve` says who set.
        for _name in sorted(n for n in element.provenance
                            if n in element.said):
            log.chose(_name, getattr(element.values, _name, None),
                      element.said[_name])
    return element


def transport_parts(task, stage: str, composed, config):
    """``(struct, configure, state)`` of one transport rung: the structure its
    decks describe, the config they render from, and the junction's
    electronic state.

    ONE DOOR, asked by the conductor through the rung
    (:func:`transport_rung_of`) and by the gather when it renders an
    upstream rung NOW (:func:`transport_rung`, plan § 5w K11, T-F30).
    """
    from ..kind_table import lead_of, rung_label
    # (i) THE STRUCTURE.  Two of them, out of the one cited file: the
    # junction, and the lead taken out of it by its region label -- same
    # atoms, same relaxation, a subset rather than a geometry derived from
    # anywhere else.  That is what lets the seam stay `spec_for(struct,
    # cfg, ...)`: a deck describes a structure, and these are two.
    region = lead_of(stage)
    if region is not None:
        # THE LEAD BY ITS REGION, as the kind table names it (`leads`).
        from .sort import REGION_LEFT_ELECTRODE
        model = (composed.electrode_left if region == REGION_LEFT_ELECTRODE
                 else composed.electrode_right)
        struct = model.as_structure()
        # The lead's identity IS the .TSHS stem the device deck names --
        # one door, `rung_label`, which names its files too.
        label = rung_label(task.label, stage)
    else:
        # The junction -- and for the device it carries the region
        # partition the NEGF block is built from.
        struct, label = composed.junction, task.label
        # A FRAME SET (`engines/transport.md` § 2a.9): the seed runs once,
        # on frame 0; the device and the transmission run per frame, each
        # point's deck its frame (`at_frame`).
        from .stages import per_point_rungs
        if struct.n_frames > 1 and stage not in per_point_rungs():
            struct = struct.frame_at(0)

    def labelled(cfg):
        # THE RUNG FIXES ITS LABEL (C2): every rung's `SystemLabel` is spelt
        # from `task.label` -- the junction's rungs by it, a lead by
        # `rung_label` -- so the device deck's `TS.Elec.<name> HS` line, the
        # gather and the run door name the files the lead wrote.  The
        # template's `system_label` never reaches a transport deck.  Written
        # through the engine's seam, on the field its record names.
        from ..jobset.engines import engine_seam
        return engine_seam(task.engine).relabel(cfg, label)
    # THE ELECTRONIC STATE BELONGS TO THE CALCULATION (ES1,
    # `science/chemistry-correctness.md` § 2a) -- and on a transport ladder
    # that is physics, not bookkeeping: TranSIESTA joins the leads'
    # self-energies to the device, so every rung must solve the same spin
    # channels.  A blank spin is decided ONCE, on the JUNCTION, and every
    # rung -- a lead included -- is handed that answer.  Decided per rung, a
    # molecule with an open-d centre would polarize the device beside
    # non-polarized leads.
    # The VALUES are folded into the config, so every reader of the rung's
    # config -- the gate, the record, the pseudopotential screening -- reads
    # the junction's answer; the STATE itself, with where each value came
    # from, is handed to the deck writer, which says so in every rung
    # (§ 2a.5).  Folded alone, the rungs read it as *stated*.
    from ..electronic_state import electronic_state
    state = electronic_state(composed.junction, labelled(config),
                             kind=task.calculation)
    # A STATE TRANSIESTA CANNOT RUN is refused HERE, on the junction, where
    # each value still says where it came from (`template.md` § 6.3a, ES4):
    # once folded into the rungs' configs, every rung's gate would call a
    # recorded or detected value *stated*.
    from ..validation.chemistry import check_electronic_state
    refused = [i for i in check_electronic_state(
        composed.junction, labelled(config), calculation=task.calculation,
        engine=task.engine, state=state)
        if i.severity == "error"]
    if refused:
        raise PrepError(refused[0].message)

    def configure(cfg):
        return dataclasses.replace(
            labelled(cfg),
            spin_treatment=state.spin_treatment.value,
            unpaired_electrons=state.unpaired_electrons.value)
    return struct, configure, state


def transport_rung(task, stage: str, composed, allocation, *,
                   template_text, run_settings=None, log=None):
    """``(struct, config, state, element)`` of one transport rung, resolved
    NOW from the current template, junction and run card -- what the gather
    renders an upstream rung from (:func:`rung_deck_now`): the template ⊕
    the rung's overrides ⊕ its run card (:func:`resolve_transport`), then
    the parts the conductor's rung reads too (:func:`transport_parts`)."""
    element = resolve_transport(task, stage, allocation,
                                 template_text=template_text,
                                 run_settings=run_settings, log=log)
    struct, configure, state = transport_parts(task, stage, composed,
                                                element.render_config())
    return struct, configure(element.render_config()), state, element


def transport_sources(element, *, engine: str):
    """Who set each value a transport rung's decks write: `resolve`'s rows,
    the label said as the rung's -- it spells every rung's ``SystemLabel``
    from the calculation's (:func:`transport_parts`, C2).  Asked by both
    roads that render a rung, so the deck the gather compares with says
    what the prepared one says."""
    from ..engine_table import engine as _engine
    from ..resolve import SET_AFTER, said_by
    return said_by(element.deck_sources,
                   _engine(engine).identity_field,
                   SET_AFTER["rung_label"])


def compose_refusals() -> tuple:
    """What reading or composing a junction raises for the person to fix --
    the composition's, the sort's, the order record's and a cited run's
    deck record -- said by every door that asks as its refusal
    (:class:`PrepError`): prep (`gather.composed_for_prep`) and the gather
    (`gather.composed_junction`) alike."""
    from ..atom_permutation import PermutationError
    from ..deck_record import DeckRecordUnreadable
    from .compose import ComposeError
    from .sort import SortError
    return (ComposeError, SortError, PermutationError, DeckRecordUnreadable)


def junction_record(calc):
    """The junction record beside ``task.json``, read for THIS citation --
    ``(the junction or None, why, the projects tree's root)``: ``why`` the
    reasons a record found is not usable (`compose.load_compose_record`).
    The one read of it, by prep and the gather; refused as
    :func:`compose_refusals` say.  The tree root is resolved BEFORE the
    record is loaded: the reload holds a cited pair to the bytes it was
    composed from (`compose.load_compose_record`)."""
    from ..projects import find_projects_root
    from .compose import load_compose_record
    base, task = calc.base, calc.task
    try:
        root = find_projects_root(base)
        why: list = []
        composed = load_compose_record(base, citation=citation_of(task),
                                       engine=task.engine, tree_root=root,
                                       why=why,
                                       swap_electrodes=task.swap_electrodes)
    except compose_refusals() as exc:
        raise PrepError(str(exc)) from exc
    return composed, why, root


def written_and_read_back(base, composed, plan, *, root, task):
    """``composed``'s record, written by its one writer
    (`compose.write_compose_record`) to a temporary folder outside the
    calculation and READ BACK from there -- the same round trip a later prep
    makes from the calculation's own copy -- its bytes into ``plan``; the
    junction as read back.

    RENDER FROM THE RECORD, on this prep as on every later one.  A fresh
    composition is the cited `.XV` at full float precision; the record is
    the codec's text of it, and every later read -- the other rungs, a
    re-prep -- reads the record.  Rendering
    this one from the fresh copy gave two preps of the SAME calculation two
    sets of numbers ~1e-10 apart (measured 2026-09-25 on the ENGINE-OFFSET
    record), and a deck rendered from each differs."""
    import tempfile
    from .compose import load_compose_record, write_compose_record
    why: list = []
    with tempfile.TemporaryDirectory(prefix="molbuilder-compose-") as tmp:
        names = write_compose_record(tmp, composed)
        back = load_compose_record(tmp, citation=citation_of(task),
                                   engine=task.engine, tree_root=root,
                                   why=why,
                                   swap_electrodes=task.swap_electrodes)
        if back is None:
            raise PrepError(
                f"the composed junction was written and could not be read "
                f"back: {why[-1] if why else 'no reason given'}")
        for name in names:
            plan.bytes(Path(base) / name, (Path(tmp) / name).read_bytes())
    return back


def rung_spec(task, stage: str, struct, config, state, point=None, *,
                   base=None, contours=None):
    """One transport deck's ``(spec, cfg)`` -- its point's when ``point``
    is given: the config at its voltage (the point is the rung's answer,
    `engines/template.md` § 6.4; a single-bias rung keeps the 0 V `resolve`
    laid on) and the deck naming its run's shared files the way the point
    reaches them (`engines/transport.md` § 2a.11); ``contours`` a
    transmission point's sharing with a finished calculation's point
    (`transport.energies.Contours`, § 2a.16)."""
    from ..jobset.engines import engine_seam
    cfg = at_point(config, point)
    # THE RUNG'S NAMES -- the one composer (`StageHome.names`): a lead's
    # on its own stem, the junction's rungs on the calculation's label.
    names = stage_home(base, task, stage).names
    with user_error_as_prep():
        try:
            spec = engine_seam(task.engine).spec_for(
                struct, cfg, names=names, calculation=task.calculation,
                **spec_keywords(task, state, point, base, contours))
        except ValueError as exc:
            # The deck's spec (`transport.deck.transport_spec`, through
            # the engine's seam) refuses an unknown rung with a message
            # written FOR a person, and `user_error_as_prep` translates
            # only ValidationError / RuntimeConfigError / WrapperError --
            # deliberately, so a TypeError still looks like the bug it is.
            raise PrepError(str(exc)) from exc
    return spec, cfg


def spec_keywords(task, state, point, base, contours) -> dict:
    """WHAT A TRANSPORT DECK'S SPEC IS TOLD beside its structure and config
    -- the one composition both roads give the engine's writer (prep's
    rung, the gather's "same calculation" check, `engines/transport.md`
    § 6.1): its electronic state, its point (the device names its run's
    shared files the way its point reaches them, § 2a.11), where the
    device keeps the leads' files, and what a transmission point computes
    where it is not its window whole (§ 2a.16)."""
    return {"state": state, "point": point,
            "scratch_dir": scratch_dir_of(base, engine=task.engine,
                                          calculation=task.calculation),
            "contours": contours}


def scratch_dir_of(base, *, engine: str,
                   calculation: str) -> Optional[str]:
    """The folder a device run keeps its leads' files in -- the
    vocabulary's ``scratch_dir`` (`warm-files.toml`, the calculation's
    section on its engine), the one spelling every writer and reader
    takes."""
    from ..warmfiles import warm_list
    return warm_list(engine, calculation, base).scratch_dir


def at_frame(struct, point):
    """The structure a point's deck describes: its frame of the cited set
    (`engines/transport.md` § 2a.9), or the junction as it is for a point
    with no frame level."""
    return (struct.frame_at(point.frame)
            if point is not None and point.frame is not None else struct)


def at_point(cfg, point):
    """A transport deck's config at its point -- its voltage
    (:func:`at_bias`); a point with no voltage level runs at the 0 V
    `resolve` laid on."""
    return at_bias(cfg, point.volts if point is not None else None)


def at_bias(cfg, volts):
    """A transport deck's config at its bias point: the point is the rung's
    answer, the one `role` value the catalogue does not hold
    (`engines/template.md` § 6.4) -- the list is the bias's only home
    (`engines/transport.md` § 2a.10), and a deck with no point keeps the
    0 V `resolve` laid on."""
    return (cfg if volts is None
            else dataclasses.replace(cfg, bias_voltage_v=float(volts)))


def transport_rung_of(calc, pset, *, seam, sweep, log, plan,
                      reuse=None) -> Rung:
    """A transport rung (`engines/transport.md`): its four steps of its own.

    **compose** -- the junction and its leads, composed from the cited
    relaxation, or read from the record that travelled beside ``task.json``
    (:func:`composed_for_prep`); the rung's structure is the junction or
    the lead taken out of it (:func:`transport_parts`).  **electronic
    state** -- decided once, on the junction, folded into every rung.
    **points** -- a sweep's rung writes one deck per bias point, each in its
    own folder (`transport.stages.point_folders`).  Its data files come
    from the citation, and its job carries the rung's own restart files and,
    for the transmission, TBtrans's program.  The fifth, **gather** -- each
    rung's inputs from the runs upstream -- is its first launch's
    (:func:`gather_sources`, `job-system.md` § 6.0).  ``reuse`` is what
    the kind's own reuse decided for this prep (`reuse.decide`, its
    description's ``reuse`` part), from which the rung's points' contours
    are its own (:func:`_contours`)."""
    from .gather import composed_for_prep
    from .reuse import point_key
    base, task, stage = calc.base, calc.task, pset.stage
    contours = _contours(calc, stage, reuse)
    # NO SWEEP REACHES IT: `prep bench` refuses the kind first, in its
    # record's words (`no_benchmark`, `prep_inputs.bench_refusal`).
    composed = composed_for_prep(calc, plan)
    struct, configure, state = transport_parts(task, stage, composed,
                                                pset[0].render_config())
    from .stages import point_folders, warm_declaration
    # THE RUNG'S POINTS (`transport.stages.rung_points`): a frame of the
    # cited set and a voltage each, the frames the junction composed here.
    points = tuple(point_folders(base, task, stage,
                                 frames=composed.junction.n_frames))
    citation = citation_of(task)

    def provide_data(plan):
        # The pseudopotentials: a cited run's travel with it, a cited pair's
        # come from the library the template names (`engines/transport.md`
        # § 3.1) -- screened against the config the rung's decks render
        # from: what it checks is each file's XC family against the
        # functional the run will ask for.
        cfg = configure(pset[0].render_config())
        if composed.provenance["kind"] == "pair":
            seam.provide_data(composed.junction, cfg, base, plan=plan)
        else:
            provide_pseudos(composed.junction, cfg, base, citation,
                            plan=plan)

    def job_facts(job):
        # The rung's own restart files (`transport.stages.warm_declaration`:
        # the seed's and the device's; the leads and the transmission carry
        # none).  The program its deck is read by and the scratch folder its
        # run keeps are the deck's own statement (`DeckSpec.program`,
        # ``scratch_dir``), taken onto the job by `_job_for`.
        return dataclasses.replace(
            job, warm=warm_declaration(stage, task.label, base))
    return Rung(struct=struct, render_kind=task.calculation, configure=configure,
                sources=lambda element: transport_sources(
                    element, engine=task.engine),
                # THE DECK'S POINT reaches the spec: the device names its
                # run's shared files the way its point reaches them
                # (`engines/transport.md` § 2a.11).
                spec_extra=lambda element, cfg, point: spec_keywords(
                    task, state, point, base,
                    # a rung that runs once, its one point
                    (contours or {}).get(point_key(point))),
                points=points, at_point=at_point,
                struct_at=lambda pt: at_frame(struct, pt),
                provide_data=provide_data, job_facts=job_facts)


def _contours(calc, stage: str, reuse) -> Optional[dict]:
    """WHAT EACH POINT OF THIS RUNG COMPUTES where it is not its window
    whole, by the point's key (`reuse.point_key`;
    `engines/transport.md` § 2a.16): a launched rung's as its decks record
    it -- never decided again -- else the prep's reuse decision's
    (``reuse``), else ``None``."""
    from ..jobset.materialize import ever_launched
    from .reuse import launched_contours
    job = calc.job(stage)
    if job is not None and ever_launched(calc, job):
        return launched_contours(calc, stage)
    return reuse.contours_for(stage) if reuse is not None else None


__all__ = ["provide_pseudos", "resolve_transport", "transport_parts",
           "scratch_dir_of", "transport_rung", "transport_sources",
           "compose_refusals", "junction_record", "written_and_read_back",
           "rung_spec", "at_frame", "at_point", "at_bias",
           "transport_rung_of"]

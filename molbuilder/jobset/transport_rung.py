"""A TRANSPORT RUNG, built from the description as it reads now -- its
junction composed from the citation (:func:`composed_for_prep`), the
template ⊕ the rung's overrides ⊕ its run card resolved
(:func:`resolve_transport`), its structure and electronic state
(:func:`transport_parts`), each deck's spec at its point
(:func:`transport_spec`, :func:`at_frame`, :func:`at_point`) and its
pseudopotentials (:func:`provide_pseudos`).

Asked by prep's conductor, which writes the rung (`prep._transport_rung_of`),
and by the gather, which renders an upstream rung NOW to compare it with the
deck its run ran (`gather.rung_deck_now`; `engines/transport.md` § 6.1) --
one builder, so the two cannot render a rung two ways.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from ..pseudos import PSEUDO_DIRNAME
from ..runfiles import RunNames
from .engines import pseudo_dir, screen_pseudos
from .errors import PrepError, user_error_as_prep
from .materialize import stage_home


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
                f"{', '.join(f'{m}.psml' for m in missing)} and there is "
                f"none in {base.name}/pseudos/ or in the cited "
                f"directory ({citation}).  The pseudopotentials travel "
                f"with the citation (engines/transport.md § 3.1): a "
                f"relaxation run of molbuilder's own holds the .psml it "
                f"ran with, so relax the junction through `jobset init` -> "
                f"`prep task` -> `launch task` and cite that run.")
    screen_pseudos(species, cfg, pdir)


def resolve_transport(task, stage: str, allocation, *, template_text,
                      pins=None, log=None):
    """Floor 3's step 2 for a transport rung — the template ⊕ its overrides
    ⊕ its run card's settings (``pins``, `stages.md` § 6.8d) -- from the
    template's text as the prep read it, once (`script-preparation.md`
    § 3.0).

    **The same `resolve` every other kind uses**: a transport calculation
    has a template, so it decides through the one conductor that exists.

    Returns the rung's :class:`~molbuilder.resolve.ResolvedConfig` — **the
    whole element, not just its values**. What is gained over assembling it by
    hand is **provenance**: every value says whether the template, the stage or
    a pin set it, which is the whole of what the pipeline log had nothing to
    print.

    THE RESOURCES RIDE ON THE ELEMENT: `resolve` folds ``continue_retries``
    and ``use_gpu`` onto each element's resources, both config answers that
    the WRAPPER reads.
    """
    from ..config.siesta import SiestaConfig
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
        ps = resolve(template_text, task, SiestaConfig,
                     allocation=allocation, stage=stage, pins=pins)
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
        for _name, _src in sorted(element.provenance.items()):
            log.chose(_name, getattr(element.values, _name, None), _src)
    return element


def transport_parts(task, stage: str, composed, config):
    """``(struct, configure, state)`` of one transport rung: the structure its
    decks describe, the config they render from, and the junction's
    electronic state.

    ONE DOOR, asked by the conductor through the rung
    (:func:`_transport_rung_of`) and by the gather when it renders an
    upstream rung NOW (:func:`transport_rung`, plan § 5w K11, T-F30).
    """
    from ..transport.transiesta import electrode_hs_stem
    # (i) THE STRUCTURE.  Two of them, out of the one cited file: the
    # junction, and the lead taken out of it by its region label -- same
    # atoms, same relaxation, a subset rather than a geometry derived from
    # anywhere else.  That is what lets the seam stay `spec_for(struct,
    # cfg, ...)`: a deck describes a structure, and these are two.
    if stage in ("electrode_L", "electrode_R"):
        model = (composed.electrode_left if stage == "electrode_L"
                 else composed.electrode_right)
        struct = model.as_structure()
        # The lead's identity IS the .TSHS stem the device deck names --
        # one spelling, `electrode_hs_stem`, read by both writers.
        label = electrode_hs_stem(task.label, model.label)
    else:
        # The junction -- and for the device it carries the region
        # partition the NEGF block is built from.
        struct, label = composed.sorted.structure, task.label
        # A FRAME SET (`engines/transport.md` § 2a.9): the seed runs once,
        # on frame 0; the device and the transmission run per frame, each
        # point's deck its frame (`at_frame`).
        from ..transport.stages import per_point_rungs
        if struct.n_frames > 1 and stage not in per_point_rungs():
            struct = struct.frame_at(0)

    def labelled(cfg):
        # THE RUNG FIXES ITS LABEL (C2): every rung's `SystemLabel` is spelt
        # from `task.label` -- the junction's rungs by it, a lead by
        # `electrode_hs_stem` -- so the device deck's `TS.Elec.<name> HS`
        # line and the gather name the file the lead wrote.  The template's
        # `system_label` never reaches a transport deck.
        return dataclasses.replace(cfg, system_label=label)
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
    state = electronic_state(composed.sorted.structure, labelled(config),
                             kind="transport")
    # A STATE TRANSIESTA CANNOT RUN is refused HERE, on the junction, where
    # each value still says where it came from (`template.md` § 6.3a, ES4):
    # once folded into the rungs' configs, every rung's gate would call a
    # recorded or detected value *stated*.
    from ..validation.chemistry import check_electronic_state
    refused = [i for i in check_electronic_state(
        composed.sorted.structure, labelled(config), calculation="transport")
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
                   template_text, pins=None, log=None):
    """``(struct, config, state, element)`` of one transport rung, resolved
    NOW from the current template, junction and run card -- what the gather
    renders an upstream rung from (:func:`rung_deck_now`): the template ⊕
    the rung's overrides ⊕ its run card (:func:`resolve_transport`), then
    the parts the conductor's rung reads too (:func:`transport_parts`)."""
    element = resolve_transport(task, stage, allocation,
                                 template_text=template_text, pins=pins,
                                 log=log)
    struct, configure, state = transport_parts(task, stage, composed,
                                                element.render_config())
    return struct, configure(element.render_config()), state, element


def transport_sources(element):
    """Who set each value a transport rung's decks write: `resolve`'s rows,
    the label said as the rung's -- it spells every rung's ``SystemLabel``
    from the calculation's (:func:`transport_parts`, C2).  Asked by both
    roads that render a rung, so the deck the gather compares with says
    what the prepared one says."""
    from ..resolve import SET_AFTER, said_by
    return said_by(element.deck_sources, "system_label",
                   SET_AFTER["rung_label"])


def composed_for_prep(base, task, plan):
    """The junction a transport calculation's rungs describe: the record
    beside ``task.json`` when it is for THIS citation (the folder travels,
    `project-layout.md` § 2.1), else composed afresh from the projects tree,
    its record written -- and READ BACK from it.

    Written into ``plan`` (`jobset.planned.Plan`), with everything else this
    prep writes: the fresh record is written to a temporary folder outside
    the calculation and read back from there -- the same round trip a later
    prep makes from the calculation's own copy -- and its bytes go into the
    plan."""
    from ..atom_permutation import PermutationError
    from ..projects import find_projects_root
    from ..transport.compose import (ComposeError, compose_junction,
                                     load_compose_record,
                                     write_compose_record)
    from ..transport.sort import SortError
    citation = task.slots["junction"]
    try:
        # The tree root is resolved BEFORE the record is loaded: the
        # reload re-runs the § 3 lead gates, and their principal-layer
        # half reads the CITED directory's own .ion files, which the
        # travelled folder does not carry.  Absent (the folder moved
        # out of its tree), that half degrades to UNVERIFIED honestly.
        root = find_projects_root(base)
        why: list = []
        composed = load_compose_record(base, citation=citation,
                                       tree_root=root, why=why,
                                       swap_electrodes=task.swap_electrodes)
        if composed is not None:
            return composed
        if root is None:
            # NAME WHICH OF THE THREE: no record, one composed from another
            # citation (a slot re-pointed after a re-relaxation), or one
            # that does not read -- with the folder outside any tree, so
            # nothing can be composed afresh.
            raise PrepError(
                f"this transport calculation cannot be composed here: "
                f"{why[0] if why else 'there is no usable record'}.  "
                f"And {base} is not inside a projects tree, so the "
                f"citation {citation!r} cannot be resolved to compose "
                f"afresh.  Prep once inside the tree that holds the "
                f"cited junction -- the record then travels with the "
                f"folder (engines/transport.md § 3.1).")
        # RENDER FROM THE RECORD, on this prep as on every later one.  The
        # fresh composition is the cited `.XV` at full float precision; the
        # record is the codec's text of it, and every later prep -- the
        # other rungs, a re-prep -- reads the record.  Rendering this one
        # from the fresh copy gave two preps of the SAME calculation two
        # sets of numbers ~1e-10 apart (measured 2026-09-25 on the
        # ENGINE-OFFSET record), and the gather's `same_calculation`
        # compares them as text: a value near a rounding boundary then
        # flips, and a good seed is refused.  One source, not rounding luck.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="molbuilder-compose-") as tmp:
            names = write_compose_record(
                tmp, compose_junction(citation, tree_root=root,
                                      swap_electrodes=task.swap_electrodes))
            composed = load_compose_record(tmp, citation=citation,
                                           tree_root=root, why=why,
                                           swap_electrodes=task.swap_electrodes)
            if composed is None:
                raise PrepError(
                    f"the composed junction was written and could not be "
                    f"read back: {why[-1] if why else 'no reason given'}")
            for name in names:
                plan.bytes(Path(base) / name, (Path(tmp) / name).read_bytes())
        return composed
    except (ComposeError, SortError, PermutationError) as exc:
        raise PrepError(str(exc)) from exc


def transport_spec(task, stage: str, struct, config, state, volts=None, *,
                   base=None):
    """One transport deck's ``(spec, cfg)`` -- the bias point's when
    ``volts`` is given (the point is the rung's answer, `engines/template.md`
    § 6.4; a single-bias rung keeps the 0 V `resolve` laid on)."""
    from ..siesta.input import spec_for as _siesta_spec_for
    cfg = at_bias(config, volts)
    # THE RUNG'S NAMES -- a transport rung's deck is named on the
    # calculation's label, as prep's own loop names it.
    names = RunNames.of(task.label, stage_home(base, task, stage).token,
                        task.shape)
    with user_error_as_prep():
        try:
            spec = _siesta_spec_for(struct, cfg, names=names,
                                    calculation="transport", state=state)
        except ValueError as exc:
            # `transport_spec` refuses an unknown rung with a message
            # written FOR a person, and `user_error_as_prep` translates
            # only ValidationError / RuntimeConfigError / WrapperError --
            # deliberately, so a TypeError still looks like the bug it is.
            raise PrepError(str(exc)) from exc
    return spec, cfg


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


__all__ = ["provide_pseudos", "resolve_transport", "transport_parts",
           "transport_rung", "transport_sources", "composed_for_prep",
           "transport_spec", "at_frame", "at_point", "at_bias"]

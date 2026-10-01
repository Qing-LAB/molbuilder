"""The transport composite's five stages — `archive/2026-09-01-transport-design.md`
§ 4.2, build step P4b.

This module owns TWO facts and the renders that follow from them:

* **The ladder** (:data:`TRANSPORT_STAGES`): the five stages in
  dependency order.  Fixed by design, not configurable — skipping is
  the per-stage ``enabled`` flag in ``task.json`` (the seed's Q4 skip),
  never a different ladder.
* **The per-rung bags, as stages** (:func:`stages_for_transport`).  Which
  rung READS an override is the catalogue's ``stages`` declaration, asked
  through the one door every kind's description check and ``resolve`` ask
  (``template.unread_overrides``, `engines/template.md` § 6.4) -- what puts
  a person's T(E) window into the deck ``tbtrans`` runs rather than the one
  ``siesta`` runs.  It was this module's own check, for transport alone,
  until 2026-09-30 (plan § 5w K4).

**NOTHING IN THIS MODULE RENDERS A DECK ANY MORE.**  All five rungs go
through the framework's own pipeline — ``siesta.input.spec_for`` with
``calculation="transport"`` -> :mod:`molbuilder.transport.deck`, whose
``SHAPE_OF_RUNG`` tables the four deck shapes — which is what lets the
template's items reach them.  :func:`config_for` below is what is LEFT of
the pre-seam path: it has no production caller
(`engines/transport.md` § 6.1a), and the electronic contract is resolved
by ``prep._resolve_transport`` from the calculation's own template.
:func:`render_stage_deck` stood beside it until 2026-09-17 and is deleted;
the tombstone at the end of this file says why a second renderer could not
be left standing.

That template is also why ruling Q5 no longer holds as written: § 2a.7
reversed it.  The cited relaxation DEFAULTS the electronic description
at ``jobset init``; it does not seal it.  The invariant survives untouched,
because ONE value shared by five rungs cannot disagree with itself.

P5 added the launch half, whose facts also live here: the § 4.2 DAG
(:func:`stage_inputs`, read by prep's gather), the continuation rows
(:func:`warm_declaration` — the seed's ``.DM``, the device's
``.TSDE``), and the bias scan's points (:func:`bias_points`; their folder
names are the codec's one spelling, ``task.bias_token`` — plain v-dirs,
ruled 2026-08-29).  The chain
walker itself is `jobset/submit.submit_transport_chain`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..config.transport import TransportConfig
from ..template import KIND_ROLES
from .compose import ComposedJunction

#: The composite's fixed ladder (§ 4.2) -- the five stages in
#: dependency order.  ``jobset init`` writes exactly these into
#: ``task.json``; `prep` names its rungs from the same tuple.  They are the
#: kind's rung ROLES too, so their one home is the catalogue's role
#: vocabulary (`template.KIND_ROLES`, `engines/template.md` § 6.4).
TRANSPORT_STAGES = KIND_ROLES["transport"]

#: **WHICH FACT EACH RUNG ANSWERS WITH** — a column, because the five rungs
#: do not answer the same question and reading them as if they did is a
#: defect with a visible face.
#:
#: * ``"scf"`` — did the SCF converge, and at what energy.  The seed hands the
#:   device a density; the device hands TBtrans a Hamiltonian.
#: * ``"fermi"`` — an SCF *and* the lead's Fermi level, which is the reference
#:   the whole junction is measured against (§ 2a.13): T(E) is relative to it
#:   and ``G = G0 * T(E_F)`` is evaluated at it.
#: * ``"product"`` — **not an SCF at all.**  TBtrans runs none: there is
#:   nothing to converge and no total energy, and its result is the
#:   transmission the record already carries in its own ``points`` blocks.
#:
#: `record.py` asked every rung the ``"scf"`` question and read the answer out
#: of its ``.out``.  For the transmission rung no parser claims that file --
#: correctly, it is not a SIESTA run -- so the ladder rendered the registry's
#: entire "supported file formats" list, two hundred words of it, in the cell
#: where the final rung's state belongs (measured 2026-09-19).  The fix is not
#: a TBtrans parser; it is to stop asking.
STAGE_FACT = {
    "seed":         "scf",
    "electrode_L":  "fermi",
    "electrode_R":  "fermi",
    "device":       "scf",
    "transmission": "product",
}


class StageError(Exception):
    """A stage whose deck cannot be rendered — the message names the
    stage and what blocks it, ready to surface verbatim."""


#: The TransportConfig fields a stage override may NOT set — the
#: electronic contract is the citation's to say (ruling Q5: electrode
#: and device must stay unable to disagree), the bias axis is
#: ``task.bias``'s, the identity the task's.  ONE spelling: `config_for`
#: refuses on it at prep, and the web hand-over refuses on it at send —
#: same set, same reason, both name the field.
#: A FIELD WHOSE KEYWORD IS UNRESOLVED — rendered nowhere, so it cannot be
#: a control that does nothing.
#:
#: `transmission_relative_to_ef` wrote `TS.TBT.Erange.RelToEF`, which the
#: 5.4.2 tbtrans does not know.  Its replacement is not a rename — whether a
#: `%block TBT.Contour` line's energies are absolute or relative to the
#: device E_F could not be settled from the binary — and inventing a mapping
#: is what put four dead keywords in the deck (`plan.md` § 5o).  It stays a
#: field so the question has an address, and stays out of the form so nobody
#: is offered a switch that moves nothing.  It is NOT sealed: an override
#: naming it is still refused by nothing here, because there is nothing to
#: refuse — it reaches no deck.
UNRESOLVED_FIELDS = frozenset({"transmission_relative_to_ef"})

#: The description's OWN facts — refused as overrides for EVERY
#: citation form: the engine is fixed, the label names the job, the
#: bias is the description's `--bias`.
SEALED_ALWAYS = frozenset({"engine", "job_name", "bias_voltages_v"})

#: The electronic contract.  Sealed when the citation carries a deck
#: (form A — fdf-is-truth, ruling Q5); OPEN when it is a labeled
#: structure pair (form B — there is no deck to be truth, so these are
#: the description's own fields).  transport-design.md § 4.1b.
#:
#: THE RECORD'S OWN KEYS -- `parse/contract.py`'s one table (the record's
#: names against ``SiestaConfig``'s), plus the k-point mesh it carries apart
#: -- so the recorded vocabulary and this set cannot drift.  They did: this
#: was a hand-written seven, and when the record began carrying the
#: electronic state (ES7, 2026-09-28) only the table learned it.
from ..parse.contract import K_MESH_RECORD_KEYS as _K_MESH_KEYS
from ..parse.contract import RECORD_TO_SIESTA_FIELD as _RECORD_FIELDS
CONTRACT_FIELDS = frozenset(_RECORD_FIELDS) | frozenset(_K_MESH_KEYS)

#: Both sets together — what a form-A citation refuses.
SEALED_TRANSPORT_FIELDS = SEALED_ALWAYS | CONTRACT_FIELDS


def resolvable_override_names() -> frozenset:
    """The names a per-stage override may carry — **asked, never listed.**

    `prep` resolves a rung's config through
    :func:`molbuilder.resolve.resolve` against
    :class:`~molbuilder.config.siesta.SiestaConfig`, so a name that is not one
    of its fields is refused there by name, and a RUN SETTING -- the
    catalogue's ``execution`` items, the machine's answers among them -- is
    refused as an override: it is the rung's run card's (`engines/stages.md`
    § 6.8d, plan § 5w K5). Those two facts are the whole vocabulary, and
    this is where they are turned into one question both doors ask.

    **Both doors had a different, wider answer, and the gap was measured
    2026-09-16.** The web form offered nineteen controls and the describe door
    checked membership of ``TransportConfig ∪ SiestaConfig``; sixteen resolve,
    and three do not — ``num_threads`` and ``log_level`` are
    ``TransportConfig`` fields that no catalogue row declares, and
    ``max_memory_mb`` is a machine fact. A person setting the thread count on
    the Transport tab got *"Described"*, and then **every** later `prep`, on
    either road, refused the whole calculation while naming a field they had
    never typed (*"did you mean 'omp_threads'?"*).

    Which names are run settings is the catalogue's own answer
    (``template.run_settings``), rather than a fourth frozenset.  Until the
    K5 review this subtracted the machine's answers alone, so the rung tabs
    offered the solver, ``block_size`` and ``parallel_over_k`` -- each then
    refused at the save with *"put it on the run card"*.
    """
    import dataclasses

    from ..config.siesta import SiestaConfig
    from ..template import run_settings

    on_the_card = run_settings("siesta")
    return frozenset(f.name for f in dataclasses.fields(SiestaConfig)
                     if f.name not in on_the_card)


def bias_points(task) -> Tuple[float, ...]:
    """The scan's points, in the order the description states them —
    the codec already enforced that the list starts at 0.0 (the chain
    starts from equilibrium).  ``()`` and a single entry mean NO scan:
    the stage keeps its plain single-deck layout, because the v-dir
    layer exists for the axis, not for every calculation
    (architecture § 0: a list with more than one element)."""
    bias = tuple(getattr(task, "bias", ()) or ())
    return bias if len(bias) > 1 else ()


# --------------------------------------------------------------------- #
#  Where a rung's attempts are (`engines/transport.md` § 2a.11; plan    #
#  § 5w K10)                                                            #
# --------------------------------------------------------------------- #

def per_point_rungs() -> frozenset:
    """The rungs a bias scan runs once per point: the bias item's own
    `stages` (`template.PER_POINT`, the role item no rung answers with one
    value) -- the device and the transmission, declared once, in the
    catalogue."""
    from ..template import PER_POINT, catalogue, one
    return frozenset(rung for name in PER_POINT
                     for rung in one(catalogue(), name, engine="siesta").stages)


def scan_points(task, stage: str) -> Tuple[float, ...]:
    """The bias points ``stage`` runs at -- the scan's (:func:`bias_points`)
    for a rung that runs once per point (:func:`per_point_rungs`), ``()``
    for every other rung and every calculation that is not a scan."""
    return bias_points(task) if stage in per_point_rungs() else ()


def rung_containers(base, task, stage: str) -> List[Tuple[Path, Optional[float]]]:
    """WHERE A RUNG'S ATTEMPTS ARE -- ``[(folder, volts)]``: one folder per
    bias point for a rung a scan runs at each point (``<token>/v<V>``,
    § 2a.11), the stage folder (``volts`` ``None``) otherwise.

    The one door every reader of a rung's attempts asks (plan § 5w K10):
    the record looked in the point folders for the transmission alone, so
    a finished device scan read *not run*, and Task setup's count and
    prep's *already under way* looked in the stage folder alone (the M11
    review's T-F27, T-F13)."""
    from ..identity import StageRef
    from ..paths import Shape
    from ..task import bias_token
    token = next(r.token for r in StageRef.ladder([s.name for s in task.stages])
                 if r.name == stage)
    sd = Shape.named(task.shape).stage_dir(token)
    stage_dir = Path(base) if sd == "." else Path(base) / sd
    points = scan_points(task, stage)
    if not points:
        return [(stage_dir, None)]
    return [(stage_dir / bias_token(v), float(v)) for v in points]


def rung_container(base, task, stage: str, volts: Optional[float]) -> Path:
    """The folder holding ``stage``'s attempts at ``volts`` -- its point's
    for a rung that runs once per point, the stage folder for one that does
    not (a lead, read by every point alike).  A point the scan does not
    hold is refused by name: a mismatch is a mistake, never a fallback."""
    containers = rung_containers(base, task, stage)
    for folder, v in containers:
        if v is None or (volts is not None and v == float(volts)):
            return folder
    raise ValueError(
        f"the {stage} stage has no folder at {volts!r} V: this scan runs it "
        f"at {', '.join(f'{v:g} V' for _f, v in containers)}")


def stage_inputs(stage: str, task_label: str, *,
                 seed_enabled: bool = True):
    """The § 4.2 DAG, as data: ``[(upstream stage, filename)]`` this
    stage CONSUMES from the concluded attempts of the stages before it.

    The filenames are derived from the SystemLabel rules the renderers
    already fixed — the seed and device share the task's label, each
    electrode's label IS its ``.TSHS`` stem (`electrode_hs_stem`, one
    spelling) — so producer and consumer cannot disagree about a name.

    A disabled seed (ruling Q4: skippable scaffolding) simply drops its
    row: the device SCF then starts from atomic densities.  The
    electrode and device rows are never optional — they are the physics
    (the self-energies, and the converged H the transmission reads).
    """
    from ..config.transport import (REGION_LEFT_ELECTRODE,
                                    REGION_RIGHT_ELECTRODE)
    from .transiesta import electrode_hs_stem
    elec = [
        ("electrode_L",
         f"{electrode_hs_stem(task_label, REGION_LEFT_ELECTRODE)}.TSHS"),
        ("electrode_R",
         f"{electrode_hs_stem(task_label, REGION_RIGHT_ELECTRODE)}.TSHS"),
    ]
    if stage == "device":
        seed = [("seed", f"{task_label}.DM")] if seed_enabled else []
        return seed + elec
    if stage == "transmission":
        # TBtrans reads the device's converged H -- which SIESTA 5.x
        # writes as <label>.TS.HSX (the sparse container that replaced
        # the 4.x device .TSHS; measured live 2026-08-29 on 5.4.2) --
        # plus the electrode .TSHS the TS.Elec blocks in the (shared)
        # deck text reference.  The .TSDE is NOT consumed: the TS.HSX
        # already carries the bias point's converged potential.
        return [("device", f"{task_label}.TS.HSX")] + elec
    return []


def warm_declaration(stage: str, task_label: str, base_dir=None):
    """What a transport rung takes from a run it CONTINUES (``--from``
    an earlier attempt of the SAME stage) — the § 4.2a vocabulary rows
    for the transport type, on the stages where continuing means
    anything: the seed (its ``.DM``) and the device (its ``.TSDE``, the
    NEGF density; read by presence, no deck keyword).

    The electrode single-points and the transmission post-processing
    declare NOTHING — re-running them is cheaper than reasoning about a
    half-finished copy, and an empty declaration is what makes
    ``--from`` refuse there by name instead of copying dead weight.
    """
    if stage not in ("seed", "device"):
        return []
    from ..jobset.model import WarmFile
    from ..warmfiles import rules_for
    return [WarmFile(f"{task_label}{r.suffix}",
                     requires_same=r.requires_same)
            for r in rules_for("siesta", "transport", base_dir) if r.carry]


#: What each rung computes and hands on -- the ONE-LINE NOTE a surface puts
#: beside the rung's name (`engines/transport.md` § 2a.3 "the results that
#: propagate", § 6.1; the tab strip of § 3.8.2a).  The index a surface shows
#: is the rung's position in TRANSPORT_STAGES, one-based.
RUNG_NOTES = {
    "seed": ("A closed periodic SCF of the whole junction.  It writes the "
             "density every later rung starts from -- a starting guess, "
             "never truth; skipping it costs the device iterations, not "
             "correctness."),
    "electrode_L": ("The left lead as a bulk crystal, sampled densely along "
                    "the transport axis.  It writes the Hamiltonian the "
                    "device folds in as the left self-energy -- read as "
                    "truth."),
    "electrode_R": ("The right lead, likewise.  Two lead runs even when the "
                    "leads are identical, so the record stays auditable."),
    "device": ("The open-boundary NEGF SCF of the junction between the two "
               "leads, at the bias.  It writes the converged Hamiltonian "
               "the transmission reads -- one per bias point."),
    "transmission": ("T(E) over the energy window, from that Hamiltonian.  "
                     "Nothing consumes its output, so tuning the window "
                     "re-runs seconds, never an NEGF cycle."),
}


def stages_for_transport(bags=None):
    """The composite's five rungs as ``Stage`` objects, each carrying the
    overrides that are ITS OWN -- *bags* maps a rung name to that rung's
    mapping, and a rung not named gets an empty bag.

    One door, so the two construction sites -- `jobset init` and the web
    describe door -- cannot build the ladder differently.  A rung this
    ladder does not have is refused by name.

    **Ownership is not decided here.**  Until 2026-09-24 this took ONE flat
    mapping and routed it by the `stages` declaration, parking an item that
    declared no rung on the device -- the holding position the per-stage
    surface (TR7) was to replace.  The per-rung form now asks the question
    per rung (`engines/transport.md` § 3.8.2a), so a bag arrives already
    placed, and the description's own check and ``resolve`` refuse a value
    on a rung that does not read it (``template.unread_overrides``).
    """
    from ..task import Stage

    bags = dict(bags or {})
    unknown = sorted(set(bags) - set(TRANSPORT_STAGES))
    if unknown:
        raise ValueError(
            f"no such rung {', '.join(map(repr, unknown))} -- a transport "
            f"ladder's rungs are {', '.join(TRANSPORT_STAGES)}.")
    return [Stage(name=n, enabled=True, overrides=dict(bags.get(n) or {}))
            for n in TRANSPORT_STAGES]


def config_for(task, composed: ComposedJunction, *,
               stage: str = None) -> TransportConfig:
    """The config ONE stage renders from.

    **Per rung, not per calculation.**  Until 2026-09-15 this merged all
    five override bags into a single config, so a name carried by two
    rungs took whichever bag came last in ladder order -- and it took it
    for EVERY rung.  Nothing observable broke while the ``device`` bag
    was the only one ever filled, which is exactly why it wanted pinning
    before that stopped being true.  *stage* names the rung whose bag
    applies; ``None`` applies none of them.

    EVERY bag is still validated whatever *stage* asks for, so an
    unknown or sealed name in any rung refuses at the first prep rather
    than at the rung that happens to carry it.

    Identity from the task (label, bias); the electronic contract from
    the cited attempt's own deck (`compose.fdf_params` — fdf-is-truth);
    the transverse k from the relaxation's k-grid, its transport axis laid
    on by the k-point mesh's rule where the value is born
    (``kmesh.with_fixed``, `engines/siesta.md` § 6.1 -- the NEGF open
    boundary is never BZ-sampled).  Transport-only knobs
    (transmission window / grid, contour) keep their defaults until the
    Transport tab describes them (P7).
    """
    from ..kmesh import with_fixed
    fdf = composed.fdf_params
    kw = {}
    recorded = getattr(composed, "recorded_contract", None)
    if recorded is not None and composed.deck_text is None:
        # The RECORDED contract (4.1b's third shade, structure-info-plan
        # I6): the pair's sidecar carries the finished run's own values
        # (info.calculation, written by the Results tab from the deck),
        # and they fill the config exactly as a cited deck would --
        # fdf-is-truth transferred to the recorded copy.  Only KNOWN
        # contract fields apply; kz is forced 1 like every fill here.
        import dataclasses as _dcf
        _holds = {f.name for f in _dcf.fields(TransportConfig)}
        for name, value in dict(recorded.get("contract") or {}).items():
            # A recorded name this older config does not hold -- the
            # electronic state -- reaches the decks through the template and
            # `resolve`, the live path.
            if name not in CONTRACT_FIELDS or name not in _holds:
                continue
            if name == "k_mesh_transverse":
                try:
                    counts = tuple(int(v) for v in value[:3])
                except (TypeError, ValueError):
                    continue
                if len(counts) != 3:
                    continue
                kw[name] = with_fixed("kgrid", counts, "transport")
            elif name == "siesta_mesh_cutoff_ry":
                kw[name] = int(round(float(value)))
            else:
                kw[name] = value
    if getattr(fdf, "kgrid", None):
        kw["k_mesh_transverse"] = with_fixed(
            "kgrid", tuple(int(v) for v in fdf.kgrid), "transport")
    if getattr(fdf, "mesh_cutoff_ry", None):
        kw["siesta_mesh_cutoff_ry"] = int(round(fdf.mesh_cutoff_ry))
    if getattr(fdf, "energy_shift_ry", None):
        kw["energy_shift_ry"] = float(fdf.energy_shift_ry)
    if getattr(fdf, "basis_size", None):
        kw["basis_size"] = str(fdf.basis_size)
    # The verbatim spelling, not the normalised comparison key `.xc` --
    # the decks this config renders should say what the citation said.
    if getattr(fdf, "xc_functional", None):
        kw["xc_functional"] = str(fdf.xc_functional)
    if getattr(fdf, "xc_authors", None):
        kw["xc_authors"] = str(fdf.xc_authors)
    if getattr(fdf, "electronic_temperature_k", None):
        kw["electronic_temperature_k"] = float(fdf.electronic_temperature_k)
    # THE TRANSPORT-ONLY KNOBS (transmission window/grid, contour,
    # electrode kz-adjacent fields) travel as STAGE OVERRIDES in
    # task.json -- the composite has no template, so the stages' own
    # override bags are the description's one place for them (P7b,
    # 2026-08-29; the Transport tab writes them there).  All five bags
    # merge in ladder order into the ONE config every deck renders
    # from; an unknown name refuses here, before anything renders.
    import dataclasses as _dc

    from ..config.siesta import SiestaConfig

    # THE VOCABULARY IS THE ENGINE'S, and `TransportConfig`'s names are
    # accepted beside it only while that class survives.
    #
    # An override names a CATALOGUE row now -- `electrode_kz`,
    # `transmission_emin_ev`, `negf_eq_pole_ev` -- because that is what the
    # template declares and what `resolve` resolves.  Checking against
    # `TransportConfig` alone refused a person's own lead k-density by
    # telling them it "is not a transport parameter", which it plainly is
    # (measured 2026-09-16, routing `electrode_kz` to the two lead rungs).
    #
    # What this function still uniquely guards is IDENTITY -- `job_name`,
    # `engine`, the bias axis -- which `resolve` would let a stage override
    # because they are ordinary schema fields to it.  That is why it is
    # still called, and it is what has to survive when TransportConfig goes
    # (TR6).
    known = ({f.name for f in _dc.fields(TransportConfig)}
             | {f.name for f in _dc.fields(SiestaConfig)})
    # What remains after SEALED_TRANSPORT_FIELDS IS the transport-only
    # vocabulary (window, grid, contour, runtime).
    for bag in (task.stages or ()):
        for name, value in (bag.overrides or {}).items():
            if name not in known:
                raise StageError(
                    f"stage {bag.name!r} overrides {name!r}, which is "
                    f"not a transport parameter (TransportConfig field "
                    f"names are the vocabulary; transport-design.md "
                    f"4.2).")
            if name in SEALED_ALWAYS:
                raise StageError(
                    f"stage {bag.name!r} overrides {name!r}, which is "
                    f"the description's own field (identity, bias) -- "
                    f"set it where the description sets it, never as a "
                    f"stage override.")
            if name in CONTRACT_FIELDS:
                # SHARED, therefore not a per-stage override.  The reason
                # changed on 2026-09-16 and the refusal did not: § 2a.7
                # ruled the cited run DEFAULTS these values rather than
                # sealing them, so "the citation's to say" is no longer
                # true -- but "every rung shares one value" still is, and
                # a per-stage override is exactly what would break it.
                raise StageError(
                    f"stage {bag.name!r} overrides {name!r}, which is "
                    f"SHARED by every stage of this calculation -- the "
                    f"electrode and the device must not be able to "
                    f"disagree about it.  Change it in the template, "
                    f"where it applies to all five rungs at once; it was "
                    f"filled in from the run you cited and it is yours "
                    f"to change (engines/transport.md 2a.7).")
            # VALIDATED for every rung; APPLIED only for the one asked
            # about, so one rung's tuning cannot reach another's deck.
            # A name this older config does not hold is legitimate and
            # simply not its business: it reaches the deck through the
            # template and `resolve`, which is the live path.
            if name not in {f.name for f in _dc.fields(TransportConfig)}:
                continue
            if bag.name == stage:
                kw[name] = value
    return TransportConfig(
        engine="transiesta",
        job_name=task.label,
        bias_voltages_v=list(task.bias) or [0.0],
        **kw)


# `render_stage_deck` DELETED 2026-09-17 -- a SECOND writer of the transport
# decks, with no caller left.
#
# It rendered the electrode rungs through `wizard.render_electrode_fdf` and the
# device/transmission rungs through `transiesta.render_script`, and its seed arm
# already raised: "the seed deck renders through the framework, not here".  The
# rest of the ladder followed the seed onto that framework on 2026-09-16, which
# left this function with nothing calling it and its own warning applying to
# every arm.
#
# Deleted rather than kept beside its replacement, for the reason its seed arm
# gave: two renderers for one deck can disagree, and the only thing stopping
# them was `prep` happening to branch before this call.  They HAD disagreed --
# the pole energy this path emitted stopped SIESTA before the SCF loop for two
# days, while the live path was correct, because the two read different config
# classes (`template.md` 2.1a, the second duplication).

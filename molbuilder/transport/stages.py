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
template's items reach them; the electronic contract is resolved by
``prep._resolve_transport`` from the calculation's own template.
``config_for``, the pre-seam path's projection onto ``TransportConfig``,
had no production caller and went with that class (M5 step 3, 2026-10-02);
:func:`render_stage_deck` went on 2026-09-17 -- the tombstone at the end of
this file says why a second renderer could not be left standing.

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

from ..template import KIND_ROLES

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
    from .sort import REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE
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

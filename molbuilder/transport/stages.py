"""The transport composite's five stages — `plans/transport-design.md`
§ 4.2, build step P4b.

This module owns TWO facts and the renders that follow from them:

* **The ladder** (:data:`TRANSPORT_STAGES`): the five stages in
  dependency order.  Fixed by design, not configurable — skipping is
  the per-stage ``enabled`` flag in ``task.json`` (the seed's Q4 skip),
  never a different ladder.
* **The one config** (:func:`config_for`): every stage's deck renders
  from a single :class:`~molbuilder.config.transport.TransportConfig`,
  filled from the composed junction's own ``.fdf`` snapshot — the deck
  that actually ran is the truth about a result — which is what makes
  basis · XC · mesh · electronic T identical between electrode, seed
  and device BY CONSTRUCTION (§ 3, ruling Q5: one template governs
  everything).

The renders reuse the existing emitters whole: the electrode decks are
the wizard's (:func:`~molbuilder.transport.wizard.render_electrode_fdf`),
the device and transmission decks are the registered TranSIESTA
engine's, and the seed deck is assembled from the same emitter pieces
(geometry / basis+XC / k-mesh) with ``SolutionMethod diagon`` — an
ordinary periodic pass whose ``.DM`` starts the device SCF.

P5 added the launch half, whose facts also live here: the § 4.2 DAG
(:func:`stage_inputs`, read by prep's gather), the continuation rows
(:func:`warm_declaration` — the seed's ``.DM``, the device's
``.TSDE``), and the bias scan's spellings (:func:`bias_token`,
:func:`bias_points` — plain v-dirs, ruled 2026-08-29).  The chain
walker itself is `jobset/submit.submit_transport_chain`.
"""
from __future__ import annotations

from typing import Tuple

from ..config.transport import TransportConfig
from ..structure import Structure
from .compose import ComposedJunction

#: The composite's fixed ladder (§ 4.2) -- the five stages in
#: dependency order.  ``jobset init`` writes exactly these into
#: ``task.json``; `prep` names its rungs from the same tuple.
TRANSPORT_STAGES = ("seed", "electrode_L", "electrode_R",
                    "device", "transmission")


class StageError(Exception):
    """A stage whose deck cannot be rendered — the message names the
    stage and what blocks it, ready to surface verbatim."""


def bias_token(v: float) -> str:
    """The ONE spelling of a bias point's directory name — ``v0``,
    ``v0.2``, ``v-0.5`` (user ruling 2026-08-29: plain v-dirs, production
    names — never the bench spelling).  Deck naming, the per-point
    attempt layout, the gather and the chain walker all read this."""
    return f"v{v:g}"


def bias_points(task) -> Tuple[float, ...]:
    """The scan's points, in the order the description states them —
    the codec already enforced that the list starts at 0.0 (the chain
    starts from equilibrium).  ``()`` and a single entry mean NO scan:
    the stage keeps its plain single-deck layout, because the v-dir
    layer exists for the axis, not for every calculation
    (architecture § 0: a list with more than one element)."""
    bias = tuple(getattr(task, "bias", ()) or ())
    return bias if len(bias) > 1 else ()


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
        # TBtrans reads the device's converged H (.TSHS) and its NEGF
        # density (.TSDE), plus the electrode .TSHS the TS.Elec blocks
        # in the (shared) deck text reference.
        return ([("device", f"{task_label}.TSHS"),
                 ("device", f"{task_label}.TSDE")] + elec)
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


def config_for(task, composed: ComposedJunction) -> TransportConfig:
    """The ONE config every stage renders from.

    Identity from the task (label, bias); the electronic contract from
    the cited attempt's own deck (`compose.fdf_params` — fdf-is-truth);
    the transverse k from the relaxation's k-grid with the transport
    axis forced to 1 (the NEGF open boundary is never BZ-sampled — the
    engine preflight refuses kz != 1, so forcing it here is the same
    rule applied where the value is born).  Transport-only knobs
    (transmission window / grid, contour) keep their defaults until the
    Transport tab describes them (P7).
    """
    fdf = composed.fdf_params
    kw = {}
    if getattr(fdf, "kgrid", None):
        kx, ky, _kz = fdf.kgrid
        kw["k_mesh_transverse"] = (int(kx), int(ky), 1)
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
    return TransportConfig(
        engine="transiesta",
        job_name=task.label,
        bias_voltages_v=list(task.bias) or [0.0],
        **kw)


def _render_seed(struct: Structure, cfg: TransportConfig) -> str:
    """The seed deck: an ordinary periodic SIESTA pass on the sorted
    sandwich (§ 4.2 stage 1 — default on, skippable, ruling Q4).

    Same geometry, same electronic contract and same transverse k as
    the device — assembled from the SAME emitter pieces — solved with
    ``SolutionMethod diagon``.  Its converged ``.DM`` (written under
    the shared ``SystemLabel``) is what the device SCF starts from;
    the seed itself starts fresh (§ 4.1a: nothing order-dependent
    crosses the sort, and there is nothing before the seed anyway).
    No MD block = a single point, the same convention the electrode
    deck states.
    """
    from ..script_emit import emit_atom_metadata
    from .transiesta import (_emit_basis_and_xc, _emit_geometry,
                             _emit_k_mesh)
    lines = [
        "# ================================================================== #",
        f"#  Transport SEED .fdf — {cfg.job_name}",
        "#  An ordinary periodic SIESTA single point on the composed,",
        "#  SORTED junction (transport-design.md 4.2, stage 1).  Its",
        f"#  converged {cfg.job_name}.DM starts the device NEGF SCF;",
        "#  scaffolding for convergence, no effect on the converged",
        "#  answer (skippable -- ruling Q4).",
        "# ================================================================== #",
        "",
        f"SystemLabel            {cfg.job_name}",
        f"SystemName             Transport seed for {cfg.job_name}",
        "",
    ]
    lines.extend(_emit_geometry(struct))
    block = emit_atom_metadata(
        regions=struct.regions or {},
        annotations=dict(getattr(struct, "annotations", {}) or {}),
        n_atoms_total=struct.n_atoms,
    )
    if block:
        lines.append(block)
        lines.append("")
    lines.extend(_emit_basis_and_xc(cfg))
    lines.extend(_emit_k_mesh(cfg))
    lines.extend([
        "# Ordinary diagonalisation (NOT transiesta): the seed is a",
        "# periodic warm-up.  SIESTA writes <SystemLabel>.DM as the",
        "# SCF converges; no MD block = single point.",
        "SolutionMethod         diagon",
        "",
    ])
    return "\n".join(lines) + "\n"


def render_stage_deck(stage: str, composed: ComposedJunction,
                      cfg: TransportConfig) -> str:
    """One stage's deck text, from the composed junction + the one
    config.  Raises :class:`StageError` naming what blocks — including
    the engine preflight's errors for the device/transmission decks,
    surfaced here so `prep` refuses before anything lands on disk.
    """
    from .transiesta import TransiestaEngine, electrode_hs_stem
    from .wizard import render_electrode_fdf

    dev = composed.sorted.structure
    if stage == "seed":
        return _render_seed(dev, cfg)
    if stage in ("electrode_L", "electrode_R"):
        model = (composed.electrode_left if stage == "electrode_L"
                 else composed.electrode_right)
        # The SystemLabel IS the .TSHS stem the device deck references
        # -- one spelling, `electrode_hs_stem`, both writers.
        return render_electrode_fdf(
            model, cfg,
            job_name=electrode_hs_stem(cfg.job_name, model.label))
    if stage in ("device", "transmission"):
        # The engine's own preflight runs first; prep sorted the atoms,
        # so its ordering error can never fire -- everything else it
        # checks (kz=1, regions, open-shell chemistry) still gates.
        errors = [i for i in TransiestaEngine.preflight(dev, cfg)
                  if i.severity == "error"]
        if errors:
            raise StageError(
                f"the {stage} deck cannot render:\n  - "
                + "\n  - ".join(i.message for i in errors))
        # The transmission deck is the SAME text: TBtrans reads the
        # device deck (geometry + TS.Elecs + the TS.TBT.* window) and
        # post-processes the converged device's files; only the binary
        # differs, and the binary is launch routing (P5).
        return TransiestaEngine.render_script(dev, cfg)
    raise StageError(
        f"{stage!r} is not a transport stage; the ladder is "
        f"{', '.join(TRANSPORT_STAGES)} (transport-design.md 4.2).")

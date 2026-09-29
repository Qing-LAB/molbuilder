"""The transport calculation's deck — SIESTA, on the framework's seam.

Contract: [`engines/transport.md`](?doc=engines/transport.md) §§ 3.2, 3.6, 6.1
(what transport's decks are and why they were not on the seam) +
`script-preparation.md` § 4 (the seam this serves).

WHAT THIS IS.  ``transport_spec(struct, cfg, stage_token)`` returns the
:class:`~molbuilder.script_emit.DeckSpec` for ``calculation = "transport"``.
It is the exact shape :mod:`molbuilder.pyscf.vibration_deck` has for
``calculation = "vibration"``: the KIND is a render argument, the seam stays
ONE per engine, and the kind's own module owns its layout.

WHY IT EXISTS.  `transport.md` § 3.2 measured the cost of transport never
joining floor 3: ``transport/transiesta.py::render_script`` (2026-06-10)
predates the render pipeline (2026-08-19) and concatenates literal
f-strings, so the keyword set is fixed in code — **13 keywords and 4 blocks
against a template offering 45 deck-reaching items**.  Twenty-one SIESTA
keywords with catalogue rows could reach no transport deck at all, among them
``MaxSCFIterations`` and ``DM.Tolerance``: absent from the file, so SIESTA
used its own defaults and a seed ran to 1000 iterations and died
``SCF_NOT_CONV`` with no way for anyone to ask for a looser budget.

THE LIFT, and its boundary.  :mod:`molbuilder.pyscf.vibration_deck` set the
direction — *"a move, not a rewrite"* — and that holds for the one piece where
it can: ``_emit_geometry`` is imported and composed unchanged.  **The other two
blocks are honest rewrites**, and saying otherwise would be this module lying
about itself: ``_emit_seed_header`` restates the old header's text and
``_emit_kgrid_block`` restates ``_emit_k_mesh``, both because their originals
read a ``TransportConfig`` and these read the engine's own config.  The old
seed emitter is DELETED rather than left beside them, so the reflowed text
exists in one place.
The boundary is drawn by a single question, and it is the question
`script-preparation.md` § 4.1 asks:

* **a keyword with a value is a SECTION ITEM**, resolved from its catalogue
  declaration.  This is the whole fix: the 21 arrive because
  :mod:`molbuilder.siesta.layout`'s sections already name them, and transport
  is a calculation KIND on the siesta engine (39 shared rows, measured in
  § 3.3) so it reuses those sections rather than restating them.
* **structural text is a BLOCK** — the coordinate table, ``%block TS.Elecs``
  and the reservoir blocks.  Those are lifted whole; a block whose VALUES are
  catalogue items (the T(E) window) asks each value through the framework's
  door and writes it with its note.

``_emit_basis_and_xc`` is therefore NOT lifted: its six keywords are exactly
``BASIS_SECTION`` + ``XC_SECTION`` + ``electronic_temperature``, and lifting
it beside them would write each twice — which ``layout.check_rules`` now
catches ("written twice with different values"), because a migrated deck gets
the engine's check gate for the first time.

NO ADAPTER ANY MORE.  The lifted NEGF emitter read a ``TransportConfig``,
so this module projected the engine's config into one (``_legacy_view``) --
the last place the two vocabularies met.  Since 2026-09-29 its VALUES are
catalogue items written by the section walk with their notes (the three
sections below), and what is left of it -- the electrode and reservoir
declarations -- reads the engine's own config
(`engines/transport.md` § 6.1b).

**ALL FIVE RUNGS ARE ON THE SEAM.**  Four deck shapes serve them
(:data:`SHAPE_OF_RUNG`), and each is a layout in this module.  The seam
question that held ``electrode`` and the NEGF rungs back — *what does a composite
kind hand its renderer, when the deck describes something the citation was
composed into?* — is answered, and the answer is that it hands a
**structure**, like every other kind: `prep` picks WHICH structure the rung
describes (the junction, or the lead taken out of it by its region label) and
``spec_for(struct, cfg, stage_token=)`` is unchanged.  Nothing reaches for the
``ComposedJunction`` from inside here.

Which shape a rung gets is a TABLE.  Choosing the LAYOUT for a shape is a
dispatch in :func:`transport_spec` over that table's four values.  A rung the
table does not name is refused BY NAME rather than rendered partially; `prep`
translates that refusal, so it reaches a person as a message and not a
traceback.

"""
from __future__ import annotations

from typing import Optional

from .. import script_emit as _sc
from ..siesta import layout as _sl
from ..structure import Structure

#: Which deck SHAPE each rung renders — a table, because FOUR texts serve
#: FIVE rungs (`engines/transport.md` § 6.1) and which text a rung gets is a
#: fact about the ladder, not a decision to re-derive per call.
#:
#: The device and the transmission are two texts because two PROGRAMS read
#: them (§ 6.1b): `siesta` reads no `TBT.*` keyword, so the device deck
#: carries none, and `tbtrans` reads the `TS.*` junction description, so the
#: transmission deck carries it beside its own settings.  What keeps them
#: from drifting apart about what the junction IS is shared text, not shared
#: bytes: one geometry block, one electrode-declaration block
#: (`transiesta.emit_electrode_declarations`), one electronic description.
SHAPE_OF_RUNG = {
    "seed":         "seed",
    "electrode_L":  "electrode",
    "electrode_R":  "electrode",
    "device":       "device",
    "transmission": "transmission",
}


def rung_of(stage_token: Optional[str]) -> str:
    """The rung name inside a ``<NN>_<name>`` stage token.

    ``prep`` holds the :class:`~molbuilder.identity.StageRef` and hands the
    token down as a render argument (`engines/stages.md` § 1.1 — the emitter
    never learns the word), so this reads the name back out rather than being
    told it twice.
    """
    tok = str(stage_token or "")
    head, sep, tail = tok.partition("_")
    return tail if (sep and head.isdigit()) else tok


# ===================================================================== #
#  The layouts, as tables.                                              #
# ===================================================================== #

#: TranSIESTA's own settings for the device's self-consistent run -- read by
#: `siesta` in NEGF mode.  The voltage is not here: it is a `role` item, the
#: bias point this rung is for, so the rung's own block writes it
#: (`_emit_device_run`).
TS_DEVICE_SECTION = _sc.Section(
    "TranSIESTA -- how the device's open-boundary run is solved",
    ("electrodes_bulk", "negf_eq_pole_ev", "negf_neq_eta_ev"),
    note=(
        "# Read by siesta in its NEGF mode (TranSIESTA) -- this rung.  An item",
        "# absent below was left at 0, which leaves it to TranSIESTA: its own",
        "# default there is a rule, not a number (engines/transport.md 6.1b).",
    ))

#: What tbtrans reads of TranSIESTA's settings: `TBT.Elecs.Bulk` defaults to
#: `TS.Elecs.Bulk` (SIESTA 5.4.2, `m_tbt_options.F90`), so the one shared
#: value is written here too.  The contour settings are not: they say how the
#: device's density is integrated, and tbtrans integrates none.
TS_READ_BY_TBTRANS_SECTION = _sc.Section(
    "TranSIESTA settings tbtrans reads",
    ("electrodes_bulk",),
    note=(
        "# tbtrans takes this as the default of its own TBT.Elecs.Bulk, so",
        "# the transmission treats the leads as the device run did.",
    ))

#: TBtrans's own settings -- read by `tbtrans` alone; `siesta` holds no
#: `TBT.*` keyword, which is why the device deck carries none.
TBT_SECTION = _sc.Section(
    "TBtrans -- how T(E) is computed from the device's Hamiltonian",
    ("tbt_k_grid", "tbt_elecs_eta_ev", "tbt_contours_eta_ev", "tbt_spin",
     "tbt_dos_gf", "tbt_dos_a", "tbt_dos_elecs", "tbt_t_eig", "tbt_t_bulk",
     "tbt_t_all", "tbt_verbosity"),
    note=(
        "# Read by tbtrans, which runs no SCF: it takes the converged",
        "# Hamiltonian the device rung wrote (TBT.HS above) and the leads'",
        "# .TSHS, and evaluates T(E) on the window above.  An item absent",
        "# below was left at 0, which leaves it to tbtrans's own rule.",
    ))


def _device_layout(derived, frame, state_block):
    """The device rung -- TranSIESTA's open-boundary NEGF run, at one bias.

    Read by `siesta` alone, so it carries TranSIESTA's settings and no
    `TBT.*` line (`engines/transport.md` § 6.1b: the binary holds no `TBT.`
    label).  The electrode declarations stay a ``Block``: they are derived
    from the structure's own ``regions`` -- which atoms are a lead, and
    therefore which contiguous range TranSIESTA is told about -- and no
    parameter models that.
    """
    return (
        _sc.Block("identity and what this rung computes", _emit_device_header),
        _sc.Block("cell, coordinates and region metadata",
                  _geometry_block(frame)),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the transverse k-mesh (the transport axis is not sampled)",
                  _emit_kgrid_block),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        # NO RESTART BLOCK HERE.  `_emit_device_run` writes `DM.UseSaveDM`
        # -- it is how the seed's density is picked up -- and writing it
        # twice is what the check gate refuses.  The boundary is drawn at
        # the keyword, not at the topic.
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(block_size=derived.get("block_size"),
                        algorithm=derived.get("algorithm")),
        _sc.Block("what this rung runs, and at which bias",
                  _device_run_block(derived)),
        _sc.Block("the junction: its electrodes and reservoirs",
                  _emit_electrode_block),
        TS_DEVICE_SECTION,
        _sl.OUTPUT_SECTION,
    )


def _transmission_layout(derived, frame, state_block):
    """The transmission rung -- `tbtrans` on the device's converged
    Hamiltonian.

    Read by `tbtrans`, which reads the junction description (the electrode
    and reservoir blocks, the voltage, the bulk treatment) as TranSIESTA
    wrote it and its own `TBT.*` settings.  **It keeps the SIESTA settings
    the rungs share**: `tbtrans` reads at least one of them -- its
    temperature starts from `ElectronicTemperature` (`m_tbt_options.F90`) --
    and which others it reads is an audit of its source not yet made, so
    none is dropped until that says it may be (§ 6.1b).
    """
    return (
        _sc.Block("identity and what this rung computes",
                  _emit_transmission_header),
        _sc.Block("cell, coordinates and region metadata",
                  _geometry_block(frame)),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the transverse k-mesh (the transport axis is not sampled)",
                  _emit_kgrid_block),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(block_size=derived.get("block_size"),
                        algorithm=derived.get("algorithm")),
        _sc.Block("what this rung reads, and at which bias",
                  _transmission_run_block(derived)),
        _sc.Block("the junction: its electrodes and reservoirs",
                  _emit_electrode_block),
        TS_READ_BY_TBTRANS_SECTION,
        _sc.Block("the energy window T(E) is computed on", _emit_tbt_window),
        TBT_SECTION,
        _sl.OUTPUT_SECTION,
    )


def _emit_device_header(struct, cfg) -> str:
    label = cfg.system_label
    return "\n".join([
        "# ================================================================== #",
        f"#  TranSIESTA DEVICE .fdf — {label}",
        "#  The open-boundary NEGF calculation on the composed junction, at",
        "#  one bias.  The leads are not solved here: each one's Hamiltonian",
        "#  was computed by its own rung and is read from the .TSHS named in",
        "#  the TS.Elec blocks below, which is what makes this an OPEN",
        "#  boundary rather than a bigger periodic cell.",
        "#",
        "#  Read by siesta (TranSIESTA) alone.  The transmission rung has its",
        "#  own deck for tbtrans; siesta reads no TBT.* keyword, so none is",
        "#  written here (engines/transport.md 6.1b).",
        "# ================================================================== #",
        "",
        f"SystemLabel            {label}",
        f"SystemName             Transport device for {label}",
    ])


def _emit_transmission_header(struct, cfg) -> str:
    label = cfg.system_label
    return "\n".join([
        "# ================================================================== #",
        f"#  TBtrans TRANSMISSION .fdf — {label}",
        "#  T(E) from the device rung's converged Hamiltonian, at the same",
        "#  bias.  No self-consistent run happens here: tbtrans reads the",
        f"#  device's {label}.TS.HSX and the leads' .TSHS and evaluates the",
        "#  transmission on the energy window below.",
        "#",
        "#  Read by tbtrans.  It reads the junction as TranSIESTA wrote it --",
        "#  the TS.Elec and TS.ChemPot blocks, TS.Voltage, TS.Elecs.Bulk --",
        "#  and its own TBT.* settings (engines/transport.md 6.1b).  The",
        "#  SIESTA settings are kept: tbtrans reads at least",
        "#  ElectronicTemperature among them.",
        "# ================================================================== #",
        "",
        f"SystemLabel            {label}",
        f"SystemName             Transport transmission for {label}",
    ])


def _notes(cfg, lines):
    """A block's explanation, dropped when the deck is asked to be quiet --
    what the section walk does with a section's note (`verbose`)."""
    return list(lines) if getattr(cfg, "verbose_comments", True) else []


def _voltage_lines(cfg, derived):
    """``TS.Voltage`` with its note -- a `role` item (the bias point this rung
    is for), so the rung writes it, through the catalogue's door and the
    engine's one syntax door, handed THIS deck's context (W10)."""
    p = _sc.parameter("bias_voltage_v", "siesta", config=cfg)
    return [*_notes(cfg, p.note(*_sl.note_lead(p))), _sl.line(derived)(p)]


def _device_run_block(derived):
    return lambda struct, cfg: _emit_device_run(struct, cfg, derived)


def _transmission_run_block(derived):
    return lambda struct, cfg: _emit_transmission_run(struct, cfg, derived)


def _emit_device_run(struct, cfg, derived) -> str:
    """What makes this deck the device's: TranSIESTA, started from the seed's
    density, at this rung's bias point."""
    return "\n".join([
        "# --- TranSIESTA, at this rung's bias point ---",
        "#",
        "# `SolutionMethod transiesta` switches the SCF cycle to NEGF -- the",
        "# rung's identity, not a setting (`role`).  (`TS.SolutionMethod` is a",
        "# different keyword -- the NEGF inversion algorithm -- and naming the",
        "# engine there stops SIESTA 5.4.2 with 'Unrecognized TranSiesta",
        "# solution method'; measured 2026-06-18.)",
        "SolutionMethod         transiesta",
        "",
        "# Start the NEGF SCF from a saved density when one is present: the",
        "# seed rung leaves <SystemLabel>.DM beside this deck, and SIESTA's",
        "# default for this keyword is false, so without it the seed would sit",
        "# unread.  With no file, SIESTA starts from atomic densities; a .TSDE",
        "# needs no keyword -- TranSIESTA reads it by presence.",
        "DM.UseSaveDM           true",
        *_voltage_lines(cfg, derived),
        "",
    ])


def _emit_transmission_run(struct, cfg, derived) -> str:
    """What makes this deck the transmission's: the device Hamiltonian it
    reads, named, and the bias point it reads it at."""
    label = cfg.system_label
    return "\n".join([
        "# --- tbtrans, on the device rung's converged Hamiltonian ---",
        "#",
        "# No SolutionMethod: tbtrans runs no SCF.  The bias below is the",
        "# device rung's own point -- tbtrans reads TS.Voltage as the default",
        "# of TBT.Voltage (m_tbt_hs.F90), and it must be the voltage the",
        "# device was converged at.",
        *_voltage_lines(cfg, derived),
        "",
        "# WHERE the device Hamiltonian is.  tbtrans would pick the first of",
        f"# {label}.TS.HSX, .TSHS and .HSX that exists (m_tbt_hs.F90); naming it",
        "# says which one this transmission is of, rather than leaving that to",
        "# whichever file is present.",
        f"TBT.HS                 {label}.TS.HSX",
        "",
    ])


def _emit_electrode_block(struct, cfg) -> str:
    """The electrode and reservoir declarations -- one text for both NEGF
    rungs (`transiesta.emit_electrode_declarations`)."""
    from .transiesta import emit_electrode_declarations
    return "\n".join(emit_electrode_declarations(struct, cfg))


def _emit_tbt_window(struct, cfg) -> str:
    """``%block TBT.Contour.window`` -- the energy grid T(E) is computed on.

    A ``%block`` is structural, so it is a block; its three VALUES are three
    catalogue rows, each asked through the framework's door so each arrives
    with its note -- the k-grid block's precedent (`_emit_kgrid_block`).
    """
    lo = _sc.parameter("transmission_emin_ev", "siesta", config=cfg)
    hi = _sc.parameter("transmission_emax_ev", "siesta", config=cfg)
    n = _sc.parameter("transmission_n_points", "siesta", config=cfg)
    return "\n".join([
        *_notes(cfg, [*lo.note(), *hi.note(), *n.note()]),
        "",
        "# A CONTOUR BLOCK, NOT SCALARS.  tbtrans reads its energy grid only",
        "# as `TBT.Contours` naming one or more line contours; `part line` is",
        "# not a choice -- tbtrans refuses anything else with \"Unrecognized",
        "# contour type for tbtrans, MUST be a line part\".",
        "%block TBT.Contours",
        "  window",
        "%endblock TBT.Contours",
        "",
        "%block TBT.Contour.window",
        "  part line",
        f"   from {float(lo.value):.5f} eV to {float(hi.value):.5f} eV",
        f"    points {int(n.value)}",
        "     method mid-rule",
        "%endblock TBT.Contour.window",
        "",
    ])


def _electrode_layout(derived, frame, state_block):
    """An electrode rung: a genuinely periodic BULK calculation.

    Read down it and the difference from the seed is three lines — and each
    is the physics of what a lead is, not a preference:

    * the k-mesh samples the **transport axis densely**, because a lead is
      periodic along it.  This is the one axis where the lead and the device
      deliberately disagree, and the density is what resolves the Fermi level
      every downstream stage is measured against;
    * ``TS.HS.Save`` is on, so the run writes the Hamiltonian the device's
      ``TS.Elec`` reference reads.  It is `role`-declared: a lead that omits
      it converges happily and produces nothing the device can attach to;
    * the structure is the **extracted lead**, not the junction.

    Everything else is the same engine's section set, because a lead run is
    an ordinary SIESTA single point.
    """
    return (
        _sc.Block("identity and what this lead is for", _emit_electrode_header),
        _sc.Block("cell and coordinates", _geometry_block(frame)),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the k-mesh — transverse shared, transport axis DENSE",
                  _emit_electrode_kgrid_block),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(block_size=derived.get("block_size"),
                        algorithm=derived.get("algorithm")),
        _sc.Block("what this rung must write", _emit_electrode_outputs),
        _sl.OUTPUT_SECTION,
    )


def _emit_electrode_header(struct, cfg) -> str:
    label = cfg.system_label
    return "\n".join([
        "# ================================================================== #",
        f"#  TranSIESTA ELECTRODE (bulk lead) .fdf — {label}",
        "#  A single point on the lead region taken OUT of the cited",
        "#  junction by its label -- same atoms, same relaxation, a subset",
        "#  rather than a geometry derived from somewhere else.  That is what",
        "#  makes this lead and the device consistent by construction.",
        "#",
        f"#  Writes {label}.TSHS, which the device deck's TS.Elec reference",
        "#  reads to build this lead's self-energy.",
        "# ================================================================== #",
        "",
        f"SystemLabel            {label}",
        "SystemName             Bulk electrode for transport",
    ])


def _emit_electrode_kgrid_block(struct, cfg) -> str:
    """``%block kgrid_Monkhorst_Pack`` — transverse shared, transport DENSE.

    The one place a lead and the device deliberately differ. The device is an
    open boundary and is not sampled along transport at all; the lead is a
    genuinely periodic bulk crystal, and **its Fermi level is the reference
    energy the whole calculation is measured against**, so that axis must be
    converged. Under-sample it and every transmission feature sits at the
    wrong energy.

    The transverse pair is the same one the device uses, and must be: the
    self-energy is built per transverse k-point and folded into the device at
    that same point.
    """
    kx, ky, _ = tuple(cfg.kgrid or (1, 1, 1))
    # ONE READ, and it is the framework's.  `parameter(..., config=cfg)`
    # resolves the row by `getattr(config, name)` and the value it resolved is
    # what the note states -- so reading the field a second time here could
    # print one number and write another.  It also spelled a literal `40` that
    # already had three homes (the catalogue row, the `SiestaConfig` default
    # and `wizard.DEFAULT_ELECTRODE_KZ`), behind a `getattr` default for a
    # field that certainly exists, with an `or` that silently rewrote a
    # deliberate 0.
    p = _sc.parameter("electrode_kz", "siesta", config=cfg)
    kz = int(p.value)
    out = list(p.note())
    out += [
        "%block kgrid_Monkhorst_Pack",
        f"  {int(kx):>3}    0    0      0.0",
        f"    0  {int(ky):>3}    0      0.0",
        f"    0    0  {kz:>3}      0.0",
        "%endblock kgrid_Monkhorst_Pack",
    ]
    return "\n".join(out)


def _emit_electrode_outputs(struct, cfg) -> str:
    """``TS.HS.Save`` — the rung's reason for existing.

    A `role` item (`engines/template.md` § 6.4): the stage decides it, nobody
    is offered a switch, and the template of this kind carries no value for
    it. Written here rather than as a section item for that reason — a
    section resolves an item's value from the config, and this one has none
    to resolve.
    """
    return "\n".join([
        "# --- The Hamiltonian this lead exists to write ---",
        "#",
        "# A lead run that omits this converges happily and produces nothing",
        "# the device can attach to, so it is the STAGE'S OWN answer rather",
        "# than a setting: `TS.HS.Save` writes <SystemLabel>.TSHS, which the",
        "# device deck names in its TS.Elec block.",
        "#",
        "# NOT the same keyword as `SaveHS` further down, which writes the",
        "# .HSX a post-processor reads.  That one is the engine's own output",
        "# group and is on by default for every SIESTA run; this ladder does",
        "# not consume it, and it costs disk, not correctness.",
        "#",
        "# Nor `TS.DE.Save`, which would write this lead's DENSITY MATRIX.",
        "# TranSIESTA asks for that file only when TS.Elecs.DM.Init is",
        "# `bulk` or `force-bulk` AND the device runs at zero bias; the",
        "# default chain lands on `diagon`, and a finite bias forces it off",
        "# regardless, so nothing in this ladder reads it.  Set that option",
        "# by hand at V = 0 and you would need TS.DE.Save true here.",
        "TS.HS.Save             true",
        "",
        "# An ordinary diagonalisation: a lead is a periodic bulk crystal,",
        "# not an open boundary.  `role`-declared like TS.HS.Save above, so",
        "# the rung writes it and no section does.",
        "SolutionMethod         diagon",
    ])


def _seed_layout(derived, frame, state_block):
    """The seed rung: an ordinary periodic SIESTA pass (§ 4.2 stage 1).

    Read down it and you have read the deck's SCIENCE, in order.  Not the
    whole file: the framework adds the banner, the USER-CUSTOM fence and the
    record sections around this, and those are about half the lines.
    Everything after the geometry is the SIESTA engine's own section set — which is the measured claim of § 3.3 made structural:
    transport is a kind on this engine, so its SCF, its convergence pair, its
    iteration limit, its spin, its parallel split and its output group are
    that engine's, not a second copy.
    """
    return (
        _sc.Block("identity and the seed's purpose", _emit_seed_header),
        _sc.Block("cell, coordinates and region metadata", _geometry_block(frame)),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the transverse k-mesh (kz forced to 1)", _emit_kgrid_block),
        _sc.Block("what the seed's solver must be, and must not",
                  _emit_solver_note),
        _sc.Block("the restart group", _emit_restart_group),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(block_size=derived.get("block_size"),
                        algorithm=derived.get("algorithm")),
        _sl.OUTPUT_SECTION,
    )


# ===================================================================== #
#  The blocks — structural text, lifted.                                #
# ===================================================================== #

def _emit_seed_header(struct, cfg) -> str:
    label = cfg.system_label
    return "\n".join([
        "# ================================================================== #",
        f"#  Transport SEED .fdf — {label}",
        "#  An ordinary periodic SIESTA single point on the composed,",
        "#  SORTED junction (engines/transport.md 4.2, stage 1).  Its",
        f"#  converged {label}.DM starts the device NEGF SCF; scaffolding",
        "#  for convergence, no effect on the converged answer",
        "#  (skippable -- ruling Q4).",
        "# ================================================================== #",
        "",
        f"SystemLabel            {label}",
        f"SystemName             Transport seed for {label}",
    ])


def _geometry_block(frame):
    """The coordinate Block for a deck whose atoms were placed with ``frame``
    -- the frame the spec records (`model/structure-periodicity.md` § 6.0)."""
    return lambda struct, cfg: _emit_geometry_block(struct, cfg, frame)


def _emit_geometry_block(struct, cfg, frame=None) -> str:
    """Cell + coordinates + the region/annotation metadata.

    Lifted whole: no parameter models a coordinate table, which is what
    :class:`~molbuilder.script_emit.Block` is for.
    """
    from .transiesta import _emit_geometry

    # NO `emit_atom_metadata` CALL HERE.  The FRAMEWORK emits that fence
    # once, in the record section (`script_emit.py`, the only caller among
    # the engines).  `_render_seed` called it itself because it was not on
    # the seam and nothing else would; lifting that call produced the fence
    # TWICE with different provenance, and `_extract_atom_metadata_dict`
    # stops at the first END marker -- so the in-body copy won and the
    # framework's richer one was dead text.  Two on-disk sources of truth
    # for the region partition the whole ladder is built on.
    # `cfg` THROUGH, because the species order is a value a person may set
    # and this block writes `ChemicalSpeciesLabel`.  It was received and
    # dropped here, which is the whole of why `species_order` reached every
    # SIESTA deck and no transport deck (`model/chemistry.md` § 3a).
    return "\n".join(_emit_geometry(struct, cfg=cfg, frame=frame))


def _emit_restart_group(struct, cfg) -> str:
    """``DM.UseSaveDM`` / ``MD.UseSaveXV`` — written in BOTH states.

    `siesta/input.py` records the measured reason this is not optional:
    *"SIESTA reads `<SystemLabel>.DM` when the file is there whatever the deck
    omits."*  A deck that says nothing therefore warm-starts from whatever the
    directory happens to hold, which is how a rung told to start clean
    silently continued.  The old transport seed omitted the group entirely and
    its own docstring claimed *"the seed itself starts fresh"* — a claim the
    file could not keep.

    The keys and the on/off come from the ONE declaration
    (:func:`molbuilder.siesta.input._restart_group_lines`), so this cannot
    drift from what `warm_declaration("seed", …)` promises to carry.
    """
    from ..siesta.input import _restart_group_lines

    return "\n".join([
        "# --- Restart: what this rung reads if it is there ---",
        "#",
        "# Written in BOTH states on purpose: SIESTA reads <SystemLabel>.DM",
        "# whenever the file exists, whatever the deck leaves out, so an",
        "# omitted group means 'warm-start from whatever is in this",
        "# directory' rather than 'start clean'.",
        "#",
        "# For the SEED the file in question is its OWN previous attempt's",
        "# density, which is what `--from` carries and what makes re-running",
        "# an unconverged seed cheap.  It is never the device's: the arrow",
        "# runs the other way (seed .DM -> device SCF).",
        *_restart_group_lines(cfg),
        "",
    ])


def _emit_solver_note(struct, cfg) -> str:
    """Why the seed solves with ``diagon`` — restored from the deck this
    layout replaced.

    ``SCF_SECTION`` is the ENGINE's object, shared with every other kind, so
    its ``solution_method`` help is necessarily a generic three-option menu.
    The transport-specific half — *the seed must NOT be transiesta* — lived at
    the keyword in the old hand-written deck and was lost when the generic
    section took over.  It goes back adjacent to the value, because that is
    where someone about to edit the value will read it, and it is a Block
    rather than a note on the section because mutating a shared section would
    put this text into every kind's deck.
    """
    return "\n".join([
        "# --- The seed's solver: ordinary diagonalisation, NOT transiesta ---",
        "#",
        "# This rung is a periodic WARM-UP, not the NEGF calculation.  Leave",
        "# `SolutionMethod` at `diagon` below: setting it to `transiesta`",
        "# here would make the seed attempt an open-boundary solve with no",
        "# electrode self-energies defined, which is not what this deck is",
        "# and not what the ladder needs from it.",
        "#",
        "# SIESTA writes <SystemLabel>.DM as this SCF converges, and that",
        "# file -- nothing else from this rung -- is what the device stage",
        "# reads as its starting density.  There is no MD block anywhere in",
        "# this deck, and that absence is what makes it a single point: the",
        "# geometry was relaxed upstream and moving it here would invalidate",
        "# the electrode partition the whole ladder is built on.",
        "#",
        "# WRITTEN HERE, not by a section: `solution_method` is `role`-",
        "# declared for transport, so the rung answers it and the template",
        "# carries no value for it.  A section rendering it would resolve",
        "# the config DEFAULT instead, which is how the device deck came to",
        "# say `diagon` from a section and `transiesta` from its own block",
        "# -- in that order, with libfdf taking the first.",
        "SolutionMethod         diagon",
        "#",
        "# PSEUDOPOTENTIALS: this rung does not name a `psml_lib`.  Its",
        "# .psml files travel with the CITED junction -- `prep` copies them",
        "# into the calculation's `pseudos/` directory beside this deck, and",
        "# `jobset init` refuses a --psml-lib for a transport calculation",
        "# for exactly that reason.  If SIESTA cannot find a pseudo, the",
        "# citation did not carry it; do not add a library path here.",
        "",
    ])


def _emit_kgrid_block(struct, cfg) -> str:
    """``%block kgrid_Monkhorst_Pack`` with the transport axis forced to 1.

    A ``%block`` is structural, so it is a block — but the VALUE is the
    template's ``kgrid`` row, and the forced third component is a DERIVED
    value, which has its own framework door
    (:func:`~molbuilder.script_emit.parameter` with ``value=``) rather than
    falling to free-form text where the note-with-the-value rule cannot reach
    it.
    """
    kx, ky, _kz = tuple(cfg.kgrid or (1, 1, 1))
    p = _sc.parameter("kgrid", "siesta", value=(int(kx), int(ky), 1))
    out = list(p.note())
    out += [
        "# The transport direction is NOT BZ-summed -- NEGF handles it, and",
        "# the engine preflight refuses kz != 1 -- so the third component is",
        "# 1 whatever the citation's own k-grid said.  (`config_for` already",
        "# forced it when it read the citation; this writes what it was",
        "# given and is not a second enforcer.)",
        "#",
        "# THE TRANSVERSE COUNTS ARE YOURS, AND 1 x 1 IS RARELY RIGHT.",
        "# For a finite molecule between leads, (1, 1) is correct.  For a",
        "# laterally PERIODIC electrode -- an Au(111) surface cell, a",
        "# nanowire -- set Nx, Ny to that lead's periodicities: a metallic",
        "# lead sampled 1 x 1 is badly under-converged, and the error lands",
        "# in the interface charge the device SCF then has to reproduce.",
        "# Nz stays 1 regardless.  (This deck's transverse pair comes from",
        "# the cited junction's own k-grid -- see the note above the value.)",
        "%block kgrid_Monkhorst_Pack",
        f"  {int(kx):>3}    0    0      0.0",
        f"    0  {int(ky):>3}    0      0.0",
        "    0    0    1      0.0",
        "%endblock kgrid_Monkhorst_Pack",
    ]
    return "\n".join(out)


# ===================================================================== #
#  The spec.                                                            #
# ===================================================================== #

def transport_spec(struct: Structure, cfg, *,
                   stage_token: Optional[str] = None,
                   state=None) -> "_sc.DeckSpec":
    """The ``DeckSpec`` for one transport rung.

    *cfg* is a :class:`~molbuilder.config.siesta.SiestaConfig` — carrying the
    calculation's own template ⊕ this rung's overrides, as
    :func:`molbuilder.jobset.prep._resolve_transport` resolves it (TR4) —
    because the sections above name catalogue rows and
    :func:`~molbuilder.script_emit.parameter` resolves a row by
    ``getattr(config, name)``.  That is not a preference: an item whose name
    is not a field on the config resolves to ``None`` *silently*, so a
    transport deck rendered from a config with different field names would
    quietly omit every one of the 21 rather than fail.
    """
    rung = rung_of(stage_token)
    shape = SHAPE_OF_RUNG.get(rung)
    if shape is None:
        raise ValueError(
            f"{rung!r} is not a transport rung; the ladder is "
            f"{', '.join(SHAPE_OF_RUNG)} (engines/transport.md 4.2).  "
            f"`prep` names the rung and hands it down as `stage_token`.")

    # NO SECOND REFUSAL HERE.  One stood between these two lines, for a shape
    # "not on the seam yet" -- and `SHAPE_OF_RUNG`'s value set is exactly the
    # four keys below, so it could not fire.  It was the last text describing
    # a migration this module has finished.
    # THE ONE STATE of the calculation (`science/chemistry-correctness.md`
    # § 2a): `prep` resolves it ONCE, on the whole junction, and hands it to
    # every rung -- a rung's own structure (a lead, the device) is not where
    # a blank spin is decided.  A caller that hands none is answered on this
    # structure.
    handed = state is not None
    if state is None:
        from ..electronic_state import electronic_state
        state = electronic_state(struct, cfg, kind="transport")
    derived = _derived_for(state, cfg)
    from .transiesta import engine_frame_for
    frame = engine_frame_for(struct)
    layout = {"seed": _seed_layout,
              "electrode": _electrode_layout,
              "device": _device_layout,
              "transmission": _transmission_layout}[shape](
                  derived, frame, _state_block(state, on_junction=handed))
    return _sc.DeckSpec(
        engine="siesta",
        engine_frame=frame,
        calculation="transport",
        layout=layout,
        line=_sl.line(derived),
        derived=derived,
        note_lead=_sl.note_lead,
        check_rules=_sl.check_rules,
        created_by="molbuilder transport prep",
    )


def _derived_for(state, cfg) -> dict:
    """What this deck worked out — W10's one per-render context.

    DECLARED on the form rather than only closed over, so a reader outside
    this module can see where a value came from.  The three groups
    :func:`molbuilder.siesta.layout.line` needs are the same ones the
    optimization deck derives; they are computed by the engine's own helper so
    the two kinds cannot answer them differently.  The spin's are the
    calculation's one state -- for the transport kind the charge is 0 by rule
    (the leads set the electron number) and the spin is the rungs' shared
    answer.
    """
    from ..siesta.input import _parallel_facts, _spin_facts

    derived = {}
    derived.update(_spin_facts(state))
    derived.update(_parallel_facts(cfg))
    return derived


def _state_block(state, *, on_junction: bool):
    """WHERE THE CHARGE AND SPIN CAME FROM, in every rung's deck
    (`science/chemistry-correctness.md` § 2a.5, ES2) -- each value beside its
    source, as the optimization deck writes them.  The rungs wrote none until
    the M6 review, and the spin `prep` handed them read as *stated* whatever
    decided it."""
    t, c, q = state.spin_treatment, state.unpaired_electrons, state.net_charge

    def emit(struct, cfg) -> str:
        lines = [f"# Spin: {t.value} ({t.said});",
                 f"#   unpaired electrons (2S): {c.value} ({c.said})."]
        if on_junction:
            lines += ["#   Decided ONCE, on the whole junction, and the same "
                      "on every rung:",
                      "#   TranSIESTA joins the leads' self-energies to the "
                      "device, so all",
                      "#   five rungs solve the same spin channels."]
        lines.append(f"# NetCharge: not written -- {q.value:+d} ({q.said}).")
        return "\n".join(lines)

    return _sc.Block("where the charge and spin came from", emit)

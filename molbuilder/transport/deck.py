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
* **structural text is a BLOCK** — the coordinate table, ``%block TS.Elecs``,
  the contour blocks.  Those are lifted whole.

``_emit_basis_and_xc`` is therefore NOT lifted: its six keywords are exactly
``BASIS_SECTION`` + ``XC_SECTION`` + ``electronic_temperature``, and lifting
it beside them would write each twice — which ``layout.check_rules`` now
catches ("written twice with different values"), because a migrated deck gets
the engine's check gate for the first time.

THE ADAPTER, and why it is here rather than in the emitters.  The lifted
emitters read four names the shared config spells differently —
``siesta_mesh_cutoff_ry``, ``energy_shift_ry``, ``electronic_temperature_k``,
``k_mesh_transverse`` against the catalogue's ``mesh_cutoff``,
``pao_energy_shift``, ``electronic_temperature``, ``kgrid``.  That is the same
count and the same shape as vibration's four, and vibration's module records
why the adapter is the seam's right answer: *"the kind's science and the
emitters both read one view, so a check and the deck it checks cannot
disagree about a value"*.  Renaming inside the old emitters instead would be
repatching the old path.

**ALL FIVE RUNGS ARE ON THE SEAM.**  Three deck shapes serve them
(:data:`SHAPE_OF_RUNG`), and each is a layout in this module.  The seam
question that held ``electrode`` and ``negf`` back — *what does a composite
kind hand its renderer, when the deck describes something the citation was
composed into?* — is answered, and the answer is that it hands a
**structure**, like every other kind: `prep` picks WHICH structure the rung
describes (the junction, or the lead taken out of it by its region label) and
``spec_for(struct, cfg, stage_token=)`` is unchanged.  Nothing reaches for the
``ComposedJunction`` from inside here.

Which shape a rung gets is a TABLE.  Choosing the LAYOUT for a shape is a
dispatch in :func:`transport_spec` over that table's three values.  A rung the
table does not name is refused BY NAME rather than rendered partially; `prep`
translates that refusal, so it reaches a person as a message and not a
traceback.

"""
from __future__ import annotations

import dataclasses as _dc
from typing import Optional

from .. import script_emit as _sc
from ..siesta import layout as _sl
from ..structure import Structure

#: Which deck SHAPE each rung renders — a table, because THREE texts serve
#: FIVE rungs (`engines/transport.md` § 6.1) and which text a rung gets is a
#: fact about the ladder, not a decision to re-derive per call.
#:
#: The device and the transmission share one SHAPE deliberately -- not the
#: same bytes, which is a retired claim (`_negf_layout`): each rung resolves
#: its own config, so a tuned transmission parameter makes the two decks
#: differ in exactly that value.  What the shared shape buys is that they
#: cannot drift apart about what the junction IS -- same geometry, same
#: electrode declarations, same electronic description.
SHAPE_OF_RUNG = {
    "seed":         "seed",
    "electrode_L":  "electrode",
    "electrode_R":  "electrode",
    "device":       "negf",
    "transmission": "negf",
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

def _negf_layout(derived):
    """The device and transmission rungs — an open-boundary NEGF deck.

    One layout serves both, and **that is not the old "same bytes" claim
    returning.**  Each rung resolves its OWN config, so where a person has
    tuned a transmission parameter the two decks differ in exactly that value
    — which is what § 2a.7's ruling asks for.  What they share is the shape:
    the same junction, the same electrode declarations, the same electronic
    description, so the two runs cannot drift apart about what the junction
    IS.

    The electrode declarations stay a ``Block``.  ``%block TS.Elecs`` and the
    per-electrode blocks are derived from the structure's own ``regions`` —
    which atoms are a lead, and therefore which contiguous range TranSIESTA
    is told about — and no parameter models that.  It is lifted whole from
    the emitter that has been getting it right, rather than rewritten: this
    is the part where a mistake is silent and expensive.
    """
    return (
        _sc.Block("identity and what this rung computes", _emit_negf_header),
        _sc.Block("cell, coordinates and region metadata",
                  _emit_geometry_block),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the transverse k-mesh (the transport axis is not sampled)",
                  _emit_kgrid_block),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        # NO RESTART BLOCK HERE.  The NEGF block below writes
        # `DM.UseSaveDM` itself -- it is how the seed's density is picked
        # up -- and writing it twice is what the check gate refuses.  The
        # lift boundary is drawn at the keyword, not at the topic.
        _sl.spin_section(polarized=derived["spin_polarized"],
                         fixed=derived["spin_fixed"]),
        _sl.mpi_section(block_size=derived.get("block_size"),
                        algorithm=derived.get("algorithm")),
        CONTOUR_SECTION,
        _sc.Block("the NEGF electrode declarations", _emit_negf_block),
        _sl.OUTPUT_SECTION,
    )


#: The equilibrium contour's POLE COUNT — a keyword with a value, therefore a
#: section item and not part of the lifted block beside it.
#:
#: The block below writes ``TS.Contours.Eq.Pole``, the pole ENERGY, and by this
#: module's own boundary rule that belongs here too -- it stays there because
#: the block has a SECOND caller, ``transiesta.render_script`` (the
#: `/api/transport/render` validation surface), which composes no sections at
#: all.  Moving the line would drop the keyword from that deck silently, and
#: writing it in both places is what ``layout.check_rules`` refuses.  The two
#: rejoin when that surface retires.
#:
#: The COUNT is a different keyword and a ``SiestaConfig`` field, so the
#: projection in :func:`_legacy_view` never carried it and no layout named it:
#: the catalogue declared ``TS.Contours.Eq.Pole.N`` with an anchor, a default
#: of 20 and a range, and **nothing wrote it into any deck** (2026-09-16).
#: The row's own help says what that cost — *"a device run could abort with
#: `the continued fraction method requires at least 20 poles` after the queue
#: wait: the count fell back to a default the deck never stated and could not
#: raise."*
CONTOUR_SECTION = _sc.Section(
    "The equilibrium contour's pole COUNT (its energy is in the block below)",
    ("negf_eq_pole_n",))


def _emit_negf_header(struct, cfg) -> str:
    label = cfg.system_label
    return "\n".join([
        "# ================================================================== #",
        f"#  TranSIESTA DEVICE .fdf — {label}",
        "#  The open-boundary NEGF calculation on the composed junction.",
        "#  The leads are not solved here: each one's Hamiltonian was",
        "#  computed by its own rung and is read from the .TSHS named in",
        "#  the TS.Elec block below, which is what makes this an OPEN",
        "#  boundary rather than a bigger periodic cell.",
        "#",
        "#  The same text serves the transmission rung, run under tbtrans:",
        "#  TS.* keywords are inert to tbtrans and TBT.* to siesta, so each",
        "#  binary reads its own half.",
        "# ================================================================== #",
        "",
        f"SystemLabel            {label}",
        f"SystemName             Transport device for {label}",
    ])


def _emit_negf_block(struct, cfg) -> str:
    """The NEGF half, LIFTED from the emitter that has been getting it right.

    ``%block TS.Elecs``, one ``%block TS.Elec.<name>`` per side with its
    ``.TSHS`` filename, atom count, chemical potential, semi-infinite
    direction and explicit position; the buffer atoms; the contour settings;
    and the TBtrans window.

    **Not rewritten, and deliberately so.** This is where a mistake is silent
    and expensive: TranSIESTA identifies each electrode by a CONTIGUOUS ATOM
    RANGE, so an off-by-one in a position line computes transmission through
    a region that is not the molecule, and converges while doing it. The
    emitter that produces it has been measured against a live 5.4.2 binary;
    a second implementation would have to earn that again for no gain.
    """
    from .transiesta import _emit_transiesta_block

    # The lifted emitter reads a TransportConfig.  It is projected here, at
    # the boundary, exactly as `vibration_deck` projects for its own lifted
    # emitters -- and it dies when that emitter is tabled (TR5c).
    view = _legacy_view(cfg)
    return "\n".join(_emit_transiesta_block(struct, view))


def _legacy_view(cfg):
    """A ``TransportConfig`` carrying this rung's answers.

    The NEGF emitter above predates the seam and reads the older config. One
    projection, at the one place the two vocabularies still meet, rather than
    a rename inside a proven emitter — which is the direction the rulings
    forbid (*"repatching the old path"*).

    It is the last of its kind: TR4 deleted the general projection when the
    template made it unnecessary, and this one goes when the NEGF block is
    tabled.
    """
    from ..config.transport import TransportConfig

    known = {f.name for f in _dc.fields(TransportConfig)}
    kw = {}
    for src, dst in (("system_label", "job_name"),
                     ("mesh_cutoff", "siesta_mesh_cutoff_ry"),
                     ("pao_energy_shift", "energy_shift_ry"),
                     ("electronic_temperature", "electronic_temperature_k"),
                     ("kgrid", "k_mesh_transverse")):
        if hasattr(cfg, src) and dst in known:
            kw[dst] = getattr(cfg, src)
    for f in _dc.fields(type(cfg)):
        if f.name in known and f.name not in kw:
            kw[f.name] = getattr(cfg, f.name)
    kw.pop("engine", None)
    if "siesta_mesh_cutoff_ry" in kw and kw["siesta_mesh_cutoff_ry"] is not None:
        kw["siesta_mesh_cutoff_ry"] = int(round(float(kw["siesta_mesh_cutoff_ry"])))
    # THE BIAS this rung runs at.  `bias_voltage_v` is the template's single
    # value; the emitter takes a list because it predates the axis rule.
    kw["bias_voltages_v"] = [float(getattr(cfg, "bias_voltage_v", 0.0) or 0.0)]
    return TransportConfig(engine="transiesta", **kw)


def _electrode_layout(derived):
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
        _sc.Block("cell and coordinates", _emit_geometry_block),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the k-mesh — transverse shared, transport axis DENSE",
                  _emit_electrode_kgrid_block),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(polarized=derived["spin_polarized"],
                         fixed=derived["spin_fixed"]),
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
        "TS.HS.Save             true",
        "",
        "# An ordinary diagonalisation: a lead is a periodic bulk crystal,",
        "# not an open boundary.  `role`-declared like TS.HS.Save above, so",
        "# the rung writes it and no section does.",
        "SolutionMethod         diagon",
    ])


def _seed_layout(derived):
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
        _sc.Block("cell, coordinates and region metadata", _emit_geometry_block),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the transverse k-mesh (kz forced to 1)", _emit_kgrid_block),
        _sc.Block("what the seed's solver must be, and must not",
                  _emit_solver_note),
        _sc.Block("the restart group", _emit_restart_group),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(polarized=derived["spin_polarized"],
                         fixed=derived["spin_fixed"]),
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


def _emit_geometry_block(struct, cfg) -> str:
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
    return "\n".join(_emit_geometry(struct))


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
                   stage_token: Optional[str] = None) -> "_sc.DeckSpec":
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
    # three keys below, so it could not fire.  It was the last text describing
    # a migration this module has finished.
    derived = _derived_for(struct, cfg)
    layout = {"seed": _seed_layout,
              "electrode": _electrode_layout,
              "negf": _negf_layout}[shape](derived)
    return _sc.DeckSpec(
        engine="siesta",
        calculation="transport",
        layout=layout,
        line=_sl.line(derived),
        derived=derived,
        note_lead=_sl.note_lead,
        check_rules=_sl.check_rules,
        created_by="molbuilder transport prep",
    )


def _derived_for(struct, cfg) -> dict:
    """What this deck worked out — W10's one per-render context.

    DECLARED on the form rather than only closed over, so a reader outside
    this module can see where a value came from.  The three groups
    :func:`molbuilder.siesta.layout.line` needs are the same ones the
    optimization deck derives; they are computed by the engine's own helper so
    the two kinds cannot answer them differently.
    """
    from ..siesta.input import _parallel_facts, _spin_facts

    derived = {}
    derived.update(_spin_facts(cfg))
    derived.update(_parallel_facts(cfg))
    return derived

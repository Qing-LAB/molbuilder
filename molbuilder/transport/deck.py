"""The transport calculation's deck — SIESTA, on the framework's seam.

Contract: [`engines/transport.md`](?doc=engines/transport.md) §§ 3.2, 3.6, 6.1
(what transport's decks are) +
`script-preparation.md` § 4 (the seam this serves).

WHAT THIS IS.  ``transport_spec(struct, cfg, stage_token)`` returns the
:class:`~molbuilder.script_emit.DeckSpec` for ``calculation = "transport"``.
It is the exact shape :mod:`molbuilder.pyscf.vibration_deck` has for
``calculation = "vibration"``: the KIND is a render argument, the seam stays
ONE per engine, and the kind's own module owns its layout.

WHY IT EXISTS.  A transport rung's deck is rendered on the seam, so every
item its sections name reaches it from the template; a keyword absent from
the file would run at SIESTA's own default with no way to ask for another
(`transport.md` § 3.2).

THE BOUNDARY between items and blocks is drawn by a single question, and it
is the question `script-preparation.md` § 4.1 asks:

* **a keyword with a value is a SECTION ITEM**, resolved from its catalogue
  declaration.  The template's items arrive because
  :mod:`molbuilder.siesta.layout`'s sections already name them, and transport
  is a calculation KIND on the siesta engine (39 shared rows, measured in
  § 2a.13) so it reuses those sections rather than restating them.
* **structural text is a BLOCK** — the coordinate table, ``%block TS.Elecs``
  and the reservoir blocks.  Those are written whole; a block whose VALUES are
  catalogue items (the T(E) window) asks each value through the framework's
  door and writes it with its note.

The NEGF settings are catalogue items written by the section walk with their
notes (the three sections below); the electrode and reservoir declarations
read the engine's own config (`engines/transport.md` § 6.1b).

**ALL FIVE RUNGS ARE ON THE SEAM.**  Four deck shapes serve them
(:data:`SHAPE_OF_RUNG`), and each is a layout in this module.  A composite
kind hands its renderer a **structure**, like every other kind: `prep` picks WHICH structure the rung
describes (the junction, or the lead taken out of it by its region label) and
``spec_for(struct, cfg, names=)`` is unchanged.  Nothing reaches for the
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
from ..parse.engines import siesta_grammar as _G
from ..runfiles import compose
from ..siesta import layout as _sl
from ..structure import Structure
from ..kmesh import TRANSMISSION_PER_SCF as _TBT_FACTOR

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
    from ..identity import parse_token
    tok = str(stage_token or "")
    got = parse_token(tok)
    return got[1] if got else tok


# ===================================================================== #
#  The layouts, as tables.                                              #
# ===================================================================== #

#: TranSIESTA's own settings for the device's self-consistent run -- read by
#: `siesta` in NEGF mode.  The voltage leads it: a `role` item, the bias point
#: this rung runs at, which `resolve` and the point `prep` renders put into
#: the rung's config (`engines/template.md` § 6.4).
TS_DEVICE_SECTION = _sc.Section(
    "TranSIESTA -- how the device's open-boundary run is solved",
    ("bias_voltage_v", "electrodes_bulk", "ts_elecs_eta_ev",
     "negf_eq_pole_ev", "negf_neq_eta_ev", "negf_neq_fermi_cutoff_kt",
     "ts_scf_dq_converge", "ts_scf_dq_tolerance",
     "ts_solution_method", "ts_hartree_fix", "ts_elecs_out_of_core"),
    note=(
        "# Read by siesta in its NEGF mode (TranSIESTA) -- this rung.  An item",
        "# not set is not written, and TranSIESTA applies its own rule there:",
        "# its line says which.  A 0 is written as 0 -- a value of its own",
        "# (engines/transport.md 6.1b).",
    ))

#: What tbtrans reads of TranSIESTA's settings: `TBT.Voltage` defaults to
#: `TS.Voltage` (`m_tbt_hs.F90`, `m_tbt_contour.F90`) and `TBT.Elecs.Bulk` to
#: `TS.Elecs.Bulk` (`m_tbt_options.F90`, SIESTA 5.4.2), so the device's values
#: are written here too.  The contour settings are not: they say how the
#: device's density is integrated, and tbtrans integrates none.
TS_READ_BY_TBTRANS_SECTION = _sc.Section(
    "TranSIESTA settings tbtrans reads",
    ("bias_voltage_v", "electrodes_bulk"),
    note=(
        "# tbtrans takes these as the defaults of its own TBT.Voltage and",
        "# TBT.Elecs.Bulk: the bias is the point the device was converged",
        "# at, and the leads are treated as the device run treated them.",
    ))

#: What a lead and the device exist to write -- ``TS.HS.Save``, a `role` item
#: whose value is the three rungs' answer (`engines/template.md` § 6.4): a
#: lead's ``.TSHS`` the device attaches to, the device's ``.TS.HSX`` the
#: transmission attaches to.  Its note says why it is this keyword and not
#: ``SaveHS`` or ``TS.DE.Save``.
TS_ELECTRODE_SECTION = _sc.Section(
    "The Hamiltonian this rung exists to write",
    ("ts_hs_save",))

#: WHAT TBTRANS READS OF THE SCF SETTINGS: the electronic temperature alone
#: -- its own starts from ``ElectronicTemperature`` (``m_tbt_options.F90``).
#: It runs no SCF, so no solver, mixer, tolerance or iteration limit is
#: written here (§ 6.1b's audit, 2026-09-29 / 2026-10-08: tbtrans compiles
#: none of the files that read them); those items declare the SCF rungs as
#: their `stages`, so no surface offers them on this rung either.
_TRANSMISSION_TEMPERATURE_SECTION = _sc.Section(
    "The electronic temperature tbtrans starts from",
    ("electronic_temperature",))

#: TBtrans's own settings -- read by `tbtrans` alone; `siesta` holds no
#: `TBT.*` keyword, which is why the device deck carries none.
TBT_SECTION = _sc.Section(
    "TBtrans -- how T(E) is computed from the device's Hamiltonian",
    ("tbt_solution_method", "tbt_elecs_eta_ev", "tbt_contours_eta_ev",
     "tbt_spin",
     "tbt_dos_gf", "tbt_dos_a", "tbt_dos_elecs", "tbt_t_eig", "tbt_t_bulk",
     "tbt_t_all", "tbt_verbosity", "tbt_progress"),
    note=(
        "# Read by tbtrans, which runs no SCF: it takes the converged",
        "# Hamiltonian the device rung wrote (TBT.HS above) and the leads'",
        "# .TSHS, and evaluates T(E) on the window above.  An item not set",
        "# is not written, and tbtrans applies its own rule there: its line",
        "# says which.  A 0 is written as 0 -- a value of its own.",
    ))


def _device_layout(derived, frame, state_block, point, scratch_dir):
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
        _sc.Block("the k-point grid -- one k-point along z",
                  _k_mesh_block(derived["k_mesh"], advice=_TRANSVERSE_ADVICE)),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        # NO RESTART BLOCK HERE.  `_emit_device_run` writes `DM.UseSaveDM`
        # -- it is how the seed's density is picked up -- and writing it
        # twice is what the check gate refuses.  The boundary is drawn at
        # the keyword, not at the topic.
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(algorithm=derived.get("algorithm")),
        _sc.Block("what this rung runs", _emit_device_run),
        # THE LEADS' SURFACE GREEN'S FUNCTIONS ARE THE RUN'S, in its
        # scratch folder (§ 2a.11, *What a run's points share*), or beside
        # the run's files where the vocabulary states none: the block names
        # them there from the point's folder.
        _sc.Block("the junction: its electrodes and reservoirs",
                  _electrode_block(_gf_folder(point, scratch_dir))),
        TS_DEVICE_SECTION,
        # THE BIAS WINDOW'S LINE, written at every point with the window the
        # cutoff gives; TranSIESTA reads it under bias alone (§ 2a.11).
        _sc.Block("the non-equilibrium contour -- the bias window's line",
                  _emit_neq_contour),
        TS_ELECTRODE_SECTION,
        _sl.OUTPUT_SECTION,
    )


def _transmission_layout(derived, frame, state_block, point, scratch_dir,
                         contours=None):
    """The transmission rung -- `tbtrans` on the device's converged
    Hamiltonian.

    Read by `tbtrans`, which reads the junction description (the electrode
    and reservoir blocks, the voltage, the bulk treatment) as TranSIESTA
    wrote it and its own `TBT.*` settings.  **Of the SIESTA settings the
    rungs share it keeps what it reads** (§ 6.1b's audit of its source:
    `SystemLabel`, `SystemName`, `ElectronicTemperature`, `Spin`, the
    `TS.Elec*` / `TS.ChemPot*` blocks, `TS.Voltage`, `TS.Elecs.Bulk`, its
    own `TBT.*`): the basis and functional sections name the Hamiltonian it
    was built with, the temperature its Fermi functions start from
    (`m_tbt_options.F90`).  The SCF settings, the solver, the MPI
    diagonaliser lines (`Diag.ParallelOverK`, `BlockSize`, the ELPA pair)
    and the output group are not written -- tbtrans compiles none of the
    files that read them (``read_options.F90``, ``bands.F``,
    ``write_subs.F``, ``outcoor.f``).
    """
    return (
        _sc.Block("identity and what this rung computes",
                  _emit_transmission_header),
        _sc.Block("cell, coordinates and region metadata",
                  _geometry_block(frame)),
        _sl.BASIS_SECTION,
        _sl.XC_SECTION,
        _sc.Block("the shared k-point grid (tbtrans reads TBT.k below)",
                  _k_mesh_block(derived["k_mesh"])),
        _TRANSMISSION_TEMPERATURE_SECTION,
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sc.Block("what this rung reads", _emit_transmission_run),
        # tbtrans writes no Green's function file (§ 2a.11): no `GF` line.
        _sc.Block("the junction: its electrodes and reservoirs",
                  _electrode_block(None)),
        TS_READ_BY_TBTRANS_SECTION,
        _sc.Block("the energy window T(E) is computed on",
                  _tbt_energies(contours)),
        _sc.Block("the transmission k-point grid",
                  _k_mesh_block(derived["tbt_k_mesh"])),
        TBT_SECTION,
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
        "#  and its own TBT.* settings (engines/transport.md 6.1b).  Of the",
        "#  SIESTA settings only what it reads is here: the label, the",
        "#  basis and functional, the electronic temperature, the spin.",
        "#",
        "#  What tbtrans actually read -- every keyword it asked for and the",
        "#  value it used, this deck's or its own default -- is in its own",
        "#  log beside the output: fdf.<date>T<time>.log.",
        "# ================================================================== #",
        "",
        f"SystemLabel            {label}",
        f"SystemName             Transport transmission for {label}",
    ])


def _notes(cfg, lines):
    """A block's explanation, dropped when the deck is asked to be quiet --
    what the section walk does with a section's note (`verbose`)."""
    return list(lines) if getattr(cfg, "verbose_comments", True) else []


def _emit_device_run(struct, cfg) -> str:
    """What makes this deck the device's: TranSIESTA, started from the seed's
    density.  Its bias point is `TS.Voltage` in the TranSIESTA section."""
    return "\n".join([
        "# --- TranSIESTA, at this rung's bias point ---",
        "#",
        "# `SolutionMethod transiesta` -- in the SCF section above, fixed by",
        "# this rung -- switches the SCF cycle to NEGF.  (`TS.SolutionMethod`,",
        "# in the TranSIESTA section below, is a different keyword: how the",
        "# NEGF Green's function is inverted.)",
        "",
        "# Start the NEGF SCF from a saved density when one is present: the",
        "# seed rung leaves <SystemLabel>.DM beside this deck, and a .TSDE",
        "# is read through the same switch.  Written, though SIESTA's default",
        "# is true, so the deck states its answer.  With no file, SIESTA",
        "# starts from atomic densities.",
        "DM.UseSaveDM           true",
        "",
    ])


def _emit_transmission_run(struct, cfg) -> str:
    """What makes this deck the transmission's: the device Hamiltonian it
    reads, named.  The bias point it reads it at is `TS.Voltage` in the
    section of TranSIESTA settings tbtrans reads."""
    label = cfg.system_label
    return "\n".join([
        "# --- tbtrans, on the device rung's converged Hamiltonian ---",
        "#",
        "# No SolutionMethod: tbtrans runs no SCF.",
        "",
        "# WHERE the device Hamiltonian is.  tbtrans would pick the first of",
        f"# {label}.TS.HSX, .TSHS and .HSX that exists (m_tbt_hs.F90); naming it",
        "# says which one this transmission is of, rather than leaving that to",
        "# whichever file is present.",
        f"TBT.HS                 {compose(label, _G.TS_HSX)}",
        "",
    ])


def _electrode_block(gf_in):
    """The electrode and reservoir declarations' Block -- one text for both
    NEGF rungs (`transiesta.emit_electrode_declarations`), the device's
    naming the leads' surface Green's functions in ``gf_in``, its run's
    folder as the point reaches it; the transmission's (``None``) naming
    none (`engines/transport.md` § 2a.11, *What a run's points share*)."""
    from .transiesta import emit_electrode_declarations

    def render(struct, cfg) -> str:
        return "\n".join(emit_electrode_declarations(struct, cfg,
                                                     gf_in=gf_in))
    return render


def _gf_folder(point, scratch_dir: Optional[str]) -> str:
    """Where a device point's deck names the leads' surface Green's
    functions, from the point's folder: the run's scratch folder -- the
    calculation's vocabulary's ``scratch_dir`` (`warm-files.toml`
    [transport], `engines/transport.md` § 2a.11) -- or, where it states
    none, beside the run's files, as the vocabulary says."""
    to_run = point.to_run if point is not None else "."
    return to_run if scratch_dir is None else f"{to_run}/{scratch_dir}"


def _emit_neq_contour(struct, cfg) -> str:
    """``%block TS.Contours.nEq`` and ``%block TS.Contour.nEq.line`` -- the
    bias window's line, the block TranSIESTA builds itself when none is
    written (`m_ts_contour_neq.f90`), written at every device point so the
    spacing is a choice on the card and never the engine's silent one
    (`engines/transport.md` § 2a.11, *The levers*).  Its window is the
    cutoff's (`negf_neq_fermi_cutoff_kt`), rendered from that item and never
    typed a second time; its spacing the card's (`negf_neq_spacing_ev`).
    TranSIESTA reads the block under bias alone."""
    cutoff = _sc.parameter("negf_neq_fermi_cutoff_kt", "siesta", config=cfg)
    delta = _sc.parameter("negf_neq_spacing_ev", "siesta", config=cfg)
    kt = float(cutoff.value)
    return "\n".join([
        *_notes(cfg, [*delta.note()]),
        "",
        "# THE BIAS WINDOW'S LINE, as TranSIESTA would build it when none is",
        "# written (m_ts_contour_neq.f90): from -cutoff kT - |V|/2 to |V|/2 +",
        "# cutoff kT, the cutoff TS.Contours.nEq.Fermi.Cutoff's above, at the",
        "# spacing set on the card.  Read under bias alone.",
        "%block TS.Contours.nEq",
        "  line",
        "%endblock TS.Contours.nEq",
        "",
        "%block TS.Contour.nEq.line",
        "  part line",
        f"    from - {kt:g} kT - |V|/2 to |V|/2 + {kt:g} kT",
        f"      delta {float(delta.value):g} eV",
        "        method mid-rule",
        "%endblock TS.Contour.nEq.line",
    ])


def _emit_tbt_window(struct, cfg) -> str:
    """``%block TBT.Contour.window`` -- the energy grid T(E) is computed on.

    A ``%block`` is structural, so it is a block; its three VALUES are three
    catalogue rows, each asked through the framework's door so each arrives
    with its note, as the k-point mesh's items do (`siesta.layout.k_mesh_lines`).
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
        # ITS ENDS AT FULL PRECISION: tbtrans computes its energies from
        # them, and an end rounded is another energy (`engines/transport.md`
        # § 2a.16).
        f"   from {float(lo.value)!r} {_sl.unit_word('transmission_emin_ev')}"
        f" to {float(hi.value)!r} {_sl.unit_word('transmission_emax_ev')}",
        f"    points {int(n.value)}",
        "     method mid-rule",
        "%endblock TBT.Contour.window",
        "",
    ])


def _tbt_energies(contours):
    """The transmission's energies: its window whole, or -- for a point
    that shares energies with a finished calculation's point
    (`energies.Contours`, `engines/transport.md` § 2a.16) -- the energies
    it computes, as line contours, every one named in ``TBT.Contours``, and
    the record of its sharing, molbuilder's, which the engine does not
    read."""
    if contours is None:
        return _emit_tbt_window

    def emit(struct, cfg) -> str:
        import json
        from ..deck_record import begin_marker, end_marker
        from .energies import BLOCK_SHARED
        names = [f"computed{i + 1}" for i in range(len(contours.lines))]
        out = [
            "",
            f"# THE ENERGIES THIS POINT COMPUTES: {contours.computed} of the "
            f"{contours.window[2]} on its",
            f"# grid.  The other {contours.held} are the finished point's, "
            f"which this point shares",
            "# (its shared/):",
            f"#   {contours.source}",
            "# Line contours, each a run of these energies at one spacing --",
            "# tbtrans says they are not connected (`engines/transport.md`",
            "# § 2a.16).",
            begin_marker(BLOCK_SHARED),
            *(f"# {ln}" for ln in json.dumps(contours.to_record(),
                                            indent=2).splitlines()),
            end_marker(BLOCK_SHARED),
            "%block TBT.Contours",
            *(f"  {name}" for name in names),
            "%endblock TBT.Contours",
        ]
        unit = _sl.unit_word("transmission_emin_ev")
        for name, (a, b, n) in zip(names, contours.lines):
            out += ["", f"%block TBT.Contour.{name}", "  part line",
                    f"   from {a!r} {unit} to {b!r} {unit}",
                    f"    points {n}", "     method mid-rule",
                    f"%endblock TBT.Contour.{name}"]
        return "\n".join(out + [""])
    return emit


def _electrode_layout(derived, frame, state_block, point, scratch_dir):
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
        _sc.Block("the k-point grid -- the shared grid in x, y; many k-points along z",
                  _k_mesh_block(derived["k_mesh"])),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(algorithm=derived.get("algorithm")),
        TS_ELECTRODE_SECTION,
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


def _seed_layout(derived, frame, state_block, point, scratch_dir):
    """The seed rung: an ordinary periodic SIESTA pass (§ 1, stage 1).

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
        _sc.Block("the k-point grid -- one k-point along z",
                  _k_mesh_block(derived["k_mesh"], advice=_TRANSVERSE_ADVICE)),
        _sc.Block("what the seed's solver must be, and must not",
                  _emit_solver_note),
        _sc.Block("the restart group", _emit_restart_group),
        _sl.SCF_SECTION,
        _sl.FREE_ENERGY_SECTION,
        _sl.SCF_TAIL_SECTION,
        _sl.spin_section(fixed=derived["spin_fixed"]),
        state_block,
        _sl.mpi_section(algorithm=derived.get("algorithm")),
        _sl.OUTPUT_SECTION,
    )


# ===================================================================== #
#  The blocks — structural text.                                        #
# ===================================================================== #

def _emit_seed_header(struct, cfg) -> str:
    label = cfg.system_label
    return "\n".join([
        "# ================================================================== #",
        f"#  Transport SEED .fdf — {label}",
        "#  An ordinary periodic SIESTA single point on the composed,",
        "#  SORTED junction (engines/transport.md 4.2, stage 1).  Its",
        f"#  converged {label}.DM starts the device NEGF SCF; scaffolding",
        "#  for convergence, no effect on the converged answer.",
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

    Written whole: no parameter models a coordinate table, which is what
    :class:`~molbuilder.script_emit.Block` is for.
    """
    from .transiesta import _emit_geometry

    # NO `emit_atom_metadata` CALL HERE.  The FRAMEWORK emits that fence
    # once, in the record section (`script_emit.py`, the only caller among
    # the engines): `deck_record.extract_atom_metadata` stops at the first END
    # marker, so a second copy here would win over the framework's.
    # `cfg` THROUGH, because the species order is a value a person may set
    # and this block writes `ChemicalSpeciesLabel` (`model/chemistry.md` § 3a).
    return "\n".join(_emit_geometry(struct, cfg=cfg, frame=frame))


def _emit_restart_group(struct, cfg) -> str:
    """``DM.UseSaveDM`` / ``MD.UseSaveXV`` — written in BOTH states.

    SIESTA reads `<SystemLabel>.DM` when the file is there whatever the deck
    omits (`siesta/input.py`, `_restart_group_lines`), so a deck that says
    nothing warm-starts from whatever the directory holds.

    The keys and the on/off come from the ONE declaration
    (:func:`molbuilder.siesta.input._restart_group_lines`).  The seed relaxes
    nothing, so it has no CG history to answer for.
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
        *_restart_group_lines(cfg, relax_type="none"),
        "",
    ])


def _emit_solver_note(struct, cfg) -> str:
    """Why the seed solves with ``diagon``.

    ``SCF_SECTION`` is the ENGINE's object, shared with every other kind, so
    its ``solution_method`` help is necessarily a generic three-option menu.
    The transport-specific half — *the seed must NOT be transiesta* — sits
    adjacent to the value, because that is
    where someone about to edit the value will read it, and it is a Block
    rather than a note on the section because mutating a shared section would
    put this text into every kind's deck.
    """
    return "\n".join([
        "# --- The seed's solver: ordinary diagonalisation, NOT transiesta ---",
        "#",
        "# This rung is a periodic WARM-UP, not the NEGF calculation, so its",
        "# `SolutionMethod` below is `diagon`, fixed by the rung: `transiesta`",
        "# here would attempt an open-boundary solve with no electrode",
        "# self-energies defined, which is not what this deck is and not",
        "# what the ladder needs from it.",
        "#",
        "# SIESTA writes <SystemLabel>.DM as this SCF converges, and that",
        "# file -- nothing else from this rung -- is what the device stage",
        "# reads as its starting density.  There is no MD block anywhere in",
        "# this deck, and that absence is what makes it a single point: the",
        "# geometry was relaxed upstream and moving it here would invalidate",
        "# the electrode partition the whole ladder is built on.",
        "#",
        *_pseudopotential_note(cfg),
        "",
    ])


def _pseudopotential_note(cfg) -> list:
    """Where this calculation's ``.psml`` files came from, by the citation's
    kind (`engines/transport.md` § 3.1): a cited pair's from the directory
    the template's ``psml_lib`` names, a cited run's with the run."""
    lib = getattr(cfg, "psml_lib", None)
    if lib:
        return [
            "# PSEUDOPOTENTIALS: `prep` copies this calculation's .psml files",
            f"# from {lib} -- the template's `psml_lib`, named at `jobset",
            "# init` for a cited structure pair, which carries none -- into",
            "# the calculation's `pseudos/` directory beside this deck.",
        ]
    return [
        "# PSEUDOPOTENTIALS: this calculation's .psml files travel with the",
        "# CITED run -- `prep` copies them into the calculation's `pseudos/`",
        "# directory beside this deck.  If SIESTA cannot find a pseudo, the",
        "# cited run did not carry it.",
    ]


#: What an SCF rung's mesh says beside its values -- ADVICE, not the rule: the
#: rule is the k-point mesh's (`engines/siesta.md` § 6.1), which writes one
#: point along transport on the seed and the device whatever the template's
#: third component says, because no rung reads it.
_TRANSVERSE_ADVICE = (
    "# THE COUNTS IN THE ELECTRODE PLANE (x, y) ARE YOURS; 1 x 1 IS RARELY",
    "# RIGHT.  (1, 1) is correct for a finite molecule between leads, and",
    "# for a wire or chain lead isolated in x and y, where nothing repeats",
    "# sideways.  For an electrode that repeats in the electrode plane --",
    "# an Au(111) surface cell -- set Nx, Ny to sample that repetition: a",
    "# metallic lead sampled 1 x 1 is badly under-converged, and the error",
    "# lands in the interface charge the device SCF then has to reproduce.",
    "# This pair is the shared k-point grid: the cited junction's own until",
    "# you change it in the template, and one pair for every SCF stage --",
    "# TranSIESTA stops on a lead sampled on other k-points.  Along z this",
    "# stage takes one k-point.  The transmission's TBT.k is its own grid,",
    f"# tbt_k_grid: {_TBT_FACTOR} times this pair unless you set it",
    "# (engines/transport.md 0.2).",
)


def _k_mesh_block(mesh, *, advice=()):
    """A block writing ``mesh`` -- the rung's k-point mesh, worked out once
    by :func:`transport_spec` (`kmesh.mesh_for`) -- with each deciding
    item's note, through the one writer (`siesta.layout.k_mesh_lines`) --
    the notes and the advice dropped when the deck is asked to be quiet, as
    every other block's are (:func:`_notes`)."""
    def render(struct, cfg) -> str:
        quiet = not getattr(cfg, "verbose_comments", True)
        return "\n".join([*_notes(cfg, advice),
                          *_sl.k_mesh_lines(mesh, notes=not quiet)])
    return render


# ===================================================================== #
#  The spec.                                                            #
# ===================================================================== #

def transport_spec(struct: Structure, cfg, *, state,
                   stage_token: Optional[str] = None,
                   point=None,
                   scratch_dir: Optional[str] = None,
                   contours=None) -> "_sc.DeckSpec":
    """The ``DeckSpec`` for one transport rung.

    ``point`` is the deck's point of a rung that runs per point
    (`transport.stages.Point`: a frame and a voltage), from which the
    device deck names the run's shared files the way the point reaches
    them (`Point.to_run`); ``None`` for a rung that runs once, or a deck
    rendered for no point in particular, whose way to the run is ``.``.
    ``scratch_dir`` is the folder of the run's own the leads' files live in
    -- the calculation's vocabulary's (`warm-files.toml` [transport]
    ``scratch_dir``, `engines/transport.md` § 2a.11), which prep hands down;
    ``None`` where it states none, the files then beside the run's.
    ``state`` is the calculation's electronic state, decided ONCE on the
    whole junction and handed to every rung
    (`science/chemistry-correctness.md` § 2a).  ``contours`` is a transmission
    point's sharing with a finished calculation's point
    (`energies.Contours`; `engines/transport.md` § 2a.16), ``None`` for a
    point that computes its window whole.

    *cfg* is a :class:`~molbuilder.config.siesta.SiestaConfig` — carrying the
    calculation's own template ⊕ this rung's overrides, as
    :func:`molbuilder.jobset.transport_rung.resolve_transport` resolves it (TR4) —
    because the sections above name catalogue rows and
    :func:`~molbuilder.script_emit.parameter` resolves a row by
    ``getattr(config, name)``.  That is not a preference: an item whose name
    is not a field on the config resolves to ``None`` *silently*, so a
    transport deck rendered from a config with different field names would
    quietly omit every catalogue item rather than fail.
    """
    rung = rung_of(stage_token)
    shape = SHAPE_OF_RUNG.get(rung)
    if shape is None:
        raise ValueError(
            f"{rung!r} is not a transport rung; the ladder is "
            f"{', '.join(SHAPE_OF_RUNG)} (engines/transport.md 4.2).  "
            f"`prep` names the rung and hands it down as `stage_token`.")

    # THE RUNG'S K-POINT MESH(ES), worked out once (`kmesh.mesh_for`,
    # `engines/siesta.md` § 6.1): read by the block that writes each, the
    # parallel split and the settings gate.  The transmission carries two --
    # the ladder's SCF mesh and its own `TBT.k`.
    k_mesh, tbt_mesh = rung_meshes(cfg, struct.axis_kind, rung)
    derived = _derived_for(state, cfg, k_mesh, tbt_mesh)
    from .transiesta import engine_frame_for
    frame = engine_frame_for(struct)
    layout = ({"seed": _seed_layout,
               "electrode": _electrode_layout,
               "device": _device_layout}[shape](
                   derived, frame, _state_block(state), point, scratch_dir)
              if shape != "transmission" else _transmission_layout(
                  derived, frame, _state_block(state), point, scratch_dir,
                  contours=contours))
    return _sc.DeckSpec(
        engine="siesta",
        engine_frame=frame,
        calculation="transport",
        layout=layout,
        line=_sl.line(derived),
        derived=derived,
        note_lead=_sl.note_lead,
        check_rules=_sl.check_rules,
        # WHAT THE SETTINGS GATE JUDGES besides the structure as it arrived:
        # the meshes this rung writes -- which the configuration alone cannot
        # say, since the rung decides its transport axis -- and which rung
        # this is, by its name, for a rule that holds on some rungs alone
        # (the transmission's window against its bias, § 2a.10; an item
        # judged only where it is read, `template.reads`).
        validate_subject=lambda s, c: (s, {"k_meshes": tuple(
            m for m in (derived["k_mesh"], derived["tbt_k_mesh"])
            if m is not None), "rung": rung}),
        created_by="molbuilder transport prep",
    )


def rung_meshes(cfg, axis_kind, rung: str):
    """``(SCF mesh, transmission mesh)`` the rung ``rung`` writes -- the
    ladder's SCF mesh on every rung and, on the transmission, its own
    ``TBT.k`` (``None`` elsewhere) -- through the one derivation,
    ``kmesh.mesh_for``.  The deck and the Transport tab's describe ask it, so
    what the tab judges is what prep writes (`engines/transport.md`
    § 0.3b)."""
    from .. import kmesh as _kmesh
    k_mesh = _kmesh.mesh_for(cfg, axis_kind, kind="transport", rung=rung)
    tbt_mesh = (_kmesh.mesh_for(cfg, axis_kind, kind="transport",
                                rung=rung, program="tbtrans")
                if SHAPE_OF_RUNG[rung] == "transmission" else None)
    return k_mesh, tbt_mesh


def ladder_mesh_findings(template_text: str, task, axis_kind) -> list:
    """The k-point mesh findings of every rung of ``task`` as the description
    resolves it (the template and each rung's own values,
    ``resolve.resolved_ladder``), on the junction's ``axis_kind`` -- judged by
    ``kmesh.check``, the one check prep's settings gate runs on each rung's
    deck.  A finding several rungs share (the one SCF mesh every SCF rung
    writes) is said once; the transmission's own grid's carries its stage,
    so it lands on that rung's tab (`engines/transport.md` § 0.3b).  An
    unresolvable description answers none: prep refuses it in its own
    words."""
    import dataclasses
    from ..config.siesta import SiestaConfig
    from ..kmesh import check
    from ..resolve import resolved_ladder
    try:
        ladder = resolved_ladder(template_text, task, SiestaConfig)
    except Exception:          # noqa: BLE001 -- prep's refusal, not ours
        return []
    out: list = []
    seen: set = set()
    for name, cfg in ladder:
        shape = SHAPE_OF_RUNG.get(name)
        if shape is None:
            continue
        meshes = [m for m in rung_meshes(cfg, axis_kind, name)
                  if m is not None]
        for issue in check(meshes, None):
            if (issue.where, issue.message) in seen:
                continue
            seen.add((issue.where, issue.message))
            if issue.where == "config.tbt_k_grid":
                issue = dataclasses.replace(issue, stage=name)
            out.append(issue)
    return out


def _derived_for(state, cfg, k_mesh, tbt_mesh=None) -> dict:
    """What this deck worked out — W10's one per-render context.

    DECLARED on the form rather than only closed over, so a reader outside
    this module can see where a value came from.  The three groups
    :func:`molbuilder.siesta.layout.line` needs are the same ones the
    optimization deck derives; they are computed by the engine's own helper so
    the two kinds cannot answer them differently.  The spin's are the
    calculation's one state -- for the transport kind the charge is 0 by rule
    (the leads set the electron number) and the spin is the rungs' shared
    answer.  The k-point meshes are the rung's (`kmesh.mesh_for`), and the
    parallel split counts its points on the SCF one.
    """
    from ..siesta.input import _parallel_facts, _spin_facts

    derived = {"k_mesh": k_mesh, "tbt_k_mesh": tbt_mesh}
    derived.update(_spin_facts(state))
    derived.update(_parallel_facts(cfg, k_mesh))
    derived.update(_contour_facts(cfg))
    return derived


def _contour_facts(cfg) -> dict:
    """The count the pole energy gives, said beside it (`engines/transport.md`
    § 6.1c).

    The deck states an ENERGY and TranSIESTA derives the count from it at the
    run's temperature, so a reader of the deck is told the count the engine
    will take -- by the engine's own rule, asked of its one home.  Nothing is
    said where the rule has no answer (no temperature, or one at or below
    zero), and the settings gate refuses that deck.
    """
    from .transiesta import pole_count
    energy = getattr(cfg, "negf_eq_pole_ev", None)
    temp = getattr(cfg, "electronic_temperature", None)
    if energy is None or temp is None or float(temp) <= 0:
        return {}
    return {"beside": {"negf_eq_pole_ev":
                       f"{pole_count(energy, temp)} poles at "
                       f"{float(temp):g} K"}}


def _state_block(state):
    """WHERE THE CHARGE AND SPIN CAME FROM, in every rung's deck
    (`science/chemistry-correctness.md` § 2a.5, ES2) -- each value beside its
    source, as the optimization deck writes them."""
    t, c, q = state.spin_treatment, state.unpaired_electrons, state.net_charge

    def emit(struct, cfg) -> str:
        lines = [f"# Spin: {t.value} ({t.said});",
                 f"#   unpaired electrons (2S): {c.value} ({c.said}).",
                 "#   Decided ONCE, on the whole junction, and the same "
                 "on every rung:",
                 "#   TranSIESTA joins the leads' self-energies to the "
                 "device, so all",
                 "#   five rungs solve the same spin channels."]
        lines.append(f"# NetCharge: not written -- {q.value:+d} ({q.said}).")
        return "\n".join(lines)

    return _sc.Block("where the charge and spin came from", emit)

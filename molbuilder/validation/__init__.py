"""Pre-emission validation for SIESTA / PySCF / Spectra / Transport.

The machinery lives in ``docs/science/validation.md``.  Every deck is
rendered by ``script_emit.render_deck``, which calls :func:`validate`
before a line is written -- a transport rung's too, through the SIESTA
validator; errors block emission, warnings print to stderr.

Design principles realised by this module:

  * **Principle #1** ("the dataclass is the lingua franca"):
    config-field validators (range / `validate=` callable) are read
    off ``dataclasses.field(metadata=...)`` -- no parallel lookup
    table, so adding a new field with a range is a one-line change
    to the dataclass.

  * **Principle #6** ("pre-emission geometry validation"): every
    structure-side check from the table is enforced here, before any
    SIESTA / PySCF text is emitted.

The output is a ``List[Issue]``.  Callers decide what to do with it;
:func:`report` is the "raise errors, print warnings to stderr"
helper ``script_emit.render_deck`` uses.

Package layout (see ``docs/science/validation.md`` § 10):

* :mod:`molbuilder.validation.geometry`  — engine-agnostic geometry
* :mod:`molbuilder.validation.metadata`  — dataclass-field driven
* :mod:`molbuilder.validation.chemistry` — the shared chemistry rules, the
  electronic state's one family among them
* :mod:`molbuilder.validation.sidecar`   — frozen-atoms / region INFO
* :mod:`molbuilder.validation.siesta`    — SIESTA-specific + aggregator
* :mod:`molbuilder.validation.pyscf`     — PySCF-specific + aggregator
"""

from __future__ import annotations

import math
import sys
from contextvars import ContextVar
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from ..issues import Issue, ValidationError
from ..structure import Structure

# Re-exported, so callers import these names from the package.
from .chemistry import (_check_metal_basis_adequacy,
                        check_electronic_state)
from .geometry import (_check_polymer_orientation,
                       _min_image_distance,
                       validate_geometry)
from .metadata import (_check_fixed_on_every_rung, _check_values,
                       _validate_config_metadata, not_carried)
from .pyscf import _validate_pyscf
from .sidecar import _check_frozen_atoms_consumed
from .siesta import (_check_siesta_charged_makov_payne_notice,
                     _check_siesta_mesh_cutoff,
                     _check_siesta_pseudo_coverage,
                     _validate_siesta)


# --------------------------------------------------------------------- #
#  Engine-validator registry                                            #
#                                                                        #
#  Type-keyed dispatch.  Each engine config class registers an          #
#  engine-specific validator; `validate()` looks it up by              #
#  isinstance().  Adding a new engine is a `_ENGINE_VALIDATORS[T] = fn` #
#  line, not a string-compare in `validate()`.                          #
# --------------------------------------------------------------------- #


_ENGINE_VALIDATORS: Dict[Type, Callable[[Structure, object, Optional[np.ndarray]], List[Issue]]] = {}


def _register_engine_validator(cfg_cls: Type):
    """Decorator: register a validator for a specific config class.

    The validator receives (struct, cfg, cell) and returns a list of
    Issues.  ``cell`` may be None for engines that don't have a
    periodic cell concept (PySCF gas-phase / PCM); each registered
    validator decides what to do with it.
    """
    def deco(fn):
        _ENGINE_VALIDATORS[cfg_cls] = fn
        return fn
    return deco


# --------------------------------------------------------------------- #
#  Top-level entry point                                                #
# --------------------------------------------------------------------- #


def _structure_declares_a_box(struct: Structure) -> bool:
    """True when this structure asks for a unit cell at all (F4).

    ``resolve_cell()`` will happily hand back a bare bounding box for a
    gas-phase molecule that never asked for one -- and for a PLANAR molecule
    (water, benzene) that box has zero thickness, so a determinant check on it
    reports a degenerate cell for a calculation that has no cell.  PySCF /
    Spectra are gas-phase: there is no lattice to check.

    A box is declared by an explicit ``cell``, a vacuum the structure STATES
    (any value, including all-zero), or a non-isolated axis kind (periodic /
    transport).  Anything else is a molecule in free space, and the
    cell-dependent checks have nothing to say about it."""
    if struct.cell is not None:
        return True
    if struct.vacuum is not None:
        # STATING one is the declaration -- including an all-zero triple, which
        # means "no gap, deliberately" and is a real (and unusable) box rather
        # than an absence.
        return True
    return any(k != "isolated" for k in (struct.axis_kind or ()))


def validate(struct: Structure, cfg, *,
             cell: Optional[np.ndarray] = None,
             dest_dir: "Optional[object]" = None,
             prior: "Optional[object]" = None,
             calculation: str = "optimization",
             design: "Optional[Structure]" = None,
             k_meshes=None,
             rung: Optional[str] = None) -> List[Issue]:
    """Run every applicable validation check and return the findings.

    Parameters
    ----------
    struct
        The Structure about to be emitted.
    cfg
        SiestaConfig or PySCFConfig (or any dataclass; the generic
        config-field metadata pass runs on anything dataclass-shaped).
    calculation
        The described KIND this configuration is for ("optimization"
        unless the description says otherwise).  The kind's own science
        joins the same pass through ``_KIND_VALIDATORS`` — a fact-keyed
        step of the pipeline, not a hook a caller may forget: the deck
        conductor reads the kind off the spec (a declared fact) and
        every calculation type flows through this one function.
    cell
        Optional pre-computed (3, 3) lattice the generator is going
        to use.  **Omit it and the gate resolves the structure's own
        cell** (``struct.resolve_cell()``) -- clause F4 of the
        delivery contract (science/validation.md 4.1): a check must
        never be silently skipped because a caller forgot an
        argument.  Pass it only to validate a cell that differs from
        the structure's (a generator overriding the box); an explicit
        ``cell`` still wins.
    dest_dir
        Optional destination directory hint -- the path the user is
        about to save the rendered .fdf into.  Used by the SIESTA
        validator to resolve dest-relative ``cfg.psml_lib`` paths
        (the portable form the web Save handler persists).  When
        None, dest-relative paths fall back to projects/-anchored
        resolution; the file-existence check may then misfire and is
        downgraded to a WARN.
    prior
        What an EARLIER STAGE of this calculation left that this stage's
        checks judge -- today, at a SIESTA force-constant stage, the ladder's
        `relax` stage's relaxation record (`engines/vibration.md` § 5.2a),
        handed in by the deck's spec from the `vibration` block `prep`
        built.  ``None`` everywhere else.

    design
        The structure AS THE PERSON HOLDS IT, when ``struct`` is the placed
        copy a deck writes (its validation subject: the coordinates plus the
        engine offset).  A fact recorded ABOUT the file -- the relaxation
        record's geometry fingerprint -- is judged against these, because a
        placement is a rigid shift the record never saw.  None means ``struct`` is the file.
    k_meshes
        The k-point meshes the deck writes (`kmesh.mesh_for`,
        `engines/siesta.md` § 6.1), when its spec built them: a transport
        rung's depend on the rung, which the configuration alone cannot say.
        None derives the one this configuration writes, through the same
        door.

    The returned list is in deterministic order: generic geometry
    checks first, generic config-field checks next, then engine-
    specific checks.  Callers can sort / filter as they please.

    This is THE single per-engine validation gate: every engine config
    (SiestaConfig, PySCFConfig) registers ONE validator, and a kind's
    science joins through ``_KIND_VALIDATORS``, so a caller runs ``validate(struct, cfg)`` once instead of hand-
    concatenating a separate engine ``preflight()`` (the cross-tab
    silent-skip class the backend-architecture review flagged; V1/V2).
    """
    issues: List[Issue] = []
    # WHETHER THE ENGINE COMPUTES IN THE BOX (`model/structure-periodicity.md`
    # § 2.1, plan § 5w K8): PySCF builds a molecule in free space, so the
    # box's advice about a calculation in it -- vacuum, images, faces -- is
    # not its.
    # What binds every engine it still hears (`cell.box_findings_for`).
    from ..cell import box_findings_for, computes_in_cell
    from ..template import engine_name
    engine = engine_name(type(cfg))
    in_cell = computes_in_cell(engine)
    # F4: resolve the structure's own box when the caller passed none, so the
    # cell-dependent checks cannot go silent on an omitted argument.  A
    # structure that cannot resolve a cell (a periodic axis with no explicit
    # lattice, an empty structure) says so as `info` rather than skipping in
    # silence -- the contract forbids a check disappearing without a trace.
    if cell is None and _structure_declares_a_box(struct):
        try:
            cell = struct.resolve_cell()
        except Exception as exc:              # noqa: BLE001 -- reported, not raised
            cell = None
            issues.append(Issue(
                "info",
                f"cell-dependent checks (volume, determinant, image distance) "
                f"were skipped: this structure has no resolvable cell ({exc})",
                "cell.unresolved"))

    # THE STRUCTURAL CELL FACTS, from the one checker.  As Issues they reach
    # the preflight panel and the CLI alike, and `report()` turns the
    # error-severity ones into the refusal the emit doors already promise.
    #
    # Only when the structure DECLARES a box, for the same reason the block
    # above is gated: `resolve_cell` hands back a bounding box for a gas-phase
    # molecule that never asked for one, and judging that box would invent
    # findings about a cell the calculation does not have.
    # ``cell`` here is the box that will actually be EMITTED -- a generator may
    # have chosen one that differs from the structure's, and judging the
    # structure's instead would answer about a box nobody runs.
    if cell is not None or _structure_declares_a_box(struct):
        from ..cell import check as _check_cell, resolve as _resolve_cell
        issues += box_findings_for(
            engine, _check_cell(_resolve_cell(struct, box=cell)))

    # ...and the geometry's own measurements of the box (images, volume) are
    # a calculation in a cell's too.
    if not in_cell:
        cell = None
    issues += validate_geometry(struct, cell)
    # WHAT MAY STAND FOR EACH ITEM ON THIS KIND, asked first so a refused
    # value's range warning can stand aside (`engines/template.md` § 5.3):
    # a component the kind fixes, a choice it does not offer -- a relaxer a
    # vibration cannot relax with -- a value past a hard limit -- a
    # displacement SIESTA divides by.  Reported after the metadata's own, in
    # the order this function has always emitted.
    refusals = _check_values(cfg, calculation)
    issues += _validate_config_metadata(
        cfg, refused={i.where.split(".", 1)[1] for i in refusals},
        foreign=not_carried(cfg, calculation))
    # WHAT EVERY RUNG FIXES ALIKE (`engines/template.md` § 6.4): a config
    # that skipped `resolve` holds whatever its caller put there.
    issues += _check_fixed_on_every_rung(cfg, calculation)
    issues += refusals
    # THE ELECTRONIC STATE, once, for every engine and every kind
    # (`science/chemistry-correctness.md` § 2a): the state the deck will be
    # written from, judged by one family of findings.
    issues += check_electronic_state(struct, cfg, calculation=calculation)

    # Engine-specific dispatch via the registry.  isinstance() picks
    # up subclasses too, so a future engine config that subclasses
    # an existing one inherits its validator unless it registers its
    # own.  Extra kwargs (dest_dir / prior) are forwarded only when
    # set; every registered validator accepts **_ and ignores the ones
    # it doesn't use.
    engine_kw = {}
    if dest_dir is not None:
        engine_kw["dest_dir"] = dest_dir
    if prior is not None:
        engine_kw["prior"] = prior
    if design is not None:
        engine_kw["design"] = design
    # THE K-POINT MESH(ES) THE DECK WRITES (`engines/siesta.md` § 6.1), when
    # the spec built them -- a transport rung's are its rung's, which the
    # configuration alone cannot say.  Absent, the SIESTA validator derives
    # the one this configuration writes, through the same door.
    if k_meshes is not None:
        engine_kw["k_meshes"] = tuple(k_meshes)
    # THE RUNG this deck is, when the rung's spec says (`transport/deck.py`'s
    # `validate_subject`, as the k-meshes arrive): a kind's rule that holds
    # on one rung alone -- the transmission's window -- is keyed on it.
    if rung is not None:
        engine_kw["rung"] = str(rung)
    # ...AND WHAT THE ONE PER-VALUE DOOR REFUSED, so a check on a value
    # built from a refused one stands aside: a value refused draws that
    # refusal alone (`engines/template.md` § 5.3).
    engine_kw["refused"] = frozenset(i.where.split(".", 1)[1]
                                     for i in refusals)
    # The KIND rides along so an engine validator can defer a family the
    # kind's own science owns (the double-fire dedup, ruled 2026-08-21:
    # one fact, one finding -- on a vibration deck the grid and frozen-atom
    # verdicts are the kind's).  The
    # charge and spin are neither's: they are the electronic state's one
    # family, asked once below for every engine and kind.  Validators that
    # do not branch on it ignore it through **_.
    engine_kw["calculation"] = calculation
    for cfg_cls, fn in _ENGINE_VALIDATORS.items():
        if isinstance(cfg, cfg_cls):
            issues += fn(struct, cfg, cell, **engine_kw)
            break

    # The CALCULATION KIND's own science — the same step, keyed by the
    # described fact.  A kind registered here cannot be forgotten, because
    # every deck route ends in this function.
    kind_fn = _KIND_VALIDATORS.get(calculation)
    if kind_fn is not None:
        issues += kind_fn(struct, cfg, cell, **engine_kw)
    return issues


#: Where :func:`report` writes when it is given no stream: this context's
#: own, set by a caller for the span of one act -- `prep` shows a sweep's
#: repeated warnings once -- and ``sys.stderr`` when none is.  A context
#: variable, never ``sys.stderr`` swapped: the server preps on several
#: threads at once, and a swap one thread puts back is another's (W55 D8).
REPORT_STREAM: "ContextVar" = ContextVar("report_stream", default=None)


def report(issues: List[Issue], *,
           raise_on_error: bool = True,
           stream=None) -> None:
    """Print warnings and advisories to stderr; raise ValidationError on errors.

    The two-pass shape (findings first, then maybe-raise) lets the
    user see *all* of them even when an error is also present --
    helpful when triaging a misconfigured run.

    **`info` is printed too**: `science/validation.md` R4 is explicit: *"no
    surface downgrades a severity to keep a screen quiet, and the CLI prints
    the same three."*  Errors still raise rather than print, which is R4's own
    distinction and not a downgrade.
    """
    if stream is None:
        stream = REPORT_STREAM.get() or sys.stderr
    for i in issues:
        if i.severity in ("warn", "info"):
            tag = f" [{i.where}]" if i.where else ""
            print(f"{i.severity}{tag}: {i.message}", file=stream)
    errors = [i for i in issues if i.severity == "error"]
    if errors and raise_on_error:
        raise ValidationError(issues)


# --- The calculation kinds' validators ---------------------------------- #
# A kind's science runs beside the engine's, keyed by what the description
# says it is, so every deck of that kind is checked on every route.  What an
# earlier stage left -- a SIESTA force-constant stage's `relax` record --
# arrives as ``validate``'s ``prior`` and is forwarded here (V1.36).
#: The calculation KIND's science, keyed by the described fact
#: (``task.calculation``).  "optimization" deliberately has no entry:
#: its science IS the engine validators above.  A new kind registers
#: here and its checks run for every deck of that kind, on every
#: route, with nothing for a deck author to remember.
_KIND_VALIDATORS: dict = {}


def _validate_vibration_kind(struct: Structure, cfg, cell, *,
                             prior=None, design=None, **_) -> List[Issue]:
    """The vibration kind's science (grid / amplitude / frozen atoms /
    the relaxation record), over the deck's own config view -- the charge
    and spin are the electronic state's one family, asked by ``validate``."""
    from ..config.pyscf import PySCFConfig
    if isinstance(cfg, PySCFConfig):
        from ..pyscf.vibration_deck import science_view
        from .spectra import spectra_render_checks
        return list(spectra_render_checks(struct, science_view(cfg, struct)))
    from ..config.siesta import SiestaConfig
    if isinstance(cfg, SiestaConfig):
        from .spectra import siesta_vibration_checks
        return list(siesta_vibration_checks(struct, cfg, design=design,
                                            relaxed_by=prior))
    # The kind's science is written per engine, and an engine this dispatch
    # does not name has NO science here -- a gap to refuse, never an empty
    # verdict: an empty list reads as "checked, nothing found" on every
    # surface, which is the silent skip science/validation.md F4 forbids.
    raise TypeError(
        f"the vibration kind has no science for {type(cfg).__name__}: "
        f"its checks are written for PySCFConfig and SiestaConfig, and a "
        f"config class with no checks cannot pass this gate as validated "
        f"(science/validation.md F4)")


def _validate_transport_kind(struct: Structure, cfg, cell, *,
                             prior=None, rung=None, **_) -> List[Issue]:
    """The transport KIND's science — keyed on ``task.calculation``, so it
    fires whatever config class the deck renders from.  A rule that only
    runs for one of two config classes is not a gate; this one runs for the
    kind.

    Its science: the bias advisory, the pole energy against the temperature,
    the vacuum where the crystal continues (I12).  **The k-point sampling is
    not here** -- the transport axis's one point, a lead's own count, the
    transmission's grid are the k-point mesh's (`kmesh.py`,
    `engines/siesta.md` § 6.1), refused on every door through
    ``template.why_not`` rather than on this one alone.
    """
    out: List[Issue] = []
    # THE BIAS ADVISORY.  Bias is the one axis a transport calculation
    # exists to sweep, so without it a person could describe 3 V and be told
    # nothing on the road that runs.
    bias = getattr(cfg, "bias_voltage_v", None)
    if bias is not None and abs(float(bias)) > 2.0:
        out.append(Issue(
            "warn",
            f"bias |V| = {abs(float(bias)):.2f} V is above the ~2 V "
            f"linear-response limit for typical molecular junctions.  "
            f"TranSIESTA will still converge, but read the result as a "
            f"single point on a NONLINEAR I-V curve, not as a linearized "
            f"Landauer conductance (di Ventra, Electrical Transport in "
            f"Nanoscale Systems, 2008; Reed et al. 2006).",
            where="config.bias_voltage_v"))
    # THE POLE ENERGY AND THE TEMPERATURE ARE ONE QUESTION, so neither can be
    # checked alone.  The rule is TranSIESTA's own -- SIESTA 5.4.2
    # `Src/m_ts_chem_pot.F90`, read 2026-09-16 rather than inferred -- and it
    # has one home, `transiesta.pole_count`, which the device deck also asks
    # for the count it states beside the energy.  TranSIESTA stops a run under
    # twenty poles, after the queue wait; this says so before it.
    #
    # 0 IS REFUSED TOO (`engines/transport.md` § 6.1c): TranSIESTA's own
    # choice -- about 42 poles at 300 K -- lost the charge on a real Au-BDT-Au
    # device, where 10 eV (123 poles) held it.  So the energy is
    # always written; and an energy at or below zero is not taken as one --
    # TranSIESTA keeps its 8-pole count and stops (`pole_count` says why).
    #
    # AND THE TEMPERATURE HAS ITS OWN FLOOR: TranSIESTA stops below 10 K
    # before any pole is counted (`transiesta.MIN_TS_TEMPERATURE_K`).
    pole = getattr(cfg, "negf_eq_pole_ev", None)
    temp = getattr(cfg, "electronic_temperature", None)
    if pole is not None:
        from ..transport.transiesta import (MIN_EQ_POLES,
                                            MIN_TS_TEMPERATURE_K, pole_count,
                                            pole_energy_for)
        if temp is None or float(temp) < MIN_TS_TEMPERATURE_K:
            out.append(Issue(
                "error",
                f"the electronic temperature is "
                f"{0.0 if temp is None else float(temp):g} K, and TranSIESTA "
                f"stops a device run below {MIN_TS_TEMPERATURE_K:g} K before "
                f"it starts -- \"TranSiesta electronic temperature *must* be "
                f"larger than 10 kT\".  Set it to at least "
                f"{MIN_TS_TEMPERATURE_K:g} K; 300 K is the default "
                f"(engines/transport.md 6.1c).",
                where="config.electronic_temperature"))
        else:
            n = pole_count(pole, temp)
            if n < MIN_EQ_POLES:
                need = pole_energy_for(MIN_EQ_POLES, temp)
                said = (f"{float(pole):g} eV is not an energy TranSIESTA "
                        f"takes -- it keeps its own count of {n} poles"
                        if float(pole) <= 0 else
                        f"{float(pole):g} eV gives {n} poles at "
                        f"{float(temp):g} K")
                out.append(Issue(
                    "error",
                    f"the equilibrium contour's pole energy: {said}, and "
                    f"TranSIESTA needs at least {MIN_EQ_POLES}.  It stops "
                    f"with \"the continued fraction method requires at least "
                    f"20 poles\", after the queue wait.  The count is "
                    f"DERIVED, N = int(E / (pi kT)), so it moves with the "
                    f"temperature: at {float(temp):g} K the least energy is "
                    f"{need:.2f} eV.  The default is 10 eV "
                    f"(engines/transport.md 6.1c).",
                    where="config.negf_eq_pole_ev"))
    # THE TRANSMISSION WINDOW REACHES THE BIAS WINDOW (P2, `engines/
    # transport.md` § 2a.10, § 6.1c): on the transmission rung alone, whose
    # deck writes the window and the voltage.  One rule for the record, this
    # gate and the description's preflight (`record.window_short_of`):
    # TBtrans integrates the current over its window and says nothing when
    # the window cuts it short.
    if rung == "transmission" and bias is not None and temp is not None:
        emin = getattr(cfg, "transmission_emin_ev", None)
        emax = getattr(cfg, "transmission_emax_ev", None)
        if emin is not None and emax is not None:
            from ..constants import BOLTZMANN_EV_K
            from ..transport.record import window_short_of
            short = window_short_of(emin, emax, bias,
                                    BOLTZMANN_EV_K * float(temp))
            if short:
                out.append(Issue("error",
                                 short + " (engines/transport.md 2a.10)",
                                 where="config.transmission_emax_ev"))
    # THE K-POINT SAMPLING IS NOT HERE: it is the k-point mesh's
    # (`kmesh.py`, `engines/siesta.md` § 6.1): the third
    # component of `kgrid` and `tbt_k_grid` is one a transport calculation
    # fixes, and a lead's count is refused at 1 by its own limit and warned
    # below 20 by its range -- every door, through `template.why_not`.

    # ---- I12: no vacuum where the crystal continues (§ 5, § 6.1c) ----
    #
    # MEASURED FROM THE LEAD (TD3, user 2026-09-29): each rule reads the
    # lead's own spacing, and both are refusals -- a gap along transport
    # severs the lead, and vacuum on a periodic axis contradicts the
    # declaration.  An ISOLATED transverse axis is left alone: a wire or chain
    # lead is vacuum-surrounded across transport, and that vacuum is the one
    # the structure states.  The threshold is the seam rule's, one rule for
    # "is this boundary vacuum" (`cell.SEAM_VACUUM_FACTOR`).
    #
    # This is the reverse of the advice an isolated molecule gets, which is
    # why it is keyed on the calculation kind and not on the cell alone --
    # `cell.vacuum_thin` is right for a molecule and is gated on `axis_kind`.
    _cell = cell if cell is not None else getattr(struct, "cell", None)
    if _cell is not None and getattr(struct, "n_atoms", 0):
        from ..cell import SEAM_VACUUM_FACTOR, transverse_reach
        from ..transport.sort import ELECTRODE_LABELS
        from .sidecar import check_junction_boundary
        # THE ROOM ALONG TRANSPORT, one rule both ways (vacuum, collision)
        # -- a refusal on every transport rung; the same rule warns at a
        # labelled junction's relaxation (`sidecar.check_junction_boundary`).
        out += check_junction_boundary(struct, _cell, severity="error",
                                       whole_structure_is_the_lead=True)
        # THE LEAD: the electrode-labelled atoms of a junction, one list per
        # lead; an electrode rung's structure IS the lead and carries no
        # labels, so all of it.
        leads = [list(idx) for label, idx in
                 (getattr(struct, "regions", None) or {}).items()
                 if label in ELECTRODE_LABELS and idx]
        if not leads:
            leads = [list(range(struct.n_atoms))]
        # LEAD BY LEAD, as the rule is stated: pooled, two leads answer for
        # each other -- one that tiles hides one that does not on the
        # junction's rungs, while the electrode rung that sees it alone
        # refuses the same calculation; and two one-atom leads measure the
        # distance between the leads as a "bond".
        kinds = getattr(struct, "axis_kind", None) or ("isolated",) * 3
        named = ([label for label, idx in
                  (getattr(struct, "regions", None) or {}).items()
                  if label in ELECTRODE_LABELS and idx]
                 or ["the lead"])
        for ax, name in ((0, "a"), (1, "b")):
            if kinds[ax] != "periodic":
                continue
            for label, lead in zip(named, leads):
                reach, bond = transverse_reach(struct.positions, _cell, ax,
                                               lead)
                if bond is None:
                    out.append(Issue(
                        "warn",
                        f"axis {name} is declared periodic, but {label} is "
                        f"one atom, so whether it reaches across that "
                        f"boundary cannot be measured (engines/transport.md "
                        f"6.1c).",
                        where="cell.transverse_vacuum"))
                elif reach > SEAM_VACUUM_FACTOR * bond:
                    out.append(Issue(
                        "error",
                        f"axis {name} is declared periodic, but across its "
                        f"boundary the nearest atom of {label} is "
                        f"{reach:.2f} Å away -- {reach / bond:.1f} of its own "
                        f"{bond:.2f} Å bond, so the crystal does not continue "
                        f"there: that is vacuum.  A wire or chain lead is "
                        f"isolated across the transport axis: declare the "
                        f"axis isolated on the structure (the Cell page), "
                        f"relax it, and cite that relaxation -- its deck "
                        f"records the axis kinds (engines/transport.md "
                        f"6.1c).",
                        where="cell.transverse_vacuum"))
    return out


def _register_default_engines() -> None:
    """Register each engine's validator and each kind's -- AT IMPORT, called
    just below.  The engine config classes come from `config/`, where they
    are defined, and are imported in here only to keep the registrations
    together: nothing in `siesta/`, `pyscf/` or `config/` imports
    `validation`, so there is no cycle to avoid.

    **An import that fails here RAISES.**  A missing engine config is a
    broken install, and swallowing it would leave that engine's scientific
    validation silently absent -- every deck of it passing a gate that
    checked nothing.
    Only the config CLASS is imported eagerly (as a registry key)."""
    from ..config.siesta import SiestaConfig
    from ..config.pyscf import PySCFConfig
    _ENGINE_VALIDATORS[SiestaConfig] = _validate_siesta
    _ENGINE_VALIDATORS[PySCFConfig] = _validate_pyscf
    # A vibration's and a transport calculation's science is the KIND's,
    # keyed by `task.calculation` below; every transport rung resolves a
    # `SiestaConfig` (`engines/transport.md` 2a.14).
    _KIND_VALIDATORS["vibration"] = _validate_vibration_kind
    _KIND_VALIDATORS["transport"] = _validate_transport_kind


_register_default_engines()


__all__ = ["validate", "report"]

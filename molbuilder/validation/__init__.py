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

import sys
from contextvars import ContextVar
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from ..issues import Issue, ValidationError
from ..structure import Structure

# The checks this entry point runs.
from .chemistry import check_electronic_state
from .geometry import validate_geometry
from .metadata import (_check_fixed_on_every_rung, _check_inactive, _inactive,
                       _check_values, _validate_config_metadata,
                       not_carried)
from .pyscf import _validate_pyscf
from .siesta import _validate_siesta


# --------------------------------------------------------------------- #
#  Engine-validator registry                                            #
#                                                                        #
#  Type-keyed dispatch.  Each engine config class registers an          #
#  engine-specific validator; `validate()` looks it up by              #
#  isinstance().  Adding a new engine is a `_ENGINE_VALIDATORS[T] = fn` #
#  line, not a string-compare in `validate()`.                          #
# --------------------------------------------------------------------- #


_ENGINE_VALIDATORS: Dict[Type, Callable[[Structure, object, Optional[np.ndarray]], List[Issue]]] = {}


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


def structure_findings(struct: Structure, *,
                       cell: Optional[np.ndarray] = None,
                       engine: Optional[str] = None) -> List[Issue]:
    """THE STRUCTURE'S OWN FINDINGS (:func:`_the_structure`'s) -- what
    `molbuilder validate` says with no engine named, and :func:`validate`
    first."""
    return _the_structure(struct, cell=cell, engine=engine)[1]


def _the_structure(struct: Structure, *, cell: Optional[np.ndarray],
                   engine: Optional[str]
                   ) -> "Tuple[Optional[np.ndarray], List[Issue]]":
    """``(box, findings)`` -- THE STRUCTURE'S OWN FINDINGS -- its box's facts (`cell.check`) and its
    geometry's (:func:`validate_geometry`) -- the structural half of
    :func:`validate`, which asks it first.  ``engine`` names who will
    compute on it, and so which of the box's findings it hears
    (`cell.box_findings_for`); ``None`` -- `molbuilder validate` with no
    engine -- is whatever will be run on it, every fact of the box: a box
    that is no box (`cell.no_volume`, `cell.left_handed`) is an error on
    every road (`science/validation.md` § 4.1, *representability is
    blocking*).  ``cell`` as :func:`validate` takes it (F4: omitted, the
    structure's own, when it declares one); ``box`` is the one every
    engine's and kind's check then reads -- resolved, and ``None`` for an
    engine that computes in no box."""
    issues: List[Issue] = []
    # WHETHER THE ENGINE COMPUTES IN THE BOX (`model/structure-periodicity.md`
    # § 2.1, plan § 5w K8): PySCF builds a molecule in free space, so the
    # box's advice about a calculation in it -- vacuum, images, faces -- is
    # not its.
    # What binds every engine it still hears (`cell.box_findings_for`).
    from ..cell import box_findings_for, computes_in_cell
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
    return cell, issues


def validate(struct: Structure, cfg, *,
             cell: Optional[np.ndarray] = None,
             dest_dir: "Optional[object]" = None,
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
        joins the same pass -- the part its record in the kind table names
        (``validator``), a step of the pipeline, not a hook a caller may
        forget: the deck
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
    science joins through its record's ``validator``, so a caller runs
    ``validate(struct, cfg)`` once instead of hand-
    concatenating a separate engine ``preflight()`` (the cross-tab
    silent-skip class the backend-architecture review flagged; V1/V2).
    """
    from ..template import engine_name
    engine = engine_name(type(cfg))
    # THE BOX, worked out once with the structure's findings, and the one
    # every check below reads.
    cell, issues = _the_structure(struct, cell=cell, engine=engine)
    # WHAT MAY STAND FOR EACH ITEM ON THIS KIND, asked first so a refused
    # value's range warning can stand aside (`engines/template.md` § 5.3):
    # a component the kind fixes, a choice it does not offer -- a relaxer a
    # vibration cannot relax with -- a value past a hard limit -- a
    # displacement SIESTA divides by.  Reported after the metadata's own, in
    # the order this function has always emitted.
    refusals = _check_values(cfg, calculation)
    # ...AND WHAT ENTERS NOTHING HERE -- a greyed-out field's (`engines/
    # template.md` § 5, `active_when`): said once below, and no advice is
    # given about it -- no recommended range, no probe's displacement.
    inactive = _inactive(cfg, calculation)
    issues += _validate_config_metadata(
        cfg, refused={i.where.split(".", 1)[1] for i in refusals},
        foreign=not_carried(cfg, calculation), inactive=frozenset(inactive))
    # WHAT EVERY RUNG FIXES ALIKE (`engines/template.md` § 6.4): a config
    # that skipped `resolve` holds whatever its caller put there.
    issues += _check_fixed_on_every_rung(cfg, calculation)
    issues += refusals
    # A VALUE THAT ENTERS NOTHING HERE: said once.
    issues += _check_inactive(cfg, calculation, inactive)
    # THE ELECTRONIC STATE, once, for every engine and every kind
    # (`science/chemistry-correctness.md` § 2a): the state the deck will be
    # written from, judged by one family of findings.
    issues += check_electronic_state(struct, cfg, calculation=calculation)

    # Engine-specific dispatch via the registry.  isinstance() picks
    # up subclasses too, so a future engine config that subclasses
    # an existing one inherits its validator unless it registers its
    # own.  Extra kwargs (dest_dir / design) are forwarded only when
    # set; every registered validator accepts **_ and ignores the ones
    # it doesn't use.
    engine_kw = {}
    if dest_dir is not None:
        engine_kw["dest_dir"] = dest_dir
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
    engine_kw["inactive"] = frozenset(inactive)
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

    # The CALCULATION KIND's own science — the same step, the part its
    # record in the kind table names (``validator``; an optimization's
    # science IS the engine validators above).  A kind that declares one
    # cannot have it forgotten, because every deck route ends in this
    # function.
    from ..kind_table import kind as _kind, part
    ref = _kind(calculation).validator
    if ref is not None:
        kind_fn = part(ref, record=f"kind {calculation}", key="validator")
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


_register_default_engines()


__all__ = ["validate", "report", "structure_findings"]

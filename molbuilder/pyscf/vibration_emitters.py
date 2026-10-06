"""The vibration deck's emission library -- PySCF block emitters.

MOVED 2026-08-21 (spectra-migration plan P3) from
``spectra/pyscf_script.py``, where these emitters were the old
standalone generator's body; the P1 lift composed them into
:func:`molbuilder.pyscf.vibration_deck.vibration_spec` call-for-call,
and P3 retired the old generator (``render_spectra_script``) around
them.  The lift is a move, not a rewrite -- the science below is the
2026-05 validated code.

Each ``_emit_*`` function returns lines of the runnable deck.  The
deck writes ``<job>.spectra.json`` incrementally, atomically replacing
the file at each phase boundary so a live-watch poller never sees a
torn document (spec § 6.1).  The wire format is the engine-agnostic
:class:`SpectraResults` shape (spec § 5 / § 6).

This module is the only place where the actual scientific choices
land in code form:

  * ONE harmonic analysis for free and held atoms alike: the free-free
    block of the true Hessian, mass-weighted, diagonalised in the
    complement of the whole-body motions that survive holding the frozen
    set -- `spectra.normal_modes`, which the deck imports from
    ``mb_pyscf.pyz`` (science/normal-modes.md R1-R4);
  * eigenvalue → cm⁻¹ through the one constant both routes share
    (`spectra.normal_modes.frequencies_cm1`, the eigenvalues in
    Eh/(Bohr²·amu));
  * Raman activity via finite-difference dα/dR_k (k over
    free Cartesians), projected onto the mass-weighted mode
    eigenvectors → Raman activity scalar via the standard
    45·α'² + 7·γ'² formula [Wilson1955, Komornicki1979];
  * per-mode displaced SCFs at q ± A·Q_n in real-space along the
    mode eigenvector for the modes selected by
    :func:`spectra.selection.select_modes`.

The emitted deck's header docstring is the Methods paragraph from
:func:`spectra.methods.render_methods_md` (the engine-specific
fragment is :func:`pyscf_methods_fragment` below) plus a bibliography
listing -- so a user reading the file can distil a Methods section
verbatim (spec § 11.2).

Every molbuilder function the deck runs is imported from ``mb_pyscf.pyz``
beside it -- ``engines/pyscf.md`` § 3 lists them, and what stays written as
text: the run itself, its values, its dressers and set-up, its per-run
helpers and the branchless IR and Raman formulas.

PROVENANCE -- WHERE EVERY NUMBER COMES FROM.  This module writes a script whose
output is a quantitative result, so the chain from PySCF's own objects to each
sidecar key is stated in `engines/vibration.md` § 6.4 and must be kept true here.  The
distinction that matters:

  * PASSED THROUGH -- `scf_energy_eh` is `mf.kernel()`'s return,
    `mo_energies_eh` is `mf.mo_energy`.  These are PySCF's numbers.  What is
    ours is that they reach the right key in the right unit, unrounded.
    `frequency_cm1` is ours: the eigenvalues of molbuilder's one harmonic
    path (`spectra.normal_modes`), converted by its one constant.
  * DERIVED -- `homo_idx` (from `mf.mo_occ`), `ir_intensity_km_mol` (from
    `mf.dip_moment` at displaced geometries) and `raman_activity_a4_amu` (from
    the polarizability, in Bohr^3, converted once).  **These are ours to get
    wrong**, and they are what this module's tests can meaningfully guard.

A DERIVED rule that has a BRANCH lives as a callable function, which the
script imports (see `spectra.pyscf_vibration.homo_index`), so one
implementation runs and is tested.  A branchless one may stay inline; if it
grows a branch, it moves.  That rule is stated once in
`siesta/makov_payne.py` and applies here too.
"""
from __future__ import annotations

from typing import List, Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:            # annotation only -- importing
    # vibration_deck at run time would cycle (it imports this).
    from .vibration_deck import VibrationConfigView
from ..runfiles import tail as _rf_tail
from ..structure import Structure

#: The engine display string for deck headers -- was
#: ``PySCFSpectraEngine.label`` until P3 retired that class.
PYSCF_ENGINE_LABEL = "PySCF (analytic Hessian + dα/dR)"

#: THE LINE THIS DECK PRINTS WHEN IT REACHES ITS OWN END -- the spectrum
#: deck's own, declared in `end_lines` beside the relaxation deck's.
from .end_lines import SPECTRUM_END_MARKER   # noqa: E402

from ..spectra.results import SCHEMA_VERSION




# Default finite-difference step for Raman dα/dR.
#
# Central-difference truncation error scales as (δ²·α'''(R) / 6) where
# α'''(R) is the third derivative of the polarizability with respect
# to nuclear position.  For typical molecular polarizabilities α(R)
# is smooth on the Bohr scale (α''' ~ 10⁻³ a.u.); at δ = 0.005 Å this
# gives a relative error in dα/dR of roughly 1e-6, which is well below
# the SCF noise floor at conv_tol=1e-9.  The 0.005 Å choice matches
# the FD step Gaussian and ORCA use for their static polarizability-
# derivative paths.  Not exposed as a config knob because (a) the
# defensible range is narrow (~0.001 Å to ~0.01 Å), (b) tuning it
# without changing the SCF convergence tolerance accomplishes
# nothing, and (c) it has no scientific interpretation -- it's a
# numerical-stability knob, not a physical one.
_RAMAN_FD_STEP_ANG = 0.005


# --------------------------------------------------------------------- #
# Header docstring + Methods paragraph                                  #
# --------------------------------------------------------------------- #


def _emit_header_docstring(struct: Structure,
                           cfg: "VibrationConfigView",
                           *,
                           methods_md: str,
                           stage_token: Optional[str] = None) -> List[str]:
    """Triple-quoted Methods + Outputs + Dependencies block.

    The Methods paragraph is the same prose the Results panel
    surfaces in its "Methods text" block, so a reader sees identical
    content in the file, the viewer and the JSON.  ONE DIFFERENCE, and
    it is deliberate: the viewer's copy gains a sentence naming which
    dmu/dR route the run took (`spectra.methods.with_ir_route`), which
    cannot be known here because the deck is written before it runs.
    ``methods_md`` is computed once in the deck composer
    (:func:`molbuilder.pyscf.vibration_deck.vibration_spec`)
    and threaded through here + ``_emit_constants``.
    """
    out: List[str] = []
    out.append('"""PySCF Spectra input script generated by molbuilder.')
    out.append("")
    out.append(f"System    : {getattr(struct, 'title', None) or 'untitled'}")
    out.append(f"Engine    : {PYSCF_ENGINE_LABEL}")
    # The level of theory as run: a Hartree-Fock run names no functional,
    # since it has none, whatever that item holds; the dispersion
    # correction applies to either method (vibration.md § 4.10).
    if cfg.is_dft:
        out.append(f"Method    : {cfg.scf_class} / {cfg.functional} / {cfg.basis}")
    else:
        out.append(f"Method    : {cfg.scf_class} (Hartree-Fock) / {cfg.basis}")
    if cfg.dispersion != "none":
        out.append(f"Dispersion: {cfg.dispersion}")
    out.append(f"Atoms     : {getattr(struct, 'n_atoms', len(struct.elements))}")
    out.append(f"Job name  : {cfg.job_name}")
    out.append("")
    # The stage by its NAME, through the one printer (`job-system.md` § 5.3).
    from ..identity import LAUNCH_MODE_NOTE, deck_launch
    _launch = deck_launch(stage_token)
    if _launch:
        out.append("Run it the way it was prepared (the described route), "
                   "from the calculation's folder:")
        out.append(f"    {_launch}")
        out.append(f"    {LAUNCH_MODE_NOTE}")
        out.append("    -- the wrapper beside this deck activates the env and "
                   "logs the run.")
    out.append("A bare `python <this file>` also works from this directory for a")
    out.append("manual run -- with mb_pyscf.pyz beside it, the molbuilder code it")
    out.append("imports -- but nothing records it.")
    out.append("Layout: one job per directory (docs/execution/job-contracts.md);")
    out.append("this deck was written into its own by `prep`.")
    out.append("")
    # THE OUTPUTS, the one list both decks write (`input.emit_outputs_block`).
    from .input import emit_outputs_block
    _relaxes = not bool(getattr(cfg, "already_relaxed", False))
    out += emit_outputs_block(
        cfg.job_name, stage_token or None, spectra=True,
        log=bool(getattr(cfg, "log_file", False)),
        chk=bool(getattr(cfg, "chkfile", False)),
        initial=bool(getattr(cfg, "save_initial_xyz", False)),
        optimized=_relaxes and bool(getattr(cfg, "save_optimized_xyz", False)),
        trajectory=_relaxes and bool(getattr(cfg, "write_trajectory", False)),
        progress_log=bool(getattr(cfg, "write_molwatch_log", False)),
        held=_relaxes and bool(getattr(cfg, "frozen_indices", None)))
    out.append("")
    if cfg.compute_ir:
        out.append("*** IR INTENSITIES -- BAND-LEVEL VALIDATED ***")
        out.append("    `ir_intensity_km_mol` values in the JSON are the")
        out.append("    dipole-moment derivative projected onto each mode,")
        out.append("    times the Gaussian/ORCA 42.2561 km/mol per")
        out.append("    (D/Å)²/amu prefactor.  HOW dmu/dR was obtained is")
        out.append("    decided at RUN time and recorded in the JSON as")
        out.append("    `ir_route` -- analytic, off the Hessian's own CPHF")
        out.append("    solution, or central finite differences.  This")
        out.append("    header cannot say which: the deck is written before")
        out.append("    it runs, and the analytic route needs a module that")
        out.append("    no PyPI release of pyscf-properties carries.")
        out.append("    Band-level validation 2026-08-20: water at")
        out.append("    B3LYP/def2-SVP lands in the literature windows with")
        # Both pointers must NAME a document that exists.  A prior wording
        # cited "§ 13.1" with no document after the 2026-07 docs migration
        # retired the spec that owned that section -- so a scientist told to
        # check the validation status had nowhere to go.
        out.append("    the right band ordering (docs/archive/2026-09-01-roadmap.md § 5 records")
        out.append("    the closure; docs/web/spectra.md holds the Raman")
        out.append("    validation).  A mode-by-mode cross-check against an")
        out.append("    external code has not been run;")
        out.append("    quote absolute values only with the caveat.")
        out.append("    For CHARGED molecules (charge != 0) IR intensities")
        out.append("    are origin-dependent and physically ill-defined; the")
        out.append("    values here may be contaminated by origin-shift terms.")
        out.append("")
    out.append("Dependencies:")
    out.append("    Use the generated molbuilder run wrapper after bootstrap:")
    out.append("        bash scripts/install-env.sh bootstrap --yes")
    out.append("")
    # ---- Methods paragraph (verbatim).  Indented one space so the
    # triple-quoted docstring doesn't get confused by `"""` inside
    # nested example markdown.  Markdown survives unmodified.
    out.append("Methods (manuscript-ready prose):")
    out.append("")
    for line in methods_md.splitlines():
        out.append(f"  {line}" if line else "")
    out.append('"""')
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Imports                                                               #
# --------------------------------------------------------------------- #

def _emit_imports(cfg: "VibrationConfigView") -> List[str]:
    out: List[str] = []
    out.append("import time")
    out.append("from datetime import datetime, timezone")
    out.append("")
    out.append("import numpy as np")
    # PySCF imports -- pin to the modules we actually use so the
    # error trail on a missing-dep is targeted.
    if cfg.is_dft:
        out.append("from pyscf import gto, scf, dft")
    else:
        out.append("from pyscf import gto, scf")
    out.append("from pyscf.hessian import thermo as _pyscf_thermo")
    out.append("")
    return out


def _emit_constants(struct: Structure,
                    cfg: "VibrationConfigView",
                    *,
                    methods_md: str,
                    bibliography_keys: List[str]) -> List[str]:
    """Pin runtime constants the user can tweak inline if they
    re-run the script with a small parameter change.

    ``methods_md`` and ``bibliography_keys`` are computed once in the
    deck composer (`vibration_deck.vibration_spec`) and threaded
    through.
    """
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Constants  (mirrored from VibrationConfigView at render time)")
    out.append("# ============================================================")
    out.append(f"SCHEMA_VERSION = {int(SCHEMA_VERSION)}")
    out.append(f"JOB            = {cfg.job_name!r}")
    # _mb_outfile is emitted ONCE, by the deck's own block from the one
    # definition every PySCF deck carries (`input.emit_outfile_helper`,
    # beside the script): the lifted resolve(__file__) form that stood
    # here wrote artifacts one level up through the bundle-root link.
    out.append("")
    out.append("# Phase status vocabulary -- matches molbuilder.spectra.results")
    out.append("# so the on-disk JSON round-trips into the typed SpectraResults.")
    out.append("PHASE_EMPTY    = 'empty'")
    out.append("PHASE_RUNNING  = 'running'")
    out.append("PHASE_COMPLETE = 'complete'")
    out.append("PHASE_NOT_REQUESTED = 'not requested'   # never asked for; terminal (vibration.md 4.9)")
    out.append("")
    out.append("# The SCF class + functional + basis + dispersion.  The class is")
    out.append("# composed from the method and the spin treatment (pyscf/layout.py")
    out.append("# scf_class) -- never left for PySCF to re-rule.")
    out.append(f"SCF_CLASS                  = {cfg.scf_class!r}")
    # A Hartree-Fock run has no functional (`PySCFConfig.is_dft`); the
    # constant says so rather than carry a value nothing reads
    # (vibration.md § 4.10).  The dispersion correction applies to either
    # method.
    out.append(f"FUNCTIONAL                 = "
               f"{(cfg.functional if cfg.is_dft else None)!r}")
    out.append(f"BASIS                      = {cfg.basis!r}")
    out.append(f"DISPERSION                 = {cfg.dispersion!r}   "
               f"# 'd3bj' / 'd3zero' / 'd4' / 'none'")
    out.append(f"DENSITY_FIT                = {bool(cfg.density_fit)!r}")
    out.append("")
    out.append("# SCF knobs.")
    out.append(f"SCF_CONV_TOL               = {float(cfg.scf_conv_tol)!r}  "
               f"# Hartree (energy)")
    out.append(f"SCF_MAX_CYCLE              = {int(cfg.scf_max_cycle)!r}")
    # No grid under Hartree-Fock either -- the same rule as FUNCTIONAL's.
    out.append(f"GRID_LEVEL                 = "
               f"{(int(cfg.grid_level) if cfg.is_dft else None)!r}")
    # A CAP LEFT UNSET IS NO CAP (`template.md` § 2), and the deck says so:
    # `None`, which PySCF's `Mole.build` reads as not given, keeping its own
    # setting.  4000 stood in for the blank until 2026-10-06, and the run's
    # recorded facts said it was asked for.
    out.append(f"MAX_MEMORY_MB              = "
               f"{(int(cfg.max_memory_mb) if cfg.max_memory_mb else None)!r}"
               + ("" if cfg.max_memory_mb else
                  "  # not stated: PySCF's own setting"))
    out.append(f"VERBOSE                    = {int(cfg.verbose)!r}")
    out.append(f"USE_GPU                    = {bool(cfg.use_gpu)!r}  "
               f"# probe gpu4pyscf at run start; STOP if unusable "
               f"(no CPU fallback)")
    out.append("")
    out.append("# Frozen atoms: the structure-side set, by 0-based index.")
    out.append("# (Element / residue selectors resolved to indices upstream --")
    out.append("# the /modify panel writes indices, and the union machinery")
    out.append("# that re-derived them here from always-empty lists retired")
    out.append("# 2026-08-21.)")
    out.append(f"FROZEN_INDICES_USER        = {list(cfg.frozen_indices)!r}  "
               f"# 0-based")
    # THE AXES gto.M COMPUTES ON -- a cluster's, from the one door
    # (`cell.engine_axis_kinds`), the axes the Methods count and the
    # settings check counted the same motions on.
    out.append(f"AXIS_KIND                  = {tuple(cfg.axis_kind)!r}  "
               f"# a molecule in free space")
    out.append("")
    out.append("# Spectrum knobs.")
    out.append(f"COMPUTE_RAMAN              = {bool(cfg.compute_raman)!r}")
    out.append(f"COMPUTE_IR                 = {bool(cfg.compute_ir)!r}  "
               f"# band-level validated 2026-08-20 (see the IR banner "
               f"below); no external mode-by-mode cross-check yet")
    out.append(f"DISPLACEMENT_AMPLITUDE_ANG = {float(cfg.displacement_amplitude_ang)!r}  "
               f"# L4 amplitude A; ±A·Q_i along each mode")
    out.append(f"RAMAN_FD_STEP_ANG          = {_RAMAN_FD_STEP_ANG!r}  "
               f"# FD step for dα/dR_k (Raman)")
    out.append("")
    out.append("# Electronic-structure (L4) selection.")
    out.append(f"ES_MODE_SELECTION          = {cfg.es_mode_selection!r}  "
               f"# skip / all / explicit")
    # The modes as numbers, read by the config's one reader (the item is
    # text, "3-7, 12"); `list()` of that text was its characters, and the
    # selector's `int(',')` stopped Phase 4 (vibration.md § 4.8).  Only
    # `explicit` reads the list, so only `explicit` has one.
    _explicit = (cfg.explicit_modes if cfg.es_mode_selection == "explicit"
                 else [])
    out.append(f"ES_EXPLICIT_INDICES        = {_explicit!r}  "
               f"# 1-based; read from {cfg.es_explicit_indices!r}")
    out.append(f"FREQ_MIN_CM1               = {cfg.freq_min_cm1!r}")
    out.append(f"FREQ_MAX_CM1               = {cfg.freq_max_cm1!r}")
    out.append(f"ES_N_HOMO_BELOW            = {int(cfg.es_n_homo_below)!r}")
    out.append(f"ES_N_LUMO_ABOVE            = {int(cfg.es_n_lumo_above)!r}")
    out.append("")

    out.append("# Bibliography keys used in the Methods text + inline comments.")
    out.append("# Verified entries live in docs/science/references.bib.")
    out.append(f"BIBLIOGRAPHY_KEYS          = {list(bibliography_keys)!r}")
    out.append("")
    out.append("# A snapshot of the VibrationConfigView that produced this script,")
    out.append("# round-tripped through plain dict + JSON-safe primitives so")
    out.append("# the value mirrors what lands in spectra.json.config.")
    out.append("# Pretty-printed one key per line so a user opening the script")
    out.append("# can read the actual run parameters without horizontal scroll.")
    # pprint.pformat preserves dict-insertion order with sort_dicts=False
    # so the field order in the script mirrors the dataclass declaration.
    import pprint as _pprint
    _cfg_repr = _pprint.pformat(
        _config_to_jsonable_dict(cfg),
        indent=4, width=80, sort_dicts=False,
    )
    out.append("CONFIG = " + _cfg_repr)
    out.append("")
    out.append("# Methods text, verbatim as composed at render time.")
    # Build the triple-quoted string carefully; escape any """ inside.
    escaped = methods_md.replace('"""', "'''")
    out.append("METHODS_TEXT = \"\"\"" + escaped + "\"\"\"")
    out.append("")
    # Pull the actual molbuilder version from the package metadata so
    # the JSON's provenance.molbuilder_version reflects reality, not
    # a stub string.  Recorded in spectra.json under molbuilder_version
    # for run-provenance auditing (which release rendered this script).
    from .. import __version__ as _MB_VERSION
    out.append(f"MOLBUILDER_VERSION = {_MB_VERSION!r}")
    out.append("")
    return out


def _config_to_jsonable_dict(cfg: "VibrationConfigView") -> dict:
    """Reduce a VibrationConfigView to a plain JSON-safe dict (provenance
    payload for spectra.json.config).  Uses dataclasses.asdict so
    new fields land in the snapshot automatically.

    **And the electronic state as RESOLVED** (`electronic_state`,
    `science/chemistry-correctness.md` § 2a): the config holds a blank where
    the charge or the spin was worked out, so the config alone could not
    say what charge this run carried (plan § 5s.4).  The resolved values and
    where each came from ride beside it."""
    import dataclasses
    out = dataclasses.asdict(cfg)
    out["electronic_state"] = cfg.state.as_dict()
    return out


# --------------------------------------------------------------------- #
# Build mol                                                             #
# --------------------------------------------------------------------- #


def _emit_build_mol(struct: Structure, cfg: "VibrationConfigView",
                    stage_token: str = "", *, positions) -> List[str]:
    """gto.M(...) molecule construction.  The atom geometry is
    inlined as a Python list-of-lists rather than a multi-line
    string so a user can scroll the script and read coordinates
    in Å directly.  ``stage_token`` suffixes the engine log's name
    (stages.md § 1.1a consequence 1), exactly as the optimization
    deck's rungs do."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Build molecule")
    out.append("# ============================================================")
    out.append("t0 = time.time()")
    out.append("print('=== Stage: build molecule ===')")
    out.append("")
    # Format the atoms as a list of [element, x, y, z].
    out.append("ATOMS = [")
    for el, (x, y, z) in zip(struct.elements, positions):
        out.append(f"    ({el!r:>4s}, {x:14.8f}, {y:14.8f}, {z:14.8f}),")
    out.append("]")
    out.append("")
    # ECP: shared resolver with Build's PySCF generator
    # (chemistry.resolve_pyscf_ecp).  ``cfg.ecp`` names the ECP and
    # ``cfg.ecp_atoms`` names which elements get it; anything empty on
    # either side means no ECP.  Nothing is auto-picked.
    from ..chemistry import resolve_pyscf_ecp
    ecp_chosen = resolve_pyscf_ecp(struct, cfg.ecp, cfg.ecp_atoms)
    # ECP is resolved ONCE above (`ecp_chosen`) and emitted once below.
    # A second resolution+emission landed here 2026-08-21 when the
    # render-probe honesty gate flagged the fields "silent" -- they were
    # silent only because the water probe holds no ECP candidate; the
    # gold-dimer probe then saw MY added lines change the text while the
    # original pair emitted too, and `gto.M(ecp=..., ecp=...)` is a
    # SyntaxError in every ECP deck.  Caught by the full-text review's
    # compile probe the same day; the gate now compiles every render.
    # symmetry (category 2, probed 2026-08-21): honored ONLY on the
    # already-relaxed path.  The equilibrium Hessian runs fine under
    # the point group (with PCM too), but a geomeTRIC step or an FD
    # displacement leaves the group, and PySCF's re-symmetrization
    # would fight the very coordinates the derivative is taken at --
    # so the relax-enabled path refuses upstream (kind validator) and
    # the displaced builder forces symmetry off below.
    _use_symmetry = bool(getattr(cfg, "symmetry", False)) and bool(
        getattr(cfg, "already_relaxed", False))
    out.append("mol = gto.M(")
    if _use_symmetry:
        out.append("    symmetry   = True,")
    out.append("    atom       = [[a[0], (a[1], a[2], a[3])] for a in ATOMS],")
    out.append("    basis      = BASIS,")
    if ecp_chosen is not None:
        # ONE shape: the resolver returns ``{element: name}``, emitted as
        # a Python dict-literal so PySCF sees a per-element mapping and
        # not a string containing braces (which it rejects as an unknown
        # ECP name).  The str branch retired with the field, 2026-08-13.
        out.append(f"    ecp        = {dict(ecp_chosen)!r},")
    if getattr(cfg, "log_file", False):
        # The engine's own verbose log, named like the optimization
        # deck names its rungs' (stages.md § 1.1a consequence 1) --
        # the token rides the name, so two rungs in one folder cannot
        # overwrite each other's log.  (The comment claimed this while
        # the emission stayed unsuffixed until 2026-08-21.)
        # RE-APPLIED 2026-08-21: the first landing of this branch was
        # wiped by a baseline restore the same day; the honesty gate
        # (render-probe, config echo stripped) is what caught the loss.
        # `or None` is the caller saying WHICH CASE it is: this signature
        # spells "no ladder" as "" and the grammar spells it as None, and
        # the grammar refuses the empty string rather than reading it as
        # None -- the two produce different filenames.
        from .input import ROLE_LOG
        _logsuf = _rf_tail(ROLE_LOG, stage_token or None)
        out.append(f"    output     = str(_mb_outfile(JOB + {_logsuf!r})),")
    out.append("    verbose    = VERBOSE,")
    out.append("    max_memory = MAX_MEMORY_MB,")
    out.append("    unit       = 'Angstrom',")
    # THE ELECTRONIC STATE's values, each with where it came from
    # (`science/chemistry-correctness.md` § 2a, ES2).
    _st = cfg.state
    out.append(f"    charge     = {_st.net_charge.value},   "
               f"# {_st.net_charge.said}")
    out.append(f"    spin       = {_st.unpaired_electrons.value},   # 2S -- "
               f"{_st.unpaired_electrons.said}")
    out.append(")")
    out.append("ELEMENTS    = [a[0] for a in ATOMS]")
    out.append("N_ATOMS     = mol.natm")
    out.append("# ONE mass array, and `isotope_avg=True` is not a")
    out.append("# preference -- `atom_mass_list()` DEFAULTS to integer mass")
    out.append("# NUMBERS (H=1, Cl=35), while PySCF's own harmonic_analysis")
    out.append("# and thermo.thermo() both use the isotope-averaged masses")
    out.append("# (H=1.008, Cl=35.45).  The deck took the default and the")
    out.append("# all-free path took PySCF's, so freezing an atom silently")
    out.append("# moved every C-H stretch by ~12 cm-1 (2026-09-21).")
    out.append("# thermo.thermo() hardcodes it and takes no mass argument,")
    out.append("# so this is the only convention that agrees with itself.")
    out.append("MASSES_AMU  = np.asarray(mol.atom_mass_list(isotope_avg=True),")
    out.append("                         dtype=float)")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Frozen mask                                                           #
# --------------------------------------------------------------------- #


def _emit_frozen_mask() -> List[str]:
    """Compute the free-atom index list from the frozen indices.

    Indices are the ONE selector: the /modify panel resolves element and
    residue picks to indices before they reach the structure's label
    store, so the union machinery that stood here (a loop over an
    always-empty FROZEN_ELEMENTS, a comment-only residue arm) computed
    nothing and retired 2026-08-21."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Frozen-atom mask")
    out.append("# ============================================================")
    out.append("_frozen = set(int(i) for i in FROZEN_INDICES_USER if 0 <= int(i) < N_ATOMS)")
    out.append("FROZEN_ATOM_IDXS = sorted(_frozen)")
    out.append("FREE_ATOM_IDXS   = [i for i in range(N_ATOMS) if i not in _frozen]")
    out.append("N_FREE           = len(FREE_ATOM_IDXS)")
    out.append("print(f'Atoms: {N_ATOMS} total, {N_FREE} free, "
               "{len(FROZEN_ATOM_IDXS)} frozen')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Initial state (Setup-complete)                                        #
# --------------------------------------------------------------------- #


def _emit_initial_state() -> List[str]:
    """Initialise the in-memory SpectraResults-shape dict and
    write the first checkpoint marking phase_frequencies=running
    (Setup complete; harmonic analysis about to start)."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Initial state -- write before any heavy compute")
    out.append("# ============================================================")
    out.append("# Live-watch picks this up immediately so the UI can show the")
    out.append("# input geometry + 'about to run' phase status.")
    out.append("# Provenance hash of the starting geometry.  Format:")
    out.append("#")
    out.append("#   line 0:    N_ATOMS")
    out.append("#   line 1:    JOB (the job_name)")
    out.append("#   line k+2:  '<element:left-aligned width 3>"
               " <x:14.8f> <y:14.8f> <z:14.8f>'  in Å")
    out.append("#   joined with '\\n'; SHA-256 of UTF-8 bytes.")
    out.append("#")
    out.append("# Recorded in the JSON under 'structure_hash' so a future")
    out.append("# loader can detect a mismatch between the saved spectrum")
    out.append("# and the user's current geometry (e.g. you re-loaded a file")
    out.append("# and edited a coordinate -- the spectrum no longer applies).")
    out.append("# The hash is provenance / audit data; no production code")
    out.append("# enforces it today.")
    # ONE spelling of the hash, the artifact writer's own, imported from
    # mb_pyscf.pyz so the SIESTA derivation and this deck cannot compute two
    # different ones.
    out.append("STRUCTURE_HASH = _mb_structure_hash_text(")
    out.append("    N_ATOMS, JOB, ELEMENTS, [(a[1], a[2], a[3]) for a in ATOMS])")
    out.append("")
    out.append("# Read pyscf's installed version from packaging metadata --")
    out.append("# more reliable than getattr(pyscf, '__version__') because")
    out.append("# some environments drop dunders during repackaging.")
    out.append("try:")
    out.append("    from importlib.metadata import version as _pkg_version")
    out.append("    ENGINE_VERSION = _pkg_version('pyscf')")
    out.append("except Exception:")
    out.append("    import pyscf as _pyscf")
    out.append("    ENGINE_VERSION = getattr(_pyscf, '__version__', '?')")
    out.append("")
    out.append("state = {")
    out.append("    'schema_version':     SCHEMA_VERSION,")
    out.append("    'engine':             'pyscf',")
    out.append("    'engine_version':     ENGINE_VERSION,")
    out.append("    'molbuilder_version': MOLBUILDER_VERSION,")
    out.append("    'timestamp':          datetime.now(timezone.utc).isoformat()"
               ".replace('+00:00', 'Z'),")
    out.append("    'structure_hash':     STRUCTURE_HASH,")
    out.append("    'n_atoms_total':      N_ATOMS,")
    out.append("    'free_atom_idxs':     FREE_ATOM_IDXS,")
    out.append("    'frozen_atom_idxs':   FROZEN_ATOM_IDXS,")
    out.append("    'equilibrium':        {")
    out.append("        'scf_energy_eh':  0.0,            # placeholder until SCF")
    out.append("        'mo_energies_eh': [],")
    out.append("        'homo_idx':       0,")
    out.append("    },")
    out.append("    'modes':                    [],")
    out.append("    'selected_mode_idxs_1based': [],")
    out.append("    'config':                    CONFIG,")
    out.append("    'methods_text':              METHODS_TEXT,")
    out.append("    'bibliography_keys':         BIBLIOGRAPHY_KEYS,")
    out.append("    'phase_frequencies':         PHASE_RUNNING,")
    out.append("    'phase_raman':               (PHASE_EMPTY if COMPUTE_RAMAN else PHASE_NOT_REQUESTED),")
    out.append("    'phase_ir':                  (PHASE_EMPTY if COMPUTE_IR else PHASE_NOT_REQUESTED),")
    out.append("    'raman_route':               'none',")
    out.append("    'raman_fd_step_ang':         None,")
    out.append("    'phase_es':                  (PHASE_EMPTY if ES_MODE_SELECTION != 'skip' else PHASE_NOT_REQUESTED),")
    out.append("    'engine_metadata':           {},")
    out.append("    # Runtime facts collected by _emit_threading_setup +")
    out.append("    # _emit_gpu_setup (n_threads, gpu name, etc.).  Visible")
    out.append("    # on the /results page so users can verify the run")
    out.append("    # actually used the resources they expected.")
    out.append("    'runtime_info':              dict(_RUNTIME_INFO),")
    out.append("}")
    out.append("# We deliberately don't write the initial state yet -- the")
    out.append("# SpectraResults shape requires non-empty MO energies +")
    out.append("# valid homo_idx for the partition check, so the first")
    out.append("# meaningful checkpoint is after the equilibrium SCF.")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Equilibrium SCF                                                       #
# --------------------------------------------------------------------- #


def _emit_equilibrium_scf(cfg: "VibrationConfigView", struct: Structure) -> List[str]:
    """Run the SCF at the input geometry; populate the
    equilibrium sub-dict of state and write the first JSON
    checkpoint."""
    scf_class = cfg.scf_class        # THE one composition (layout.scf_class)

    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Equilibrium SCF")
    out.append("# ============================================================")
    out.append("print('=== Stage: equilibrium SCF ===')")
    if cfg.is_dft:
        # _dft is gpu4pyscf.dft when USE_GPU AND the import succeeded;
        # plain pyscf.dft otherwise.  Same RKS / UKS class names in
        # both, so the rest of the SCF setup is identical.
        out.append(f"mf = _dft.{scf_class}(mol)")
    else:
        out.append(f"mf = _scf.{scf_class}(mol)")
    # Every method: the dispersion correction applies to Hartree-Fock too.
    out.append("mf = _mb_configure_theory(mf)   # the one spelling (§ 7a)")
    out.append("if DENSITY_FIT:")
    out.append("    mf = mf.density_fit(**_MB_DF_KW)")
    out.append("mf = _mb_apply_solvent(mf)")
    out.append("mf = _mb_configure_scf(mf)")
    # Hard-SCF hint when an open-shell metal is present -- the one wording
    # (`pyscf/layout.hard_scf_hint`), shared with the optimization deck.
    from .layout import hard_scf_hint
    out += hard_scf_hint(cfg.state)
    # Site extras per the § 7a role table: checkpoint write and the
    # Newton wrap ride the EQUILIBRIUM mf only (render-time branches
    # on the config -- self-documenting in the emitted text).
    if getattr(cfg, "chkfile", False):
        from .input import ROLE_CHK
        out.append(f'mf.chkfile = _mb_outfile(JOB + "{ROLE_CHK}")')
    if getattr(cfg, "scf_soscf", False):
        out.append("# Second-order SCF (engines/pyscf.md § 7): Newton solver;")
        out.append("# DIIS/damp stop applying under it, by design.")
        out.append("mf = mf.newton()")
    # THE RECORD, before the equilibrium SCF runs (`model/parse.md` § 5d.3a):
    # every item this kind carries, what it was set to and what the live
    # objects hold -- the optimization deck's own emitter, so the two decks
    # cannot record differently.  This deck printed none until 2026-09-26.
    from .input import _emit_effective_parameters
    out.extend(_emit_effective_parameters(cfg, cfg.is_dft,
                                          calculation="vibration",
                                          state=cfg.state))
    out.append("E_eq = mf.kernel()")
    out.append("if not mf.converged:")
    # The equilibrium SCF halts UNCONDITIONALLY on non-convergence --
    # restored 2026-08-21 after a same-day mis-wiring: on_nonconvergence
    # is the RELAXATION's policy (its catalogue help says so: what to do
    # when geomeTRIC's criteria are not met), and no policy makes an
    # unconverged equilibrium density acceptable HERE -- the Hessian,
    # every intensity and the thermochemistry are read from it.
    out.append("    raise SystemExit(")
    out.append("        f'SCF did not converge (E={E_eq!r}); '")
    out.append("        f'increase scf_max_cycle or revisit '")
    out.append("        f'the input geometry'")
    out.append("    )")
    out.append("MO_ENERGIES_EQ = _mb_as_numpy(mf.mo_energy).copy()")
    # THE HOMO RULE IS IMPORTED, NOT RETYPED (2026-09-09; from mb_pyscf.pyz
    # since 2026-10-05).  It has a branch -- RHF/RKS gives a 1-D mo_occ,
    # UHF/UKS a 2-D (alpha, beta) that must be summed -- and a second copy of
    # a branch is a second thing to get wrong.  `spectra.pyscf_vibration.
    # homo_index` is the one implementation and the one the tests call.
    out.append("HOMO_IDX = _mb_homo_index(_mb_as_numpy(mf.mo_occ))")
    out.append("")
    out.append("state['equilibrium'] = {")
    out.append("    'scf_energy_eh':  float(E_eq),")
    out.append("    'mo_energies_eh': _mb_finite_or_none(MO_ENERGIES_EQ),")
    out.append("    'homo_idx':       HOMO_IDX,")
    out.append("    # THE GEOMETRY THE HESSIAN IS TAKEN AT -- the relaxed one when")
    out.append("    # this deck relaxed, the input otherwise (COORDS_EQ_ANG is")
    out.append("    # rebound by the relaxation).  The viewer animates modes from")
    out.append("    # it, and a mode's eigenvectors belong to THIS geometry, not")
    out.append("    # to the input's.")
    out.append("    'elements':       list(ELEMENTS),")
    out.append("    'positions_ang':  np.asarray(COORDS_EQ_ANG, dtype=float).tolist(),")
    out.append("}")
    out.append("_mb_write_spectra_payload(state, JSON_PATH)")
    out.append("print(f'Equilibrium SCF: E = {E_eq:.10f} Ha; HOMO index = {HOMO_IDX}')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Hessian / L2                                                          #
# --------------------------------------------------------------------- #


def _emit_gpu_coverage_probe(cfg: "VibrationConfigView") -> List[str]:
    """Probe what gpu4pyscf can do for the current SCF type.

    Sets two flags the downstream stages read:

      _GPU_HAS_HESSIAN
          True if and only if ``mf.Hessian()`` returns a gpu4pyscf-
          backed object for THIS SCF type.  We check the class's
          module because ``hasattr(mf, 'Hessian')`` is True even
          when gpu4pyscf only inherits pyscf's CPU implementation
          (which would TypeError on CuPy mo_coeff downstream).
      _GPU_HAS_POLARIZABILITY
          Always False today -- gpu4pyscf does not yet expose
          analytic CPHF polarizability.  Reported here for
          diagnostic completeness; the Raman block already forces
          CPU at the ``_build_mf_at`` boundary.

    When ``USE_GPU`` is False the probe is a no-op (both flags
    False, no warning).  The probe itself is cheap -- constructing
    a Hessian object does no compute.
    """
    out: List[str] = []
    out.append("")
    out.append("# ============================================================")
    out.append("#  GPU coverage probe (decides which stages run on GPU)")
    out.append("# ============================================================")
    out.append("# gpu4pyscf's coverage is a moving target: as of 2026-05 it")
    out.append("# supports analytic Hessian for RKS/UKS but lags on others,")
    out.append("# and does not expose analytic CPHF polarizability at all.")
    out.append("# Rather than hard-coding which (SCF_CLASS, stage) pairs work,")
    out.append("# we probe the actual mf object after SCF: if Hessian()")
    out.append("# returns a gpu4pyscf-module object, the kernel works on")
    out.append("# GPU; otherwise we rebuild mf on CPU for the Hessian step.")
    out.append("_GPU_HAS_HESSIAN        = False")
    out.append("_GPU_HAS_POLARIZABILITY = False  # by gpu4pyscf design (as of 2026-05)")
    out.append("if _USING_GPU:")
    out.append("    try:")
    out.append("        _h_probe = mf.Hessian()")
    out.append("        _GPU_HAS_HESSIAN = (")
    out.append("            type(_h_probe).__module__.startswith('gpu4pyscf')")
    out.append("        )")
    out.append("    except (AttributeError, NotImplementedError):")
    out.append("        _GPU_HAS_HESSIAN = False")
    out.append("    _gaps = [k for k, ok in {")
    out.append("        'Hessian':        _GPU_HAS_HESSIAN,")
    out.append("        'Polarizability': _GPU_HAS_POLARIZABILITY,")
    out.append("    }.items() if not ok]")
    out.append("    if _gaps:")
    out.append("        print(f'GPU coverage gaps (will use CPU for these): {_gaps}')")
    out.append("    else:")
    out.append("        print('GPU coverage: SCF + Hessian.')")
    out.append("")
    return out


def _emit_hessian_block(cfg: "VibrationConfigView") -> List[str]:
    """Analytic Hessian -> the one harmonic path -> frequencies +
    eigenvectors.

    Free and held atoms take the SAME path: `vibrational_modes` (imported
    from `spectra.normal_modes`) keeps the free-free block of the true
    Hessian, mass-weights it, and diagonalises it in the complement of the
    whole-body motions that survive holding the frozen set -- six or five
    for a free molecule, fewer or none with atoms held, decided by a rank
    and never by a table.  Exactly ``3 * N_FREE - N_RIGID`` modes come
    out and every one is a vibration (science/normal-modes.md R2-R4).

    GPU branching at the kernel call (driven by the
    ``_GPU_HAS_HESSIAN`` flag set by :func:`_emit_gpu_coverage_probe`):
    when gpu4pyscf covers Hessian for this SCF type, use ``mf``
    directly and bridge the CuPy return to NumPy.  When it does not,
    rebuild ``mf`` on CPU and run the Hessian there.  Downstream
    harmonic-analysis and mass-weighting code is the same in either
    case because it sees only the NumPy Hessian.
    """
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Phase 2: Hessian -> frequencies + eigenvectors")
    out.append("# ============================================================")
    out.append("print('=== Stage: analytic Hessian ===')")
    out.append("# Branch on the GPU-coverage probe set above:")
    out.append("#   _GPU_HAS_HESSIAN True   -> use mf directly, bridge CuPy -> NumPy")
    out.append("#                              right at the kernel() boundary so")
    out.append("#                              the harmonic path (CPU-only) gets")
    out.append("#                              a NumPy array.")
    out.append("#   _GPU_HAS_HESSIAN False  -> rebuild mf on CPU and run Hessian")
    out.append("#                              there.  Costs one extra SCF but is")
    out.append("#                              the only path when gpu4pyscf does")
    out.append("#                              not cover Hessian for this SCF type.")
    out.append("if _GPU_HAS_HESSIAN or not _USING_GPU:")
    out.append("    _mf_for_hess = mf")
    out.append("else:")
    out.append("    print('  rebuilding mf on CPU for the Hessian step')")
    out.append("    _mf_for_hess = _build_mf_at(COORDS_EQ_ANG, force_cpu=True)")
    out.append("")
    out.append("# The Hessian and -- when IR is the ONLY intensity asked for --")
    out.append("# dmu/dR come from the same CPHF solve.  When Raman IS on, its")
    out.append("# displacement loop runs regardless and reads the dipole at each")
    out.append("# point for free, so the analytic route would buy nothing: the")
    out.append("# 6N SCFs are already being spent on dalpha/dR.  That is the whole")
    out.append("# rule for which route runs.")
    out.append("# ...and only with every atom free: the analytic route takes no atom")
    out.append("# list, and with atoms held the Hessian is computed for the free atoms")
    out.append("# only (the reduced calculation -- see dipole_derivatives).")
    out.append("_WANT_ANALYTIC_IR = COMPUTE_IR and not COMPUTE_RAMAN and N_FREE == N_ATOMS")
    out.append("# The reduced route runs on a plain mean field: PySCF's density-fitted")
    out.append("# Hessian class takes no atom list.  One extra SCF, then a Hessian over")
    out.append("# the free atoms only.  With nothing held the mean field is the SCF's own.")
    out.append("HESSIAN_DENSITY_FIT = bool(DENSITY_FIT) and N_FREE == N_ATOMS")
    out.append("if N_FREE < N_ATOMS and DENSITY_FIT:")
    out.append("    print('  rebuilding mf without density fitting for the free-atom Hessian')")
    out.append("    _mf_for_hess = _build_mf_at(COORDS_EQ_ANG, density_fit=False,")
    out.append("                                force_cpu=(not _GPU_HAS_HESSIAN))")
    out.append("_HESS_RAW, DMU_DR, IR_ROUTE = _mb_dipole_derivatives(")
    out.append("    _mf_for_hess, FREE_ATOM_IDXS, _WANT_ANALYTIC_IR)")
    out.append("HESS = _mb_as_numpy(_HESS_RAW)")
    out.append("# What the Hessian covered, recorded for the reader and the Methods text:")
    out.append("# 'free' -- second derivatives for the free atoms only (atoms are held);")
    out.append("# 'all'  -- every atom (nothing held; the free molecule).")
    out.append("HESSIAN_SCOPE = 'free' if N_FREE < N_ATOMS else 'all'")
    out.append("state['hessian_scope'] = HESSIAN_SCOPE")
    out.append("state['n_atoms_in_hessian'] = int(N_FREE)")
    out.append("state['hessian_density_fit'] = HESSIAN_DENSITY_FIT")
    out.append("print(f'  Hessian scope: {HESSIAN_SCOPE} ({N_FREE} of {N_ATOMS} atoms)')")
    out.append("if COMPUTE_IR and not _WANT_ANALYTIC_IR:")
    out.append("    IR_ROUTE = 'finite-difference'   # the Raman loop supplies it")
    out.append("state['ir_route'] = IR_ROUTE")
    out.append("if IR_ROUTE == 'finite-difference':")
    out.append("    # The step belongs to the RESULT, not to whichever block")
    out.append("    # happened to spend it: both finite-difference paths use")
    out.append("    # the one step (the Raman loop's, when Raman is on and")
    out.append("    # the dipole rides along), and a Methods section quoting")
    out.append("    # a finite-difference derivative has to state it.")
    out.append("    state['ir_fd_step_ang'] = RAMAN_FD_STEP_ANG")
    out.append("print(f'  IR dipole-derivative route: {IR_ROUTE}')")
    out.append("# HESS shape: (n_atoms, n_atoms, 3, 3) in Hartree / Bohr².")
    out.append("")
    out.append("# ------------------------------------------------------------")
    out.append("# Two normal-mode arrays are computed and used DIFFERENTLY:")
    out.append("#")
    out.append("#   NORM_MODES_CANONICAL  (n_modes, N_FREE, 3)")
    out.append("#       Cartesian normal modes L_cart in the *canonical*")
    out.append("#       mass-weighted normalisation: sum_k m_k |L_cart_k|^2 = 1")
    out.append("#       (mass in amu).  This is the form the standard")
    out.append("#       Placzek Raman-activity formula expects.  The 45 a^2 +")
    out.append("#       7 gamma^2 scalar comes out in (a.u. polarizability)² /")
    out.append("#       (Å² · amu), which is rescaled by BOHR_ANGSTROM**6 in")
    out.append("#       Phase 3 to get the textbook Å^4/amu (see comments in")
    out.append("#       _emit_raman_block).  CONSUMED BY: the Raman projection")
    out.append("#       (dα/dQ = Σ_k dα/dR_k · L_cart_k).")
    out.append("#")
    out.append("#   NORM_MODES_DISPLAY    (n_modes, N_FREE, 3)")
    out.append("#       Per-mode rescaling of NORM_MODES_CANONICAL so that")
    out.append("#       max(|L_display|) = 1.  Dimensionless.  CONSUMED BY:")
    out.append("#       the 3Dmol mode-animation viewer in the spectra tab,")
    out.append("#       and the per-mode ES displacement (q ± A · L_display)")
    out.append("#       in Phase 4 -- both want a deterministic peak")
    out.append("#       displacement at the user's chosen amplitude rather")
    out.append("#       than the canonical mass-weighted scale (where heavy")
    out.append("#       atoms barely move and light ones move a lot).")
    out.append("#")
    out.append("# Both forms land in the JSON under explicit labels:")
    out.append("#   eigenvector_canonical -> the SCIENCE form")
    out.append("#   eigenvector_display   -> the UI form")
    out.append("# The Raman projection in Phase 3 reads NORM_MODES_CANONICAL")
    out.append("# directly; the ES loop in Phase 4 reads")
    out.append("# eigenvector_display out of the JSON.")
    out.append("# ------------------------------------------------------------")
    out.append("")
    out.append("# THE ONE PATH (science/normal-modes.md R1-R4): the free-free block")
    out.append("# of the TRUE Hessian, mass-weighted, diagonalised in the complement")
    out.append("# of the whole-body motions that survive holding FROZEN_ATOM_IDXS")
    out.append("# still.  With nothing frozen that is the free molecule's six (or")
    out.append("# five) motions removed; with atoms frozen it is the same calculation")
    out.append("# over what survives -- one code path, so the two cannot disagree.")
    out.append("#")
    out.append("# gto.M builds a molecule in free space, so the motions the energy is")
    out.append("# invariant under are an isolated system's (AXIS_KIND).  A structure")
    out.append("# that repeats along an axis is computed as this cluster: its cell is")
    out.append("# not respected, and the settings check said so before this was written.")
    out.append("_LAMBDA, NORM_MODES_CANONICAL, RIGID_PATTERNS = _mb_vibrational_modes(")
    out.append("    HESS, MASSES_AMU, COORDS_EQ_ANG, FROZEN_ATOM_IDXS, AXIS_KIND)")
    out.append("N_RIGID = int(len(RIGID_PATTERNS))")
    out.append("# omega = sign(lambda) * sqrt(|lambda|): a negative eigenvalue of the")
    out.append("# mass-weighted Hessian is an imaginary mode, reported as a negative")
    out.append("# wavenumber -- the steps both routes take after diagonalising, from")
    out.append("# their one home (spectra/normal_modes).")
    out.append("_OMEGA_AU = _mb_signed_omega(_LAMBDA)")
    out.append("FREQ_CM1  = _mb_frequencies_cm1(_LAMBDA)")
    out.append("HAS_IMAG  = [bool(f < 0) for f in FREQ_CM1]")
    out.append("print(f'  whole-body motions removed before diagonalising: {N_RIGID}; '")
    out.append("      f'{len(FREQ_CM1)} modes = 3*{N_FREE} - {N_RIGID}')")
    out.append("")
    out.append("# The DISPLAY form (max(|L|)=1 per mode), from the canonical form.")
    out.append("# Both forms ship in the JSON under explicit names so consumers")
    out.append("# don't have to compute one from the other.")
    out.append("NORM_MODES_DISPLAY = _mb_display_form(NORM_MODES_CANONICAL)")
    out.append("")
    out.append("# Build the modes payload, one record per mode.  The JSON keys")
    out.append("# below are the SCHEMA_VERSION contract (spectra/results.py):")
    out.append("#")
    out.append("#   eigenvector_canonical -- (N_FREE, 3) Cartesian normal mode")
    out.append("#       with the canonical mass-weighted unit norm")
    out.append("#       Σ_k m_k |L_k|^2 = 1.  USE FOR: physical-amplitude")
    out.append("#       quantities (Placzek Raman activity, IR intensities,")
    out.append("#       electron-phonon coupling gradients).")
    out.append("#")
    out.append("#   eigenvector_display   -- same mode rescaled so max(|L_k|)=1.")
    out.append("#       Dimensionless.  USE FOR: 3D animation in the viewer")
    out.append("#       and the fixed-amplitude electron-phonon probe in Phase 4.")
    out.append("#       Do NOT plug into physical-amplitude formulas.")
    out.append("modes_payload = []")
    out.append("for _i, _f in enumerate(FREQ_CM1):")
    out.append("    _L_canonical = NORM_MODES_CANONICAL[_i]")
    out.append("    _L_display   = NORM_MODES_DISPLAY[_i]")
    out.append("    # Serialise each eigenvector exactly once -- _mb_finite_or_none")
    out.append("    # converts NaN/Inf to JSON-safe None and returns a plain list.")
    out.append("    _evec_canon_json = _mb_finite_or_none(_L_canonical)")
    out.append("    _evec_disp_json  = _mb_finite_or_none(_L_display)")
    out.append("    modes_payload.append({")
    out.append("        'index_1based':          int(_i + 1),")
    out.append("        'frequency_cm1':         float(_f),")
    out.append("        'raman_activity_a4_amu': None,")
    out.append("        'ir_intensity_km_mol':   None,")
    out.append("        'eigenvector_canonical': _evec_canon_json,")
    out.append("        'eigenvector_display':   _evec_disp_json,")
    out.append("        'has_imag':              bool(HAS_IMAG[_i]),")
    out.append("        'electronic_structure':  None,")
    out.append("    })")
    out.append("state['modes'] = modes_payload")
    out.append("# What was removed, beside what was kept (R7): the count and the")
    out.append("# Cartesian patterns over the free atoms, so a reader can see that")
    out.append("# a mode list of 3*N_FREE - N_RIGID is complete and what the")
    out.append("# difference was.")
    out.append("state['removed_motions'] = {")
    out.append("    'count':    N_RIGID,")
    out.append("    'patterns': [_mb_finite_or_none(_p) for _p in RIGID_PATTERNS],")
    out.append("}")
    out.append("state['phase_frequencies'] = PHASE_COMPLETE")
    out.append("_mb_write_spectra_payload(state, JSON_PATH)")
    out.append("print(f'Phase 2 done: {len(modes_payload)} modes; "
               "{sum(HAS_IMAG)} imaginary')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Raman / L3                                                            #
# --------------------------------------------------------------------- #


def _emit_displaced_scf_helpers(cfg: "VibrationConfigView") -> List[str]:
    """COORDS_EQ_ANG + _build_mf_at -- shared between L3 (Raman FD)
    and L4 (per-mode ES).  Emit ONCE per script whenever either
    phase will run, so the names exist regardless of which phase
    block is enabled."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Shared helpers for L3 / L4 (displaced-geometry SCFs)")
    out.append("# ============================================================")
    out.append("# Equilibrium coords (Å).  Used as the reference geometry")
    out.append("# both for Raman finite-difference (L3) and per-mode")
    out.append("# displacement (L4).")
    out.append("COORDS_EQ_ANG = np.asarray([[a[1], a[2], a[3]] for a in ATOMS])")
    out.append("")
    out.append("def _build_mf_at(coords, *, density_fit=None, force_cpu=False):")
    out.append("    '''Re-build mol at new coords + reconverge SCF.")
    out.append("")
    out.append("    density_fit=None  -> follow the global DENSITY_FIT flag.")
    out.append("    density_fit=False -> force the non-DF code path (the")
    out.append("                         polarizability CPHF in pyscf-properties")
    out.append("                         doesn't have a DF implementation yet, so")
    out.append("                         the Raman finite-difference calls force this).")
    out.append("    density_fit=True  -> force DF on regardless of global.")
    out.append("    force_cpu=True    -> use stock PySCF even when _USING_GPU is True.")
    out.append("                         The Raman polarizability path needs this")
    out.append("                         because gpu4pyscf doesn't yet expose")
    out.append("                         analytic CPHF polarizability.'''")
    out.append("    _mol_new = mol.copy()")
    out.append("    # displacements leave the point group; re-symmetrization")
    out.append("    # would reorient the frame under the derivative (probed:")
    out.append("    # C2v -> Cs on a 0.005 A displacement).  Always off here.")
    out.append("    _mol_new.symmetry = False")
    out.append("    _mol_new.atom = [[ELEMENTS[_i], tuple(coords[_i])]")
    out.append("                     for _i in range(N_ATOMS)]")
    out.append("    _mol_new.unit = 'Angstrom'")
    out.append("    _mol_new.build(dump_input=False)   # engines/pyscf.md § 3, (2)")
    # THE METHOD IS A RENDER-TIME FACT, so only the live arm is
    # emitted (the E-M4.7 shape, taken one step further at the U6
    # close): an HF deck used to carry the DFT arm as dead text, with
    # a reference -- `dft` -- that exists only on DFT decks.  Dead text
    # with dead names is exactly where that NameError class hides.
    if cfg.is_dft:
        out.append("    # _dft is gpu4pyscf when _USING_GPU else stock pyscf;")
        out.append("    # force_cpu overrides to stock pyscf regardless.")
        out.append("    _dft_mod = dft if force_cpu else _dft")
        out.append("    _cls = getattr(_dft_mod, SCF_CLASS)")
        out.append("    _mf2 = _cls(_mol_new)")
    else:
        out.append("    _scf_mod = scf if force_cpu else _scf")
        out.append("    _cls = getattr(_scf_mod, SCF_CLASS)")
        out.append("    _mf2 = _cls(_mol_new)")
    out.append("    _mf2 = _mb_configure_theory(_mf2)  # the one spelling (§ 7a)")
    out.append("    _use_df = DENSITY_FIT if density_fit is None else density_fit")
    out.append("    if _use_df:")
    out.append("        _mf2 = _mf2.density_fit(**_MB_DF_KW)")
    out.append("    _mf2 = _mb_apply_solvent(_mf2)")
    out.append("    # The one SCF dresser (pyscf.md § 7a): the displaced and")
    out.append("    # relaxation cycles run the same machinery the equilibrium")
    out.append("    # one does -- init guess, level shift, damping, DIIS.")
    out.append("    _mf2 = _mb_configure_scf(_mf2)")
    out.append("    _mf2.kernel()")
    out.append("    if not _mf2.converged:")
    out.append("        raise SystemExit('displaced SCF did not converge at "
               "FD step')")
    out.append("    return _mf2")
    out.append("")
    return out


def _emit_ir_projection() -> List[str]:
    """Per-mode IR intensity (km/mol) from the dipole-moment
    derivative collected in the Raman FD loop.

    TWO composers emit this block: ``_emit_raman_block`` (after the
    Raman activity loop) and the IR-only arm
    (``vibration_deck._vib_ir_only_block``, whose dipole sweep fills
    the same ``DMU_DR``).  Either way it relies on ``DMU_DR``,
    ``modes_payload`` and ``NORM_MODES_CANONICAL`` -- the last defined
    by the Hessian block on every path -- already being in scope.

    SCIENTIFIC VALIDATION STATUS: BAND-LEVEL VALIDATED
    (2026-08-20, water at B3LYP/def2-SVP against literature
    windows with the right band ordering -- docs/archive/2026-09-01-roadmap.md § 5
    records the closure).  The projection math + Gaussian/ORCA
    km/mol prefactor are textbook; a mode-by-mode cross-check
    against an external code (Gaussian / ORCA / Turbomole) has
    not been run, so quote absolute values with that caveat.
    """
    out: List[str] = []
    out.append("")
    out.append("# ----- IR (band-level validated; see header banner) -----")
    out.append("# Standard IR intensity formula for a normal mode of frequency ν_n")
    out.append("# in km/mol, given the dipole-moment derivative dμ/dQ_n (3-vector):")
    out.append("#")
    out.append("#     I_n  =  (N_A · π) / (3 · c²)  ·  |dμ/dQ_n|²")
    out.append("#")
    out.append("# With μ in Debye, R in Å, Q the canonical mass-weighted normal")
    out.append("# coordinate (units Å·√amu), the prefactor that converts the")
    out.append("# squared derivative |dμ/dQ|² [D²/(Å²·amu)] to km/mol is the")
    out.append("# Gaussian / ORCA / literature constant 42.2561 .  Derivation:")
    out.append("#   K = N_A · π / (3·c²) · (D/Å)² / amu  →  km/mol")
    out.append("# (CODATA 2018 N_A, c, e·a₀ → D; 1 Å = 10⁻¹⁰ m; 1 amu = 1.66054e-27 kg)")
    out.append("# Same value cited by Gaussian whitepaper on IR intensities,")
    out.append("# ORCA manual, and the psi4 source.")
    out.append("#")
    out.append("# NOTE: not yet cross-validated against Gaussian/ORCA numerically;")
    out.append("# see docs/web/spectra.md for validation status.")
    out.append("#")
    out.append("# Charged-molecule caveat: PySCF's mf.dip_moment() picks an origin")
    out.append("# (geometric center of atoms by default).  The dipole is origin-")
    out.append("# invariant only for NEUTRAL systems.  For a charged molecule,")
    out.append("# dμ/dR_k captured by central difference picks up a non-physical")
    out.append("# Q_total·(∂R_origin/∂R_k) term that contaminates the projection.")
    out.append("# IR intensity is physically ill-defined for charged molecules")
    out.append("# anyway (any IR code has this caveat), so we don't try to fix it")
    out.append("# here -- but a user computing IR on a cation/anion should treat")
    out.append("# absolute values with extra suspicion.")
    out.append("_IR_PREFACTOR_KM_MOL_PER_D2_PER_A2_PER_AMU = 42.2561")
    out.append("for _n in range(len(modes_payload)):")
    out.append("    _L_canonical = NORM_MODES_CANONICAL[_n]")
    out.append("    # dμ/dQ_n is a 3-vector; einsum sums DMU_DR over its k (atom)")
    out.append("    # and α (direction) axes weighted by L_canonical, leaving the")
    out.append("    # remaining axis i (dipole Cartesian component).")
    out.append("    _dmudq = np.einsum('kai,ka->i', DMU_DR, _L_canonical)")
    out.append("    _ir_intensity = (")
    out.append("        _IR_PREFACTOR_KM_MOL_PER_D2_PER_A2_PER_AMU")
    out.append("        * float(np.dot(_dmudq, _dmudq))")
    out.append("    )")
    out.append("    modes_payload[_n]['ir_intensity_km_mol'] = _ir_intensity")
    return out


def _emit_raman_block(cfg: "VibrationConfigView") -> List[str]:
    """Finite-difference dα/dR_k for k over free Cartesians;
    project onto modes; Raman activity per mode.

    Requires COORDS_EQ_ANG and _build_mf_at from the shared
    displaced-SCF helper block (always emitted before this when
    compute_raman is True).

    When ``cfg.compute_ir`` is also True, this block additionally
    captures dipole moments at each displaced SCF and projects
    them onto the normal modes for IR intensities -- essentially
    free, since the SCFs already converged for polarizability."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Phase 3: Raman activities (finite-difference dα/dR)")
    out.append("# ============================================================")
    out.append("# Cost note: this stage runs ~6*N_FREE displaced-geometry SCFs")
    out.append("# (±FD step in each of 3 Cartesian directions per free atom).")
    out.append("# Each SCF is ~10-30% of the equilibrium SCF wall time.  For")
    out.append("# N_FREE > ~50 atoms this can dominate the L2 cost.")
    out.append("#")
    out.append("# Method: at each q ± δ·e_kα (δ = RAMAN_FD_STEP_ANG, k atom,")
    out.append("# α direction), recompute the static polarizability α(q).")
    out.append("# Central difference gives dα/dR_kα.  Project onto each mode's")
    out.append("# normal-mode eigenvector to get dα/dQ_n.")
    out.append("# Raman activity per mode = 45·a^2 + 7·γ^2 where:")
    out.append("#   a (mean polarizability deriv) = trace/3")
    out.append("#   γ (anisotropy)                = sqrt(sum of squared diffs/2)")
    out.append("# [Wilson1955 ch. 4; Komornicki1979 for the analytic dα/dR")
    out.append("# theory, here approximated by central-difference dα/dR].")
    out.append("print('=== Stage: Raman activities ===')")
    out.append("state['phase_raman'] = PHASE_RUNNING")
    if cfg.compute_ir:
        # the dipole rides along the same displaced points (vibration.md 4.9)
        out.append("state['phase_ir'] = PHASE_RUNNING")
    out.append("_mb_write_spectra_payload(state, JSON_PATH)")
    out.append("")
    out.append("# Polarizability requires pyscf-properties, installed in the managed")
    out.append("# molbuilder-pySCF environment by bootstrap. Core PySCF doesn't ship")
    out.append("# the analytic CPHF polarizability; pyscf-properties adds it as")
    out.append("# `mf.Polarizability().polarizability()`.")
    out.append("#")
    out.append("# The DF (density_fit) variant is NOT YET implemented in")
    out.append("# pyscf-properties 0.1.x; the Raman block forces non-DF SCFs")
    out.append("# for the polarizability evaluations.  The Hessian path stays")
    out.append("# DF (controlled by the global DENSITY_FIT flag) so we only")
    out.append("# pay the non-DF cost for the 6*N_FREE polarizability points.")
    out.append("try:")
    out.append("    import pyscf.prop.polarizability  # noqa: F401")
    out.append("except ImportError:")
    out.append("    raise SystemExit(")
    out.append("        'COMPUTE_RAMAN=True requires the optional pyscf-properties '")
    out.append("        'package.  Bootstrap or repair the managed backend: bash scripts/install-env.sh bootstrap --yes'")
    out.append("    )")
    out.append("")
    out.append("def _polarizability(_mf):")
    out.append("    '''Static dipole polarizability at the converged mf.'''")
    out.append("    # _mf is force_cpu=True for polarizability (gpu4pyscf")
    out.append("    # doesn't expose analytic CPHF), so this is CPU NumPy --")
    out.append("    # the bridge is defensive in case the call path changes.")
    out.append("    return _mb_as_numpy(_mf.Polarizability().polarizability())")
    out.append("")
    if cfg.compute_ir:
        out.append("def _dipole_debye(_mf):")
        out.append("    '''Dipole moment in Debye at the converged mf.'''")
        out.append("    # mf.dip_moment() defaults to unit='Debye'; we pass")
        out.append("    # it explicitly so a future PySCF version that flips")
        out.append("    # the default can't silently change our units.")
        out.append("    # verbose=0 suppresses the per-call print; the mf is")
        out.append("    # already converged so dip_moment() is essentially a")
        out.append("    # one-line integral, not another SCF.")
        out.append("    return _mb_as_numpy(_mf.dip_moment(unit='Debye', verbose=0))")
        out.append("")
    out.append("def _displace(coords, atom_idx, direction, delta):")
    out.append("    '''Return a copy of coords with one Cartesian shifted.'''")
    out.append("    new = coords.copy()")
    out.append("    new[atom_idx, direction] += delta")
    out.append("    return new")
    out.append("")
    # (An equilibrium polarizability was computed here and never read -- one
    # non-density-fitted SCF and a CPHF solve per Raman run, for nothing;
    # removed 2026-10-05, the M11 review's C21.)
    out.append("# Build dα/dR_kα by central difference for each free-atom Cartesian.")
    out.append("# Each displaced mean field is built on the CPU (gpu4pyscf exposes")
    out.append("# no analytic CPHF polarizability) and without density fitting")
    out.append("# (pyscf-properties has no DF polarizability) -- the two flags are")
    out.append("# orthogonal, and both apply to this step only.")
    out.append("DALPHA_DR = np.zeros((N_FREE, 3, 3, 3))   # (k, α_dir, i, j)")
    if cfg.compute_ir:
        out.append("# IR: dμ/dR_kα captured in the SAME displaced SCFs that")
        out.append("# Raman uses -- the dipole moment is a one-line integral on")
        out.append("# an already-converged mf, so this is essentially free.")
        out.append("# Units: μ in Debye (set explicitly in _dipole_debye), R in Å.")
        out.append("DMU_DR    = np.zeros((N_FREE, 3, 3))      # (k, α_dir, i)")
    out.append("for _k_idx, _atom_idx in enumerate(FREE_ATOM_IDXS):")
    out.append("    for _dir in range(3):")
    out.append("        _mf_plus  = _build_mf_at(_displace(COORDS_EQ_ANG, ")
    out.append("                                          _atom_idx, _dir, ")
    out.append("                                          +RAMAN_FD_STEP_ANG),")
    out.append("                                 density_fit=False,")
    out.append("                                 force_cpu=True)")
    out.append("        _mf_minus = _build_mf_at(_displace(COORDS_EQ_ANG, ")
    out.append("                                           _atom_idx, _dir, ")
    out.append("                                           -RAMAN_FD_STEP_ANG),")
    out.append("                                 density_fit=False,")
    out.append("                                 force_cpu=True)")
    out.append("        _ap = _polarizability(_mf_plus)")
    out.append("        _am = _polarizability(_mf_minus)")
    out.append("        DALPHA_DR[_k_idx, _dir] = (_ap - _am) / (2 * RAMAN_FD_STEP_ANG)")
    if cfg.compute_ir:
        out.append("        _dp = _dipole_debye(_mf_plus)")
        out.append("        _dm = _dipole_debye(_mf_minus)")
        out.append("        DMU_DR[_k_idx, _dir] = (_dp - _dm) / (2 * RAMAN_FD_STEP_ANG)")
    out.append("    print(f'  Raman FD: atom {_atom_idx + 1}/{N_ATOMS} done')")
    out.append("")
    out.append("# Per-mode Raman activity in Å^4 / amu via Placzek's formula.")
    out.append("#")
    out.append("# We project the per-Cartesian polarizability derivative tensor")
    out.append("#     dα/dR_{k,α}       shape (3, 3)   [a.u. polarizability / Å]")
    out.append("# onto each normal mode's eigenvector, summing over free-atom")
    out.append("# index k and Cartesian direction α:")
    out.append("#     dα/dQ_n = Σ_{k,α} (dα/dR_{k,α}) · L_canonical_{k,α,n}")
    out.append("# Substituted into the Placzek scalar, this yields a quantity")
    out.append("# with units (a.u. polarizability)² / (Å² · amu) -- NOT yet the")
    out.append("# textbook Å^4/amu -- because PySCF reports polarizability in")
    out.append("# atomic units (volume = Bohr³).  The conversion is exact and")
    out.append("# global: multiply by (Bohr/Å)^6 = BOHR_ANGSTROM^6 ≈ 0.02197.  We")
    out.append("# apply that factor once on the final scalar (see the loop")
    out.append("# below), so what lands in JSON under 'raman_activity_a4_amu'")
    out.append("# is in genuine Å^4/amu -- comparable to Gaussian/ORCA Raman")
    out.append("# activity columns.  Without this factor the relative spectrum")
    out.append("# shape is still correct (uniform scale), but absolute")
    out.append("# intensities are ~50× too small.")
    out.append("#")
    out.append("# (Note: using the *display* form (max|L|=1) instead of the")
    out.append("# canonical mass-weighted L_cart would additionally shift")
    out.append("# activities by a mass-distribution-dependent factor PER MODE")
    out.append("# -- that was the partial-Hessian-path bug fixed by the v2")
    out.append("# canonical/display split.)")
    out.append("#")
    out.append("# Placzek (isotropic Raman) activity for plane-polarised light,")
    out.append("# averaged over molecular orientation:")
    out.append("#     S_n = 45 · a_n² + 7 · γ_n²")
    out.append("# with the mean polarizability derivative")
    out.append("#     a_n = (dα_xx + dα_yy + dα_zz) / 3")
    out.append("# and the anisotropy of the polarizability derivative")
    out.append("#     γ_n² = ½[(dα_xx - dα_yy)² + (dα_yy - dα_zz)² + (dα_zz - dα_xx)²]")
    out.append("#          + 3·(dα_xy² + dα_yz² + dα_xz²)")
    out.append("# (where dα_ij is the ij-component of dα/dQ_n) -- see Wilson1955")
    out.append("# ch.4 and Komornicki1979 for the analytic-CPHF version we")
    out.append("# approximate here via the finite-difference dα/dR.")
    out.append("def _raman_activity(d_alpha_d_Q):")
    out.append("    '''45 a² + 7 γ² in (a.u. polariz)² / (Å² · amu).")
    out.append("")
    out.append("    The caller multiplies by BOHR_ANGSTROM**6 to convert to the")
    out.append("    standard Å^4/amu units used in literature reports.")
    out.append("    '''")
    out.append("    _a = (d_alpha_d_Q[0, 0] + d_alpha_d_Q[1, 1] +")
    out.append("          d_alpha_d_Q[2, 2]) / 3.0")
    out.append("    _xx, _yy, _zz = (d_alpha_d_Q[0, 0], d_alpha_d_Q[1, 1],")
    out.append("                     d_alpha_d_Q[2, 2])")
    out.append("    _gamma_sq = 0.5 * (")
    out.append("        (_xx - _yy)**2 + (_yy - _zz)**2 + (_zz - _xx)**2")
    out.append("    ) + 3.0 * (d_alpha_d_Q[0, 1]**2 + d_alpha_d_Q[1, 2]**2 +")
    out.append("               d_alpha_d_Q[0, 2]**2)")
    out.append("    return 45.0 * _a**2 + 7.0 * _gamma_sq")
    out.append("")
    out.append("# Conversion from (a.u. polariz)² / (Å²·amu) to Å^4/amu:")
    out.append("# polarizability has units of volume; PySCF reports a.u. (Bohr³),")
    out.append("# and the textbook Raman activity formula expects Å³.")
    out.append("# (Bohr/Å)^6 = BOHR_ANGSTROM^6  ≈ 0.02197 -- the one home's")
    out.append("# factor, imported with the constants module from mb_pyscf.pyz.")
    out.append("_RAMAN_AU2_TO_A4AMU = _mb_constants.BOHR_ANGSTROM ** 6")
    out.append("")
    out.append("for _n in range(len(modes_payload)):")
    out.append("    # NORM_MODES_CANONICAL is the canonical mass-weighted form")
    out.append("    # (sum_k m_k |L_k|² = 1).  Using the *display* form here")
    out.append("    # would give activities in different units per mode.")
    out.append("    _L_canonical = NORM_MODES_CANONICAL[_n]")
    out.append("    # dα/dQ_n -- the einsum sums DALPHA_DR over its k (atom) and")
    out.append("    # α (direction) axes, weighted by the eigenvector L_canonical;")
    out.append("    # remaining axes (i, j) are the polarizability Cartesian pair.")
    out.append("    _dadq = np.einsum('kaij,ka->ij', DALPHA_DR, _L_canonical)")
    out.append("    _act = _raman_activity(_dadq) * _RAMAN_AU2_TO_A4AMU")
    out.append("    modes_payload[_n]['raman_activity_a4_amu'] = float(_act)")
    if cfg.compute_ir:
        out.extend(_emit_ir_projection())
    out.append("state['modes'] = modes_payload")
    out.append("state['phase_raman'] = PHASE_COMPLETE")
    if cfg.compute_ir:
        out.append("state['phase_ir'] = PHASE_COMPLETE")
    out.append("# How dalpha/dR was obtained, for the run as a whole -- the one")
    out.append("# Raman route there is, and the step a Methods paragraph must")
    out.append("# quote to be reproducible (SpectraResults.raman_route).")
    out.append("state['raman_route'] = 'finite-difference'")
    out.append("state['raman_fd_step_ang'] = RAMAN_FD_STEP_ANG")
    out.append("_mb_write_spectra_payload(state, JSON_PATH)")
    if cfg.compute_ir:
        out.append("print(f'Phase 3 done: Raman + IR for "
                   "{len(modes_payload)} modes')")
    else:
        out.append("print(f'Phase 3 done: Raman activities for "
                   "{len(modes_payload)} modes')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# ES loop / L4                                                          #
# --------------------------------------------------------------------- #


def _emit_es_loop(cfg: "VibrationConfigView") -> List[str]:
    """Per selected mode: displace q ± A·L_n, run SCF at each
    displaced geometry, record the MO window around HOMO/LUMO."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Phase 4: per-mode displaced electronic structure")
    out.append("# ============================================================")
    out.append("# Cost: 2 SCFs per selected mode (q+A·Q and q-A·Q).")
    out.append("# Output: MO-energy window [HOMO - ES_N_HOMO_BELOW,")
    out.append("# LUMO + ES_N_LUMO_ABOVE] at each displaced geometry, plus")
    out.append("# the SCF energy.  Used for electron-phonon coupling")
    out.append("# analysis [Galperin2007, Frederiksen2007].")
    out.append("print('=== Stage: per-mode electronic structure ===')")
    out.append("state['phase_es'] = PHASE_RUNNING")
    out.append("_mb_write_spectra_payload(state, JSON_PATH)")
    out.append("")
    # Which modes: molbuilder's own selector, imported from mb_pyscf.pyz at
    # the top of the deck (`engines/pyscf.md` § 3) -- a hand-written copy of
    # it stood here until 2026-10-05, held equal to it by a test.
    out.append("# Which modes get the probe (vibration.md 4.8): molbuilder's")
    out.append("# own selector, imported at the top of this script.")
    out.append("_selected = _mb_select_modes(")
    out.append("    [m['frequency_cm1'] for m in modes_payload], ES_MODE_SELECTION,")
    out.append("    explicit=ES_EXPLICIT_INDICES,")
    out.append("    freq_min_cm1=FREQ_MIN_CM1, freq_max_cm1=FREQ_MAX_CM1)")
    # A listed number that names no mode is dropped by the selector, and
    # said here -- the mode count is known only now.
    out.append("_dropped = [i for i in ES_EXPLICIT_INDICES if i not in _selected]")
    out.append("if _dropped:")
    out.append("    print(f'  WARN: no mode numbered {_dropped}; the modes are '")
    out.append("          f'1..{len(modes_payload)}')")
    out.append("state['selected_mode_idxs_1based'] = list(_selected)")
    out.append("")
    out.append("for _idx_1 in _selected:")
    out.append("    _mode_pos = _idx_1 - 1")
    out.append("    # ES displacement uses the DISPLAY form (max|L|=1) so the")
    out.append("    # user-specified DISPLACEMENT_AMPLITUDE_ANG is a deterministic")
    out.append("    # peak displacement per mode, independent of mass distribution.")
    out.append("    # This is a probe geometry for electron-phonon sensitivity, not")
    out.append("    # a physically-amplitude vibrational sample -- the choice of")
    out.append("    # display-form is intentional here.")
    out.append("    _evec = np.asarray(modes_payload[_mode_pos]['eigenvector_display'])")
    out.append("    # Displace all free atoms; the mode's eigenvector is")
    out.append("    # already restricted to free atoms.")
    out.append("    _disp_plus  = COORDS_EQ_ANG.copy()")
    out.append("    _disp_minus = COORDS_EQ_ANG.copy()")
    out.append("    for _k_idx, _atom_idx in enumerate(FREE_ATOM_IDXS):")
    out.append("        _disp_plus[_atom_idx]  += DISPLACEMENT_AMPLITUDE_ANG * _evec[_k_idx]")
    out.append("        _disp_minus[_atom_idx] -= DISPLACEMENT_AMPLITUDE_ANG * _evec[_k_idx]")
    out.append("    # Displaced-geometry SCFs.  Same setup as equilibrium")
    out.append("    # (SCF_CLASS / FUNCTIONAL / BASIS / GRID_LEVEL / DENSITY_FIT);")
    out.append("    # _build_mf_at handles the gpu4pyscf vs CPU branching")
    out.append("    # internally via the _USING_GPU flag.")
    out.append("    _mfp = _build_mf_at(_disp_plus)")
    out.append("    _mfm = _build_mf_at(_disp_minus)")
    # The window around each geometry's own HOMO -- the HOMO rule and the
    # window rule, molbuilder's, imported (spectra/pyscf_vibration).
    out.append("    _mos_p, _ = _mb_mo_window(_mb_as_numpy(_mfp.mo_energy),")
    out.append("                              _mb_homo_index(_mb_as_numpy(_mfp.mo_occ)),")
    out.append("                              ES_N_HOMO_BELOW, ES_N_LUMO_ABOVE)")
    out.append("    _mos_m, _ = _mb_mo_window(_mb_as_numpy(_mfm.mo_energy),")
    out.append("                              _mb_homo_index(_mb_as_numpy(_mfm.mo_occ)),")
    out.append("                              ES_N_HOMO_BELOW, ES_N_LUMO_ABOVE)")
    out.append("    # Use the EQUILIBRIUM window for the 'eq' slice so the")
    out.append("    # three arrays share length even when an orbital swap")
    out.append("    # shifts the HOMO index at a displaced geometry.")
    out.append("    _mos_eq, _homo_in_eq = _mb_mo_window(MO_ENERGIES_EQ, HOMO_IDX,")
    out.append("                                         ES_N_HOMO_BELOW, ES_N_LUMO_ABOVE)")
    out.append("    _n_win = len(_mos_eq)")
    out.append("    # Re-slice ± arrays to match the equilibrium window size.")
    out.append("    # If a displaced HOMO shifted, take the same-length slice.")
    out.append("    _mos_p = _mos_p[:_n_win] if len(_mos_p) >= _n_win else (")
    out.append("        np.concatenate([_mos_p, np.full(_n_win - len(_mos_p), np.nan)])")
    out.append("    )")
    out.append("    _mos_m = _mos_m[:_n_win] if len(_mos_m) >= _n_win else (")
    out.append("        np.concatenate([_mos_m, np.full(_n_win - len(_mos_m), np.nan)])")
    out.append("    )")
    out.append("    modes_payload[_mode_pos]['electronic_structure'] = {")
    out.append("        'amplitude_ang':        float(DISPLACEMENT_AMPLITUDE_ANG),")
    out.append("        'mo_energies_eq_eh':    _mb_finite_or_none(_mos_eq),")
    out.append("        'mo_energies_minus_eh': _mb_finite_or_none(_mos_m),")
    out.append("        'mo_energies_plus_eh':  _mb_finite_or_none(_mos_p),")
    out.append("        'homo_index_in_window': int(_homo_in_eq),")
    out.append("        'scf_energy_eq_eh':     float(E_eq),")
    out.append("        'scf_energy_minus_eh':  float(_mfm.e_tot),")
    out.append("        'scf_energy_plus_eh':   float(_mfp.e_tot),")
    out.append("    }")
    out.append("    # Per-mode checkpoint -- live-watch can show ES")
    out.append("    # incrementally as each mode completes.")
    out.append("    state['modes'] = modes_payload")
    out.append("    _mb_write_spectra_payload(state, JSON_PATH)")
    out.append("    print(f'  Mode {_idx_1}: ES recorded')")
    out.append("")
    out.append("state['phase_es'] = PHASE_COMPLETE")
    out.append("_mb_write_spectra_payload(state, JSON_PATH)")
    out.append("print(f'Phase 4 done: {len(_selected)} modes with ES data')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Final summary                                                         #
# --------------------------------------------------------------------- #


def _emit_final_summary() -> List[str]:
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Done")
    out.append("# ============================================================")
    out.append("t1 = time.time()")
    out.append("print(f'" + SPECTRUM_END_MARKER + " {t1 - t0:.1f} s')")
    out.append("print(f'Results: {JSON_PATH}')")
    out.append("")
    return out


# The module's one public name: the deck composes the private
# emitters via explicit imports, and the Methods paragraph is what
# spectra/methods.py documents as this engine's fragment.  (The old
# generator's render_spectra_script died with the class; its name
# in __all__ made `import *` raise.)
__all__ = ["pyscf_methods_fragment"]

# --------------------------------------------------------------------- #
# Methods paragraph (engine-specific fragment)                          #
# --------------------------------------------------------------------- #


def pyscf_methods_fragment(cfg: "VibrationConfigView") -> str:
    """Engine-specific paragraph for the Methods section.

    MOVED at P3 from ``PySCFSpectraEngine.methods_fragment`` (the class
    retired with the old generator); the unused ``modes`` parameter was
    dropped.  Names PySCF + the specific Hessian /
    polarizability-derivative APIs used, with citation keys that
    resolve against ``docs/science/references.bib`` and bubble up into
    the trailing bibliography of the full Methods text
    (:func:`spectra.methods.render_methods_md` composes it in).
    """
    # Method-class-specific Hessian module name -- the PySCF
    # API splits hessian.RKS / UKS / RHF / UHF.  Sun2020 +
    # Sun2018 cite the package itself; the analytic Hessian
    # API is covered by both.
    # The module is the composed class's own name (layout.scf_class):
    # rks / uks / rhf / uhf -- the four PySCF has analytic Hessians for; a
    # restricted-open vibration is refused before a deck exists (ES4).
    hessian_module = f"pyscf.hessian.{cfg.scf_class.lower()}"

    parts = [
        "All electronic-structure calculations were performed "
        "with PySCF [Sun2020, Sun2018], a Python-based ab "
        "initio package."
    ]

    # Analytic over finite-difference is a load-bearing claim for the
    # Methods reader (no FD noise on frequencies).  ONE sentence for free
    # and held atoms alike, because there is one path: the free-free block
    # of the true Hessian, the whole-body motions the geometry permits
    # projected out before diagonalisation.  NO COUNT HERE: how many were
    # removed is the geometry's answer (spectra.methods states it from the
    # one derivation, and the artifact records it), and this function is
    # handed `cfg` alone.
    frozen = list(getattr(cfg, "frozen_indices", []) or [])
    if frozen:
        parts.append(
            f"Analytic second derivatives were computed via "
            f"`{hessian_module}` for the free atoms only (PySCF's "
            f"`atmlst`), the held atoms entering through the "
            f"self-consistent field -- partial Hessian vibrational "
            f"analysis [Head1997, LiJensen2002], with the block taken "
            f"from the full adsorbate-plus-environment energy [Besley2008] "
            f"as in Q-Chem's implementation [QChemPHVA]; the partial "
            f"Hessian was mass-weighted and diagonalized after projecting "
            f"out the whole-body motions the held geometry permits "
            f"[Ghysels2008] (the number removed is recorded with the "
            f"results)."
        )
    else:
        parts.append(
            f"The harmonic Hessian was obtained analytically via "
            f"`{hessian_module}`, mass-weighted and diagonalized after "
            f"projecting out the whole-body motions of the free molecule."
        )
    if frozen:
        # THE FROZEN SET, SAID OUT LOUD: freezing is the user's own
        # choice, and what it means must be explicit.  This paragraph is
        # what a reader of the results sees first, so the regime statement
        # lives here.
        # WHAT HELD THEM is the relaxation only when one ran: under
        # `already_relaxed` the deck relaxes nothing (`engines/vibration.md`
        # § 2.2), and the sentence must not claim a step that never happened.
        _held_by = ("the geometry relaxation constrained them (geomeTRIC "
                    "`$freeze`) and"
                    if not bool(getattr(cfg, "already_relaxed", False))
                    else "no relaxation ran (the structure was stated relaxed) "
                         "and")
        parts.append(
            f"{len(frozen)} atom(s) (0-based indices {sorted(frozen)}) "
            f"were held fixed throughout: {_held_by} no second "
            f"derivative was taken with respect to them, so the reported "
            f"frequencies are those of the free atoms moving in the "
            f"static field of the fixed ones.  Thermochemistry is "
            f"vibrational-only (an anchored system has no gas-phase "
            f"translational or rotational partition function)."
        )

    # Raman path: α is analytic (CPHF) at each displaced point; the
    # DERIVATIVE dα/dR is central finite differences over those
    # points -- claiming an analytic derivative overstated the method.
    if cfg.compute_raman:
        parts.append(
            f"Polarizability derivatives dα/dR were computed by "
            f"central finite differences (±{_RAMAN_FD_STEP_ANG:g} Å per "
            f"free Cartesian coordinate) of analytic CPHF "
            f"polarizabilities (`pyscf.prop.polarizability`) at "
            f"displaced geometries, then projected onto the "
            f"mass-weighted mode eigenvectors to obtain Raman "
            f"activities in Å⁴/amu [Komornicki1979]."
        )

    # Density fitting note.  The emitted deck applies DF to the SCF
    # + Hessian path only; when Raman is computed the polarizability
    # points are forced NON-DF (pyscf-properties 0.1.x has no DF
    # polarizability -- see ``_emit_raman_block`` above), so the
    # Methods text must not claim DF across the board.  The Raman
    # caveat rides ONLY when compute_raman is on (mirrors the Raman
    # prose above) so a non-Raman run's Methods never mentions Raman.
    if cfg.density_fit:
        df_note = (
            "Density fitting (RIJK) was used for the SCF"
            + (" and analytic Hessian" if not frozen else
               "; the free-atom Hessian was evaluated without it, on a "
               "plain mean field at the same geometry, because PySCF's "
               "density-fitted Hessian takes no atom list")
            + f"; the auxiliary basis was selected automatically by "
            f"PySCF for the production {cfg.basis} basis."
        )
        if cfg.compute_raman:
            df_note += (
                "  The polarizability evaluations that yield Raman "
                "activities are run without density fitting "
                "(pyscf-properties 0.1.x has no DF polarizability "
                "implementation)."
            )
        parts.append(df_note)

    # Grid level for DFT runs.
    if cfg.is_dft:
        parts.append(
            f"DFT integration used PySCF's grid level "
            f"{cfg.grid_level} (production setting for hybrid "
            f"functionals; the v1 spec § 11.4 sets level 4 as "
            f"the recommended minimum)."
        )

    return " ".join(parts)

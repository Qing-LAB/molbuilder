"""Script-template generator for :class:`PySCFSpectraEngine`.

``render_spectra_script(struct, cfg)`` produces the runnable
``<job>.spectra.py`` script as a single string.  The script is
self-contained: ``python <job>.spectra.py`` on any machine with
the required PySCF stack reproduces the run end-to-end.

The script writes ``<job>.spectra.json`` incrementally, atomically
replacing the file at each phase boundary so a live-watch
poller can pick up partial state without ever seeing a torn
document (spec § 6.1).  The wire format is the engine-agnostic
:class:`SpectraResults` shape (spec § 5 / § 6).

This module is the only place where the actual scientific
choices land in code form:

  * frozen-atom partial Hessian by row/col deletion of the full
    analytic Hessian, mass-weighted on the free-atom subspace;
  * eigenvalue → cm⁻¹ via the PySCF helper
    ``hessian.thermo._freq_from_force_constant`` (the conversion
    factor that pins atomic-units frequencies to wavenumbers);
  * Raman activity via finite-difference dα/dR_k (k over
    free Cartesians), projected onto the mass-weighted mode
    eigenvectors → Raman activity scalar via the standard
    45·α'² + 7·γ'² formula [Wilson1955, Komornicki1979];
  * per-mode displaced SCFs at q ± A·Q_n in real-space along the
    mode eigenvector for the modes selected by
    :func:`spectra.selection.select_modes`.

The emitted script's header docstring is the Methods paragraph
from :func:`spectra.methods.render_methods_md` plus a
bibliography listing -- so a user reading the file can distil a
Methods section verbatim (spec § 11.2).

The atomic JSON writer is INLINED into every emitted script
(no molbuilder import at runtime) so cluster nodes need only
the PySCF stack.  It mirrors the safety rules of
:func:`molbuilder.parsers.spectra_json.dump_spectra_json`
(``allow_nan=False``, ``ensure_ascii=False``, tempfile + replace,
no BOM).
"""

from __future__ import annotations

from typing import List

from ..config.spectra import SpectraConfig
from ..structure import Structure
# Eager imports were lazy before to break a circular dep, but only one
# direction actually cycles (pyscf_engine.render_script -> pyscf_script,
# which stays lazy inside that method body).  Importing pyscf_engine +
# methods at module load here loads the engine class without touching
# pyscf_script, so no cycle.
from .methods import render_methods_md, extract_citation_keys
from .pyscf_engine import PySCFSpectraEngine
from .results import SCHEMA_VERSION


# Conversion factors used in the script.  Pinned here so the
# string-emit and the unit-test cross-check from the same source.
# Hartree-to-cm⁻¹ for ω = sqrt(force constant / mass) follows the
# PySCF convention (see pyscf.hessian.thermo).  We use the explicit
# constant in the script for transparency rather than hiding it
# behind an import.
_CM1_PER_AU_FREQ = 219474.6313632   # 1 atomic unit of frequency (= 2*Rydberg) in cm⁻¹
_BOHR_TO_ANG     = 0.529177210903


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


def render_spectra_script(struct: Structure, cfg: SpectraConfig) -> str:
    """Emit the complete ``<job>.spectra.py`` script as a string.

    The order of emitted blocks matters -- earlier blocks define
    state used by later ones.  Each block is its own
    ``_emit_<name>`` helper so the structure stays inspectable
    and the unit tests can assert on individual block contents.
    """
    # Compute the Methods prose + bibliography ONCE -- the header
    # docstring and the constants block both inline the same prose,
    # so calling render_methods_md three times (as the earlier code
    # did) was wasted work and a consistency risk.
    methods_md = render_methods_md(
        cfg, engine=PySCFSpectraEngine, struct=struct,
    )
    bibliography_keys = extract_citation_keys(methods_md)

    lines: List[str] = []
    lines += _emit_header_docstring(struct, cfg, methods_md=methods_md)
    lines += _emit_imports(cfg)
    lines += _emit_constants(
        struct, cfg,
        methods_md=methods_md,
        bibliography_keys=bibliography_keys,
    )
    lines += _emit_atomic_writer()
    lines += _emit_build_mol(struct, cfg)
    lines += _emit_frozen_mask()
    lines += _emit_initial_state()
    lines += _emit_equilibrium_scf(cfg)
    lines += _emit_hessian_block(cfg)
    # Shared helpers for L3 and L4: both phases run SCFs at displaced
    # geometries via _build_mf_at(coords).  Emit ONCE whenever
    # either L3 or L4 will run -- previously _build_mf_at lived
    # inside the Raman block, so `compute_raman=False` with
    # `es_mode_selection != "none"` crashed the script with NameError.
    needs_displaced_scf = cfg.compute_raman or cfg.es_mode_selection != "none"
    if needs_displaced_scf:
        lines += _emit_displaced_scf_helpers(cfg)
    if cfg.compute_raman:
        lines += _emit_raman_block(cfg)
    if cfg.es_mode_selection != "none":
        lines += _emit_es_loop(cfg)
    lines += _emit_final_summary()
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# Header docstring + Methods paragraph                                  #
# --------------------------------------------------------------------- #


def _emit_header_docstring(struct: Structure,
                           cfg: SpectraConfig,
                           *,
                           methods_md: str) -> List[str]:
    """Triple-quoted Methods + Outputs + Dependencies block.

    The Methods paragraph is the same prose the UI surfaces in
    the Methods-preview modal (spec § 9.4 / § 11.2), so a reader
    sees identical content in the file, the form, and the JSON.
    ``methods_md`` is computed once in :func:`render_spectra_script`
    and threaded through here + ``_emit_constants``.
    """
    out: List[str] = []
    out.append('"""PySCF Spectra input script generated by molbuilder.')
    out.append("")
    out.append(f"System    : {getattr(struct, 'title', None) or 'untitled'}")
    out.append(f"Engine    : PySCF ({PySCFSpectraEngine.label})")
    out.append(f"Method    : {cfg.method} / {cfg.functional} / {cfg.basis}")
    if cfg.dispersion and cfg.dispersion.lower() != "none":
        out.append(f"Dispersion: {cfg.dispersion}")
    out.append(f"Atoms     : {getattr(struct, 'n_atoms', len(struct.elements))}")
    out.append(f"Job name  : {cfg.job_name}")
    out.append("")
    out.append("Run with:")
    out.append(f"    python {cfg.job_name}.spectra.py")
    out.append("")
    out.append("Outputs (atomic-replace at each phase boundary):")
    out.append(f"    {cfg.job_name}.spectra.json     -- typed SpectraResults")
    out.append("                                          (see spec § 5 / § 6)")
    out.append("")
    out.append("Dependencies:")
    out.append("    pip install pyscf")
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


def _emit_imports(cfg: SpectraConfig) -> List[str]:
    out: List[str] = []
    out.append("import json")
    out.append("import math")
    out.append("import os")
    out.append("import tempfile")
    out.append("import time")
    out.append("from datetime import datetime, timezone")
    out.append("")
    out.append("import numpy as np")
    # PySCF imports -- pin to the modules we actually use so the
    # error trail on a missing-dep is targeted.
    method = cfg.method.upper()
    is_dft = method in ("RKS", "UKS")
    if is_dft:
        out.append("from pyscf import gto, scf, dft")
    else:
        out.append("from pyscf import gto, scf")
    out.append("from pyscf import hessian")
    out.append("from pyscf.hessian import thermo as _mb_thermo")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Constants + bibliography                                              #
# --------------------------------------------------------------------- #


def _emit_constants(struct: Structure,
                    cfg: SpectraConfig,
                    *,
                    methods_md: str,
                    bibliography_keys: List[str]) -> List[str]:
    """Pin runtime constants the user can tweak inline if they
    re-run the script with a small parameter change.

    ``methods_md`` and ``bibliography_keys`` are computed once in
    :func:`render_spectra_script` and threaded through.
    """
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Constants  (mirrored from SpectraConfig at render time)")
    out.append("# ============================================================")
    out.append(f"SCHEMA_VERSION = {int(SCHEMA_VERSION)}")
    out.append(f"JOB            = {cfg.job_name!r}")
    out.append("JSON_PATH      = JOB + '.spectra.json'")
    out.append("")
    out.append("# Phase status vocabulary -- matches molbuilder.spectra.results")
    out.append("# so the on-disk JSON round-trips into the typed SpectraResults.")
    out.append("PHASE_EMPTY    = 'empty'")
    out.append("PHASE_RUNNING  = 'running'")
    out.append("PHASE_COMPLETE = 'complete'")
    out.append("")
    out.append("# Method + functional + basis + dispersion.")
    out.append(f"METHOD                     = {cfg.method!r}")
    out.append(f"FUNCTIONAL                 = {cfg.functional!r}")
    out.append(f"BASIS                      = {cfg.basis!r}")
    out.append(f"DISPERSION                 = {cfg.dispersion!r}   "
               f"# None / 'd3' / 'd3bj' / 'd4'")
    out.append(f"DENSITY_FIT                = {bool(cfg.density_fit)!r}")
    out.append("")
    out.append("# SCF knobs.")
    out.append(f"SCF_CONV_TOL               = {float(cfg.scf_conv_tol)!r}  "
               f"# Hartree (energy)")
    out.append(f"SCF_MAX_CYCLE              = {int(cfg.scf_max_cycle)!r}")
    out.append(f"GRID_LEVEL                 = {int(cfg.grid_level)!r}")
    out.append(f"MAX_MEMORY_MB              = {int(cfg.max_memory_mb)!r}")
    out.append(f"VERBOSE                    = {int(cfg.verbose)!r}")
    out.append("")
    out.append("# Frozen-atom mask (UNION of element + residue + explicit).")
    out.append("# The runtime block computes the final FREE_ATOM_IDXS from")
    out.append("# this triplet against the molecule built below.")
    out.append(f"FIXED_ELEMENTS             = {list(cfg.fixed_elements)!r}")
    out.append(f"FIXED_RESIDUE_NAMES        = {list(cfg.fixed_residue_names)!r}")
    out.append(f"FIXED_INDICES_USER         = {list(cfg.fixed_indices)!r}  "
               f"# 0-based")
    out.append("")
    out.append("# Spectrum knobs.")
    out.append(f"COMPUTE_RAMAN              = {bool(cfg.compute_raman)!r}")
    out.append(f"DISPLACEMENT_AMPLITUDE_ANG = {float(cfg.displacement_amplitude_ang)!r}  "
               f"# L4 amplitude A; ±A·Q_i along each mode")
    out.append(f"RAMAN_FD_STEP_ANG          = {_RAMAN_FD_STEP_ANG!r}  "
               f"# FD step for dα/dR_k (Raman)")
    out.append("")
    out.append("# Electronic-structure (L4) selection.")
    out.append(f"ES_MODE_SELECTION          = {cfg.es_mode_selection!r}  "
               f"# none / all / top_n / threshold / explicit")
    out.append(f"ES_TOP_N                   = {int(cfg.es_top_n)!r}")
    out.append(f"ES_THRESHOLD               = {float(cfg.es_threshold)!r}  "
               f"# Å⁴/amu Raman activity cutoff")
    out.append(f"ES_EXPLICIT_INDICES        = {list(cfg.es_explicit_indices)!r}")
    out.append(f"FREQ_MIN_CM1               = {cfg.freq_min_cm1!r}")
    out.append(f"FREQ_MAX_CM1               = {cfg.freq_max_cm1!r}")
    out.append(f"ES_N_HOMO_BELOW            = {int(cfg.es_n_homo_below)!r}")
    out.append(f"ES_N_LUMO_ABOVE            = {int(cfg.es_n_lumo_above)!r}")
    out.append("")
    out.append("# Unit conversions (kept inline so the math in the script")
    out.append("# is self-contained without needing molbuilder at runtime).")
    out.append(f"CM1_PER_AU_FREQ            = {_CM1_PER_AU_FREQ!r}  "
               f"# 1 a.u. of freq in cm⁻¹")
    out.append(f"BOHR_TO_ANG                = {_BOHR_TO_ANG!r}")
    out.append("")
    out.append("# Bibliography keys used in the Methods text + inline comments.")
    out.append("# Verified entries live in docs/tabs/spectra/references.bib.")
    out.append(f"BIBLIOGRAPHY_KEYS          = {list(bibliography_keys)!r}")
    out.append("")
    out.append("# A snapshot of the SpectraConfig that produced this script,")
    out.append("# round-tripped through plain dict + JSON-safe primitives so")
    out.append("# the value mirrors what lands in spectra.json.config.")
    out.append(f"CONFIG = {_config_to_jsonable_dict(cfg)!r}")
    out.append("")
    out.append("# Methods-preview text (verbatim what the UI showed at render).")
    # Build the triple-quoted string carefully; escape any """ inside.
    escaped = methods_md.replace('"""', "'''")
    out.append("METHODS_TEXT = \"\"\"" + escaped + "\"\"\"")
    out.append("")
    # Pull the actual molbuilder version from the package metadata so
    # the JSON's provenance.molbuilder_version reflects reality, not
    # a stub string.
    from .. import __version__ as _MB_VERSION
    out.append(f"MOLBUILDER_VERSION = {_MB_VERSION!r}")
    out.append("")
    return out


def _config_to_jsonable_dict(cfg: SpectraConfig) -> dict:
    """Reduce a SpectraConfig to a plain JSON-safe dict (provenance
    payload for spectra.json.config).  Uses dataclasses.asdict so
    new fields land in the snapshot automatically."""
    import dataclasses
    return dataclasses.asdict(cfg)


# --------------------------------------------------------------------- #
# Atomic JSON writer (inlined into the script)                          #
# --------------------------------------------------------------------- #


def _emit_atomic_writer() -> List[str]:
    """Inline the same atomic-write helper as
    :func:`molbuilder.parsers.spectra_json.dump_spectra_json` so
    the script doesn't need molbuilder at runtime.

    Safety rules (mirror the dump helper):
      * allow_nan=False    -- a non-finite value raises before
                              touching disk.
      * ensure_ascii=False -- cm⁻¹ / Å survive verbatim.
      * tempfile-in-same-dir + os.replace -- atomic on POSIX +
        same-FS Windows.
      * fsync before replace so a crash leaves either the prior
        file or the new file intact, never a half-written one.
    """
    out: List[str] = []
    out.append("")
    out.append("# ============================================================")
    out.append("#  Atomic JSON writer (inlined; mirrors molbuilder's helper)")
    out.append("# ============================================================")
    out.append("def _filter_finite(arr):")
    out.append("    '''Replace NaN/Inf in a numeric array with None for JSON.")
    out.append("    The dump helper rejects allow_nan, so we have to scrub")
    out.append("    BEFORE handing the payload over.  None survives JSON as")
    out.append("    null which the parser handles for optional fields.'''")
    out.append("    a = np.asarray(arr, dtype=float)")
    out.append("    if np.isfinite(a).all():")
    out.append("        return a.tolist()")
    out.append("    out = []")
    out.append("    flat_iter = a.flat")
    out.append("    shape = a.shape")
    out.append("    flat = [float(x) if math.isfinite(x) else None for x in flat_iter]")
    out.append("    # Re-shape via nested lists.  For 1-D this is trivial.")
    out.append("    if a.ndim == 1:")
    out.append("        return flat")
    out.append("    return np.asarray(flat, dtype=object).reshape(shape).tolist()")
    out.append("")
    out.append("def _atomic_write_json(payload, path):")
    out.append("    text = json.dumps(payload, indent=2, ensure_ascii=False,")
    out.append("                      allow_nan=False, sort_keys=False)")
    out.append("    parent = os.path.dirname(os.path.abspath(path)) or '.'")
    out.append("    fd, tmp = tempfile.mkstemp(")
    out.append("        prefix=os.path.basename(path) + '.',")
    out.append("        suffix='.tmp',")
    out.append("        dir=parent,")
    out.append("    )")
    out.append("    try:")
    out.append("        with os.fdopen(fd, 'w', encoding='utf-8') as fh:")
    out.append("            fh.write(text)")
    out.append("            fh.flush()")
    out.append("            try:")
    out.append("                os.fsync(fh.fileno())")
    out.append("            except OSError:")
    out.append("                pass")
    out.append("        os.replace(tmp, path)")
    out.append("    except BaseException:")
    out.append("        try: os.unlink(tmp)")
    out.append("        except OSError: pass")
    out.append("        raise")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Build mol                                                             #
# --------------------------------------------------------------------- #


def _emit_build_mol(struct: Structure, cfg: SpectraConfig) -> List[str]:
    """gto.M(...) molecule construction.  The atom geometry is
    inlined as a Python list-of-lists rather than a multi-line
    string so a user can scroll the script and read coordinates
    in Å directly."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Build molecule")
    out.append("# ============================================================")
    out.append("t0 = time.time()")
    out.append("print('=== Stage: build molecule ===')")
    out.append("")
    # Format the atoms as a list of [element, x, y, z].
    out.append("ATOMS = [")
    for el, (x, y, z) in zip(struct.elements, struct.positions):
        out.append(f"    ({el!r:>4s}, {x:14.8f}, {y:14.8f}, {z:14.8f}),")
    out.append("]")
    out.append("")
    out.append("mol = gto.M(")
    out.append("    atom       = [[a[0], (a[1], a[2], a[3])] for a in ATOMS],")
    out.append("    basis      = BASIS,")
    out.append("    verbose    = VERBOSE,")
    out.append("    max_memory = MAX_MEMORY_MB,")
    out.append("    unit       = 'Angstrom',")
    out.append(")")
    out.append("ELEMENTS    = [a[0] for a in ATOMS]")
    out.append("N_ATOMS     = mol.natm")
    out.append("MASSES_AMU  = np.asarray(mol.atom_mass_list(), dtype=float)")
    out.append("# Convert masses to atomic units (electron masses).  The")
    out.append("# Hessian below is in Hartree/Bohr² and we want frequencies")
    out.append("# in a.u. before the wavenumber conversion.")
    out.append("AMU_TO_AU   = 1822.888486209  # CODATA 2018; 1 Da in m_e units")
    out.append("MASSES_AU   = MASSES_AMU * AMU_TO_AU")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Frozen mask                                                           #
# --------------------------------------------------------------------- #


def _emit_frozen_mask() -> List[str]:
    """Compute the free-atom index list at runtime from the
    user-supplied freeze rules (element / residue / indices).
    Residue-name freezing is a no-op here -- the script has no
    PDB; the molbuilder wrapper that produced this struct would
    have resolved residue names to indices before calling
    render_script (or warned that PDB info wasn't available)."""
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Frozen-atom mask")
    out.append("# ============================================================")
    out.append("# Union of three rules; an atom is FIXED if it matches any:")
    out.append("#   1. its element is in FIXED_ELEMENTS")
    out.append("#   2. its 0-based index is in FIXED_INDICES_USER")
    out.append("#   3. (residue freezing is no-op without PDB info -- the")
    out.append("#       molbuilder layer that emitted this script would have")
    out.append("#       resolved residue names to indices already.)")
    out.append("_fixed = set(int(i) for i in FIXED_INDICES_USER if 0 <= int(i) < N_ATOMS)")
    out.append("for _i, _el in enumerate(ELEMENTS):")
    out.append("    if _el in FIXED_ELEMENTS:")
    out.append("        _fixed.add(_i)")
    out.append("FIXED_ATOM_IDXS = sorted(_fixed)")
    out.append("FREE_ATOM_IDXS  = [i for i in range(N_ATOMS) if i not in _fixed]")
    out.append("N_FREE          = len(FREE_ATOM_IDXS)")
    out.append("print(f'Atoms: {N_ATOMS} total, {N_FREE} free, "
               "{len(FIXED_ATOM_IDXS)} fixed')")
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
    out.append("import hashlib")
    out.append("_xyz_lines = [f'{N_ATOMS}', f'{JOB}']")
    out.append("for _el, (_x, _y, _z) in zip(ELEMENTS, [(a[1], a[2], a[3]) for a in ATOMS]):")
    out.append("    _xyz_lines.append(f'{_el:<3s} {_x:14.8f} {_y:14.8f} {_z:14.8f}')")
    out.append("STRUCTURE_HASH = 'sha256:' + hashlib.sha256(")
    out.append("    '\\n'.join(_xyz_lines).encode('utf-8')")
    out.append(").hexdigest()")
    out.append("")
    out.append("import pyscf as _pyscf")
    out.append("ENGINE_VERSION = getattr(_pyscf, '__version__', '?')")
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
    out.append("    'fixed_atom_idxs':    FIXED_ATOM_IDXS,")
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
    out.append("    'phase_raman':               PHASE_EMPTY,")
    out.append("    'phase_es':                  PHASE_EMPTY,")
    out.append("    'engine_metadata':           {},")
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


def _emit_equilibrium_scf(cfg: SpectraConfig) -> List[str]:
    """Run the SCF at the input geometry; populate the
    equilibrium sub-dict of state and write the first JSON
    checkpoint."""
    method = cfg.method.upper()
    if method in ("RKS", "RHF"):
        scf_class = "RKS" if method == "RKS" else "RHF"
    else:
        scf_class = "UKS" if method == "UKS" else "UHF"

    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Equilibrium SCF")
    out.append("# ============================================================")
    out.append("print('=== Stage: equilibrium SCF ===')")
    if method.endswith("KS"):
        out.append(f"mf = dft.{scf_class}(mol)")
        out.append("mf.xc = FUNCTIONAL")
        out.append("if DISPERSION and DISPERSION.lower() != 'none':")
        out.append("    mf.disp = DISPERSION")
        out.append("mf.grids.level = GRID_LEVEL")
    else:
        out.append(f"mf = scf.{scf_class}(mol)")
    out.append("if DENSITY_FIT:")
    out.append("    mf = mf.density_fit()")
    out.append("mf.conv_tol  = SCF_CONV_TOL")
    out.append("mf.max_cycle = SCF_MAX_CYCLE")
    out.append("E_eq = mf.kernel()")
    out.append("if not mf.converged:")
    out.append("    raise SystemExit(")
    out.append("        f'SCF did not converge (E={E_eq!r}); '")
    out.append("        f'increase scf_max_cycle or revisit '")
    out.append("        f'the input geometry'")
    out.append("    )")
    out.append("MO_ENERGIES_EQ = np.asarray(mf.mo_energy).copy()")
    out.append("# HOMO index: highest occupied MO.  For UHF/UKS the mo_occ")
    out.append("# is 2-D (alpha, beta) -- we sum to total occupancy and find")
    out.append("# the highest level with occupancy > 0.  For RHF/RKS it's 1-D.")
    out.append("_occ = np.asarray(mf.mo_occ)")
    out.append("if _occ.ndim == 2:")
    out.append("    _total_occ = _occ.sum(axis=0)")
    out.append("else:")
    out.append("    _total_occ = _occ")
    out.append("HOMO_IDX = int(np.max(np.where(_total_occ > 0.5)[0]))")
    out.append("")
    out.append("state['equilibrium'] = {")
    out.append("    'scf_energy_eh':  float(E_eq),")
    out.append("    'mo_energies_eh': _filter_finite(MO_ENERGIES_EQ),")
    out.append("    'homo_idx':       HOMO_IDX,")
    out.append("}")
    out.append("_atomic_write_json(state, JSON_PATH)")
    out.append("print(f'Equilibrium SCF: E = {E_eq:.10f} Ha; HOMO index = {HOMO_IDX}')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Hessian / L2                                                          #
# --------------------------------------------------------------------- #


def _emit_hessian_block(cfg: SpectraConfig) -> List[str]:
    """Analytic Hessian -> mass-weighted -> diagonalize ->
    frequencies + eigenvectors.

    The frozen-atom case takes the (N_free × 3) block of the full
    Hessian and mass-weights with the free-atom masses only.  No
    rotation/translation projection in that path -- the fixed
    atoms anchor the system.

    The all-free case calls PySCF's
    ``thermo.harmonic_analysis(mol, hess)`` which handles
    projection internally, then exposes ``freq_au`` and the
    normal-mode eigenvectors.
    """
    out: List[str] = []
    out.append("# ============================================================")
    out.append("#  Phase 2: Hessian -> frequencies + eigenvectors")
    out.append("# ============================================================")
    out.append("print('=== Stage: analytic Hessian ===')")
    out.append("HESS = mf.Hessian().kernel()")
    out.append("# HESS shape: (n_atoms, n_atoms, 3, 3) in Hartree / Bohr².")
    out.append("")
    out.append("if N_FREE == N_ATOMS:")
    out.append("    # All-free path: use PySCF's harmonic_analysis which")
    out.append("    # projects out the 6 translation/rotation modes.")
    out.append("    _ha = _mb_thermo.harmonic_analysis(mol, HESS)")
    out.append("    # freq_wavenumber may be complex with `imag` set for")
    out.append("    # imaginary modes; flatten to signed real wavenumber.")
    out.append("    FREQ_CM1 = np.asarray([")
    out.append("        float(w.real) if abs(w.imag) > 0 else float(w)")
    out.append("        for w in _ha['freq_wavenumber']")
    out.append("    ])")
    out.append("    # Imaginary flag from the original PySCF token.")
    out.append("    HAS_IMAG = [abs(getattr(w, 'imag', 0.0)) > 0")
    out.append("                for w in _ha['freq_wavenumber']]")
    out.append("    # PySCF returns norm_mode of shape (n_modes, n_atoms, 3)")
    out.append("    # -- already mass-weighted normal modes.  Each mode is")
    out.append("    # spanning the FULL atom set; since N_FREE == N_ATOMS")
    out.append("    # the per-mode eigenvector is just norm_mode[i].")
    out.append("    NORM_MODES = np.asarray(_ha['norm_mode'])")
    out.append("else:")
    out.append("    # Partial-Hessian path: cut to the free-atom (N_FREE×3,")
    out.append("    # N_FREE×3) block, mass-weight, diagonalize.  No")
    out.append("    # rotation/translation projection -- the fixed atoms")
    out.append("    # anchor the system in space.")
    out.append("    _free_idx = np.asarray(FREE_ATOM_IDXS, dtype=int)")
    out.append("    _hess_free = HESS[_free_idx][:, _free_idx]")
    out.append("    # Reshape (N_FREE, N_FREE, 3, 3) -> (3*N_FREE, 3*N_FREE)")
    out.append("    _h2 = _hess_free.transpose(0, 2, 1, 3).reshape(")
    out.append("        3 * N_FREE, 3 * N_FREE)")
    out.append("    _masses_free = MASSES_AU[_free_idx]")
    out.append("    _msqrt_inv = 1.0 / np.sqrt(_masses_free)")
    out.append("    # Mass-weight: H_ij / sqrt(m_i * m_j) for each 3x3 block")
    out.append("    _weights = np.repeat(_msqrt_inv, 3)")
    out.append("    _hmw = _h2 * np.outer(_weights, _weights)")
    out.append("    # Symmetrise to clean up numerical asymmetry in the Hessian")
    out.append("    _hmw = 0.5 * (_hmw + _hmw.T)")
    out.append("    _eigvals, _eigvecs = np.linalg.eigh(_hmw)")
    out.append("    # ω in a.u. = sign(λ) · sqrt(|λ|).  Convert to cm⁻¹.")
    out.append("    _omega_au = np.sign(_eigvals) * np.sqrt(np.abs(_eigvals))")
    out.append("    FREQ_CM1  = _omega_au * CM1_PER_AU_FREQ")
    out.append("    HAS_IMAG  = [bool(f < 0) for f in FREQ_CM1]")
    out.append("    # Reshape each eigenvector back into (N_FREE, 3) and")
    out.append("    # convert mass-weighted -> Cartesian normal mode.")
    out.append("    # The eigenvector of the mass-weighted Hessian is L_mw;")
    out.append("    # the per-atom Cartesian displacement direction is")
    out.append("    # L_cart_k = L_mw_k / sqrt(m_k).  _msqrt_inv = 1/sqrt(m)")
    out.append("    # so multiplying by it gives the correct conversion.")
    out.append("    # (Bug fix: earlier code divided by _msqrt_inv, which")
    out.append("    # is equivalent to multiplying by sqrt(m) -- the")
    out.append("    # opposite direction.  Visible in heavy/light atom")
    out.append("    # mixed systems where the light atom should move much")
    out.append("    # more than the heavy one for a typical stretch.)")
    out.append("    NORM_MODES = np.zeros((len(_eigvals), N_FREE, 3))")
    out.append("    for _k in range(len(_eigvals)):")
    out.append("        _v = _eigvecs[:, _k].reshape(N_FREE, 3)")
    out.append("        _v = _v * _msqrt_inv[:, None]  # L_cart = L_mw / sqrt(m)")
    out.append("        # Normalise so max-abs displacement is 1.  This")
    out.append("        # makes the eigenvectors visually comparable in the")
    out.append("        # UI; the L4 amplitude knob then scales them.")
    out.append("        _norm = np.max(np.abs(_v))")
    out.append("        if _norm > 0:")
    out.append("            _v = _v / _norm")
    out.append("        NORM_MODES[_k] = _v")
    out.append("")
    out.append("# Build the modes list.  For the all-free path, every mode")
    out.append("# spans N_ATOMS atoms; we trim to N_FREE = N_ATOMS rows.  For")
    out.append("# the partial path it's already (N_FREE, 3).")
    out.append("modes_payload = []")
    out.append("for _i, _f in enumerate(FREQ_CM1):")
    out.append("    _evec = NORM_MODES[_i]")
    out.append("    if _evec.shape[0] != N_FREE:")
    out.append("        # All-free path returns (N_ATOMS, 3) which equals (N_FREE, 3).")
    out.append("        _evec = _evec.reshape(-1, 3)")
    out.append("    modes_payload.append({")
    out.append("        'index_1based':          int(_i + 1),")
    out.append("        'frequency_cm1':         float(_f),")
    out.append("        'raman_activity_a4_amu': None,")
    out.append("        'ir_intensity_km_mol':   None,")
    out.append("        'eigenvector_free':      _filter_finite(_evec),")
    out.append("        'has_imag':              bool(HAS_IMAG[_i]),")
    out.append("        'electronic_structure':  None,")
    out.append("    })")
    out.append("state['modes'] = modes_payload")
    out.append("state['phase_frequencies'] = PHASE_COMPLETE")
    out.append("_atomic_write_json(state, JSON_PATH)")
    out.append("print(f'Phase 2 done: {len(modes_payload)} modes; "
               "{sum(HAS_IMAG)} imaginary')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Raman / L3                                                            #
# --------------------------------------------------------------------- #


def _emit_displaced_scf_helpers(cfg: SpectraConfig) -> List[str]:
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
    out.append("def _build_mf_at(coords, *, density_fit=None):")
    out.append("    '''Re-build mol at new coords + reconverge SCF.")
    out.append("")
    out.append("    density_fit=None  -> follow the global DENSITY_FIT flag.")
    out.append("    density_fit=False -> force the non-DF code path (the")
    out.append("                         polarizability CPHF in pyscf-properties")
    out.append("                         doesn't have a DF implementation yet, so")
    out.append("                         the Raman FD calls force this).")
    out.append("    density_fit=True  -> force DF on regardless of global.'''")
    out.append("    _mol_new = mol.copy()")
    out.append("    _mol_new.atom = [[ELEMENTS[_i], tuple(coords[_i])]")
    out.append("                     for _i in range(N_ATOMS)]")
    out.append("    _mol_new.unit = 'Angstrom'")
    out.append("    _mol_new.build()")
    out.append("    if METHOD.upper() in ('RKS', 'UKS'):")
    out.append("        _mf2 = (dft.RKS if METHOD.upper() == 'RKS' else dft.UKS)(_mol_new)")
    out.append("        _mf2.xc = FUNCTIONAL")
    out.append("        if DISPERSION and DISPERSION.lower() != 'none':")
    out.append("            _mf2.disp = DISPERSION")
    out.append("        _mf2.grids.level = GRID_LEVEL")
    out.append("    else:")
    out.append("        _mf2 = (scf.RHF if METHOD.upper() == 'RHF' else scf.UHF)(_mol_new)")
    out.append("    _use_df = DENSITY_FIT if density_fit is None else density_fit")
    out.append("    if _use_df:")
    out.append("        _mf2 = _mf2.density_fit()")
    out.append("    _mf2.conv_tol  = SCF_CONV_TOL")
    out.append("    _mf2.max_cycle = SCF_MAX_CYCLE")
    out.append("    _mf2.kernel()")
    out.append("    if not _mf2.converged:")
    out.append("        raise SystemExit('displaced SCF did not converge at "
               "FD step')")
    out.append("    return _mf2")
    out.append("")
    return out


def _emit_raman_block(cfg: SpectraConfig) -> List[str]:
    """Finite-difference dα/dR_k for k over free Cartesians;
    project onto modes; Raman activity per mode.

    Requires COORDS_EQ_ANG and _build_mf_at from the shared
    displaced-SCF helper block (always emitted before this when
    compute_raman is True)."""
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
    out.append("_atomic_write_json(state, JSON_PATH)")
    out.append("")
    out.append("# Polarizability requires the optional pyscf-properties package")
    out.append("# (`pip install pyscf-properties`).  Core PySCF doesn't ship")
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
    out.append("        'package.  Install with:  pip install pyscf-properties'")
    out.append("    )")
    out.append("")
    out.append("def _polarizability(_mf):")
    out.append("    '''Static dipole polarizability at the converged mf.'''")
    out.append("    return np.asarray(_mf.Polarizability().polarizability())")
    out.append("")
    out.append("def _displace(coords, atom_idx, direction, delta):")
    out.append("    '''Return a copy of coords with one Cartesian shifted.'''")
    out.append("    new = coords.copy()")
    out.append("    new[atom_idx, direction] += delta")
    out.append("    return new")
    out.append("")
    out.append("# Build a non-DF mf at the equilibrium geometry for the")
    out.append("# polarizability calculations.  See module-docstring note re:")
    out.append("# DF + Polarizability incompatibility.")
    out.append("_mf_nodf_eq = _build_mf_at(COORDS_EQ_ANG, density_fit=False)")
    out.append("alpha_eq = _polarizability(_mf_nodf_eq)")
    out.append("")
    out.append("# Build dα/dR_kα by central difference for each free-atom Cartesian.")
    out.append("DALPHA_DR = np.zeros((N_FREE, 3, 3, 3))   # (k, α_dir, i, j)")
    out.append("for _k_idx, _atom_idx in enumerate(FREE_ATOM_IDXS):")
    out.append("    for _dir in range(3):")
    out.append("        _mf_plus  = _build_mf_at(_displace(COORDS_EQ_ANG, ")
    out.append("                                          _atom_idx, _dir, ")
    out.append("                                          +RAMAN_FD_STEP_ANG),")
    out.append("                                 density_fit=False)")
    out.append("        _mf_minus = _build_mf_at(_displace(COORDS_EQ_ANG, ")
    out.append("                                           _atom_idx, _dir, ")
    out.append("                                           -RAMAN_FD_STEP_ANG),")
    out.append("                                 density_fit=False)")
    out.append("        _ap = _polarizability(_mf_plus)")
    out.append("        _am = _polarizability(_mf_minus)")
    out.append("        DALPHA_DR[_k_idx, _dir] = (_ap - _am) / (2 * RAMAN_FD_STEP_ANG)")
    out.append("    print(f'  Raman FD: atom {_atom_idx + 1}/{N_ATOMS} done')")
    out.append("")
    out.append("# Project onto each mode: dα/dQ_n = Σ_k Σ_α (dα/dR_kα) · L_kα,n")
    out.append("# where L_kα,n is the mode-n displacement of free-atom k in")
    out.append("# direction α.  L is in our NORM_MODES (n, n_free, 3).")
    out.append("def _raman_activity(d_alpha_d_Q):")
    out.append("    '''Standard Raman activity scalar: 45 a^2 + 7 γ^2.'''")
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
    out.append("for _n, _mode in enumerate(modes_payload):")
    out.append("    _evec = np.asarray(_mode['eigenvector_free'])")
    out.append("    # dα/dQ_n = sum over (k, α) of dα/dR_kα * L_n,kα")
    out.append("    _dadq = np.einsum('kaij,ka->ij', DALPHA_DR, _evec)")
    out.append("    _act = _raman_activity(_dadq)")
    out.append("    modes_payload[_n]['raman_activity_a4_amu'] = float(_act)")
    out.append("state['modes'] = modes_payload")
    out.append("state['phase_raman'] = PHASE_COMPLETE")
    out.append("_atomic_write_json(state, JSON_PATH)")
    out.append("print(f'Phase 3 done: Raman activities for {len(modes_payload)} modes')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# ES loop / L4                                                          #
# --------------------------------------------------------------------- #


def _emit_es_loop(cfg: SpectraConfig) -> List[str]:
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
    out.append("_atomic_write_json(state, JSON_PATH)")
    out.append("")
    out.append("# Resolve which modes to compute.  Logic mirrors")
    out.append("# molbuilder.spectra.selection.select_modes so the script")
    out.append("# behaves identically to the form-side preview.  We inline")
    out.append("# the selector here instead of importing molbuilder at")
    out.append("# runtime (cluster nodes don't need molbuilder).")
    out.append("def _passes_freq_window(freq_cm1):")
    out.append("    if FREQ_MIN_CM1 is not None and freq_cm1 < FREQ_MIN_CM1:")
    out.append("        return False")
    out.append("    if FREQ_MAX_CM1 is not None and freq_cm1 > FREQ_MAX_CM1:")
    out.append("        return False")
    out.append("    return True")
    out.append("")
    out.append("if ES_MODE_SELECTION == 'all':")
    out.append("    _selected = [m['index_1based'] for m in modes_payload")
    out.append("                 if _passes_freq_window(m['frequency_cm1'])]")
    out.append("elif ES_MODE_SELECTION == 'top_n':")
    out.append("    _ranked = sorted(")
    out.append("        [m for m in modes_payload")
    out.append("         if m['raman_activity_a4_amu'] is not None")
    out.append("            and _passes_freq_window(m['frequency_cm1'])],")
    out.append("        key=lambda m: (-m['raman_activity_a4_amu'], m['index_1based']),")
    out.append("    )")
    out.append("    _selected = [m['index_1based'] for m in _ranked[:ES_TOP_N]]")
    out.append("elif ES_MODE_SELECTION == 'threshold':")
    out.append("    _selected = [m['index_1based'] for m in modes_payload")
    out.append("                 if m['raman_activity_a4_amu'] is not None")
    out.append("                    and m['raman_activity_a4_amu'] > ES_THRESHOLD")
    out.append("                    and _passes_freq_window(m['frequency_cm1'])]")
    out.append("elif ES_MODE_SELECTION == 'explicit':")
    out.append("    _selected = [int(i) for i in ES_EXPLICIT_INDICES]")
    out.append("else:")
    out.append("    _selected = []")
    out.append("state['selected_mode_idxs_1based'] = list(_selected)")
    out.append("")
    out.append("def _displaced_scf(disp_coords):")
    out.append("    '''Same SCF setup as equilibrium, at displaced coords.'''")
    out.append("    return _build_mf_at(disp_coords)")
    out.append("")
    out.append("def _mo_window(_mf2):")
    out.append("    '''Slice the MO array to [HOMO-N, LUMO+M] around the")
    out.append("    equilibrium HOMO.  At a displaced geometry orbitals can")
    out.append("    swap; the spec accepts that and lets downstream")
    out.append("    EPC analysis handle alignment.'''")
    out.append("    _mos = np.asarray(_mf2.mo_energy)")
    out.append("    _occ = np.asarray(_mf2.mo_occ)")
    out.append("    if _occ.ndim == 2:")
    out.append("        _tot = _occ.sum(axis=0)")
    out.append("    else:")
    out.append("        _tot = _occ")
    out.append("    _homo = int(np.max(np.where(_tot > 0.5)[0]))")
    out.append("    _lo = max(0, _homo - ES_N_HOMO_BELOW)")
    out.append("    _hi = min(len(_mos), _homo + 1 + ES_N_LUMO_ABOVE)")
    out.append("    return _mos[_lo:_hi].copy(), _homo - _lo")
    out.append("")
    out.append("# Defensive: skip out-of-range explicit indices.  The")
    out.append("# pre-render validator can't range-check explicit indices")
    out.append("# because the mode count isn't known until L2 completes,")
    out.append("# so a user typo (es_explicit_indices=[1, 99] on a 12-mode")
    out.append("# system) would otherwise crash here with IndexError after")
    out.append("# L2 + L3 already burned wall time.  Print + skip instead.")
    out.append("_n_modes_available = len(modes_payload)")
    out.append("_skipped_oor = [i for i in _selected")
    out.append("                if not 1 <= i <= _n_modes_available]")
    out.append("if _skipped_oor:")
    out.append("    print(f'  WARN: skipping out-of-range mode indices "
               "{_skipped_oor}; ' f'valid range is 1..{_n_modes_available}')")
    out.append("_selected = [i for i in _selected")
    out.append("             if 1 <= i <= _n_modes_available]")
    out.append("for _idx_1 in _selected:")
    out.append("    _mode_pos = _idx_1 - 1")
    out.append("    _evec = np.asarray(modes_payload[_mode_pos]['eigenvector_free'])")
    out.append("    # Displace all free atoms; the mode's eigenvector is")
    out.append("    # already restricted to free atoms.")
    out.append("    _disp_plus  = COORDS_EQ_ANG.copy()")
    out.append("    _disp_minus = COORDS_EQ_ANG.copy()")
    out.append("    for _k_idx, _atom_idx in enumerate(FREE_ATOM_IDXS):")
    out.append("        _disp_plus[_atom_idx]  += DISPLACEMENT_AMPLITUDE_ANG * _evec[_k_idx]")
    out.append("        _disp_minus[_atom_idx] -= DISPLACEMENT_AMPLITUDE_ANG * _evec[_k_idx]")
    out.append("    _mfp = _displaced_scf(_disp_plus)")
    out.append("    _mfm = _displaced_scf(_disp_minus)")
    out.append("    _mos_p, _homo_in_win_p = _mo_window(_mfp)")
    out.append("    _mos_m, _homo_in_win_m = _mo_window(_mfm)")
    out.append("    # Use the EQUILIBRIUM window for the 'eq' slice so the")
    out.append("    # three arrays share length even when an orbital swap")
    out.append("    # shifts the HOMO index at a displaced geometry.")
    out.append("    _lo = max(0, HOMO_IDX - ES_N_HOMO_BELOW)")
    out.append("    _hi = min(len(MO_ENERGIES_EQ), HOMO_IDX + 1 + ES_N_LUMO_ABOVE)")
    out.append("    _mos_eq = MO_ENERGIES_EQ[_lo:_hi]")
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
    out.append("        'mo_energies_eq_eh':    _filter_finite(_mos_eq),")
    out.append("        'mo_energies_minus_eh': _filter_finite(_mos_m),")
    out.append("        'mo_energies_plus_eh':  _filter_finite(_mos_p),")
    out.append("        'homo_index_in_window': int(HOMO_IDX - _lo),")
    out.append("        'scf_energy_eq_eh':     float(E_eq),")
    out.append("        'scf_energy_minus_eh':  float(_mfm.e_tot),")
    out.append("        'scf_energy_plus_eh':   float(_mfp.e_tot),")
    out.append("    }")
    out.append("    # Per-mode checkpoint -- live-watch can show ES")
    out.append("    # incrementally as each mode completes.")
    out.append("    state['modes'] = modes_payload")
    out.append("    _atomic_write_json(state, JSON_PATH)")
    out.append("    print(f'  Mode {_idx_1}: ES recorded')")
    out.append("")
    out.append("state['phase_es'] = PHASE_COMPLETE")
    out.append("_atomic_write_json(state, JSON_PATH)")
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
    out.append("print(f'Total wall time: {t1 - t0:.1f} s')")
    out.append("print(f'Results: {JSON_PATH}')")
    out.append("")
    return out


__all__ = ["render_spectra_script"]

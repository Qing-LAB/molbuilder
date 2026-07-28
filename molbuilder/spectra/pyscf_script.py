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
    # v1 IR constraint: IR rides on the same displaced SCFs that
    # Raman runs (dipole moment is essentially free after each
    # converged mf), so compute_ir requires compute_raman.  A
    # standalone IR FD loop would just duplicate that work; if a
    # user ever wants IR-without-Raman the right answer is to enable
    # both flags, accept the polarizability cost, and revisit later.
    if cfg.compute_ir and not cfg.compute_raman:
        raise ValueError(
            "compute_ir=True requires compute_raman=True in v1 "
            "(IR piggybacks on the Raman finite-difference loop; a "
            "standalone IR-only FD path is a future feature)."
        )

    # Cross-cutting validation (open-shell metal, parity, frozen-atom-
    # consumed, peptide protonation, config-metadata range checks).
    # SIESTA + PySCF generators have always called this from inside
    # render_fdf / render_script; the spectra generator was missing
    # the call so CLI / library callers bypassed every check (the
    # web /api/spectra/render endpoint runs ``engine.preflight`` so
    # the web path was covered).  Mirror the SIESTA pattern: validate
    # the struct + cfg, surface warnings to stderr via ``report``,
    # raise ValidationError on hard-error issues so the caller can
    # catch + display.  Caught by the 2026-05-26 fresh-eyes review.
    from ..validation import validate, report
    report(validate(struct, cfg))

    # Compute the Methods prose + bibliography ONCE -- the header
    # docstring and the constants block both inline the same prose,
    # so calling render_methods_md three times (as the earlier code
    # did) was wasted work and a consistency risk.
    methods_md = render_methods_md(
        cfg, engine=PySCFSpectraEngine, struct=struct,
    )
    bibliography_keys = extract_citation_keys(methods_md)

    # Threading + runtime-info + GPU probe are CROSS-CUTTING -- they
    # live in molbuilder.runtime_info and are shared verbatim with
    # Build's PySCF script generator (pyscf/input.py).  Single source
    # of truth for the OMP/BLAS recipe + GPU detection so the two
    # generators can't drift.
    from ..runtime_info import (
        emit_threading_setup_lines,
        emit_runtime_info_capture_lines,
        emit_pyscf_post_import_lines,
        emit_gpu_probe_lines,
    )
    lines: List[str] = []
    lines += _emit_header_docstring(struct, cfg, methods_md=methods_md)
    lines += emit_threading_setup_lines(cfg.threads)
    lines += emit_runtime_info_capture_lines(
        use_gpu=bool(cfg.use_gpu),
        max_memory_mb=int(cfg.max_memory_mb) if cfg.max_memory_mb else None,
    )
    lines += _emit_imports(cfg)
    lines += emit_pyscf_post_import_lines()
    lines.append("_RUNTIME_INFO['n_threads_pyscf'] = int(_pyscf_lib.num_threads())")
    lines += _emit_constants(
        struct, cfg,
        methods_md=methods_md,
        bibliography_keys=bibliography_keys,
    )
    lines += emit_gpu_probe_lines(
        use_gpu=bool(cfg.use_gpu),
        min_compute_capability=int(
            __import__("molbuilder.spectra.pyscf_engine",
                       fromlist=["PySCFSpectraEngine"]
                      ).PySCFSpectraEngine.GPU4PYSCF_MIN_COMPUTE_CAPABILITY
        ),
    )
    # Legacy compat: the existing spectra script uses _scf / _dft
    # pointer rebind for GPU dispatch.  Keep that pattern (it's how
    # _emit_build_mol picks SCF class) by aliasing.
    lines.append("if _USING_GPU:")
    lines.append("    from gpu4pyscf import scf as _gpu_scf")
    if cfg.method.upper() in ("RKS", "UKS"):
        lines.append("    from gpu4pyscf import dft as _gpu_dft")
        lines.append("    _scf = _gpu_scf")
        lines.append("    _dft = _gpu_dft")
    else:
        lines.append("    _scf = _gpu_scf")
        lines.append("    _dft = None")
    lines.append("else:")
    lines.append("    _scf = scf")
    if cfg.method.upper() in ("RKS", "UKS"):
        lines.append("    _dft = dft")
    else:
        lines.append("    _dft = None")
    lines.append("")
    lines += _emit_atomic_writer()
    lines += _emit_build_mol(struct, cfg)
    lines += _emit_frozen_mask()
    lines += _emit_initial_state()
    lines += _emit_equilibrium_scf(cfg, struct)
    lines += _emit_gpu_coverage_probe(cfg)
    lines += _emit_hessian_block(cfg)
    # Shared helpers for L3 and L4: both phases run SCFs at displaced
    # geometries via _build_mf_at(coords).  Emit ONCE whenever
    # either L3 or L4 will run -- previously _build_mf_at lived
    # inside the Raman block, so `compute_raman=False` with
    # `es_mode_selection != "skip"` crashed the script with NameError.
    needs_displaced_scf = cfg.compute_raman or cfg.es_mode_selection != "skip"
    if needs_displaced_scf:
        lines += _emit_displaced_scf_helpers(cfg)
    if cfg.compute_raman:
        lines += _emit_raman_block(cfg)
    if cfg.es_mode_selection != "skip":
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
    out.append("Recommended layout (one job per directory, see")
    out.append("docs/protocols/job-layout.md):")
    out.append(f"    projects/<project>/spectrum/<structure>/{cfg.job_name}.spectra.py")
    out.append("Canonical topics: optimization, frequency, spectrum,")
    out.append("                  transport, single-point, scan.")
    out.append("")
    out.append("Outputs (atomic-replace at each phase boundary):")
    out.append(f"    {cfg.job_name}.spectra.json     -- typed SpectraResults")
    out.append("                                          (see spec § 5 / § 6)")
    out.append("")
    if cfg.compute_ir:
        out.append("*** IR INTENSITY SCAFFOLD -- NOT YET VALIDATED ***")
        out.append("    `ir_intensity_km_mol` values in the JSON are computed")
        out.append("    from finite-difference dipole derivatives + the")
        out.append("    Gaussian/ORCA 42.2561 km/mol per (D/Å)²/amu prefactor.")
        out.append("    The math is textbook, but absolute magnitudes have NOT")
        out.append("    been cross-checked against an external code the way")
        out.append("    Raman was (see docs/tabs/spectra/spec.md § 12.1 for")
        out.append("    the Raman validation; § 13.1 for IR validation status).")
        out.append("    Use for relative IR intensities + qualitative work;")
        out.append("    quote absolute values only with the caveat.")
        out.append("    For CHARGED molecules (charge != 0) IR intensities")
        out.append("    are origin-dependent and physically ill-defined; the")
        out.append("    values here may be contaminated by origin-shift terms.")
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
    # _mb_outfile: ALL output paths resolve relative to the SCRIPT
    # directory, NOT the process cwd.  See the matching helper in
    # ``molbuilder/pyscf/input.py``.  PySCF / geomeTRIC may chdir
    # mid-run (esp. during optimisation), and the .run.sh wrapper's
    # chdir doesn't protect users who invoke the script directly
    # from a different cwd.  Resolving against ``__file__`` makes
    # the script land its outputs next to itself, regardless.
    out.append("from pathlib import Path as _MB_Path")
    out.append("_MB_SCRIPT_DIR = _MB_Path(__file__).resolve().parent")
    out.append("def _mb_outfile(name):")
    out.append("    p = _MB_Path(name)")
    out.append("    return str(p if p.is_absolute() else _MB_SCRIPT_DIR / p)")
    out.append("JSON_PATH      = _mb_outfile(JOB + '.spectra.json')")
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
    out.append(f"USE_GPU                    = {bool(cfg.use_gpu)!r}  "
               f"# try gpu4pyscf at runtime; fall back to CPU if it "
               f"isn't installed")
    out.append("")
    out.append("# Frozen-atom mask (UNION of element + residue + explicit).")
    out.append("# The runtime block computes the final FREE_ATOM_IDXS from")
    out.append("# this triplet against the molecule built below.")
    out.append(f"FROZEN_ELEMENTS            = {list(cfg.frozen_elements)!r}")
    out.append(f"FROZEN_RESIDUE_NAMES       = {list(cfg.frozen_residue_names)!r}")
    out.append(f"FROZEN_INDICES_USER        = {list(cfg.frozen_indices)!r}  "
               f"# 0-based")
    out.append("")
    out.append("# Spectrum knobs.")
    out.append(f"COMPUTE_RAMAN              = {bool(cfg.compute_raman)!r}")
    out.append(f"COMPUTE_IR                 = {bool(cfg.compute_ir)!r}  "
               f"# scaffold; values not yet validated against external code")
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
    out.append("# Verified entries live in docs/science/references.bib.")
    out.append(f"BIBLIOGRAPHY_KEYS          = {list(bibliography_keys)!r}")
    out.append("")
    out.append("# A snapshot of the SpectraConfig that produced this script,")
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
    out.append("# Methods-preview text (verbatim what the UI showed at render).")
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
    out.append("#  Array bridge:  GPU (CuPy) <-> CPU (NumPy)")
    out.append("# ============================================================")
    out.append("# On GPU runs (gpu4pyscf) attributes such as mf.mo_energy,")
    out.append("# mf.mo_occ, mf.Hessian().kernel() come back as CuPy arrays.")
    out.append("# Modern CuPy refuses implicit conversion via __array__ and")
    out.append("# raises TypeError; downstream code (pyscf.hessian.thermo,")
    out.append("# np.linalg.eigh, np.where, json serialisation) is all CPU.")
    out.append("# _as_numpy() does the explicit .get() round-trip ONCE at the")
    out.append("# crossing point and is a no-op for NumPy arrays / lists /")
    out.append("# scalars, so the same code runs unchanged on CPU and GPU.")
    out.append("def _as_numpy(x):")
    out.append("    '''Coerce a CuPy or NumPy array (or list / scalar) to NumPy.'''")
    out.append("    # Detect CuPy by module name to avoid an unconditional")
    out.append("    # `import cupy` (which would fail in pure-CPU envs).")
    out.append("    if type(x).__module__.startswith('cupy'):")
    out.append("        return x.get()")
    out.append("    return np.asarray(x)")
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
    # ECP: shared resolver with Build's PySCF generator
    # (chemistry.resolve_pyscf_ecp).  For Z > 36 + non-def2 basis,
    # auto-picks "lanl2dz"; for def2-* basis, returns None because
    # def2-* bundles its own Stuttgart ECP.  An explicit user value
    # (cfg.ecp = "lanl2dz" / dict / "none") wins.  See chemistry.py
    # for the full decision rule.
    from ..chemistry import resolve_pyscf_ecp
    ecp_chosen = resolve_pyscf_ecp(struct, cfg.ecp, cfg.basis)
    out.append("mol = gto.M(")
    out.append("    atom       = [[a[0], (a[1], a[2], a[3])] for a in ATOMS],")
    out.append("    basis      = BASIS,")
    if ecp_chosen is not None:
        # str -> emit as quoted literal; dict -> emit as Python dict
        # literal so PySCF sees the per-element mapping, not a string
        # containing braces.
        if isinstance(ecp_chosen, str):
            out.append(f"    ecp        = {ecp_chosen!r},")
        else:
            out.append(f"    ecp        = {dict(ecp_chosen)!r},")
    out.append("    verbose    = VERBOSE,")
    out.append("    max_memory = MAX_MEMORY_MB,")
    out.append("    unit       = 'Angstrom',")
    out.append(f"    charge     = {int(cfg.charge)},")
    out.append(f"    spin       = {int(cfg.spin)},   # 2S = # unpaired electrons")
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
    out.append("# Union of three rules; an atom is FROZEN if it matches any:")
    out.append("#   1. its element is in FROZEN_ELEMENTS")
    out.append("#   2. its 0-based index is in FROZEN_INDICES_USER")
    out.append("#   3. (residue freezing is no-op without PDB info -- the")
    out.append("#       molbuilder layer that emitted this script would have")
    out.append("#       resolved residue names to indices already.)")
    out.append("_frozen = set(int(i) for i in FROZEN_INDICES_USER if 0 <= int(i) < N_ATOMS)")
    out.append("for _i, _el in enumerate(ELEMENTS):")
    out.append("    if _el in FROZEN_ELEMENTS:")
    out.append("        _frozen.add(_i)")
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
    out.append("import hashlib")
    out.append("_xyz_lines = [f'{N_ATOMS}', f'{JOB}']")
    out.append("for _el, (_x, _y, _z) in zip(ELEMENTS, [(a[1], a[2], a[3]) for a in ATOMS]):")
    out.append("    _xyz_lines.append(f'{_el:<3s} {_x:14.8f} {_y:14.8f} {_z:14.8f}')")
    out.append("STRUCTURE_HASH = 'sha256:' + hashlib.sha256(")
    out.append("    '\\n'.join(_xyz_lines).encode('utf-8')")
    out.append(").hexdigest()")
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
    out.append("    'phase_raman':               PHASE_EMPTY,")
    out.append("    'phase_es':                  PHASE_EMPTY,")
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


def _emit_equilibrium_scf(cfg: SpectraConfig, struct: Structure) -> List[str]:
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
        # _dft is gpu4pyscf.dft when USE_GPU AND the import succeeded;
        # plain pyscf.dft otherwise.  Same RKS / UKS class names in
        # both, so the rest of the SCF setup is identical.
        out.append(f"mf = _dft.{scf_class}(mol)")
        out.append("mf.xc = FUNCTIONAL")
        out.append("if DISPERSION and DISPERSION.lower() != 'none':")
        out.append("    mf.disp = DISPERSION")
        out.append("mf.grids.level = GRID_LEVEL")
    else:
        out.append(f"mf = _scf.{scf_class}(mol)")
    out.append("if DENSITY_FIT:")
    out.append("    mf = mf.density_fit()")
    out.append("mf.conv_tol  = SCF_CONV_TOL")
    out.append("mf.max_cycle = SCF_MAX_CYCLE")
    # Hard-SCF hint when an open-shell metal is present.  Commented
    # template -- discoverable without being prescriptive; the user
    # uncomments + tunes if the equilibrium SCF won't converge.
    from ..chemistry import detect_open_shell_metals
    _metals = detect_open_shell_metals(struct)
    if _metals:
        out.append("# Hard SCF (typical for open-shell metals like "
                   f"{', '.join(_metals)}):")
        out.append("# Uncomment to apply a virtual-orbital level shift "
                   "(Eh).  Typical 0.1-0.3;")
        out.append("# helps when the HOMO-LUMO gap is small / open-shell "
                   "mixing causes oscillation.")
        out.append("# mf.level_shift = 0.2")
    out.append("E_eq = mf.kernel()")
    out.append("if not mf.converged:")
    out.append("    raise SystemExit(")
    out.append("        f'SCF did not converge (E={E_eq!r}); '")
    out.append("        f'increase scf_max_cycle or revisit '")
    out.append("        f'the input geometry'")
    out.append("    )")
    out.append("MO_ENERGIES_EQ = _as_numpy(mf.mo_energy).copy()")
    out.append("# HOMO index: highest occupied MO.  For UHF/UKS the mo_occ")
    out.append("# is 2-D (alpha, beta) -- we sum to total occupancy and find")
    out.append("# the highest level with occupancy > 0.  For RHF/RKS it's 1-D.")
    out.append("_occ = _as_numpy(mf.mo_occ)")
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
    out.append("    # Geometry is included so the spectra-tab UI")
    out.append("    # can animate vibrational modes directly from")
    out.append("    # the loaded results -- no need for the user")
    out.append("    # to keep the source XYZ around.")
    out.append("    'elements':       list(ELEMENTS),")
    out.append("    'positions_ang':  [[a[1], a[2], a[3]] for a in ATOMS],")
    out.append("}")
    out.append("_atomic_write_json(state, JSON_PATH)")
    out.append("print(f'Equilibrium SCF: E = {E_eq:.10f} Ha; HOMO index = {HOMO_IDX}')")
    out.append("")
    return out


# --------------------------------------------------------------------- #
# Hessian / L2                                                          #
# --------------------------------------------------------------------- #


def _emit_gpu_coverage_probe(cfg: SpectraConfig) -> List[str]:
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
    out.append("# Rather than hard-coding which (METHOD, stage) pairs work,")
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
    out.append("#                              harmonic_analysis (CPU-only) gets")
    out.append("#                              a NumPy array.")
    out.append("#   _GPU_HAS_HESSIAN False  -> rebuild mf on CPU and run Hessian")
    out.append("#                              there.  Costs one extra SCF but is")
    out.append("#                              the only path when gpu4pyscf does")
    out.append("#                              not cover Hessian for this SCF type.")
    out.append("if _GPU_HAS_HESSIAN or not _USING_GPU:")
    out.append("    HESS = _as_numpy(mf.Hessian().kernel())")
    out.append("else:")
    out.append("    print('  rebuilding mf on CPU for the Hessian step')")
    out.append("    _mf_cpu_for_hess = _build_mf_at(COORDS_EQ_ANG, force_cpu=True)")
    out.append("    HESS = _as_numpy(_mf_cpu_for_hess.Hessian().kernel())")
    out.append("# HESS shape: (n_atoms, n_atoms, 3, 3) in Hartree / Bohr².")
    out.append("")
    out.append("# ------------------------------------------------------------")
    out.append("# Two normal-mode arrays are computed and used DIFFERENTLY:")
    out.append("#")
    out.append("#   NORM_MODES_CANONICAL  (n_modes, N_FREE, 3)")
    out.append("#       Cartesian normal modes L_cart in the *canonical*")
    out.append("#       mass-weighted normalisation: sum_k m_k |L_cart_k|^2 = 1")
    out.append("#       (mass in atomic units).  This is the form the standard")
    out.append("#       Placzek Raman-activity formula expects.  The 45 a^2 +")
    out.append("#       7 gamma^2 scalar comes out in (a.u. polarizability)² /")
    out.append("#       (Å² · amu), which is rescaled by BOHR_TO_ANG**6 in")
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
    out.append("def _signed_wavenumber(w):")
    out.append("    '''Normalise a per-mode wavenumber to signed real cm⁻¹.")
    out.append("")
    out.append("    PySCF's harmonic_analysis may emit ``freq_wavenumber`` in")
    out.append("    one of two shapes for imaginary modes:")
    out.append("      * signed real:        w = -500.0 for a 500i mode")
    out.append("      * imaginary complex:  w = 0 + 500j for the same")
    out.append("    Whichever shape is in hand, we want a single convention")
    out.append("    downstream: a negative real for imaginary modes, positive")
    out.append("    real for real modes.  Then HAS_IMAG falls out as ``f < 0``")
    out.append("    and matches the partial-Hessian path below.'''")
    out.append("    if hasattr(w, 'imag') and abs(w.imag) > 0:")
    out.append("        return -abs(float(w.imag))")
    out.append("    return float(w.real if hasattr(w, 'real') else w)")
    out.append("")
    out.append("if N_FREE == N_ATOMS:")
    out.append("    # ----- All-free path -----")
    out.append("    # PySCF's harmonic_analysis projects out the 6 (or 5 for")
    out.append("    # linear molecules) translation+rotation modes from the")
    out.append("    # mass-weighted Hessian internally, then returns:")
    out.append("    #   freq_wavenumber : per-mode wavenumber (real or complex,")
    out.append("    #                     see _signed_wavenumber above)")
    out.append("    #   norm_mode       : (n_modes, N_ATOMS, 3) Cartesian normal")
    out.append("    #                     modes in the canonical mass-weighted")
    out.append("    #                     unit-norm convention.")
    out.append("    _ha = _mb_thermo.harmonic_analysis(mol, HESS)")
    out.append("    FREQ_CM1 = np.asarray([_signed_wavenumber(w)")
    out.append("                            for w in _ha['freq_wavenumber']])")
    out.append("    HAS_IMAG = [bool(f < 0) for f in FREQ_CM1]")
    out.append("    NORM_MODES_CANONICAL = _as_numpy(_ha['norm_mode'])")
    out.append("else:")
    out.append("    # ----- Partial-Hessian path (frozen atoms anchor the system) -----")
    out.append("    # No translation/rotation projection: the frozen atoms")
    out.append("    # break full T/R invariance, so the 6 (or 5) zero-frequency")
    out.append("    # modes the all-free path projects out simply do not exist")
    out.append("    # here.  All 3·N_FREE eigenvalues are physical.")
    out.append("    _free_idx = np.asarray(FREE_ATOM_IDXS, dtype=int)")
    out.append("    # Slice HESS (N_atoms × N_atoms × 3 × 3) down to the")
    out.append("    # (N_FREE × N_FREE × 3 × 3) block coupling free atoms.")
    out.append("    _hess_free = HESS[_free_idx][:, _free_idx]")
    out.append("    # Reshape (N_FREE, N_FREE, 3, 3) -> (3*N_FREE, 3*N_FREE)")
    out.append("    # with axis order (atom_i, dir_i, atom_j, dir_j) so the flat")
    out.append("    # 2-D index k = 3*atom + direction.")
    out.append("    _h2 = _hess_free.transpose(0, 2, 1, 3).reshape(")
    out.append("        3 * N_FREE, 3 * N_FREE)")
    out.append("    _masses_free = MASSES_AU[_free_idx]")
    out.append("    _msqrt_inv = 1.0 / np.sqrt(_masses_free)")
    out.append("    # Mass-weight: H_ij <- H_ij / sqrt(m_i * m_j) for each 3x3")
    out.append("    # atom-atom block, by broadcasting the per-atom 1/sqrt(m)")
    out.append("    # vector (length 3*N_FREE after np.repeat) onto both axes.")
    out.append("    _weights = np.repeat(_msqrt_inv, 3)")
    out.append("    _hmw = _h2 * np.outer(_weights, _weights)")
    out.append("    # Symmetrise to clean up the numerical asymmetry that")
    out.append("    # accumulates in the analytic Hessian (small enough that")
    out.append("    # eigh would silently average anyway; doing it here makes")
    out.append("    # the result deterministic across BLAS implementations).")
    out.append("    _hmw = 0.5 * (_hmw + _hmw.T)")
    out.append("    _eigvals, _eigvecs = np.linalg.eigh(_hmw)")
    out.append("    # Frequency in atomic units of frequency:")
    out.append("    #     ω_au = sign(λ) · sqrt(|λ|)")
    out.append("    # negative ω_au for an imaginary mode (negative eigenvalue")
    out.append("    # of the mass-weighted Hessian).  Then convert a.u. -> cm⁻¹.")
    out.append("    _omega_au = np.sign(_eigvals) * np.sqrt(np.abs(_eigvals))")
    out.append("    FREQ_CM1  = _omega_au * CM1_PER_AU_FREQ")
    out.append("    HAS_IMAG  = [bool(f < 0) for f in FREQ_CM1]")
    out.append("    # Convert each eigenvector L_mw of the mass-weighted Hessian")
    out.append("    # back to a Cartesian normal mode L_cart via")
    out.append("    #     L_cart_k = L_mw_k / sqrt(m_k)")
    out.append("    # which automatically preserves the canonical mass-weighted")
    out.append("    # unit norm  Σ_k m_k |L_cart_k|^2 = 1  (verifiable: substitute")
    out.append("    # L_cart and the m_k cancels against the 1/sqrt(m_k)^2 to")
    out.append("    # give Σ_k |L_mw_k|^2, which is 1 because eigh-eigenvectors")
    out.append("    # of a symmetric matrix have unit Euclidean norm).")
    out.append("    NORM_MODES_CANONICAL = np.zeros((len(_eigvals), N_FREE, 3))")
    out.append("    for _k in range(len(_eigvals)):")
    out.append("        _L_mw   = _eigvecs[:, _k].reshape(N_FREE, 3)")
    out.append("        _L_cart = _L_mw * _msqrt_inv[:, None]")
    out.append("        NORM_MODES_CANONICAL[_k] = _L_cart")
    out.append("")
    out.append("# Derive the DISPLAY form (max(|L|)=1 per mode) from the canonical")
    out.append("# form.  Both forms ship in the JSON under explicit names so")
    out.append("# consumers don't have to compute one from the other.")
    out.append("NORM_MODES_DISPLAY = np.zeros_like(NORM_MODES_CANONICAL)")
    out.append("for _k in range(NORM_MODES_CANONICAL.shape[0]):")
    out.append("    _max = float(np.max(np.abs(NORM_MODES_CANONICAL[_k])))")
    out.append("    if _max > 0:")
    out.append("        NORM_MODES_DISPLAY[_k] = NORM_MODES_CANONICAL[_k] / _max")
    out.append("")
    out.append("# Build the modes payload, one record per mode.  The JSON keys")
    out.append("# below are the SCHEMA_VERSION=2 contract:")
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
    out.append("    # Defensive reshape: PySCF's all-free path returns")
    out.append("    # norm_mode shape (N_ATOMS, 3) which equals (N_FREE, 3)")
    out.append("    # here because N_FREE == N_ATOMS in that branch -- so the")
    out.append("    # reshape is a no-op for normal-shaped returns.  Triggers")
    out.append("    # only if a future PySCF version returns the flat")
    out.append("    # (N_ATOMS * 3,) shape some upstream code uses.")
    out.append("    if _L_canonical.shape[0] != N_FREE:")
    out.append("        _L_canonical = _L_canonical.reshape(-1, 3)")
    out.append("        _L_display   = _L_display.reshape(-1, 3)")
    out.append("    # Serialise each eigenvector exactly once -- _filter_finite")
    out.append("    # converts NaN/Inf to JSON-safe None and returns a plain list.")
    out.append("    _evec_canon_json = _filter_finite(_L_canonical)")
    out.append("    _evec_disp_json  = _filter_finite(_L_display)")
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
    out.append("    _mol_new.atom = [[ELEMENTS[_i], tuple(coords[_i])]")
    out.append("                     for _i in range(N_ATOMS)]")
    out.append("    _mol_new.unit = 'Angstrom'")
    out.append("    _mol_new.build()")
    out.append("    # Pick the right dft / scf module for this call.  _dft / _scf")
    out.append("    # are gpu4pyscf when _USING_GPU else stock pyscf; force_cpu")
    out.append("    # overrides to stock pyscf regardless.")
    out.append("    _dft_mod = dft if force_cpu else _dft")
    out.append("    _scf_mod = scf if force_cpu else _scf")
    out.append("    if METHOD.upper() in ('RKS', 'UKS'):")
    out.append("        _cls = _dft_mod.RKS if METHOD.upper() == 'RKS' else _dft_mod.UKS")
    out.append("        _mf2 = _cls(_mol_new)")
    out.append("        _mf2.xc = FUNCTIONAL")
    out.append("        if DISPERSION and DISPERSION.lower() != 'none':")
    out.append("            _mf2.disp = DISPERSION")
    out.append("        _mf2.grids.level = GRID_LEVEL")
    out.append("    else:")
    out.append("        _cls = _scf_mod.RHF if METHOD.upper() == 'RHF' else _scf_mod.UHF")
    out.append("        _mf2 = _cls(_mol_new)")
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


def _emit_ir_projection() -> List[str]:
    """Per-mode IR intensity (km/mol) from the dipole-moment
    derivative collected in the Raman FD loop.

    This block emits inside ``_emit_raman_block`` after the Raman
    activity loop has run -- it relies on ``DMU_DR`` and the
    canonical normal modes already being in scope.

    SCIENTIFIC VALIDATION STATUS: NOT YET VALIDATED against an
    external code (Gaussian / ORCA / Turbomole).  The projection
    math + Gaussian/ORCA km/mol prefactor are textbook, but the
    absolute magnitudes have not been cross-checked the way the
    Raman path was (see ``docs/tabs/spectra/spec.md § 12.1``).
    Use for relative IR intensities and qualitative analysis;
    quote absolute values only with the caveat.
    """
    out: List[str] = []
    out.append("")
    out.append("# ----- IR (scaffold; absolute magnitudes not yet validated) -----")
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
    out.append("# see docs/tabs/spectra/spec.md § 13.1 for validation status.")
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


def _emit_raman_block(cfg: SpectraConfig) -> List[str]:
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
    out.append("    # _mf is force_cpu=True for polarizability (gpu4pyscf")
    out.append("    # doesn't expose analytic CPHF), so this is CPU NumPy --")
    out.append("    # the bridge is defensive in case the call path changes.")
    out.append("    return _as_numpy(_mf.Polarizability().polarizability())")
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
        out.append("    return _as_numpy(_mf.dip_moment(unit='Debye', verbose=0))")
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
    out.append("# Polarizability needs CPU (gpu4pyscf doesn't expose")
    out.append("# analytic CPHF polarizability) AND non-DF (pyscf-properties")
    out.append("# doesn't have a DF implementation).  These two flags are")
    out.append("# orthogonal but both kick in for the Raman FD step only.")
    out.append("_mf_nodf_eq = _build_mf_at(COORDS_EQ_ANG, density_fit=False,")
    out.append("                           force_cpu=True)")
    out.append("alpha_eq = _polarizability(_mf_nodf_eq)")
    out.append("")
    out.append("# Build dα/dR_kα by central difference for each free-atom Cartesian.")
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
    out.append("# global: multiply by (Bohr/Å)^6 = BOHR_TO_ANG^6 ≈ 0.02197.  We")
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
    out.append("    The caller multiplies by BOHR_TO_ANG**6 to convert to the")
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
    out.append("# (Bohr/Å)^6 = BOHR_TO_ANG^6  ≈ 0.02197 .")
    out.append("_RAMAN_AU2_TO_A4AMU = BOHR_TO_ANG ** 6")
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
    out.append("_atomic_write_json(state, JSON_PATH)")
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
    out.append("def _mo_window(_mf2):")
    out.append("    '''Slice the MO array to [HOMO-N, LUMO+M] around the")
    out.append("    equilibrium HOMO.  At a displaced geometry orbitals can")
    out.append("    swap; the spec accepts that and lets downstream")
    out.append("    EPC analysis handle alignment.'''")
    out.append("    _mos = _as_numpy(_mf2.mo_energy)")
    out.append("    _occ = _as_numpy(_mf2.mo_occ)")
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
    out.append("    # (METHOD / FUNCTIONAL / BASIS / GRID_LEVEL / DENSITY_FIT);")
    out.append("    # _build_mf_at handles the gpu4pyscf vs CPU branching")
    out.append("    # internally via the _USING_GPU flag.")
    out.append("    _mfp = _build_mf_at(_disp_plus)")
    out.append("    _mfm = _build_mf_at(_disp_minus)")
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

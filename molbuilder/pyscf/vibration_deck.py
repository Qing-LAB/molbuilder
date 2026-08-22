"""The vibration calculation's deck — PySCF, on the framework's seam.

Contract: ``docs/archive/2026-08-20-spectra-migration-plan.md`` § 2 (the rulings of
2026-08-20) + `script-preparation.md` § 4 (the seam this serves).

WHAT THIS IS.  ``vibration_spec(struct, cfg, stage_token)`` returns the
:class:`~molbuilder.script_emit.DeckSpec` for ``calculation = "vibration"``:
one deck, one run — the relaxation as its mandatory first act (geomeTRIC
straight into the Hessian, in-process; D3's final form), the harmonic
analysis with the frozen-atom partial-Hessian path, IR and Raman as
INDEPENDENT toggles over shared machinery, RRHO thermochemistry into the
artifact's v5 ``thermo`` block, the per-mode electronic-structure loop,
and the phase-writing ``.spectra.json`` the viewer live-watches.

THE LIFT (the plan's § 2: *"the old code transitions onto the new
framework"*).  The proven science emitters are IMPORTED from
``spectra/pyscf_script.py`` and composed here — a move, not a rewrite.
P3 (2026-08-21) deleted the old module; the surviving emitters live in
this package (`vibration_emitters.py`) and this file is their one
composer.

THE LIFT BOUNDARY'S ADAPTER (kept past P3, deliberately).  P3's plan had
it dissolving with the old module; what P3 actually showed is that the
adapter IS the seam's right shape — the kind's science and the emitters
both read one view, so a check and the deck it checks cannot disagree
about a value (`science_view`'s contract).  The lifted emitters read
four names the shared config spells differently or does not hold:
``charge`` (the shared item is ``net_charge``) and the three frozen
SELECTORS — which are structure-side facts here (the sidecar three-stage
contract, `engines/overview.md` § 3), so :class:`VibrationConfigView` feeds the
frozen indices from the STRUCTURE's own label store and everything else
straight through.  Modifying the old emitters instead would be repatching
the old path — the direction the rulings forbid.

NEW BLOCKS (authored here, not lifted): the tracked relaxation
(``phase_relaxation`` + step/max-force progress, complete-by-assertion
under ``already_relaxed`` with the post-SCF gradient check that WARNS and
never refuses), the thermochemistry (headline at (T, P), the documented
default temperature grid the viewer's G/H/S curves draw, and the regime
word — ``rrho`` for a free molecule, ``vibrational-only`` when atoms are
frozen: an anchored molecule does not rotate), and the IR-only
displacement loop (a dipole read per displacement — far cheaper than
Raman; the old ``compute_ir``-requires-``compute_raman`` coupling was an
implementation artifact and retires with the old entry point).
"""
from __future__ import annotations

from typing import List, Optional

from .. import script_emit as _sc
from ..structure import FROZEN_LABEL, Structure

# The lifted emitters — the old generator's proven blocks, composed anew.
from .vibration_emitters import (
    pyscf_methods_fragment,
    _emit_atomic_writer,
    _emit_build_mol,
    _emit_constants,
    _emit_displaced_scf_helpers,
    _emit_equilibrium_scf,
    _emit_es_loop,
    _emit_final_summary,
    _emit_frozen_mask,
    _emit_gpu_coverage_probe,
    _emit_header_docstring,
    _emit_hessian_block,
    _emit_imports,
    _emit_ir_projection,
    _emit_initial_state,
    _emit_raman_block,
)

#: The viewer's G/H/S curves run over this grid (plan § 2b: a documented
#: presentation default, not a scientific knob — the headline (T, P) items
#: are the knobs).  50–1500 K, 30 points: wide enough to show the trend,
#: cheap enough to be free (arithmetic over frequencies already in hand).
_THERMO_GRID_K = "np.linspace(50.0, 1500.0, 30)"


class VibrationConfigView:
    # Public because it is the SHAPE, not an implementation detail: the
    # emitters and the Methods fragment annotate against it now that
    # `SpectraConfig` -- a 33-field dataclass nothing ever constructed --
    # is retired.  Named `_LiftView` while it was private to this file.
    """The lift boundary's adapter — see the module header (kept past P3
    as the one view both the emitters and the kind's science read).

    ``dataclasses.asdict`` works on it (the lifted constants emitter
    records the config into the artifact that way) by advertising the
    REAL config's fields — which is also the right record: the artifact's
    ``config`` should say what the described PySCFConfig held, not the
    adapter's four bridged spellings."""

    def __init__(self, cfg, struct: Structure):
        from ..config.pyscf import PySCFConfig
        # asdict() asks type(self) for __dataclass_fields__ and getattrs
        # each name off the instance -- __getattr__ forwards to the real
        # config, so the recorded dict is the config's own.
        type(self).__dataclass_fields__ = PySCFConfig.__dataclass_fields__
        type(self).__dataclass_params__ = PySCFConfig.__dataclass_params__
        self._cfg = cfg
        frozen = ()
        try:
            frozen = tuple(int(i) for i in
                           (struct.regions or {}).get(FROZEN_LABEL, ()))
        except Exception:  # noqa: BLE001 -- absent store reads as none frozen
            frozen = ()
        # Indices only.  `frozen_elements` / `frozen_residue_names` were
        # set here as empty lists and read by one unreachable branch in the
        # Methods renderer; both went with `SpectraConfig` (2026-08-22).
        self.frozen_indices = list(frozen)
        # THE one charge rule (chemistry.resolve_net_charge), resolved
        # once at the lift boundary: explicit wins, 0 included; only an
        # unset charge runs the phosphate auto-detection.  `net_charge
        # or 0` silently dropped that detection, so a nucleic-acid
        # vibration with charge unset was a DIFFERENT calculation than
        # its optimization sibling.
        from ..chemistry import resolve_net_charge
        self._charge = int(resolve_net_charge(struct, cfg.net_charge))

    @property
    def charge(self) -> int:
        return self._charge

    @property
    def max_memory_mb(self) -> int:
        # The lifted constants emitter takes an int only; the shared
        # item's UNSET means "the engine's own default", which for PySCF
        # is 4000 MB (gto.M's max_memory).  Supplied at the lift boundary
        # rather than invented in the deck.  (P3 kept the adapter; teaching
        # the emitter the unset state remains open and small.)
        v = self._cfg.max_memory_mb
        return int(v) if v else 4000

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def science_view(cfg, struct: Structure) -> "VibrationConfigView":
    """The vibration deck's config view, for the KIND's validation step
    (validation/__init__._validate_vibration_kind).  One adapter, two
    readers: the emitters and the science speak the same vocabulary,
    so a check and the deck it checks cannot disagree about a value."""
    return VibrationConfigView(cfg, struct)


def _vib_constants(cfg) -> List[str]:
    """The vibration deck's own constants, beside the lifted ones."""
    out = ["",
           "# ---- vibration-kind constants (framework deck) ----",
           f"ALREADY_RELAXED = {bool(cfg.already_relaxed)}",
           "# The relaxation's convergence criteria (the shared geometry",
           "# items; the vibration template defaults them to the tight",
           "# tier -- a frequency deserves a real stationary point).",
           f"GEOM_GMAX      = {float(cfg.geom_gmax)!r}",
           f"GEOM_GRMS      = {float(cfg.geom_grms)!r}",
           f"GEOM_DMAX      = {float(cfg.geom_dmax)!r}",
           f"GEOM_DRMS      = {float(cfg.geom_drms)!r}",
           f"GEOM_MAX_STEPS = {int(cfg.geom_max_steps)}",
           f"GEOM_ETOL      = {float(cfg.geom_etol)!r}",
           f"THERMO_T_K     = {float(cfg.temperature_K)!r}",
           f"THERMO_P_ATM   = {float(cfg.pressure_atm)!r}",
           f"THERMO_T_GRID  = {_THERMO_GRID_K}",
           "# The framework runs a deck from its ATTEMPT directory and",
           "# links the deck text from the bundle root --",
           "# resolve(__file__) follows the link, so a script-dir rule",
           "# would write the artifact one level up (found by the first",
           "# live E2E).  Outputs belong where the run ran: the cwd.",
           "# THE one definition (the lifted script-dir form it once",
           "# shadowed is gone).",
           "def _mb_outfile(name):",
           "    from pathlib import Path as _P",
           "    _p = _P(name)",
           "    return str(_p if _p.is_absolute() else _P.cwd() / _p)",
           "JSON_PATH = _mb_outfile(JOB + '.spectra.json')",
           ]
    return out


def _vib_state_init() -> List[str]:
    """v5's state keys + the first write: the relaxation is a TRACKED
    step from the first byte the viewer polls."""
    return [
        "",
        "# v5: the relaxation is a tracked phase (spectra-migration plan",
        "# D4); the thermo block rides the same schema bump (D2).",
        "state['phase_relaxation'] = 'empty'",
        "state['relaxation'] = {'enabled': (not ALREADY_RELAXED),",
        "                       'already_relaxed': ALREADY_RELAXED,",
        "                       'n_steps': 0, 'max_force_eh_a': None,",
        "                       'converged': None}",
        "state['thermo'] = {}",
        "_atomic_write_json(state, JSON_PATH)",
    ]


def _vib_relax_block(cfg, stage_token=None) -> List[str]:
    """The mandatory precondition (D3's final form): geomeTRIC, in-process,
    BEFORE the equilibrium SCF — which then runs on the relaxed geometry
    unchanged, because this block rebinds ``mol`` and ``COORDS_EQ_ANG``.
    Every step writes the artifact, so the viewer's chip shows
    'step N, max force F' ticking down.

    Category 3 of the integration plan (2026-08-21): the workflow knobs
    are honored here — ``geom_etol`` joins the convergence dict,
    ``on_nonconvergence='continue'`` wraps the call in the SAME retry
    budget the optimization deck uses, ``write_trajectory`` hands
    geomeTRIC its streaming-XYZ prefix, ``write_molwatch_log`` composes
    the molwatch hooks beside the artifact callback, and the two
    ``save_*_xyz`` flags write their standalone files.  ``optimizer``
    is geomeTRIC by REFUSAL upstream (the kind validator: pyberny is
    absent from the run environment, probed 2026-08-21, and its solver
    has no step callback for the tracked phase)."""
    _rung = f"_{stage_token}" if stage_token else ""
    out: List[str] = [
        "",
        "# ============================================================",
        "#  Phase 0: relaxation -- the measurement's precondition",
        "# ============================================================",
        "if not ALREADY_RELAXED:",
        "    print('=== Stage: relaxation (geomeTRIC) ===')",
        "    state['phase_relaxation'] = 'running'",
        "    _atomic_write_json(state, JSON_PATH)",
        "    from pyscf.geomopt.geometric_solver import optimize as _geom_opt",
        "    _mf_relax = _build_mf_at(COORDS_EQ_ANG)",
    ]
    if getattr(cfg, "scf_soscf", False):
        out += [
            "    # Second-order SCF at the relax site too (the § 7a role",
            "    # table): every geometry step's SCF runs the same Newton",
            "    # solver the equilibrium one does; DIIS/damp stop",
            "    # applying under it, by design.",
            "    _mf_relax = _mf_relax.newton()",
        ]
    if getattr(cfg, "write_molwatch_log", False):
        out += [
            "    # Live-watch: the relaxation phase streams the same",
            "    # molwatch log the optimization deck writes, so the",
            "    # browser's watcher covers this run's first phase too.",
            "    _mf_relax.callback = _molwatch.scf_cycle_hook",
        ]
    out += [
        "    def _relax_cb(envs):",
        "        # Tolerant by design: geomeTRIC's callback dict has varied",
        "        # across versions; a missing key must not kill the run.",
        "        try:",
        "            _g = envs.get('gradients')",
        "            _mx = (float(np.abs(np.asarray(_g)).max())",
        "                   if _g is not None else None)",
        "        except Exception:",
        "            _mx = None",
        "        state['relaxation']['n_steps'] += 1",
        "        if _mx is not None:",
        "            state['relaxation']['max_force_eh_a'] = _mx",
        "        _atomic_write_json(state, JSON_PATH)",
    ]
    if getattr(cfg, "write_molwatch_log", False):
        out += [
            "    def _relax_cb_both(envs):",
            "        _molwatch.opt_step_hook(envs)",
            "        _relax_cb(envs)",
        ]
        _cb = "_relax_cb_both"
    else:
        _cb = "_relax_cb"
    # The geomeTRIC keyword spellings come from the ONE mapping the
    # optimization deck's section uses (layout._GEOM_KWARG) -- this dict
    # was a hand-spelled second copy of five of its six rows.
    from .layout import _GEOM_KWARG, GEOMETRY_SECTION
    _conv_rows = [
        f"'{_GEOM_KWARG[_n]}': GEOM_{_n[5:].upper()}"
        for _n in GEOMETRY_SECTION.items if _n != "geom_max_steps"]
    out += [
        "    _conv = {" + ",\n             ".join(_conv_rows) + "}",
    ]
    _opt_kw = f"maxsteps=GEOM_MAX_STEPS, callback={_cb}"
    _frozen = list(getattr(cfg, "frozen_indices", []) or [])
    if _frozen:
        from ..engine_atom_index import geometric_atom_index
        _ids_1based = ",".join(str(geometric_atom_index(i)) for i in _frozen)
        out += [
            "    # Frozen atoms stay frozen through the pre-Hessian",
            "    # relaxation (frozen means frozen -- user ruling",
            "    # 2026-08-21; engines/pyscf.md § 7a role table).  geomeTRIC",
            "    # takes the set as a $freeze constraints file, the",
            "    # optimization deck's own mechanism; indices are 1-based",
            "    # there.  The SAME set is excluded from the Hessian below",
            "    # (partial Hessian), so the geometry the frequencies are",
            "    # computed at is one where the fixed atoms never moved.",
            f"    # Source: frozen set (0-based) = {_frozen!r}",
            "    _FROZEN_CONSTRAINTS_PATH = _mb_outfile(JOB "
            "+ '.constraints.txt')",
            "    with open(_FROZEN_CONSTRAINTS_PATH, 'w') as _fh:",
            "        _fh.write('$freeze\\n')",
            f"        _fh.write('xyz {_ids_1based}\\n')",
        ]
        _opt_kw += ", constraints=str(_FROZEN_CONSTRAINTS_PATH)"
    if getattr(cfg, "write_trajectory", False):
        out += [
            "    # geomeTRIC streams its own multi-frame XYZ under this",
            "    # prefix (<prefix>_optim.xyz) -- the same file the",
            "    # optimization deck's rungs write.",
        ]
        _opt_kw += f", prefix=str(_mb_outfile(JOB + '_geom{_rung}'))"
    # The on_nonconvergence policy is the RELAXATION's (its catalogue
    # help says so: what to do when geomeTRIC's criteria are not met).
    # proceed = accept the last geometry UNASSERTED; continue = retry
    # the same budget; halt = raise.  The mechanism is the same
    # `assert_convergence` kwarg the optimization deck uses.
    _policy = str(getattr(cfg, "on_nonconvergence", "halt") or "halt").lower()
    if _policy == "continue":
        _retries = int(getattr(cfg, "geom_continue_retries", 0) or 0)
        out += [
            f"    _budget = 1 + {_retries}",
            "    for _attempt in range(_budget):",
            "        try:",
            f"            mol = _geom_opt(_mf_relax, {_opt_kw},",
            "                            assert_convergence=True, **_conv)",
            "            break",
            "        except RuntimeError as _e:",
            "            if 'not converged' not in str(_e).lower():",
            "                raise            # a genuinely different error",
            "            if _attempt == _budget - 1:",
            "                raise            # exhausted -> halt",
            "            print(f'WARN: relaxation did not converge in '",
            "                  f'{GEOM_MAX_STEPS} steps; retrying '",
            "                  f'({_budget - 1 - _attempt} left)')",
            "    state['relaxation']['converged'] = True",
        ]
    elif _policy == "proceed":
        out += [
            f"    mol = _geom_opt(_mf_relax, {_opt_kw},",
            "                    assert_convergence=False, **_conv)",
            "    # UNASSERTED by policy: the geometry is whatever the step",
            "    # budget bought.  Recorded as unknown, never claimed True.",
            "    state['relaxation']['converged'] = None",
            "    state['relaxation']['warning'] = (",
            "        'on_nonconvergence=proceed: convergence was not '",
            "        'asserted; frequencies below are for the geometry the '",
            "        'step budget produced')",
        ]
    else:
        out += [
            f"    mol = _geom_opt(_mf_relax, {_opt_kw},",
            "                    assert_convergence=True, **_conv)",
            "    state['relaxation']['converged'] = True",
        ]
    if getattr(cfg, "save_optimized_xyz", False):
        out += [
            "    _save_xyz(mol, _mb_outfile(JOB + '_optimized.xyz'),",
            "              'Relaxed geometry (vibration deck)')",
        ]
    out += [
        "    COORDS_EQ_ANG = mol.atom_coords(unit='Angstrom')",
        "    state['phase_relaxation'] = 'complete'",
        "    _atomic_write_json(state, JSON_PATH)",
        "else:",
        "    # Complete-BY-ASSERTION: the user stated it; the gradient is",
        "    # still checked (after the equilibrium SCF below, where the",
        "    # mean field exists) and a warning -- never a refusal --",
        "    # carries the number.",
        "    state['phase_relaxation'] = 'complete'",
        "    _atomic_write_json(state, JSON_PATH)",
    ]
    return out


def _vib_gradient_check() -> List[str]:
    """The already-relaxed assertion's honesty check — placed AFTER the
    equilibrium SCF (the mean field exists there), warns with the numbers,
    never refuses: skipping was a deliberate choice."""
    return [
        "",
        "if ALREADY_RELAXED:",
        "    print('=== gradient check (already_relaxed asserted) ===')",
        "    try:",
        "        _g0 = _as_numpy(mf.nuc_grad_method().kernel())",
        "        _maxf = float(np.abs(_g0).max())",
        "        state['relaxation']['max_force_eh_a'] = _maxf",
        "        if _maxf > GEOM_GMAX * 10.0:",
        "            _w = (f'input geometry is not a stationary point '",
        "                  f'(max |dE/dR| = {_maxf:.2e} Eh/Bohr, vs the '",
        "                  f'gmax criterion {GEOM_GMAX:.1e}); frequencies '",
        "                  f'-- the low ones especially -- may be '",
        "                  f'unreliable.  already_relaxed was asserted, '",
        "                  f'so this run carries on.')",
        "            print('WARNING: ' + _w)",
        "            state['relaxation']['warning'] = _w",
        "        _atomic_write_json(state, JSON_PATH)",
        "    except Exception as _e:  # the check must never kill the run",
        "        print(f'gradient check skipped: {_e}')",
    ]


def _vib_thermo_block() -> List[str]:
    """RRHO thermochemistry into the v5 ``thermo`` block (D2's re-homing),
    from the mode list the hessian block just wrote — the one source both
    Hessian paths (full and partial) share.  Regime honesty: a free
    molecule gets full RRHO via PySCF's own ``thermo.thermo``; a
    frozen-atom system gets the VIBRATIONAL contributions only, stated —
    an anchored molecule does not rotate."""
    return [
        "",
        "# ============================================================",
        "#  Thermochemistry (v5 `thermo`; D2's re-homing)",
        "# ============================================================",
        "print('=== Stage: thermochemistry ===')",
        "_freqs_cm1 = np.array([m['frequency_cm1'] for m in",
        "                       state['modes'] if not m['has_imag']])",
        "_freqs_cm1 = _freqs_cm1[_freqs_cm1 > 0]",
        "_KB_EH = 3.166811563e-6        # Boltzmann, Eh/K",
        "_CM1_TO_EH = 1.0 / 219474.6313632",
        "def _vib_thermo_at(T):",
        "    '''ZPE + vibrational U, S at T (Eh, Eh/K) -- harmonic.'''",
        "    if not len(_freqs_cm1) or T <= 0:",
        "        return 0.0, 0.0, 0.0",
        "    _w = _freqs_cm1 * _CM1_TO_EH",
        "    _x = _w / (_KB_EH * T)",
        "    _x = np.clip(_x, 1e-12, 700.0)",
        "    _zpe = float(0.5 * _w.sum())",
        "    _u = float((_w / (np.exp(_x) - 1.0)).sum())",
        "    _s = float(_KB_EH * (( _x / (np.exp(_x) - 1.0)",
        "               - np.log1p(-np.exp(-_x))).sum()))",
        "    return _zpe, _u, _s",
        "_regime = 'rrho' if N_FREE == N_ATOMS else 'vibrational-only'",
        "_zpe0, _u0, _s0 = _vib_thermo_at(THERMO_T_K)",
        "_grid = {'temperatures_K': [], 'zpe_eh': [], 'u_vib_eh': [],",
        "         'h_eh': [], 's_eh_k': [], 'g_eh': []}",
        "state['thermo'] = {",
        "    'regime': _regime,",
        "    'temperature_K': THERMO_T_K, 'pressure_atm': THERMO_P_ATM,",
        "    'zpe_eh': _zpe0,",
        "    'note': ('full RRHO (rot+trans+vib) via pyscf thermo' if",
        "             _regime == 'rrho' else",
        "             'VIBRATIONAL contributions only: atoms are frozen, '",
        "             'so the gas-phase rotational/translational partition '",
        "             'functions do not apply (an anchored molecule does '",
        "             'not rotate); low-frequency modes make the entropy '",
        "             'soft even so'),",
        "}",
        "if _regime == 'rrho':",
        "    try:",
        "        _t_res = _mb_thermo.thermo(",
        "            mf, _ha['freq_au'], THERMO_T_K, THERMO_P_ATM * 101325.0)",
        "        state['thermo']['h_eh'] = float(_t_res['H_tot'][0])",
        "        state['thermo']['s_eh_k'] = float(_t_res['S_tot'][0])",
        "        state['thermo']['g_eh'] = float(_t_res['G_tot'][0])",
        "    except Exception as _e:",
        "        print(f'full-RRHO thermo unavailable ({_e}); '",
        "              'vibrational-only values recorded')",
        "for _T in THERMO_T_GRID:",
        "    _z, _u, _s = _vib_thermo_at(float(_T))",
        "    _h = float(E_eq) + _z + _u + _KB_EH * float(_T)",
        "    _grid['temperatures_K'].append(float(_T))",
        "    _grid['zpe_eh'].append(_z)",
        "    _grid['u_vib_eh'].append(_u)",
        "    _grid['h_eh'].append(_h)",
        "    _grid['s_eh_k'].append(_s)",
        "    _grid['g_eh'].append(_h - float(_T) * _s)",
        "state['thermo']['grid'] = _grid",
        "_atomic_write_json(state, JSON_PATH)",
    ]


def _vib_ir_only_block(cfg) -> List[str]:
    """IR without Raman — the decoupling the 2026-08-20 ruling asked for:
    a dipole read per ± displacement over the free Cartesians (far cheaper
    than Raman's response calculations), then the LIFTED per-mode IR
    projection verbatim, so the two paths' formula cannot differ."""
    out = [
        "",
        "# ============================================================",
        "#  Phase 3-IR: IR intensities (dipole finite differences)",
        "# ============================================================",
        "print('=== Stage: IR intensities (dipole FD) ===')",
        "state['phase_raman'] = 'complete'   # not requested; nothing owed",
        "_h_ang = RAMAN_FD_STEP_ANG   # ONE step for both dmu/dR paths",
        "DMU_DR = np.zeros((N_FREE, 3, 3))",
        "for _k, _atom in enumerate(FREE_ATOM_IDXS):",
        "    for _a in range(3):",
        "        _cp = np.array(COORDS_EQ_ANG, dtype=float)",
        "        _cm = np.array(COORDS_EQ_ANG, dtype=float)",
        "        _cp[_atom, _a] += _h_ang",
        "        _cm[_atom, _a] -= _h_ang",
        "        # _build_mf_at CONVERGES the SCF before returning (and",
        "        # halts if it cannot) -- a second kernel() here re-ran the",
        "        # whole SCF per displaced point, doubling the loop's cost",
        "        # for identical numbers.",
        "        _mfp = _build_mf_at(_cp)",
        "        _mfm = _build_mf_at(_cm)",
        "        _dp = _as_numpy(_mfp.dip_moment(unit='Debye', verbose=0))",
        "        _dm = _as_numpy(_mfm.dip_moment(unit='Debye', verbose=0))",
        "        DMU_DR[_k, _a, :] = (_dp - _dm) / (2.0 * _h_ang)",
        "modes_payload = state['modes']",
    ]
    # The projection emits ITS OWN per-mode loop (reading
    # NORM_MODES_CANONICAL, which the Hessian block defines on every
    # path) -- wrapping it in a second per-mode loop here ran the whole
    # projection once PER MODE: N² idempotent work whose outer loop
    # variable nothing read.
    out.extend(_emit_ir_projection())
    out += [
        "_atomic_write_json(state, JSON_PATH)",
    ]
    return out


def vibration_spec(struct: Structure, cfg, *,
                   stage_token: Optional[str] = None) -> "_sc.DeckSpec":
    """The vibration calculation's DeckSpec — the seam's answer for
    ``calculation = 'vibration'`` (PySCF first; the shape admits others)."""
    view = VibrationConfigView(cfg, struct)

    def _vib_deck(struct_, cfg_) -> str:
        # THE COMPOSITION MIRRORS THE OLD GENERATOR'S (render_spectra_script
        # 105-200) call for call -- the validation, the Methods prose, the
        # shared runtime/threading/GPU glue, the aliasing block -- with the
        # vibration insertions at their run positions: the kind's constants,
        # the v5 state, the hoisted displaced helpers (pure defs; the
        # relaxation's driver needs `_build_mf_at` early), the relaxation
        # before the equilibrium SCF, the gradient check after it, the
        # thermochemistry after the Hessian, and the IR-only arm.
        # NO validation call here -- deliberately.  The settings gate is
        # ONE step of the pipeline (render_deck STEP 3.3), and it reads
        # `calculation` off this spec to compose the kind's science
        # (validation/__init__._KIND_VALIDATORS).  The old generator
        # validated inside its own body because it WAS the whole
        # pipeline; under the framework a second call here is the
        # two-gates drift the 2026-08-19 rule abolished.
        from ..spectra.methods import (extract_citation_keys,
                                       render_methods_md)
        methods_md = render_methods_md(
            view, fragment_md=pyscf_methods_fragment(view), struct=struct)
        bibliography_keys = extract_citation_keys(methods_md)
        from ..runtime_info import (
            emit_threading_setup_lines,
            emit_runtime_info_capture_lines,
            emit_pyscf_post_import_lines,
            emit_gpu_probe_lines,
            GPU4PYSCF_MIN_COMPUTE_CAPABILITY,
        )
        out: List[str] = []
        out += _emit_header_docstring(struct, view, methods_md=methods_md)
        out += emit_threading_setup_lines(view.threads)
        out += emit_runtime_info_capture_lines(
            use_gpu=bool(view.use_gpu),
            max_memory_mb=(int(view.max_memory_mb)
                           if view.max_memory_mb else None),
        )
        out += _emit_imports(view)
        out += emit_pyscf_post_import_lines()
        out.append("_RUNTIME_INFO['n_threads_pyscf'] = "
                   "int(_pyscf_lib.num_threads())")
        out += _emit_constants(struct, view, methods_md=methods_md,
                               bibliography_keys=bibliography_keys)
        out += _vib_constants(cfg)
        # The SCF dresser + density_fit kw -- generated from
        # SCF_SECTION + layout.line (pyscf.md § 7a): every mf this
        # deck builds calls _mb_configure_scf, so the machinery knobs
        # apply identically at the equilibrium, displaced and
        # relaxation sites.
        from .scf_setup import (emit_dft_configure_fn,
                                emit_scf_configure_fn,
                                emit_density_fit_kw,
                                emit_solvent_apply_fn)
        out += emit_scf_configure_fn(cfg, verbose=bool(getattr(cfg, 'verbose_comments', True)))
        # The DFT dresser beside the SCF one (M1.2): functional / grid /
        # dispersion get ONE spelling, generated from the same section
        # and line the optimization deck's layout walks.
        out += emit_dft_configure_fn(cfg, verbose=bool(getattr(cfg, 'verbose_comments', True)))
        out += [""] + emit_density_fit_kw(cfg)
        out += emit_solvent_apply_fn(cfg)
        # with_promotion_helper=False: this deck's ONE GPU mechanism is
        # class selection (see runtime_info.emit_gpu_probe_lines).
        out += emit_gpu_probe_lines(
            use_gpu=bool(view.use_gpu),
            min_compute_capability=int(
                GPU4PYSCF_MIN_COMPUTE_CAPABILITY),
            with_promotion_helper=False,
        )
        out.append("if _USING_GPU:")
        out.append("    from gpu4pyscf import scf as _gpu_scf")
        if str(view.method).upper() in ("RKS", "UKS"):
            out.append("    from gpu4pyscf import dft as _gpu_dft")
            out.append("    _scf = _gpu_scf")
            out.append("    _dft = _gpu_dft")
        else:
            out.append("    _scf = _gpu_scf")
            out.append("    _dft = None")
        out.append("else:")
        out.append("    _scf = scf")
        if str(view.method).upper() in ("RKS", "UKS"):
            out.append("    _dft = dft")
        else:
            out.append("    _dft = None")
        out += _emit_atomic_writer()
        out += _emit_build_mol(struct, view, stage_token=stage_token or "")
        # Category 3 (integration plan): the standalone-geometry writer
        # and the live-watch emitter ride the same homes the
        # optimization deck uses -- emit_save_helper and
        # _emit_molwatch_emitter are input.py's own, imported, so the
        # two decks cannot drift about either text.
        if getattr(cfg, "save_initial_xyz", False) or getattr(
                cfg, "save_optimized_xyz", False):
            from .input import emit_save_helper
            out += emit_save_helper(bool(getattr(cfg, "verbose_comments", True)))
        if getattr(cfg, "save_initial_xyz", False):
            out.append("_save_xyz(mol, _mb_outfile(JOB + '_initial.xyz'),")
            out.append("          'Input geometry (pre-relaxation)')")
        if getattr(cfg, "write_molwatch_log", False):
            from .input import _emit_molwatch_emitter
            out += _emit_molwatch_emitter(
                bool(getattr(cfg, "verbose_comments", True)), cfg,
                stage_token=stage_token)
        out += _emit_frozen_mask()
        out += _emit_initial_state()
        out += _vib_state_init()
        out += _emit_displaced_scf_helpers(view)
        # The VIEW, not the raw config: the relax block needs the frozen
        # set (a structure-side fact the view lifted), and the view
        # forwards every config field it does not bridge.
        out += _vib_relax_block(view, stage_token=stage_token)
        out += _emit_equilibrium_scf(view, struct)
        out += _vib_gradient_check()
        out += _emit_gpu_coverage_probe(view)
        out += _emit_hessian_block(view)
        out += _vib_thermo_block()
        if view.compute_raman:
            out += _emit_raman_block(view)
        elif view.compute_ir:
            out += _vib_ir_only_block(cfg)
        else:
            out += ["", "state['phase_raman'] = 'complete'  "
                        "# neither IR nor Raman requested",
                    "_atomic_write_json(state, JSON_PATH)"]
        if view.es_mode_selection != "skip":
            out += _emit_es_loop(view)
        else:
            out += ["", "state['phase_es'] = 'complete'  # selector: skip",
                    "_atomic_write_json(state, JSON_PATH)"]
        out += _emit_final_summary()
        return "\n".join(out) + "\n"

    from . import layout as _layout
    return _sc.DeckSpec(
        engine="pyscf",
        calculation="vibration",
        layout=(_sc.Block("the vibration deck (lifted emitters + the "
                          "relaxation/thermo/IR blocks)", _vib_deck),),
        # The engine's own line/provenance answers, shared with the
        # optimization deck -- one syntax per engine, not per kind.
        # The DFT test is membership, not "anything but HF": RHF/UHF are
        # Hartree-Fock spellings too, and classifying them as DFT would
        # emit mf.xc / grids lines into a wavefunction-method deck.
        line=_layout.line(cfg,
                          is_dft=(str(cfg.method).upper() in ("RKS", "UKS"))),
        provenance_defaults=lambda c: {
            "use_gpu":     str(bool(getattr(c, "use_gpu", False))).lower(),
            "density_fit": str(bool(getattr(c, "density_fit", True))).lower(),
        },
        created_by="molbuilder vibration deck",
        # The engine's own deck gate, shared with the optimization deck:
        # it parses (a non-compiling deck dies on the queue after the
        # wait -- the shipped ECP double-kwarg was exactly this class),
        # it builds a molecule, and its JOB literal is the stamped
        # identity.  The artifact bar (.spectra.json, schema-valid)
        # remains the E2E proof; this gate is the prep-time one.
        check_rules=_layout.check_rules,
    )

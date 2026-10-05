"""The vibration calculation's deck — PySCF, on the framework's seam.

Contract: ``docs/archive/2026-08-20-spectra-migration-plan.md`` § 2 (the rulings of
2026-08-20) + `script-preparation.md` § 4 (the seam this serves).

WHAT THIS IS.  ``vibration_spec(struct, cfg, stage_token)`` returns the
:class:`~molbuilder.script_emit.DeckSpec` for ``calculation = "vibration"``:
one deck, one run — the relaxation as its mandatory first act (geomeTRIC
straight into the Hessian, in-process; D3's final form), ONE harmonic
analysis for free and held atoms (the whole-body motions that survive the
frozen set projected out before diagonalising -- `spectra.normal_modes`,
spliced), IR and Raman as
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

from ..constants import BOLTZMANN_HARTREE_K as _BOLTZMANN_HARTREE_K
from ..constants import HARTREE_CM1 as _HARTREE_CM1
from .. import script_emit as _sc
from ..runfiles import tail as _rf_tail
from ..structure import FROZEN_LABEL, Structure
from .input import (GEOMETRIC_APPENDS, ROLE_CONSTRAINTS, emit_bundle_imports,
                    emit_outfile_helper, ROLE_GEOM_TRAJ)

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
#: are the knobs).  ONE home for the numbers -- `spectra/normal_modes.py`,
#: which the SIESTA derivation reads too -- written into the deck as a
#: literal list; the deck adds the headline temperature to it at run time.
def _nm_thermo_grid_home():
    from ..spectra.normal_modes import THERMO_GRID_K
    return THERMO_GRID_K
_THERMO_GRID_K = repr(list(_nm_thermo_grid_home()))


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

    #: WHOSE VIEW THIS IS -- `template.engine_name` reads it, so a reader
    #: holding the view gets PySCF's answers, never the class name's (the K8
    #: review: the frame door answered `vibrationconfigview` as in-cell).
    ENGINE = "pyscf"

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
        self._struct = struct
        self._state = None

    @property
    def state(self):
        """THE ELECTRONIC STATE, resolved once at the lift boundary
        (`science/chemistry-correctness.md` § 2a) -- the charge, the count
        and the SCF class the deck writes all come from it.  Lazily, because
        the kind's science reads this view too and must not fail on a label
        naming no element: the state is an electron count, and the label
        check owns that finding.  (`net_charge or 0` stood here once and
        silently dropped the phosphate rule, so a nucleic-acid vibration was
        a different calculation from its optimization sibling.)"""
        if self._state is None:
            from ..electronic_state import electronic_state
            self._state = electronic_state(self._struct, self._cfg,
                                           kind="vibration")
        return self._state

    @property
    def axis_kind(self):
        """THE AXES THE DECK COMPUTES ON (`model/structure-periodicity.md`
        § 2.1, plan § 5w K8): ``gto.M`` builds a molecule in free space, so a
        cluster's, whatever the structure's -- asked of the one door
        (`cell.engine_axis_kinds`), and read by the deck's count, the
        Methods count and the settings check's note alike.  The latter two
        counted the structure's own until 2026-10-01 (PS-C4)."""
        from ..cell import engine_axis_kinds
        from ..template import engine_name
        return engine_axis_kinds(engine_name(type(self._cfg)), self._struct)

    @property
    def scf_class(self) -> str:
        """``RKS`` / ``UKS`` / ``RHF`` / ``UHF`` -- composed from the state
        (`pyscf/layout.scf_class`), never read off one field."""
        from .layout import scf_class
        return scf_class(self.state)

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
    from ..spectra.selection import select_modes
    from ..trajectory_log.emitter import MolwatchEmitter
    from ..workingcopy_structure import StructureCodec
    from .relax_policy import policy_of
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
           "# What the relaxation does when geomeTRIC reports its criteria",
           "# unmet at GEOM_MAX_STEPS (engines/pyscf.md 3): halt, continue",
           "# from the geometry reached, or proceed with it.",
           f"ON_NONCONVERGENCE     = {policy_of(cfg)[0]!r}",
           f"GEOM_CONTINUE_RETRIES = {policy_of(cfg)[1]}",
           f"THERMO_T_K     = {float(cfg.temperature_K)!r}",
           f"THERMO_P_ATM   = {float(cfg.pressure_atm)!r}",
           f"THERMO_T_GRID  = {_THERMO_GRID_K}",
           # Where every output lands: beside the script -- the one
           # definition every PySCF deck carries (`input.emit_outfile_helper`).
           *emit_outfile_helper(),
           # And the molbuilder code the deck calls, from the file beside it
           # (`engines/pyscf.md` § 3): the pair writer's codec, the
           # progress-log writer and the mode selector (§ 4.8).
           *emit_bundle_imports(StructureCodec, MolwatchEmitter,
                                select_modes),
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
        "                       'n_steps': 0, 'max_force_eh_bohr': None,",
        "                       'converged': None}",
        "state['thermo'] = {}",
        "# What the harmonic analysis removes before diagonalising, filled",
        "# beside the modes (science/normal-modes.md R7).",
        "state['removed_motions'] = {}",
        "_atomic_write_json(state, JSON_PATH)",
    ]


def _vib_relax_block(cfg, stage_token=None) -> List[str]:
    """The mandatory precondition (D3's final form): geomeTRIC, in-process,
    BEFORE the equilibrium SCF — which then runs on the relaxed geometry
    unchanged, because this block rebinds ``mol`` and ``COORDS_EQ_ANG``.
    Every step writes the artifact, so the viewer's chip shows
    'step N, max force F' ticking down.

    The workflow knobs are honored here — ``geom_etol`` joins the
    convergence dict, ``on_nonconvergence`` and its retry budget go to the
    ONE relaxation function both PySCF decks run (`relax_policy.relax`,
    `engines/pyscf.md` § 3), which asks geomeTRIC whether it converged;
    ``write_trajectory`` hands geomeTRIC its streaming-XYZ prefix,
    ``write_molwatch_log`` composes the molwatch hooks beside the artifact
    callback, and the two ``save_*_xyz`` flags write their standalone
    files.  geomeTRIC is the one optimizer (`engines/pyscf.md` § 3).

    ``relaxation.converged`` is the judged force at the geometry reached
    against ``geom_gmax`` — the meaning the key has on every route
    (`engines/vibration.md` § 4.2, § 4.3, § 5.5); geomeTRIC's own verdict,
    all of its criteria, decides the policy."""
    out: List[str] = [
        "",
        "# ============================================================",
        "#  Phase 0: relaxation -- the measurement's precondition",
        "# ============================================================",
        "if not ALREADY_RELAXED:",
        "    print('=== Stage: relaxation (geomeTRIC) ===')",
        "    state['phase_relaxation'] = 'running'",
        "    _atomic_write_json(state, JSON_PATH)",
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
        "            # The free atoms' force (science/normal-modes.md R5):",
        "            # a held atom carries the constraint force by",
        "            # definition, and geomeTRIC converges the projected",
        "            # gradient, so the number the chip ticks down is the",
        "            # one the criterion is about.",
        "            _mx = None",
        "            if _g is not None:",
        "                _ga = np.abs(np.asarray(_g, dtype=float)).reshape(-1, 3)",
        "                _mx = float(_ga[FREE_ATOM_IDXS].max()) if len(_ga) == N_ATOMS else float(_ga.max())",
        "        except Exception:",
        "            _mx = None",
        "        state['relaxation']['n_steps'] += 1",
        "        if _mx is not None:",
        "            state['relaxation']['max_force_eh_bohr'] = _mx",
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
            # ONE SPELLING of the frozen-set line, the optimization deck's
            # (`pyscf/input.py`): the run wrapper reads the constraint back
            # off the deck by that exact comment, so a second wording here
            # made every held-atom vibration run report "no frozen_atoms --
            # all atoms free" in its own header.
            f"    # Source: Structure.frozen_atoms = {_frozen!r}  (0-based)",
            "    _FROZEN_CONSTRAINTS_PATH = _mb_outfile(JOB "
            f"+ {ROLE_CONSTRAINTS!r})",
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
        # THE PREFIX IS DERIVED FROM THE ROLE, exactly as the optimization
        # deck's is (`pyscf/input.py`; `job-contracts.md` § 2.2a): compose the
        # name the file will HAVE, then take off the tail geomeTRIC appends.
        #
        # It was `JOB + '_geom_<stage>'` until 2026-09-07 -- the token INSIDE
        # the role, giving `<JOB>_geom_<stage>_optim.xyz`, which the declared
        # role `_geom_optim.xyz` cannot match.  The optimization deck was
        # fixed the same week and this one, the vibration deck, was missed:
        # the same defect, in the second place that writes the same file.
        _pfx = _rf_tail(ROLE_GEOM_TRAJ,
                        stage_token or None)[:-len(GEOMETRIC_APPENDS)]
        _opt_kw += f", prefix=str(_mb_outfile(JOB + {_pfx!r}))"
    # THE ONE RELAXATION FUNCTION (`relax_policy.relax`, spliced above):
    # it asks geomeTRIC whether it converged and applies this rung's
    # on_nonconvergence to the answer -- halt stops the run here, before
    # the Hessian; continue re-enters from the geometry reached; proceed
    # returns it with False.  The recorded verdict is the judged force at
    # that geometry (R5), the key's one meaning on every route.
    out += [
        "    mol, _geometric_converged = relax(",
        "        _mf_relax, ON_NONCONVERGENCE, GEOM_CONTINUE_RETRIES,",
        f"        {_opt_kw}, **_conv)",
        "    _judged = state['relaxation']['max_force_eh_bohr']",
        "    state['relaxation']['converged'] = (",
        "        None if _judged is None else bool(_judged <= GEOM_GMAX))",
        "    if not _geometric_converged:",
        "        state['relaxation']['warning'] = (",
        '            f"the relaxation did not meet geomeTRIC\'s criteria in "',
        '            f"{GEOM_MAX_STEPS} steps; on_nonconvergence = proceed kept "',
        '            f"the geometry it reached, so the frequencies below are the "',
        '            f"curvature there, not at the minimum")',
    ]
    if getattr(cfg, "save_optimized_xyz", False):
        out += [
            "    _save_structure(mol, _mb_outfile(JOB + '_optimized.xyz'),",
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
    never refuses: skipping was a deliberate choice.  Its remedy is THE one
    text the SIESTA finish and `prep` write too
    (`vibrational_analysis.nonstationary_remedy`, `engines/vibration.md`
    § 5.5), written into the deck as a literal at render time."""
    from ..spectra.vibrational_analysis import nonstationary_remedy
    _remedy = nonstationary_remedy(None, "pyscf")
    return [
        "",
        "if ALREADY_RELAXED:",
        "    print('=== gradient check (already_relaxed asserted) ===')",
        "    try:",
        "        _g0 = _as_numpy(mf.nuc_grad_method().kernel())",
        "        # Judged in the subspace that is diagonalised (science/",
        "        # normal-modes.md R5): a held atom carries the constraint",
        "        # force by definition, and that number says nothing about",
        "        # whether the free atoms sit at their stationary point.",
        "        # Both are recorded; only the free one is judged.",
        "        _maxf_all = float(np.abs(_g0).max())",
        "        _maxf = float(np.abs(_g0[FREE_ATOM_IDXS]).max())",
        "        state['relaxation']['max_force_eh_bohr'] = _maxf",
        "        state['relaxation']['max_force_all_atoms_eh_bohr'] = _maxf_all",
        "        state['relaxation']['converged'] = bool(_maxf <= GEOM_GMAX)",
        "        if _maxf > GEOM_GMAX:",
        "            _w = (f'the input geometry is not a stationary point '",
        "                  f'at this level of theory: the largest force on '",
        "                  f'the free atoms is {_maxf:.2e} Eh/Bohr against '",
        "                  f'the geom_gmax of this calculation, {GEOM_GMAX:.1e}. '",
        "                  f'The frequencies are the curvature at this point, '",
        "                  f'not at the minimum, and will be off -- the low '",
        "                  f'ones most.  already_relaxed was stated, so this '",
        "                  f'run carries on.  '",
        f"                  + {_remedy!r})",
        "            print('WARNING: ' + _w)",
        "            state['relaxation']['warning'] = _w",
        "        _atomic_write_json(state, JSON_PATH)",
        "    except Exception as _e:  # the check must never kill the run",
        "        print(f'gradient check skipped: {_e}')",
    ]


def _vib_thermo_block() -> List[str]:
    """RRHO thermochemistry into the ``thermo`` block, from the mode list
    the hessian block just wrote.  Every entry of that list is a vibration
    (the whole-body motions were projected out before diagonalising), so
    nothing here filters for one; an IMAGINARY mode has no harmonic
    partition function and is excluded with its count recorded.  Regime
    honesty: a free molecule gets full RRHO via PySCF's own
    ``thermo.thermo``; a system with atoms held gets the VIBRATIONAL
    contributions only, stated -- there is no gas-phase translational or
    rotational partition function to add.  ONE quantity under one label
    (engines/vibration.md § 4.7): the headline and every point of the grid are the
    same sum, and the headline temperature is on the grid."""
    return [
        "",
        "# ============================================================",
        "#  Thermochemistry (v5 `thermo`; D2's re-homing)",
        "# ============================================================",
        "print('=== Stage: thermochemistry ===')",
        "# Every mode is a vibration -- the whole-body motions were removed",
        "# before diagonalising -- so nothing here filters for one.  An",
        "# imaginary mode has no harmonic partition function; it is left out",
        "# and the count is recorded, never dropped in silence.",
        "_freqs_cm1 = np.array([m['frequency_cm1'] for m in",
        "                       state['modes'] if not m['has_imag']])",
        "_N_IMAG_EXCLUDED = int(sum(1 for m in state['modes'] if m['has_imag']))",
        f"_KB_EH = {_BOLTZMANN_HARTREE_K!r}        # Boltzmann, Eh/K",
        f"_CM1_TO_EH = 1.0 / {_HARTREE_CM1!r}",
        "# The harmonic sums come from the one home both engines share",
        "# (spectra/normal_modes.vibrational_thermo, spliced above).",
        "def _vib_thermo_at(T):",
        "    return vibrational_thermo(_freqs_cm1, T, _KB_EH, _CM1_TO_EH)",
        "_regime = 'rrho' if not FROZEN_ATOM_IDXS else 'vibrational-only'",
        "_zpe0, _u0, _s0 = _vib_thermo_at(THERMO_T_K)",
        "# The grid the viewer draws, with the headline temperature ON it, so",
        "# the curve passes through the headline number and the two cannot",
        "# disagree.",
        "_T_GRID = sorted(set(float(_t) for _t in THERMO_T_GRID) | {float(THERMO_T_K)})",
        "state['thermo'] = {",
        "    'regime': _regime,",
        "    'temperature_K': THERMO_T_K, 'pressure_atm': THERMO_P_ATM,",
        "    'zpe_eh': _zpe0,",
        "    'n_modes': int(len(state['modes'])),",
        "    'n_imag_excluded': _N_IMAG_EXCLUDED,",
        "    'n_rigid_removed': N_RIGID,",
        "}",
        "# ONE QUANTITY UNDER ONE LABEL: the headline and every grid point are",
        "# the same sum.  Free molecule: PySCF's full RRHO (electronic +",
        "# translational + rotational + vibrational) at each T.  Atoms held:",
        "# the vibrational sums above the electronic energy at each T -- there",
        "# is no gas-phase translational or rotational partition function to",
        "# add, and the whole-body motions of the free atoms were removed",
        "# before diagonalising.",
        "_grid = None",
        "# Why a vibrational-only answer is one, said in its note: atoms held,",
        "# or a free molecule whose full RRHO failed.",
        "_why_vib_only = ('atoms are held, so the whole has no gas-phase '",
        "                 'translation or rotation to add')",
        "if _regime == 'rrho':",
        "    try:",
        "        def _rrho_at(_T):",
        "            # The same signed omega the modes were reported from;",
        "            # pyscf's thermo keeps the positive ones itself.",
        "            _r = _mb_thermo.thermo(mf, _OMEGA_AU, float(_T),",
        "                                   THERMO_P_ATM * 101325.0)",
        "            return (float(_r['ZPE'][0]),",
        "                    float(_r['E_vib'][0]) - float(_r['ZPE'][0]),",
        "                    float(_r['H_tot'][0]), float(_r['S_tot'][0]),",
        "                    float(_r['G_tot'][0]))",
        "        _grid = {'temperatures_K': [], 'zpe_eh': [], 'u_vib_eh': [],",
        "                 'h_eh': [], 's_eh_k': [], 'g_eh': []}",
        "        for _T in _T_GRID:",
        "            _z, _u, _h, _s, _g = _rrho_at(_T)",
        "            _grid['temperatures_K'].append(float(_T))",
        "            _grid['zpe_eh'].append(_z)",
        "            _grid['u_vib_eh'].append(_u)",
        "            _grid['h_eh'].append(_h)",
        "            _grid['s_eh_k'].append(_s)",
        "            _grid['g_eh'].append(_g)",
        "        state['thermo']['note'] = (",
        "            'Full rigid-rotor harmonic-oscillator (RRHO) thermochemistry of '",
        "            'the free molecule as an ideal gas at this temperature and '",
        "            'pressure -- electronic, translational, rotational and '",
        "            'vibrational -- via PySCF thermo, the headline and the grid '",
        "            'alike.  Good for gas-phase reaction and binding free energies '",
        "            'against species computed the same way; the vibrations are '",
        "            'harmonic, so modes below about 100 cm-1 (torsions, hindered '",
        "            'rotations) make the entropy the least reliable number')",
        "    except Exception as _e:",
        "        print(f'full-RRHO thermo unavailable ({_e}); '",
        "              'vibrational-only values recorded')",
        "        _regime = 'vibrational-only'",
        "        state['thermo']['regime'] = _regime",
        "        _why_vib_only = (f'PySCF thermo could not give the full RRHO '",
        "                         f'for this free molecule ({_e})')",
        "        _grid = None",
        "if _grid is None:",
        "    _grid = vibrational_thermo_grid(_freqs_cm1, _T_GRID, float(E_eq),",
        "                                    _KB_EH, _CM1_TO_EH)",
        "    # No pressure enters the vibrational sums -- only the gas-phase",
        "    # translational term takes one (engines/vibration.md 4.7).",
        "    state['thermo']['pressure_atm'] = None",
        "    state['thermo']['note'] = (",
        "        'The harmonic VIBRATIONAL contributions of the free atoms at this '",
        "        'temperature, above the electronic energy -- ZPE, U_vib, S_vib '",
        "        'and F_vib = ZPE + U_vib - T*S_vib -- the headline and the grid '",
        "        'alike, because ' + _why_vib_only + '.  Good for the zero-point '",
        "        'and vibrational free-energy correction of one structure against '",
        "        'another computed the same way; not the free energy of a '",
        "        'molecule in the gas phase: no translation or rotation is added '",
        "        'and no pressure enters.  Modes below about 100 cm-1 are '",
        "        'anharmonic in practice and carry most of S_vib, so the entropy '",
        "        'is the least reliable number; the whole-body motions of the free '",
        "        'atoms were removed before diagonalising')",
        "_k = _grid['temperatures_K'].index(float(THERMO_T_K))",
        "state['thermo']['h_eh'] = _grid['h_eh'][_k]",
        "state['thermo']['s_eh_k'] = _grid['s_eh_k'][_k]",
        "state['thermo']['g_eh'] = _grid['g_eh'][_k]",
        "state['thermo']['grid'] = _grid",
        "_atomic_write_json(state, JSON_PATH)",
    ]


def _vib_ir_only_block(cfg) -> List[str]:
    """IR without Raman -- the decoupling the 2026-08-20 ruling asked for.

    TWO ROUTES TO THE SAME TENSOR, chosen at RUN time because the choice
    depends on the env the deck lands in, not the machine that wrote it:

    * ``dipole_derivatives`` already filled ``DMU_DR`` analytically, off
      the Hessian's own CPHF solution -- see the spliced rule in
      ``vibration_emitters``.  Nothing to do here but project.
    * It could not (``pyscf.prop.infrared`` is absent, or the reference
      is one it does not cover), so the dipole is finite-differenced
      over the free Cartesians: 6N extra SCFs for the same numbers.

    Either way the LIFTED per-mode projection runs verbatim, so the two
    paths cannot disagree about the formula -- only about how dmu/dR was
    obtained, which ``state['ir_route']`` records for the reader.
    """
    out = [
        "",
        "# ============================================================",
        "#  Phase 3-IR: IR intensities",
        "# ============================================================",
        "state['phase_raman'] = PHASE_NOT_REQUESTED   # no Raman sweep was asked for (vibration.md 4.9)",
        "# The dipole sweep writes nothing until it ends, so the flag is what",
        "# tells a reader the intensities are still coming (vibration.md 4.9).",
        "state['phase_ir'] = PHASE_RUNNING",
        "_atomic_write_json(state, JSON_PATH)",
        "if DMU_DR is None:",
        "    print('=== Stage: IR intensities (dipole finite differences) ===')",
        "    _h_ang = RAMAN_FD_STEP_ANG   # ONE step for both dmu/dR paths",
        "    DMU_DR = np.zeros((N_FREE, 3, 3))",
        "    for _k, _atom in enumerate(FREE_ATOM_IDXS):",
        "        for _a in range(3):",
        "            _cp = np.array(COORDS_EQ_ANG, dtype=float)",
        "            _cm = np.array(COORDS_EQ_ANG, dtype=float)",
        "            _cp[_atom, _a] += _h_ang",
        "            _cm[_atom, _a] -= _h_ang",
        "            # _build_mf_at CONVERGES the SCF before returning (and",
        "            # halts if it cannot) -- a second kernel() here re-ran",
        "            # the whole SCF per displaced point, doubling the",
        "            # loop's cost for identical numbers.",
        "            _mfp = _build_mf_at(_cp)",
        "            _mfm = _build_mf_at(_cm)",
        "            _dp = _as_numpy(_mfp.dip_moment(unit='Debye', verbose=0))",
        "            _dm = _as_numpy(_mfm.dip_moment(unit='Debye', verbose=0))",
        "            DMU_DR[_k, _a, :] = (_dp - _dm) / (2.0 * _h_ang)",
        "else:",
        "    print('=== Stage: IR intensities (analytic dmu/dR, no extra SCFs) ===')",
        "modes_payload = state['modes']",
    ]
    # The projection emits ITS OWN per-mode loop (reading
    # NORM_MODES_CANONICAL, which the Hessian block defines on every
    # path) -- wrapping it in a second per-mode loop here ran the whole
    # projection once PER MODE: N² idempotent work whose outer loop
    # variable nothing read.
    out.extend(_emit_ir_projection())
    out += [
        "state['phase_ir'] = PHASE_COMPLETE",
        "_atomic_write_json(state, JSON_PATH)",
    ]
    return out


def vibration_spec(struct: Structure, cfg, *,
                   stage_token: Optional[str] = None) -> "_sc.DeckSpec":
    """The vibration calculation's DeckSpec — the seam's answer for
    ``calculation = 'vibration'`` (PySCF first; the shape admits others)."""
    view = VibrationConfigView(cfg, struct)
    # THE ONE PLACEMENT, computed once (`model/structure-periodicity.md`
    # § 6.0): the molecule block writes these positions and the deck's
    # ENGINE-OFFSET record states the same frame.
    from ..cell import to_engine as _to_engine
    _frame = _to_engine(struct)

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
        out += _emit_header_docstring(struct, view, methods_md=methods_md,
                                      stage_token=stage_token)
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
        from .scf_setup import (emit_theory_configure_fn,
                                emit_scf_configure_fn,
                                emit_density_fit_kw,
                                emit_solvent_apply_fn)
        out += emit_scf_configure_fn(cfg, verbose=bool(getattr(cfg, 'verbose_comments', True)))
        # The level-of-theory dresser beside the SCF one (M1.2):
        # functional / grid / dispersion get ONE spelling, generated from
        # the same section and line the optimization deck's layout walks
        # -- on every deck, since the dispersion correction applies to
        # Hartree-Fock too (pyscf.md § 7a).
        out += emit_theory_configure_fn(cfg, verbose=bool(getattr(cfg, 'verbose_comments', True)))
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
        if view.is_dft:
            out.append("    from gpu4pyscf import dft as _gpu_dft")
            out.append("    _scf = _gpu_scf")
            out.append("    _dft = _gpu_dft")
        else:
            out.append("    _scf = _gpu_scf")
            out.append("    _dft = None")
        out.append("else:")
        out.append("    _scf = scf")
        if view.is_dft:
            out.append("    _dft = dft")
        else:
            out.append("    _dft = None")
        out += _emit_atomic_writer()
        out += _emit_build_mol(struct, view, stage_token=stage_token or "",
                               positions=_frame.positions)
        # Category 3 (integration plan): the standalone-geometry writer
        # and the live-watch emitter ride the same homes the
        # optimization deck uses -- emit_save_helper and
        # _emit_molwatch_emitter are input.py's own, imported, so the
        # two decks cannot drift about either text; the code both call
        # is imported from mb_pyscf.pyz at the top (_vib_constants).
        if getattr(cfg, "save_initial_xyz", False) or getattr(
                cfg, "save_optimized_xyz", False):
            from .input import _sidecar_for, emit_save_helper
            out += emit_save_helper(bool(getattr(cfg, "verbose_comments", True)),
                                    _sidecar_for(struct))
        if getattr(cfg, "save_initial_xyz", False):
            out.append("_save_structure(mol, _mb_outfile(JOB + '_initial.xyz'),")
            out.append("          'Input geometry (pre-relaxation)')")
        if getattr(cfg, "write_molwatch_log", False):
            # THE SCF'S CRITERIA, read off a solver this deck's own dresser
            # configures -- every mf here is built through it -- so the
            # progress log states them as the optimization deck's does.
            from .input import (_emit_molwatch_emitter,
                                emit_scf_criteria_readback)
            # The probe is THIS run's class (`layout.scf_class`), not a
            # hard-coded `scf.RHF` -- which PySCF turned into ROHF under a
            # nonzero spin, and which was never the solver the run built.
            from .layout import scf_module
            out.append(f"_mb_scf_probe = _mb_configure_scf("
                       f"{scf_module(view.state)}.{view.scf_class}(mol))")
            out += emit_scf_criteria_readback("_mb_scf_probe")
            out.append("del _mb_scf_probe")
            # THE ATOMS THIS RUN HOLDS -- the set its relaxation freezes
            # (`_vib_relax_block`) -- go into its log's header, as the
            # optimization deck's do (`model/parse.md` § 5.3).
            out += _emit_molwatch_emitter(
                bool(getattr(cfg, "verbose_comments", True)), cfg,
                stage_token=stage_token,
                frozen_atoms=list(view.frozen_indices))
        out += _emit_frozen_mask()
        out += _emit_initial_state()
        out += _vib_state_init()
        out += _emit_displaced_scf_helpers(view)
        # The one relaxation function both PySCF decks run (`engines/pyscf.md`
        # § 3), spliced before the phase that calls it.
        from .relax_policy import emit_relax
        out += [""] + emit_relax()
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
            out += ["", "state['phase_raman'] = PHASE_NOT_REQUESTED  "
                        "# neither IR nor Raman requested (vibration.md 4.9)",
                    "_atomic_write_json(state, JSON_PATH)"]
        if view.es_mode_selection != "skip":
            out += _emit_es_loop(view)
        else:
            out += ["", "state['phase_es'] = PHASE_NOT_REQUESTED  # selector: skip -- nothing was asked (vibration.md 4.9)",
                    "_atomic_write_json(state, JSON_PATH)"]
        out += _emit_final_summary()
        return "\n".join(out) + "\n"

    from . import layout as _layout
    return _sc.DeckSpec(
        engine="pyscf",
        calculation="vibration",
        engine_frame=_frame,
        layout=(_sc.Block("the vibration deck (lifted emitters + the "
                          "relaxation/thermo/IR blocks)", _vib_deck),),
        # The engine's own line/provenance answers, shared with the
        # optimization deck -- one syntax per engine, not per kind.
        # The DFT test reads the method item -- DFT or HF, nothing else
        # since 2026-09-28 -- so a Hartree-Fock deck writes no mf.xc /
        # grids lines.
        line=_layout.line(cfg,
                          is_dft=cfg.is_dft),
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

"""PySCFConfig -- every parameter the PySCF script generator emits.

L1 dataclass.  Its field metadata is read by the validation pass
(``molbuilder/validation/``; the readers are `web/form-schema.md` § 1a's)
-- the forms are drawn from the catalogue; the PySCF deck's spec
(``molbuilder/pyscf/input.py:spec_for``, and ``pyscf/vibration_deck.py``
for a vibration) renders the configured values.

Defaults are tuned for "build a small/medium molecule and relax it":

    * B3LYP+D3BJ/def2-SVP  (modern hybrid, dispersion-corrected)
    * Density fitting on -- bare ``mf.density_fit()``, so PySCF auto-picks
      the *basis-matched* JK-fit set (``def2-svp-jkfit`` for this def2-SVP
      default, ``def2-tzvp-jkfit`` for def2-TZVP, ...), verified via
      ``mf.with_df.auxbasis`` on a real def2 hybrid.
    * geomeTRIC optimizer with maxsteps=200, grms=3e-4 Ha/Bohr
    * Kohn-Sham DFT, with the charge and the spin worked out from the
      structure when left blank (`electronic_state`,
      `science/chemistry-correctness.md` § 2a): the phosphate rule for the
      charge, then open or closed shell at that charge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..identity import RestartGroup
from ..selection import parse_index_list
from .siesta import _validate_basename     # shared with SiestaConfig


# --------------------------------------------------------------------- #
#  The ladder's per-tier science                                        #
# --------------------------------------------------------------------- #

#: What each rung of a PySCF ladder is tuned to, tier by tier.
#:
#: Read across from ``SIESTA_STAGE_PRESETS``: one table per engine, keyed by
#: the same three tiers, and ``<engine>/stages.py::default_<engine>_stages``
#: turns it into the shipped ladder of :class:`~molbuilder.task.Stage`
#: objects.  The two engines differ in which parameters a tier names and in
#: nothing else (`stages.md` § 1.1a).
#:
#: **Every number is `tuning.md` § 2.4's and § 2.5's**, column by column --
#: loose preopt, publishable, tight.  That table is what a reviewer is
#: pointed at, so it is what this states; a value here that disagrees with it
#: is a bug here rather than a second opinion.
#:
#: The keys are catalogue items, which is what makes them legal ``overrides``
#: on a stage (`stages.md` § 2).  ``restart`` is NOT among them: a rung's
#: position does not answer *is there anything to continue from*
#: (`run-identity.md` § 4 rule 3), so neither engine's ladder sets it.
#:
#: Units: ``geom_gmax`` / ``geom_grms`` in Ha/Bohr; ``geom_dmax`` /
#: ``geom_drms`` in Angstrom -- NOT Bohr, a long-standing geomeTRIC doc bug
#: whose source uses Angstrom; ``geom_etol`` and ``scf_conv_tol`` in Hartree.
PYSCF_STAGE_PRESETS: Dict[int, Dict[str, Any]] = {
    1: {   # loose preopt
        "scf_conv_tol":   1.0e-7,
        "geom_gmax":      2.0e-3,
        "geom_grms":      1.3e-3,
        "geom_dmax":      7.2e-3,
        "geom_drms":      4.8e-3,
        "geom_etol":      1.0e-5,
        "geom_max_steps": 50,
    },
    2: {   # publishable -- geomeTRIC's GAU preset
        "scf_conv_tol":   1.0e-9,
        "geom_gmax":      4.5e-4,
        "geom_grms":      3.0e-4,
        "geom_dmax":      1.8e-3,
        "geom_drms":      1.2e-3,
        "geom_etol":      1.0e-6,
        "geom_max_steps": 200,
    },
    3: {   # tight
        "scf_conv_tol":   1.0e-10,
        "geom_gmax":      2.0e-4,
        "geom_grms":      1.0e-4,
        "geom_dmax":      1.0e-3,
        "geom_drms":      5.0e-4,
        "geom_etol":      1.0e-6,
        "geom_max_steps": 100,
    },
}


#: § 4 rule 1 — PySCF's identity group.
#:
#: Read across from ``SIESTA_RESTART_GROUP`` and the contract's point is
#: visible: the identity is the same *idea* in both engines and a different
#: *mechanism*. SIESTA declares three keys; PySCF carries generated control
#: flow, so ``keys`` is empty and ``mechanism`` says what actually happens.
#: An empty tuple alone would read as "nothing is bound", which is the
#: opposite of true.
#:
#:
#: **Rule 2 is answered by the ``restart`` field below**, and by the same one
#: field SIESTA answers it with: two values, ``clean`` and ``continue``, and
#: a rerun that says ``clean`` writes its checkpoint without reading the one
#: already beside it.
PYSCF_RESTART_GROUP = RestartGroup(
    literal="JOB",
    keys=(),
    mechanism="generated control flow: mf.chkfile + init_guess='chkfile' "
              "when the file exists, and <JOB>_optimized.xyz overriding the "
              "literal geometry",
    field="job_name",
)


@dataclass
class PySCFConfig:
    # ---------------- System ----------------
    job_name: str = field(
        default="pyscf_relax",
        metadata={"validate": _validate_basename("job_name")})
    # THE ELECTRONIC STATE (`science/chemistry-correctness.md` § 2a) -- the
    # three merged items, declared once in `config/state.py` for both
    # engines, and `method` below.  A blank means *work it out*;
    # `electronic_state` does, for the form, the checks and the deck alike.
    # THE ELECTRONIC STATE'S ITEMS -- `net_charge`, `spin_treatment`,
    # `unpaired_electrons` -- one question asked of both engines, declared
    # once in the catalogue (`engines/template.md` § 6.3); a blank means
    # *work it out* (`electronic_state.electronic_state`).
    net_charge: Optional[int] = None
    spin_treatment: Optional[str] = None
    unpaired_electrons: Optional[Union[int, str]] = None
    symmetry: bool = False

    # ---------------- Method (main run) ----------------
    # WHICH THEORY -- Kohn-Sham DFT or Hartree-Fock -- and nothing else.  The
    # SCF class is COMPOSED from this and `spin_treatment` and written
    # explicitly (`dft.UKS`, `scf.ROHF`, ...; `pyscf/layout.scf_class`), never left
    # to PySCF to re-rule.
    method: str = "DFT"
    functional: str = "B3LYP"
    basis: str = "def2-SVP"
    # auxbasis: rarely set -- the auto-pick from density_fit() is the
    # right default.
    auxbasis: Optional[str] = None
    density_fit: bool = True
    dispersion: str = "d3bj"
    # Effective Core Potential -- TWO plain fields, ONE format each.  The
    # user's rulings (2026-08-13):
    #
    #   * *"there is no point to limit matching to heavy -- who defines
    #     heavy? there is no clear reasoning or standard"* -- so no Z
    #     threshold decides anything.  ``["*"]`` means ALL atoms.
    #   * *"empty means empty"* -- an empty name or an empty list means
    #     no ECP.  It never means "pick one for me".
    #   * *"one choice, one explicit format"*, *"do not invent too many
    #     options/alias"*.
    #
    # Nothing is added behind the user's back.  ``validation`` still
    # HINTS when a structure looks like it wants an ECP and none was
    # declared -- a hint the user confirms, never a choice made for them.
    ecp: str = ""
    ecp_atoms: List[str] = field(default_factory=list)

    # ---------------- SCF ----------------
    scf_conv_tol: float = 1e-9
    scf_conv_tol_grad: float = 0.0
    scf_soscf: bool = False
    scf_max_cycle: int = 100
    scf_init_guess: str = "minao"
    grid_level: int = 4
    level_shift: float = 0.0
    # Hard-SCF troubleshooting knobs.  Defaults preserve PySCF
    # behaviour for the easy-converge case.
    diis_space: int = 8
    damp: float = 0.0

    # ---------------- Main optimization ----------------
    optimize: bool = True
    # ---------------- geomeTRIC convergence (per-stage knobs) ----------
    #
    # **Flat, one value each** -- exactly as SIESTA's per-rung knobs
    # (``relax_force_tol``, ``relax_max_displ``, …) are.  An engine config
    # holds what ONE run does; `task.json`'s stages override it per rung, and
    # a ladder is N of these decks (`engines/stages.md` § 1.1, § 1.1a).  A
    # list-of-rungs field here would be a second place to declare the ladder,
    # free to disagree with the description -- which is what § 1.1 forbids.
    #
    # The SCF tolerance is not among them because it is already above:
    # ``scf_conv_tol`` declares ``mf.conv_tol``, and a per-stage twin of it
    # would be the same knob with two homes.
    geom_gmax: float = 4.5e-4
    geom_grms: float = 3.0e-4
    geom_dmax: float = 1.8e-3
    geom_drms: float = 1.2e-3
    geom_etol: float = 1.0e-6
    geom_max_steps: int = 200
    on_nonconvergence: str = "halt"
    geom_continue_retries: int = 1
    #: ``wall_slack`` -- seconds before its wall the deck stops itself
    #: (`engines/pyscf.md` § 3.9): the deadline it keeps.
    wall_slack: int = 600

    # ---------------- Solvent (optional) ----------------
    solvent: Optional[str] = None
    solvent_method: str = "IEF-PCM"

    # ---------------- Runtime ----------------
    # UNLIMITED unless the user asks for a cap -- the typical memory limit is
    # all physical memory (user, ruled 2026-08-13 and again 2026-08-14).
    max_memory_mb: Optional[int] = None
    threads: Optional[int] = None
    gpu_count: Optional[int] = None
    use_gpu: bool = False
    verbose: int = 4
    restart: str = "continue"
    chkfile: bool = True
    log_file: bool = True
    # Always-on output knobs.
    save_optimized_xyz: bool = True
    save_initial_xyz: bool = True
    write_trajectory: bool = True
    # Match SiestaConfig's naming (``write_molwatch_log``) so the two
    # configs read the same way.  Emission also requires ``optimize=True`` -- the
    # molwatch hooks ride on the SCF and geomeTRIC opt-step callbacks,
    # so a single-point run has nowhere to attach.
    write_molwatch_log: bool = True


    # ----- Vibrational spectroscopy (the vibration calculation kind) -----
    already_relaxed: bool = False
    compute_raman: bool = True
    compute_ir: bool = False
    displacement_amplitude_ang: float = 0.02
    es_mode_selection: str = 'skip'
    es_explicit_indices: str = ''
    freq_min_cm1: Optional[float] = None
    freq_max_cm1: Optional[float] = None
    es_n_homo_below: int = 5
    es_n_lumo_above: int = 5
    temperature_K: float = 298.15
    pressure_atm: float = 1.0

    # ---------------- Comments ----------------
    verbose_comments: bool = True

    @property
    def is_dft(self) -> bool:
        """Whether the method is a density functional.  THE one answer every
        reader of the level of theory asks: the decks' SCF construction,
        headers, constants and line spellings, the grid advisory and the
        Methods paragraph (`engines/vibration.md` § 4.10).  Hartree-Fock has
        no functional and no integration grid, whatever those items hold.
        The dispersion correction is not asked of this answer: HF takes it
        like any method -- it has no correlation at all, so it misses
        dispersion entirely, D3 and D4 carry parameters fitted for it, and
        PySCF applies them through the energy, gradient and Hessian, reading
        the method as ``hf`` (`engines/pyscf.md` § 7a).  `method` is never
        blank, so this asks the field directly."""
        return self.method == "DFT"

    @property
    def explicit_modes(self) -> List[int]:
        """The modes ``es_explicit_indices`` names: 1-based, sorted, each once.
        THE one reading of that text -- the deck's constant (which the
        script's selector reads), the Methods paragraph and the kind's check
        all take it from here (`engines/vibration.md` § 4.8).  The grammar is the atom
        index list's (``selection.parse_index_list``): ``"3, 7, 12"``,
        ``"3-7, 12"``.  Raises ``SelectionError`` (a ``ValueError``) on text
        it cannot read; the kind's check turns that into a refusal at
        ``prep``, so no deck is written from it."""
        return parse_index_list(self.es_explicit_indices, first=1)

    # ``molwatch_log`` reads and writes ``write_molwatch_log``.
    @property
    def molwatch_log(self) -> bool:                  # pragma: no cover
        return self.write_molwatch_log

    @molwatch_log.setter
    def molwatch_log(self, value: bool) -> None:     # pragma: no cover
        self.write_molwatch_log = value


__all__ = ["PySCFConfig"]

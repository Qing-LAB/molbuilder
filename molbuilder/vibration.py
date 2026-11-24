"""A HARMONIC MODE'S MECHANICS, AND THE MOLECULE'S MOTION -- every per-mode
quantity, its definition and its derivation, and the motion built from them,
from one module (`engines/vibration.md` § 6.11; `engines/vibration-frames.md`
§ 1-§ 4, § 7; `science/vibrational-averaging.md` § 2, § 2a-§ 2c, § 5.4,
§ 5.5).

MODULE  vibration (L2; numpy and the constants, nothing else of molbuilder)
ROLE    from what a result states of a mode -- its frequency, whether it is
        imaginary, its canonical eigenvector over the free atoms, the masses
        the modes were weighted by -- every quantity of the mode:

        * its zero-point amplitude ``Q_zp = sqrt(hbar / 2 omega)``, from any
          frequency (§ 2b.1), its mean number of quanta and its spread at a
          temperature (§ 2, § 2a);
        * its geometry: the pattern over every atom, each atom's move for a
          coordinate ``u_A = Q L_A`` and the coordinate back by the
          mass-weighted projection ``Q = sum_A m_A L_A . u_A`` (§ 2b.2), the
          displaced geometry ``R + Q L``, the largest move, and the largest
          at the zero-point amplitude;
        * each atom's share of its motion, ``m_A |L_A|^2`` over the total,
          and those shares summed by any grouping -- element, species label,
          region (`engines/vibration.md` § 6.11 § 3);
        * its reduced mass and force constant by Gaussian's convention, and
          its spread and mean quanta over a list of temperatures (the row
          the record states, § 6.11 § 3);
        * where the molecule sits along it in each state (§ 2b.3), the states
          one table (:data:`_STATES`, `engines/vibration-frames.md` § 2) --
          its density, distribution, moments, spread and mean quanta, its
          draws from the mode's own streams of a seed, the frames of its own
          Gauss rule (§ 2b.4) and the weights for frames placed elsewhere
          (§ 5.4);
        * :class:`Mode` and :class:`Vibration`, the view of a result a person
          computes with (`postprocess.vibration`, § 6.11 § 6) -- the shared
          fields the vibration's -- and the molecule's motion: a state per
          mode and the drive groups (:class:`Motion`), its snapshots and
          their analyses (:class:`Snapshots`, :func:`pooled`).
USED-BY the result's derived values (`spectra.derived`), the thermochemistry
        (`spectra.normal_modes`), the frame-set maker (`spectra.mode_frames`),
        the weights for a frame set's rows (`vibration_frames`), the SIESTA
        vibration's Raman displacements (`spectra.siesta_vibration`), the
        spectrum tab's load door (`web.blueprints.spectra`), a result's
        modes (`spectra.results.SpectraResults.vibration`) and the script
        door (`postprocess.vibration`)
TRAVELS in both bundles -- the PySCF script's and a SIESTA vibration's
        (`runwrap.members`) -- so it imports the constants two ways and
        nothing else of molbuilder.  The masses and element symbols it takes
        are its caller's: a result's own (`equilibrium.masses_amu`), or the
        reader's table for a result that states none (`chemistry`, which
        does not travel).

**Every temperature is explicit** (the user, 2026-11-22, § 6.11 § 9 item 6):
a call given one uses it, and one taken is said -- the rule's one table is
`engines/vibration-frames.md` § 2 (:attr:`ModeState.temperature_from`).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from .constants import (CM1_KELVIN, FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2,
                            ZERO_POINT_Q2_AMU_ANG2_CM1)
except ImportError:                         # beside a job, in a bundle
    from constants import (  # type: ignore[no-redef]
        CM1_KELVIN, FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2,
        ZERO_POINT_Q2_AMU_ANG2_CM1)


# --------------------------------------------------------------------- #
#  The mechanics: the zero-point amplitude, quanta, spread               #
# --------------------------------------------------------------------- #

def _finite_non_negative(raw: Any) -> Optional[float]:
    """``raw`` as a finite number of 0 or more -- never a boolean -- or
    ``None``: the one rule a temperature and a number of quanta follow."""
    try:
        v = float("nan") if isinstance(raw, bool) else float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0.0 else None


def checked_temperature_k(raw: Any) -> float:
    """A temperature as every door takes one: a number of kelvin, 0 or more
    -- 0 K the zero-point motion alone (`science/vibrational-averaging.md`
    § 2) -- refused by name otherwise."""
    t = _finite_non_negative(raw)
    if t is None:
        raise ValueError(f"temperature_k is a number of kelvin, 0 or more; "
                         f"got {raw!r}")
    return t


def zero_point_amplitude_amu12_ang(frequency_cm1: float,
                                   has_imag: bool = False) -> Optional[float]:
    """``Q_zp = sqrt(hbar / 2 omega) = sqrt(C / nu)`` in amu^1/2.A, ``C``
    the constants' ``ZERO_POINT_Q2_AMU_ANG2_CM1`` (§ 2b.1) -- from any
    frequency, the mode's pattern kept; ``None`` for an imaginary mode,
    which has no harmonic width."""
    nu = float(frequency_cm1)
    if has_imag or not nu > 0.0:
        return None
    return math.sqrt(ZERO_POINT_Q2_AMU_ANG2_CM1 / nu)


def mean_quanta(frequency_cm1, temperature_k):
    """A harmonic mode's mean number of quanta at ``temperature_k``, Bose and
    Einstein's ``n = 1 / (exp(hbar omega / k_B T) - 1)`` -- 0 at 0 K (§ 2a).
    Written as ``e^-x / (1 - e^-x)``, which neither overflows for a stiff
    mode in the cold nor loses digits for a soft one in the heat.  A number
    of kelvin, or an array of frequencies (the thermochemistry's sums);
    each frequency positive -- an imaginary mode, or one of no frequency,
    holds no thermal state, and is refused by name."""
    temperature_k = checked_temperature_k(temperature_k)
    nu = np.asarray(frequency_cm1, dtype=float)
    if not np.all(np.isfinite(nu) & (nu > 0.0)):
        raise ValueError(f"a mode's mean quanta need a positive frequency, "
                         f"in cm-1; got {frequency_cm1!r}")
    if temperature_k == 0.0:
        out = np.zeros_like(nu)
    else:
        x = nu * CM1_KELVIN / temperature_k
        out = np.exp(-x) / -np.expm1(-x)
    return float(out) if out.ndim == 0 else out


#: :func:`mean_quanta`, by a name a parameter called ``mean_quanta`` does not
#: hide (:func:`state`).
_occupation = mean_quanta


def spread_of_quanta(zero_point_amplitude: float, quanta: float) -> float:
    """The spread a mean number of quanta gives, in every state,
    ``sqrt(<Q^2>) = Q_zp sqrt(2 n + 1)`` (§ 2a, § 2b.3)."""
    return float(zero_point_amplitude) * math.sqrt(
        2.0 * _checked_quanta(quanta) + 1.0)


def thermal_spread_amu12_ang(zero_point_amplitude: float,
                             frequency_cm1: float,
                             temperature_k: float) -> float:
    """The spread of a harmonic mode's coordinate at ``temperature_k``,
    ``sigma = Q_zp sqrt(2 n(T) + 1) = Q_zp sqrt(coth(hbar omega / 2 k_B T))``
    in amu^1/2.A -- exactly the Gaussian's width at every temperature,
    ``Q_zp`` at 0 K (§ 2): the spread a mean number of quanta gives
    (:func:`spread_of_quanta`), at the temperature's (:func:`mean_quanta`)."""
    return spread_of_quanta(zero_point_amplitude,
                            mean_quanta(frequency_cm1, temperature_k))


def classical_spread_amu12_ang(frequency_cm1: float,
                               temperature_k: float) -> float:
    """The classical spread at ``temperature_k``, ``sqrt(k_B T / omega^2)``
    = ``Q_zp sqrt(2 k_B T / hbar omega)`` -- the high-temperature limit of
    the thermal one (§ 2b.3); 0 at 0 K, the classical mode at rest."""
    temperature_k = checked_temperature_k(temperature_k)
    q_zp = zero_point_amplitude_amu12_ang(frequency_cm1)
    if q_zp is None:
        raise ValueError("an imaginary mode has no classical spread")
    return q_zp * math.sqrt(2.0 * temperature_k
                            / (float(frequency_cm1) * CM1_KELVIN))


#: THE NAMES A MODE'S THERMAL ROW GOES BY -- :attr:`Mode.thermal`'s keys and
#: the record's ``modes_thermal`` block's (`engines/vibration.md` § 6.2): the
#: temperatures, and the spread and mean quanta at each.
THERMAL_ROW_KEYS = ("temperature_k", "thermal_spread_amu12_ang", "mean_quanta")


def thermal_row(frequency_cm1: float, has_imag: bool,
                temperatures_k: Sequence[float]):
    """A mode's spread and mean quanta at each of ``temperatures_k`` --
    ``(spreads, quanta)``, lists in amu^1/2.Å and quanta, or ``(None,
    None)`` for an imaginary mode, which holds no thermal state: the row the
    record states over the modes' own temperatures (`engines/vibration.md`
    § 6.11 § 3)."""
    q = zero_point_amplitude_amu12_ang(frequency_cm1, has_imag)
    if q is None:
        return None, None
    quanta = [mean_quanta(frequency_cm1, t) for t in temperatures_k]
    return [spread_of_quanta(q, n) for n in quanta], quanta


def thermal_spread_of(mode: Mapping[str, Any],
                      temperature_k: float) -> Optional[float]:
    """A mode row's thermal spread at ``temperature_k``, from the values the
    row states -- its zero-point amplitude and its frequency -- or ``None``
    for a mode with no zero-point amplitude (imaginary, or a row written
    before the key): what the spectrum tab's load door serves per mode
    (`web/spectra.md` § 4.1)."""
    temperature_k = checked_temperature_k(temperature_k)
    q = mode.get("zero_point_amplitude_amu12_ang")
    if q is None:
        return None
    return thermal_spread_amu12_ang(float(q), float(mode["frequency_cm1"]),
                                    temperature_k)


# --------------------------------------------------------------------- #
#  The geometry: a coordinate to each atom's move, and back              #
# --------------------------------------------------------------------- #

def mode_over_atoms(mode_free, free_idxs, n_atoms) -> np.ndarray:
    """A mode's pattern over EVERY atom -- its canonical eigenvector on the
    free atoms, zero on every held one, so no held atom moves
    (`engines/vibration.md` § 5.10 ③, § 5.11 § 3): shape ``(n_atoms, 3)``."""
    rows = np.zeros((int(n_atoms), 3))
    rows[list(free_idxs)] = np.asarray(mode_free, dtype=float)
    return rows


def _row_lengths(mode_rows) -> np.ndarray:
    """Each atom's ``|L_A|``, the length of its row of the mode, in Å per
    amu^1/2.Å."""
    return np.linalg.norm(np.asarray(mode_rows, dtype=float), axis=1)


def largest_move(mode_rows) -> float:
    """How far the atom that moves most moves per unit ``Q``: ``max_A
    |L_A|``, the largest of the mode's per-atom vector LENGTHS, in Å per
    amu^1/2.Å -- the one meaning of "the largest move" (§ 6.11 § 2).
    ``mode_rows`` is the mode's pattern, a row per atom (the free atoms' or
    every atom's)."""
    return float(_row_lengths(mode_rows).max())


def q_for_largest_move(mode_rows, move_ang: float) -> float:
    """The coordinate ``Q`` in amu^1/2.Å at which the atom that moves most
    moves ``move_ang`` Å: ``move / max_A |L_A|`` (:func:`largest_move`) --
    a SIESTA Raman phase's ``±δQ`` (`engines/vibration.md` § 5.11 § 3)."""
    return float(move_ang) / largest_move(mode_rows)


def atom_moves(mode_rows, q: float) -> np.ndarray:
    """Every atom's move at the coordinate ``q`` (amu^1/2.Å), in Å along x,
    y, z: ``u_A = q L_A`` -- a held atom's row is zero (§ 2b.2)."""
    return float(q) * np.asarray(mode_rows, dtype=float)


def displaced(positions, mode_rows, q: float) -> np.ndarray:
    """Every atom moved along the mode by ``q`` (amu^1/2.Å): ``R + q L`` --
    the one displacement rule the frame-set maker and a SIESTA Raman
    phase's plan apply."""
    return np.asarray(positions, dtype=float) + atom_moves(mode_rows, q)


def coordinate_of(mode_rows, masses_amu, moves_ang) -> float:
    """The coordinate a set of atom moves gives along the mode, by the
    mass-weighted projection ``Q = sum_A m_A L_A . u_A`` (§ 2b.2) -- exact
    for moves along the mode, since the modes are orthonormal with the
    masses; the part of a move along other modes projects to nothing."""
    rows = np.asarray(mode_rows, dtype=float)
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    moves = np.asarray(moves_ang, dtype=float)
    if masses.shape[0] != rows.shape[0]:
        raise ValueError(f"{masses.shape[0]} masses for a mode over "
                         f"{rows.shape[0]} atoms")
    if moves.shape != rows.shape:
        raise ValueError(f"the moves are a row of x, y, z per atom, every "
                         f"atom's -- shape {rows.shape}; got {moves.shape}")
    return float(np.einsum("a,ai,ai->", masses, rows, moves))


def motion_shares_of(mode_rows, masses_amu) -> np.ndarray:
    """Each atom's share of a mode's motion: ``m_A |L_A|^2`` over its sum
    across the atoms -- the kinetic energy's share, the same at every phase
    of the motion, so a property of the mode (`engines/vibration.md`
    § 6.11 § 3); for the canonical form, over the free atoms, the sum is 1
    already.  A row and a mass per atom, the shares in the rows' order,
    summing to 1; all 0 for a mode with no motion."""
    rows = np.asarray(mode_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"a mode is a row of x, y, z per atom; got the "
                         f"shape {rows.shape}")
    masses = np.asarray(masses_amu, dtype=float).reshape(-1)
    if masses.shape[0] != rows.shape[0]:
        raise ValueError(
            f"{rows.shape[0]} rows of the mode and {masses.shape[0]} masses: "
            f"the mode does not match its atoms")
    weight = masses * np.einsum("ai,ai->a", rows, rows)
    total = float(weight.sum())
    return np.zeros_like(weight) if total <= 0.0 else weight / total


def shares_by(groups: Sequence[Any], shares) -> Dict[str, float]:
    """The shares of :func:`motion_shares_of` summed by each atom's group --
    its element, its species label -- or groups, a tuple of them (the
    structure's regions, which may overlap; an atom in none adds to none):
    one entry per share, the sums largest first; a mode with no motion
    gives ``{}``."""
    values = np.asarray(shares, dtype=float).reshape(-1)
    if len(groups) != values.shape[0]:
        raise ValueError(f"{values.shape[0]} shares and {len(groups)} "
                         f"groups: the mode does not match its atoms")
    if not values.any():
        return {}
    summed: Dict[str, float] = {}
    for group, share in zip(groups, values):
        for name in ((group,) if isinstance(group, str) else group):
            summed[str(name)] = summed.get(str(name), 0.0) + float(share)
    return dict(sorted(summed.items(), key=lambda kv: -kv[1]))


def zero_point_displacement_ang(frequency_cm1: float, has_imag: bool,
                                mode_rows) -> Optional[np.ndarray]:
    """Each atom's move at the zero-point amplitude, in Å, a row per row of
    ``mode_rows`` -- ``u_A = Q_zp L_A`` (§ 2b.2, `engines/vibration.md`
    § 6.3); ``None`` for an imaginary mode, which has no zero-point
    amplitude."""
    q = zero_point_amplitude_amu12_ang(frequency_cm1, has_imag)
    return None if q is None else atom_moves(mode_rows, q)


def largest_zero_point_move_ang(frequency_cm1: float, has_imag: bool,
                                mode_rows) -> Optional[float]:
    """The largest single atom's move at the zero-point amplitude, in Å --
    ``Q_zp`` times the longest row (`engines/vibration.md` § 6.11 § 3);
    ``None`` for an imaginary mode, which has no zero-point amplitude."""
    q = zero_point_amplitude_amu12_ang(frequency_cm1, has_imag)
    return None if q is None else q * largest_move(mode_rows)


def reduced_mass_amu(mode_rows) -> Optional[float]:
    """The mode's reduced mass by Gaussian's convention [Ochterski1999],
    ``mu = 1 / sum_A |L_A|^2`` over the canonical form's rows, in amu --
    what Gaussian prints beside every frequency, so a person compares the
    two directly (`engines/vibration.md` § 6.11 § 3); for H2's stretch the
    atom's mass, 1.008.  ``None`` for a mode with no motion, which has
    none -- the record states it ``null``."""
    rows = np.asarray(mode_rows, dtype=float)
    total = float(np.einsum("ai,ai->", rows, rows))
    return 1.0 / total if total > 0.0 else None


def force_constant_mdyn_per_ang(frequency_cm1: float,
                                reduced_mass: Optional[float]
                                ) -> Optional[float]:
    """The mode's force constant by Gaussian's convention, ``k = mu omega^2``
    in mdyn/Å (`constants.FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2`) -- an
    imaginary mode's positive, ``mu |omega|^2``, as Gaussian prints it;
    ``None`` where the mode has no reduced mass (:func:`reduced_mass_amu`)."""
    if reduced_mass is None:
        return None
    nu = float(frequency_cm1)
    return FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2 * float(reduced_mass) * nu * nu


# --------------------------------------------------------------------- #
#  Where the molecule sits along a mode -- the states (§ 2b.3)           #
# --------------------------------------------------------------------- #

#: THE STATES a mode's coordinate can be in (§ 2b.3), each a kind word.
THERMAL, NUMBER, SWING, CLASSICAL, CLASSICAL_SWING = (
    "thermal", "number", "swing", "classical", "classical swing")


@dataclass(frozen=True, eq=False)
class Frames:
    """A mode's own frames, standing for the set as written
    (`engines/vibration-frames.md` § 3.3, § 4): item ``i`` is frame ``i``,
    frame 0 the relaxed geometry -- an odd rule's middle node, at its
    weight, or, for an even rule, prepended at weight 0.  ``q_amu12_ang``
    each frame's coordinate along the mode (frames x 1), the rule's nodes
    in the order ``(0, +q1, -q1, +q2, -q2, ...)``; ``weights`` each frame's
    share of the average, summing to 1; ``exact_to_degree`` the polynomial
    degree the rule averages exactly; ``state`` the mode's state the frames
    were made for (``None`` for a bare :class:`State`'s, which has no
    atoms); ``positions_ang`` each frame's geometry, every atom at
    ``R + Q L`` (frames x atoms x 3; ``None`` without a mode)."""
    q_amu12_ang: np.ndarray
    weights: np.ndarray
    exact_to_degree: int
    state: Optional["ModeState"] = field(default=None, repr=False)
    positions_ang: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def mode_index_1based(self) -> np.ndarray:
        """The column's mode, ``(1,)`` -- 0 for a bare state's frames."""
        mode = None if self.state is None else self.state.mode
        return np.array([0 if mode is None else int(mode.index_1based)])

    @property
    def atom_moves_ang(self) -> Optional[np.ndarray]:
        """Each frame's every-atom move, ``Q L`` (frames x atoms x 3) -- the
        vibration's one rule (:meth:`Vibration._moves`)."""
        if self.state is None:
            return None
        return self.state.mode._v._moves(self.mode_index_1based,
                                         self.q_amu12_ang)

    @property
    def max_atom_displacement_ang(self) -> Optional[np.ndarray]:
        """Each frame's largest single-atom move, Å (frames,)."""
        return None if self.state is None else self._largest[0]

    @property
    def max_atom_mode_index_1based(self) -> Optional[np.ndarray]:
        """The mode moving each frame's most-moved atom -- this set's one
        mode, 0 for a frame that moves nothing (frame 0) (frames,)."""
        return None if self.state is None else self._largest[1]

    @property
    def _largest(self) -> Tuple[np.ndarray, np.ndarray]:
        return _largest_moves(self.state.mode._v, self.mode_index_1based,
                              self.q_amu12_ang)

    @property
    def said(self) -> Tuple[str, ...]:
        """What the frames say (`engines/vibration-frames.md` § 4): the
        largest single-atom move of the outermost frames -- the check a
        person reads against a bond length."""
        largest = self.max_atom_displacement_ang
        if largest is None or not len(largest):
            return ()
        i = int(np.argmax(largest))
        return (f"the outermost frames move an atom by at most "
                f"{largest[i]:.4f} Å ({_frame_words(i, len(largest))}, q = "
                f"{self.q_amu12_ang[i, 0]:+.4f} amu^1/2·Å)",)


@dataclass(frozen=True, eq=False)
class Weights:
    """The weights for a state of frames (`engines/vibration-frames.md`
    § 7; science § 5.4): ``weights`` one per frame, summing to 1;
    ``beyond`` the share of the state's motion beyond the outermost frames,
    both sides; ``exact_to_degree`` the degree they reproduce the state's
    averages to (:meth:`State.weights_at`); ``said`` -- for frames placed for
    another motion -- the share beyond and, for three frames, that they are
    blind to the state's shape (for fewer, their degree), and for any frames
    each weight below zero: said, never refused."""
    weights: np.ndarray
    beyond: float
    exact_to_degree: int
    said: Tuple[str, ...] = ()


def _largest_moves(vibration: "Vibration", numbers, q_amu12_ang
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Each frame's largest single-atom move, and the mode moving that atom
    most, ``|Q_k| |L_k,A|`` at it -- 0 for a frame that moves nothing; one
    rule for a mode's frames and for snapshots."""
    q = np.atleast_2d(np.asarray(q_amu12_ang, dtype=float))
    lengths = np.linalg.norm(vibration._moves(numbers, q), axis=-1)
    largest = lengths.max(axis=1)
    atom = lengths.argmax(axis=1)
    rows = np.linalg.norm(vibration._rows(numbers), axis=-1)   # modes x atoms
    mover = np.asarray(numbers)[(np.abs(q) * rows[:, atom].T).argmax(axis=1)]
    return largest, np.where(largest > 0.0, mover, 0)


def _gauss_hermite(nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """The Gauss–Hermite rule for the standard normal: points and weights
    summing to 1 (§ 5.1)."""
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    return math.sqrt(2.0) * x, w / math.sqrt(math.pi)


def _gauss_rule_of(points, probabilities, nodes: int
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """The ``nodes``-point Gauss rule of a discrete measure -- its points
    and their probabilities -- by Lanczos with full reorthogonalisation on
    the diagonal operator from the probabilities' square roots and Golub
    and Welsch's eigenproblem [GolubWelsch1969]: exact for the measure's
    moments to degree ``2 nodes - 1`` (the measure holding ``nodes``
    distinct points or more)."""
    x = np.asarray(points, dtype=float)
    q = np.sqrt(np.asarray(probabilities, dtype=float))
    basis: List[np.ndarray] = []
    alpha: List[float] = []
    beta: List[float] = []
    for _ in range(nodes):
        basis.append(q)
        w = x * q
        a = float(q @ w)
        alpha.append(a)
        for v in basis:                     # full reorthogonalisation
            w = w - (v @ w) * v
        b = float(np.linalg.norm(w))
        beta.append(b)
        if b == 0.0:
            break
        q = w / b
    return _jacobi_rule(alpha, beta)


def _jacobi_rule(alpha: Sequence[float], beta: Sequence[float]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Golub and Welsch's step [GolubWelsch1969]: the Gauss rule of a
    recurrence's coefficients -- the Jacobi matrix's eigenvalues its nodes,
    the squares of its eigenvectors' first components its weights."""
    m = len(alpha)
    off = np.array(beta[:m - 1])
    jacobi = np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1)
    node, vec = np.linalg.eigh(jacobi)
    return node, vec[0] ** 2


def _symmetric_ordered(nodes, weights) -> Tuple[np.ndarray, np.ndarray]:
    """An even density's rule made exactly symmetric and put in the set's
    order (`engines/vibration-frames.md` § 4): an odd rule's middle node 0
    exactly; each pair ``+/-(x_+ - x_-)/2`` with the pair's two weights
    averaged; then ``(0, +q1, -q1, +q2, -q2, ...)``, nearest first, its
    positive side first -- an even rule's frame 0, the relaxed geometry,
    prepended at weight 0."""
    order = np.argsort(np.asarray(nodes, dtype=float))
    x = np.asarray(nodes, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    m = len(x)
    q_out: List[float] = [0.0]
    w_out: List[float] = [float(w[m // 2]) if m % 2 else 0.0]
    for i in range(m // 2 - 1, -1, -1):     # the pair nearest first
        lo, hi = i, m - 1 - i
        q = (x[hi] - x[lo]) / 2.0
        pw = (w[hi] + w[lo]) / 2.0
        q_out += [q, -q]
        w_out += [pw, pw]
    return np.array(q_out), np.array(w_out)


def _whole(value: Any, what: str, least: int) -> int:
    """``value`` as a whole number of ``least`` or more -- any integer type,
    numpy's too, never a boolean -- refused by name otherwise."""
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or int(value) < least):
        raise ValueError(f"{what} is a whole number, {least} or more; got "
                         f"{value!r}")
    return int(value)


_erf = np.frompyfunc(math.erf, 1, 1)
_erfc = np.frompyfunc(math.erfc, 1, 1)


def _at_rest_density(x):
    """A coordinate at rest at 0: all of it there, a point -- its density
    infinite at 0 and 0 elsewhere."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0.0, np.inf, 0.0)


def _gaussian_density(x, sigma: float):
    """The centred Gaussian of spread ``sigma``, its density at ``x`` --
    at rest for a spread of 0."""
    if sigma == 0.0:
        return _at_rest_density(x)
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def _gaussian_distribution(x, sigma: float):
    """The centred Gaussian of spread ``sigma``, the chance of ``x`` or
    less: ``(1 + erf(x / sqrt(2) sigma)) / 2`` -- for a spread of 0, 1 from
    0 on; arrays in, arrays out."""
    x = np.asarray(x, dtype=float)
    if sigma == 0.0:
        return np.where(x >= 0.0, 1.0, 0.0)
    return 0.5 * (1.0 + np.asarray(_erf(x / (math.sqrt(2.0) * sigma)),
                                   dtype=float))


def _gaussian_moment(sigma: float, k: int) -> float:
    """``<Q^k>`` of a centred Gaussian: ``sigma^k (k-1)!!`` for even ``k``,
    0 for odd."""
    if k % 2:
        return 0.0
    return sigma ** k * math.prod(range(k - 1, 0, -2))


def _cos_power_mean(p: int) -> float:
    """The mean of ``cos^p`` over a period: ``C(p, p/2) / 2^p`` for even
    ``p``, 0 for odd."""
    if p % 2:
        return 0.0
    return math.comb(p, p // 2) / 2.0 ** p


def _eigenfunction(n: int, xi) -> np.ndarray:
    """The oscillator's ``n``-th eigenfunction, ``psi_n(xi)``, normalised in
    ``xi`` -- by the recurrence ``h_{k+1} = sqrt(2/(k+1)) xi h_k -
    sqrt(k/(k+1)) h_{k-1}`` from ``h_0 = pi^-1/4`` on ``H_n`` scaled, each
    value rescaled where it grows past 1e100 and the scale kept as a
    logarithm, the Gaussian applied once at the end: ``psi_n = h_n
    e^{log scale - xi^2/2}`` -- no ``n`` and no ``xi`` overflows or loses
    the weight near the turning points."""
    xi = np.asarray(xi, dtype=float)
    h_prev = np.zeros_like(xi)
    h = np.full_like(xi, math.pi ** -0.25)
    log_scale = np.zeros_like(xi)
    for k in range(n):
        h_prev, h = h, (math.sqrt(2.0 / (k + 1)) * xi * h
                        - math.sqrt(k / (k + 1)) * h_prev)
        big = np.abs(h) > 1e100
        if np.any(big):
            h = np.where(big, h * 1e-100, h)
            h_prev = np.where(big, h_prev * 1e-100, h_prev)
            log_scale = np.where(big, log_scale + 100.0 * math.log(10.0),
                                 log_scale)
    return h * np.exp(log_scale - 0.5 * xi * xi)


def _position_operator(size: int) -> np.ndarray:
    """``xi = (a + a^dagger) / sqrt(2)`` on the oscillator's first ``size``
    states: the matrix ``<j|xi|j+1> = sqrt((j+1)/2)`` -- exact on every state
    ``xi`` raises from below ``size - 1``."""
    off = np.sqrt(np.arange(1, size) / 2.0)
    return np.diag(off, 1) + np.diag(off, -1)


def _number_state_rule(n: int, nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """The ``nodes``-point Gauss rule whose weight function is ``P_n`` (in
    ``xi``): the spectral measure of ``xi`` in the state ``|n>``, by
    Lanczos from ``|n>`` and Golub and Welsch's eigenproblem -- exact for
    its polynomials to degree ``2 nodes - 1``, for any ``n``."""
    size = n + nodes + 1
    x_op = _position_operator(size)
    q = np.zeros(size)
    q[n] = 1.0
    q_prev = np.zeros(size)
    alpha: List[float] = []
    beta: List[float] = []
    b = 0.0
    for _ in range(nodes):
        w = x_op @ q - b * q_prev
        a = float(q @ w)
        w = w - a * q
        alpha.append(a)
        b = float(np.linalg.norm(w))
        beta.append(b)
        q_prev, q = q, (w / b if b > 0.0 else w)
    return _jacobi_rule(alpha, beta)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


class _NumberDistribution:
    """``P_n``'s distribution in ``xi = Q / sqrt(2) Q_zp``, tabulated once
    per ``n``: ``2 (n + 20)`` panels -- one to two between neighbouring
    zeros of ``P_n``, whose ``n`` zeros lie within ``±sqrt(2n + 1)`` -- each by
    40-point Gauss–Legendre, exact to degree 79, so each panel's integral to
    its last digits; out to nine past the far turning point, ``e^-81`` of the
    density beyond; the running integral at every panel's edge -- the chance
    of ``xi`` or less at any point the table's edge plus that panel's own
    piece, every point asked at once."""

    def __init__(self, n: int):
        self.n = n
        self.reach = math.sqrt(2.0 * n + 1.0) + 9.0
        panels = 2 * (n + 20)
        self.edges = np.linspace(-self.reach, self.reach, panels + 1)
        half = (self.edges[1] - self.edges[0]) / 2.0
        mids = (self.edges[:-1] + self.edges[1:]) / 2.0
        points = mids[:, None] + half * _GL_X[None, :]
        pieces = half * (self.density(points) * _GL_W[None, :]).sum(axis=1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(pieces)))

    def density(self, xi) -> np.ndarray:
        psi = _eigenfunction(self.n, xi)
        return psi * psi

    def below(self, xi) -> np.ndarray:
        """The chance of ``xi`` or less, ``xi`` any array."""
        xi = np.clip(np.asarray(xi, dtype=float), -self.reach, self.reach)
        k = np.clip(np.searchsorted(self.edges, xi, side="right") - 1, 0,
                    len(self.edges) - 2)
        left = self.edges[k]
        half = (xi - left) / 2.0
        points = left[..., None] + half[..., None] * (_GL_X + 1.0)
        piece = half * (self.density(points) * _GL_W).sum(axis=-1)
        return self.cumulative[k] + piece

    def lower_quantile(self, p) -> np.ndarray:
        """The ``xi`` whose chance of ``xi`` or less is ``p``, for ``p`` of
        one half or less -- found from the left, where the running integral
        keeps its digits: the table's panel, then Newton within it, kept in
        the panel by bisection, each point stopped once its step is within
        four units of its last place, where Newton's last steps ring -- sixty
        steps at most, enough for bisection alone to halve a panel below its
        last place: a point beside a zero of the density may ring wider and
        run them all, its answer still good to the distribution's last
        bit."""
        p = np.asarray(p, dtype=float)
        shape, p = p.shape, p.reshape(-1)
        k = np.clip(np.searchsorted(self.cumulative, p, side="right") - 1, 0,
                    len(self.edges) - 2)
        lo, hi = self.edges[k].copy(), self.edges[k + 1].copy()
        span = np.maximum(self.cumulative[k + 1] - self.cumulative[k],
                          1e-300)
        x = lo + (hi - lo) * np.clip((p - self.cumulative[k]) / span, 0, 1)
        active = np.ones(p.shape[0], dtype=bool)
        for _ in range(60):
            at = np.flatnonzero(active)
            if not at.size:
                break
            xa = x[at]
            f = self.below(xa) - p[at]
            lo[at] = np.where(f < 0.0, xa, lo[at])
            hi[at] = np.where(f > 0.0, xa, hi[at])
            d = self.density(xa)
            newton = xa - np.where(d > 0.0, f / np.where(d > 0.0, d, 1.0),
                                   np.inf)
            inside = (newton >= lo[at]) & (newton <= hi[at])
            step = np.where(inside, newton, (lo[at] + hi[at]) / 2.0)
            done = (f == 0.0) | (np.abs(step - xa) <= 4.0 * np.spacing(
                np.maximum(1.0, np.abs(xa))))
            x[at] = step
            active[at[done]] = False
        return x.reshape(shape)


_NUMBER_TABLES: Dict[int, _NumberDistribution] = {}


def _number_distribution(n: int) -> _NumberDistribution:
    if n not in _NUMBER_TABLES:
        _NUMBER_TABLES[n] = _NumberDistribution(n)
    return _NUMBER_TABLES[n]


def _stream(key: Tuple[int, ...], size: int, start: int, *,
            uniform: bool = False) -> np.ndarray:
    """One stream of the seed (`engines/vibration-frames.md` § 3.3):
    ``numpy.random.default_rng([seed, mode, 0])`` a mode's normal numbers,
    ``[seed, mode, 1]`` its uniforms for its phases, ``[seed, 0, *modes]`` a
    drive group's, keyed by the modes it was made with -- drawn in one block
    from the first, so draws ``start ... start + size - 1`` are the same
    however many were asked."""
    rng = np.random.default_rng(list(key))
    block = (rng.random(start + size) if uniform
             else rng.standard_normal(start + size))
    return block[start:]


def _phases_of(uniforms) -> np.ndarray:
    """A swing's phases from a stream's uniforms, ``2 pi u``."""
    return 2.0 * math.pi * np.asarray(uniforms, dtype=float)


def _checked_seed(seed: Any) -> int:
    """A seed, a whole number 0 or more -- required, refused by name."""
    if seed is None:
        raise ValueError("a draw needs its seed, a whole number 0 or more: "
                         "the same seed gives the same draws")
    return _whole(seed, "seed", 0)


@dataclass(frozen=True)
class State:
    """Where a mode's coordinate ``Q`` (amu^1/2.Å) sits, in one state
    (§ 2b.3).  ``kind`` one of :data:`THERMAL`, :data:`NUMBER`,
    :data:`SWING`, :data:`CLASSICAL`, :data:`CLASSICAL_SWING`;
    ``zero_point_amplitude_amu12_ang`` the mode's ``Q_zp``; ``bath_quanta``
    the thermal (or random) quanta -- a driven mode's warmth, 0 for the
    coherent state; ``quanta`` a number state's; ``swing_quanta`` a swing's;
    ``classical_spread_amu12_ang`` a classical bell curve's, alone or
    carried by a classical swing."""
    kind: str
    zero_point_amplitude_amu12_ang: float
    bath_quanta: float = 0.0
    quanta: int = 0
    swing_quanta: float = 0.0
    classical_spread_amu12_ang: float = 0.0

    # ---- the numbers every state states ----------------------------------
    @property
    def _q_zp(self) -> float:
        return float(self.zero_point_amplitude_amu12_ang)

    @property
    def _bath_spread(self) -> float:
        return spread_of_quanta(self._q_zp, self.bath_quanta)

    @property
    def swing_amplitude_amu12_ang(self) -> float:
        """A swing's amplitude ``Q_0 = 2 Q_zp sqrt(n_s)`` -- its quanta
        ``Q_0^2 / 4 Q_zp^2`` (§ 2a.2); 0 for another state."""
        return 2.0 * self._q_zp * math.sqrt(self.swing_quanta)

    @property
    def carried_spread_amu12_ang(self) -> Optional[float]:
        """The bell curve a swing carries: the zero-point width ``Q_zp`` of
        the coherent state, a warm driven mode's thermal spread at that
        temperature, the classical bell curve a classical swing carries (0
        for the bare classical swing) (§ 2a.2); ``None`` for a state that
        does not swing."""
        if self.kind == SWING:
            return self._bath_spread
        if self.kind == CLASSICAL_SWING:
            return float(self.classical_spread_amu12_ang)
        return None

    @property
    def mean_quanta(self) -> Optional[float]:
        """The state's total mean number of quanta: a number state's own,
        the bath's, a swing's added to its warmth's (§ 2a.2); ``None`` for a
        classical state, which counts none."""
        if self.kind == NUMBER:
            return float(self.quanta)
        if self.kind in (CLASSICAL, CLASSICAL_SWING):
            return None
        return float(self.bath_quanta) + float(self.swing_quanta)

    @property
    def spread_amu12_ang(self) -> float:
        """The r.m.s. coordinate, ``sqrt(<Q^2>)``: in every quantum state the
        spread its mean quanta give, ``Q_zp sqrt(2 <n> + 1)``
        (:func:`spread_of_quanta`, § 2b.3); a classical state's from its
        second moment."""
        quanta = self.mean_quanta
        if quanta is None:                          # a classical state
            return math.sqrt(self.moment(2))
        return spread_of_quanta(self._q_zp, quanta)

    @property
    def _gaussian_spread(self) -> float:
        if self.kind == CLASSICAL:
            return float(self.classical_spread_amu12_ang)
        return self._bath_spread

    # ---- the density, the distribution, the moments ----------------------
    def pdf(self, q):
        """The probability density of ``Q`` at ``q`` (per amu^1/2.Å)."""
        q = np.asarray(q, dtype=float)
        if self.kind in (THERMAL, CLASSICAL):
            out = _gaussian_density(q, self._gaussian_spread)
        elif self.kind == NUMBER:
            xi = q / (math.sqrt(2.0) * self._q_zp)
            psi = _eigenfunction(self.quanta, xi)
            out = psi * psi / (math.sqrt(2.0) * self._q_zp)
        else:
            s, q0 = self.carried_spread_amu12_ang, self.swing_amplitude_amu12_ang
            if s > 0.0:
                centres = q0 * np.cos(_swing_phases(q0, s))
                out = _gaussian_density(q[..., None] - centres,
                                        s).mean(axis=-1)
            else:                                   # the bare classical swing
                inside = np.abs(q) < q0
                out = (_at_rest_density(q) if q0 == 0.0 else
                       np.where(inside, 1.0 / (math.pi * np.sqrt(
                           np.where(inside, q0 * q0 - q * q, 1.0))), 0.0))
        return float(out) if out.ndim == 0 else out

    def cdf(self, q):
        """The probability that ``Q`` is ``q`` or less -- arrays in, arrays
        out."""
        q = np.asarray(q, dtype=float)
        if self.kind in (THERMAL, CLASSICAL):
            out = _gaussian_distribution(q, self._gaussian_spread)
        elif self.kind == NUMBER:
            table = _number_distribution(self.quanta)
            out = np.clip(table.below(q / (math.sqrt(2.0) * self._q_zp)),
                          0.0, 1.0)
        else:
            s, q0 = self.carried_spread_amu12_ang, self.swing_amplitude_amu12_ang
            if s > 0.0:
                centres = q0 * np.cos(_swing_phases(q0, s))
                out = _gaussian_distribution(q[..., None] - centres,
                                             s).mean(axis=-1)
            elif q0 == 0.0:
                out = np.where(q >= 0.0, 1.0, 0.0)
            else:                                   # the bare classical swing
                out = 0.5 + np.arcsin(np.clip(q / q0, -1.0, 1.0)) / math.pi
        return float(out) if np.ndim(out) == 0 else out

    def moment(self, k: int) -> float:
        """``<Q^k>`` exactly, for a whole ``k`` of 0 or more: the Gaussian's
        ``sigma^k (k-1)!!``; a number state's ``<n|xi^k|n>`` by the
        oscillator's own position operator; a swing's by the binomial over
        the bell curve it carries and its phase (§ 2b.3)."""
        k = _whole(k, "a moment's order", 0)
        if self.kind in (THERMAL, CLASSICAL):
            return _gaussian_moment(self._gaussian_spread, k)
        if self.kind == NUMBER:
            if k % 2:
                return 0.0
            n = self.quanta
            x_op = _position_operator(n + k // 2 + 1)
            v = np.zeros(n + k // 2 + 1)
            v[n] = 1.0
            for _ in range(k // 2):
                v = x_op @ v
            return float(v @ v) * (math.sqrt(2.0) * self._q_zp) ** k
        q0, s = self.swing_amplitude_amu12_ang, self.carried_spread_amu12_ang
        return sum(math.comb(k, j) * _gaussian_moment(s, j)
                   * q0 ** (k - j) * _cos_power_mean(k - j)
                   for j in range(k + 1))

    # ---- the frames ------------------------------------------------------
    def _rule(self, m: int) -> Tuple[np.ndarray, np.ndarray]:
        """The ``m``-point Gauss rule of this state's own distribution
        (science § 2a.3, § 2b.4), nodes and weights as found."""
        if self.kind in (THERMAL, CLASSICAL):
            x, w = _gauss_hermite(m)
            return self._gaussian_spread * x, w
        if self.kind == NUMBER:
            x, w = _number_state_rule(self.quanta, m)
            return math.sqrt(2.0) * self._q_zp * x, w
        q0, s = self.swing_amplitude_amu12_ang, self.carried_spread_amu12_ang
        if s == 0.0:                       # Chebyshev's: the bare swing
            j = np.arange(1, m + 1)
            return q0 * np.cos((2 * j - 1) * math.pi / (2 * m)), np.full(m, 1.0 / m)
        if q0 == 0.0:                      # no swing: the bell curve it carries
            x, w = _gauss_hermite(m)
            return s * x, w
        # the swing's own measure, exact to degree 2m - 1: 2m + 2 equal
        # phases, each carrying the bell curve's (m + 1)-point rule
        phases = 2.0 * math.pi * np.arange(2 * m + 2) / (2 * m + 2)
        xb, wb = _gauss_hermite(m + 1)
        points = (q0 * np.cos(phases)[:, None] + s * xb[None, :]).ravel()
        probabilities = (np.full(len(phases), 1.0 / len(phases))[:, None]
                         * wb[None, :]).ravel()
        return _gauss_rule_of(points, probabilities, m)

    def frames(self, nodes: int = 3) -> Frames:
        """The frames that average a response over the state (§ 2b.4,
        `engines/vibration-frames.md` § 4): the Gauss rule of the state's own
        distribution -- Gauss–Hermite for a bell curve, the rule whose
        weight function is the state's density for n quanta and for a
        swing -- exact for a response polynomial to degree ``2 nodes - 1``,
        exactly symmetric, in the order ``(0, +q1, -q1, ...)``, an even
        rule's frame 0 prepended at weight 0."""
        m = _whole(nodes, "nodes", 1)
        x, w = self._rule(m)
        q, weights = _symmetric_ordered(x, w)
        return Frames(q[:, None], weights, 2 * m - 1)

    # ---- the draws -------------------------------------------------------
    def draw(self, size: int, *, seed: int, start: int = 0,
             phase=None, mode: int = 0) -> np.ndarray:
        """``size`` coordinates drawn from ``P(Q)`` -- draws ``start`` ...
        ``start + size - 1`` of the seed's sequence -- from the mode's own
        streams (`engines/vibration-frames.md` § 2, § 3.3): one standard
        normal number ``z`` per draw for every state -- a bell curve, its
        spread times ``z``; a swing, ``Q_0 cos(phi) + s z``, ``s`` the
        bell curve it carries and ``phi`` its phase (its own uniform stream,
        or ``phase`` where a drive group gives it); n quanta, the ``Q``
        where the chance of ``Q`` or less is ``Phi(z)``, taken from the left
        with ``z``'s sign, so ``-z`` gives ``-Q`` exactly.  ``mode`` the
        mode's number the streams are keyed by (0 for a bare state)."""
        size = _whole(size, "size", 0)
        seed = _checked_seed(seed)
        start = _whole(start, "start", 0)
        z = _stream((seed, int(mode), 0), size, start)
        return self._at(z, self._phases(size, seed, start, phase, mode))

    def _phases(self, size, seed, start, phase, mode) -> Optional[np.ndarray]:
        """A swing's phases -- given, or its own uniform stream's."""
        if self.kind not in (SWING, CLASSICAL_SWING):
            return None
        if phase is not None:
            phase = np.asarray(phase, dtype=float).reshape(-1)
            if phase.shape[0] != size:
                raise ValueError(f"{phase.shape[0]} phases for {size} draws")
            return phase
        return _phases_of(_stream((seed, int(mode), 1), size, start,
                                  uniform=True))

    def _at(self, z, phases) -> np.ndarray:
        """The coordinate each normal number ``z`` (and phase) gives."""
        z = np.asarray(z, dtype=float)
        if self.kind in (THERMAL, CLASSICAL):
            return self._gaussian_spread * z
        if self.kind == NUMBER:
            table = _number_distribution(self.quanta)
            left = table.lower_quantile(0.5 * np.asarray(
                _erfc(np.abs(z) / math.sqrt(2.0)), dtype=float))
            xi = np.where(z == 0.0, 0.0, -np.sign(z) * left)
            return math.sqrt(2.0) * self._q_zp * xi
        return (self.swing_amplitude_amu12_ang * np.cos(phases)
                + self.carried_spread_amu12_ang * z)

    # ---- the weights for frames placed elsewhere -------------------------
    def weights_at(self, q_amu12_ang, *, written=None) -> Weights:
        """The weights with which frames at ``q_amu12_ang`` reproduce this
        state's averages of ``1, Q, ..., Q^(m-1)`` (science § 5.4) -- and
        the degree they reproduce its averages to, ``exact_to_degree``: at
        least ``m - 1``, one more for an odd number of symmetric frames and
        a symmetric state, up to ``2m - 1`` where the m frames are the
        state's own Gauss rule (§ 2b.4).  ``written``, the weights the frames
        were written with: where this state's are those, the frames were
        placed for it -- an even rule's prepended frame 0 among them.  For
        frames placed for another motion, said, never refused: the share of
        the state's motion beyond the outermost frames, and for three frames
        that they are blind to the state's shape (for fewer, their degree);
        a weight below zero, past rounding, for any.  Two frames at one
        position refused by name, no weights separating them."""
        q = np.asarray(q_amu12_ang, dtype=float).reshape(-1)
        m = q.shape[0]
        if m == 0:
            raise ValueError("weights need one frame at least")
        for i in range(m):
            for j in range(i + 1, m):
                if q[i] == q[j]:
                    raise ValueError(
                        f"{_frame_words(i, m)} and {_frame_words(j, m)} sit "
                        f"at one position, q = {q[i]:g} amu^1/2·Å: no "
                        f"weights separate them")
        scale = self.spread_amu12_ang or 1.0
        x = q / scale
        a = np.vander(x, m, increasing=True).T
        b = np.array([self.moment(k) / scale ** k for k in range(m)])
        w = np.linalg.solve(a, b)
        degree = self._degree_reproduced(w, x, scale, m)
        # THE STATE'S OWN RULE -- its m frames' weights reproduce its
        # averages as far as a Gauss rule on m frames can -- is no
        # re-weighting to another motion: science § 5.4's items 1 and 2 (the
        # share beyond, the blindness) are not said of it; a weight below
        # zero, item 3, is said of any frames.  Counted on every frame: frames placed for another
        # motion whose weights happen to fall on a shorter rule of this
        # state (a node at weight 0) are still said of.
        own = degree >= 2 * m - 1 or (
            written is not None
            and np.allclose(w, np.asarray(written, dtype=float).reshape(-1),
                            rtol=0.0, atol=_WEIGHT_ROUNDING))
        cdf = np.atleast_1d(self.cdf(np.array([q.min(), q.max()])))
        beyond = float(max(0.0, cdf[0]) + max(0.0, 1.0 - cdf[1]))
        said: List[str] = []
        if not own and beyond > 0.0:
            said.append(f"{100.0 * beyond:.3g} % of the state's motion lies "
                        f"beyond the outermost frames (q from {q.min():+.4g} "
                        f"to {q.max():+.4g} amu^1/2·Å), averaged by values "
                        f"the frames never computed")
        if not own and m == 3:
            said.append("three frames carry only the curvature: up to it a "
                        "swing and a bell curve of one spread agree, so the "
                        "re-averaged number is blind to the state's shape")
        elif not own and m < 3:
            said.append(f"{'one frame is' if m == 1 else 'two frames are'} "
                        f"exact only to degree {degree}")
        for i, wi in enumerate(w):
            if wi < -_WEIGHT_ROUNDING:
                said.append(f"{_frame_words(i, m)}'s weight is {wi:.4g}, below "
                            f"zero: the frames do not suit the state's shape "
                            f"well")
        return Weights(w, beyond, degree, tuple(said))

    def _degree_reproduced(self, w, x, scale: float, m: int) -> int:
        """The highest degree ``d`` (``m - 1`` up to ``2m - 1``) to which the
        weights ``w`` at the scaled positions ``x`` reproduce every average
        of ``1, x, ..., x^d`` -- each within 1e-9 of the sums it is: a
        state's own rule reproduces its averages to ~1e-14 of them, and the
        first degree frames do not reproduce is off by ~4e-5 or more (every
        state kind, 1 to 9 frames, worked in exact arithmetic, R3d2b2b1's
        review); the bound sits between them."""
        degree = m - 1
        for k in range(m, 2 * m):
            want = self.moment(k) / scale ** k
            terms = w * x ** k
            if abs(float(terms.sum()) - want) > 1e-9 * max(
                    float(np.abs(terms).sum()), abs(want), 1.0):
                break
            degree = k
        return degree


def _swing_phases(amplitude: float, carried_spread: float) -> np.ndarray:
    """The equal phases a swing's density and distribution are averaged over
    (§ 2a.3): sixteen per carried spread of the swing's amplitude, 256 at
    least, so the swing's centre moves at most ``2 pi / 16`` of that spread
    between two -- the equal-phase rule on a periodic Gaussian then
    converged to the last digits for a swing of any size."""
    count = max(256, 16 * math.ceil(amplitude / carried_spread))
    return 2.0 * math.pi * np.arange(count) / count


#: WHAT NAMES EACH STATE (§ 2b.3, `engines/vibration-frames.md` § 2): the
#: arguments a person gives, exactly one of these sets -- a bath named by
#: its temperature or by its mean quanta; a driven mode by its swing quanta
#: alone (the coherent state) or with its warmth's temperature.
_STATES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (NUMBER, ("quanta",)),
    (THERMAL, ("temperature_k",)),
    (THERMAL, ("mean_quanta",)),
    (SWING, ("swing_quanta",)),
    (SWING, ("swing_quanta", "temperature_k")),
    (CLASSICAL, ("classical", "temperature_k")),
    (CLASSICAL_SWING, ("classical", "swing_quanta")),
    (CLASSICAL_SWING, ("classical", "swing_quanta", "temperature_k")),
)
_STATE_NAMED_BY: Dict[frozenset, str] = {
    frozenset(names): kind for kind, names in _STATES}


def _frame_words(frame: int, n_frames: int) -> str:
    """A frame as a message says it, from 1: ``frame 2 of 3`` -- the
    spelling of `structure.frame_words`, the one spelling of a frame in a
    message (`model/structure.md` § 2.2e), kept here because this module
    travels with numpy and the constants alone."""
    return f"frame {int(frame) + 1} of {int(n_frames)}"


#: A WEIGHT WITHIN ROUNDING OF ZERO: a frame the solve weights 0 in exact
#: arithmetic -- a prepended frame 0, a node the state does not reach --
#: comes out a few ulps either side of it, never "below zero".
_WEIGHT_ROUNDING = 1e-12


#: WHERE A STATE'S TEMPERATURE CAME FROM (§ 6.11 § 9 item 6,
#: `engines/vibration-frames.md` § 2's table) -- a set's for its rows'
#: weights (`vibration_frames.weights_for_set`, § 7).
GIVEN, THE_RECORDS, THE_MOTIONS, THE_SETS = (
    "given", "the record's", "the motion's", "the set's")

#: WHAT HOLDS EACH TEMPERATURE TAKEN, as its refusal names it when it states
#: none.
_HOLDERS = {THE_RECORDS: "the record", THE_MOTIONS: "the motion",
            THE_SETS: "the set"}


def _temperature_of(named: frozenset, temperature_k: Optional[float],
                    default: Optional[float], source: str):
    """The temperature a state is asked at and where it came from -- § 2's
    one table: given, used (:data:`GIVEN`); not given where the arguments
    name a state once a temperature is added (a thermal or classical state),
    ``default`` and its ``source`` (:data:`THE_RECORDS`,
    :data:`THE_MOTIONS`, :data:`THE_SETS`), a ``default`` of none refused by
    name; otherwise none -- n quanta, mean quanta, a driven mode, a
    classical swing alone."""
    if temperature_k is not None:
        return temperature_k, GIVEN
    if (named not in _STATE_NAMED_BY
            and named | {"temperature_k"} in _STATE_NAMED_BY):
        if default is None:
            raise ValueError(f"name a temperature: {_HOLDERS[source]} "
                             f"states none")
        return default, source
    return None, None


def _named(temperature_k, mean_quanta, quanta, swing_quanta,
           classical) -> frozenset:
    """The arguments a call names: each one given, ``classical`` when
    true."""
    given = {"temperature_k": temperature_k, "mean_quanta": mean_quanta,
             "quanta": quanta, "swing_quanta": swing_quanta,
             "classical": True if classical else None}
    return frozenset(name for name, value in given.items()
                     if value is not None)


def state(zero_point_amplitude: float, frequency_cm1: float, *,
          temperature_k: Optional[float] = None,
          mean_quanta: Optional[float] = None,
          quanta: Optional[int] = None,
          swing_quanta: Optional[float] = None,
          classical: bool = False) -> State:
    """The state a person names (§ 2b.3), by one of the sets
    :data:`_STATES` lists:

    * ``quanta`` -- n quanta exactly;
    * ``temperature_k`` or ``mean_quanta`` -- thermal (or random quanta);
    * ``swing_quanta`` -- driven, the coherent state: a bell curve of the
      zero-point width centred on the swing; with ``temperature_k``, a warm
      driven mode, the swing carrying that temperature's bell curve;
    * ``classical`` with ``temperature_k`` -- the classical bell curve, at
      rest at 0 K; with ``swing_quanta``, the classical swing alone; with
      both, the classical swing carrying the classical bell curve.

    Any other set is refused, naming the sets."""
    q_zp = float(zero_point_amplitude)
    if not (math.isfinite(q_zp) and q_zp > 0.0):
        raise ValueError("a state needs the mode's zero-point amplitude -- "
                         "an imaginary mode has none")
    named = _named(temperature_k, mean_quanta, quanta, swing_quanta,
                   classical)
    kind = _STATE_NAMED_BY.get(named)
    if kind is None:
        ways = "; ".join(" with ".join(names) for _, names in _STATES)
        raise ValueError(f"a state is named one of these ways: {ways} -- "
                         f"got {', '.join(sorted(named)) or 'nothing'}")
    if kind == NUMBER:
        return State(NUMBER, q_zp, quanta=_whole(quanta, "quanta", 0))
    classical_spread = (0.0 if not classical or temperature_k is None else
                        classical_spread_amu12_ang(frequency_cm1,
                                                   temperature_k))
    if kind == CLASSICAL:
        return State(CLASSICAL, q_zp,
                     classical_spread_amu12_ang=classical_spread)
    n_s = (0.0 if swing_quanta is None
           else _checked_quanta(swing_quanta, "swing_quanta"))
    if kind == CLASSICAL_SWING:
        return State(CLASSICAL_SWING, q_zp, swing_quanta=n_s,
                     classical_spread_amu12_ang=classical_spread)
    if kind == SWING:
        bath = (0.0 if temperature_k is None else
                _checked_quanta(_occupation(frequency_cm1, temperature_k)))
        return State(SWING, q_zp, bath_quanta=bath, swing_quanta=n_s)
    bath = _checked_quanta(_occupation(frequency_cm1, temperature_k)
                           if temperature_k is not None else mean_quanta)
    return State(THERMAL, q_zp, bath_quanta=bath)


def _checked_quanta(n: Any, what: str = "a mean number of quanta") -> float:
    """``n`` as a number of quanta, 0 or more, refused by name otherwise."""
    v = _finite_non_negative(n)
    if v is None:
        raise ValueError(f"{what} is 0 or more; got {n!r}")
    return v


def temperature_of_quanta(frequency_cm1: float, mean_quanta: float) -> float:
    """The temperature whose thermal state holds ``mean_quanta`` quanta of a
    mode at ``frequency_cm1`` -- a mode pumped at random moments, its
    effective temperature (science § 2a.1, § 2a.3):
    ``T = h c nu / (k_B ln(1 + 1/n))``, 0 K at no quanta."""
    nu = float(frequency_cm1)
    if not (math.isfinite(nu) and nu > 0.0):
        raise ValueError(f"a temperature of quanta needs a positive "
                         f"frequency, in cm-1; got {frequency_cm1!r}")
    n = _checked_quanta(mean_quanta, "mean_quanta")
    if n == 0.0:
        return 0.0
    return nu * CM1_KELVIN / math.log1p(1.0 / n)


# --------------------------------------------------------------------- #
#  One mode, as a person computes with it (§ 6.11 § 6)                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModeState:
    """A state of one mode, with the mode's atoms: the state's numbers
    (:class:`State`), each atom's r.m.s. move in it, its draws and frames as
    geometries.  ``temperature_k`` the temperature it was taken at, and
    ``temperature_from`` where it came from -- :data:`GIVEN`,
    :data:`THE_RECORDS` or, set apart in a motion, :data:`THE_MOTIONS`
    (`engines/vibration-frames.md` § 2) -- ``None`` for a state of no
    temperature."""
    state: State
    mode: "Mode" = field(repr=False)
    temperature_k: Optional[float] = None
    temperature_from: Optional[str] = None

    def __getattr__(self, name: str):
        # THE STATE'S OWN FIELDS handed on (`engines/vibration-frames.md`
        # § 1) -- its quanta, swing quanta, warmth, zero-point amplitude and
        # classical spread; nothing else.
        if name in State.__dataclass_fields__:
            return getattr(self.state, name)
        raise AttributeError(f"{type(self).__name__!r} has no attribute "
                             f"{name!r}")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(State.__dataclass_fields__))

    def pdf(self, q):
        return self.state.pdf(q)

    def cdf(self, q):
        return self.state.cdf(q)

    def moment(self, k: int) -> float:
        return self.state.moment(k)

    @property
    def kind(self) -> str:
        return self.state.kind

    @property
    def spread_amu12_ang(self) -> float:
        return self.state.spread_amu12_ang

    @property
    def mean_quanta(self) -> Optional[float]:
        return self.state.mean_quanta

    @property
    def swing_amplitude_amu12_ang(self) -> float:
        return self.state.swing_amplitude_amu12_ang

    @property
    def carried_spread_amu12_ang(self) -> Optional[float]:
        return self.state.carried_spread_amu12_ang

    @property
    def atom_rms_ang(self) -> np.ndarray:
        """Each atom's r.m.s. move in the state, Å: the spread times its
        row's length, ``sqrt(<Q^2>) |L_A|`` -- 0 for a held atom (§ 2b.2)."""
        return self.spread_amu12_ang * _row_lengths(
            self.mode.eigenvector_canonical_over_atoms_per_amu12)

    def draw(self, size: int, *, seed: int, start: int = 0,
             phase=None) -> np.ndarray:
        """The state's draws (:meth:`State.draw`), from this mode's own
        streams of the seed -- keyed by its number."""
        return self.state.draw(size, seed=seed, start=start, phase=phase,
                               mode=self.mode.index_1based)

    def weights_at(self, q_amu12_ang, *, written=None) -> Weights:
        """The weights for this state of frames (:meth:`State.weights_at`)."""
        return self.state.weights_at(q_amu12_ang, written=written)

    def frames(self, nodes: int = 3) -> Frames:
        """The state's frames (:meth:`State.frames`) with the geometries a
        calculation reads: each frame every atom at ``R + Q_j L``, frame 0
        the relaxed geometry exactly."""
        f = self.state.frames(nodes)
        positions = np.stack([self.mode.displaced(float(q))
                              for q in f.q_amu12_ang[:, 0]])
        return replace(f, state=self, positions_ang=positions)


@dataclass(frozen=True)
class Mode:
    """One normal mode of a vibration -- every number of § 6.11 § 6 derived
    by this module from what the result states, each field the record's, by
    the data file's name, its unit in it: a value the record stores is
    worked out here by the function the record was written with, and the
    reader has checked the two agree (§ 6.6).  ``index_1based`` its number,
    as the result counts it; ``eigenvector_canonical_per_amu12`` the
    record's canonical eigenvector, a row per free atom (§ 6.3);
    ``vibration`` the result it belongs to (left out of its repr and
    comparison), which holds what every mode shares -- the free atoms, the
    masses, the elements, the geometry, the record's temperature, the
    labels and regions, the modes' temperatures -- each reached here by its
    name; ``result_frequency_cm1`` the result's own frequency where the
    mode's is scaled or replaced (``None`` otherwise).  The pattern over
    every atom is :attr:`eigenvector_canonical_over_atoms_per_amu12`."""
    index_1based: int
    frequency_cm1: float
    has_imag: bool
    eigenvector_canonical_per_amu12: np.ndarray = field(repr=False,
                                                        compare=False)
    vibration: Optional["Vibration"] = field(default=None, repr=False,
                                             compare=False)
    result_frequency_cm1: Optional[float] = field(default=None, repr=False)

    # ---- what every mode shares, its vibration's -------------------------
    @property
    def _v(self) -> "Vibration":
        if self.vibration is None:
            raise ValueError(f"mode {self.index_1based} belongs to no "
                             f"vibration: a mode is built through a "
                             f"Vibration, which holds its atoms")
        return self.vibration

    @property
    def free_atom_idxs(self) -> Tuple[int, ...]:
        return self._v.free_atom_idxs

    @property
    def masses_amu(self) -> np.ndarray:
        return self._v.masses_amu

    @property
    def element_symbols(self) -> Tuple[str, ...]:
        return self._v.element_symbols

    @property
    def positions_ang(self) -> np.ndarray:
        return self._v.positions_ang

    @property
    def record_temperature_k(self) -> Optional[float]:
        return self._v.record_temperature_k

    @property
    def species_labels(self) -> Tuple[str, ...]:
        return self._v.species_labels

    @property
    def regions(self) -> Mapping[str, Tuple[int, ...]]:
        return self._v.regions

    @property
    def mode_temperatures_k(self) -> Tuple[float, ...]:
        return self._v.mode_temperatures_k

    # ---- the mode's own numbers ------------------------------------------
    @property
    def eigenvector_canonical_over_atoms_per_amu12(self) -> np.ndarray:
        """The canonical eigenvector over EVERY atom in the user's order, a
        held atom's row zero -- what each atom's move is ``Q`` times
        (:func:`mode_over_atoms`)."""
        return mode_over_atoms(self.eigenvector_canonical_per_amu12,
                               self.free_atom_idxs, len(self.masses_amu))

    @property
    def zero_point_amplitude_amu12_ang(self) -> Optional[float]:
        """``Q_zp`` at the mode's frequency; ``None`` for an imaginary mode."""
        return zero_point_amplitude_amu12_ang(self.frequency_cm1, self.has_imag)

    @property
    def zero_point_displacement_ang(self) -> Optional[np.ndarray]:
        """Each FREE atom's move at ``Q_zp``, Å, a row per atom of
        :attr:`free_atom_idxs` -- the record's own field and shape (§ 6.3,
        :func:`zero_point_displacement_ang`); ``None`` for an imaginary
        mode.  Every atom's is ``displacement(zero_point_widths=1)``."""
        return zero_point_displacement_ang(
            self.frequency_cm1, self.has_imag,
            self.eigenvector_canonical_per_amu12)

    @property
    def largest_zero_point_move_ang(self) -> Optional[float]:
        """The atom that moves most at ``Q_zp``, its move's length in Å --
        the record's `largest_zero_point_move_ang`
        (:func:`largest_zero_point_move_ang`)."""
        return largest_zero_point_move_ang(
            self.frequency_cm1, self.has_imag,
            self.eigenvector_canonical_per_amu12)

    @property
    def motion_share(self) -> np.ndarray:
        """Each FREE atom's share of the mode's motion, in the order of
        :attr:`free_atom_idxs` -- the record's `motion_share` (§ 6.11 § 3)."""
        return motion_shares_of(self.eigenvector_canonical_per_amu12,
                                self.masses_amu[list(self.free_atom_idxs)])

    def motion_shares(self, by: str = "element") -> Dict[str, float]:
        """The free atoms' shares summed by group (§ 6.11 § 6, :func:`shares_by`),
        largest first, each group holding a free atom: ``by="element"`` --
        each atom's element, its species label's name; ``"label"`` -- its
        species label as written; ``"region"`` -- each of the structure's
        region labels as written (they may overlap, so the sums may add to
        more than 1; ``{}`` where the structure has none).  A held atom adds
        nothing; a mode with no motion gives ``{}``."""
        free = list(self.free_atom_idxs)
        shares = self.motion_share
        if by == "element":
            return shares_by([self.element_symbols[i] for i in free], shares)
        if by == "label":
            return shares_by([self.species_labels[i] for i in free], shares)
        if by == "region":
            return shares_by([tuple(label for label, atoms
                                    in self.regions.items() if i in atoms)
                              for i in free], shares)
        raise ValueError(f"motion shares are summed by element, label or "
                         f"region; got {by!r}")

    @property
    def reduced_mass_amu(self) -> Optional[float]:
        """The reduced mass by Gaussian's convention (:func:`reduced_mass_amu`);
        ``None`` for a mode with no motion."""
        return reduced_mass_amu(self.eigenvector_canonical_per_amu12)

    @property
    def force_constant_mdyn_per_ang(self) -> Optional[float]:
        """The force constant by Gaussian's convention
        (:func:`force_constant_mdyn_per_ang`); ``None`` for a mode with no
        motion."""
        return force_constant_mdyn_per_ang(self.frequency_cm1,
                                           self.reduced_mass_amu)

    @property
    def thermal(self) -> Dict[str, Any]:
        """The mode's spread and mean quanta at the modes' own temperatures
        (§ 6.11 § 3), ``{temperature_k, thermal_spread_amu12_ang,
        mean_quanta}`` -- each a list over :attr:`mode_temperatures_k`,
        ``None`` for an imaginary mode, which holds no thermal state."""
        temps = [float(t) for t in self.mode_temperatures_k]
        spreads, quanta = thermal_row(self.frequency_cm1, self.has_imag,
                                      temps)
        return dict(zip(THERMAL_ROW_KEYS, (temps, spreads, quanta)))

    def displacement(self, q: Optional[float] = None, *,
                     zero_point_widths: Optional[float] = None) -> np.ndarray:
        """Each atom's move (Å, x y z), ``u_A = Q L_A``, at the coordinate
        ``q`` (amu^1/2.Å) or at ``zero_point_widths`` times ``Q_zp`` -- one
        of the two (§ 2b.2)."""
        return atom_moves(self.eigenvector_canonical_over_atoms_per_amu12,
                          self._coordinate(q, zero_point_widths))

    def displaced(self, q: Optional[float] = None, *,
                  zero_point_widths: Optional[float] = None) -> np.ndarray:
        """The geometry at the coordinate: every atom at ``R + Q L``."""
        return displaced(self.positions_ang,
                         self.eigenvector_canonical_over_atoms_per_amu12,
                         self._coordinate(q, zero_point_widths))

    def coordinate(self, moves_ang) -> float:
        """The coordinate a set of atom moves gives along this mode,
        ``Q = sum_A m_A L_A . u_A`` (§ 2b.2)."""
        return coordinate_of(self.eigenvector_canonical_over_atoms_per_amu12,
                             self.masses_amu, moves_ang)

    def state(self, temperature_k: Optional[float] = None, *,
              mean_quanta: Optional[float] = None,
              quanta: Optional[int] = None,
              swing_quanta: Optional[float] = None,
              classical: bool = False) -> ModeState:
        """The mode in a state (§ 2b.3, :func:`state`).  Arguments that name
        a state once a temperature is added (:data:`_STATE_NAMED_BY`) -- a
        thermal or classical state given neither a temperature nor mean
        quanta -- take the temperature the record states, and say so (§ 6.11
        § 9 item 6); a record stating none is asked for one.  n quanta, mean
        quanta, a driven mode (the coherent state) and a classical swing
        alone take none."""
        q_zp = self.zero_point_amplitude_amu12_ang
        if q_zp is None:
            raise ValueError(f"mode {self.index_1based} is imaginary: it has "
                             f"no harmonic state")
        named = _named(temperature_k, mean_quanta, quanta, swing_quanta,
                       classical)
        temperature_k, source = _temperature_of(
            named, temperature_k, self.record_temperature_k, THE_RECORDS)
        return ModeState(
            state(q_zp, self.frequency_cm1, temperature_k=temperature_k,
                  mean_quanta=mean_quanta, quanta=quanta,
                  swing_quanta=swing_quanta, classical=classical),
            self, temperature_k=temperature_k, temperature_from=source)

    def with_frequency(self, frequency_cm1: float) -> "Mode":
        """The mode at another frequency -- a scaled or a measured one -- its
        pattern kept (§ 2b.1): ``Q_zp``, its displacement and every spread
        follow from the new frequency; the result's own frequency kept as
        :attr:`result_frequency_cm1`."""
        nu = float(frequency_cm1)
        if not (math.isfinite(nu) and nu > 0.0):
            raise ValueError(f"a frequency is a positive number of cm-1; got "
                             f"{frequency_cm1!r}")
        return replace(self, frequency_cm1=nu, has_imag=False,
                       result_frequency_cm1=self._result_frequency)

    @property
    def _result_frequency(self) -> float:
        return (self.frequency_cm1 if self.result_frequency_cm1 is None
                else self.result_frequency_cm1)

    def _coordinate(self, q: Optional[float],
                    zero_point_widths: Optional[float]) -> float:
        if (q is None) == (zero_point_widths is None):
            raise ValueError("give the coordinate q (amu^1/2.Å) or "
                             "zero_point_widths, one of the two")
        if q is not None:
            return float(q)
        q_zp = self.zero_point_amplitude_amu12_ang
        if q_zp is None:
            raise ValueError(f"mode {self.index_1based} is imaginary: it has "
                             f"no zero-point width")
        return float(zero_point_widths) * q_zp


@dataclass(frozen=True, eq=False)
class Vibration:
    """A vibration result's modes, as a person computes with them (§ 6.11
    § 6, `engines/vibration-frames.md` § 1): ``modes`` in the result's order
    (``modes[6]`` is mode 7; :meth:`mode` by number), each bound to this
    vibration; what every mode shares -- ``free_atom_idxs``, every atom's
    ``masses_amu`` and ``element_symbols`` (its species label's name),
    ``positions_ang`` the geometry the modes were taken at, the record's
    temperature (its thermochemistry's, which a state asked without one
    takes), ``species_labels`` and ``regions`` as written, the modes'
    temperatures the record states its spreads at; ``frequency_scale`` the
    factor every mode's frequency was scaled by (1 for the result's own);
    and where it came from -- ``read_path`` the file read, as given;
    ``structure`` the structure the result states (its reader fills it);
    ``result_sha256`` the result JSON's, ``run`` and
    ``relaxation_calculation`` -- filled from R3d2b2b2, at the read door and
    where a run's folder is read (`vibration_frames.vibration_of_run`);
    ``None`` until then."""
    modes: Tuple[Mode, ...]
    free_atom_idxs: Tuple[int, ...] = field(repr=False)
    masses_amu: np.ndarray = field(repr=False)
    element_symbols: Tuple[str, ...] = field(repr=False)
    positions_ang: np.ndarray = field(repr=False)
    record_temperature_k: Optional[float] = None
    species_labels: Tuple[str, ...] = field(default=(), repr=False)
    regions: Mapping[str, Tuple[int, ...]] = field(default_factory=dict,
                                                   repr=False)
    mode_temperatures_k: Tuple[float, ...] = field(default=(), repr=False)
    frequency_scale: float = 1.0
    read_path: str = ""
    run: Optional[Any] = field(default=None, repr=False)
    result_sha256: Optional[str] = field(default=None, repr=False)
    structure: Optional[Any] = field(default=None, repr=False)
    relaxation_calculation: Optional[Mapping[str, Any]] = field(
        default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "masses_amu",
                           np.asarray(self.masses_amu, dtype=float))
        object.__setattr__(self, "positions_ang",
                           np.asarray(self.positions_ang, dtype=float))
        object.__setattr__(self, "modes", tuple(
            replace(m, vibration=self) for m in self.modes))

    def mode(self, number: int) -> Mode:
        """The mode numbered ``number``, as the result counts it
        (``index_1based``) -- refused by name where the result has none."""
        for m in self.modes:
            if m.index_1based == number:
                return m
        raise ValueError(f"the result has no mode {number!r}: its modes are "
                         f"numbered {self.modes[0].index_1based} to "
                         f"{self.modes[-1].index_1based}" if self.modes else
                         "the result has no modes")

    def _rows(self, numbers: Sequence[int]) -> np.ndarray:
        """The modes numbered ``numbers`` over every atom (modes x atoms x
        3)."""
        if not len(numbers):
            return np.zeros((0, len(self.masses_amu), 3))
        return np.stack([self.mode(int(k)).eigenvector_canonical_over_atoms_per_amu12
                         for k in numbers])

    def _moves(self, numbers: Sequence[int], q_amu12_ang) -> np.ndarray:
        """Every atom's move for each row of coordinates ``q_amu12_ang``
        (frames x modes) along the modes numbered ``numbers``,
        ``sum_k Q_k L_k`` (frames x atoms x 3) -- the one rule every
        geometry of several modes is built by (§ 3.2)."""
        q = np.atleast_2d(np.asarray(q_amu12_ang, dtype=float))
        return np.einsum("fk,kai->fai", q, self._rows(numbers))

    def displaced(self, coordinates: Mapping[int, float]) -> np.ndarray:
        """The geometry for coordinates a person gives
        (`engines/vibration-frames.md` § 3.2): every atom at
        ``R^0 + sum_k Q_k L_k``, the modes by number, ``Q`` in amu^1/2.Å --
        the relaxed geometry itself for none."""
        numbers = [int(k) for k in dict(coordinates)]
        if not numbers:
            return np.array(self.positions_ang, dtype=float, copy=True)
        q = [[float(dict(coordinates)[k]) for k in numbers]]
        return self.positions_ang + self._moves(numbers, q)[0]

    @property
    def _content(self) -> Tuple[Any, ...]:
        """What makes two vibrations one: each mode's number, frequency,
        whether imaginary and pattern; the masses, the geometry, the free
        atoms and the frequency scale -- however many times it was read."""
        return (tuple(self.free_atom_idxs), self.masses_amu.tobytes(),
                self.positions_ang.tobytes(), float(self.frequency_scale),
                tuple((m.index_1based, float(m.frequency_cm1), m.has_imag,
                       np.asarray(m.eigenvector_canonical_per_amu12,
                                  dtype=float).tobytes())
                      for m in self.modes))

    def with_frequency_scale(self, factor: float) -> "Vibration":
        """Every mode at ``factor`` times its frequency, each pattern kept
        (science § 2b.1) -- applied to the frequencies it holds, so two scales
        compound; the result's own frequencies kept on each mode
        (:attr:`Mode.result_frequency_cm1`) and the factor from the result's
        stated (``frequency_scale``)."""
        f = float(factor)
        if not (math.isfinite(f) and f > 0.0):
            raise ValueError(f"a frequency scale is a positive number; got "
                             f"{factor!r}")
        total = f * self.frequency_scale
        modes = tuple(replace(m, frequency_cm1=m._result_frequency * total,
                              result_frequency_cm1=m._result_frequency)
                      for m in self.modes)
        return replace(self, modes=modes, frequency_scale=total)

    def motion(self, temperature_k: Optional[float] = None, *,
               modes: Optional[Sequence[int]] = None,
               classical: bool = False) -> "Motion":
        """The molecule's motion (`engines/vibration-frames.md` § 3.1): every
        real mode -- or the numbers in ``modes`` -- thermal at
        ``temperature_k`` (the record's when none, said), quantum, or the
        classical bell curves with ``classical``; its modes in the result's
        order.  An imaginary mode among them is left out, said; none left to
        move is refused by name."""
        asked = ({m.index_1based for m in self.modes} if modes is None
                 else {int(k) for k in modes})
        for k in asked:
            self.mode(k)
        notes: List[str] = []
        imaginary = [m.index_1based for m in self.modes
                     if m.index_1based in asked and m.has_imag]
        moving = [m.index_1based for m in self.modes
                  if m.index_1based in asked and not m.has_imag]
        if not moving:
            raise ValueError("a motion needs a real mode to move: "
                             + ("none were named" if not asked else
                                "every mode named is imaginary"))
        temperature_k, source = _temperature_of(
            frozenset(), temperature_k, self.record_temperature_k, THE_RECORDS)
        if source == THE_RECORDS:
            notes.append(f"the temperature is the record's, "
                         f"{temperature_k:g} K")
        states = {k: replace(self.mode(k).state(temperature_k,
                                                classical=classical),
                             temperature_from=source)
                  for k in moving}
        return Motion(self, states, (), float(temperature_k), source,
                      bool(classical), tuple(notes), (), tuple(imaginary))


def _numbers_words(numbers: Sequence[Any]) -> str:
    """``9``, ``9 and 11``, ``9, 11 and 13``."""
    words = [f"{n:g}" if isinstance(n, float) else str(n) for n in numbers]
    return (words[0] if len(words) == 1 else
            ", ".join(words[:-1]) + " and " + words[-1])


def _quanta_words(n: float) -> str:
    return f"{n:g} swing quant{'um' if n == 1.0 else 'a'}"


@dataclass(frozen=True)
class Drive:
    """One laser's drive group (`engines/vibration-frames.md` § 3.1;
    science § 2c.2): its modes, by number, sharing one phase per snapshot,
    each at its own lag behind the field, ``lags_rad``, the person's;
    ``key`` the modes it was made with -- its phase stream's,
    ``[seed, 0, *key]``, unique among the groups a motion holds (a new
    group's modes are in none) and kept when a member is set apart again,
    so the others draw as they did."""
    modes: Tuple[int, ...]
    lags_rad: Mapping[int, float]
    key: Tuple[int, ...]


@dataclass(frozen=True, eq=False)
class Motion:
    """The molecule's motion (`engines/vibration-frames.md` § 3.1): a state
    for each mode, by number, in the result's order (``states``), and the
    drive groups (``drives``); ``temperature_k`` the thermal background's,
    ``temperature_from`` whether it was ``"given"`` or ``"the record's"``;
    ``classical`` whether the background is classical; ``set_apart`` the
    modes given a state of their own, said as :attr:`set_apart_modes`;
    ``left_out`` the modes left out as imaginary when it was made;
    ``said`` what the motion says -- information, never a decision."""
    vibration: Vibration = field(repr=False)
    states: Mapping[int, ModeState]
    drives: Tuple[Drive, ...]
    temperature_k: float
    temperature_from: str
    classical: bool
    notes: Tuple[str, ...] = field(default=(), repr=False)
    set_apart: Tuple[int, ...] = ()
    left_out: Tuple[int, ...] = field(default=(), repr=False)

    def _group_of(self, k: int) -> Optional[Drive]:
        for d in self.drives:
            if k in d.modes:
                return d
        return None

    @property
    def set_apart_modes(self) -> str:
        """Each mode given a state of its own, in words
        (`engines/vibration-frames.md` § 5.1's ``set_apart_modes``): a mode
        alone, its state -- a driven one with its own phase -- and its
        frequency where it is not the result's; each drive group, its modes,
        swing quanta and lags; ``""`` when none."""
        parts: List[str] = []
        done = set()
        for k in self.set_apart:
            if k in done:
                continue
            group = self._group_of(k)
            if group is None:
                parts.append(_state_words(k, self.states[k]))
                done.add(k)
                continue
            states = [self.states[m] for m in group.modes]
            lags = _numbers_words([group.lags_rad[m] for m in group.modes])
            if len(states) == 1:
                head = (f"mode {group.modes[0]}: one drive, "
                        f"{_quanta_words(states[0].state.swing_quanta)}, "
                        f"lag {lags} rad")
            else:
                quanta = _numbers_words([st.state.swing_quanta
                                         for st in states])
                head = (f"modes {_numbers_words(list(group.modes))}: one "
                        f"drive, {quanta} swing quanta, lags {lags} rad")
            parts.append(f"{head}, {_swing_words(states[0])}" + "".join(
                _frequency_words(st.mode) for st in states))
            done.update(group.modes)
        return "; ".join(parts)

    @property
    def said(self) -> Tuple[str, ...]:
        """Summary sentences: the modes left out as imaginary and still out of
        the motion, the temperature taken from the record, a mode that left
        its drive group, a mode set apart again, a cold driven mode in a warm
        motion with its thermal quanta and its two baselines (science § 6), a
        bare classical swing in a warm motion."""
        out = []
        out_still = [k for k in self.left_out if k not in self.states]
        if out_still:
            out.append(f"mode{'s' if len(out_still) > 1 else ''} "
                       f"{_numbers_words(out_still)} left out: imaginary, "
                       f"no harmonic state")
        out += list(self.notes)
        t = self.temperature_k
        if t > 0.0:
            for k, s in self.states.items():
                grouped = self._group_of(k) is not None
                if s.kind == SWING and s.state.bath_quanta == 0.0:
                    n_th = float(mean_quanta(s.mode.frequency_cm1, t))
                    warm = (f"the same drive made with temperature_k={t:g} "
                            f"(with_drive on the motion before it)" if grouped
                            else f"with_mode({k}, swing_quanta="
                                 f"{s.state.swing_quanta:g}, temperature_k={t:g})")
                    rest = (f"with_mode({k}, temperature_k=0.0"
                            + (", classical=False" if self.classical else "")
                            + ")")
                    out.append(
                        f"mode {k} is driven from its ground state -- the "
                        f"coherent state, the zero-point width; at {t:g} K "
                        f"its thermal quanta would be {n_th:.3g} (sigma_T/Q_zp "
                        f"{spread_of_quanta(1.0, n_th):.3f}): its exact "
                        f"baseline is {rest} beside the "
                        f"{'classical' if self.classical else 'warm'} others; "
                        f"the warm driven mode is {warm}")
                elif (s.kind == CLASSICAL_SWING
                      and s.state.classical_spread_amu12_ang == 0.0):
                    carried = (f"the same drive made with temperature_k={t:g}, "
                               f"classical=True (with_drive on the motion "
                               f"before it)" if grouped else
                               f"with_mode({k}, swing_quanta="
                               f"{s.state.swing_quanta:g}, temperature_k={t:g}, "
                               f"classical=True)")
                    out.append(
                        f"mode {k} is a bare classical swing in a motion at "
                        f"{t:g} K: the swing carrying the classical bell curve "
                        f"is {carried}")
        return tuple(out)

    def __repr__(self) -> str:
        head = (f"Motion({len(self.states)} "
                f"mode{'' if len(self.states) == 1 else 's'}, "
                f"{'classical' if self.classical else 'quantum'} at "
                f"{self.temperature_k:g} K ({self.temperature_from})"
                + (f"; set apart: {self.set_apart_modes}"
                   if self.set_apart else "") + ")")
        return "\n".join([head] + [f"  {s}" for s in self.said])

    def with_mode(self, number: int, *,
                  temperature_k: Optional[float] = None,
                  mean_quanta: Optional[float] = None,
                  quanta: Optional[int] = None,
                  swing_quanta: Optional[float] = None,
                  classical: Optional[bool] = None,
                  state: Optional[ModeState] = None) -> "Motion":
        """A new motion with mode ``number`` set apart -- in any state of § 2
        by its arguments, or a :class:`ModeState` of that mode, never both --
        replacing its state; a mode outside the motion added.  A thermal or
        classical state given no temperature takes the motion's, said; in a
        classical motion a thermal state is classical unless
        ``classical=False``, a driven one quantum unless ``classical=True``.
        Set apart again, the later call's state, said; a member of a drive
        group leaves it, the group drawing as before."""
        k = int(number)
        mode = self.vibration.mode(k)
        asked = _named(temperature_k, mean_quanta, quanta, swing_quanta,
                       classical)
        if state is not None:
            if not isinstance(state, ModeState):
                raise ValueError(
                    f"with_mode({k}, state=...) takes a state of the mode, as "
                    f"v.mode({k}).state(...) gives it; got a "
                    f"{type(state).__name__}")
            if asked:
                raise ValueError(
                    f"with_mode({k}, ...) takes a state, or a state's "
                    f"arguments -- not both; got state= with "
                    f"{', '.join(sorted(asked))}")
            if (state.mode.index_1based != k or state.mode.vibration is None
                    or state.mode.vibration._content
                    != self.vibration._content):
                raise ValueError(
                    f"with_mode({k}, state=...) takes a state of mode {k} of "
                    f"this vibration; got mode {state.mode.index_1based}'s"
                    + ("" if state.mode.vibration is not None
                       and state.mode.vibration._content
                       == self.vibration._content else
                       " of another vibration"))
            return self._replaced(k, state)
        if classical is None:
            kind = _STATE_NAMED_BY.get(asked | {"classical", "temperature_k"})
            classical = self.classical and kind == CLASSICAL
        named = _named(temperature_k, mean_quanta, quanta, swing_quanta,
                       classical)
        t, source = _temperature_of(named, temperature_k, self.temperature_k,
                                    THE_MOTIONS)
        mode_state = mode.state(t, mean_quanta=mean_quanta, quanta=quanta,
                                swing_quanta=swing_quanta,
                                classical=bool(classical))
        return self._replaced(k, replace(mode_state, temperature_from=source))

    def with_drive(self, swing_quanta: Mapping[int, float], *,
                   lags_rad: Mapping[int, float],
                   temperature_k: Optional[float] = None,
                   classical: bool = False) -> "Motion":
        """A new motion with one laser's drive group
        (`engines/vibration-frames.md` § 3.1; science § 2c.2): each mode
        driven with its swing quanta, one phase per snapshot shared by the
        group, each mode at its lag behind the field -- every lag the
        person's, stated for every mode, never defaulted.  The modes' warmth
        with ``temperature_k``, their swings classical with ``classical``.
        A mode in another group is refused by name; a mode set apart before
        takes this call's state, said."""
        quanta = {int(k): v for k, v in dict(swing_quanta).items()}
        lags = {int(k): float(v) for k, v in dict(lags_rad).items()}
        if not quanta:
            raise ValueError("a drive group drives one mode at least")
        if set(lags) != set(quanta):
            missing = sorted(set(quanta) - set(lags))
            extra = sorted(set(lags) - set(quanta))
            raise ValueError(
                "every driven mode's lag is the person's, stated for each: "
                + (f"no lag for mode{'s' if len(missing) > 1 else ''} "
                   f"{_numbers_words(missing)}" if missing else "")
                + ("; " if missing and extra else "")
                + (f"a lag for mode{'s' if len(extra) > 1 else ''} "
                   f"{_numbers_words(extra)} not driven" if extra else ""))
        for k in quanta:
            group = self._group_of(k)
            if group is not None:
                raise ValueError(
                    f"mode {k} is in a drive group already (modes "
                    f"{_numbers_words(list(group.modes))}): a mode belongs to "
                    f"one group at most")
        motion = self
        for k, n_s in sorted(quanta.items()):
            motion = motion._replaced(k, self.vibration.mode(k).state(
                temperature_k, swing_quanta=n_s, classical=classical))
        group = Drive(tuple(sorted(quanta)), dict(sorted(lags.items())),
                      tuple(sorted(quanta)))
        return replace(motion, drives=motion.drives + (group,))

    def _replaced(self, k: int, mode_state: ModeState) -> "Motion":
        """This motion with mode ``k``'s state replaced, its modes in the
        result's order: out of any drive group it was in -- the group keeping
        its phase stream -- and set apart again, said."""
        notes = list(self.notes)
        group = self._group_of(k)
        if group is not None:
            notes.append(f"mode {k} left its drive" if len(group.modes) == 1
                         else f"mode {k} left the drive of modes "
                              f"{_numbers_words(list(group.modes))}: the "
                              f"others keep its phases")
        elif k in self.set_apart:
            notes.append(f"mode {k} set apart again: the later call's state "
                         f"replaces the earlier")
        drives = []
        for d in self.drives:
            kept = tuple(m for m in d.modes if m != k)
            if kept:
                drives.append(d if kept == d.modes else Drive(
                    kept, {m: d.lags_rad[m] for m in kept}, d.key))
        states = dict(self.states)
        states[k] = mode_state
        states = {m.index_1based: states[m.index_1based]
                  for m in self.vibration.modes if m.index_1based in states}
        set_apart = self.set_apart + ((k,) if k not in self.set_apart else ())
        return replace(self, states=states, drives=tuple(drives),
                       notes=tuple(notes), set_apart=set_apart)

    def sample(self, n: int, *, seed: int, antithetic: bool,
               start: int = 0) -> "Snapshots":
        """``n`` snapshots of the motion (`engines/vibration-frames.md` § 3.3;
        science § 2c.3): each mode's coordinate drawn from its own state and
        its own streams of ``seed`` -- a drive group's modes with one shared
        phase from ``[seed, 0, *key]``, each less its lag -- snapshots
        ``start`` ... ``start + n - 1`` of the seed's sequence.  With
        ``antithetic`` -- the person's choice, no default -- each snapshot is
        followed by its mirror, every ``Q`` negated; ``n`` and ``start`` even
        then, an odd one refused by name."""
        n = _whole(n, "the number of snapshots", 1)
        start = _whole(start, "start", 0)
        seed = _checked_seed(seed)
        if not isinstance(antithetic, (bool, np.bool_)):
            raise ValueError(f"antithetic is True or False, the person's "
                             f"choice; got {antithetic!r}")
        antithetic = bool(antithetic)
        if antithetic and (n % 2 or start % 2):
            raise ValueError(
                f"mirrored pairs need an even number of snapshots and an even "
                f"start: n = {n}, start = {start}")
        base_n, base_start = ((n // 2, start // 2) if antithetic
                              else (n, start))
        cols = list(self.states)
        shared = {}
        for d in self.drives:
            phase = _phases_of(_stream((seed, 0) + tuple(d.key), base_n,
                                       base_start, uniform=True))
            for k in d.modes:
                shared[k] = phase - d.lags_rad[k]
        base = np.empty((base_n, len(cols)))
        for c, k in enumerate(cols):
            base[:, c] = self.states[k].draw(base_n, seed=seed,
                                             start=base_start,
                                             phase=shared.get(k))
        if antithetic:
            drawn = np.empty((n, len(cols)))
            drawn[0::2] = base
            drawn[1::2] = -base
        else:
            drawn = base
        q = np.vstack([np.zeros((1, len(cols))), drawn])
        return Snapshots(self, np.array(cols, dtype=int), q, antithetic,
                         ((seed, start, n),))


@dataclass(frozen=True, eq=False)
class Estimate:
    """A snapshot set's average and its error (`engines/vibration-frames.md`
    § 7; science § 5.5): ``value`` the average over the snapshots,
    ``error`` its standard error, ``n`` the snapshots counted; ``said``
    what was dropped, or why there is no error."""
    value: Any
    error: Any
    n: int
    said: Tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Parts:
    """A paired set's spread split (science § 5.5): ``even`` the spread of
    the pair means -- the even part's; ``odd`` the spread of the pair
    half-differences -- the odd part's; ``said`` what was dropped."""
    even: Any
    odd: Any
    said: Tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Snapshots:
    """Snapshots of a motion, standing for the set as written
    (`engines/vibration-frames.md` § 3.3): item ``i`` is frame ``i``, frame
    0 the relaxed geometry at weight 0, the snapshot drawn ``j``-th (from 0)
    frame ``j + 1``; with pairs, frames ``2j + 1`` and ``2j + 2`` the
    ``j``-th pair.  ``mode_index_1based`` the columns' modes, in the result's
    order; ``q_amu12_ang`` each frame's coordinate along each (frames x
    modes); ``antithetic`` whether pairs; ``drawings`` each drawing's
    ``(seed, start, samples)`` -- one for a set drawn at once, one per set
    for a pooled one; ``pieces`` the snapshots of each set a pooled set was
    given, in order (a pooled set among them one piece)."""
    motion: Motion = field(repr=False)
    mode_index_1based: np.ndarray = field(repr=False)
    q_amu12_ang: np.ndarray = field(repr=False)
    antithetic: bool
    drawings: Tuple[Tuple[int, int, int], ...]
    pieces: Tuple[int, ...] = ()

    @property
    def seed(self) -> Optional[int]:
        """The seed -- ``None`` for a pooled set."""
        return self.drawings[0][0] if len(self.drawings) == 1 else None

    @property
    def start(self) -> Optional[int]:
        return self.drawings[0][1] if len(self.drawings) == 1 else None

    @property
    def samples(self) -> int:
        """The snapshots, the frames after frame 0."""
        return int(self.q_amu12_ang.shape[0] - 1)

    @property
    def pooled(self) -> bool:
        return len(self.drawings) > 1

    @property
    def weights(self) -> np.ndarray:
        """0 for frame 0, then ``1/N`` each."""
        w = np.full(self.q_amu12_ang.shape[0], 1.0 / self.samples)
        w[0] = 0.0
        return w

    @property
    def atom_moves_ang(self) -> np.ndarray:
        """Each frame's every-atom move, ``sum_k Q_k L_k`` (frames x atoms x
        3) -- the displacement map, by the vibration's one rule."""
        return self.motion.vibration._moves(self.mode_index_1based,
                                            self.q_amu12_ang)

    @property
    def positions_ang(self) -> np.ndarray:
        """Each frame's geometry, frame 0 the relaxed one (frames x atoms x
        3)."""
        return self.motion.vibration.positions_ang[None] + self.atom_moves_ang

    @property
    def max_atom_displacement_ang(self) -> np.ndarray:
        """Each frame's largest single-atom move, Å (frames,)."""
        return _largest_moves(self.motion.vibration, self.mode_index_1based,
                              self.q_amu12_ang)[0]

    @property
    def max_atom_mode_index_1based(self) -> np.ndarray:
        """The mode moving each frame's most-moved atom most -- 0 for a frame
        that moves nothing (frame 0) (frames,)."""
        return _largest_moves(self.motion.vibration, self.mode_index_1based,
                              self.q_amu12_ang)[1]

    @property
    def said(self) -> Tuple[str, ...]:
        """Summary sentences (`engines/vibration-frames.md` § 3.3): the
        largest single-atom move over the set, its frame and the mode moving
        it, beside each mode's spread -- read against a bond length; one
        pair alone, its error having nothing to be taken from."""
        largest = self.max_atom_displacement_ang
        i = int(np.argmax(largest))
        spreads = ", ".join(
            f"mode {k} {self.motion.states[int(k)].spread_amu12_ang:.3g}"
            for k in self.mode_index_1based)
        out = [f"the largest single-atom move is {largest[i]:.4f} Å, frame "
               f"{i}, mode {int(self.max_atom_mode_index_1based[i])} moving "
               f"it most; each mode's spread (amu^1/2·Å): {spreads}"]
        if self.antithetic and self.samples == 2:
            out.append("one mirrored pair: its error has nothing to be taken "
                       "from")
        return tuple(out)

    def __repr__(self) -> str:
        modes = len(self.mode_index_1based)
        head = (f"Snapshots({self.samples} snapshot"
                f"{'' if self.samples == 1 else 's'} of {modes} "
                f"mode{'' if modes == 1 else 's'}, "
                + ("mirrored pairs" if self.antithetic else "independent")
                + (f", seed {self.seed}, start {self.start}" if not self.pooled
                   else f", pooled from {len(self.pieces)} sets") + ")")
        return "\n".join([head] + [f"  {s}" for s in self.said])

    # ---- the analyses of a set's results (§ 7) ---------------------------
    def _values(self, values) -> np.ndarray:
        """The snapshots' values, frame 0 left out: one array for the set as
        written -- one value per frame, frame 0 first -- or, for a pooled
        set, one array per set it was given, each its own frame 0 first, in
        order."""
        pieces = self.pieces or (self.samples,)
        if (len(values) == 1
                and np.asarray(values[0]).shape[0] == self.samples + 1):
            return np.asarray(values[0], dtype=float)[1:]
        if len(values) != len(pieces):
            raise ValueError(
                f"one array of values, one value per frame of the set as "
                f"written ({self.samples + 1}, frame 0 first)"
                + (f", or one per set pooled ({len(pieces)}), each its own "
                   f"frame 0 first" if len(pieces) > 1 else "")
                + f"; got {len(values)} array"
                + ("s" if len(values) != 1 else "")
                + (f" of {np.asarray(values[0]).shape[0]} values"
                   if len(values) == 1 else ""))
        out = []
        for samples, v in zip(pieces, values):
            v = np.asarray(v, dtype=float)
            if v.shape[0] != samples + 1:
                raise ValueError(
                    f"{v.shape[0]} values for a set of {samples + 1} frames: "
                    f"one value per frame, frame 0 first")
            out.append(v[1:])
        return np.concatenate(out, axis=0)

    def _units(self, values) -> Tuple[np.ndarray, int, List[str]]:
        """The independent units of the snapshots' values -- each frame, or
        each pair -- a unit with a missing value dropped, said."""
        x = self._values(values)
        said: List[str] = []
        missing = np.isnan(x.reshape(x.shape[0], -1)).any(axis=1)
        if self.antithetic:
            pair_missing = missing.reshape(-1, 2).any(axis=1)
            pairs = x.reshape((-1, 2) + x.shape[1:])
            if pair_missing.any():
                said.append(f"{int(pair_missing.sum())} pair"
                            f"{'s' if pair_missing.sum() > 1 else ''} left out:"
                            f" a value missing")
            return pairs[~pair_missing], int(2 * (~pair_missing).sum()), said
        if missing.any():
            said.append(f"{int(missing.sum())} snapshot"
                        f"{'s' if missing.sum() > 1 else ''} left out: a value"
                        f" missing")
        return x[~missing], int((~missing).sum()), said

    def _none_left(self, count: int, said: List[str]) -> bool:
        unit = "pair" if self.antithetic else "snapshot"
        if count == 0:
            said.append(f"no {unit} left: every one has a value missing")
        elif count == 1:
            said.append(f"one {unit}: its error has nothing to be taken from")
        return count < 2

    def standard_error(self, *values) -> Estimate:
        """The average over the snapshots and its error (science § 5.5) --
        ``values`` one per frame of the set as written, frame 0 first (a
        pooled set's, or one array per set pooled, in order), any trailing
        axes kept: for independent frames the frames' spread over
        ``sqrt(N)``, for pairs the pair means' over ``sqrt(N/2)``; frame 0
        left out."""
        units, n, said = self._units(values)
        if self.antithetic:
            units = units.mean(axis=1)
        count = units.shape[0]
        value = units.mean(axis=0) if count else np.full(units.shape[1:],
                                                         np.nan)
        if self._none_left(count, said):
            error = np.full_like(np.asarray(value, dtype=float), np.nan)
        else:
            error = units.std(axis=0, ddof=1) / math.sqrt(count)
        return Estimate(value, error, n, tuple(said))

    def parts(self, *values) -> Parts:
        """A paired set's spread split into its even and odd parts (science
        § 5.5): the spread of the pair means, the even part's, and of the
        pair half-differences, the odd part's -- whether pairs helped;
        ``values`` as :meth:`standard_error` takes them.  A set drawn without
        pairs is refused by name."""
        if not self.antithetic:
            raise ValueError("parts split a paired set's spread: this set's "
                             "snapshots were drawn alone, not in mirrored "
                             "pairs")
        units, _, said = self._units(values)
        if self._none_left(units.shape[0], said):
            nan = np.full(units.shape[2:], np.nan)
            return Parts(nan, nan, tuple(said))
        means = units.mean(axis=1)
        halves = (units[:, 0] - units[:, 1]) / 2.0
        return Parts(means.std(axis=0, ddof=1), halves.std(axis=0, ddof=1),
                     tuple(said))


def _frequency_words(mode: "Mode") -> str:
    """A mode's frequency where it is not the result's."""
    if mode.result_frequency_cm1 is None:
        return ""
    return (f"; mode {mode.index_1based} at {mode.frequency_cm1:g} cm-1, the "
            f"result's {mode.result_frequency_cm1:g}")


def _temperature_words(s: ModeState) -> str:
    """A state's temperature, and where it came from where not given."""
    return f"{s.temperature_k:g} K" + (
        f" ({s.temperature_from})" if s.temperature_from in (THE_RECORDS,
                                                             THE_MOTIONS)
        else "")


def _swing_words(s: ModeState) -> str:
    """A driven state's warmth, in words: the coherent state's zero-point
    width, a warm driven mode's temperature, a classical swing bare or
    carrying the classical bell curve at its temperature."""
    st = s.state
    if st.kind == SWING:
        return ("the zero-point width" if st.bath_quanta == 0.0 else
                f"warm at {_temperature_words(s)}")
    return ("a bare classical swing" if st.classical_spread_amu12_ang == 0.0
            else f"a classical swing carrying the classical bell curve at "
                 f"{_temperature_words(s)}")


def _state_words(k: int, s: ModeState) -> str:
    """A mode set apart alone, its state in words
    (`engines/vibration-frames.md` § 5.1's ``set_apart_modes``)."""
    st = s.state
    t = "" if s.temperature_k is None else _temperature_words(s)
    if st.kind in (SWING, CLASSICAL_SWING):
        words = (f"mode {k}: driven, {_quanta_words(st.swing_quanta)}, "
                 f"{_swing_words(s)}, its own phase")
    elif st.kind == NUMBER:
        words = f"mode {k}: {st.quanta} quant{'um' if st.quanta == 1 else 'a'} exactly"
    elif st.kind == CLASSICAL:
        words = f"mode {k}: classical at {t}"
    elif s.temperature_k is not None:
        words = f"mode {k}: thermal at {t}"
    else:
        words = f"mode {k}: thermal, {st.bath_quanta:g} mean quanta"
    return words + _frequency_words(s.mode)


def pooled(*sets: Snapshots) -> Snapshots:
    """Snapshot sets of one motion joined into one
    (`engines/vibration-frames.md` § 3.3; science § 5.5): the same states,
    drive groups and pairing -- one seed drawn in sequence, or distinct
    seeds -- in the order given; two motions, pairs with unpaired, or two
    ranges of one seed that overlap, refused by name.  Its analyses take
    one array for the set as written, or each set's values whole, in the
    order of the sets."""
    if not sets:
        raise ValueError("pooling needs a set to pool")
    if len(sets) == 1:
        return sets[0]
    first = sets[0]
    for s in sets[1:]:
        if s.antithetic != first.antithetic:
            raise ValueError("pairs pool only with pairs: one set is mirrored "
                             "pairs, another is not")
        differ = [what for what, a, b in zip(
            ("the vibration", "the modes' states", "the drive groups"),
            _motion_key(s.motion), _motion_key(first.motion)) if a != b]
        if differ:
            raise ValueError(f"pooled sets are of one motion; "
                             f"{' and '.join(differ)} differ")
    drawn = [p for s in sets for p in s.drawings]
    for i, (seed_a, start_a, n_a) in enumerate(drawn):
        for seed_b, start_b, n_b in drawn[i + 1:]:
            if seed_a == seed_b and start_a < start_b + n_b \
                    and start_b < start_a + n_a:
                raise ValueError(
                    f"two sets of seed {seed_a} overlap (snapshots "
                    f"{start_a}-{start_a + n_a - 1} and "
                    f"{start_b}-{start_b + n_b - 1}): pooled sets draw "
                    f"different snapshots")
    q = np.vstack([first.q_amu12_ang[:1]]
                  + [s.q_amu12_ang[1:] for s in sets])
    return Snapshots(first.motion, first.mode_index_1based, q,
                     first.antithetic, tuple(drawn),
                     tuple(s.samples for s in sets))


def _motion_key(motion: Motion) -> Tuple[Any, ...]:
    """What makes two motions one: the vibration's content, each mode's
    state (`State`'s own equality), the drive groups."""
    return (motion.vibration._content,
            tuple((k, s.state) for k, s in motion.states.items()),
            tuple((d.modes, tuple(sorted(d.lags_rad.items())), d.key)
                  for d in motion.drives))


__all__ = ["checked_temperature_k", "zero_point_amplitude_amu12_ang",
           "mean_quanta", "spread_of_quanta", "thermal_spread_amu12_ang",
           "classical_spread_amu12_ang",
           "thermal_spread_of", "THERMAL_ROW_KEYS", "thermal_row",
           "mode_over_atoms", "largest_move",
           "q_for_largest_move", "atom_moves", "displaced", "coordinate_of",
           "zero_point_displacement_ang", "largest_zero_point_move_ang",
           "motion_shares_of", "shares_by", "reduced_mass_amu",
           "force_constant_mdyn_per_ang", "THERMAL", "NUMBER", "SWING", "CLASSICAL",
           "CLASSICAL_SWING", "Frames", "Weights", "State", "state",
           "temperature_of_quanta", "GIVEN", "THE_RECORDS", "THE_MOTIONS",
           "THE_SETS",
           "ModeState", "Mode", "Vibration", "Drive",
           "Motion", "Estimate", "Parts", "Snapshots", "pooled"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.constants",)

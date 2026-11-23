"""A HARMONIC MODE'S MECHANICS -- every per-mode quantity, its definition and
its derivation, from one module (`engines/vibration.md` § 6.11;
`science/vibrational-averaging.md` § 2, § 2a, § 2b).

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
          displaced geometry ``R + Q L``, the largest move;
        * each element's share of its motion, ``m_i |L_i|^2`` summed by
          element over the total (`engines/vibration.md` § 6.6);
        * where the molecule sits along it in each state (§ 2b.3) -- n quanta
          exactly, thermal (at a temperature or at a mean number of quanta),
          a steady swing over a thermal bath, and the classical limits -- its
          density, distribution, moments, spread and mean quanta, and the
          frames that average over it (§ 2b.4);
        * :class:`Mode`, the one view of a mode a person computes with
          (`postprocess.vibration`, § 6.11 § 6).
USED-BY the result's derived values (`spectra.derived`), the thermochemistry
        (`spectra.normal_modes`), the frame-set maker (`spectra.mode_frames`),
        the transport average's motion (`frame_average`), the SIESTA
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
a call given one uses it; :class:`Mode` given none takes the temperature its
record states, and says so (:attr:`ModeState.temperature_from`).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from .constants import CM1_KELVIN, ZERO_POINT_Q2_AMU_ANG2_CM1
except ImportError:                         # beside a job, in a bundle
    from constants import CM1_KELVIN, ZERO_POINT_Q2_AMU_ANG2_CM1  # type: ignore[no-redef]


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
    of kelvin, or an array of frequencies (the thermochemistry's sums)."""
    temperature_k = checked_temperature_k(temperature_k)
    nu = np.asarray(frequency_cm1, dtype=float)
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


def element_shares(elements: Sequence[str], mode_rows,
                   masses_amu: Sequence[float]) -> Dict[str, float]:
    """Each element's share of a mode's motion, summing to 1: atom ``A``'s
    share is ``m_A |L_A|^2`` over the sum across atoms, summed by element
    (`engines/vibration.md` § 6.6) -- the kinetic energy's share, the same at
    every phase of the motion, so a property of the mode.  ``elements`` and
    ``masses_amu`` are a row's each, as the caller resolved them (a species
    label to its element; the result's masses, or the table's for a result
    stating none).  Largest first; a mode with no motion gives ``{}``."""
    rows = np.asarray(mode_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError(f"a mode is a row of x, y, z per atom; got the "
                         f"shape {rows.shape}")
    if len(elements) != rows.shape[0] or len(masses_amu) != rows.shape[0]:
        raise ValueError(
            f"{rows.shape[0]} rows of the mode, {len(elements)} elements and "
            f"{len(masses_amu)} masses: the mode does not match its atoms")
    weight: Dict[str, float] = {}
    total = 0.0
    for el, m, row in zip(elements, masses_amu, rows):
        w = float(m) * float(np.dot(row, row))
        weight[str(el)] = weight.get(str(el), 0.0) + w
        total += w
    if total <= 0.0:
        return {}
    return {el: w / total
            for el, w in sorted(weight.items(), key=lambda kv: -kv[1])}


# --------------------------------------------------------------------- #
#  Where the molecule sits along a mode -- the states (§ 2b.3)           #
# --------------------------------------------------------------------- #

#: THE STATES a mode's coordinate can be in (§ 2b.3), each a kind word.
THERMAL, NUMBER, SWING, CLASSICAL, CLASSICAL_SWING = (
    "thermal", "number", "swing", "classical", "classical swing")


@dataclass(frozen=True)
class Frames:
    """Where to compute, and with what weight, to average a response over a
    state (§ 2b.4): the rule's nodes along the mode ``nodes_amu12_ang`` and
    their ``weights``, summing to 1, the polynomial degree the rule is exact
    to, and -- the frames of a mode (:meth:`ModeState.frames`) -- each
    frame's geometry ``positions_ang``, every atom at ``R + Q_j L``;
    ``None`` for a state's alone, which has no atoms."""
    nodes_amu12_ang: Tuple[float, ...]
    weights: Tuple[float, ...]
    exact_to_degree: int
    positions_ang: Optional[Tuple[np.ndarray, ...]] = field(
        default=None, repr=False, compare=False)


def _gauss_hermite(nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """The Gauss–Hermite rule for the standard normal: points and weights
    summing to 1 (§ 5.1)."""
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    return math.sqrt(2.0) * x, w / math.sqrt(math.pi)


def _whole(value: Any, what: str, least: int) -> int:
    """``value`` as a whole number of ``least`` or more -- any integer type,
    numpy's too, never a boolean -- refused by name otherwise."""
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or int(value) < least):
        raise ValueError(f"{what} is a whole number, {least} or more; got "
                         f"{value!r}")
    return int(value)


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


def _gaussian_distribution(x: float, sigma: float) -> float:
    """The centred Gaussian of spread ``sigma``, the chance of ``x`` or
    less: ``(1 + erf(x / sqrt(2) sigma)) / 2`` -- for a spread of 0, 1 from
    0 on."""
    if sigma == 0.0:
        return 1.0 if float(x) >= 0.0 else 0.0
    return 0.5 * (1.0 + math.erf(float(x) / (math.sqrt(2.0) * sigma)))


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
    off = np.array(beta[:-1])
    jacobi = np.diag(alpha) + np.diag(off, 1) + np.diag(off, -1)
    x, vec = np.linalg.eigh(jacobi)
    return x, vec[0] ** 2


def _integrated(pdf, q: float, lo: float, panels: int = 20) -> float:
    """``int_lo^q pdf``, by 40-point Gauss–Legendre on ``panels`` equal
    panels, the density asked at every point at once -- to the last digits
    for the smooth densities here."""
    if q <= lo:
        return 0.0
    x, w = np.polynomial.legendre.leggauss(40)
    edges = np.linspace(lo, q, int(panels) + 1)
    mids = (edges[:-1] + edges[1:]) / 2.0
    half = (edges[1] - edges[0]) / 2.0
    points = (mids[:, None] + half * x[None, :]).ravel()
    values = np.asarray(pdf(points), dtype=float).reshape(len(mids), len(x))
    return float(half * np.sum(values * w[None, :]))


@dataclass(frozen=True)
class State:
    """Where a mode's coordinate ``Q`` (amu^1/2.Å) sits, in one state
    (§ 2b.3).  ``kind`` one of :data:`THERMAL`, :data:`NUMBER`,
    :data:`SWING`, :data:`CLASSICAL`, :data:`CLASSICAL_SWING`;
    ``zero_point_amplitude_amu12_ang`` the mode's ``Q_zp``; ``bath_quanta``
    the thermal (or random) quanta, a number state's ``quanta``, a swing's
    ``swing_quanta``; a classical Gaussian's ``classical_spread_amu12_ang``."""
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
    def mean_quanta(self) -> Optional[float]:
        """The state's mean number of quanta: a number state's own, the
        bath's, a swing's added to its bath's (§ 2a.2); ``None`` for a
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
        elif self.kind == SWING:
            s, q0 = self._bath_spread, self.swing_amplitude_amu12_ang
            centres = q0 * np.cos(_swing_phases(q0, s))
            out = _gaussian_density(q[..., None] - centres, s).mean(axis=-1)
        else:                                       # the classical swing
            q0 = self.swing_amplitude_amu12_ang
            inside = np.abs(q) < q0
            out = (_at_rest_density(q) if q0 == 0.0 else
                   np.where(inside, 1.0 / (math.pi * np.sqrt(
                       np.where(inside, q0 * q0 - q * q, 1.0))), 0.0))
        return float(out) if out.ndim == 0 else out

    def cdf(self, q: float) -> float:
        """The probability that ``Q`` is ``q`` or less."""
        q = float(q)
        if self.kind in (THERMAL, CLASSICAL):
            return _gaussian_distribution(q, self._gaussian_spread)
        if self.kind == SWING:
            s, q0 = self._bath_spread, self.swing_amplitude_amu12_ang
            return float(np.mean([_gaussian_distribution(q - c, s) for c in
                                  q0 * np.cos(_swing_phases(q0, s))]))
        if self.kind == CLASSICAL_SWING:
            q0 = self.swing_amplitude_amu12_ang
            if q >= q0:
                return 1.0
            if q <= -q0:
                return 0.0
            return 0.5 + math.asin(q / q0) / math.pi
        # n quanta: the density integrated from beyond its far turning point,
        # in panels fine enough for its n zeros
        reach = self._q_zp * (math.sqrt(2.0 * (2 * self.quanta + 1)) + 12.0)
        if q >= reach:
            return 1.0
        return min(1.0, _integrated(self.pdf, q, -reach,
                                    panels=20 + self.quanta // 4))

    def moment(self, k: int) -> float:
        """``<Q^k>`` exactly, for a whole ``k`` of 0 or more: the Gaussian's
        ``sigma^k (k-1)!!``; a number state's ``<n|xi^k|n>`` by the
        oscillator's own position operator; a swing's by the binomial over
        its bath and its phase (§ 2b.3)."""
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
        q0 = self.swing_amplitude_amu12_ang
        if self.kind == CLASSICAL_SWING:
            return q0 ** k * _cos_power_mean(k)
        s = self._bath_spread
        return sum(math.comb(k, j) * _gaussian_moment(s, j)
                   * q0 ** (k - j) * _cos_power_mean(k - j)
                   for j in range(k + 1))

    @property
    def _gaussian_spread(self) -> float:
        if self.kind == CLASSICAL:
            return float(self.classical_spread_amu12_ang)
        return self._bath_spread

    # ---- the frames ------------------------------------------------------
    def frames(self, nodes: int = 3) -> Frames:
        """The frames that average a response over the state (§ 2b.4): for
        a Gaussian, the Gauss–Hermite rule ``Q_j = sqrt(2) sigma x_j``; for
        n quanta, the Gauss rule whose weight function is that state's own
        density; for a swing, ``2 (nodes - 1)`` equal phases folded onto
        ``nodes`` positions -- one node, the swing's centre alone, exact to
        degree 1 -- each carrying the bath's own ``nodes``-point rule.  Each
        rule exact to the degree it states."""
        m = _whole(nodes, "nodes", 1)
        if self.kind in (THERMAL, CLASSICAL):
            x, w = _gauss_hermite(m)
            return Frames(tuple(float(v) for v in self._gaussian_spread * x),
                          tuple(float(v) for v in w), 2 * m - 1)
        if self.kind == NUMBER:
            x, wx = _number_state_rule(self.quanta, m)
            return Frames(tuple(float(v) for v in
                                math.sqrt(2.0) * self._q_zp * x),
                          tuple(float(v) for v in wx), 2 * m - 1)
        # a swing: its equal phases, folded where two land together
        q0 = self.swing_amplitude_amu12_ang
        if m == 1:
            positions, weights, degree = [0.0], [1.0], 1
        else:
            n_phase = 2 * (m - 1)
            positions = [q0 * math.cos(math.pi * j / (m - 1))
                         for j in range(m)]
            weights = [(1.0 if j in (0, m - 1) else 2.0) / n_phase
                       for j in range(m)]
            degree = n_phase - 1
        if self.kind == CLASSICAL_SWING:
            return Frames(tuple(positions), tuple(weights), degree)
        x, w = _gauss_hermite(m)
        s = self._bath_spread
        out_q, out_w = [], []
        for p, pw in zip(positions, weights):
            for xb, wb in zip(x, w):
                out_q.append(float(p + s * xb))
                out_w.append(float(pw * wb))
        return Frames(tuple(out_q), tuple(out_w), min(degree, 2 * m - 1))


def _swing_phases(amplitude: float, bath_spread: float) -> np.ndarray:
    """The equal phases a swing's density and distribution are averaged over
    (§ 2a.3): sixteen per bath spread of the swing's amplitude, 256 at least,
    so the swing's centre moves at most ``2 pi / 16`` of its bath's spread
    between two -- the equal-phase rule on a periodic Gaussian then
    converged to the last digits for a swing of any size."""
    count = max(256, 16 * math.ceil(amplitude / bath_spread))
    return 2.0 * math.pi * np.arange(count) / count


#: WHAT NAMES EACH STATE (§ 2b.3): the arguments a person gives, exactly one
#: of these sets -- a bath named by its temperature or by its mean quanta;
#: a classical swing alone has none.
_STATES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (NUMBER, ("quanta",)),
    (THERMAL, ("temperature_k",)),
    (THERMAL, ("mean_quanta",)),
    (SWING, ("swing_quanta", "temperature_k")),
    (SWING, ("swing_quanta", "mean_quanta")),
    (CLASSICAL, ("classical", "temperature_k")),
    (CLASSICAL_SWING, ("classical", "swing_quanta")),
)
_STATE_NAMED_BY: Dict[frozenset, str] = {
    frozenset(names): kind for kind, names in _STATES}


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
    * ``swing_quanta`` with ``temperature_k`` or ``mean_quanta`` -- a steady
      swing over a thermal bath;
    * ``classical`` with ``temperature_k`` -- the classical Gaussian, at
      rest at 0 K; with ``swing_quanta``, the classical swing alone.

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
    if kind == CLASSICAL:
        return State(CLASSICAL, q_zp, classical_spread_amu12_ang=(
            classical_spread_amu12_ang(frequency_cm1, temperature_k)))
    n_s = (0.0 if swing_quanta is None
           else _checked_quanta(swing_quanta, "swing_quanta"))
    if kind == CLASSICAL_SWING:
        return State(CLASSICAL_SWING, q_zp, swing_quanta=n_s)
    bath = _checked_quanta(_occupation(frequency_cm1, temperature_k)
                           if temperature_k is not None else mean_quanta)
    return State(kind, q_zp, bath_quanta=bath, swing_quanta=n_s)


def _checked_quanta(n: Any, what: str = "a mean number of quanta") -> float:
    """``n`` as a number of quanta, 0 or more, refused by name otherwise."""
    v = _finite_non_negative(n)
    if v is None:
        raise ValueError(f"{what} is 0 or more; got {n!r}")
    return v


# --------------------------------------------------------------------- #
#  One mode, as a person computes with it (§ 6.11 § 6)                  #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModeState:
    """A state of one mode, with the mode's atoms: the state's numbers
    (:class:`State`), each atom's r.m.s. move in it and frames as
    geometries.  ``temperature_k`` the temperature it was taken at, and
    ``temperature_from`` whether it was ``"given"`` or ``"the record's"``
    (§ 6.11 § 9 item 6) -- ``None`` for a state of no temperature."""
    state: State
    mode: "Mode"
    temperature_k: Optional[float] = None
    temperature_from: Optional[str] = None

    def pdf(self, q):
        return self.state.pdf(q)

    def cdf(self, q: float) -> float:
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
    def atom_rms_ang(self) -> np.ndarray:
        """Each atom's r.m.s. move in the state, Å: the spread times its
        row's length, ``sqrt(<Q^2>) |L_A|`` -- 0 for a held atom (§ 2b.2)."""
        return self.spread_amu12_ang * _row_lengths(
            self.mode.eigenvector_canonical_over_atoms_per_amu12)

    def frames(self, nodes: int = 3) -> Frames:
        """The state's frames (:meth:`State.frames`) with the geometries a
        calculation reads: each frame every atom at ``R + Q_j L``."""
        f = self.state.frames(nodes)
        return replace(f, positions_ang=tuple(
            self.mode.displaced(q) for q in f.nodes_amu12_ang))


@dataclass(frozen=True)
class Mode:
    """One normal mode with its atoms -- every number of § 6.11 § 6 derived
    by this module from what the result states, each field the record's, by
    the data file's name, its unit in it.
    ``eigenvector_canonical_per_amu12`` the record's canonical eigenvector, a
    row per free atom (§ 6.3), over ``free_atom_idxs``, the atoms the mode
    moves; ``masses_amu`` every atom's, in the user's order;
    ``element_symbols`` every atom's element, its species label resolved
    (``Au`` for ``Au1``); ``positions_ang`` the geometry the modes were
    taken at; ``record_temperature_k`` the temperature the record states
    (its thermochemistry's), ``None`` where it states none.  The pattern
    over every atom is :attr:`eigenvector_canonical_over_atoms_per_amu12`."""
    index_1based: int
    frequency_cm1: float
    has_imag: bool
    eigenvector_canonical_per_amu12: np.ndarray = field(repr=False)
    free_atom_idxs: Tuple[int, ...] = field(repr=False)
    masses_amu: np.ndarray = field(repr=False)
    element_symbols: Tuple[str, ...] = field(repr=False)
    positions_ang: np.ndarray = field(repr=False)
    record_temperature_k: Optional[float] = None

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
        :attr:`free_atom_idxs` -- the record's own field and shape (§ 6.3);
        ``None`` for an imaginary mode.  Every atom's is
        ``displacement(zero_point_widths=1)``."""
        q = self.zero_point_amplitude_amu12_ang
        return None if q is None else atom_moves(
            self.eigenvector_canonical_per_amu12, q)

    @property
    def largest_zero_point_move_ang(self) -> Optional[float]:
        """The atom that moves most at ``Q_zp``, its move's length in Å."""
        q = self.zero_point_amplitude_amu12_ang
        return None if q is None else q * largest_move(
            self.eigenvector_canonical_per_amu12)

    @property
    def element_shares(self) -> Dict[str, float]:
        """Each element's share of the motion, summed over the free atoms
        (§ 6.11 § 3) -- a held atom adds nothing to its element's share."""
        free = list(self.free_atom_idxs)
        return element_shares([self.element_symbols[i] for i in free],
                              self.eigenvector_canonical_per_amu12,
                              self.masses_amu[free])

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
        thermal or classical state, a swing's bath, given neither a
        temperature nor mean quanta -- take the temperature the record
        states, and say so (§ 6.11 § 9 item 6); a record stating none is
        asked for one.  n quanta, mean quanta and a classical swing alone
        take none."""
        q_zp = self.zero_point_amplitude_amu12_ang
        if q_zp is None:
            raise ValueError(f"mode {self.index_1based} is imaginary: it has "
                             f"no harmonic state")
        named = _named(temperature_k, mean_quanta, quanta, swing_quanta,
                       classical)
        source: Optional[str] = "given" if temperature_k is not None else None
        if (named not in _STATE_NAMED_BY
                and named | {"temperature_k"} in _STATE_NAMED_BY):
            if self.record_temperature_k is None:
                raise ValueError(
                    "name a temperature: the record states none")
            temperature_k = self.record_temperature_k
            source = "the record's"
        return ModeState(
            state(q_zp, self.frequency_cm1, temperature_k=temperature_k,
                  mean_quanta=mean_quanta, quanta=quanta,
                  swing_quanta=swing_quanta, classical=classical),
            self, temperature_k=temperature_k, temperature_from=source)

    def with_frequency(self, frequency_cm1: float) -> "Mode":
        """The mode at another frequency -- a scaled or a measured one -- its
        pattern kept (§ 2b.1): ``Q_zp``, its displacement and every spread
        follow from the new frequency."""
        nu = float(frequency_cm1)
        if not (math.isfinite(nu) and nu > 0.0):
            raise ValueError(f"a frequency is a positive number of cm-1; got "
                             f"{frequency_cm1!r}")
        return replace(self, frequency_cm1=nu, has_imag=False)

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


@dataclass(frozen=True)
class Vibration:
    """A vibration result's modes, as a person computes with them (§ 6.11
    § 6): ``modes`` in the result's order (``modes[6]`` is mode 7), every
    atom in the user's order; ``record_temperature_k`` the temperature the
    record states (its thermochemistry's), which a state asked without one
    takes; ``source`` the file it was read from."""
    modes: Tuple[Mode, ...]
    record_temperature_k: Optional[float]
    source: str = ""


__all__ = ["checked_temperature_k", "zero_point_amplitude_amu12_ang",
           "mean_quanta", "spread_of_quanta", "thermal_spread_amu12_ang",
           "classical_spread_amu12_ang",
           "thermal_spread_of", "mode_over_atoms", "largest_move",
           "q_for_largest_move", "atom_moves", "displaced", "coordinate_of",
           "element_shares", "THERMAL", "NUMBER", "SWING", "CLASSICAL",
           "CLASSICAL_SWING", "Frames", "State", "state", "ModeState", "Mode",
           "Vibration"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.constants",)

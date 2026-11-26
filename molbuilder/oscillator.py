"""ONE HARMONIC OSCILLATOR OF A FREQUENCY -- its zero-point amplitude, its
thermal quanta and spreads, and its levels in its own unit
(`engines/vibration-frames.md` § 9.1 module 1;
`science/vibrational-averaging.md` § 2, § 2b, § 2d.2, § 2d.5, § 2e).

MODULE  oscillator (L2; numpy, math and the constants, nothing else of
        molbuilder)
ROLE    the quantum mechanics of one harmonic oscillator, nothing molecular:

        * the mechanics every mode's numbers take -- the zero-point
          amplitude ``Q_zp = sqrt(hbar / 2 omega)`` from a frequency, Bose
          and Einstein's mean number of quanta at a temperature and the
          temperature of a mean, the spread a mean gives, the classical
          spread -- each refused by name on a value that is none;
        * :class:`Oscillator`, one oscillator of a frequency: the same in
          its own unit ``x = Q / Q_zp``;
        * its levels ``|n>`` in ``x``: each level's wavefunction and density
          (:func:`level_wavefunction`, :func:`level_density`), its averages
          (:func:`level_moment`), and the ladder and position operators on
          a window of levels (:func:`lowering_matrix`,
          :func:`raising_matrix`, :func:`position_matrix`) -- a level's
          distribution function is `distribution.Level`'s, from the
          level's one table here (:func:`_number_distribution`).
USED-BY `vibration` (which hands the mechanics on under their names -- read
        there by the record's writers, `spectra.derived` and
        `spectra.normal_modes`, the frame-set doors, `vibration_frames` and
        `cli`, and the spectrum tab's door, `web.blueprints.spectra`),
        `oscillator_state`, `distribution`, `sampling` (the whole-number
        rule), and a person's script (`postprocess` hands the module on).
TRAVELS with `vibration` in both vibration bundles (`runwrap.bundle_members`),
        so it imports the constants two ways and nothing else of molbuilder.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from .constants import (CM1_EV, CM1_KELVIN,
                            ZERO_POINT_Q2_AMU_ANG2_CM1)
except ImportError:                         # beside a job, in a bundle
    from constants import (  # type: ignore[no-redef]
        CM1_EV, CM1_KELVIN, ZERO_POINT_Q2_AMU_ANG2_CM1)


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


def _checked_frequency(raw: Any, what: str):
    """A frequency, or an array of them, each a positive number of cm-1 --
    an imaginary mode, or one of no frequency, holds no oscillator -- refused
    by name otherwise (``what`` says whose); a truth value or text is none."""
    try:
        nu = np.asarray(float("nan") if isinstance(raw, (bool, str, bytes))
                        else raw, dtype=float)
    except (TypeError, ValueError):
        nu = np.asarray(float("nan"))
    if nu.size == 0 or not np.all(np.isfinite(nu) & (nu > 0.0)):
        raise ValueError(f"{what} needs a positive frequency, in cm-1; got "
                         f"{raw!r}")
    return float(nu) if nu.ndim == 0 else nu


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


def bose_einstein_mean_quanta(frequency_cm1, temperature_k):
    """A harmonic mode's mean number of quanta at ``temperature_k``, Bose and
    Einstein's ``n = 1 / (exp(hbar omega / k_B T) - 1)`` -- 0 at 0 K (§ 2a).
    Written as ``e^-x / (1 - e^-x)``, which neither overflows for a stiff
    mode in the cold nor loses digits for a soft one in the heat.  A number
    of kelvin, or an array of frequencies (the thermochemistry's sums);
    each frequency positive -- an imaginary mode, or one of no frequency,
    holds no thermal state, and is refused by name."""
    temperature_k = checked_temperature_k(temperature_k)
    nu = np.asarray(_checked_frequency(frequency_cm1, "a mode's mean quanta"))
    if temperature_k == 0.0:
        out = np.zeros_like(nu)
    else:
        x = nu * CM1_KELVIN / temperature_k
        out = np.exp(-x) / -np.expm1(-x)
    return float(out) if out.ndim == 0 else out


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
    (:func:`spread_of_quanta`), at the temperature's (:func:`bose_einstein_mean_quanta`)."""
    return spread_of_quanta(zero_point_amplitude,
                            bose_einstein_mean_quanta(frequency_cm1,
                                                      temperature_k))


def classical_spread_amu12_ang(frequency_cm1: float,
                               temperature_k: float) -> float:
    """The classical spread at ``temperature_k``, ``sqrt(k_B T / omega^2)``
    = ``Q_zp sqrt(2 k_B T / hbar omega)`` -- the high-temperature limit of
    the thermal one (§ 2b.3); 0 at 0 K, the classical mode at rest."""
    temperature_k = checked_temperature_k(temperature_k)
    nu = _checked_frequency(frequency_cm1, "a classical spread")
    return zero_point_amplitude_amu12_ang(nu) * math.sqrt(
        2.0 * temperature_k / (nu * CM1_KELVIN))


def _checked_quanta(n: Any, what: str = "a mean number of quanta") -> float:
    """``n`` as a number of quanta, 0 or more, refused by name otherwise."""
    v = _finite_non_negative(n)
    if v is None:
        raise ValueError(f"{what} is 0 or more; got {n!r}")
    return v


def bose_einstein_temperature(frequency_cm1: float,
                              mean_quanta: float) -> float:
    """The temperature whose thermal state holds ``mean_quanta`` quanta of a
    mode at ``frequency_cm1`` -- a mode pumped at random moments, its
    effective temperature (science § 2a.1, § 2a.3):
    ``T = h c nu / (k_B ln(1 + 1/n))``, 0 K at no quanta."""
    nu = _checked_frequency(frequency_cm1, "a Bose-Einstein temperature")
    n = _checked_quanta(mean_quanta)
    if n == 0.0:
        return 0.0
    return nu * CM1_KELVIN / math.log1p(1.0 / n)


def _quanta_words(n: float, kind: str = "", spec: str = "g") -> str:
    """A COUNT OF QUANTA AS SAID: ``n`` written by ``spec``, ``kind`` the
    quanta's (``"mean "``, or none) -- the noun chosen on the figure
    written, so a mean that rounds to one is "1 quantum" (`web/spectra.md`
    § 4.3)."""
    text = format(float(n), spec)
    return f"{text} {kind}quant{'um' if text == '1' else 'a'}"


def _whole(value: Any, what: str, least: int) -> int:
    """``value`` as a whole number of ``least`` or more -- any integer type,
    numpy's too, never a boolean -- refused by name otherwise."""
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or int(value) < least):
        raise ValueError(f"{what} is a whole number, {least} or more; got "
                         f"{value!r}")
    return int(value)

# --------------------------------------------------------------------- #
#  The levels, in xi = Q / sqrt(2) Q_zp (the recurrence's own unit)      #
# --------------------------------------------------------------------- #

def _eigenfunction(n: int, xi) -> np.ndarray:
    """The oscillator's ``n``-th eigenfunction, ``psi_n(xi)``, normalised in
    ``xi``: the recurrence's level ``n`` (:func:`_level_sweep`), the
    Gaussian applied once at the end."""
    xi = np.asarray(xi, dtype=float)
    for _, h, log_scale in _level_sweep(n + 1, xi):
        pass
    return h * np.exp(log_scale - 0.5 * xi * xi)


def _level_sweep(levels: int, xi):
    """THE ONE RECURRENCE every level is taken by: for ``k = 0 ...
    levels - 1``, ``(k, h_k, log_scale)`` with ``psi_k(xi) = h_k
    e^{log_scale - xi^2/2}`` -- ``h_{k+1} = sqrt(2/(k+1)) xi h_k -
    sqrt(k/(k+1)) h_{k-1}`` from ``h_0 = pi^-1/4``, each value rescaled where
    it grows past 1e100 and the scale kept as a logarithm, so no level and no
    ``xi`` overflows (:func:`_eigenfunction`; `distribution`'s lists)."""
    xi = np.asarray(xi, dtype=float)
    h_prev = np.zeros_like(xi)
    h = np.full_like(xi, math.pi ** -0.25)
    log_scale = np.zeros_like(xi)
    for k in range(levels):
        if k:
            h_prev, h = h, (math.sqrt(2.0 / k) * xi * h
                            - math.sqrt((k - 1) / k) * h_prev)
            big = np.abs(h) > 1e100
            if np.any(big):
                h = np.where(big, h * 1e-100, h)
                h_prev = np.where(big, h_prev * 1e-100, h_prev)
                log_scale = np.where(big, log_scale + 100.0 * math.log(10.0),
                                     log_scale)
        yield k, h, log_scale


def _table_shape(top: int) -> Tuple[float, int]:
    """Where a table of levels up to ``top`` reaches and how many panels it
    takes (:class:`_Table`): nine past the far turning point
    ``sqrt(2 top + 1)``, ``e^-81`` of the density beyond, in ``2 (top + 20)``
    panels -- one or more between neighbouring zeros of the top level."""
    return math.sqrt(2.0 * top + 1.0) + 9.0, 2 * (top + 20)


def _bracketed_newton(chance, density, p, lo, hi, x, steps: int = 200):
    """THE POINTS WHOSE CHANCE IS ``p`` -- each from its start ``x``,
    inside its bracket ``[lo, hi]`` (science § 2e.6).  Each pass asks first
    how far the point misses, ``chance(x) - p``: a point that misses by
    nothing is the answer, and is kept.  Otherwise the miss's sign narrows
    the bracket, and the next point is a Newton step where that stays inside
    the bracket and is at most half the last step, the bracket's middle
    otherwise -- so a flat start (density 0) halves the bracket, and no point
    bounces between two places (a two-humped density's flat middle).  A
    point stops once its step is within four units of the last digit of
    ``max(1, |x|)`` -- near 0, within about 9e-16 -- and after ``steps``
    passes in any case: each pass at least halves a bracket or takes a
    Newton step at most half the last, so 200 passes narrow any bracket a
    double holds below its spacing.  ``chance`` and ``density`` take an
    array of points."""
    p = np.asarray(p, dtype=float).reshape(-1)
    lo = np.array(lo, dtype=float).reshape(-1) * np.ones_like(p)
    hi = np.array(hi, dtype=float).reshape(-1) * np.ones_like(p)
    x = np.array(x, dtype=float).reshape(-1) * np.ones_like(p)
    last = np.full_like(p, np.inf)
    active = np.ones(p.shape[0], dtype=bool)
    for _ in range(steps):
        at = np.flatnonzero(active)
        if not at.size:
            break
        miss = np.asarray(chance(x[at]), dtype=float) - p[at]
        active[at[miss == 0.0]] = False          # already the answer: kept
        at, miss = at[miss != 0.0], miss[miss != 0.0]
        if not at.size:
            break
        xa = x[at]
        lo[at] = np.where(miss < 0.0, xa, lo[at])
        hi[at] = np.where(miss > 0.0, xa, hi[at])
        d = np.asarray(density(xa), dtype=float)
        newton = xa - np.where(d > 0.0, miss / np.where(d > 0.0, d, 1.0),
                               np.inf)
        good = ((newton >= lo[at]) & (newton <= hi[at])
                & (np.abs(newton - xa) <= 0.5 * last[at]))
        step = np.where(good, newton, (lo[at] + hi[at]) / 2.0)
        last[at] = np.abs(step - xa)
        x[at] = step
        active[at[last[at] <= 4.0 * np.spacing(np.maximum(1.0, np.abs(xa)))]] \
            = False
    return x


def _position_times(v, first: int = 0) -> np.ndarray:
    """``xi`` applied to ``v`` -- its rows the levels ``first, first + 1,
    ...``, any columns -- as the tridiagonal it is, ``<j|xi|j+1> =
    sqrt((j+1)/2)``, the one place the position's elements are written: in
    time and memory of ``v``'s size, not its square."""
    v = np.asarray(v)
    off = np.sqrt(np.arange(first + 1, first + v.shape[0]) / 2.0)
    if v.ndim > 1:
        off = off.reshape((-1,) + (1,) * (v.ndim - 1))
    out = np.zeros_like(v)
    out[:-1] += off * v[1:]
    out[1:] += off * v[:-1]
    return out


def _position_operator(lo: int, hi: int) -> np.ndarray:
    """``xi = (a + a^dagger) / sqrt(2)`` on the oscillator's states ``lo`` to
    ``hi - 1`` as a matrix, :func:`_position_times` on each level -- exact
    on every vector it moves without leaving them: :func:`position_matrix`,
    and `vibration.State`'s moments on the window of levels ``xi`` reaches
    from ``|n>``, a few states wide."""
    return _position_times(np.eye(hi - lo), lo)


#: THE ONE EPSILON: two numbers are one, or a number is zero, within this
#: share of the size of what it was computed from -- every "is this zero,
#: are these equal" question the tools ask (`distribution`: a typed list's
#: chance just below 0, the degree frames reproduce, whether weights are the
#: ones written, a weight below zero); why 1e-12, science § 2e.6.
_EPSILON = 1e-12

_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


class _Table:
    """A one-dimensional distribution tabulated once: its ``density`` on
    ``[-reach, reach]`` cut into ``panels`` panels, each integrated by
    40-point Gauss–Legendre, exact to degree 79, so each panel's integral to
    its last digits; the running integral at every panel's edge -- the
    chance of a point or less the table's edge plus that panel's own piece,
    every point asked at once (`science/vibrational-averaging.md` § 2d.5,
    § 2e.3)."""

    def __init__(self, density, reach: float, panels: int):
        self._density = density
        self.reach = float(reach)
        self.edges = np.linspace(-self.reach, self.reach, int(panels) + 1)
        half = (self.edges[1] - self.edges[0]) / 2.0
        mids = (self.edges[:-1] + self.edges[1:]) / 2.0
        points = mids[:, None] + half * _GL_X[None, :]
        pieces = half * (self.density(points) * _GL_W[None, :]).sum(axis=1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(pieces)))

    def density(self, xi) -> np.ndarray:
        return self._density(xi)

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
        """The ``xi`` whose chance of ``xi`` or less is ``p`` -- for ``p`` of
        one half or less found from the left, where the running integral
        keeps its digits: from the table's panel, by the one bracketed Newton
        (:func:`_bracketed_newton`), the panel its first bracket."""
        p = np.asarray(p, dtype=float)
        shape, p = p.shape, p.reshape(-1)
        k = np.clip(np.searchsorted(self.cumulative, p, side="right") - 1, 0,
                    len(self.edges) - 2)
        lo, hi = self.edges[k], self.edges[k + 1]
        span = np.maximum(self.cumulative[k + 1] - self.cumulative[k],
                          1e-300)
        x = lo + (hi - lo) * np.clip((p - self.cumulative[k]) / span, 0, 1)
        return _bracketed_newton(self.below, self.density, p, lo, hi,
                                 x).reshape(shape)


class _NumberDistribution(_Table):
    """``P_n``'s distribution in ``xi = Q / sqrt(2) Q_zp``, tabulated once
    per ``n`` (:class:`_Table`, its shape :func:`_table_shape`'s)."""

    def __init__(self, n: int):
        self.n = n
        super().__init__(lambda xi: _eigenfunction(n, xi) ** 2,
                         *_table_shape(n))


_NUMBER_TABLES: Dict[int, _NumberDistribution] = {}


def _number_distribution(n: int) -> _NumberDistribution:
    if n not in _NUMBER_TABLES:
        _NUMBER_TABLES[n] = _NumberDistribution(n)
    return _NUMBER_TABLES[n]

# --------------------------------------------------------------------- #
#  The levels, in the oscillator's own unit x = Q / Q_zp                 #
# --------------------------------------------------------------------- #

_SQRT2 = math.sqrt(2.0)


def level_wavefunction(n: int, x) -> np.ndarray:
    """Level ``n``'s wavefunction at ``x = Q / Q_zp``, normalised in ``x``,
    ``int psi_n(x)^2 dx = 1`` (science § 2d.2): the recurrence's in ``xi =
    x / sqrt 2`` (:func:`_eigenfunction`), times ``2^(-1/4)``."""
    n = _whole(n, "a level", 0)
    return _eigenfunction(n, np.asarray(x, dtype=float) / _SQRT2) / 2.0 ** 0.25


def level_density(n: int, x) -> np.ndarray:
    """Level ``n``'s density at ``x = Q / Q_zp``, ``psi_n(x)^2``."""
    psi = level_wavefunction(n, x)
    return psi * psi


def _window(lo: Any, hi: Any) -> Tuple[int, int]:
    """A window of levels ``lo`` to ``hi - 1``, one level or more -- refused
    by name otherwise."""
    lo = _whole(lo, "lo, a window's first level", 0)
    return lo, _whole(hi, "hi, one past a window's last level", lo + 1)


def lowering_matrix(lo: int, hi: int) -> np.ndarray:
    """The lowering operator ``b`` on the levels ``lo`` to ``hi - 1``:
    ``b|n> = sqrt(n) |n-1>`` -- the matrix ``<n-1|b|n> = sqrt(n)`` (science
    § 2d.2)."""
    lo, hi = _window(lo, hi)
    return np.diag(np.sqrt(np.arange(lo + 1, hi, dtype=float)), 1)


def raising_matrix(lo: int, hi: int) -> np.ndarray:
    """The raising operator ``b^dagger`` on the levels ``lo`` to ``hi - 1``:
    ``b^dagger|n> = sqrt(n+1) |n+1>``, the transpose of
    :func:`lowering_matrix`."""
    return lowering_matrix(lo, hi).T


def position_matrix(lo: int, hi: int) -> np.ndarray:
    """The position ``x = b + b^dagger`` on the levels ``lo`` to ``hi - 1``:
    ``<j|x|j+1> = sqrt(j + 1)`` -- exact on every vector it moves without
    leaving them (:func:`_position_operator`, in ``xi``, times
    ``sqrt 2``)."""
    return _SQRT2 * _position_operator(*_window(lo, hi))


def level_moment(n: int, k: int) -> float:
    """``<n|x^k|n>`` exactly: 0 for an odd ``k``; for an even one, ``x``
    applied ``k/2`` times to ``|n>`` on the window of levels it reaches,
    ``n - k/2`` to ``n + k/2`` -- ``<n|x^2|n> = 2n + 1``, ``<n|x^4|n> =
    3(2n^2 + 2n + 1)`` (science § 2e.3)."""
    n = _whole(n, "a level", 0)
    k = _whole(k, "a moment's order", 0)
    if k % 2:
        return 0.0
    lo, hi = max(0, n - k // 2), n + k // 2 + 1
    v = np.zeros(hi - lo)
    v[n - lo] = 1.0
    for _ in range(k // 2):
        v = _SQRT2 * _position_times(v, lo)
    return float(v @ v)


# --------------------------------------------------------------------- #
#  One oscillator of a frequency                                         #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Oscillator:
    """ONE HARMONIC OSCILLATOR OF A FREQUENCY, ``frequency_cm1`` in cm-1
    (`engines/vibration-frames.md` § 9.1 module 1): the oscillator a chain
    of steps is placed on (`oscillator_state`), its own unit ``x = Q /
    Q_zp``.  A frequency that is no positive number is refused by name: an
    imaginary mode has no oscillator."""
    frequency_cm1: float

    def __post_init__(self):
        nu = self.frequency_cm1
        if np.ndim(nu):
            raise ValueError(f"an oscillator has one frequency, in cm-1; got "
                             f"{nu!r}")
        object.__setattr__(self, "frequency_cm1",
                           _checked_frequency(nu, "an oscillator"))

    @property
    def zero_point_amplitude_amu12_ang(self) -> float:
        """The zero-point amplitude ``Q_zp`` in amu^1/2.Å (science § 2b.1)."""
        return zero_point_amplitude_amu12_ang(self.frequency_cm1)

    @property
    def quantum_ev(self) -> float:
        """Its quantum ``hbar omega``, in eV."""
        return self.frequency_cm1 * CM1_EV

    def bose_einstein_mean_quanta(self, temperature_k: float) -> float:
        """Bose and Einstein's mean number of quanta at ``temperature_k``
        (:func:`bose_einstein_mean_quanta`)."""
        return bose_einstein_mean_quanta(self.frequency_cm1, temperature_k)

    def bose_einstein_temperature(self, mean_quanta: float) -> float:
        """The temperature whose thermal state holds ``mean_quanta`` quanta
        (:func:`bose_einstein_temperature`)."""
        return bose_einstein_temperature(self.frequency_cm1, mean_quanta)

    def classical_spread(self, temperature_k: float) -> float:
        """The classical bell curve's width at ``temperature_k`` in ``x``,
        ``sqrt(2 k_B T / hbar omega)`` -- no zero-point motion (science
        § 2b.3; :func:`classical_spread_amu12_ang` over ``Q_zp``)."""
        return (classical_spread_amu12_ang(self.frequency_cm1, temperature_k)
                / self.zero_point_amplitude_amu12_ang)


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.constants",)

__all__ = ["checked_temperature_k", "zero_point_amplitude_amu12_ang",
           "bose_einstein_mean_quanta", "spread_of_quanta",
           "thermal_spread_amu12_ang", "classical_spread_amu12_ang",
           "bose_einstein_temperature",
           "Oscillator", "level_wavefunction", "level_density",
           "lowering_matrix", "raising_matrix", "position_matrix",
           "level_moment"]

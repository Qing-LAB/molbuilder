"""ONE COORDINATE'S DISTRIBUTION -- and every number a frame set takes from
it: its density, its distribution function, its averages, a quantile, its
Gauss rule, a draw from one normal number, and the weights for
points placed elsewhere (`engines/vibration-frames.md` § 9.1 module 3;
`science/vibrational-averaging.md` § 2a.3, § 2b.4, § 2d.5, § 2e.3, § 5.4).

MODULE  distribution (L2; numpy and math, and `oscillator` -- nothing else of
        molbuilder)
ROLE    statistics of one coordinate, in the unit it is given -- a mode's
        states in ``x = Q / Q_zp``:

        * the families, each with a closed form: a bell curve
          (:class:`Gaussian`); one quantum on a warm mode, the bell curve
          times a quadratic (:class:`QuantumAdded`); one level
          (:class:`Level`);
        * the general route: a list over the levels (:class:`LevelList`),
          science § 2e.3 -- every distribution here symmetric about 0;
        * :func:`of_state`, a state's distribution by the route its reduced
          chain takes (`engines/vibration-frames.md` § 9.1's diagram);
        * the numerics every rule takes: Gauss--Hermite, a discrete
          measure's Gauss rule by Lanczos and Golub--Welsch, a level's rule.
USED-BY `vibration` (a mode's states and frames), and a person's script
        (`postprocess` hands the module on as ``dist``).
TRAVELS with `vibration` in both vibration bundles: numpy, math and
        `oscillator`, imported two ways.
"""
from __future__ import annotations

import functools
import math
import statistics
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from .oscillator import (_EPSILON, _bracketed_newton, _checked_quanta,
                             _finite_non_negative, _level_sweep,
                             _number_distribution, _position_times,
                             _table_shape, _Table, _whole, level_density,
                             level_moment, spread_of_quanta)
except ImportError:                         # beside a job, in a bundle
    from oscillator import (  # type: ignore[no-redef]
        _EPSILON, _bracketed_newton, _checked_quanta, _finite_non_negative,
        _level_sweep, _number_distribution, _position_times, _table_shape,
        _Table, _whole, level_density, level_moment, spread_of_quanta)

_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------- #
#  The numerics every rule takes                                         #
# --------------------------------------------------------------------- #

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


def _number_state_rule(n: int, nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """The ``nodes``-point Gauss rule whose weight function is level ``n``'s
    density ``P_n`` (in ``xi``): by Lanczos from ``|n>`` on ``xi`` and Golub
    and Welsch's eigenproblem [GolubWelsch1969], exact for its polynomials
    to degree ``2 nodes - 1``.  ``nodes`` steps reach ``nodes`` levels either
    side of ``n`` -- the window the position is applied on, level to
    neighbour (`oscillator._position_times`), so the work is in proportion
    to the window, never its square; nodes in ``xi``."""
    lo, hi = max(0, n - nodes), n + 1 + nodes
    q = np.zeros(hi - lo)
    q[n - lo] = 1.0
    q_prev = np.zeros_like(q)
    alpha: List[float] = []
    beta: List[float] = []
    b = 0.0
    for _ in range(nodes):
        w = _position_times(q, lo) - b * q_prev
        a = float(q @ w)
        w = w - a * q
        alpha.append(a)
        b = float(np.linalg.norm(w))
        beta.append(b)
        if b == 0.0:
            break
        q_prev, q = q, w / b
    return _jacobi_rule(alpha, beta)


def _standard_normal(z):
    """The standard normal's chance of ``z`` or less, ``Phi(z)``, from its
    lower tail for ``z < 0`` so the far tails keep their digits."""
    z = np.asarray(z, dtype=float)
    return np.where(z < 0.0,
                    0.5 * np.asarray(_erfc(-z / _SQRT2), dtype=float),
                    1.0 - 0.5 * np.asarray(_erfc(z / _SQRT2), dtype=float))


_NORMAL = statistics.NormalDist()


def _normal_quantile_one(p: float) -> float:
    """``Phi^-1(p)``, the standard library's (``statistics.NormalDist``,
    Wichura's algorithm AS 241, good to about 1e-16) -- ``-inf`` at 0 and
    ``inf`` at 1, where no finite point has that chance."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    return _NORMAL.inv_cdf(p)


#: ``Phi^-1`` over an array (:func:`_normal_quantile_one`).
_standard_normal_quantile = np.vectorize(_normal_quantile_one, otypes=[float])


def _invert(cdf, pdf, p, lo: float, hi: float):
    """The points whose chance is ``p``, each inside ``[lo, hi]`` -- the one
    bracketed Newton (`oscillator._bracketed_newton`), from the bracket's
    middle."""
    p = np.asarray(p, dtype=float)
    return _bracketed_newton(cdf, pdf, p, lo, hi,
                             (lo + hi) / 2.0).reshape(p.shape)


def _degree_reproduced(w, x, moment, scale: float, m: int) -> int:
    """The highest degree ``d`` (``m - 1`` up to ``2m - 1``) to which the
    weights ``w`` at the scaled points ``x`` reproduce every average of
    ``1, x, ..., x^d`` -- each within the one epsilon of the sums it is
    (`oscillator._EPSILON`; science § 2e.6)."""
    degree = m - 1
    for k in range(m, 2 * m):
        want = moment(k) / scale ** k
        terms = w * x ** k
        if abs(float(terms.sum()) - want) > _EPSILON * max(
                float(np.abs(terms).sum()), abs(want), 1.0):
            break
        degree = k
    return degree


@dataclass(frozen=True)
class PointWeights:
    """The weights with which points reproduce a distribution's averages
    (:func:`points_weights`): ``weights``; ``exact_to_degree``; ``own``,
    whether the points are the distribution's own rule (its weights for
    them those ``written``, or reproducing it to ``2m - 1``); ``beyond``,
    the share of the distribution beyond the outermost points, both sides;
    ``below_zero``, the indices of the weights below zero past rounding."""
    weights: np.ndarray
    exact_to_degree: int
    own: bool
    beyond: float
    below_zero: Tuple[int, ...]


def points_weights(points, moment: Callable[[int], float], cdf, scale: float,
                   written=None) -> PointWeights:
    """THE WEIGHTS FOR POINTS PLACED ELSEWHERE (science § 5.4): the weights
    with which ``points`` reproduce a distribution's averages of ``1, x,
    ..., x^(m-1)`` -- ``moment(k)`` its averages ``<x^k>``, ``cdf`` its
    distribution function, ``scale`` its spread, the unit the solve works in
    -- and what they come to (:class:`PointWeights`).  Two points at one
    position are refused by name: no weights separate them."""
    q = np.asarray(points, dtype=float).reshape(-1)
    m = q.shape[0]
    if m == 0:
        raise ValueError("weights need one point at least")
    if len(set(q.tolist())) < m:
        raise ValueError("two points sit at one position: no weights "
                         "separate them")
    scale = float(scale) or 1.0
    x = q / scale
    a = np.vander(x, m, increasing=True).T
    b = np.array([moment(k) / scale ** k for k in range(m)])
    w = np.linalg.solve(a, b)
    degree = _degree_reproduced(w, x, moment, scale, m)
    own = degree >= 2 * m - 1 or (
        written is not None
        and np.allclose(w, np.asarray(written, dtype=float).reshape(-1),
                        rtol=0.0, atol=_EPSILON))
    ends = np.atleast_1d(cdf(np.array([q.min(), q.max()])))
    beyond = float(max(0.0, ends[0]) + max(0.0, 1.0 - ends[1]))
    below = tuple(int(i) for i, wi in enumerate(w) if wi < -_EPSILON)
    return PointWeights(w, degree, own, beyond, below)


# --------------------------------------------------------------------- #
#  The distributions                                                     #
# --------------------------------------------------------------------- #

class Distribution:
    """ONE COORDINATE'S DISTRIBUTION, in the unit it is given: its density
    ``pdf``, its distribution function ``cdf``, its averages ``moment(k)``,
    ``quantile(p)`` (the point with ``cdf = p``), its ``gauss_rule(nodes)`` --
    ``nodes`` points and weights reproducing its averages to degree ``2 nodes
    - 1``,
    as found -- a draw from one standard normal number ``z``, the point with
    ``cdf = Phi(z)`` (science § 2d.5), and ``weights_at(points)``.  Every
    distribution here is symmetric about 0, ``P(x) = P(-x)``: a list's
    density is a sum of the levels' even densities, and every family's is
    even (science § 2e.3)."""

    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def moment(self, k: int) -> float:
        raise NotImplementedError

    def gauss_rule(self, nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _reach(self) -> float:
        """A point beyond which nothing of the distribution is left."""
        return 12.0 * math.sqrt(max(self.moment(2), 1e-300))

    def quantile(self, p):
        """The point whose chance of it or less is ``p``."""
        r = self._reach()
        return _invert(self.cdf, self.pdf, p, -r, r)

    def draw(self, z):
        """The coordinate one standard normal number ``z`` gives, the point
        where ``cdf`` reaches ``Phi(z)`` -- taken from the lower tail with
        ``z``'s sign, the distribution symmetric, so ``-z`` gives exactly
        ``-x`` and the far tails keep their digits (science § 2d.5)."""
        z = np.asarray(z, dtype=float)
        left = self.quantile(0.5 * np.asarray(_erfc(np.abs(z) / _SQRT2),
                                              dtype=float))
        return np.where(z == 0.0, 0.0, -np.sign(z) * left)

    def weights_at(self, points, written=None) -> PointWeights:
        """The weights for points placed elsewhere (:func:`points_weights`)
        -- ``points`` in the distribution's own unit, a state's ``x = Q /
        Q_zp`` (`engines/vibration-frames.md` § 9.1)."""
        return points_weights(points, self.moment, self.cdf,
                              math.sqrt(self.moment(2)), written)


class Gaussian(Distribution):
    """A bell curve of spread ``sigma`` about 0 -- at rest, a point, where
    ``sigma`` is 0."""

    def __init__(self, sigma: float):
        width = _finite_non_negative(sigma)
        if width is None:
            raise ValueError(f"a bell curve's spread is 0 or more; got "
                             f"{sigma!r}")
        self.sigma = width

    def pdf(self, x):
        out = _gaussian_density(x, self.sigma)
        return float(out) if np.ndim(out) == 0 else out

    def cdf(self, x):
        out = _gaussian_distribution(x, self.sigma)
        return float(out) if np.ndim(out) == 0 else out

    def moment(self, k: int) -> float:
        return _gaussian_moment(self.sigma, _whole(k, "a moment's order", 0))

    def quantile(self, p):
        return self.sigma * _standard_normal_quantile(p)

    def draw(self, z):
        return self.sigma * np.asarray(z, dtype=float)

    def gauss_rule(self, nodes: int):
        x, w = _gauss_hermite(_whole(nodes, "nodes", 1))
        return self.sigma * x, w


class QuantumAdded(Distribution):
    """One quantum added to a thermal mode of mean ``nbar`` (science § 2d.4,
    § 2e.1): the thermal bell curve, of variance ``s = 2 nbar + 1``, times a
    quadratic -- the mix of that bell curve, weight ``nbar / s``, and the
    one-quantum shape stretched to it, weight ``(nbar + 1) / s``."""

    def __init__(self, nbar: float):
        self.nbar = _checked_quanta(nbar)
        self.var = 2.0 * self.nbar + 1.0
        self.w_bell = self.nbar / self.var
        self.w_quantum = (self.nbar + 1.0) / self.var

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        bell = _gaussian_density(x, math.sqrt(self.var))
        out = (self.w_bell + self.w_quantum * x * x / self.var) * bell
        return float(out) if np.ndim(out) == 0 else out

    def cdf(self, x):
        y = np.asarray(x, dtype=float) / math.sqrt(self.var)
        phi = np.exp(-0.5 * y * y) / math.sqrt(2.0 * math.pi)
        out = _standard_normal(y) - self.w_quantum * y * phi
        return float(out) if np.ndim(out) == 0 else out

    def moment(self, k: int) -> float:
        k = _whole(k, "a moment's order", 0)
        if k % 2:
            return 0.0
        double = math.prod(range(k - 1, 0, -2))
        return self.var ** (k // 2) * (self.w_bell * double
                                       + self.w_quantum * double * (k + 1))

    def gauss_rule(self, nodes: int):
        m = _whole(nodes, "nodes", 1)
        root = math.sqrt(self.var)
        xb, wb = _gauss_hermite(m + 1)
        xq, wq = _number_state_rule(1, m + 1)
        points = np.concatenate((root * xb, root * _SQRT2 * xq))
        weights = np.concatenate((self.w_bell * wb, self.w_quantum * wq))
        return _gauss_rule_of(points, weights, m)


class Level(Distribution):
    """One level ``n`` of the oscillator, exactly ``n`` quanta: its density
    ``psi_n(x)^2`` (`oscillator.level_density`), its averages exactly
    (`oscillator.level_moment`), its distribution function from the level's
    one table (`oscillator._number_distribution`, in ``xi = x / sqrt 2``)."""

    def __init__(self, n: int):
        self.n = _whole(n, "a level", 0)

    @functools.cached_property
    def _table(self) -> _Table:
        """The level's table, built on first use -- a draw or a quantile asks
        it; the density, averages and Gauss rule never do."""
        return _number_distribution(self.n)

    def pdf(self, x):
        out = level_density(self.n, x)
        return float(out) if np.ndim(out) == 0 else out

    def cdf(self, x):
        out = np.clip(self._table.below(np.asarray(x, dtype=float) / _SQRT2),
                      0.0, 1.0)
        return float(out) if np.ndim(out) == 0 else out

    def moment(self, k: int) -> float:
        return level_moment(self.n, k)

    def quantile(self, p):
        return _SQRT2 * self._table.lower_quantile(p)

    def gauss_rule(self, nodes: int):
        x, w = _number_state_rule(self.n, _whole(nodes, "nodes", 1))
        return _SQRT2 * x, w


def _levels_density(populations, xi) -> np.ndarray:
    """``sum_n p_n psi_n(xi)^2``, every level from one sweep of the
    recurrence (`oscillator._level_sweep`), each term taken in logarithms so
    no level overflows or underflows (science § 2e.3)."""
    xi = np.asarray(xi, dtype=float)
    p = np.asarray(populations, dtype=float)
    out = np.zeros_like(xi)
    with np.errstate(divide="ignore"):
        for k, h, log_scale in _level_sweep(p.size, xi):
            if p[k]:
                out += p[k] * np.exp(2.0 * (np.log(np.abs(h)) + log_scale)
                                     - xi * xi)
    return out


class LevelList(Distribution):
    """A LIST OVER THE LEVELS -- ``populations[n]`` the chance of ``n``
    quanta -- and its coordinate's distribution, ``sum_n p_n psi_n(x)^2``
    (science § 2e.1, § 2e.3): its density by one sweep over the levels; its
    distribution function by one table of the whole list; its averages
    level by level; its Gauss rule from each level's own, pooled.  Even:
    every level's density is."""

    def __init__(self, populations):
        p = np.asarray(populations, dtype=float).reshape(-1)
        if (p.size == 0 or not np.all(np.isfinite(p))
                or np.any(p < -_EPSILON * np.max(np.abs(p)))):
            raise ValueError("a list's chances are numbers, 0 or more, one "
                             "per level")
        p = np.where(p > 0.0, p, 0.0)        # within the one epsilon of 0
        if not p.sum() > 0.0:
            raise ValueError("a list holds a chance above 0 on one level at "
                             "least; every chance given is 0")
        self.populations = p / p.sum()

    @functools.cached_property
    def _table(self) -> _Table:
        """Its distribution function, tabulated on first use -- a draw or a
        quantile asks it; the density, averages and Gauss rule never do."""
        top = int(np.flatnonzero(self.populations)[-1])
        return _Table(lambda xi: _levels_density(self.populations, xi),
                      *_table_shape(top))

    def pdf(self, x):
        xi = np.asarray(x, dtype=float) / _SQRT2
        out = _levels_density(self.populations, xi) / _SQRT2
        return float(out) if np.ndim(out) == 0 else out

    def cdf(self, x):
        out = np.clip(self._table.below(np.asarray(x, dtype=float) / _SQRT2),
                      0.0, 1.0)
        return float(out) if np.ndim(out) == 0 else out

    def moment(self, k: int) -> float:
        k = _whole(k, "a moment's order", 0)
        return float(sum(pn * level_moment(n, k)
                         for n, pn in enumerate(self.populations) if pn))

    def quantile(self, p):
        return _SQRT2 * self._table.lower_quantile(p)

    def gauss_rule(self, nodes: int):
        m = _whole(nodes, "nodes", 1)
        points, weights = [], []
        for n, pn in enumerate(self.populations):
            if pn:
                x, w = _number_state_rule(n, m + 1)
                points.append(_SQRT2 * x)
                weights.append(pn * w)
        return _gauss_rule_of(np.concatenate(points), np.concatenate(weights),
                              m)


def of_state(state) -> Distribution:
    """A STATE'S DISTRIBUTION, in ``x = Q / Q_zp`` -- by the route its
    reduced chain takes (`engines/vibration-frames.md` § 9.1's diagram), the
    family read once (`oscillator_state._family`): the ground state or a
    thermal one, the bell curve; a thermal state with one quantum added, the
    bell curve times a quadratic; a level, its level; the classical bell
    curve (a point at 0 K); any other quantum state its list
    (:class:`LevelList`).  ``state`` is a chain placed on an oscillator --
    ``chain.on(Oscillator(nu))``; a chain alone is refused by name, its
    numbers not yet known."""
    if not (hasattr(state, "chain") and hasattr(state, "oscillator")):
        raise ValueError(
            f"of_state takes a state -- a chain placed on an oscillator, "
            f"chain.on(Oscillator(frequency_cm1)), which gives it its "
            f"numbers; got {state!r}")
    family = state._family
    start = state.chain.start
    if state.is_classical:
        return Gaussian(state.oscillator.classical_spread(
            start.get("temperature_k")))
    if family == "bell":
        return Gaussian(spread_of_quanta(1.0, start.get("mean_quanta", 0.0)))
    if family == "quantum added":
        return QuantumAdded(start.get("mean_quanta"))
    if family == "level":
        return Level(start.get("quanta"))
    return LevelList(state.populations)


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.oscillator",)

__all__ = ["Distribution", "Gaussian", "QuantumAdded", "Level",
           "LevelList", "of_state",
           "PointWeights", "points_weights"]

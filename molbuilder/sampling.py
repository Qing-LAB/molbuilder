"""THE SEED'S KEYED RANDOM STREAMS, AND THE STATISTICS OF ARRAYS
(`engines/vibration-frames.md` § 3.3, § 7, § 9.1 module 4;
`science/vibrational-averaging.md` § 2d.6, § 5.5).

MODULE  sampling (L2; numpy and math, and `oscillator`'s whole-number rule
        -- nothing else of molbuilder)
ROLE    random sampling, nothing vibrational:

        * the seed's streams of standard normal numbers, one per key --
          ``numpy.random.default_rng(key)`` seeded through ``SeedSequence``
          -- drawn in one block from the first, so draws ``start ... start +
          size - 1`` are the same however many are asked
          (:func:`normal_stream`), and the seed's check
          (:func:`_checked_seed`);
        * the statistics of independent units at the weights given -- a
          mean and its error (:func:`mean_with_error`) -- and of mirrored
          pairs, their means and weights and their even and odd parts
          (:func:`pair_means`, :func:`even_odd_parts`): the weights a set
          stores, never counted out here (the user, 2026-11-26).
USED-BY `vibration` (a mode's draws and a motion's snapshots, their
        analyses), and a person's script (`postprocess` hands the module on).
TRAVELS with `vibration` in both vibration bundles: numpy, math and
        `oscillator`, imported two ways.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple, Tuple

import numpy as np

# TWO WAYS, because this module travels beside the job (see the header).
try:                                        # inside molbuilder
    from .oscillator import _whole
except ImportError:                         # beside a job, in a bundle
    from oscillator import _whole  # type: ignore[no-redef]


# --------------------------------------------------------------------- #
#  The seed's streams                                                    #
# --------------------------------------------------------------------- #

def _key(key: Any) -> list:
    """A stream's key, a sequence of whole numbers 0 or more -- ``(seed, mode,
    0)`` a mode's normal numbers -- refused by name otherwise."""
    try:
        entries = list(key)
    except TypeError:
        entries = None
    if not entries:
        raise ValueError(f"a stream's key is a sequence of whole numbers -- "
                         f"(seed, mode, 0) a mode's normal numbers; got "
                         f"{key!r}")
    return [_whole(k, "a stream key's entry", 0) for k in entries]


def normal_stream(key: Tuple[int, ...], size: int, start: int = 0) -> np.ndarray:
    """Standard normal numbers ``start ... start + size - 1`` of the stream
    ``key`` -- ``(seed, mode, 0)`` a mode's (`engines/vibration-frames.md`
    § 3.3): ``numpy.random.default_rng(key)``, drawn in one block from the
    first, so draws ``start ... start + size - 1`` are the same however many
    were asked."""
    size, start = _whole(size, "size", 0), _whole(start, "start", 0)
    return np.random.default_rng(_key(key)).standard_normal(start
                                                            + size)[start:]


def _checked_seed(seed: Any) -> int:
    """A seed, a whole number 0 or more -- required, refused by name: the
    same seed gives the same draws."""
    if seed is None:
        raise ValueError("a draw needs its seed, a whole number 0 or more: "
                         "the same seed gives the same draws")
    return _whole(seed, "seed", 0)


# --------------------------------------------------------------------- #
#  The statistics of arrays (science § 5.5)                              #
# --------------------------------------------------------------------- #

#: HOW FAR THE WEIGHTS' SUM MAY BE FROM 1 -- the container's
#: (`frameset.WEIGHT_SUM_TOLERANCE`, `model/structure.md` § 2.2f), restated:
#: this module travels in a job's bundle without the structure, and a test
#: holds the two one.
WEIGHT_SUM_TOLERANCE = 1e-6


class PairMeans(NamedTuple):
    """A paired set's independent units: each pair's mean at its two values'
    weights, and its weight, their sum -- ``mean_with_error(*pair_means(
    values, weights))`` the average and its error."""
    means: np.ndarray
    weights: np.ndarray


def _checked_weights(weights, count: int) -> np.ndarray:
    """``weights`` as the statistics take them: one finite number per value,
    0 or more, summing to 1 within :data:`WEIGHT_SUM_TOLERANCE` -- each
    value's share of the mean, a set's stored weights, never counted out
    here (the user, 2026-11-26).  Refused by name otherwise."""
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError):
        w = None
    if w is None or w.ndim != 1 or w.shape[0] != count:
        raise ValueError(
            f"the weights are one per value, {count} here -- each value's "
            f"share of the mean; got "
            + (f"a {type(weights).__name__}" if w is None
               else _count_words(w)))
    if not np.isfinite(w).all():
        raise ValueError("the weights are finite numbers -- each value's "
                         "share of the mean")
    if (w < 0).any():
        raise ValueError(f"a weight is 0 or more -- each value's share of "
                         f"the mean; got {float(w[w < 0][0])!r}")
    total = math.fsum(w)
    if count and abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"the weights sum to {total!r}, not 1 within "
            f"{WEIGHT_SUM_TOLERANCE:g} -- each value's share of the mean, "
            f"summing to 1")
    return w


def _along_first(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``w`` shaped to multiply ``x`` along its first axis, its trailing
    axes kept."""
    return w.reshape(w.shape + (1,) * (x.ndim - w.ndim))


def _spread_at(units: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The spread of independent ``units`` (axis 0) at their weights
    (science § 5.5): ``sqrt(sum_i w_i (u_i - u)^2 / (1 - sum_i w_i^2))``,
    ``u = sum_i w_i u_i`` -- the sample spread, ``ddof`` 1, at equal
    weights; NaN with fewer than two of weight above 0, which leave nothing
    to take a spread from.  A unit of weight 0 adds nothing, its value
    whatever it is (science § 5.5)."""
    if int((w > 0).sum()) < 2:
        return np.full(units.shape[1:], np.nan)
    units, w = units[w > 0], w[w > 0]
    mean = (_along_first(w, units) * units).sum(axis=0)
    square = (_along_first(w, units) * (units - mean) ** 2).sum(axis=0)
    return np.sqrt(square / (1.0 - float(np.sum(w ** 2))))


def _as_units(values) -> np.ndarray:
    """``values`` along a first axis -- a single number refused by name."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 0:
        raise ValueError(f"the values run along a first axis, one per unit; "
                         f"got {_count_words(v)}")
    return v


def mean_with_error(units, weights) -> Tuple[np.ndarray, np.ndarray]:
    """The mean over independent ``units`` (axis 0, any trailing axes kept)
    at their ``weights`` -- one per unit, summing to 1 (:func:`_checked_
    weights`) -- and its error (science § 5.5): ``sqrt(sum_i w_i^2)`` times
    their spread at the weights (:func:`_spread_at`), their spread over
    ``sqrt(N)`` at equal weights; a unit of weight 0 adding nothing, its
    value whatever it is; the mean NaN with no unit, the error with fewer
    than two of weight above 0."""
    units = _as_units(units)
    w = _checked_weights(weights, units.shape[0])
    if not units.shape[0]:
        nan = np.full(units.shape[1:], np.nan)
        return nan, nan
    kept = w > 0
    value = (_along_first(w[kept], units[kept]) * units[kept]).sum(axis=0)
    return value, _spread_at(units, w) * math.sqrt(float(np.sum(w ** 2)))


def _count_words(v: np.ndarray) -> str:
    """How many values ``v`` holds, in words: *a single number* for none
    along an axis, else *1 value*, *3 values*."""
    if v.ndim == 0:
        return "a single number"
    return f"{v.shape[0]} value{'' if v.shape[0] == 1 else 's'}"


def _in_pairs(values) -> np.ndarray:
    """THE PAIRS, GROUPED HERE: ``values`` one per snapshot, in the order the
    snapshots were drawn -- each snapshot followed by its mirror, so values
    1 and 2 are the first pair, 3 and 4 the second, and so on (frame 0
    left out) -- any trailing axes kept (a value per energy, say), grouped
    two by two along the first axis into ``(pairs, 2, ...)``.  An odd count
    is refused by name: a value without its mirror, often a set's frame 0
    left in."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 0 or v.shape[0] % 2:
        raise ValueError(
            f"mirrored pairs come two by two -- each snapshot followed by its "
            f"mirror, frame 0 left out; got {_count_words(v)}")
    return v.reshape((v.shape[0] // 2, 2) + v.shape[1:])


def pair_means(values, weights) -> PairMeans:
    """Each mirrored pair's mean and weight, the independent units of a
    paired set: ``values`` one per snapshot in the order drawn, grouped two
    by two here (:func:`_in_pairs`), ``weights`` one per value as
    :func:`mean_with_error` takes them -- each pair's weight its two
    values', its mean at them, ``(v1 + v2)/2`` at equal weights, any
    trailing axes kept."""
    pairs = _in_pairs(values)
    w = _checked_weights(weights, 2 * pairs.shape[0]).reshape(-1, 2)
    total = w.sum(axis=1)
    # A PAIR OF WEIGHT 0 adds nothing (:func:`mean_with_error`); its mean is
    # its two values' plain one, never a division by its weight.
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = ((_along_first(w, pairs) * pairs).sum(axis=1)
                    / _along_first(total, pairs[:, 0]))
    means = np.where(_along_first(total, pairs[:, 0]) > 0, weighted,
                     pairs.mean(axis=1))
    return PairMeans(means, total)


def even_odd_parts(values, weights) -> Tuple[np.ndarray, np.ndarray]:
    """A paired set's spread split in two (science § 5.5): the spread of
    the pair means -- the response's even part -- and of the pair
    half-differences ``(v1 - v2)/2, (v3 - v4)/2, ...`` -- its odd part, which
    the pairs cancel; ``values`` and ``weights`` as :func:`pair_means` takes
    them, each spread at the pairs' weights (:func:`_spread_at`), NaN with
    fewer than two pairs."""
    pairs = _in_pairs(values)
    units = pair_means(values, weights)
    halves = (pairs[:, 0] - pairs[:, 1]) / 2.0
    return (_spread_at(units.means, units.weights),
            _spread_at(halves, units.weights))


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.oscillator",)

__all__ = ["WEIGHT_SUM_TOLERANCE", "PairMeans", "normal_stream",
           "mean_with_error", "pair_means", "even_odd_parts"]

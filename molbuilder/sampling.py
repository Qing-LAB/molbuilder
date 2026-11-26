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
        * the statistics of independent units -- a mean and its error
          (:func:`mean_with_error`) -- and of mirrored pairs, their means
          and their even and odd parts (:func:`pair_means`,
          :func:`even_odd_parts`).
USED-BY `vibration` (a mode's draws and a motion's snapshots, their
        analyses), and a person's script (`postprocess` hands the module on).
TRAVELS with `vibration` in both vibration bundles: numpy, math and
        `oscillator`, imported two ways.
"""
from __future__ import annotations

import math
from typing import Any, Tuple

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

def _spread_of(units: np.ndarray) -> np.ndarray:
    """The spread of independent ``units`` over axis 0, ``ddof`` 1 -- NaN
    with fewer than two, which leave nothing to take a spread from."""
    if units.shape[0] < 2:
        return np.full(units.shape[1:], np.nan)
    return units.std(axis=0, ddof=1)


def mean_with_error(units) -> Tuple[np.ndarray, np.ndarray]:
    """The mean over independent ``units`` (axis 0, any trailing axes kept)
    and its error, their spread over ``sqrt(N)`` (:func:`_spread_of`) --
    the mean NaN with no unit, the error with fewer than two."""
    units = np.asarray(units, dtype=float)
    count = units.shape[0]
    value = (units.mean(axis=0) if count
             else np.full(units.shape[1:], np.nan))
    return value, _spread_of(units) / math.sqrt(max(count, 1))


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
            f"mirror, frame 0 left out; got "
            f"{v.shape[0] if v.ndim else 'one'} value"
            f"{'' if v.ndim and v.shape[0] == 1 else 's'}")
    return v.reshape((v.shape[0] // 2, 2) + v.shape[1:])


def pair_means(values) -> np.ndarray:
    """Each mirrored pair's mean, the independent units of a paired set:
    ``values`` one per snapshot in the order drawn, grouped two by two here
    (:func:`_in_pairs`) -- ``(v1 + v2)/2, (v3 + v4)/2, ...``, any trailing
    axes kept."""
    return _in_pairs(values).mean(axis=1)


def even_odd_parts(values) -> Tuple[np.ndarray, np.ndarray]:
    """A paired set's spread split in two (science § 5.5): the spread of
    the pair means -- the response's even part -- and of the pair
    half-differences ``(v1 - v2)/2, (v3 - v4)/2, ...`` -- its odd part, which
    the pairs cancel; ``values`` one per snapshot in the order drawn,
    grouped two by two here (:func:`_in_pairs`); each a spread of
    independent units (:func:`_spread_of`), NaN with fewer than two pairs."""
    pairs = _in_pairs(values)
    halves = (pairs[:, 0] - pairs[:, 1]) / 2.0
    return _spread_of(pairs.mean(axis=1)), _spread_of(halves)


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.oscillator",)

__all__ = ["normal_stream",
           "mean_with_error", "pair_means", "even_odd_parts"]

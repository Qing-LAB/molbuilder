"""A HARMONIC MODE'S NUMBERS, AND THE MOLECULE'S MOTION -- every per-mode
quantity, on one oscillator's mechanics (`oscillator`), and the motion built
from them (`engines/vibration.md` § 6.11; `engines/vibration-frames.md`
§ 1-§ 4, § 7; `science/vibrational-averaging.md` § 2, § 2a-§ 2c, § 5.4,
§ 5.5).

MODULE  vibration (L2; numpy, the constants, the generic tools --
        `oscillator`, `sampling`, `distribution` -- and the frequency-window
        rule `spectra.selection`, nothing else of molbuilder)
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
          one table (:data:`STATES`, `engines/vibration-frames.md` § 2) --
          its density, distribution, moments, spread and mean quanta, its
          draws from the mode's own stream of a seed, the frames of its own
          Gauss rule (§ 2b.4) and the weights for frames placed elsewhere
          (§ 5.4);
        * :class:`Mode` and :class:`Vibration`, the view of a result a person
          computes with (`postprocess.vibration`, § 6.11 § 6) -- the shared
          fields the vibration's -- and the molecule's motion: the modes that
          move, a list or chosen in steps (:class:`ModeSelection`), a state
          per mode, each drawn on its own (:class:`Motion`), its snapshots
          and their analyses (:class:`Snapshots`).
USED-BY the result's derived values (`spectra.derived`), the thermochemistry
        (`spectra.normal_modes`), the motion toolkit's frame sets
        (`vibration_frames`: the writer and reader, the doors' set, the
        outermost move, the weights for a set's rows), the SIESTA
        vibration's Raman displacements (`spectra.siesta_vibration`), the
        spectrum tab's load door (`web.blueprints.spectra`), a result's
        modes (`spectra.results.SpectraResults.vibration`) and the script
        door (`postprocess.vibration`)
TRAVELS in both bundles -- the PySCF script's and a SIESTA vibration's
        (`runwrap.members`) -- so it imports the constants, the generic
        tools and the window rule two ways (:data:`TRAVELS_WITH`; the rule
        beside a job as ``mode_selection``), and nothing else of
        molbuilder.  The masses and element symbols it takes
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
# The mechanics and a level's numbers are one oscillator's (`oscillator`),
# the seed's streams and the arrays' statistics `sampling`'s, the
# distributions' numerics `distribution`'s.  The mechanics and the quanta
# words are handed on here under their names, read here by the record's
# writers (`spectra.derived`, `spectra.normal_modes`), the frame-set doors
# (`vibration_frames`, `cli`) and the spectrum tab's door
# (`web.blueprints.spectra`).
try:                                        # inside molbuilder
    from .constants import FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2
    from .oscillator import (
        _checked_frequency, _checked_quanta, _eigenfunction, _number_distribution,
        _position_operator, _quanta_words, _whole, checked_temperature_k, classical_spread_amu12_ang,
        bose_einstein_mean_quanta, spread_of_quanta, bose_einstein_temperature,
        thermal_spread_amu12_ang, zero_point_amplitude_amu12_ang)
    from .sampling import (_checked_seed, even_odd_parts, mean_with_error,
                           normal_stream, pair_means)
    from .distribution import (
        _erfc, _gauss_hermite, _gaussian_density, _gaussian_distribution,
        _gaussian_moment, _number_state_rule, points_weights)
except ImportError:                         # beside a job, in a bundle
    from constants import (  # type: ignore[no-redef]
        FORCE_CONSTANT_MDYN_ANG_PER_AMU_CM2)
    from oscillator import (  # type: ignore[no-redef]
        _checked_frequency, _checked_quanta, _eigenfunction, _number_distribution,
        _position_operator, _quanta_words, _whole, checked_temperature_k, classical_spread_amu12_ang,
        bose_einstein_mean_quanta, spread_of_quanta, bose_einstein_temperature,
        thermal_spread_amu12_ang, zero_point_amplitude_amu12_ang)
    from sampling import (  # type: ignore[no-redef]
        _checked_seed, even_odd_parts, mean_with_error, normal_stream,
        pair_means)
    from distribution import (  # type: ignore[no-redef]
        _erfc, _gauss_hermite, _gaussian_density, _gaussian_distribution,
        _gaussian_moment, _number_state_rule, points_weights)



# --------------------------------------------------------------------- #
#  A mode's thermal row: the oscillator's mechanics at its temperatures  #
# --------------------------------------------------------------------- #

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
    quanta = [bose_einstein_mean_quanta(frequency_cm1, t)
              for t in temperatures_k]
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


def outermost_move_ang(state: "State", mode_rows,
                       nodes: Optional[int] = None) -> float:
    """THE OUTERMOST MOVE of a mode in ``state`` for its ``nodes``-point
    rule, in Angstrom (`engines/vibration-frames.md` § 4, § 5.1's
    ``max_displacement_ang``): the largest single atom's move at the
    outermost frames, ``|q_out| max_A |L_A|`` -- ``q_out`` the outermost
    node of the state's own rule (:meth:`State.frames`, its default rule
    where ``nodes`` is unsaid), ``mode_rows`` the mode over its atoms.  The
    one rule every door asks: a set's row, the page's readout,
    ``postprocess.outermost_move``."""
    rule = state.frames() if nodes is None else state.frames(nodes)
    q = np.abs(rule.q_amu12_ang)
    return float(q.max()) * largest_move(mode_rows)


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
THERMAL, LEVEL, CLASSICAL = ("thermal", "level", "classical")

#: A STATE'S OWN RULE'S NODES WHEN UNSAID -- three: the bell curve's
#: three frames at 0 and +-sqrt(3) sigma, exact to degree 5, five and seven
#: the check on them (`science/vibrational-averaging.md` § 5.1, § 5.3); the
#: doors offer it first (`vibration_frames.OFFERED_NODES`).
DEFAULT_NODES = 3


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
    ``R + Q L`` (frames x atoms x 3; ``None`` without a mode);
    ``read_said`` what reading a written set back said of it
    (`vibration_frames.read_frame_set`), first in :attr:`said`, and
    ``read_mode`` the mode a set read back without its vibration states it
    is of."""
    q_amu12_ang: np.ndarray
    weights: np.ndarray
    exact_to_degree: int
    state: Optional["ModeState"] = field(default=None, repr=False)
    positions_ang: Optional[np.ndarray] = field(default=None, repr=False)
    read_said: Tuple[str, ...] = field(default=(), repr=False)
    read_mode: Optional[int] = field(default=None, repr=False)

    @property
    def mode_index_1based(self) -> np.ndarray:
        """The column's mode, ``(1,)`` -- the mode a set read back without
        its vibration states, 0 for a bare state's frames."""
        mode = None if self.state is None else self.state.mode
        if mode is None:
            return np.array([0 if self.read_mode is None
                             else int(self.read_mode)])
        return np.array([int(mode.index_1based)])

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
        """Each frame's largest single-atom move, Å (frames,) -- from the
        geometries where the frames have no mode (a set read back without
        its vibration); ``None`` with neither."""
        return None if self._largest is None else self._largest[0]

    @property
    def max_atom_mode_index_1based(self) -> Optional[np.ndarray]:
        """The mode moving each frame's most-moved atom -- this set's one
        mode, 0 for a frame that moves nothing (frame 0) (frames,)."""
        return None if self._largest is None else self._largest[1]

    @property
    def _largest(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.state is not None:
            return _largest_moves(self.state.mode._v, self.mode_index_1based,
                                  self.q_amu12_ang)
        return _largest_of_geometries(self.positions_ang,
                                      self.mode_index_1based)

    @property
    def said(self) -> Tuple[str, ...]:
        """What the frames say (`engines/vibration-frames.md` § 4): the
        largest single-atom move of the outermost frames -- the check a
        person reads against a bond length."""
        largest = self.max_atom_displacement_ang
        if largest is None or not len(largest):
            return tuple(self.read_said)
        i = int(np.argmax(largest))
        return (*self.read_said,
                f"the outermost frames move an atom by at most "
                f"{largest[i]:.4f} Å ({_frame_words(i, len(largest))}, q = "
                f"{self.q_amu12_ang[i, 0]:+.4f} amu^1/2·Å)")


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


def _largest_of_geometries(positions_ang, numbers
                           ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Each frame's largest single-atom move from frame 0, from the
    geometries alone -- a set read back without its vibration -- and the
    mode moving it: the set's one mode, or 0 where the modes are not known
    or the frame moves nothing; ``None`` with no geometries."""
    if positions_ang is None:
        return None
    p = np.asarray(positions_ang, dtype=float)
    largest = np.linalg.norm(p - p[:1], axis=-1).max(axis=1)
    one = int(numbers[0]) if len(numbers) == 1 else 0
    return largest, np.where(largest > 0.0, one, 0)


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


@dataclass(frozen=True)
class State:
    """Where a mode's coordinate ``Q`` (amu^1/2.Å) sits, in one state
    (§ 2b.3).  ``kind`` one of :data:`THERMAL`, :data:`LEVEL`,
    :data:`CLASSICAL`; ``zero_point_amplitude_amu12_ang`` the mode's
    ``Q_zp``; ``thermal_quanta`` a thermal state's mean quanta; ``quanta``
    a level's; ``classical_spread_amu12_ang`` the classical bell
    curve's."""
    kind: str
    zero_point_amplitude_amu12_ang: float
    thermal_quanta: float = 0.0
    quanta: int = 0
    classical_spread_amu12_ang: float = 0.0

    # ---- the numbers every state states ----------------------------------
    @property
    def _q_zp(self) -> float:
        return float(self.zero_point_amplitude_amu12_ang)

    @property
    def mean_quanta(self) -> Optional[float]:
        """The state's mean number of quanta: a level's own, a
        thermal state's; ``None`` for a classical state, which counts
        none."""
        if self.kind == LEVEL:
            return float(self.quanta)
        if self.kind == CLASSICAL:
            return None
        return float(self.thermal_quanta)

    @property
    def spread_amu12_ang(self) -> float:
        """The r.m.s. coordinate, ``sqrt(<Q^2>)``: in every quantum state the
        spread its mean quanta give, ``Q_zp sqrt(2 <n> + 1)``
        (:func:`spread_of_quanta`, § 2b.3); a classical state's its bell
        curve's."""
        quanta = self.mean_quanta
        if quanta is None:                          # a classical state
            return float(self.classical_spread_amu12_ang)
        return spread_of_quanta(self._q_zp, quanta)

    # ---- the density, the distribution, the moments ----------------------
    def pdf(self, q):
        """The probability density of ``Q`` at ``q`` (per amu^1/2.Å)."""
        q = np.asarray(q, dtype=float)
        if self.kind == LEVEL:
            xi = q / (math.sqrt(2.0) * self._q_zp)
            psi = _eigenfunction(self.quanta, xi)
            out = psi * psi / (math.sqrt(2.0) * self._q_zp)
        else:
            out = _gaussian_density(q, self.spread_amu12_ang)
        return float(out) if out.ndim == 0 else out

    def cdf(self, q):
        """The probability that ``Q`` is ``q`` or less -- arrays in, arrays
        out."""
        q = np.asarray(q, dtype=float)
        if self.kind == LEVEL:
            table = _number_distribution(self.quanta)
            out = np.clip(table.below(q / (math.sqrt(2.0) * self._q_zp)),
                          0.0, 1.0)
        else:
            out = _gaussian_distribution(q, self.spread_amu12_ang)
        return float(out) if np.ndim(out) == 0 else out

    def moment(self, k: int) -> float:
        """``<Q^k>`` exactly, for a whole ``k`` of 0 or more: the Gaussian's
        ``sigma^k (k-1)!!``; a level's ``<n|xi^k|n>`` by the
        oscillator's own position operator (§ 2b.3)."""
        k = _whole(k, "a moment's order", 0)
        if self.kind != LEVEL:
            return _gaussian_moment(self.spread_amu12_ang, k)
        if k % 2:
            return 0.0
        n = self.quanta
        lo, hi = max(0, n - k // 2), n + k // 2 + 1
        x_op = _position_operator(lo, hi)
        v = np.zeros(hi - lo)
        v[n - lo] = 1.0
        for _ in range(k // 2):
            v = x_op @ v
        return float(v @ v) * (math.sqrt(2.0) * self._q_zp) ** k

    # ---- the frames ------------------------------------------------------
    def _rule(self, m: int) -> Tuple[np.ndarray, np.ndarray]:
        """The ``m``-point Gauss rule of this state's own distribution
        (science § 2a.3, § 2b.4), nodes and weights as found."""
        if self.kind == LEVEL:
            x, w = _number_state_rule(self.quanta, m)
            return math.sqrt(2.0) * self._q_zp * x, w
        x, w = _gauss_hermite(m)
        return self.spread_amu12_ang * x, w

    def frames(self, nodes: int = DEFAULT_NODES) -> Frames:
        """The frames that average a response over the state (§ 2b.4,
        `engines/vibration-frames.md` § 4): the Gauss rule of the state's own
        distribution -- Gauss–Hermite for a bell curve, the rule whose
        weight function is the state's density for n quanta -- exact for a
        response polynomial to degree ``2 nodes - 1``, exactly symmetric, in
        the order ``(0, +q1, -q1, ...)``, an even rule's frame 0 prepended at
        weight 0."""
        m = _whole(nodes, "nodes", 1)
        x, w = self._rule(m)
        q, weights = _symmetric_ordered(x, w)
        return Frames(q[:, None], weights, 2 * m - 1)

    # ---- the draws -------------------------------------------------------
    def draw(self, size: int, *, seed: int, start: int = 0,
             mode: int = 0) -> np.ndarray:
        """``size`` coordinates drawn from ``P(Q)`` -- draws ``start`` ...
        ``start + size - 1`` of the seed's sequence -- from the mode's own
        stream (`engines/vibration-frames.md` § 2, § 3.3): one standard
        normal number ``z`` per draw -- a bell curve, its spread times ``z``;
        n quanta, the ``Q`` where the chance of ``Q`` or less is ``Phi(z)``,
        taken from the left with ``z``'s sign, so ``-z`` gives ``-Q``
        exactly.  ``mode`` the mode's number the stream is keyed by (0 for a
        bare state)."""
        size = _whole(size, "size", 0)
        seed = _checked_seed(seed)
        start = _whole(start, "start", 0)
        return self._at(normal_stream((seed, int(mode), 0), size, start))

    def _at(self, z) -> np.ndarray:
        """The coordinate each normal number ``z`` gives."""
        z = np.asarray(z, dtype=float)
        if self.kind != LEVEL:
            return self.spread_amu12_ang * z
        table = _number_distribution(self.quanta)
        left = table.lower_quantile(0.5 * np.asarray(
            _erfc(np.abs(z) / math.sqrt(2.0)), dtype=float))
        xi = np.where(z == 0.0, 0.0, -np.sign(z) * left)
        return math.sqrt(2.0) * self._q_zp * xi

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
        # THE ARITHMETIC IS THE DISTRIBUTION'S (`distribution.points_weights`):
        # the solve, the degree reproduced, whether the frames are the
        # state's own rule, the share beyond them.  THE STATE'S OWN RULE --
        # its m frames' weights reproduce its averages as far as a Gauss rule
        # on m frames can -- is no re-weighting to another motion: science
        # § 5.4's items 1 and 2 (the share beyond, the blindness) are not
        # said of it; a weight below zero, item 3, is said of any frames.
        # Counted on every frame: frames placed for another motion whose
        # weights happen to fall on a shorter rule of this state (a node at
        # weight 0) are still said of.
        found = points_weights(q, self.moment, self.cdf, self.spread_amu12_ang,
                               written)
        w, degree, own, beyond = (found.weights, found.exact_to_degree,
                                  found.own, found.beyond)
        said: List[str] = []
        if not own and beyond > 0.0:
            said.append(f"{100.0 * beyond:.3g} % of the state's motion lies "
                        f"beyond the outermost frames (q from {q.min():+.4g} "
                        f"to {q.max():+.4g} amu^1/2·Å), averaged by values "
                        f"the frames never computed")
        if not own and m == 3:
            said.append("three frames carry only the curvature: up to it "
                        "every state of one spread agrees, so the "
                        "re-averaged number is blind to the state's shape")
        elif not own and m < 3:
            said.append(f"{'one frame is' if m == 1 else 'two frames are'} "
                        f"exact only to degree {degree}")
        for i in found.below_zero:
            said.append(f"{_frame_words(i, m)}'s weight is {w[i]:.4g}, below "
                        f"zero: the frames do not suit the state's shape "
                        f"well")
        return Weights(w, beyond, degree, tuple(said))


#: WHAT NAMES EACH STATE (§ 2b.3, `engines/vibration-frames.md` § 2): the
#: arguments a person gives, exactly one of these sets -- a thermal state
#: named by its temperature or by its mean quanta.
STATES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (LEVEL, ("quanta",)),
    (THERMAL, ("temperature_k",)),
    (THERMAL, ("mean_quanta",)),
    (CLASSICAL, ("classical", "temperature_k")),
)
_STATE_NAMED_BY: Dict[frozenset, str] = {
    frozenset(names): kind for kind, names in STATES}


def checked_state(*, temperature_k=None, mean_quanta=None, quanta=None,
                  classical: bool = False) -> str:
    """THE STATE A SET OF ARGUMENTS NAMES, checked before any mode is known
    (§ 2b.3; `engines/vibration-frames.md` § 2): its kind, by the one table
    (:data:`STATES` -- any other set refused, naming them), and each value
    by its own rule -- a temperature a number of kelvin, 0 or more; ``quanta``
    a whole number, 0 or more; mean quanta 0 or more.  What :func:`state`
    asks first, and the doors before they read a run."""
    named = _named(temperature_k, mean_quanta, quanta, classical)
    kind = _STATE_NAMED_BY.get(named)
    if kind is None:
        ways = "; ".join(" with ".join(names) for _, names in STATES)
        raise ValueError(f"a state is named one of these ways: {ways} -- "
                         f"got {', '.join(sorted(named)) or 'nothing'}")
    if temperature_k is not None:
        checked_temperature_k(temperature_k)
    if quanta is not None:
        _whole(quanta, "a number of quanta", 0)
    if mean_quanta is not None:
        _checked_quanta(mean_quanta)
    return kind


def _frame_words(frame: int, n_frames: int) -> str:
    """A frame as a message says it, from 1: ``frame 2 of 3`` -- the
    spelling of `structure.frame_words`, the one spelling of a frame in a
    message (`model/structure.md` § 2.2e), kept here because this module
    travels beside a job, where `structure` does not."""
    return f"frame {int(frame) + 1} of {int(n_frames)}"


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
    name; otherwise none -- n quanta, mean quanta."""
    if temperature_k is not None:
        return temperature_k, GIVEN
    if (named not in _STATE_NAMED_BY
            and named | {"temperature_k"} in _STATE_NAMED_BY):
        if default is None:
            raise ValueError(f"name a temperature: {_HOLDERS[source]} "
                             f"states none")
        return default, source
    return None, None


def _named(temperature_k, mean_quanta, quanta, classical) -> frozenset:
    """The arguments a call names: each one given, ``classical`` when
    true."""
    given = {"temperature_k": temperature_k, "mean_quanta": mean_quanta,
             "quanta": quanta, "classical": True if classical else None}
    return frozenset(name for name, value in given.items()
                     if value is not None)


def state(zero_point_amplitude: float, frequency_cm1: float, *,
          temperature_k: Optional[float] = None,
          mean_quanta: Optional[float] = None,
          quanta: Optional[int] = None,
          classical: bool = False) -> State:
    """The state a person names (§ 2b.3), by one of the sets
    :data:`STATES` lists:

    * ``quanta`` -- n quanta exactly;
    * ``temperature_k`` or ``mean_quanta`` -- thermal (or random quanta);
    * ``classical`` with ``temperature_k`` -- the classical bell curve, at
      rest at 0 K.

    Any other set is refused, naming the sets."""
    q_zp = float(zero_point_amplitude)
    if not (math.isfinite(q_zp) and q_zp > 0.0):
        raise ValueError("a state needs the mode's zero-point amplitude -- "
                         "an imaginary mode has none")
    kind = checked_state(temperature_k=temperature_k,
                         mean_quanta=mean_quanta, quanta=quanta,
                         classical=classical)
    if kind == LEVEL:
        return State(LEVEL, q_zp, quanta=_whole(quanta, "quanta", 0))
    if kind == CLASSICAL:
        return State(CLASSICAL, q_zp, classical_spread_amu12_ang=(
            classical_spread_amu12_ang(frequency_cm1, temperature_k)))
    thermal = _checked_quanta(bose_einstein_mean_quanta(frequency_cm1,
                                                        temperature_k)
                              if temperature_k is not None else mean_quanta)
    return State(THERMAL, q_zp, thermal_quanta=thermal)


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
        # § 1) -- its quanta, thermal quanta, zero-point amplitude and
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
    def atom_rms_ang(self) -> np.ndarray:
        """Each atom's r.m.s. move in the state, Å: the spread times its
        row's length, ``sqrt(<Q^2>) |L_A|`` -- 0 for a held atom (§ 2b.2)."""
        return self.spread_amu12_ang * _row_lengths(
            self.mode.eigenvector_canonical_over_atoms_per_amu12)

    def draw(self, size: int, *, seed: int, start: int = 0) -> np.ndarray:
        """The state's draws (:meth:`State.draw`), from this mode's own
        stream of the seed -- keyed by its number."""
        return self.state.draw(size, seed=seed, start=start,
                               mode=self.mode.index_1based)

    def weights_at(self, q_amu12_ang, *, written=None) -> Weights:
        """The weights for this state of frames (:meth:`State.weights_at`)."""
        return self.state.weights_at(q_amu12_ang, written=written)

    def frames(self, nodes: int = DEFAULT_NODES) -> Frames:
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
              classical: bool = False) -> ModeState:
        """The mode in a state (§ 2b.3, :func:`state`).  Arguments that name
        a state once a temperature is added (:data:`_STATE_NAMED_BY`) -- a
        thermal or classical state given neither a temperature nor mean
        quanta -- take the temperature the record states, and say so (§ 6.11
        § 9 item 6); a record stating none is asked for one.  n quanta and
        mean quanta take none."""
        q_zp = self.zero_point_amplitude_amu12_ang
        if q_zp is None:
            raise ValueError(f"mode {self.index_1based} is imaginary: it has "
                             f"no harmonic state")
        named = _named(temperature_k, mean_quanta, quanta, classical)
        temperature_k, source = _temperature_of(
            named, temperature_k, self.record_temperature_k, THE_RECORDS)
        return ModeState(
            state(q_zp, self.frequency_cm1, temperature_k=temperature_k,
                  mean_quanta=mean_quanta, quanta=quanta,
                  classical=classical),
            self, temperature_k=temperature_k, temperature_from=source)

    def with_frequency(self, frequency_cm1: float) -> "Mode":
        """The mode at another frequency -- a scaled or a measured one -- its
        pattern kept (§ 2b.1): ``Q_zp``, its displacement and every spread
        follow from the new frequency; the result's own frequency kept as
        :attr:`result_frequency_cm1`."""
        nu = _checked_frequency(frequency_cm1, "a mode at another frequency")
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
    and where it came from -- ``read_path`` the path read, as given (the
    result file, or the run's folder);
    ``structure`` the structure the result states (its reader fills it);
    ``result_sha256`` the result JSON's (the read door's); ``run`` the run's
    folder and ``relaxation_calculation`` frame 0's record of the relaxation
    its geometry came from -- filled where a run's folder is read
    (`vibration_frames.vibration_of_run`), ``None`` else -- and
    ``relaxation_said``, why there is none, in words."""
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
    relaxation_said: str = field(default="", repr=False)

    def __post_init__(self):
        object.__setattr__(self, "masses_amu",
                           np.asarray(self.masses_amu, dtype=float))
        object.__setattr__(self, "positions_ang",
                           np.asarray(self.positions_ang, dtype=float))
        object.__setattr__(self, "modes", tuple(
            replace(m, vibration=self) for m in self.modes))

    def mode(self, number: int) -> Mode:
        """The mode numbered ``number``, as the result counts it
        (``index_1based``) -- THE ONE RULE of a mode's number, every door
        asking it: a whole number, never true/false and never truncated
        (2.5 is no mode 2), refused by name, as a number the result holds
        none of is."""
        if (isinstance(number, (bool, np.bool_))
                or not isinstance(number, (numbers.Real, np.integer))
                or float(number) != int(number)):
            raise ValueError(f"a mode is named by its number, a whole "
                             f"number -- not {number!r}")
        number = int(number)
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
        given = dict(coordinates)
        numbers = [self.mode(k).index_1based for k in given]
        if not numbers:
            return np.array(self.positions_ang, dtype=float, copy=True)
        q = [[float(v) for v in given.values()]]
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
        return self._at_total(f * self.frequency_scale)

    def _at_total(self, total: float) -> "Vibration":
        """The vibration at ``total`` times the result's frequencies -- THE
        ONE PLACE a vibration is set at a scale (:meth:`with_frequency_scale`
        and a frame set read back at its own scale ask it): at 1 the
        result's own modes, as read; else each mode at ``total`` times the
        result's frequency, the result's own kept."""
        total = float(total)
        if total == 1.0:
            modes = tuple(replace(m, frequency_cm1=m._result_frequency,
                                  result_frequency_cm1=None)
                          for m in self.modes)
        else:
            modes = tuple(replace(m, frequency_cm1=m._result_frequency * total,
                                  result_frequency_cm1=m._result_frequency)
                          for m in self.modes)
        return replace(self, modes=modes, frequency_scale=total)

    def mode_selection(self) -> "ModeSelection":
        """Which modes move, chosen in steps (:class:`ModeSelection`) --
        starting as every real mode."""
        return ModeSelection(self, tuple(m.index_1based for m in self.modes
                                         if not m.has_imag),
                             ("every real mode",))

    def motion(self, temperature_k: Optional[float] = None, *,
               modes: Any = None,
               classical: bool = False) -> "Motion":
        """The molecule's motion (`engines/vibration-frames.md` § 3.1): every
        real mode -- or the numbers in ``modes``, a list or a
        :class:`ModeSelection` (its steps said and written with the set) --
        thermal at
        ``temperature_k`` (the record's when none, said), quantum, or the
        classical bell curves with ``classical``; its modes in the result's
        order.  An imaginary mode among them is left out, said; none left to
        move is refused by name."""
        chosen_by: Tuple[str, ...] = ()
        if isinstance(modes, ModeSelection):
            if modes.vibration._content != self._content:
                raise ValueError("a mode selection is of the vibration it was "
                                 "made from: v.mode_selection() on this one")
            chosen_by, modes = modes.steps, modes.final()
        asked = ({m.index_1based for m in self.modes} if modes is None
                 else {self.mode(k).index_1based for k in modes})
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
        return Motion(self, states, float(temperature_k), source,
                      bool(classical), tuple(notes), (), tuple(imaginary),
                      chosen_by)


def _numbers_words(numbers: Sequence[Any]) -> str:
    """``9``, ``9 and 11``, ``9, 11 and 13``."""
    words = [str(n) for n in numbers]
    return (words[0] if len(words) == 1 else
            ", ".join(words[:-1]) + " and " + words[-1])


@dataclass(frozen=True, eq=False)
class ModeSelection:
    """WHICH MODES MOVE, chosen in steps (`engines/vibration-frames.md`
    § 3.1): it starts as every real mode of ``vibration``
    (:meth:`Vibration.mode_selection`), and each step acts on what the step
    before left, in the order written --

    * ``window(low, high)`` keeps the modes whose frequency, in cm-1, is
      inside ``[low, high]``, either end ``None`` for open -- the rule
      molbuilder's per-mode steps choose by (`spectra.selection`; an
      imaginary mode's frequency negative);
    * ``only(modes)`` keeps, of the modes the steps before left, only those
      listed;
    * ``add(modes)`` puts the modes listed in, ``drop(modes)`` takes them
      out;

    -- and :meth:`final` gives the list, an empty one refused by name.
    ``steps`` the steps in words, which the motion says and the set
    records.  A mode named by a number the result holds none of is refused
    by name (:meth:`Vibration.mode`); an imaginary mode added is left out of
    the motion, said."""
    vibration: "Vibration" = field(repr=False)
    modes: Tuple[int, ...]
    steps: Tuple[str, ...]

    def _numbers(self, modes) -> Tuple[int, ...]:
        """The modes listed, by number -- one number or a list of them; each
        refused by name where the result holds no such mode
        (:meth:`Vibration.mode`), a truth value too."""
        if isinstance(modes, (numbers.Real, np.integer, bool, np.bool_)):
            modes = (modes,)
        return tuple(self.vibration.mode(k).index_1based for k in modes)

    def _then(self, kept, words: str) -> "ModeSelection":
        kept = set(kept)
        return replace(self, modes=tuple(m.index_1based
                                         for m in self.vibration.modes
                                         if m.index_1based in kept),
                       steps=self.steps + (words,))

    def window(self, low_cm1: Optional[float] = None,
               high_cm1: Optional[float] = None) -> "ModeSelection":
        """The modes kept whose frequency is inside ``[low_cm1, high_cm1]``,
        either end ``None`` for open."""
        for end, name in ((low_cm1, "low"), (high_cm1, "high")):
            if end is not None and (isinstance(end, bool)
                                    or not isinstance(end, numbers.Real)
                                    or not math.isfinite(float(end))):
                raise ValueError(f"a window's {name} end is a number of cm-1, "
                                 f"or None for open; got {end!r}")
        if low_cm1 is not None and high_cm1 is not None and low_cm1 > high_cm1:
            raise ValueError(f"a window's low end is at its high end or "
                             f"below; got {low_cm1!r} and {high_cm1!r}")
        # imported here, not at the top: `spectra`'s package imports this
        # module (`spectra.derived`), so a top-level import would cycle
        try:                                # inside molbuilder
            from .spectra.selection import select_modes
        except ImportError:                 # beside a job, in a bundle
            from mode_selection import select_modes  # type: ignore
        inside = select_modes([m.frequency_cm1 for m in self.vibration.modes],
                              "range", freq_min_cm1=low_cm1,
                              freq_max_cm1=high_cm1)
        words = ("every frequency" if low_cm1 is None and high_cm1 is None
                 else f"{low_cm1:g} cm-1 and up" if high_cm1 is None
                 else f"up to {high_cm1:g} cm-1" if low_cm1 is None
                 else f"{low_cm1:g} to {high_cm1:g} cm-1")
        return self._then((k for k in self.modes if k in set(inside)), words)

    def only(self, modes) -> "ModeSelection":
        """Of the modes the steps before left, only those listed."""
        listed = self._numbers(modes)
        return self._then((k for k in self.modes if k in set(listed)),
                          f"only {_modes_words(listed)}")

    def add(self, modes) -> "ModeSelection":
        """The modes listed put in."""
        listed = self._numbers(modes)
        return self._then(set(self.modes) | set(listed),
                          f"{_modes_words(listed)} added")

    def drop(self, modes) -> "ModeSelection":
        """The modes listed taken out."""
        listed = self._numbers(modes)
        return self._then(set(self.modes) - set(listed),
                          f"{_modes_words(listed)} dropped")

    def final(self) -> List[int]:
        """The modes chosen, in the result's order -- none refused by name:
        nothing would move."""
        if not self.modes:
            raise ValueError("the mode selection holds no mode -- nothing "
                             "would move: " + "; ".join(self.steps))
        return list(self.modes)

    def __repr__(self) -> str:
        return (f"ModeSelection({_modes_words(self.modes)}: "
                + "; ".join(self.steps) + ")")


def _modes_words(modes: Sequence[int]) -> str:
    """``mode 3``, ``modes 12 and 13``, ``no mode``."""
    modes = list(modes)
    if not modes:
        return "no mode"
    return f"mode{'s' if len(modes) > 1 else ''} {_numbers_words(modes)}"


@dataclass(frozen=True, eq=False)
class Motion:
    """The molecule's motion (`engines/vibration-frames.md` § 3.1): a state
    for each mode, by number, in the result's order (``states``) -- each
    drawn on its own; ``temperature_k`` the thermal background's,
    ``temperature_from`` whether it was ``"given"`` or ``"the record's"``;
    ``classical`` whether the background is classical; ``set_apart`` the
    modes given a state of their own, said as :attr:`set_apart_modes`;
    ``left_out`` the modes left out as imaginary when it was made;
    ``said`` what the motion says -- information, never a decision."""
    vibration: Vibration = field(repr=False)
    states: Mapping[int, ModeState]
    temperature_k: float
    temperature_from: str
    classical: bool
    notes: Tuple[str, ...] = field(default=(), repr=False)
    set_apart: Tuple[int, ...] = ()
    left_out: Tuple[int, ...] = field(default=(), repr=False)
    chosen_by: Tuple[str, ...] = field(default=(), repr=False)

    @property
    def set_apart_modes(self) -> str:
        """Each mode given a state of its own, in words
        (`engines/vibration-frames.md` § 5.1's ``set_apart_modes``): its
        state and its frequency where it is not the result's; ``""`` when
        none."""
        return "; ".join(_state_words(k, self.states[k])
                         for k in self.set_apart)

    @property
    def said(self) -> Tuple[str, ...]:
        """Summary sentences: how the modes were chosen (a
        :class:`ModeSelection`'s steps), the modes left out as imaginary and
        still out of the motion, the temperature taken from the record, a
        mode set apart again."""
        out = []
        if self.chosen_by:
            out.append(f"the modes chosen: {'; '.join(self.chosen_by)}")
        out_still = [k for k in self.left_out if k not in self.states]
        if out_still:
            out.append(f"{_modes_words(out_still)} left out: imaginary, "
                       f"no harmonic state")
        out += list(self.notes)
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
                  classical: Optional[bool] = None,
                  state: Optional[ModeState] = None) -> "Motion":
        """A new motion with mode ``number`` set apart -- in any state of § 2
        by its arguments, or a :class:`ModeState` of that mode, never both --
        replacing its state; a mode outside the motion added.  A thermal or
        classical state given no temperature takes the motion's, said; in a
        classical motion a thermal state is classical unless
        ``classical=False``.  Set apart again, the later call's state,
        said."""
        mode = self.vibration.mode(number)
        k = mode.index_1based
        asked = _named(temperature_k, mean_quanta, quanta, classical)
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
        named = _named(temperature_k, mean_quanta, quanta, classical)
        t, source = _temperature_of(named, temperature_k, self.temperature_k,
                                    THE_MOTIONS)
        mode_state = mode.state(t, mean_quanta=mean_quanta, quanta=quanta,
                                classical=bool(classical))
        return self._replaced(k, replace(mode_state, temperature_from=source))

    def _replaced(self, k: int, mode_state: ModeState) -> "Motion":
        """This motion with mode ``k``'s state replaced, its modes in the
        result's order; set apart again, said."""
        notes = list(self.notes)
        if k in self.set_apart:
            notes.append(f"mode {k} set apart again: the later call's state "
                         f"replaces the earlier")
        states = dict(self.states)
        states[k] = mode_state
        states = {m.index_1based: states[m.index_1based]
                  for m in self.vibration.modes if m.index_1based in states}
        set_apart = self.set_apart + ((k,) if k not in self.set_apart else ())
        return replace(self, states=states, notes=tuple(notes),
                       set_apart=set_apart)

    def sample(self, n: int, *, seed: int, antithetic: bool,
               start: int = 0) -> "Snapshots":
        """``n`` snapshots of the motion (`engines/vibration-frames.md` § 3.3;
        science § 2c.3): each mode's coordinate drawn from its own state and
        its own stream of ``seed`` -- snapshots ``start`` ... ``start + n -
        1`` of the seed's sequence, so more snapshots later are the same seed
        drawn further, a set of their own.  With
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
        base = np.empty((base_n, len(cols)))
        for c, k in enumerate(cols):
            base[:, c] = self.states[k].draw(base_n, seed=seed,
                                             start=base_start)
        if antithetic:
            drawn = np.empty((n, len(cols)))
            drawn[0::2] = base
            drawn[1::2] = -base
        else:
            drawn = base
        q = np.vstack([np.zeros((1, len(cols))), drawn])
        return Snapshots(self, np.array(cols, dtype=int), q, antithetic,
                         seed, start)


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
    modes); ``antithetic`` whether pairs; ``seed`` the seed the snapshots
    were drawn with and ``start`` the first of its sequence drawn.
    ``motion``
    ``None`` for a set read back without its vibration
    (`vibration_frames.read_frame_set`): ``written_positions_ang`` its
    geometries as written, its analyses whole, its moves from those
    geometries."""
    motion: Optional[Motion] = field(repr=False)
    mode_index_1based: np.ndarray = field(repr=False)
    q_amu12_ang: np.ndarray = field(repr=False)
    antithetic: bool
    seed: int
    start: int
    written_positions_ang: Optional[np.ndarray] = field(default=None,
                                                        repr=False)
    read_said: Tuple[str, ...] = field(default=(), repr=False)

    @property
    def samples(self) -> int:
        """The snapshots, the frames after frame 0."""
        return int(self.q_amu12_ang.shape[0] - 1)

    @property
    def weights(self) -> np.ndarray:
        """0 for frame 0, then ``1/N`` each."""
        w = np.full(self.q_amu12_ang.shape[0], 1.0 / self.samples)
        w[0] = 0.0
        return w

    @property
    def atom_moves_ang(self) -> np.ndarray:
        """Each frame's every-atom move, ``sum_k Q_k L_k`` (frames x atoms x
        3) -- the displacement map, by the vibration's one rule; read back
        without the vibration, from the geometries as written."""
        if self.motion is None:
            p = np.asarray(self.written_positions_ang, dtype=float)
            return p - p[:1]
        return self.motion.vibration._moves(self.mode_index_1based,
                                            self.q_amu12_ang)

    @property
    def positions_ang(self) -> np.ndarray:
        """Each frame's geometry, frame 0 the relaxed one (frames x atoms x
        3)."""
        if self.motion is None:
            return np.asarray(self.written_positions_ang, dtype=float)
        return self.motion.vibration.positions_ang[None] + self.atom_moves_ang

    @property
    def _largest(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.motion is None:
            return _largest_of_geometries(self.written_positions_ang, ())
        return _largest_moves(self.motion.vibration, self.mode_index_1based,
                              self.q_amu12_ang)

    @property
    def max_atom_displacement_ang(self) -> np.ndarray:
        """Each frame's largest single-atom move, Å (frames,)."""
        return self._largest[0]

    @property
    def max_atom_mode_index_1based(self) -> np.ndarray:
        """The mode moving each frame's most-moved atom most -- 0 for a frame
        that moves nothing (frame 0), and for every frame read back without
        the vibration (frames,)."""
        return self._largest[1]

    @property
    def said(self) -> Tuple[str, ...]:
        """Summary sentences (`engines/vibration-frames.md` § 3.3): the
        largest single-atom move over the set, its frame and the mode moving
        it, beside each mode's spread -- read against a bond length; one
        pair alone, its error having nothing to be taken from."""
        largest = self.max_atom_displacement_ang
        i = int(np.argmax(largest))
        if self.motion is None:
            out = [f"the largest single-atom move is {largest[i]:.4f} Å, "
                   f"{_frame_words(i, len(largest))}; read without its "
                   f"vibration, the modes moving it and their spreads are "
                   f"not known"]
        else:
            spreads = ", ".join(
                f"mode {k} {self.motion.states[int(k)].spread_amu12_ang:.3g}"
                for k in self.mode_index_1based)
            out = [f"the largest single-atom move is {largest[i]:.4f} Å, "
                   f"{_frame_words(i, len(largest))}, mode "
                   f"{int(self.max_atom_mode_index_1based[i])} moving it "
                   f"most; each mode's spread (amu^1/2·Å): {spreads}"]
        if self.antithetic and self.samples == 2:
            out.append("one mirrored pair: its error has nothing to be taken "
                       "from")
        return (*self.read_said, *out)

    def __repr__(self) -> str:
        modes = len(self.mode_index_1based)
        head = (f"Snapshots({self.samples} snapshot"
                f"{'' if self.samples == 1 else 's'} of {modes} "
                f"mode{'' if modes == 1 else 's'}, "
                + ("mirrored pairs" if self.antithetic else "independent")
                + f", seed {self.seed}, start {self.start})")
        return "\n".join([head] + [f"  {s}" for s in self.said])

    # ---- the analyses of a set's results (§ 7) ---------------------------
    def _values(self, values) -> np.ndarray:
        """The snapshots' values, frame 0 left out: one value per frame of the
        set as written, frame 0 first, any trailing axes kept."""
        v = np.asarray(values, dtype=float)
        if v.ndim == 0 or v.shape[0] != self.samples + 1:
            raise ValueError(
                f"one value per frame of the set as written "
                f"({self.samples + 1}, frame 0 first); got "
                f"{v.shape[0] if v.ndim else 'one'} value"
                f"{'' if v.ndim and v.shape[0] == 1 else 's'}")
        return v[1:]

    def _units(self, values) -> Tuple[np.ndarray, int, List[str]]:
        """The snapshots' values kept -- each frame, or with pairs each pair
        whole, still one value per snapshot in the order drawn -- a value
        missing dropping its snapshot or its pair, said."""
        x = self._values(values)
        said: List[str] = []
        missing = np.isnan(x.reshape(x.shape[0], -1)).any(axis=1)
        if self.antithetic:
            pair_missing = missing.reshape(-1, 2).any(axis=1)
            if pair_missing.any():
                said.append(f"{int(pair_missing.sum())} pair"
                            f"{'s' if pair_missing.sum() > 1 else ''} left out:"
                            f" a value missing")
            kept = np.repeat(~pair_missing, 2)
            return x[kept], int(kept.sum()), said
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

    def standard_error(self, values) -> Estimate:
        """The average over the snapshots and its error (science § 5.5) --
        ``values`` one per frame of the set as written, frame 0 first, any
        trailing axes kept: for independent frames the frames' spread over
        ``sqrt(N)``, for pairs the pair means' over ``sqrt(N/2)``; frame 0
        left out."""
        kept, n, said = self._units(values)
        units = pair_means(kept) if self.antithetic else kept
        self._none_left(units.shape[0], said)
        value, error = mean_with_error(units)
        return Estimate(value, error, n, tuple(said))

    def parts(self, values) -> Parts:
        """A paired set's spread split into its even and odd parts (science
        § 5.5): the spread of the pair means, the even part's, and of the
        pair half-differences, the odd part's -- whether pairs helped;
        ``values`` as :meth:`standard_error` takes them.  A set drawn without
        pairs is refused by name."""
        if not self.antithetic:
            raise ValueError("parts split a paired set's spread: this set's "
                             "snapshots were drawn alone, not in mirrored "
                             "pairs")
        kept, _, said = self._units(values)
        self._none_left(kept.shape[0] // 2, said)
        even, odd = even_odd_parts(kept)
        return Parts(even, odd, tuple(said))


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


def _state_words(k: int, s: ModeState) -> str:
    """A mode set apart alone, its state in words
    (`engines/vibration-frames.md` § 5.1's ``set_apart_modes``)."""
    st = s.state
    t = "" if s.temperature_k is None else _temperature_words(s)
    if st.kind == LEVEL:
        words = f"mode {k}: {_quanta_words(st.quanta)} exactly"
    elif st.kind == CLASSICAL:
        words = f"mode {k}: classical at {t}"
    elif s.temperature_k is not None:
        words = f"mode {k}: thermal at {t}"
    else:
        words = (f"mode {k}: thermal, "
                 f"{_quanta_words(st.thermal_quanta, kind='mean ')}")
    return words + _frequency_words(s.mode)


__all__ = ["DEFAULT_NODES", "STATES", "checked_state", "checked_temperature_k",
           "zero_point_amplitude_amu12_ang",
           "bose_einstein_mean_quanta", "spread_of_quanta",
           "thermal_spread_amu12_ang",
           "classical_spread_amu12_ang",
           "thermal_spread_of", "THERMAL_ROW_KEYS", "thermal_row",
           "mode_over_atoms", "largest_move", "outermost_move_ang",
           "q_for_largest_move", "atom_moves", "displaced", "coordinate_of",
           "zero_point_displacement_ang", "largest_zero_point_move_ang",
           "motion_shares_of", "shares_by", "reduced_mass_amu",
           "force_constant_mdyn_per_ang", "THERMAL", "LEVEL", "CLASSICAL",
           "Frames",
           "Weights", "State", "state",
           "bose_einstein_temperature", "GIVEN", "THE_RECORDS", "THE_MOTIONS",
           "THE_SETS",
           "ModeState", "Mode", "Vibration", "ModeSelection",
           "Motion", "Estimate", "Parts", "Snapshots"]


#: WHAT THIS MODULE NEEDS BESIDE IT in a job's bundle -- the modules it
#: imports there by their bare names (`runwrap.bundle_members`).
TRAVELS_WITH = ("molbuilder.constants", "molbuilder.oscillator",
                "molbuilder.sampling", "molbuilder.distribution",
                "molbuilder.spectra.selection")

"""A mode's state built in steps, held to the science's closed forms
(`science/vibrational-averaging.md` § 2e; `engines/vibration-frames.md`
§ 9.1): the lists each chain of steps makes, what the steps do to the mean,
when two chains are one state, where a list is cut, the worked chain's
frames, the exact families against the list's route, the draws, the search
for a point of a given chance, the statistics of snapshots, the refusals and
the words.  Every expectation is a formula of the science or a
reference outside the toolkit (the standard library's normal quantile,
numpy's generator); where two of the toolkit's routes are held to each other
-- a closed family and the list (`engines/vibration-frames.md` § 9.1's
`of_state`, "each the other's check"; the user's decision 1(b) of 2026-11-25,
`archive/2026-11-26-vibration-frames-steps-design.md`) -- the test says so."""
from __future__ import annotations

import math
import statistics
import warnings

import numpy as np
import pytest

from molbuilder.constants import CM1_KELVIN
# the tools as a person's script reaches them (`engines/vibration-frames.md`
# § 9: every tool through `molbuilder.postprocess`)
from molbuilder.postprocess import (Oscillator, dist, oscillator, sampling,
                                    states)

#: AN OSCILLATOR THE NUMBERS NEED NO FREQUENCY FOR: every chain below names
#: its thermal start by a mean number of quanta, so any frequency gives the
#: same list.
_ANY = Oscillator(100.0)


def _thermal(nbar, n):
    return nbar ** n / (nbar + 1.0) ** (n + 1)


def _bose_einstein(frequency_cm1, temperature_k):
    """Bose and Einstein's mean, ``1 / (e^(h c nu / k_B T) - 1)``."""
    return 1.0 / math.expm1(frequency_cm1 * CM1_KELVIN / temperature_k)


def _phi(z):
    """The standard normal's chance of ``z`` or less."""
    return 0.5 * np.array([math.erfc(-v / math.sqrt(2.0)) for v in z])


#: SCIENCE § 2e.1's TABLE: each state, its chain, and its list, level by level.
_LISTS = {
    "thermal": (states.thermal(mean_quanta=1.0), lambda n: _thermal(1.0, n)),
    "one photon on a warm mode": (
        states.thermal(mean_quanta=1.0).add_quantum(),
        lambda n: n * 1.0 ** (n - 1) / 2.0 ** (n + 1) if n else 0.0),
    "one quantum taken from a warm mode": (
        states.thermal(mean_quanta=3.69).remove_quantum(),
        lambda n: (n + 1) * 3.69 ** n / 4.69 ** (n + 2)),
    "two quanta taken from a warm mode": (
        states.thermal(mean_quanta=1.0).remove_quantum().remove_quantum(),
        lambda n: math.comb(n + 2, 2) / 2.0 ** (n + 3)),
    "3 quanta exactly": (states.level(3), lambda n: 1.0 if n == 3 else 0.0),
}


@pytest.mark.parametrize("name", list(_LISTS))
def test_each_chain_makes_the_sciences_list(name):
    """Science § 2e.1, § 2e.2: every chain of steps is a list over the
    levels -- the thermal geometric list, a photon on a warm mode, one and
    two quanta taken from one, a level -- each level as the closed form
    gives it.  The silent failure: a step whose weights or normalisation are
    another's, which every number downstream would carry."""
    chain, expected = _LISTS[name]
    p = chain.on(_ANY).populations
    want = np.array([expected(n) for n in range(min(p.size, 40))])
    assert p[:want.size] == pytest.approx(want, abs=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("chain, mean", [
    (states.thermal(mean_quanta=1.0).add_quantum(), 3.0),
    (states.thermal(mean_quanta=1.0).remove_quantum(), 2.0),
    (states.thermal(mean_quanta=1.0).add_quantum().add_quantum(), 5.0),
    (states.thermal(mean_quanta=3.69).add_quantum(), 8.38),
    (states.thermal(mean_quanta=3.69).remove_quantum(), 7.38),
    (states.level(4).add_quantum(), 5.0),
    (states.thermal(mean_quanta=1.0).add_quantum().remove_quantum(),
     10.0 / 3.0),
], ids=["a photon, warm", "a quantum taken, warm", "two photons, warm",
        "a photon, 50 cm-1 at 300 K", "a quantum taken, 50 cm-1 at 300 K",
        "a photon on 4 quanta", "a photon, then a quantum taken, warm"])
def test_the_steps_move_the_mean_as_the_science_says(chain, mean):
    """Science § 2e.2: one quantum added moves the mean by
    ``1 + Var(n)/(<n> + 1)``, one removed by ``-1 + Var(n)/<n>`` -- a thermal
    mode to ``2 nbar + 1`` and ``2 nbar``, a level by exactly one; a photon
    on a warm mode of ``nbar = 1`` (``p_n = n / 2^(n+1)``, ``<n> = 3``,
    ``<n^2> = 13``, ``Var(n) = 4``), then a quantum taken: ``3 - 1 + 4/3 =
    10/3`` -- and no level's chance below 0.  The silent failure: a quantum added as a
    shift of the list by one level, mean ``nbar + 1``."""
    state = chain.on(_ANY)
    assert state.mean_quanta == pytest.approx(mean, abs=1e-12)
    # THE STEPS THEMSELVES, not only the closed form a family's mean takes
    p = state.populations
    assert float((np.arange(p.size) * p).sum()) == pytest.approx(mean,
                                                                 abs=1e-10)
    assert float(np.min(p)) >= 0.0


def test_a_quantum_taken_from_a_cold_mode_leaves_twice_its_mean():
    """Science § 2e.2: a thermal mode, however cold, holds a little of
    level 1, and one quantum removed leaves ``2 nbar`` on average -- a
    1000 cm-1 mode at 4 K, ``nbar = 1 / (e^(hc nu / k_B T) - 1)``.  The
    silent failure: a mean of 0, the event lost to the cut."""
    state = states.thermal(temperature_k=4.0).remove_quantum().on(
        Oscillator(1000.0))
    # AS A RATIO: the mean is 1e-156, which any absolute tolerance holds 0 to
    assert state.mean_quanta / _bose_einstein(1000.0, 4.0) == pytest.approx(
        2.0, rel=1e-12)


def test_a_quantum_taken_from_a_mode_too_cold_for_a_number_is_its_limit():
    """Science § 2e.2: one quantum taken from a thermal state, ``p_n = (n + 1)(1 -
    t)^2 t^n``, has no division by the mean left in it -- at 3000 cm-1 and
    4 K, ``t = e^-1079`` below what a number holds, the list is the ground
    state's, its limit, not a refusal; at exactly 0 K, the person's number,
    nothing is there to take.  THE SILENT FAILURE: "holds none" said of a
    mode above 0 K because its mean rounded to 0."""
    state = states.thermal(temperature_k=4.0).remove_quantum().on(
        Oscillator(3000.0))
    assert list(state.populations) == [1.0]
    with pytest.raises(ValueError, match="exactly 0 K"):
        states.thermal(temperature_k=0.0).remove_quantum()
    # A PHOTON, THEN TWO TAKEN, at 4 K on 3000 cm-1: p_n ∝ (n+1)(n+2)^2
    # t^n, t = e^-1079 below what a double holds -- the ground state's
    # list, its limit, not a refusal; at exactly 0 K the photon's quantum is
    # taken once, and the second has none.
    warm = states.thermal(temperature_k=4.0).add_quantum().remove_quantum() \
        .remove_quantum()
    assert list(warm.on(Oscillator(3000.0)).populations) == [1.0]
    with pytest.raises(ValueError, match="holds none"):
        states.thermal(temperature_k=0.0).add_quantum().remove_quantum() \
            .remove_quantum().on(Oscillator(1000.0)).populations


def test_quanta_taken_from_a_thermal_state_are_their_formula():
    """Science § 2e.1, § 2e.2: ``k`` quanta taken from a thermal state, ``t = nbar /
    (nbar + 1)``, ``p_n = C(n + k, k) (1 - t)^(k+1) t^n`` -- for ``nbar = 1``
    and two taken, ``C(n + 2, 2) / 2^(n + 3)``, mean ``3 nbar``.  THE SILENT
    FAILURE: a second removal from the first's list read one level off."""
    state = states.thermal(mean_quanta=1.0).remove_quantum().remove_quantum().on(
        _ANY)
    p = state.populations
    assert p[:30] == pytest.approx([math.comb(n + 2, 2) / 2.0 ** (n + 3)
                                    for n in range(30)], abs=1e-15)
    assert state.mean_quanta == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize("one, other", [
    (states.level(1), states.ground().add_quantum()),
    (states.level(3), states.ground().add_quantum().add_quantum()
     .add_quantum()),
    (states.ground(), states.level(1).remove_quantum()),
    (states.level(2), states.level(3).remove_quantum()),
    (states.ground(), states.level(0)),
], ids=["a photon on the ground state", "three photons",
        "a quantum taken from one", "a quantum taken from three",
        "no quanta"])
def test_two_chains_that_make_one_state_reduce_to_one(one, other):
    """`engines/vibration-frames.md` § 9.1: two states are one where their
    reduced chains are -- the ground state or a level with quanta added or
    taken straight after it, that level; a level of none, the ground state.
    The silent failure: a chain computed the long way and its closed family
    missed -- a level drawn from its list, not its own table."""
    assert one.reduced() == other.reduced()
    assert one.on(_ANY) == other.on(_ANY)


@pytest.mark.parametrize("start", [states.thermal(temperature_k=0.0),
                                   states.thermal(mean_quanta=0.0)],
                         ids=["at 0 K", "of no quanta"])
def test_a_thermal_state_of_no_quanta_is_the_ground_states_curve(start):
    """`engines/vibration-frames.md` § 9.1 (the user's rule of 2026-10-28,
    "a thermal set is stated by its temperature"): a thermal state at 0 K
    keeps its step and its temperature, and its distribution is the ground
    state's, the bell curve of ``<x^2> = 1``.  The silent failure: a
    thermal set at 0 K recorded as the ground state, its temperature
    lost."""
    state = start.on(_ANY)
    assert state.steps[0].name == "thermal"
    assert state.temperature_k == start.start.get("temperature_k")
    curve = dist.of_state(state)
    x = np.linspace(-4.0, 4.0, 9)
    assert curve.pdf(x) == pytest.approx(
        np.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi), abs=1e-15)


@pytest.mark.parametrize("precision", [1e-8, 1e-12, 1e-16])
def test_a_lists_precision_is_a_parameter(precision):
    """Science § 2e.4, § 2e.6: ``chain.on(oscillator, precision=)`` -- a
    thermal list keeps its chances to where those left out hold less than
    ``precision``, ``t^N`` beyond level ``N``: ``N = ceil(ln(precision) /
    ln t)``, ``t = nbar/(nbar + 1)`` -- 154 levels at 1e-16 for a 50 cm-1
    mode at 300 K.  THE SILENT FAILURE: a precision asked and another
    kept."""
    nbar = _bose_einstein(50.0, 300.0)
    t = nbar / (nbar + 1.0)
    state = states.thermal(temperature_k=300.0).on(Oscillator(50.0),
                                                precision=precision)
    assert state.populations_for(0).size == math.ceil(math.log(precision)
                                                      / math.log(t))


def _left_out_share(chances, kept, order=12):
    """The share of ``<x^order>`` the levels from ``kept`` on hold, from a
    list's exact chances -- each level weighing ``(2n + 1)^(order/2)``, the
    cut's weighting (science § 2e.4)."""
    n = np.arange(chances.size)
    w = chances * (2.0 * n + 1.0) ** (order // 2)
    return float(w[kept:].sum() / w.sum())


@pytest.mark.parametrize("frequency_cm1, chain, exact, precision", [
    (30.0, states.thermal(temperature_k=300.0).add_quantum(),
     lambda n, t: n * (1.0 - t) ** 2 * t ** (n - 1.0), 1e-8),
    (150.0, states.thermal(temperature_k=300.0).add_quantum(),
     lambda n, t: n * (1.0 - t) ** 2 * t ** (n - 1.0), 1e-12),
    (10.0, states.thermal(temperature_k=300.0),
     lambda n, t: (1.0 - t) * t ** n, 1e-12),
], ids=["one quantum on a 30 cm-1 mode at 300 K, to 1e-8",
        "one quantum on a 150 cm-1 mode at 300 K, to 1e-12",
        "a 10 cm-1 mode at 300 K, to 1e-12"])
def test_a_list_leaves_out_less_than_its_precision_and_no_level_more(
        frequency_cm1, chain, exact, precision):
    """Science § 2e.4, § 2e.6 (how exactly the cut is placed): a list keeps
    its levels up to where those it leaves out hold less than the precision
    of ``<x^12>`` -- and not one level fewer.  The exact chances are the
    science's closed forms, ``p_n = (1 - t) t^n`` thermal and ``n (1 - t)^2
    t^(n-1)`` with one quantum added, ``t = nbar/(nbar + 1)``.  THE SILENT
    FAILURE: a list built only to the precision, its cut placed one level
    early, leaving out more than was asked (243 levels for 244, 1.04 of the
    precision, on the first row)."""
    nbar = _bose_einstein(frequency_cm1, 300.0)
    t = nbar / (nbar + 1.0)
    chances = exact(np.arange(20000, dtype=float), t)
    kept = chain.on(Oscillator(frequency_cm1),
                    precision=precision).populations.size
    assert _left_out_share(chances, kept) < precision
    assert _left_out_share(chances, kept - 1) >= precision


def test_a_list_is_cut_where_the_averages_asked_of_it_stop_changing():
    """Science § 2e.4: a 50 cm-1 mode at 300 K keeps 154 levels for its
    chances, 183, 206 and 227 for the averages three, five and seven frames
    match (up to ``<x^4>``, ``<x^8>``, ``<x^12>``).  The silent failure: a
    list cut for its chances alone, whose frames' high averages are off."""
    s = states.thermal(temperature_k=300.0).on(Oscillator(50.0))
    assert [s.populations_for(k).size for k in (0, 4, 8, 12)] == [
        154, 183, 206, 227]


def test_a_states_list_is_cut_for_the_largest_rule_the_doors_offer():
    """Science § 2e.4, § 2e.6: a state's list as it is placed (its
    ``populations``) is cut for the highest average the largest rule the
    doors offer reproduces -- seven frames, ``<x^12>`` -- so that rule's
    frames from the list are the state's.  THE SILENT FAILURE: the doors
    offering more frames than the list was cut for, the outermost frames'
    averages off."""
    from molbuilder.vibration import OFFERED_NODES
    order = 2 * max(OFFERED_NODES) - 2
    s = states.thermal(temperature_k=300.0).on(Oscillator(50.0))
    assert s.populations.size >= s.populations_for(order).size


@pytest.mark.parametrize("route", ["closed form", "the list's route"])
def test_the_worked_chains_three_frames_are_the_sciences(route):
    """Science § 2e.3's worked chain: thermal state to ``nbar = 1``, add a quantum --
    ``<x^2> = 7``, ``<x^4> = 99`` in units of ``Q_zp``; three frames at 0
    and ``± sqrt(99/7)``, weights ``1 - 2w`` and ``w = 7 / (2 · 99/7)`` --
    by the closed form and by the list's route alike.  The silent failure:
    the thermal stand-in's frames, at ``± sqrt(21)``."""
    state = states.thermal(mean_quanta=1.0).add_quantum().on(_ANY)
    d = (dist.of_state(state) if route == "closed form"
         else dist.LevelList(state.populations))
    a = math.sqrt(99.0 / 7.0)
    w = 7.0 / (2.0 * a * a)
    x, weights = d.gauss_rule(3)
    order = np.argsort(x)
    assert x[order] == pytest.approx([-a, 0.0, a], abs=1e-10)
    assert weights[order] == pytest.approx([w, 1.0 - 2.0 * w, w], abs=1e-12)
    assert (d.moment(2), d.moment(4)) == pytest.approx((7.0, 99.0), rel=1e-12)


@pytest.mark.parametrize("chain, family", [
    (states.thermal(mean_quanta=1.0), dist.Gaussian),
    (states.thermal(mean_quanta=1.0).add_quantum(), dist.QuantumAdded),
    (states.level(2), dist.Level),
], ids=["a thermal state", "one photon on a warm mode", "2 quanta exactly"])
def test_an_exact_family_and_the_lists_route_are_one_distribution(chain,
                                                                    family):
    """`engines/vibration-frames.md` § 9.1's `of_state` (the user's decision
    1(b) of 2026-11-25); science § 2e.3:
    where a family has a closed form, its density, distribution function,
    averages and frames are the list's -- each route the other's check.
    The silent failure: a closed form that drifted from the state it names."""
    state = chain.on(_ANY)
    closed, listed = dist.of_state(state), dist.LevelList(state.populations)
    assert isinstance(closed, family)
    x = np.linspace(-6.0, 6.0, 25)
    assert closed.pdf(x) == pytest.approx(listed.pdf(x), abs=1e-10)
    assert closed.cdf(x) == pytest.approx(listed.cdf(x), abs=1e-10)
    for k in (2, 4, 6, 8):
        assert closed.moment(k) == pytest.approx(listed.moment(k), rel=1e-10)
    assert np.sort(closed.gauss_rule(5)[0]) == pytest.approx(
        np.sort(listed.gauss_rule(5)[0]), abs=1e-8)


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_a_levels_own_rule_reproduces_its_averages(n):
    """Science § 2e.3: level ``n``'s four frames reproduce ``<n|x^k|n>`` to
    ``k = 7`` -- ``2n + 1``, ``3(2n^2 + 2n + 1)``, ``5(4n^3 + 6n^2 + 8n +
    3)``, the odd ones 0.  The silent failure: frames in the recurrence's
    unit, ``sqrt 2`` short."""
    x, w = dist.of_state(states.level(n).on(_ANY)).gauss_rule(4)
    want = {2: 2 * n + 1, 4: 3 * (2 * n * n + 2 * n + 1),
            6: 5 * (4 * n ** 3 + 6 * n * n + 8 * n + 3)}
    for k in range(8):
        terms = w * x ** k                  # odd k: terms cancelling to 0
        assert float(terms.sum()) == pytest.approx(
            want.get(k, 1.0 if k == 0 else 0.0), rel=1e-12,
            abs=1e-14 * float(np.abs(terms).sum()))


@pytest.mark.parametrize("n", [0, 1, 3, 6])
def test_a_levels_wavefunction_is_hermites(n):
    """Science § 2d.2: level ``n``'s wavefunction in ``x = Q / Q_zp``,
    ``He_n(x) e^(-x^2/4) / ((2 pi)^(1/4) sqrt(n!))`` -- ``He_n`` the
    probabilists' Hermite polynomial -- and its density the level's
    distribution's.  The silent failure: a wavefunction normalised in
    another unit, its density off by ``sqrt 2``."""
    x = np.linspace(-4.0, 4.0, 17)
    he = np.polynomial.hermite_e.hermeval(x, [0.0] * n + [1.0])
    want = he * np.exp(-x * x / 4.0) / ((2.0 * math.pi) ** 0.25
                                       * math.sqrt(math.factorial(n)))
    assert oscillator.level_wavefunction(n, x) == pytest.approx(want,
                                                                abs=1e-14)
    assert oscillator.level_density(n, x) == pytest.approx(want * want,
                                                           abs=1e-14)
    assert dist.Level(n).pdf(x) == pytest.approx(want * want, abs=1e-14)


@pytest.mark.parametrize("open_chain, given", [
    (states.thermal(), states.thermal(temperature_k=300.0)),
    (states.classical_thermal(), states.classical_thermal(temperature_k=300.0)),
], ids=["a thermal state", "a classical thermal state"])
def test_an_open_temperature_is_filled_where_the_chain_is_placed(open_chain,
                                                                 given):
    """`engines/vibration-frames.md` § 9.1: ``chain.on(oscillator,
    temperature_k=T)`` fills a thermal state left open -- the same state as the thermal state
    at ``T``, Bose and Einstein's mean on a 200 cm-1 mode at 300 K.  The
    silent failure: the record's temperature read as another."""
    o = Oscillator(200.0)
    state = open_chain.on(o, temperature_k=300.0)
    assert state == given.on(o)
    assert given.on(o, temperature_k=300.0) == given.on(o)   # the same twice
    assert state.temperature_k == 300.0
    if not state.is_classical:
        assert state.mean_quanta == pytest.approx(
            _bose_einstein(200.0, 300.0), rel=1e-12)


def test_a_thermal_states_two_routes_draw_one_coordinate_from_one_number():
    """Science § 2e.3: the bell curve and the list draw the same coordinate
    from the same ``z`` -- ``x = sqrt(2 nbar + 1) z`` -- so a set drawn by
    either route with one seed is the same set.  The silent failure: the
    list's draws mirrored, ``-sqrt(3) z``, which a Kolmogorov test passes."""
    z = np.array([-2.0, -0.5, 0.3, 1.7])
    listed = dist.LevelList(states.thermal(mean_quanta=1.0).on(_ANY).populations)
    assert listed.draw(z) == pytest.approx(math.sqrt(3.0) * z, rel=1e-12)


@pytest.mark.parametrize("distribution", [
    dist.QuantumAdded(0.62), dist.Level(3)],
    ids=["one photon on a warm mode", "3 quanta"])
def test_each_draw_is_where_the_distribution_reaches_phi_of_its_number(
        distribution):
    """Science § 2d.5: each draw is the ``x`` with ``F(x) = Phi(z)``, to
    rounding -- for every ``z``, the bands a many-humped density's flat
    stretches send a Newton step back and forth across included.  The silent
    failure: draws that miss ``F = Phi(z)`` in bands, which a Kolmogorov
    test of the whole set passes."""
    z = sampling.normal_stream((3, 1, 0), 2000)
    assert np.max(np.abs(distribution.cdf(distribution.draw(z)) - _phi(z))) \
        < 1e-12


def test_a_paired_sets_even_and_odd_parts_are_their_spreads():
    """Science § 5.5: values one per snapshot, each snapshot followed by
    its mirror -- ``1, -1, 3, -3, 2, -2`` is the pairs ``(1, -1), (3, -3),
    (2, -2)``, grouped two by two by the functions themselves: pair means 0,
    half-differences 1, 3, 2, their spread 1 (``ddof`` 1); ``1, 1, 3, 3, 2,
    2`` the other way round; a value per energy kept, each energy its own;
    an odd count refused by name (a set's frame 0 left in); each pair the
    weight of its two values, its mean at them (the weights the set stores,
    the user, 2026-11-26).  The silent failure: an odd part twice its size,
    the pairs taken across the energies, or a pair's mean that ignores its
    values' weights."""
    sixth = np.full(6, 1.0 / 6.0)
    assert sampling.even_odd_parts([1.0, -1.0, 3.0, -3.0, 2.0, -2.0],
                                   sixth) == pytest.approx((0.0, 1.0),
                                                           abs=1e-15)
    assert sampling.even_odd_parts([1.0, 1.0, 3.0, 3.0, 2.0, 2.0],
                                   sixth) == pytest.approx((1.0, 0.0),
                                                           abs=1e-15)
    per_energy = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
    got = sampling.pair_means(per_energy, np.full(4, 0.25))
    assert got.means == pytest.approx(np.array([[2.0, 20.0], [6.0, 60.0]]),
                                      abs=1e-15)
    assert got.weights == pytest.approx([0.5, 0.5], abs=1e-15)
    # A PAIR'S MEAN AT ITS VALUES' WEIGHTS: (0.1 * 1 + 0.3 * 3) / 0.4.
    lopsided = sampling.pair_means([1.0, 3.0, 5.0, 7.0], [0.1, 0.3, 0.3, 0.3])
    assert lopsided.means == pytest.approx([2.5, 6.0], abs=1e-15)
    assert lopsided.weights == pytest.approx([0.4, 0.6], abs=1e-15)
    # EACH SPREAD AT THE PAIRS' WEIGHTS, 0.2, 0.4, 0.4: the halves 1, 3, 2
    # about their weighted mean 2.2, sum_j W_j (h_j - 2.2)^2 = 0.56 over
    # 1 - sum_j W_j^2 = 0.64 -- the odd part sqrt(0.875).
    assert sampling.even_odd_parts([1.0, -1.0, 3.0, -3.0, 2.0, -2.0],
                                   [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]) == \
        pytest.approx((0.0, math.sqrt(0.875)), abs=1e-15)
    assert sampling.even_odd_parts([1.0, 1.0, 3.0, 3.0, 2.0, 2.0],
                                   [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]) == \
        pytest.approx((math.sqrt(0.875), 0.0), abs=1e-15)
    # A PAIR OF WEIGHT 0 adds nothing -- its mean its two values' plain one
    # (`engines/vibration-frames.md` § 9.1) -- the other two's own.
    assert sampling.pair_means([9.0, 7.0, 1.0, 3.0], [0.0, 0.0, 0.5, 0.5]) \
        .means == pytest.approx([8.0, 2.0], abs=1e-15)
    assert sampling.mean_with_error(*sampling.pair_means(
        [9.0, 9.0, 1.0, 3.0, 5.0, 7.0], [0.0, 0.0, 0.25, 0.25, 0.25, 0.25])) \
        == pytest.approx(sampling.mean_with_error(*sampling.pair_means(
            [1.0, 3.0, 5.0, 7.0], [0.25] * 4)), abs=1e-15)
    with pytest.raises(ValueError, match="two by two.*got 3 values$"):
        sampling.pair_means([0.0, 1.0, 3.0], [1 / 3] * 3)
    with pytest.raises(ValueError, match="got 1 value$"):
        sampling.pair_means([0.0], [1.0])
    with pytest.raises(ValueError, match="got a single number$"):
        sampling.pair_means(3.0, [1.0])


def test_a_states_list_cannot_be_edited():
    """A state's list is read-only: an edit is refused, never carried into
    another state of the same chain (the lists are made once per chain).
    The silent failure: one script's edit changing every later state."""
    first = states.thermal(mean_quanta=1.0).on(_ANY).populations
    with pytest.raises(ValueError, match="read-only"):
        first[0] = 1.0
    assert states.thermal(mean_quanta=1.0).on(_ANY).populations[0] == \
        pytest.approx(0.5, abs=1e-15)


def _normal_quantile(p):
    return statistics.NormalDist().inv_cdf(p)


#: THE SEARCH'S CASES: a distribution, the chance asked, its exact point
#: where one has a formula (else ``None``: ``F`` itself checked -- also
#: where ``F`` is flat at the answer, one quantum's middle, ``F - 1/2 ~
#: x^3 / 3 sqrt(2 pi)``, so every point within about 7e-6 of 0 has ``F``
#: exactly 1/2 in a double), the bracket, and starts chosen wrong in each
#: way -- on the
#: answer, at each end, on the wrong side, where the slope is 0, on the
#: wrong hump.
_SEARCHES = {
    "a bell curve": (dist.Gaussian(1.0), 0.3, _normal_quantile(0.3),
                     (-12.0, 12.0), [_normal_quantile(0.3), -12.0, 12.0, 5.0]),
    "one quantum, its middle": (dist.Level(1), 0.5, None, (-12.0, 12.0),
                                [0.0, 1.5, -3.0]),
    "one quantum, a quarter": (dist.Level(1), 0.25, None, (-12.0, 12.0),
                               [0.0, 2.5, -2.5]),
    "one photon on a warm mode, two humps": (
        dist.QuantumAdded(0.2), 0.4, None, (-12.0, 12.0), [0.0, 1.5, -1.5]),
}


@pytest.mark.parametrize("name", list(_SEARCHES))
def test_the_search_finds_the_point_from_every_wrong_start(name):
    """Science § 2d.5, § 2e.6: the point where ``F`` reaches ``p`` -- each
    start deliberately wrong (on the answer, at an end of the bracket, on
    the wrong side, where the density is 0, on the other hump) reaches the
    same point: the standard library's normal quantile, one quantum's middle
    0 by its symmetry, or where no formula gives it, ``F`` within rounding
    of ``p``.  THE SILENT
    FAILURE: a start already on the answer left for the bracket's middle
    (one quantum's middle at 0.361), or a step bouncing between two humps."""
    d, p, exact, (lo, hi), starts = _SEARCHES[name]
    found = oscillator._bracketed_newton(d.cdf, d.pdf,
                                         np.full(len(starts), p), lo, hi,
                                         np.array(starts))
    if exact is None:
        assert np.max(np.abs(np.asarray(d.cdf(found)) - p)) < 1e-14
    else:
        assert found == pytest.approx([exact] * len(starts), abs=1e-12)
    for start, point in zip(starts, found):
        if d.cdf(np.array([start]))[0] == p:     # a start on the answer
            assert point == start                # is kept, to the last bit


#: ONE OF EACH FAMILY a person reaches through `dist` -- every one symmetric
#: about 0 (`engines/vibration-frames.md` § 9.1 module 3).
_FAMILIES = {
    "a bell curve": lambda: dist.Gaussian(1.3),
    "one quantum on a warm mode": lambda: dist.QuantumAdded(0.62),
    "3 quanta": lambda: dist.Level(3),
    "a list": lambda: dist.LevelList(
        states.thermal(mean_quanta=1.0).add_quantum().remove_quantum()
        .on(_ANY).populations),
}


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_every_odd_average_is_zero(family):
    """`engines/vibration-frames.md` § 9.1: ``moment(k)`` is exact, 0 for an
    odd ``k`` -- every distribution here is even -- and so is a level's
    ``<n|x^k|n>``.  THE SILENT FAILURE: an odd average that is not 0,
    re-weighted frames placed for a lopsided state that is not there."""
    d = _FAMILIES[family]()
    assert [d.moment(k) for k in (1, 3, 5, 7)] == [0.0] * 4
    assert [oscillator.level_moment(3, k) for k in (1, 3, 5)] == [0.0] * 3


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_a_quantile_is_where_the_distribution_reaches_its_chance(family):
    """`engines/vibration-frames.md` § 9.1: ``quantile(p)`` is the ``x``
    with ``F(x) = p`` -- for every family, ``F`` back at ``p`` to rounding;
    the bell curve's ``sigma Phi^-1(p)``, the standard library's.  THE
    SILENT FAILURE: a quantile on the wrong side of the middle, every draw
    mirrored."""
    d = _FAMILIES[family]()
    for p in (0.05, 0.3, 0.5, 0.7, 0.95):
        assert float(d.cdf(d.quantile(p))) == pytest.approx(p, abs=1e-14)
    if family == "a bell curve":
        assert float(d.quantile(0.3)) == pytest.approx(
            1.3 * statistics.NormalDist().inv_cdf(0.3), abs=1e-15)


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_a_distributions_own_frames_are_read_as_its_own(family):
    """Science § 5.1, § 5.4; `engines/vibration-frames.md` § 9.1: a
    distribution's own four frames, weighted for it, are its own rule --
    exact to degree 7, ``own``; the same frames with the relaxed geometry
    prepended at weight 0, as an even rule's set is written, are its own
    only given the weights they were written with (``written``).  THE
    SILENT FAILURE: a set re-weighted to its own state said to be placed
    for another motion."""
    d = _FAMILIES[family]()
    x, w = d.gauss_rule(4)
    own = d.weights_at(x)
    assert (own.own, own.exact_to_degree) == (True, 7)
    assert own.weights == pytest.approx(w, abs=1e-12)
    points, written = np.concatenate(([0.0], x)), np.concatenate(([0.0], w))
    assert d.weights_at(points, written=written).own
    assert not d.weights_at(points).own


def test_one_quantums_middle_is_at_rest():
    """By symmetry half of one quantum's chance lies on each side of rest:
    its quantile at one half is 0.  THE SILENT FAILURE: 0.361 zero-point
    widths."""
    assert float(dist.Level(1).quantile(0.5)) == 0.0


def test_a_thermal_state_with_a_quantum_taken_is_its_list_distribution():
    """Science § 2e.1: one quantum taken from a thermal state of ``nbar = 1``, ``p_n =
    (n + 1)/2^(n+2)`` -- ``<x^2> = 2<n> + 1 = 5``, ``<x^4> = 3(2<n^2> + 2<n>
    + 1) = 63`` (``<n^2> = 8``).  THE SILENT FAILURE: the removal drawn as a
    photon added, ``<x^2> = 7``."""
    d = dist.of_state(states.thermal(mean_quanta=1.0).remove_quantum().on(_ANY))
    assert (d.moment(2), d.moment(4)) == pytest.approx((5.0, 63.0), rel=1e-12)


def test_a_typed_lists_rounding_speck_below_zero_is_zero():
    """Science § 2e.6: a chance a person's list holds just below 0, within
    the one epsilon of its largest, is a rounded 0 -- ``[0.5, 0.5,
    -1e-18]`` the list of levels 0 and 1, ``<x^2> = 2``.  THE SILENT
    FAILURE: a list from a person's own arithmetic refused for its
    rounding."""
    speck = dist.LevelList([0.5, 0.5, -1e-18])
    assert speck.moment(2) == pytest.approx(2.0, rel=1e-15)


def test_the_oscillators_own_numbers_are_the_constants():
    """Science § 2a.1: ``hbar omega = h c nu``, 0.1239842 eV for 1000 cm-1,
    from CODATA; the temperature whose thermal state holds half a quantum,
    ``h c nu / (k_B ln 3)``, 1309.6 K.  THE SILENT FAILURE: a quantum's
    energy off by ``2 pi``, ``h`` read for ``hbar``."""
    from scipy import constants as codata
    o = Oscillator(1000.0)
    assert o.quantum_ev == pytest.approx(
        codata.h * codata.c * 100.0 * 1000.0 / codata.e, rel=1e-9)
    assert o.bose_einstein_temperature(0.5) == pytest.approx(
        codata.h * codata.c * 100.0 * 1000.0 / (codata.k * math.log(3.0)),
        rel=1e-9)


def test_a_mean_number_of_quanta_names_its_temperature():
    """Science § 2a.1, `engines/vibration-frames.md` § 9.1: the temperature
    whose thermal state holds ``n`` quanta, ``T = h c nu / (k_B ln(1 +
    1/n))`` -- one quantum of a 1000 cm-1 mode at ``h c nu / (k_B ln 2)``,
    2075.7 K by CODATA -- 0 K for none; back through Bose and Einstein it is
    ``n``.  THE SILENT FAILURE: a pumped mode's set written at a temperature
    that holds other quanta than it was pumped to."""
    from scipy import constants as codata
    hc_nu = codata.h * codata.c * 100.0 * 1000.0
    t = oscillator.bose_einstein_temperature(1000.0, 1.0)
    assert t == pytest.approx(hc_nu / (codata.k * math.log(2.0)), rel=1e-8)
    assert oscillator.bose_einstein_mean_quanta(1000.0, t) == pytest.approx(
        1.0, rel=1e-12)
    assert oscillator.bose_einstein_temperature(1000.0, 0.0) == 0.0


def test_the_ladder_operators_are_the_sciences():
    """Science § 2d.2: on the levels 2 to 4, the lowering operator
    ``<n-1|b|n> = sqrt(n)``, the raising operator its transpose, ``<n+1|
    b^dagger|n> = sqrt(n+1)``, and the position their sum.  THE SILENT
    FAILURE: a ladder one level off, every level's averages shifted."""
    b = np.array([[0.0, math.sqrt(3.0), 0.0], [0.0, 0.0, 2.0],
                  [0.0, 0.0, 0.0]])
    assert oscillator.lowering_matrix(2, 5) == pytest.approx(b, abs=1e-15)
    assert oscillator.raising_matrix(2, 5) == pytest.approx(b.T, abs=1e-15)
    assert oscillator.position_matrix(2, 5) == pytest.approx(b + b.T,
                                                             abs=1e-15)


@pytest.mark.parametrize("values, weights, expected", [
    ([1.0, 2.0, 3.0, 4.0], [0.25] * 4, (2.5, math.sqrt(5.0 / 3.0) / 2.0)),
    ([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4], (3.0, math.sqrt(0.3 / 0.7))),
    ([1.0, 2.0, 3.0, 4.0], [0.0, 1 / 3, 1 / 3, 1 / 3],
     (3.0, 1.0 / math.sqrt(3.0))),
    ([math.nan, 2.0, 3.0, 4.0], [0.0, 1 / 3, 1 / 3, 1 / 3],
     (3.0, 1.0 / math.sqrt(3.0))),
], ids=["equal weights", "weights of their own", "one value at weight 0",
        "a value missing at weight 0"])
def test_a_means_error_is_the_spread_at_the_weights(values, weights,
                                                    expected):
    """Science § 5.5, `engines/vibration-frames.md` § 9.1: the mean over
    independent units at the weights given, ``sum_i w_i v_i``, and its
    error, ``sqrt(sum_i w_i^2)`` times their spread at the weights,
    ``sqrt(sum_i w_i (v_i - v)^2 / (1 - sum_i w_i^2))`` -- 1, 2, 3, 4: at
    equal weights 2.5 and ``sqrt(5/3)/2``, the spread ``ddof`` 1 over
    ``sqrt(N)``; at 0.1 ... 0.4, 3 and ``sqrt(0.3/0.7)``; a value at weight
    0, as a set's frame 0, adding nothing -- the other three's own, its
    value whatever it is, a missing one too.  THE SILENT FAILURES: an error
    13 % small with ``ddof`` 0; weights counted out by the function, not
    the set's; a frame left out at weight 0 turning the mean into NaN."""
    mean, error = sampling.mean_with_error(values, weights)
    assert (float(mean), float(error)) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("weights, refused", [
    ([0.5, 0.5, 0.5, 0.5], "the weights sum to 2.0, not 1 within 1e-06"),
    ([0.5, 0.5], "the weights are one per value, 4 here.*got 2 values$"),
    ([0.25, 0.25, 0.75, -0.25], "a weight is 0 or more.*got -0.25$"),
    ([0.25, 0.25, 0.25, float("nan")], "the weights are finite numbers"),
    ("equal", "the weights are one per value, 4 here.*got a str$"),
], ids=["a sum off 1", "not one per value", "a weight below 0",
        "a weight that is no number", "a word for weights"])
def test_weights_the_statistics_cannot_take_are_refused_by_name(weights,
                                                               refused):
    """`engines/vibration-frames.md` § 9.1: the statistics take the weights
    given -- one finite number per value, 0 or more, summing to 1 -- and
    count out none of their own; any other refused by name (the user,
    2026-11-26: "weight ... will be checked for normalization").  THE
    SILENT FAILURE: a mean at weights that do not sum to 1, its error
    scaled with them."""
    with pytest.raises(ValueError, match=refused):
        sampling.mean_with_error([1.0, 2.0, 3.0, 4.0], weights)


def test_the_statistics_weigh_by_the_containers_rule():
    """`model/structure.md` § 2.2f: the statistics, travelling in a job's
    bundle without the structure, restate the container's tolerance for
    the weights' sum -- held one here.  THE SILENT FAILURE: a set the
    container accepts refused by its own analyses, or the reverse."""
    from molbuilder.frameset import WEIGHT_SUM_TOLERANCE
    assert sampling.WEIGHT_SUM_TOLERANCE == WEIGHT_SUM_TOLERANCE


@pytest.mark.parametrize("spreads", [
    lambda: sampling.mean_with_error([1.0], [1.0])[1:],
    lambda: sampling.mean_with_error([1.0, 2.0], [0.0, 1.0])[1:],
    lambda: sampling.even_odd_parts([1.0, -1.0], [0.5, 0.5]),
], ids=["one snapshot", "one of weight above 0", "one mirrored pair"])
def test_fewer_than_two_units_leave_no_spread(spreads):
    """Science § 5.5: a spread needs two units of weight above 0 -- one
    snapshot, one beside a frame of weight 0, or one pair, leaves none:
    NaN, and no warning.  THE SILENT FAILURE: an error of 0 read as an
    exact mean."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert all(math.isnan(float(v)) for v in spreads())


def test_a_stream_is_numpys_generator_seeded_with_its_key():
    """Science § 2d.6: the stream keyed ``(0, 1, 0)`` -- seed 0 a seed like
    any other -- is numpy's generator seeded with that key, the numbers
    ``start ...`` of it the same however many are asked.  THE SILENT
    FAILURE: seed 0 read as no seed, the frames drawn afresh each time."""
    assert sampling.normal_stream((0, 1, 0), 3) == pytest.approx(
        np.random.default_rng([0, 1, 0]).standard_normal(3), abs=0.0)
    assert sampling.normal_stream((0, 1, 0), 2, start=1) == pytest.approx(
        np.random.default_rng([0, 1, 0]).standard_normal(3)[1:], abs=0.0)


@pytest.mark.parametrize("chain", [
    states.level(3),
    states.thermal(mean_quanta=1.0).add_quantum(),
    states.thermal(mean_quanta=1.0).add_quantum().remove_quantum(),
], ids=["3 quanta", "one photon, warm", "the list's route"])
def test_draws_from_one_normal_number_follow_the_distribution(chain):
    """Science § 2d.5, § 2e.3: a draw is the point where ``F`` reaches
    ``Phi(z)`` -- 1000 draws pass a Kolmogorov test against ``F``; the
    mirror ``-z`` gives exactly ``-x``, every distribution symmetric.  The
    silent failure: draws from another distribution than the frames'."""
    d = dist.of_state(chain.on(_ANY))
    z = sampling.normal_stream((7, 1, 0), 1000)
    x = np.sort(d.draw(z))
    ecdf = np.arange(1, x.size + 1) / x.size
    assert np.max(np.abs(ecdf - d.cdf(x))) < 1.63 / math.sqrt(x.size)
    assert np.array_equal(d.draw(-z), -d.draw(z))


def test_the_classical_forms_have_no_zero_point_motion():
    """Science § 2b.3, § 2e.5: the classical bell curve at ``T`` has
    variance ``k_B T / omega^2``, in ``x`` ``2 k_B T / hbar omega``; at 0 K
    it is a point at rest.  The silent failure: a classical comparison
    carrying the zero-point width."""
    o = Oscillator(200.0)
    bell = dist.of_state(states.classical_thermal(temperature_k=300.0).on(o))
    assert bell.moment(2) == pytest.approx(2.0 * 300.0 / (200.0 * CM1_KELVIN),
                                           rel=1e-12)
    rest = dist.of_state(states.classical_thermal(temperature_k=0.0).on(o))
    assert (rest.moment(2), rest.cdf(-1e-9), rest.cdf(0.0)) == (0.0, 0.0, 1.0)


@pytest.mark.parametrize("make, refused", [
    (lambda: states.classical_thermal(300.0).add_quantum(), "counts no quanta"),
    (lambda: states.ground().remove_quantum(), "holds none"),
    (lambda: states.Chain((states.Step("ground"), states.Step(
        "remove_quantum"))).reduced(), "holds none"),
    (lambda: states.thermal().remove_quantum().on(
        Oscillator(200.0), temperature_k=0.0).populations, "holds none"),
    (lambda: dist.of_state(states.thermal(mean_quanta=1.0)),
     "a chain placed on an oscillator"),
    (lambda: states.thermal(temperature_k=300.0).on(_ANY, temperature_k=77.0),
     "conflicts"),
    (lambda: states.thermal(mean_quanta=1.0).on(_ANY, temperature_k=77.0),
     "would name another"),
    (lambda: states.thermal(mean_quanta=1.0).on(_ANY, precision=0.0),
     "precision is a share"),
    (lambda: dist.LevelList([0.0, 0.0, 0.0]), "every chance given is 0"),
    (lambda: sampling.normal_stream(5, 3), "a stream's key is a sequence"),
    (lambda: sampling.normal_stream((1.5, 7, 0), 3), "a whole number"),
    (lambda: oscillator.lowering_matrix(5, 3), "6 or more"),
    (lambda: states.thermal(mean_quanta=0.0).remove_quantum(),
     "of no mean quanta"),
    (lambda: states.thermal(temperature_k=300.0, mean_quanta=1.0), "not both"),
    (lambda: states.thermal().on(_ANY), "names no temperature"),
    (lambda: Oscillator(0.0), "a positive frequency"),
    (lambda: Oscillator(True), "a positive frequency"),
    (lambda: states.classical_thermal(temperature_k=300.0).on(_ANY).populations,
     "counts no quanta"),
    (lambda: dist.LevelList([0.5, 0.6, -0.1]), "0 or more"),
    (lambda: dist.Gaussian(-1.0), "spread is 0 or more"),
    (lambda: dist.Gaussian(True), "spread is 0 or more"),
    (lambda: dist.QuantumAdded(-1.0), "0 or more"),
    (lambda: Oscillator(np.array([100.0, 200.0])), "one frequency"),
], ids=["a quantum on a classical start", "a quantum from the ground state",
        "a chain written with a quantum from the ground state",
        "a quantum from a thermal state placed at 0 K", "a chain asked for its numbers",
        "a temperature given twice, two values", "a temperature given to a "
        "thermal state of mean quanta", "a precision of nothing", "a list of no chance",
        "a stream key of one number", "a stream key of a fraction",
        "a window of levels backwards", "a quantum from a thermal state of none",
        "a thermal state named twice", "a temperature left open, given none",
        "an oscillator of no frequency", "an oscillator of a truth value",
        "a list of a classical state", "a list with a chance below 0",
        "a bell curve of a negative spread", "a bell curve of a truth value",
        "a photon on a thermal state of negative quanta",
        "an oscillator of two frequencies"])
def test_a_step_that_is_none_is_refused_by_name(make, refused):
    """Science § 2e.2; `engines/vibration-frames.md` § 9.1: a classical
    state counts no quanta, and has no list; the ground state, or a thermal
    state at 0 K or of no mean quanta, has none to give; a chain has numbers
    once it is placed; a thermal state is named once, and given no other
    temperature where it is placed; an open temperature is given there; a
    precision is a share; a list holds a chance, none below 0; a spread and
    a mean number of quanta are 0 or more; an oscillator has one frequency;
    a stream's key is whole numbers; a window of levels runs upward.  The silent failure: a number
    for a state that is none, or a value given and silently not used."""
    with pytest.raises(ValueError, match=refused):
        make()


@pytest.mark.parametrize("chain, words", [
    (states.thermal(temperature_k=300.0).add_quantum(),
     "thermal at 300 K (0.621 mean quanta); one quantum added"),
    (states.thermal(mean_quanta=1.0), "thermal, 1 mean quantum"),
    (states.ground(), "the ground state"),
    (states.level(1), "1 quantum exactly"),
    (states.classical_thermal(temperature_k=300.0),
     "the classical bell curve at 300 K, no zero-point motion"),
    (states.thermal(mean_quanta=1.0).remove_quantum(),
     "thermal, 1 mean quantum; one quantum removed"),
], ids=["a photon on a warm mode", "a thermal state of one quantum",
        "the ground state", "one quantum", "a classical thermal state",
        "a quantum taken"])
def test_a_states_words_are_its_steps(chain, words):
    """`engines/vibration-frames.md` § 9.1: each step says itself -- a thermal state
    at 300 K on a 200 cm-1 mode, 0.621 mean quanta (Bose--Einstein), then a
    quantum added; a thermal state of one mean quantum in the singular
    (`web/spectra.md` § 4.3); the ground state; a level; the classical
    bell curve; a quantum taken.  The silent failure: a set whose words name another
    state than its frames sample."""
    assert chain.on(Oscillator(200.0)).words == words


# --------------------------------------------------------------------- #
#  A distribution in another unit, and the records a set keeps of a     #
#  chain or a distribution (`engines/vibration-frames.md` § 2, § 5,     #
#  § 9.1)                                                               #
# --------------------------------------------------------------------- #

#: THE SCALE A MODE PUTS ITS STATE IN: about a 200 cm-1 mode's zero-point
#: amplitude, amu^1/2.Å -- ``Q = Q_zp x`` (`engines/vibration-frames.md`
#: § 2).
_SCALE = 0.29


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_a_distribution_in_another_unit_is_its_change_of_variable(family):
    """`engines/vibration-frames.md` § 2, § 9.1 (``d.scaled(s)``): the
    distribution of ``y = s x`` -- its density ``P_x(y / s) / s``, its
    distribution function ``F_x(y / s)``, its averages ``s^k <x^k>``; its
    quantile where its ``F`` reaches the chance; its own Gauss rule exact to
    ``2m - 1`` against those averages; a draw where its ``F`` reaches
    ``Phi(z)``, ``-z`` giving exactly ``-y``; and the weights of points
    placed elsewhere the base's for the same points in ``x`` -- a weight
    has no unit.  THE SILENT FAILURES: a mode's density off by its
    ``Q_zp`` (a density that no longer integrates to 1), or its frames left
    in ``x``, every geometry moved by ``1 / Q_zp`` too little."""
    base = _FAMILIES[family]()
    d = base.scaled(_SCALE)
    y = np.linspace(-6.0, 6.0, 25) * _SCALE
    assert d.pdf(y) == pytest.approx(base.pdf(y / _SCALE) / _SCALE,
                                     rel=1e-12, abs=1e-300)
    assert d.cdf(y) == pytest.approx(base.cdf(y / _SCALE), abs=1e-15)
    for k in range(9):
        assert d.moment(k) == pytest.approx(_SCALE ** k * base.moment(k),
                                            rel=1e-12, abs=1e-300), k
    for p in (0.05, 0.3, 0.5, 0.95):
        assert float(d.cdf(d.quantile(p))) == pytest.approx(p, abs=1e-14)
    x, w = d.gauss_rule(4)
    for k in range(8):
        terms = w * x ** k                  # odd k: terms cancelling to 0
        assert float(terms.sum()) == pytest.approx(
            d.moment(k), rel=1e-11, abs=1e-14 * float(np.abs(terms).sum()))
    z = sampling.normal_stream((5, 1, 0), 200)
    assert np.max(np.abs(d.cdf(d.draw(z)) - _phi(z))) < 1e-12
    assert np.array_equal(d.draw(-z), -d.draw(z))
    points = np.array([-1.5, -0.4, 0.0, 0.7, 2.0]) * _SCALE
    assert d.weights_at(points).weights == pytest.approx(
        base.weights_at(points / _SCALE).weights, abs=1e-12)


def test_a_bell_curve_in_another_unit_is_the_bell_curve_of_that_spread():
    """`engines/vibration-frames.md` § 9.1: a bell curve of spread ``sigma``
    in ``x`` is, in ``y = s x``, the bell curve of spread ``s sigma`` --
    its density, its three frames at ``0, +-sqrt(3) s sigma``, a draw ``s
    sigma z`` -- and a distribution scaled twice is one scaled by the two
    scales' product, its record nesting once.  THE SILENT FAILURE: a mode's
    state scaled by ``Q_zp`` twice, its frames ``Q_zp`` times too far."""
    sigma = 1.3
    d = dist.Gaussian(sigma).scaled(_SCALE)
    y = np.linspace(-3.0, 3.0, 13)
    width = _SCALE * sigma
    assert d.pdf(y) == pytest.approx(
        np.exp(-0.5 * (y / width) ** 2) / (math.sqrt(2.0 * math.pi) * width),
        rel=1e-13)
    x, w = d.gauss_rule(3)
    assert np.sort(x) == pytest.approx(
        [-math.sqrt(3.0) * width, 0.0, math.sqrt(3.0) * width], abs=1e-15)
    assert d.draw(np.array([1.0, -2.0])) == pytest.approx(
        [width, -2.0 * width], rel=1e-15)
    twice = dist.Gaussian(sigma).scaled(2.0).scaled(_SCALE)
    assert twice.pdf(y) == pytest.approx(
        dist.Gaussian(2.0 * width).pdf(y), rel=1e-13)
    assert twice.to_record() == {"family": "Scaled", "scale": 2.0 * _SCALE,
                                 "base": {"family": "Gaussian",
                                          "sigma": sigma}}


@pytest.mark.parametrize("scale", [0.0, -0.29, math.inf, True],
                         ids=["no scale", "a scale below zero", "an "
                              "endless scale", "a truth value"])
def test_a_scale_that_is_none_is_refused_by_name(scale):
    """`engines/vibration-frames.md` § 9.1: a scale is a positive number --
    0, one below it, an endless one or a truth value refused by name.  THE
    SILENT FAILURE: a mode of no zero-point amplitude put in a unit where
    every frame sits at rest, or mirrored."""
    with pytest.raises(ValueError, match="a scale is a positive number"):
        dist.Level(2).scaled(scale)


@pytest.mark.parametrize("distribution, words", [
    (lambda: dist.Gaussian(1.3), "a bell curve of spread 1.3"),
    (lambda: dist.Gaussian(0.0), "a point at rest at 0"),
    (lambda: dist.Level(1), "1 quantum exactly"),
    (lambda: dist.Level(3), "3 quanta exactly"),
    (lambda: dist.QuantumAdded(0.62),
     "one quantum added to a thermal state of 0.62 mean quanta"),
    (lambda: dist.LevelList([0.25, 0.5, 0.25]),
     "a list over 3 levels, 1 mean quantum"),
    (lambda: dist.LevelList([1.0]), "a list over 1 level, 0 mean quanta"),
    (lambda: dist.Gaussian(1.0).scaled(1.3),
     "a bell curve of spread 1, scaled by 1.3"),
], ids=["a bell curve", "a point at rest", "1 quantum", "3 quanta",
        "one quantum on a warm mode", "a list over 3 levels",
        "a list over 1 level", "scaled"])
def test_a_distributions_words_say_its_family(distribution, words):
    """`engines/vibration-frames.md` § 9.1's ``words``: a family says what it
    is, its count's noun chosen on the figure written -- *1 quantum*, *1
    level*, *3 levels*, *0 mean quanta* -- a ``Scaled`` its base's and its
    scale.  The silent failure: a set's assumptions row naming another
    distribution than its frames were placed for."""
    assert distribution().words == words


#: ONE OF EACH FAMILY A SET RECORDS (`engines/vibration-frames.md` § 5's
#: ``info.state``, § 9.1): the four families, a list given unnormalised,
#: and one in another unit.
_RECORDED_FAMILIES = {
    **_FAMILIES,
    "a list given unnormalised": lambda: dist.LevelList([2.0, 1.0, 1.0]),
    "3 quanta in another unit": lambda: dist.Level(3).scaled(_SCALE),
}


@pytest.mark.parametrize("family", list(_RECORDED_FAMILIES))
def test_a_distributions_record_makes_it_again(family):
    """`engines/vibration-frames.md` § 5, § 9.1: a distribution's record --
    its family and its numbers, as a set's sidecar keeps it (JSON) -- makes
    it again (``dist.from_record``): the same family, the same density,
    averages and own frames, the same record.  THE SILENT FAILURE: a set's
    state read back as another distribution than the frames were placed
    for -- a list's chances shifted, a photon's mean swapped for its
    spread."""
    import json
    d = _RECORDED_FAMILIES[family]()
    again = dist.from_record(json.loads(json.dumps(d.to_record())))
    assert type(again) is type(d)
    assert again.to_record() == d.to_record()
    x = np.linspace(-5.0, 5.0, 21)
    assert np.array_equal(again.pdf(x), d.pdf(x))
    assert [again.moment(k) for k in (2, 4, 6)] == [d.moment(k)
                                                    for k in (2, 4, 6)]
    assert np.array_equal(again.gauss_rule(3)[0], d.gauss_rule(3)[0])


class _OwnBell(dist.Distribution):
    """A person's own distribution, by a class of their own."""

    def moment(self, k):
        return dist.Gaussian(1.0).moment(k)


@pytest.mark.parametrize("record, refused", [
    ({"family": "Poisson", "mean": 3.0}, "names its family"),
    ("Gaussian", "names its family"),
    ({"family": "Gaussian"}, "a Gaussian record states its sigma"),
    ({"family": "Scaled", "base": {"family": "Level", "n": 2}},
     "a Scaled record states its scale"),
    ({"family": "Gaussian", "sigma": -1.0}, "spread is 0 or more"),
    ({"family": "Level", "n": 2.5}, "a level is a whole number"),
    ({"family": "LevelList", "populations": [0.0, 0.0]},
     "every chance given is 0"),
], ids=["a family it does not know", "no mapping", "a number missing",
        "a scale missing", "a spread below zero", "half a level",
        "a list of no chance"])
def test_a_record_that_makes_no_distribution_is_refused_by_name(record,
                                                                refused):
    """`engines/vibration-frames.md` § 9.1: a record makes its family by
    the family's own checks -- what names no family it knows, a family's
    record missing a number, and a number its family refuses, each refused
    by name.  THE SILENT FAILURE: a set's state read back as a default where
    its record names none."""
    with pytest.raises(ValueError, match=refused):
        dist.from_record(record)


def test_a_persons_own_class_records_nothing_to_make_again():
    """`engines/vibration-frames.md` § 5, § 9.1: a distribution of a
    person's own class can be drawn from and placed, but states no record
    to be made again from -- refused by name, naming its class and the
    families a set records.  THE SILENT FAILURE: a set written that reads
    back with no state, or with another."""
    with pytest.raises(ValueError, match="a _OwnBell states no record to be "
                                         "made again from"):
        _OwnBell().to_record()


#: CHAINS AS A SET RECORDS THEM (`engines/vibration-frames.md` § 5): each
#: start, open or named, and events after it, as given -- never reduced.
_CHAINS = {
    "the ground state": states.ground(),
    "3 quanta": states.level(3),
    "a level, a quantum added, as given": states.level(1).add_quantum(),
    "thermal at a temperature": states.thermal(temperature_k=300.0),
    "thermal of a mean": states.thermal(mean_quanta=0.5),
    "thermal, open": states.thermal(),
    "classical at a temperature": states.classical_thermal(temperature_k=77.0),
    "classical, open": states.classical_thermal(),
    "a photon, then a quantum taken": states.thermal(temperature_k=300.0)
    .add_quantum().remove_quantum(),
}


@pytest.mark.parametrize("name", list(_CHAINS))
def test_a_chains_record_makes_it_again_as_given(name):
    """`engines/vibration-frames.md` § 5 (``info.state``'s chain, "the chain
    as given, its steps in order"): a chain's record -- its steps, each
    ``{"step", "values"}``, as a set's sidecar keeps it (JSON) -- makes
    the chain again (``Chain.from_record``), the same chain as given, a
    start left open still open.  THE SILENT FAILURES: an open start read
    back naming a temperature, or a level and its event read back as
    another level, its record no longer the chain the person wrote."""
    import json
    chain = _CHAINS[name]
    record = json.loads(json.dumps(chain.to_record()))
    assert [step["step"] for step in record] == [s.name for s in chain.steps]
    again = states.Chain.from_record(record)
    assert again == chain
    assert again.temperature_open == chain.temperature_open


@pytest.mark.parametrize("record, refused", [
    ([], "a chain's record is its steps in order"),
    ("thermal", "a chain's record is its steps in order"),
    ([{"step": "add_quantum", "values": {}}],
     "a chain starts with ground, level, thermal or classical_thermal"),
    ([{"step": "ground", "values": {}}, {"step": "thermal", "values": {}}],
     "after its start a chain holds add_quantum and remove_quantum"),
    ([{"step": "thermal", "values": {"temperature_k": -5.0}}],
     "a number of kelvin"),
    ([{"step": "level", "values": {"quanta": 1.5}}],
     "a level is a whole number"),
    ([{"step": "classical_thermal", "values": {}},
      {"step": "add_quantum", "values": {}}], "counts no quanta"),
    ([{"step": "ground", "values": {}},
      {"step": "remove_quantum", "values": {}}], "holds none"),
    ([{"step": "thermal", "values": {"temperatur_k": 300.0}}],
     "thermal takes temperature_k or mean_quanta; got 'temperatur_k'"),
    ([{"step": "thermal", "values": [300.0]}],
     "a start's values are a mapping"),
    ([{"step": "level", "values": {"quanta": 2, "spin": 1}}],
     "level takes quanta; got 'spin'"),
    ([{"step": "ground", "values": {}},
      {"step": "add_quantum", "values": {"n": 2}}], "each of no value"),
], ids=["no steps", "no list", "an event first", "a second start",
        "a temperature below zero", "half a level", "a quantum on a "
        "classical start", "a quantum from the ground state",
        "a key the start does not take", "values that are no mapping",
        "a level's extra key", "an event holding a value"])
def test_a_record_that_makes_no_chain_is_refused_by_name(record, refused):
    """`engines/vibration-frames.md` § 9.1: a chain's record is made again
    by the starts and events themselves -- what is no list of steps, a
    first step that is no start, a later one that is no event or holds a
    value, a start's values that are no mapping or hold a key it does not
    take, and what a start or an event refuses, each refused by name.  THE
    SILENT FAILURE: a set's state read back as a chain no person could have
    written, or a misspelled key dropped and the start read as open."""
    with pytest.raises(ValueError, match=refused):
        states.Chain.from_record(record)


@pytest.mark.parametrize("steps, refused", [
    ((states.Step("thermal", (("mean_quanta", 1.0),)),
      states.Step("add_quantm")), "after its start a chain holds add_quantum"),
    ((states.Step("ground"), states.Step("add_quantum"),
      states.Step("thermal", (("temperature_k", 300.0),))),
     "after its start a chain holds add_quantum"),
    ((states.Step("classical_thermal"), states.Step("add_quantum")),
     "counts no quanta"),
    ((states.Step("warm"),), "a chain starts with ground, level, thermal"),
    ((states.Step("thermal", (("temperatur_k", 300.0),)),),
     "thermal takes temperature_k or mean_quanta"),
    ((states.Step("level", (("quanta", -1),)),), "a level is a whole number"),
    ((), "a chain is a start and the events after it"),
    ((states.Step("ground"), states.Step("add_quantum", (("n", 2),))),
     "each of no value"),
    ((states.Step("thermal", (("mean_quanta", 5.0), ("temperature_k", 300.0))),),
     "named by a temperature or by a mean number of quanta, not both"),
], ids=["a misspelled event", "a start after an event",
        "a quantum on a classical start", "no start", "a misspelled key",
        "a level below zero", "no steps", "an event holding a value",
        "a thermal start naming both"])
def test_a_chain_built_by_hand_is_checked_as_its_steps_check(steps, refused):
    """`engines/vibration-frames.md` § 9.1 (`Step`, `Chain`), science
    § 2e.2: a chain built by hand from `Step`s is checked as the starts and
    the events check them -- a first step that is no start, a key it does
    not take or a value it refuses, a thermal start naming both a
    temperature and a mean, a later step that is no event or holds a value,
    a quantum on a classical start -- refused by name.  THE SILENT FAILURES:
    a misspelled event computed and said as a quantum removed; a start
    after an event said in the words but ignored in the numbers; a mean
    said in the words and another placed."""
    with pytest.raises(ValueError, match=refused):
        states.Chain(steps)


@pytest.mark.parametrize("chain, refused", [
    (states.thermal(temperature_k=300.0), "names its own temperature or mean"),
    (states.thermal(mean_quanta=2.0), "names its own temperature or mean"),
    (states.ground(), "ground takes no temperature"),
    (states.level(2), "level takes no temperature"),
], ids=["a start at its own temperature", "a start of its own mean",
        "the ground state", "a level"])
def test_a_temperature_fills_only_an_open_start(chain, refused):
    """`engines/vibration-frames.md` § 9.1 (`with_temperature`): a
    temperature fills an open start -- *thermal at 77 K; one quantum added*
    -- and is refused by name for a start that names its own or takes none.
    THE SILENT FAILURE: a set's 300 K start quietly moved to another
    temperature, or a level given one it has no use for."""
    assert states.thermal().add_quantum().with_temperature(77.0).words == \
        "thermal at 77 K; one quantum added"
    with pytest.raises(ValueError, match=refused):
        chain.with_temperature(77.0)


@pytest.mark.parametrize("chain, says, open_", [
    (states.thermal(), ("thermal", "open"), True),
    (states.classical_thermal(), ("classical", "open"), True),
    (states.thermal().add_quantum(), ("thermal", "open", "one quantum added"),
     True),
    (states.thermal(temperature_k=300.0).add_quantum(),
     ("thermal at 300 K", "one quantum added"), False),
], ids=["thermal, open", "classical, open", "a photon on an open start",
        "a photon on a named one"])
def test_a_chains_words_say_an_open_start_is_open(chain, says, open_):
    """`engines/vibration-frames.md` § 2, § 9.1: a chain in words, before
    it is placed -- each step saying itself, an open start saying its
    temperature is open, never a number, a start naming its temperature
    saying it (*thermal at 300 K; one quantum added*).  THE SILENT FAILURE:
    a chain whose temperature is taken where it is placed said as if it
    named one."""
    words = chain.words
    assert all(w in words for w in says), words
    assert (" K" not in words) == open_, words

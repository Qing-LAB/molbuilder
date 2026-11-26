"""A mode's state built in steps, held to the science's closed forms
(`science/vibrational-averaging.md` § 2e; `engines/vibration-frames.md`
§ 9.1): the lists each chain of steps makes, what the steps do to the mean,
when two chains are one state, where a list is cut, the worked chain's
frames, the exact families against the list's route, the draws, the search
for a point of a given chance, the statistics of snapshots, the refusals and
the words.  Every expectation is a formula of the science or a
reference outside the toolkit (the standard library's normal quantile,
numpy's generator); where two of the toolkit's routes are held to each other
-- a closed family and the list (`engines/vibration-frames.md` § 9.4,
decision 1(b)) -- the test says so."""
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
    assert state.steps[0].get("temperature_k") == start.start.get(
        "temperature_k")
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
    """`engines/vibration-frames.md` § 9.4 decision 1(b); science § 2e.3:
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
    an odd count refused by name (a set's frame 0 left in).  The silent
    failure: an odd part twice its size, or the pairs taken across the
    energies."""
    assert sampling.even_odd_parts([1.0, -1.0, 3.0, -3.0, 2.0, -2.0]) == \
        pytest.approx((0.0, 1.0), abs=1e-15)
    assert sampling.even_odd_parts([1.0, 1.0, 3.0, 3.0, 2.0, 2.0]) == \
        pytest.approx((1.0, 0.0), abs=1e-15)
    per_energy = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [7.0, 70.0]])
    assert sampling.pair_means(per_energy) == pytest.approx(
        np.array([[2.0, 20.0], [6.0, 60.0]]), abs=1e-15)
    with pytest.raises(ValueError, match="two by two"):
        sampling.pair_means([0.0, 1.0, 3.0])


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


def test_a_means_error_is_the_spread_over_root_n():
    """Science § 5.5: the error of a mean over ``N`` independent units is
    their spread, ``ddof`` 1, over ``sqrt(N)`` -- 1, 2, 3, 4: 2.5 and
    ``sqrt(5/3)/2``.  THE SILENT FAILURE: an error 13 % small with ``ddof``
    0."""
    mean, error = sampling.mean_with_error([1.0, 2.0, 3.0, 4.0])
    assert (float(mean), float(error)) == pytest.approx(
        (2.5, math.sqrt(5.0 / 3.0) / 2.0), abs=1e-15)


@pytest.mark.parametrize("spreads", [
    lambda: sampling.mean_with_error([1.0])[1:],
    lambda: sampling.even_odd_parts([1.0, -1.0]),
], ids=["one snapshot", "one mirrored pair"])
def test_fewer_than_two_units_leave_no_spread(spreads):
    """Science § 5.5: a spread needs two units -- one snapshot, or one pair,
    leaves none: NaN, and no warning.  THE SILENT FAILURE: an error of 0
    read as an exact mean."""
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

"""A vibration's frame sets on the host: the one writer and its reader,
what a set assumes, and the weights its rows give for a state
(`engines/vibration-frames.md` § 4, § 5, § 5.1, § 6, § 7).

API-level, as a person's script and the web layer ask: each set is a pair on
disk -- written by the toolkit's one writer (`postprocess.write_frame_set`),
or made through the Structure API and the structure's one writer -- and read
back by `postprocess.read_frame_set` from its path; a set's rows are the
structure's own reader's (`frameset.read`), as `postprocess.frame_rows`
hands them.  Each expectation is the contract's: § 4's outermost move, § 5's
rows, refusals, what is said and what is read back, § 5.1's rule, § 7's
refusals, the science's closed forms (`science/vibrational-averaging.md`
§ 2b.3) for a set's derived rows, and the Gauss-Hermite rule from numpy
(``hermgauss``) for a set's own weights.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# the generic tools as a person's script reaches them (`engines/vibration-
# frames.md` § 9): a mode's state a chain or a distribution bound to it
from molbuilder.postprocess import dist, states

_H2 = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"


def _pair(tmp_path, rows=(), frame_rows=(), info=None, n=2, name="set.xyz"):
    """A pair of ``n`` frames of H2 at ``tmp_path / name``: the structure's
    ``rows`` and each frame's ``frame_rows`` -- each ``(name, value, unit,
    note)`` -- and ``info`` keys, written by the structure's one writer."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    s = Structure.from_xyz(_H2)
    s = s.with_frames([s.positions] * n)
    for name_, value, unit, note in rows:
        s.set_customized(name_, value, unit=unit, note=note)
    for f in range(n):
        for name_, value, unit, note in frame_rows:
            s.set_customized(name_, value(f) if callable(value) else value,
                             unit=unit, note=note, frame=f)
    for key, value in (info or {}).items():
        s.set_info(key, value)
    return StructureCodec().write(s, tmp_path / name)


# ----------------------------------------------------------------------- #
#  The weights for a set's rows at a state (§ 7)                           #
# ----------------------------------------------------------------------- #

#: A mode of 1000 cm-1 whose zero-point amplitude is 1 amu^1/2 Angstrom.
_NU, _QZP = 1000.0, 1.0


def _mode_set(m=3, temperature_k=300.0, with_temperature=True, *,
              extra=(), without=(), no_q_at=None, weight_scale=1.0):
    """A mode's own set's rows as `frameset.read` hands them, made through
    the Structure API as a script makes one: its frames at the m-point
    Gauss-Hermite rule's nodes for the thermal spread at ``temperature_k``,
    each weighted by the rule -- the set's own weights, times
    ``weight_scale``; ``extra`` structure rows ``(name, value)`` added, the
    structure rows named in ``without`` left out, frame ``no_q_at`` stating
    none."""
    from molbuilder.frameset import read
    from molbuilder.structure import Structure
    from molbuilder.oscillator import thermal_spread_amu12_ang
    sigma = thermal_spread_amu12_ang(_QZP, _NU, temperature_k)
    x, w = np.polynomial.hermite.hermgauss(m)
    # THE ROAD'S ORDER (`engines/vibration-frames.md` § 4): frame 0 at the
    # middle node, then each pair, nearest first, its positive side first.
    order = sorted(range(m), key=lambda i: (abs(x[i]), -x[i]))
    x, w = x[order], w[order]
    s = Structure.from_xyz(_H2)
    s = s.with_frames([s.positions] * m)
    stated = [("mode_index_1based", 7, "1"), ("frequency_cm1", _NU, None),
              ("zero_point_amplitude_amu12_ang", _QZP, "amu^1/2 angstrom")]
    if with_temperature:
        stated.append(("temperature_k", temperature_k, "K"))
    for name, value, unit in [*stated, *((n, v, "1") for n, v in extra)]:
        if name not in without:
            s.set_customized(name, value, unit=unit)
    for f in range(m):
        if f != no_q_at:
            s.set_customized("q_amu12_ang", math.sqrt(2.0) * sigma * x[f],
                             frame=f)
        s.set_customized("weight", weight_scale * w[f] / math.sqrt(math.pi),
                         unit="1", frame=f)
    return read(s)


#: The set's own temperature, ``_mode_set``'s ``temperature_k`` row.
_SETS_OWN = (300.0, "the set's temperature, 300 K (its temperature_k row)")


@pytest.mark.parametrize("m", [3, 5, 7])
def test_a_sets_rows_at_its_own_state_give_its_own_weights(m):
    """§ 7: at the set's own state -- an open thermal start taking the
    set's own ``temperature_k`` row, said first (*the set's temperature,
    300 K (its temperature_k row), for thermal at 300 K (… mean quanta)*) --
    the weights are the set's own, the rule's, exact to degree ``2m - 1``
    (science § 5.1), and nothing said of frames placed for another motion;
    the same state named by its temperature says nothing at all.  The
    silent failure: a re-average that disagrees with the record."""
    from molbuilder.postprocess import states
    from molbuilder.vibration_frames import weights_for_set
    rows = _mode_set(m)
    got = weights_for_set(rows, states.thermal())
    assert got.weights == pytest.approx(list(rows.weights), abs=1e-12)
    assert got.exact_to_degree == 2 * m - 1, got
    assert len(got.said) == 1, got.said
    assert got.said[0].startswith(_SETS_OWN[1]), got.said
    # THE STATE IT WAS TAKEN FOR, in its words: Bose and Einstein's mean at
    # the set's temperature on its 1000 cm-1 mode, by CODATA.
    assert got.said[0].endswith(f"thermal at 300 K ({_n_th(_NU, 300.0):.3g} "
                                f"mean quanta)"), got.said
    given = weights_for_set(rows, states.thermal(temperature_k=300.0))
    assert given.said == () and given.exact_to_degree == 2 * m - 1, given


def test_an_even_rules_set_at_its_own_state_says_nothing_of_another():
    """§ 7: an even rule's set -- frame 0, the relaxed geometry, prepended at
    weight 0 (§ 5) -- re-weighted to the state it was written for is that
    state's own rule: its written weights, and nothing of frames placed for
    another motion said.  The silent failure: a set's own average said to be
    extrapolated beyond its frames."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import Oscillator, dist, states
    from molbuilder.structure import Structure
    from molbuilder.vibration_frames import weights_for_set
    warm = states.thermal(temperature_k=300.0)
    # THE STATE'S OWN FOUR-POINT RULE, by the generic tools, in Q -- this
    # mode's Q_zp is 1 -- written as an even rule's set is (§ 4, § 5):
    # frame 0 the relaxed geometry at weight 0, then each pair, nearest
    # first, its positive side first.
    x, w4 = dist.of_state(warm.on(Oscillator(_NU))).gauss_rule(4)
    order = sorted(range(4), key=lambda i: (abs(x[i]), -x[i]))
    q, w = [0.0, *x[order]], [0.0, *w4[order]]
    s = Structure.from_xyz(_H2)
    s = s.with_frames([s.positions] * len(q))
    s.set_customized("frequency_cm1", _NU, unit="cm-1")
    s.set_customized("zero_point_amplitude_amu12_ang", _QZP,
                     unit="amu^1/2 angstrom")
    s.set_customized("temperature_k", 300.0, unit="K")
    for f in range(len(q)):
        s.set_customized("q_amu12_ang", float(q[f]), frame=f)
        s.set_customized("weight", float(w[f]), unit="1", frame=f)
    got = weights_for_set(read(s), warm)
    assert got.weights == pytest.approx(list(w), abs=1e-12)
    assert got.said == (), got.said


def test_a_state_beyond_the_frames_is_said_not_refused():
    """§ 9, § 7: a state reaching past the outermost frames -- a set written
    at 0 K re-weighted to 300 K -- is averaged and the share beyond said,
    never refused.  Three frames' blindness to the shape is said with it
    (science § 5.4).  The silent failure: a motion the frames never reached
    averaged from values extrapolated past them, unsaid -- or a person's
    question refused where an answer with its caveat stands."""
    from molbuilder.postprocess import states
    from molbuilder.vibration_frames import weights_for_set
    got = weights_for_set(_mode_set(3, temperature_k=0.0),
                          states.thermal(temperature_k=300.0))
    assert got.beyond > 0.0, got
    assert math.fsum(got.weights) == pytest.approx(1.0, abs=1e-12)
    assert any("beyond the outermost frames" in s for s in got.said), got
    assert any("blind to the state's shape" in s for s in got.said), got


@pytest.mark.parametrize("open_chain, named, words", [
    ("thermal", "thermal at 300 K", "thermal at 300 K ("),
    ("classical", "classical at 300 K",
     "the classical bell curve at 300 K, no zero-point motion"),
    ("a photon on it", "a photon at 300 K", "; one quantum added"),
], ids=["a thermal start", "a classical start", "a photon on a thermal "
        "start"])
def test_an_open_start_takes_the_sets_temperature_and_says_so_first(
        open_chain, named, words):
    """§ 2's table, § 7: an open start -- thermal, classical, whatever
    follows it -- is placed at the set's own ``temperature_k`` row, said
    first in § 7's words, naming the state it was taken for; its weights
    those of the same chain naming 300 K itself, which says nothing of it.
    The silent failure: a temperature the person never named used unsaid,
    or another than the set's."""
    from molbuilder.postprocess import states
    from molbuilder.vibration_frames import weights_for_set
    chains = {
        "thermal": states.thermal(), "thermal at 300 K":
        states.thermal(temperature_k=300.0),
        "classical": states.classical_thermal(), "classical at 300 K":
        states.classical_thermal(temperature_k=300.0),
        "a photon on it": states.thermal().add_quantum(),
        "a photon at 300 K": states.thermal(temperature_k=300.0)
        .add_quantum()}
    rows = _mode_set(5)
    taken = weights_for_set(rows, chains[open_chain])
    assert taken.said[0].startswith(_SETS_OWN[1]) and words in \
        taken.said[0], taken.said
    given = weights_for_set(rows, chains[named])
    assert not any("temperature_k row" in s for s in given.said), given.said
    assert taken.weights == pytest.approx(given.weights, abs=1e-12)


@pytest.mark.parametrize("rows, chain, refused", [
    ("snapshots", "at 300 K", "snapshot set is not re-weighted"),
    ("no mode", "at 300 K", "no mode's frequency"),
    ("a frame without q", "at 300 K", "no position along"),
    ("no temperature", "open", "name a temperature: the set states none"),
    ("no temperature", "classical, open", "the set states none"),
    ("none", "at 300 K", "no frame set's rows -- one structure"),
    ("one structure stating a row", "at 300 K",
     "no frame set's rows -- one structure"),
    ("weights off 1", "at 300 K",
     "the frames' weights sum to 1.1.*, not 1 within 1e-06: every frame's "
     "weight is required"),
    ("a path", "at 300 K", "not a str"),
], ids=["a snapshot set", "a set of no mode", "a frame stating no q",
        "a thermal open start, the set stating no temperature",
        "a classical open start, the set stating no temperature", "no rows",
        "one structure stating a row", "weights summing off 1",
        "a path, not rows"])
def test_a_set_the_state_cannot_weight_is_refused_by_name(rows, chain,
                                                          refused):
    """§ 7: refused by name -- a snapshot set (its frames have no single
    mode's coordinate and its weights are equal), a set of no mode, a frame
    stating no coordinate, an open start of a set stating no temperature
    (naming the temperature to give), rows whose own weights break their
    rule (`model/structure.md` § 2.2f: re-weighted from weights not summing
    to 1), what is no set's rows (named by its kind, never printed whole).
    A start of none -- n quanta -- needs none.  The silent failure: weights
    for a motion the frames cannot carry, or a temperature the person never
    named taken from nowhere."""
    from molbuilder.postprocess import states
    from molbuilder.vibration_frames import weights_for_set
    chain = {"at 300 K": states.thermal(temperature_k=300.0),
             "open": states.thermal(),
             "classical, open": states.classical_thermal()}[chain]
    if rows == "snapshots":
        got = _mode_set(extra=[("samples", 2)])
    elif rows == "no mode":
        got = _mode_set(without=("frequency_cm1",))
    elif rows == "a frame without q":
        got = _mode_set(no_q_at=0)
    elif rows == "no temperature":
        got = _mode_set(with_temperature=False)
    elif rows == "weights off 1":
        got = _mode_set(weight_scale=1.1)
    elif rows == "one structure stating a row":
        from molbuilder.frameset import read
        from molbuilder.structure import Structure
        lone = Structure.from_xyz(_H2)
        lone.set_customized("bond", "H-H")
        got = read(lone)
    elif rows == "a path":
        got = "F.transport.nc" * 1000
    else:
        got = None
    with pytest.raises(ValueError, match=refused) as said:
        weights_for_set(got, chain)
    # A REFUSAL NAMES WHAT IT WAS GIVEN, never prints it whole.
    assert len(str(said.value)) < 400, str(said.value)
    if rows == "no temperature":
        assert len(weights_for_set(got, states.level(1)).weights) == 3


# --------------------------------------------------------------------- #
#  The one writer and its reader (§ 5)                                   #
# --------------------------------------------------------------------- #

def _model(*, held=False):
    """The model molecule's vibration (`support.model_vibration`, the one
    builder) -- with ``held``, a fourth atom held still."""
    from support.model_vibration import model_vibration
    return model_vibration(held=held)


#: EVERY MODE THERMAL AT 300 K, a chain naming its own temperature.
_WARM = states.thermal(temperature_k=300.0)

#: A PERSON'S OWN LIST over levels 0, 1, 2 (`engines/vibration-frames.md`
#: § 2: a distribution of ``x`` given, taken as it is): ``<x^2> = sum_n p_n
#: (2n + 1) = 3``.
_LIST = dist.LevelList([0.25, 0.5, 0.25])


class _Mine(dist.Distribution):
    """A PERSON'S OWN DISTRIBUTION of ``x``, by a class of their own -- a bell
    curve of spread 1.2 -- drawn from and placed in memory, but no family a
    set records to bind again (`engines/vibration-frames.md` § 5, § 9.1's
    ``to_record``)."""

    def __init__(self):
        self._bell = dist.Gaussian(1.2)

    def pdf(self, x):
        return self._bell.pdf(x)

    def cdf(self, x):
        return self._bell.cdf(x)

    def moment(self, k):
        return self._bell.moment(k)

    def gauss_rule(self, nodes):
        return self._bell.gauss_rule(nodes)


#: THE RECIPES, each with the structure rows § 5.1 says its kind writes --
#: beside the provenance and assumptions every set states.
_RECIPES = {
    "thermal": (lambda v: v.mode(1).bind(_WARM).frames(3),
                {"sampling_rule": "gauss-hermite", "sampling_nodes": 3,
                 "quantum": 1, "mode_index_1based": 1, "temperature_k": 300.0},
                {"quanta", "result_frequency_cm1"}),
    "an even rule": (lambda v: v.mode(2).bind(_WARM).frames(2),
                     {"sampling_nodes": 2, "mode_index_1based": 2},
                     set()),
    "classical": (lambda v: v.mode(1).bind(states.classical_thermal(
                      temperature_k=300.0)).frames(3),
                  {"quantum": 0, "temperature_k": 300.0}, {"mean_quanta"}),
    "n quanta": (lambda v: v.mode(1).bind(states.level(2)).frames(3),
                 {"quanta": 2, "sampling_rule": "gauss"},
                 {"temperature_k"}),
    "mean quanta": (lambda v: v.mode(3).bind(states.thermal(
                        mean_quanta=0.4)).frames(3),
                    {"mean_quanta": 0.4}, {"temperature_k"}),
    # ONE PHOTON ON A WARM MODE (§ 5's example): the warm mode's temperature
    # its start's, quantum, its own Gauss rule, no level's quanta.
    "one photon on a warm mode": (
        lambda v: v.mode(1).bind(_WARM.add_quantum()).frames(3),
        {"sampling_rule": "gauss", "sampling_nodes": 3, "quantum": 1,
         "mode_index_1based": 1, "temperature_k": 300.0}, {"quanta"}),
    # A PERSON'S OWN LIST: no temperature, no quanta counted, neither
    # quantum nor classical said.
    "a person's list": (
        lambda v: v.mode(1).bind(_LIST).frames(3),
        {"sampling_rule": "gauss", "mode_index_1based": 1},
        {"quanta", "temperature_k", "quantum", "mean_quanta"}),
    # A PERSON'S BELL CURVE IN ANOTHER UNIT (§ 2: taken as given; § 9.1's
    # Scaled): a bell curve's Gauss rule, no temperature or quanta said.
    "a person's distribution in another unit": (
        lambda v: v.mode(1).bind(dist.Gaussian(1.0).scaled(1.3)).frames(3),
        {"sampling_rule": "gauss-hermite", "mode_index_1based": 1},
        {"quanta", "temperature_k", "quantum", "mean_quanta"}),
    "a scaled frequency": (lambda v: v.with_frequency_scale(0.96).mode(2)
                           .bind(_WARM).frames(3),
                           {"frequency_scale": 0.96,
                            "result_frequency_cm1": 700.0,
                            "frequency_cm1": 0.96 * 700.0}, set()),
    "snapshots": (lambda v: v.motion(_WARM).with_mode(2, states.level(2))
                  .with_mode(1, states.thermal(temperature_k=77.0))
                  .with_mode(3, states.level(1))
                  .sample(4, seed=7, antithetic=True),
                  {"sampling_rule": "monte-carlo", "samples": 4, "seed": 7,
                   "start": 0, "antithetic": 1, "modes": "every real mode"},
                  # EVERY MODE SET APART: no mode keeps the background, so
                  # the set states neither its temperature nor whether it
                  # is quantum (§ 5.1) -- each mode's state is its own.
                  {"mode_index_1based", "temperature_k", "quantum"}),
    # A PHOTON ON THE WARM SOFT MODE, set apart in a warm motion (§ 3.1):
    # the background's rows, the mode's own state in info.sampling.
    "snapshots, a photon set apart": (
        lambda v: v.motion(_WARM).with_mode(1, states.thermal().add_quantum())
        .sample(4, seed=9, antithetic=True),
        {"sampling_rule": "monte-carlo", "temperature_k": 300.0,
         "quantum": 1, "modes": "every real mode"}, {"mode_index_1based"}),
    # A COLD MOTION WITH A PHOTON SET APART (§ 3.1): the background's rows
    # at the motion's 77 K, and the set read back as written; that the open
    # start set apart takes 77 K, not the record's 300 K, is held by
    # test_vibration_mechanics::test_a_mode_set_apart_takes_the_motions_
    # temperature_or_the_records.
    "snapshots at 77 K, a photon set apart": (
        lambda v: v.motion(states.thermal(temperature_k=77.0))
        .with_mode(1, states.thermal().add_quantum())
        .sample(4, seed=9, antithetic=True),
        {"sampling_rule": "monte-carlo", "temperature_k": 77.0,
         "quantum": 1}, {"mode_index_1based"}),
    "snapshots of some modes": (lambda v: v.motion(_WARM, modes=[2, 3])
                                .sample(4, seed=3, antithetic=False),
                                {"modes": "2, 3", "antithetic": 0},
                                {"set_apart_modes"}),
    "classical snapshots": (lambda v: v.motion(states.classical_thermal(
                                temperature_k=300.0))
                            .sample(4, seed=5, antithetic=True),
                            {"quantum": 0, "temperature_k": 300.0},
                            {"set_apart_modes"}),
    # A PERSON'S OWN BELL CURVE on every mode: neither a temperature nor
    # whether it is quantum said.
    "snapshots of a person's distribution": (
        lambda v: v.motion(dist.Gaussian(1.5)).sample(4, seed=15,
                                                      antithetic=False),
        {"sampling_rule": "monte-carlo"},
        {"temperature_k", "quantum", "set_apart_modes"}),
    # THE MODES CHOSEN IN STEPS, the record's temperature noted: a window,
    # one mode added, one dropped (§ 3.1).
    "modes chosen in steps": (lambda v: v.motion(states.thermal(),
                                                 modes=v.mode_selection()
                                                 .window(500.0, None)
                                                 .add([1]).drop([3]))
                              .sample(4, seed=11, antithetic=True),
                              {"modes": "1, 2", "quantum": 1,
                               "temperature_k": 300.0}, set()),
    "scaled snapshots": (lambda v: v.with_frequency_scale(0.96).motion(_WARM)
                         .sample(4, seed=13, antithetic=False),
                         {"frequency_scale": 0.96}, set()),
}

#: The recipes whose frames are snapshots (§ 3), each with its
#: info.sampling.
_SNAPSHOT_KINDS = {"snapshots", "snapshots, a photon set apart",
                   "snapshots at 77 K, a photon set apart",
                   "snapshots of some modes", "classical snapshots",
                   "snapshots of a person's distribution",
                   "modes chosen in steps", "scaled snapshots"}


#: EACH MODE SET'S STATE IN ITS ASSUMPTIONS ROW, in its words
#: (`engines/vibration-frames.md` § 2's ``words``: the chain's steps, the
#: thermal start's mean quanta on the mode -- Bose and Einstein's by CODATA
#: -- or a distribution's own words, said as the person's).
_STATE_WORDS = {
    "thermal": lambda: (f"thermal at 300 K ({_n_th(120.0, 300.0):.3g} "
                        f"mean quanta)",),
    "an even rule": lambda: (f"thermal at 300 K ({_n_th(700.0, 300.0):.3g} "
                             f"mean quanta)",),
    "classical": lambda: ("the classical bell curve at 300 K, no zero-point "
                          "motion",),
    "n quanta": lambda: ("2 quanta exactly",),
    "mean quanta": lambda: ("thermal, 0.4 mean quanta",),
    "one photon on a warm mode": lambda: (
        f"thermal at 300 K ({_n_th(120.0, 300.0):.3g} mean quanta); one "
        f"quantum added",),
    # § 9.1's words: a list's levels and its mean, sum_n n p_n = 1.
    "a person's list": lambda: ("a list over 3 levels, 1 mean quantum",
                                "the person's own"),
    "a person's distribution in another unit": lambda: (
        "a bell curve of spread 1, scaled by 1.3", "the person's own"),
    "a scaled frequency": lambda: (
        f"thermal at 300 K ({_n_th(672.0, 300.0):.3g} mean quanta)",),
}


@pytest.mark.parametrize("kind", list(_RECIPES))
def test_a_recipes_set_is_written_with_the_rows_of_its_kind(tmp_path, kind):
    """§ 5, § 5.1: a recipe's set is written as it stands -- frame 0 the
    relaxed geometry, every frame's weight and measured moves, a prepended
    frame 0's rows at 0 -- with the provenance, the masses, an
    assumptions row and § 5.1's rows of its kind, each with the
    vocabulary's meaning and its unit in one place; a snapshot set its
    info.sampling; a vibration read from no run's folder said to
    state no run and no relaxation record.  The silent failures: a set
    whose rows say another state than its frames sample, or a frame 0
    that is not the relaxed geometry."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import write_frame_set
    from molbuilder.vibration_frames import MASS_CHANNEL, ROWS
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    make, expected, absent = _RECIPES[kind]
    frames = make(v)
    written = write_frame_set(tmp_path / "set.xyz", frames)
    s = StructureCodec().load(written.path, frames=True)
    rows = read(s)
    assert written.n_frames == s.n_frames == len(frames.weights)
    assert np.allclose(s.positions, v.positions_ang, atol=1e-6), \
        "frame 0 is not the relaxed geometry"
    assert rows.weights == pytest.approx(list(frames.weights), abs=1e-15)
    for name, value in expected.items():
        assert rows.value(name) == pytest.approx(value), (name, rows.value(name))
    for name in absent:
        assert rows.value(name) is None, name
    assert rows.value("assumptions") and rows.value("mass_source")
    # WHAT FRAMES OF A VIBRATION ASSUME is the set's own to say, beside the
    # harmonic picture (science § 7; the user, 2026-11-26: transport does
    # the transport calculation and that's it): the electron fast against
    # the vibration, no inelastic step in a frame average, the same state at
    # every voltage.
    for said in ("the electrons cross fast against the vibration",
                 "inelastic steps at eV = hbar omega",
                 "The state is the same at every voltage"):
        assert said in rows.value("assumptions"), (said, kind)
    vocabulary = {r.name: r for r in ROWS}
    for r in (*rows.structure, *rows.frames[0]):
        if r["name"] in vocabulary:
            assert r["note"] == vocabulary[r["name"]].means, r["name"]
            # ITS UNIT IN ONE PLACE (`model/structure.md` § 2.2d): blank
            # where its name carries it, 1 for a count, none for a text.
            assert r.get("unit") == (None if isinstance(r["value"], str)
                                     else vocabulary[r["name"]].unit_written), r
    zero = {r["name"]: r["value"] for r in rows.frames[0]}
    assert zero["displacement_ang"] == zero["max_atom_displacement_ang"] == 0.0
    assert zero.get("q_amu12_ang", 0.0) == zero.get("node_sigma", 0.0) == 0.0
    assert list(s.get_channel(MASS_CHANNEL).data.values()) == list(v.masses_amu)
    assert ("sampling" in (s.info or {})) == (kind in _SNAPSHOT_KINDS)
    # WHAT A MODE'S OWN SET WAS BOUND FROM, to bind it again (§ 5).
    assert ("state" in (s.info or {})) == (kind not in _SNAPSHOT_KINDS)
    # WHAT THE SET ASSUMES, in its own words: a mode's state as the toolkit
    # words it.
    assumptions = rows.value("assumptions")
    if getattr(frames, "mode_distribution", None) is not None:
        assert all(w in assumptions for w in _STATE_WORDS[kind]()), \
            assumptions
    else:
        assert rows.value("set_apart_modes") == (
            frames.motion.set_apart_modes or None)
        assert ("every mode not included stays at the relaxed geometry"
                in assumptions) == (kind in ("snapshots of some modes",
                                             "modes chosen in steps"))
        # THE BACKGROUND SAID WHERE A MODE KEEPS IT -- none where every
        # mode is set apart.
        assert ("each mode set apart" in assumptions) == (kind == "snapshots")
    assert any("not read from its run's folder: the set states no "
               "vibration_run and no info.calculation" in w
               for w in written.said), written.said


def _bound_said(distributions):
    """Each mode's state as a set records and says it: its state on
    the mode's oscillator, its words, its start's temperature and where that
    came from, its record."""
    return [(k, d.state, d.words, d.temperature_k, d.temperature_from,
             d.to_record()) for k, d in distributions.items()]


@pytest.mark.parametrize("kind", list(_RECIPES))
def test_a_set_read_back_is_the_set_written(tmp_path, kind):
    """§ 5: a set read back is the set as written, every frame, frame 0
    included -- with its vibration given, its geometries, coordinates,
    weights and the state it was bound from, bound again (a mode's own set
    its ``info.state`` -- a chain's or a person's list -- its words, mean
    quanta and own frames the same; a snapshot set's motion whole, each
    mode's record bound again, drawn again bit for bit); without it, its
    geometries, weights and largest moves still, and why it has no
    vibration said; written again, the same rows and records.  The silent
    failures: results paired with frames that are not the ones written, or
    a set re-read at another state than its own -- one photon on a warm
    mode read back as the thermal state of its mean."""
    from molbuilder.postprocess import read_frame_set, write_frame_set
    v = _model()
    frames = _RECIPES[kind][0](v)
    path = write_frame_set(tmp_path / "set.xyz", frames).path
    back = read_frame_set(path, vibration=v)
    assert type(back) is type(frames)
    assert np.allclose(back.positions_ang, frames.positions_ang, atol=1e-6)
    assert np.array_equal(back.q_amu12_ang, frames.q_amu12_ang)
    assert np.array_equal(back.weights, frames.weights)
    assert np.array_equal(back.mode_index_1based, frames.mode_index_1based)
    if hasattr(frames, "motion"):
        # EACH MODE'S STATE, as recorded and bound again (§ 5): its
        # state, words, temperature and its source.
        assert _bound_said(back.motion.distributions) == \
            _bound_said(frames.motion.distributions)
        assert back.motion.background_words == \
            frames.motion.background_words
        assert (back.motion.temperature_k, back.motion.temperature_from,
                back.motion.classical) == (frames.motion.temperature_k,
                                           frames.motion.temperature_from,
                                           frames.motion.classical)
        assert back.said == frames.said
        assert (back.seed, back.start) == (frames.seed, frames.start)
        assert back.motion.said == frames.motion.said
        assert back.motion.set_apart_modes == frames.motion.set_apart_modes
        assert back.motion.chosen_by == frames.motion.chosen_by
        # THE MOTION WHOLE -- states and the modes chosen: drawn again from
        # what was read, the same coordinates.
        again = back.motion.sample(frames.samples, seed=frames.seed,
                                   antithetic=frames.antithetic,
                                   start=frames.start)
        assert np.array_equal(again.q_amu12_ang, frames.q_amu12_ang)
    else:
        # ITS STATE BOUND AGAIN, at the temperature it was bound at, from
        # where it came as recorded (§ 5): its words, its mean quanta and
        # spread, and its own frames again bit for bit.
        b, f = back.mode_distribution, frames.mode_distribution
        assert _bound_said({1: b}) == _bound_said({1: f})
        assert (b.mean_quanta, b.spread_amu12_ang) == (f.mean_quanta,
                                                       f.spread_amu12_ang)
        assert back.exact_to_degree == frames.exact_to_degree
        again = b.frames((frames.exact_to_degree + 1) // 2)
        assert np.array_equal(again.q_amu12_ang, frames.q_amu12_ang)
        assert np.array_equal(again.weights, frames.weights)
    bare = read_frame_set(path)
    assert np.allclose(bare.max_atom_displacement_ang,
                       frames.max_atom_displacement_ang, atol=1e-5)
    assert np.array_equal(bare.mode_index_1based, frames.mode_index_1based)
    assert bare.said[0].startswith("the set names no vibration run")
    # WRITTEN AGAIN, given its vibration (§ 5): the same set -- its rows,
    # their units and notes, its info.sampling (numbers to the 1e-6 Å a
    # geometry is written to).
    from molbuilder.frameset import read
    from molbuilder.workingcopy_structure import StructureCodec
    again = write_frame_set(tmp_path / "again.xyz", back).path
    first_s, again_s = (StructureCodec().load(x, frames=True)
                        for x in (path, again))
    r1, r2 = read(first_s), read(again_s)
    for a, b in [(r1.structure, r2.structure), *zip(r1.frames, r2.frames)]:
        assert [r["name"] for r in a] == [r["name"] for r in b]
        for x, y in zip(a, b):
            if isinstance(x["value"], str):
                assert x == y
            else:
                assert y["value"] == pytest.approx(x["value"], rel=1e-5,
                                                   abs=1e-6), (x, y)
                assert (x.get("unit"), x.get("note")) == \
                    (y.get("unit"), y.get("note"))
    for key in ("sampling", "state"):
        assert (first_s.info or {}).get(key) == (again_s.info or {}).get(key)


#: WHAT A MODE'S OWN SET RECORDS OF THE STATE IT WAS BOUND FROM
#: (`engines/vibration-frames.md` § 5's ``info.state``), on § 5's example
#: mode -- 200 cm-1 (the model's mode 1 at a measured frequency) at the
#: record's 300 K: the chain as given, its steps in order, and the
#: temperature its start took and from where; or a distribution's family
#: and numbers.
_RECORDED = {
    "one photon on a warm mode": (
        _WARM.add_quantum(),
        {"chain": [{"step": "thermal", "values": {"temperature_k": 300.0}},
                   {"step": "add_quantum", "values": {}}],
         "temperature_k": 300.0, "temperature_from": "given"}),
    "the same, its temperature the record's": (
        states.thermal().add_quantum(),
        {"chain": [{"step": "thermal", "values": {}},
                   {"step": "add_quantum", "values": {}}],
         "temperature_k": 300.0, "temperature_from": "the record's"}),
    "a person's list": (
        _LIST, {"distribution": {"family": "LevelList",
                                 "populations": [0.25, 0.5, 0.25]}}),
    # § 2's case table: one photon on a cold mode, the same state as
    # level(1), recorded as given; its rows the level's (§ 5.1).
    "one photon on a cold mode": (
        states.ground().add_quantum(),
        {"chain": [{"step": "ground", "values": {}},
                   {"step": "add_quantum", "values": {}}],
         "temperature_k": None, "temperature_from": None}),
    # A PERSON'S LEVEL WIDENED, in another unit: its scale and base kept.
    "a person's distribution in another unit": (
        dist.Level(1).scaled(1.1),
        {"distribution": {"family": "Scaled", "scale": 1.1,
                          "base": {"family": "Level", "n": 1}}}),
}


@pytest.mark.parametrize("case", list(_RECORDED))
def test_a_mode_set_records_the_state_it_was_bound_from(tmp_path, case):
    """§ 5, its example (one photon on a 200 cm-1 mode at 300 K): a mode's
    own set records its mode's state in ``info.state`` -- the chain as
    given, the temperature its start took and where from, or a person's
    list -- beside the rows ``temperature_k`` 300, ``mean_quanta`` 2.24
    (``2 nbar + 1``, Bose and Einstein by CODATA), ``quantum`` 1,
    ``sampling_rule`` gauss and no ``quanta``, its assumptions saying the
    state in its words; read back, its state is bound again at the
    temperature it was bound at, from where it came, its words and own
    frames the same.  THE SILENT FAILURES: one photon on a warm mode read
    back as the thermal state of its mean, or an open start's temperature
    said given."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import read_frame_set, write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    chain, recorded = _RECORDED[case]
    v = _model()
    d = v.mode(1).with_frequency(200.0).bind(chain)
    path = write_frame_set(tmp_path / "set.xyz", d.frames(3)).path
    s = StructureCodec().load(path, frames=True)
    assert s.info["state"] == recorded, s.info["state"]
    rows = read(s)
    if isinstance(chain, dist.Distribution):
        assert [rows.value(n) for n in ("temperature_k", "mean_quanta",
                                        "quantum", "quanta")] == [None] * 4
    elif case == "one photon on a cold mode":
        assert [rows.value(n) for n in ("temperature_k", "mean_quanta",
                                        "quantum", "quanta")] == [
            None, 1.0, 1, 1]
    else:
        assert rows.value("temperature_k") == 300.0
        assert rows.value("mean_quanta") == pytest.approx(
            2.0 * _n_th(200.0, 300.0) + 1.0, rel=1e-6)
        assert round(rows.value("mean_quanta"), 2) == 2.24
        assert (rows.value("quantum"), rows.value("quanta")) == (1, None)
        assert "thermal at 300 K (0.621 mean quanta)" in rows.value(
            "assumptions") and "; one quantum added" in rows.value(
            "assumptions"), rows.value("assumptions")
    assert rows.value("sampling_rule") == "gauss"
    back = read_frame_set(path, vibration=v).mode_distribution
    assert (back.words, back.temperature_k, back.temperature_from) == (
        d.words, d.temperature_k, d.temperature_from)
    assert back.mean_quanta == d.mean_quanta
    assert np.array_equal(back.frames(3).q_amu12_ang, d.frames(3).q_amu12_ang)


def test_a_mode_set_that_records_no_state_is_read_without_it(tmp_path):
    """§ 5 (the user's choice of 2026-11-26): a mode's own set that records
    no ``info.state`` -- as molbuilder before 2.85.0 wrote one, or a
    person's frames written in the toolkit's rule's rows -- reads back as
    its frames, weights, coordinates and moves whole (each atom's from the
    geometries as written, frame 0 the reference), said once that it
    records no state, its state none, its run not reopened and no
    vibration asked for: no reader of its rows as a state.  THE SILENT
    FAILURES: a state guessed from a set's rows -- a thermal state read for
    frames placed for another; a script told to give a vibration that
    could bind nothing; no moves for a set whose geometries hold them."""
    from molbuilder.postprocess import read_frame_set, write_frame_set
    from molbuilder.vibration import Frames
    v = _model()
    f = v.mode(1).bind(_WARM).frames(3)
    q = f.q_amu12_ang[:, 0]
    path = write_frame_set(
        tmp_path / "set.xyz", f.positions_ang, f.weights, vibration=v,
        rows=[*_mode_rows(v),
              {"name": "sampling_rule", "value": "gauss-hermite"},
              {"name": "sampling_nodes", "value": 3, "unit": "1"},
              {"name": "assumptions", "value": "mine"}],
        frame_rows=[[{"name": "q_amu12_ang", "value": float(x)}]
                    for x in q]).path
    for given in (v, None):
        back = read_frame_set(path, vibration=given)
        assert isinstance(back, Frames) and back.mode_distribution is None
        assert sum("records no state" in w for w in back.said) == 1, \
            back.said
        assert np.array_equal(back.q_amu12_ang[:, 0], q)
        assert np.array_equal(back.weights, f.weights)
        assert back.exact_to_degree == 5
        assert not any("give one" in w or "vibration run" in w
                       for w in back.said), back.said
        # ITS MOVES, FROM THE GEOMETRIES AS WRITTEN: the .xyz's six decimals.
        assert back.atom_moves_ang is not None
        assert np.allclose(back.atom_moves_ang,
                           f.positions_ang - f.positions_ang[:1], atol=2e-6)


#: A RECORD THAT DOES NOT BIND AGAIN, one row per kind of set: the frames, the
#: sidecar's record its start's key misspelled (as a person's edit leaves
#: it), and what the set is said to be read without.
_UNBOUND = [
    ("a mode's set",
     lambda v: v.mode(1).bind(_WARM).frames(3),
     lambda info: info["state"]["chain"][0],
     "its recorded state does not bind again", "mode_distribution"),
    ("a snapshot set",
     lambda v: v.motion(_WARM).sample(4, seed=7, antithetic=False),
     lambda info: info["sampling"]["background"]["chain"][0],
     "its recorded motion does not bind again", "motion"),
]


@pytest.mark.parametrize("case, frames_of, start_of, words, held_as",
                         _UNBOUND, ids=[row[0] for row in _UNBOUND])
def test_a_recorded_state_that_does_not_bind_again_is_said(
        tmp_path, case, frames_of, start_of, words, held_as):
    """§ 5: a set whose recorded state or motion does not bind again to its
    vibration -- its start's ``temperature_k`` spelled ``temperatur_k`` in
    the sidecar -- is read without it, said once with the start's own
    refusal, its frames, weights and geometries whole.  THE SILENT
    FAILURES: a read that stops on a set whose frames are all there; a
    state bound with the misspelled value dropped -- a thermal start left
    open, bound at another temperature than the set's."""
    import json
    from molbuilder.postprocess import read_frame_set, write_frame_set
    from molbuilder.sidecars.molstruct import sidecar_path_for
    v = _model()
    f = frames_of(v)
    path = write_frame_set(tmp_path / "set.xyz", f).path
    side = sidecar_path_for(path)
    payload = json.loads(side.read_text())
    start = start_of(payload["info"])
    start["values"] = {"temperatur_k": start["values"]["temperature_k"]}
    side.write_text(json.dumps(payload))
    back = read_frame_set(path, vibration=v)
    assert getattr(back, held_as) is None
    assert [w for w in back.said if words in w] == [
        f"{words} (thermal takes temperature_k or mean_quanta; got "
        f"'temperatur_k'): it is read without its "
        f"{'state' if held_as == 'mode_distribution' else 'motion'}"], \
        back.said
    # READ WITH ITS VIBRATION, never said to be read without it.
    assert not any("without its vibration" in w for w in back.said), back.said
    assert np.array_equal(back.weights, f.weights)
    assert np.allclose(back.positions_ang, f.positions_ang, atol=1e-6)


def test_a_persons_frames_are_written_with_what_every_set_needs(tmp_path):
    """§ 3.4, § 5: a person's frames keep their first as frame 0 where it is
    the relaxed geometry (as a pair holds it), else get it prepended at
    weight 0 with the person's frame0_rows -- a frame row the frame 0 then
    lacks said -- and every set's rows besides; a set stating no
    assumptions row said;
    read back as a FrameSet.  The silent failure: a second copy of the
    relaxed geometry, or results shifted by one frame against the file."""
    from molbuilder.postprocess import read_frame_set, write_frame_set
    from molbuilder.vibration_frames import FrameSet
    v = _model()
    grid = [v.displaced({1: a, 2: b}) for a in (0.0, 0.1) for b in (0.0, -0.1)]
    marks = [[{"name": "q1_amu12_ang", "value": a}] for a in (0, 0, .1, .1)]
    kept = write_frame_set(tmp_path / "grid.xyz", grid, [0.25] * 4,
                           vibration=v, frame_rows=marks,
                           rows=[{"name": "sampling_rule", "value": "grid"}])
    assert kept.n_frames == 4
    assert any("no assumptions row" in w for w in kept.said)
    back = read_frame_set(kept.path)
    assert isinstance(back, FrameSet)
    assert np.allclose(back.positions_ang, grid, atol=1e-6)
    assert back.rows.value("sampling_rule") == "grid"
    shifted = write_frame_set(tmp_path / "shifted.xyz", grid[1:],
                              [0.5, 0.25, 0.25], vibration=v,
                              frame_rows=marks[1:],
                              rows=[{"name": "assumptions", "value": "mine"}])
    assert shifted.n_frames == 4
    assert any("prepended frame 0 states no q1_amu12_ang" in w
               for w in shifted.said)
    assert not any("no assumptions row" in w for w in shifted.said)
    again = read_frame_set(shifted.path)
    assert np.array_equal(again.weights, [0.0, 0.5, 0.25, 0.25])
    assert np.allclose(again.positions_ang[0], v.positions_ang, atol=1e-6)


@pytest.mark.parametrize("call, refused", [
    ("a pair already there", "never written over"),
    ("frames of a person's own distribution",
     "a _Mine states no record to be made again from"),
    ("snapshots of a person's own distribution",
     "a _Mine states no record to be made again from"),
    ("a frame row named weight", "the writer's own"),
    ("a weight short", "one weight per frame"),
    ("weights summing off 1", "the frames' weights sum to 1.1, not 1 within "
     "1e-06: the weights given are each frame's share of the average"),
    ("no vibration", "with the vibration they are of"),
    ("a recipe given weights", "bring their own vibration"),
    ("a row named mass_source", "the writer's own"),
    ("a set read back without its vibration",
     "read back with no state for their mode"),
    ("snapshots read back without their vibration",
     "once its vibration is given"),
    ("a person's set read back", "written again as a person's frames"),
    ("a row that is no mapping", "a row is a mapping"),
    ("a frame 0 row that is no mapping", "a row is a mapping"),
    ("a frame 0 row named weight", "the writer's own"),
    ("a set of one frame", "two frames or more"),
])
def test_the_writer_refuses_by_name(tmp_path, call, refused):
    """§ 5: refused by name, writing nothing -- a pair already at the path
    (a calculation citing it is pinned to it), a recipe's frames whose
    mode's state records nothing to bind again (a distribution of a person's
    own class, which is drawn from and placed in memory, its frames and
    snapshots made, but cannot be read back), a person's row -- in rows,
    frame_rows or frame0_rows -- that is no mapping or is named like the
    writer's own, weights not one per frame, a person's frames without
    their vibration, a recipe's frames given what it brings, a set read
    back that is written again otherwise, a set of one frame.  The silent
    failures: a cited pair overwritten, a frame's weight replaced by the
    person's row of that name, or a set written that reads back with no
    state."""
    from molbuilder.vibration_frames import FrameSetError, write_frame_set
    v = _model()
    path = tmp_path / "set.xyz"
    f = v.mode(1).bind(_WARM).frames(3)
    grid = [v.displaced({}), v.displaced({1: 0.1})]
    if call == "a pair already there":
        write_frame_set(path, f)
    with pytest.raises(FrameSetError, match=refused):
        if call in ("a pair already there", "a recipe given weights"):
            write_frame_set(path, f, *([[1.0, 0, 0]] if call ==
                                      "a recipe given weights" else []))
        elif call == "frames of a person's own distribution":
            write_frame_set(path, v.mode(1).bind(_Mine()).frames(3))
        elif call == "snapshots of a person's own distribution":
            write_frame_set(path, v.motion(_Mine()).sample(
                2, seed=1, antithetic=True))
        elif call == "a frame row named weight":
            write_frame_set(path, grid, [0.5, 0.5], vibration=v,
                            frame_rows=[[{"name": "weight", "value": 0.5,
                                          "unit": "1"}]] * 2)
        elif call == "a weight short":
            write_frame_set(path, grid, [1.0], vibration=v)
        elif call == "weights summing off 1":
            write_frame_set(path, grid, [0.5, 0.6], vibration=v)
        elif call == "a row named mass_source":
            write_frame_set(path, grid, [0.5, 0.5], vibration=v,
                            rows=[{"name": "mass_source", "value": "mine"}])
        elif call in ("a set read back without its vibration",
                      "snapshots read back without their vibration"):
            from molbuilder.vibration_frames import read_frame_set
            first = write_frame_set(tmp_path / "first.xyz", f if call ==
                                    "a set read back without its vibration"
                                    else v.motion(_WARM).sample(
                                        2, seed=1, antithetic=True)).path
            write_frame_set(path, read_frame_set(first))
        elif call == "a person's set read back":
            from molbuilder.vibration_frames import read_frame_set
            first = write_frame_set(tmp_path / "first.xyz", grid, [0.5, 0.5],
                                    vibration=v).path
            write_frame_set(path, read_frame_set(first, vibration=v))
        elif call == "a row that is no mapping":
            write_frame_set(path, grid, [0.5, 0.5], vibration=v,
                            rows=["assumptions: mine"])
        elif call in ("a frame 0 row that is no mapping",
                      "a frame 0 row named weight"):
            write_frame_set(path, grid[1:], [1.0], vibration=v,
                            frame0_rows=["mark"] if call.endswith("mapping")
                            else [{"name": "weight", "value": 0.0}])
        elif call == "a set of one frame":
            write_frame_set(path, v.mode(1).bind(_WARM).frames(1))
        else:
            write_frame_set(path, grid, [0.5, 0.5])
    assert (path.exists() and call == "a pair already there") \
        or not path.exists()


def test_the_writer_says_what_it_writes_as_given(tmp_path):
    """§ 5: written and said, never refused -- a person's frames moving a
    held atom.  The silent failure: a held atom moved unseen."""
    from molbuilder.vibration_frames import write_frame_set
    v = _model(held=True)
    moved = v.displaced({1: 0.1}).copy()
    moved[3] += 0.05
    got = write_frame_set(tmp_path / "set.xyz", [v.displaced({}), moved],
                          [0.5, 0.5], vibration=v,
                          rows=[{"name": "assumptions", "value": "mine"}])
    assert any("frame 2 of 2 moves a held atom" in w for w in got.said), \
        got.said


@pytest.mark.parametrize("chain", [
    _WARM, states.level(2), states.classical_thermal(temperature_k=300.0),
    _WARM.add_quantum()],
    ids=["thermal", "n quanta", "classical", "one photon on a warm mode"])
@pytest.mark.parametrize("nodes", [3, 4])
def test_the_outermost_move_is_the_written_sets_for_any_state(tmp_path, chain,
                                                              nodes):
    """§ 4, § 5.1: a mode's state's outermost move for its rule
    (``d.outermost_move(nodes)``) -- before a set is written -- is the
    largest atom's move at the outermost frames of the set written of it,
    its ``max_displacement_ang`` row exactly.  The silent failure: the
    page's readout and the set's row disagreeing."""
    from molbuilder.frameset import read
    from molbuilder.vibration_frames import write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    s = v.mode(1).bind(chain)
    f = s.frames(nodes)
    move = s.outermost_move(nodes)
    assert move == pytest.approx(float(np.max(f.max_atom_displacement_ang)),
                                 rel=1e-9)
    path = write_frame_set(tmp_path / "set.xyz", f).path
    assert read(StructureCodec().load(path, frames=True)).value(
        "max_displacement_ang") == move


def _mode_rows(v, number=1, temperature_k=300.0):
    """A mode's rows in the vocabulary's names -- its number, frequency and
    zero-point amplitude, a temperature -- as a person writes them."""
    m = v.mode(number)
    return [{"name": "mode_index_1based", "value": number, "unit": "1"},
            {"name": "frequency_cm1", "value": m.frequency_cm1},
            {"name": "zero_point_amplitude_amu12_ang",
             "value": m.zero_point_amplitude_amu12_ang},
            {"name": "temperature_k", "value": temperature_k}]


@pytest.mark.parametrize("rows, frame_q, said", [
    ([{"name": "mode_index_1based", "value": 7, "unit": "1"}], False,
     "it is read as a person's set"),
    ([{"name": "sampling_rule", "value": "monte-carlo"}], False,
     "it is read as a person's set"),
    ("a mode's rows, no number", False, "it is read as a person's set"),
    ("a mode's rows, its own grid", True, "it is read as a person's set"),
    ("a mode's rows, the rule's name, no nodes", True,
     "it is read as a person's set"),
], ids=["a mode's number alone", "a snapshot set's rule alone",
        "a frequency and amplitude, no mode number",
        "a mode's rows, nodes and coordinates, the person's own rule",
        "a mode's rows and coordinates, gauss-hermite, no nodes"])
def test_a_persons_set_in_the_vocabularys_names_reads_back_as_written(
        tmp_path, rows, frame_q, said):
    """§ 5, § 5.1: a person's set may write the vocabulary's names; one
    that does not state what the toolkit's own set of that kind states --
    a mode's number, frequency and zero-point amplitude, its rule
    (``sampling_rule`` gauss-hermite or gauss, ``sampling_nodes``) and every
    frame's coordinate; a snapshot set's info.sampling -- reads back as a
    FrameSet, its weights as written, and says why.  The silent failure: a
    person's set read as a recipe's -- its weights replaced, its degree
    and rows the toolkit's rule's, not its frames'."""
    from molbuilder.postprocess import read_frame_set, write_frame_set
    from molbuilder.vibration_frames import FrameSet
    v = _model()
    if rows == "a mode's rows, no number":
        rows = _mode_rows(v)[1:]
    elif rows == "a mode's rows, its own grid":
        rows = [*_mode_rows(v),
                {"name": "sampling_nodes", "value": 3, "unit": "1"},
                {"name": "sampling_rule", "value": "my grid at +-1 sigma"}]
    elif rows == "a mode's rows, the rule's name, no nodes":
        rows = [*_mode_rows(v),
                {"name": "sampling_rule", "value": "gauss-hermite"}]
    grid = [v.displaced({}), v.displaced({1: 0.1}), v.displaced({1: -0.1})]
    frame_rows = ([[{"name": "q_amu12_ang", "value": q}]
                   for q in (0.0, 0.1, -0.1)] if frame_q else ())
    path = write_frame_set(tmp_path / "set.xyz", grid, [0.5, 0.25, 0.25],
                           vibration=v, frame_rows=frame_rows,
                           rows=[*rows, {"name": "assumptions",
                                         "value": "mine"}]).path
    back = read_frame_set(path, vibration=v)
    assert isinstance(back, FrameSet)
    assert np.array_equal(back.weights, [0.5, 0.25, 0.25])
    assert any(said in w for w in back.said), back.said


@pytest.mark.parametrize("offset_ang, kept", [(1e-5, False), (4e-7, True)],
                         ids=["farther than the pair holds",
                              "within the pair's six decimals"])
def test_a_persons_frame_0_is_prepended_unless_it_is_the_relaxed_geometry(
        tmp_path, offset_ang, kept):
    """§ 5 (the user, 2026-11-24): a first frame where the relaxed geometry
    stands, as the pair holds positions -- within the structure's one rule
    of two places, a pair's six decimals -- IS frame 0, written as the
    relaxed geometry exactly; one farther off is no frame 0, the relaxed
    geometry prepended at weight 0 with the person's frame0_rows, nothing
    said of a row it states.  The silent failures: a near-copy taken as the
    reference every frame's move is measured from, or two frames the file
    cannot tell apart, one computed for nothing."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    near = v.displaced({}) + offset_ang
    written = write_frame_set(
        tmp_path / "set.xyz", [near, v.displaced({1: 0.1})], [0.5, 0.5],
        vibration=v, frame_rows=[[{"name": "mark", "value": 1, "unit": "1"}],
                                 [{"name": "mark", "value": 2, "unit": "1"}]],
        frame0_rows=[{"name": "mark", "value": 0, "unit": "1"}],
        rows=[{"name": "assumptions", "value": "mine"}])
    s = StructureCodec().load(written.path, frames=True)
    rows = read(s)
    assert np.allclose(s.positions, v.positions_ang, atol=1e-6)
    # FRAME 0 WRITTEN EXACTLY, the reference every move is measured from:
    # the displaced frame's move is from the relaxed geometry, not the
    # person's near frame.
    moved = float(np.max(np.linalg.norm(
        v.displaced({1: 0.1}) - v.displaced({}), axis=1)))
    assert rows.value("max_atom_displacement_ang", 1 if kept else 2) == \
        pytest.approx(moved, rel=1e-12)
    if kept:
        assert written.n_frames == 2
        assert [rows.value("mark", f) for f in range(2)] == [1, 2]
        assert rows.weights == (0.5, 0.5)
    else:
        assert written.n_frames == 3
        assert not any("prepended frame 0 states no" in w
                       for w in written.said)
        assert [rows.value("mark", f) for f in range(3)] == [0, 1, 2]
        assert rows.weights == (0.0, 0.5, 0.5)


def test_a_persons_set_read_back_is_written_again_as_it_was(tmp_path):
    """§ 5: a person's set read back -- frame 0 as the pair holds it, to six
    decimals -- written again as a person's frames, as the writer's refusal
    of a FrameSet says, is the set it was: the same frames, weights and
    rows, no frame prepended.  The silent failure: a second frame 0 the
    calculation citing the set computes for nothing, every result one frame
    later than in the first pair."""
    from molbuilder.postprocess import read_frame_set, write_frame_set
    v = _model()
    grid = [v.displaced({}), v.displaced({1: 0.1}), v.displaced({1: -0.1})]
    marks = [[{"name": "q1_amu12_ang", "value": q}] for q in (0.0, .1, -.1)]
    first = write_frame_set(tmp_path / "set.xyz", grid, [0.5, 0.25, 0.25],
                            vibration=v, frame_rows=marks,
                            rows=[{"name": "assumptions", "value": "mine"}])
    back = read_frame_set(first.path, vibration=v)
    again = write_frame_set(
        tmp_path / "again.xyz", back.positions_ang, back.weights, vibration=v,
        rows=[{"name": "assumptions", "value": "mine"}], frame_rows=marks)
    assert again.n_frames == first.n_frames == 3
    assert np.array_equal(read_frame_set(again.path).weights, back.weights)
    assert not any("prepended" in w for w in again.said), again.said
    from molbuilder.frameset import read
    from molbuilder.workingcopy_structure import StructureCodec
    r1, r2 = (read(StructureCodec().load(x, frames=True))
              for x in (first.path, again.path))
    # EACH FRAME'S ROWS: the person's and the weights as written, the
    # measured moves to the 1e-6 Å a pair holds a geometry to.
    for a, b in zip(r1.frames, r2.frames):
        assert [r["name"] for r in a] == [r["name"] for r in b]
        for x, y in zip(a, b):
            assert y["value"] == pytest.approx(
                x["value"], abs=0 if x["name"] in ("q1_amu12_ang", "weight")
                else 1e-5), (x, y)


@pytest.mark.parametrize("scaled", [1.0, 0.95])
def test_a_set_at_the_results_own_frequency_states_no_scale(tmp_path, scaled):
    """§ 5.1: ``frequency_scale`` and ``result_frequency_cm1`` are written
    only where the set's frequencies are not the result's -- a vibration at
    a scale of 1 is the result's own, and a set read back with a vibration
    at another scale is brought to its own, so written again it states the
    same rows.  The silent failure: a set saying it was scaled when it was
    not, its frequencies off by a rounding."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import read_frame_set, write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    first = write_frame_set(tmp_path / "set.xyz", v.with_frequency_scale(
        scaled).mode(2).bind(_WARM).frames(3)).path
    back = read_frame_set(first, vibration=v.with_frequency_scale(0.9))
    again = write_frame_set(tmp_path / "again.xyz", back).path
    for path in (first, again):
        rows = read(StructureCodec().load(path, frames=True))
        assert rows.value("frequency_cm1") == 700.0 * scaled
        if scaled == 1.0:
            assert rows.value("frequency_scale") is None
            assert rows.value("result_frequency_cm1") is None
        else:
            assert rows.value("frequency_scale") == scaled
            assert rows.value("result_frequency_cm1") == 700.0


@pytest.mark.parametrize("call, cls, refused", [
    ("another vibration", "FrameSetError", "the vibration given is not"),
    ("another vibration for a person's set", "FrameSetError",
     "the vibration given is not"),
    ("no vibration", "FrameSetError", "vibration is a vibration"),
    ("frames stating no weight", "FrameSetError",
     "bare.xyz: no frame states a weight: every frame's weight is "
     "required, their sum 1 -- assign them: postprocess.with_weights"),
    ("weights summing off 1", "FrameSetError",
     "off.xyz: the frames' weights sum to 1.2, not 1 within 1e-06"),
    ("one structure stating a row", "FrameSetError",
     "a frame set holds two frames or more"),
    ("no run", "FrameSetError",
     "no run of ours -- a vibration is read from its run's folder"),
    ("no run, at the run door", "FrameSetError",
     "no run of ours -- name the folder of the vibration run whose modes "
     "are read \\(…/01_freq/run-0\\)$"),
    ("a path that is not there", "SpectraJsonNotFoundError",
     "no file or folder .* a vibration is read from its run's folder"),
    ("a file of another name", "SpectraJsonError",
     "not a vibration result -- a vibration is read from its run's folder"),
])
def test_the_reader_and_the_run_door_refuse_by_name(tmp_path, call, cls,
                                                    refused):
    """§ 1, § 3.3, § 5: refused by name, each in its door's own class -- a
    vibration given that is not the set's (its result's sha256 not the
    set's), a person's set too, or that is no vibration; frames that state
    no frame set's rows, and one structure, whatever rows it states (a frame
    set holds two frames or more); a folder that holds no run of ours, a path
    that is not there and a file of another name, each naming the two forms
    a vibration is read from where both are taken, the run door naming the
    run's folder alone (the command line and the page ask it).  The model
    vibration has no result
    file: its sha256 is stated as the read door states one.  The silent
    failures: a set's results paired with another molecule's modes, a
    mistyped path read as nothing, or a way out the door then refuses."""
    from dataclasses import replace
    from molbuilder.postprocess import (read_frame_set, vibration,
                                        write_frame_set)
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    v = replace(_model(), result_sha256="a" * 64)
    snaps = v.motion(_WARM).sample(4, seed=7, antithetic=True)
    path = write_frame_set(tmp_path / "set.xyz", snaps).path
    from molbuilder.sidecars import spectra
    from molbuilder import vibration_frames
    expected = (getattr(vibration_frames, cls, None)
                or getattr(spectra, cls, None) or ValueError)
    with pytest.raises(expected, match=refused) as refusal:
        if call == "another vibration":
            read_frame_set(path, vibration=replace(v, result_sha256="b" * 64))
        elif call == "another vibration for a person's set":
            grid = write_frame_set(tmp_path / "grid.xyz",
                                   [v.displaced({}), v.displaced({1: 0.1})],
                                   [0.5, 0.5], vibration=v).path
            read_frame_set(grid, vibration=replace(v, result_sha256="b" * 64))
        elif call == "no vibration":
            read_frame_set(path, vibration=str(tmp_path))
        elif call == "a file of another name":
            (tmp_path / "notes.txt").write_text("mine")
            vibration(tmp_path / "notes.txt")
        elif call == "frames stating no weight":
            h2 = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
            read_frame_set(StructureCodec().write(
                h2.with_frames([h2.positions] * 2), tmp_path / "bare.xyz"))
        elif call == "weights summing off 1":
            h2 = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
            off = h2.with_frames([h2.positions] * 2)
            for f in range(2):
                off.set_customized("weight", 0.6, unit="1", frame=f)
            read_frame_set(StructureCodec().write(off, tmp_path / "off.xyz"))
        elif call == "one structure stating a row":
            lone = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
            lone.set_customized("bond", "H-H")
            read_frame_set(StructureCodec().write(lone, tmp_path / "lone.xyz"))
        elif call == "no run":
            vibration(tmp_path)
        elif call == "no run, at the run door":
            vibration_frames.vibration_of_run(tmp_path)
        else:
            vibration(tmp_path / "nowhere")
    assert type(refusal.value).__name__ == cls


# --------------------------------------------------------------------- #
#  A set's derived rows, from the science's closed forms (§ 5.1)          #
# --------------------------------------------------------------------- #

def _q_zp(nu_cm1):
    """``sqrt(hbar / 2 omega)`` in amu^1/2.Å, from CODATA."""
    from scipy import constants as codata
    amu = codata.physical_constants["atomic mass constant"][0]
    omega = 2.0 * math.pi * codata.c * nu_cm1 * 100.0
    return math.sqrt(codata.hbar / (2.0 * omega) / amu) * 1e10


def _n_th(nu_cm1, temperature_k):
    """Bose and Einstein's ``1 / (exp(h c nu / k_B T) - 1)``, from CODATA."""
    from scipy import constants as codata
    return 1.0 / math.expm1(codata.h * codata.c * nu_cm1 * 100.0
                            / (codata.k * temperature_k))


def _classical_spread(nu_cm1, temperature_k):
    """The classical bell curve's ``sqrt(k_B T) / omega`` in amu^1/2.Å."""
    from scipy import constants as codata
    amu = codata.physical_constants["atomic mass constant"][0]
    omega = 2.0 * math.pi * codata.c * nu_cm1 * 100.0
    return math.sqrt(codata.k * temperature_k / amu) / omega * 1e10


#: EACH MODE SET'S FREQUENCY, SPREAD AND MEAN QUANTA, by science § 2b.3,
#: § 2e.1: a quantum state's spread ``Q_zp sqrt(2 <n> + 1)``, its mean
#: quanta the thermal state's, a number state's own, or one photon on a warm
#: mode's ``2 nbar + 1``; a classical bell curve's ``sqrt(k_B T) / omega``,
#: and a person's list's ``Q_zp sqrt(sum_n p_n (2n + 1))``, counting none --
#: ``(mean quanta, None)`` or ``(None, spread)``.
_CLOSED_FORMS = {
    "thermal": (120.0, lambda q: (_n_th(120.0, 300.0), None)),
    "an even rule": (700.0, lambda q: (_n_th(700.0, 300.0), None)),
    "classical": (120.0, lambda q: (None, _classical_spread(120.0, 300.0))),
    "n quanta": (120.0, lambda q: (2.0, None)),
    "mean quanta": (1600.0, lambda q: (0.4, None)),
    "one photon on a warm mode": (
        120.0, lambda q: (2.0 * _n_th(120.0, 300.0) + 1.0, None)),
    "a person's list": (120.0, lambda q: (None, q * math.sqrt(3.0))),
    "a scaled frequency": (0.96 * 700.0,
                           lambda q: (_n_th(0.96 * 700.0, 300.0), None)),
}


@pytest.mark.parametrize("kind", list(_CLOSED_FORMS))
def test_a_mode_sets_derived_rows_are_the_sciences(tmp_path, kind):
    """§ 5.1, science § 2b.3: a mode's own set states its frequency, its
    zero-point amplitude, its state's spread and mean quanta (none for a
    classical state or a person's list), each displaced frame's coordinate
    in units of the
    spread, and the outermost move ``|q_out| max_A |L_A|`` -- each from
    the closed forms on CODATA's constants and the mode's own rows.  The
    silent failure: a set whose rows say another state than its frames
    sample -- a number state's quanta lost."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    frames = _RECIPES[kind][0](v)
    nu, forms = _CLOSED_FORMS[kind]
    q_zp = _q_zp(nu)
    mean, counting_none = forms(q_zp)
    spread = (counting_none if counting_none is not None
              else q_zp * math.sqrt(2.0 * mean + 1.0))
    rows = read(StructureCodec().load(
        write_frame_set(tmp_path / "set.xyz", frames).path, frames=True))
    assert rows.value("frequency_cm1") == pytest.approx(nu, rel=1e-12)
    assert rows.value("zero_point_amplitude_amu12_ang") == pytest.approx(
        q_zp, rel=1e-6)
    assert rows.value("spread_amu12_ang") == pytest.approx(spread, rel=1e-6)
    if mean is None:
        assert rows.value("mean_quanta") is None
    else:
        assert rows.value("mean_quanta") == pytest.approx(mean, rel=1e-6)
    q = [rows.value("q_amu12_ang", f) for f in range(len(rows.frames))]
    for f in range(1, len(q)):
        assert rows.value("node_sigma", f) == pytest.approx(
            q[f] / spread, rel=1e-6), f
    number = int(rows.value("mode_index_1based"))
    largest = float(np.max(np.linalg.norm(
        v.mode(number).eigenvector_canonical_per_amu12, axis=1)))
    assert rows.value("max_displacement_ang") == pytest.approx(
        max(abs(x) for x in q) * largest, rel=1e-6)


# --------------------------------------------------------------------- #
#  The outermost move's door, and what a set says of where it came from  #
# --------------------------------------------------------------------- #

def _with_imaginary_mode_1(v):
    """``v`` with its mode 1 imaginary, as the result's reader states one --
    a negative frequency, no harmonic state."""
    from dataclasses import replace
    return replace(v, modes=(replace(v.modes[0], has_imag=True,
                                     frequency_cm1=-120.0),
                             *v.modes[1:]))


@pytest.mark.parametrize("asked, answer", [
    (dict(mode=1.5), "a whole number"),
    (dict(mode=True), "a whole number"),
    (dict(mode=9), "numbered 1 to 3"),
    (dict(mode=1, nodes=0), "nodes is a whole number"),
    (dict(mode=1, chain="300 K"), "a mode is bound to a chain"),
    (dict(mode=1, imaginary=True), None),
    (dict(mode=1), "three nodes when unsaid"),
], ids=["a mode of 1.5", "a mode of True", "no such mode", "no nodes",
        "no chain", "an imaginary mode", "the default rule"])
def test_the_outermost_move_of_a_mode_by_its_number(asked, answer):
    """§ 4: ``outermost_move_of(v, mode, chain_or_distribution, *,
    nodes=None)`` -- a mode named by its whole number (1.5 is no mode 1,
    True no mode at all), a rule and what binding refuses refused by name
    (``FrameSetError``), an imaginary mode's ``None``, and three nodes when
    unsaid, the mode's state's own move.  The silent failure: a move read
    for another mode than the one named, or for another rule."""
    from molbuilder.vibration_frames import FrameSetError, outermost_move_of
    asked = dict(asked)
    v = (_with_imaginary_mode_1(_model()) if asked.pop("imaginary", False)
         else _model())
    mode, chain = asked.pop("mode"), asked.pop("chain", _WARM)
    if answer is None:
        assert outermost_move_of(v, mode, chain, **asked) is None
    elif answer == "three nodes when unsaid":
        s = v.mode(mode).bind(chain)
        assert outermost_move_of(v, mode, chain) == s.outermost_move() \
            == s.outermost_move(3)
    else:
        with pytest.raises(FrameSetError, match=answer):
            outermost_move_of(v, mode, chain, **asked)


def test_a_set_from_a_result_file_says_the_way_to_a_citation(tmp_path):
    """§ 1: a set written from a vibration read from its result file
    states no run and no relaxation record, and says the way to one that
    does -- the run's folder read instead.  The silent failure: a set that
    cannot be a transport citation, its writer saying nothing of how to
    make one that can."""
    from dataclasses import replace
    from molbuilder.postprocess import write_frame_set
    v = replace(_model(), read_path=str(
        tmp_path / "01_freq" / "run-0" / "v.spectra.json"))
    said = write_frame_set(tmp_path / "set.xyz",
                           v.mode(1).bind(_WARM).frames(3)).said
    assert any("not read from its run's folder (read from v.spectra.json)"
               in w and "vibration(run_folder)" in w for w in said), said


@pytest.mark.parametrize("snapshots", [False, True],
                         ids=["a mode's own set", "snapshots"])
def test_a_set_whose_run_does_not_read_now_is_read_without_it(tmp_path,
                                                              snapshots):
    """§ 5: a set read back without a vibration given reopens the run its
    ``vibration_run`` names; where that run does not read now, the set is
    read without it -- its geometries and weights whole -- and said, never
    refused.  The silent failure: a set that will not open once its run
    has moved, or one paired with no vibration unsaid."""
    from dataclasses import replace
    from molbuilder.postprocess import read_frame_set, write_frame_set
    v = replace(_model(), run=tmp_path / "gone", result_sha256="c" * 64)
    frames = (v.motion(_WARM).sample(2, seed=1, antithetic=True) if snapshots
              else v.mode(1).bind(_WARM).frames(3))
    path = write_frame_set(tmp_path / "set.xyz", frames).path
    back = read_frame_set(path)
    assert getattr(back, "mode_distribution" if not snapshots
                   else "motion") is None
    assert np.array_equal(back.weights, frames.weights)
    assert any("its vibration run does not read now" in w
               for w in back.said), back.said


def test_a_snapshot_set_names_the_imaginary_modes_it_leaves_still(tmp_path):
    """§ 5, § 5.1; science § 7, item 1: a snapshot set of every real mode
    on a vibration with an imaginary mode states ``modes`` "every real
    mode" and says in its assumptions that the imaginary one -- no
    harmonic state -- stays at the relaxed geometry.  The silent failure: a
    set claiming every mode moves while one is frozen."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _with_imaginary_mode_1(_model())
    rows = read(StructureCodec().load(write_frame_set(
        tmp_path / "set.xyz",
        v.motion(_WARM).sample(2, seed=1, antithetic=True)).path,
        frames=True))
    assert rows.value("modes") == "every real mode"
    assert ("every mode not included stays at the relaxed geometry "
            "(imaginary, no harmonic state: 1)") in rows.value("assumptions")
    assert rows.value("temperature_k") == 300.0
    # MODE 1 GIVEN A MEASURED FREQUENCY (§ 3.1) moves: no longer said still,
    # mode 3, left out, still said.
    measured = v.mode(1).with_frequency(100.0).bind(_WARM)
    added = read(StructureCodec().load(write_frame_set(
        tmp_path / "added.xyz", v.motion(_WARM, modes=[2]).with_mode(
            1, measured).sample(2, seed=1, antithetic=True)).path,
        frames=True)).value("assumptions")
    assert "every mode not included stays at the relaxed geometry" in added
    assert "imaginary" not in added, added


# --------------------------------------------------------------------- #
#  The doors' state and what it means before a set is written (§ 6)     #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("chain, words, mean", [
    (states.thermal(temperature_k=600.0),
     lambda: f"thermal at 600 K ({_n_th(120.0, 600.0):.3g} mean quanta)",
     lambda: _n_th(120.0, 600.0)),
    (states.thermal(mean_quanta=1.0), lambda: "thermal, 1 mean quantum",
     lambda: 1.0),
    (states.level(2), lambda: "2 quanta exactly", lambda: 2.0),
], ids=["thermal", "one mean quantum", "n quanta"])
def test_what_a_state_means_for_a_mode_before_its_set_is_written(
        chain, words, mean):
    """`web/spectra.md` § 4.3's table, science § 2b.3: the readout under the
    dialog for each of the doors' chains -- the state in the mode's state's
    words (§ 2: *thermal at 600 K (3 mean quanta)*, Bose and Einstein by
    CODATA; *thermal, 1 mean quantum*; *2 quanta exactly*), the mean quanta,
    the spread against the zero-point one (``sqrt(2 <n> + 1)``), the
    outermost move for each frame count the doors offer, the mode's state's
    own; an imaginary mode none.  The silent failure: a dialog whose
    numbers or words are another state's than the frames written."""
    from molbuilder.vibration_frames import OFFERED_NODES, readout
    m = _model().mode(1)
    said = readout(m, chain)
    assert said["state"] == words(), said["state"]
    n = mean()
    assert said["mean_quanta"] == pytest.approx(n, rel=1e-6)
    ratio = math.sqrt(2.0 * n + 1.0)
    assert said["spread_over_zero_point"] == pytest.approx(ratio, rel=1e-6)
    assert said["spread_amu12_ang"] == pytest.approx(
        ratio * _q_zp(120.0), rel=1e-6)
    assert said["outermost_move_ang"] == {
        str(k): m.bind(chain).outermost_move(k) for k in OFFERED_NODES}
    assert readout(_with_imaginary_mode_1(_model()).mode(1), chain) is None


@pytest.mark.parametrize("asked, chain", [
    ({"temperature_k": 300.0}, states.thermal(temperature_k=300.0)),
    ({"mean_quanta": 0.5}, states.thermal(mean_quanta=0.5)),
    ({"quanta": 2}, states.level(2)),
    ({"quanta": 0, "temperature_k": None}, states.level(0)),
], ids=["a temperature", "a mean number of quanta", "a number of quanta",
        "no quanta, a temperature of none beside it"])
def test_the_doors_choice_is_one_chain(asked, chain):
    """§ 6: each choice the doors take is one chain of the generic starts --
    ``--temperature T`` ``states.thermal(temperature_k=T)``,
    ``--mean-quanta N`` ``states.thermal(mean_quanta=N)``, ``--quanta N``
    ``states.level(N)`` -- and the page's ask (``frameset_state``) the same
    chain by the same rule.  The silent failures: a mean read as a
    temperature, or the page's dialog and the command line writing two
    states for one choice."""
    from molbuilder.vibration_frames import doors_chain, read_state_ask
    assert doors_chain(**asked) == chain
    assert read_state_ask(dict(asked)) == chain


@pytest.mark.parametrize("asked, refused", [
    ({}, "asked every time, never assumed"),
    ({"temperature_k": None}, "asked every time, never assumed"),
    ({"temperature_k": "warm"}, "'warm' is not one"),
    ({"temperature_k": -5}, "-5 is not one"),
    ({"quanta": 2.5}, "the number of quanta is a whole number, 0 or more; 2.5 is not one"),
    ({"quanta": 2, "temperature_k": 300.0},
     "a temperature and a number of quanta name no one state"),
    ({"mean_quanta": -0.5}, "the mean number of quanta is 0 or more; -0.5 is not one"),
], ids=["no state", "a temperature of none", "a temperature in words",
        "below zero", "half a quantum", "two states",
        "mean quanta below zero"])
def test_the_doors_ask_the_state_every_time(asked, refused):
    """§ 2's table, § 6: `frameset from-mode` and *Write frame set…* ask the
    state every time -- none named (a value of none is none named) refused,
    never the record's temperature assumed; a temperature by the rule every
    door shares; two states at once refused in the doors' words, a value by
    its start's own rule -- each a ``StateRefused``, before any run is read,
    the page's ask refused alike.  The silent failure: frames sampled at a
    state the person never stated."""
    from molbuilder.vibration_frames import (StateRefused, doors_chain,
                                             read_state_ask)
    with pytest.raises(StateRefused, match=refused):
        doors_chain(**asked)
    with pytest.raises(StateRefused, match=refused):
        read_state_ask(dict(asked))


@pytest.mark.parametrize("raw, refused", [
    (["temperature_k", 300.0], r"frameset_state is \{temperature_k, "
                               r"quanta, mean_quanta\}"),
    ({"temprature_k": 300.0}, "did you mean 'temperature_k'"),
], ids=["not a mapping", "a misspelled key"])
def test_the_pages_ask_of_another_shape_is_refused_by_name(raw, refused):
    """`web/spectra.md` § 4.3, § 6: the load door's ``frameset_state`` is
    ``{temperature_k?, quanta?, mean_quanta?}`` -- another shape, or a key it
    does not take (with the key it is close to), refused by name
    (``StateRefused``).  The silent failure: a misspelled temperature left
    out, the readout at a state nobody said."""
    from molbuilder.vibration_frames import StateRefused, read_state_ask
    with pytest.raises(StateRefused, match=refused):
        read_state_ask(raw)


@pytest.mark.parametrize("quanta, warned", [(5, False), (6, True)])
def test_the_doors_warn_of_many_quanta(quanta, warned):
    """§ 6 (the user's choice, 2026-11-24): the readout says where a state's
    quanta put more than 1 eV into the mode -- ``<n> h c nu``, far up its
    ladder, where the harmonic picture may no longer hold -- and never
    refuses: at 1600 cm-1 (0.198 eV a quantum, CODATA) five quanta are
    0.99 eV, six 1.19 eV.  The silent failure: frames written far outside
    the picture they are drawn from, unsaid."""
    from scipy import constants as codata
    from molbuilder.vibration_frames import readout
    m = _model().mode(3)
    energy = quanta * codata.h * codata.c * 1600.0 * 100.0 / codata.e
    assert (energy > 1.0) == warned
    said = readout(m, states.level(quanta))
    assert said["mean_quanta"] == quanta
    if warned:
        assert said["beyond_harmonic"].startswith(
            f"mode 3 holds {quanta} quanta in this state, {energy:.3g} eV")
    else:
        assert said["beyond_harmonic"] is None


def test_a_snapshot_sets_analyses_take_the_weights_it_stores(tmp_path):
    """§ 3.3, § 7, science § 5.5 (the user, 2026-11-26: "weight is
    explicitly calculated and saved with multiframe data"): a snapshot set
    read back weighs each frame as its file stores it, and its average and
    error are the statistics' own on the snapshots' values and those
    weights, frame 0 left out.  (Every set the toolkit writes stores 0, then
    1/N, so the weights read and the weights drawn agree on any set it
    writes: this holds the read and the route to the statistics.)  THE
    SILENT FAILURE: a set read back whose weights are not its file's, or an
    error taken another way than the statistics take it."""
    from molbuilder.frameset import read
    from molbuilder.postprocess import read_frame_set, sampling, write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    path = write_frame_set(tmp_path / "set.xyz", v.motion(_WARM).sample(
        6, seed=3, antithetic=True)).path
    stored = np.asarray(read(StructureCodec().load(path, frames=True))
                        .weights)
    back = read_frame_set(path, vibration=v)
    assert np.array_equal(back.weights, stored)
    values = np.array([9.0, 1.0, 3.0, 2.0, 2.0, 5.0, 1.0])
    est = back.standard_error(values)
    mean, error = sampling.mean_with_error(
        *sampling.pair_means(values[1:], stored[1:]))
    assert (est.value, est.error) == (pytest.approx(float(mean), abs=1e-15),
                                      pytest.approx(float(error), abs=1e-15))

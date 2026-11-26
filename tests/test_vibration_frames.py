"""A vibration's frame sets on the host: the one writer and its reader,
what a page says of a cited set's rows, where the set came from, and the
weights its rows give for a state (`engines/vibration-frames.md` § 4, § 5,
§ 5.1, § 6, § 7).

API-level, as a person's script and the web layer ask: each set is a pair on
disk -- written by the toolkit's one writer (`postprocess.write_frame_set`),
or made through the Structure API and the structure's one writer -- and read
back by `postprocess.read_frame_set` and by `vibration_frames.said_of` from
its path; a set's rows are the structure's own reader's (`frameset.read`), as
`postprocess.frame_rows` hands them.  Each expectation is the contract's:
§ 4's outermost move, § 5's rows, refusals, what is said and what is read
back, § 5.1's words and rule, § 6's provenance and changed-pair words, § 7's
refusals, the science's closed forms (`science/vibrational-averaging.md`
§ 2b.3) for a set's derived rows, and the Gauss-Hermite rule from numpy
(``hermgauss``) for a set's own weights.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

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


@pytest.mark.parametrize("where, name, unit, words", [
    ("frames", "displacement_ang", None, ("Displacement", "Å")),
    ("frames", "displacement_ang", "angstrom", ("Displacement", "Å")),
    ("frames", "weight", "1", ("Weight", None)),
    ("frames", "weight", "%", None),
    ("structure", "frequency_cm1", None, ("Frequency", "cm⁻¹")),
    ("structure", "sigma_amu12_ang", None, ("Spread σ", "amu½·Å")),
    ("structure", "bond", "angstrom", None),
], ids=["its unit in its name", "its unit in its name, said again", "the "
        "dimensionless unit", "a unitless name in another unit",
        "a structure row", "a name written before 2.81.0",
        "a name it does not know"])
def test_a_page_says_a_row_in_the_vocabularys_words_where_its_unit_is_the_vocabularys(
        tmp_path, where, name, unit, words):
    """§ 5.1: a row the vocabulary names is said in its words where its unit
    is the vocabulary's -- a name that carries one, `unit` blank or the same;
    a name of none, `1` or blank -- an older set's name read the same; any
    other row by its name and unit as written (no words given).  The silent
    failure:
    a person's row in its own unit labelled on the page with the
    vocabulary's."""
    from molbuilder.vibration_frames import said_of
    row = (name, 0.5, unit, "what it is")
    path = _pair(tmp_path, rows=[row] if where == "structure" else (),
                 frame_rows=[row] if where == "frames" else ())
    got = said_of(path)[where].get(name)
    # ...its words and its unit as a page shows it, the units table's.
    assert (got and (got["words"], got["page_unit"])) == words, got


@pytest.mark.parametrize("note", ["a Boltzmann weight at 300 K", None],
                         ids=["the set's", "none written"])
def test_a_rows_meaning_is_the_sets_and_one_written_without_is_said_once(
        tmp_path, note):
    """§ 5.1, `model/structure.md` § 2.2f: a page's meaning for a row is the
    note the set writes, else the vocabulary's; a vocabulary row written
    without one is said once, by name, however many frames write it -- a
    warning, never a refusal.  The silent failure: a person's own weights
    shown with the Gauss rule's meaning."""
    from molbuilder.vibration_frames import ROWS, said_of
    path = _pair(tmp_path, frame_rows=[("weight", 0.5, "1", note)], n=3)
    got = said_of(path)
    vocabulary = next(r.means for r in ROWS if r.name == "weight")
    assert got["frames"]["weight"]["means"] == (note or vocabulary)
    named = [s for s in got["said"] if "states no meaning" in s]
    assert len(named) == (0 if note else 1), got["said"]
    # ...named in the words the page says it in, never its stored name.
    assert all(s.count("Weight") == 1 and "weight" not in s
               for s in named), named


@pytest.mark.parametrize("rows, info, expected", [
    ([("vibration_run", "P/vib/run", None, None),
      ("vibration_result_sha256", "ab" * 32, None, None)], None,
     ("P/vib/run", "ab" * 32, "rows",
      ["vibration_run", "vibration_result_sha256"])),
    ((), {"vibration": {"run": "P/vib/old", "result_sha256": "cd" * 32}},
     ("P/vib/old", "cd" * 32, "info.vibration", [])),
    ((), None, None),
], ids=["its rows", "a set written before 2.81.0", "none stated"])
def test_a_set_says_where_it_came_from(tmp_path, rows, info, expected):
    """§ 6, `engines/transport.md` § 3.8.1: a vibration's set names its run
    and result in its two provenance rows -- named, so a page says them
    once, on the provenance line; an older set in its ``info.vibration`` --
    read the same, saying which it was; a set stating neither has no
    provenance.  The silent failure: a card naming no run for a set that
    states one, or a run for a set that states none."""
    from molbuilder.vibration_frames import said_of
    got = said_of(_pair(tmp_path, rows=rows, info=info))["provenance"]
    if expected is None:
        assert got is None, got
        return
    assert got is not None
    run, sha, stated_as, named = expected
    assert (got["run"], got["result_sha256"]) == (run, sha), got
    assert stated_as in got["stated_as"] and got["rows"] == named, got


@pytest.mark.parametrize("pinned, changed", [
    ("the .xyz", ["set.xyz"]),
    ("the sidecar", ["set.molstruct.json"]),
    ("both as they are", []),
])
def test_a_pair_that_changed_since_its_citation_is_said(tmp_path, pinned,
                                                         changed):
    """§ 6: given the files a calculation's record pinned, a pair either of
    whose files no longer matches is said so, naming the file; its words
    and provenance shown as the pair now states them.  The silent failure: a
    Results tab showing a set's provenance that is not the one the
    calculation ran."""
    from molbuilder.persist import sha256_of
    from molbuilder.vibration_frames import said_of
    path = _pair(tmp_path, rows=[("vibration_run", "P/vib/run", None,
                                  None)])
    sidecar = path.with_name("set.molstruct.json")
    files = {path.name: sha256_of(path), sidecar.name: sha256_of(sidecar)}
    # THE PERSON EDITS ONE FILE after the record pinned it -- as an editor
    # saves it, its bytes another.
    edited = {"the .xyz": path, "the sidecar": sidecar}.get(pinned)
    if edited is not None:
        edited.write_text(edited.read_text() + "\n")
    got = said_of(path, files)
    said = [s for s in got["said"] if "changed" in s]
    assert len(said) == (1 if changed else 0), got["said"]
    for name in changed:
        assert name in said[0], said
    if changed:
        assert said[0].endswith("cite the pair as it is now in a new "
                                "calculation, or put it back"), said
    assert got["provenance"]["run"] == "P/vib/run", got


def test_the_words_are_of_the_frames_drawn_whatever_became_of_the_pair(
        tmp_path):
    """§ 6, `engines/transport.md` § 3.8.1: given the frames a page draws --
    a calculation's composed junction -- the words, what the set is and
    where it came from are theirs; the cited pair, moved or deleted since,
    is said not to hold what the record pinned, the frames shown being the
    ones the calculation ran.  The silent failure: a report losing every
    word of its set because a person tidied the tree after the run."""
    from molbuilder.persist import sha256_of
    from molbuilder.workingcopy_structure import StructureCodec
    from molbuilder.vibration_frames import said_of
    path = _pair(tmp_path, rows=[("mode_index_1based", 6, "1", None),
                                 ("vibration_run", "P/vib/run", None, None)])
    drawn = StructureCodec().load(path, frames=True)
    files = {path.name: sha256_of(path)}
    files["set.molstruct.json"] = sha256_of(
        path.with_name("set.molstruct.json"))
    path.rename(tmp_path / "moved.xyz")
    got = said_of(path, files, structure=drawn)
    assert got["what"] == "frames of one mode", got
    assert got["structure"]["mode_index_1based"]["words"] == "Mode", got
    assert got["provenance"]["run"] == "P/vib/run", got
    changed = [x for x in got["said"] if "has changed" in x]
    assert len(changed) == 1 and "set.xyz" in changed[0], got["said"]
    assert "the frames shown are the ones the calculation ran" in \
        changed[0], changed


def test_a_pair_that_does_not_read_is_said_and_gives_no_words(tmp_path):
    """§ 6: a cited pair gone or unreadable, with no frames drawn to read
    instead, is said; the page shows the rows the calculation carries,
    without words or provenance.  The silent failure: an error page, or a
    card silently empty, for a pair a person deleted."""
    from molbuilder.vibration_frames import said_of
    got = said_of(tmp_path / "gone.xyz")
    assert got["structure"] == {} and got["frames"] == {}, got
    assert got["provenance"] is None and got["what"] is None, got
    assert len(got["said"]) == 1, got
    # ...in the page's words, its way on said, no path of the server's.
    assert got["said"][0].startswith("the cited pair gone.xyz is not there "
                                     "now"), got["said"]
    assert "put it back" in got["said"][0], got["said"]
    assert str(tmp_path) not in got["said"][0], got["said"]


@pytest.mark.parametrize("rows, what", [
    ([("mode_index_1based", 6, "1", None)], "frames of one mode"),
    ([("samples", 64, "1", None)],
     "frames, the relaxed geometry and its snapshots"),
    ([("sampling_rule", "monte-carlo", None, None)],
     "frames, the relaxed geometry and its snapshots"),
    ([("frequency_cm1", 1000.0, "cm-1", None),
      ("zero_point_amplitude_amu12_ang", 0.1, "amu^1/2 angstrom", None)],
     "frames of one mode"),
    ([("bond", "H-H", None, None)], None),
], ids=["a mode's own", "snapshots by their count", "snapshots by their rule",
        "a mode's own by its frequency and Q_zp", "a person's"])
def test_a_set_says_what_it_is(tmp_path, rows, what):
    """§ 5.1, `engines/transport.md` § 3.8.1: what the set is, the words
    after its count, by the rows its maker wrote -- a mode's own set by its
    mode, a snapshot set by its snapshots, frame 0 the relaxed geometry and
    no snapshot (*65 frames, the relaxed geometry and its snapshots*) -- so
    no page spells a row name to decide it.  The silent failure: a snapshot
    set's relaxed geometry counted as one of its snapshots."""
    from molbuilder.vibration_frames import said_of
    assert said_of(_pair(tmp_path, rows=rows))["what"] == what


@pytest.mark.parametrize("written", ["before 2.81.0, its spread's name",
                                     "before 2.81.0, its info.vibration",
                                     "by a person, no assumptions",
                                     "now"])
def test_a_mode_sets_assumptions_are_said_beside_it(tmp_path, written):
    """Science § 7 ("never dropped"), § 5.1: a mode's own set states its
    harmonic assumptions as its ``assumptions`` row, which a page draws as
    prose on a line of its own; a set written before 2.81.0 -- told by its
    own marks, the spread's older name or ``info.vibration``, every one a
    thermal set -- states none, and the toolkit says them for it, in the
    words a set states them now, at its temperature; a person's set that
    states none is said nothing of, its state not known.  The silent
    failures: an older set's average shown with its harmonic picture nowhere
    beside it; a person's own set said to be thermal."""
    from molbuilder.vibration_frames import (assumptions_of_a_mode, said_of,
                                             thermal_state_words)
    rows = [("mode_index_1based", 6, "1", None),
            ("temperature_k", 300.0, "K", None)]
    info = None
    if written == "before 2.81.0, its spread's name":
        rows.append(("sigma_amu12_ang", 0.12, "amu^1/2 angstrom", None))
    elif written == "before 2.81.0, its info.vibration":
        info = {"vibration": {"run": "P/vib/old", "result_sha256": "cd" * 32}}
    elif written == "now":
        rows.append(("assumptions", "the maker's own words", None, None))
    got = said_of(_pair(tmp_path, rows=rows, info=info))
    if written.startswith("before"):
        # THE WORDS A SET WRITTEN NOW STATES -- the contract's identity
        # (§ 6): the toolkit's writer writes its row by these same two
        # calls.
        assert got["assumptions"] == {
            "words": "Assumptions",
            "value": assumptions_of_a_mode(thermal_state_words(300.0))}, got
        return
    if written == "by a person, no assumptions":
        assert got["assumptions"] is None, got
        return
    assert got["assumptions"] is None, got
    assert got["structure"]["assumptions"]["prose"] is True, got
    assert got["structure"]["mode_index_1based"]["prose"] is False, got


# ----------------------------------------------------------------------- #
#  The weights for a set's rows at a state (§ 7)                           #
# ----------------------------------------------------------------------- #

#: A mode of 1000 cm-1 whose zero-point amplitude is 1 amu^1/2 Angstrom.
_NU, _QZP = 1000.0, 1.0


def _mode_set(m=3, temperature_k=300.0, with_temperature=True, *,
              extra=(), without=(), no_q_at=None):
    """A mode's own set's rows as `frameset.read` hands them, made through
    the Structure API as a script makes one: its frames at the m-point
    Gauss-Hermite rule's nodes for the thermal spread at ``temperature_k``,
    each weighted by the rule -- the set's own weights; ``extra`` structure
    rows ``(name, value)`` added, the structure rows named in ``without``
    left out, frame ``no_q_at`` stating none."""
    from molbuilder.frameset import read
    from molbuilder.structure import Structure
    from molbuilder.vibration import thermal_spread_amu12_ang
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
        s.set_customized("weight", w[f] / math.sqrt(math.pi), unit="1",
                         frame=f)
    return read(s)


@pytest.mark.parametrize("m", [3, 5, 7])
def test_a_sets_rows_at_its_own_state_give_its_own_weights(m):
    """§ 7: at the set's own temperature -- taken from its own
    ``temperature_k`` row, said -- the weights are the set's own, the
    rule's; the re-averaged number is the record's where nothing was asked
    to change.  The silent failure: a re-average that disagrees with the
    record."""
    from molbuilder.vibration_frames import weights_for_set
    rows = _mode_set(m)
    got = weights_for_set(rows)
    assert got.weights == pytest.approx(list(rows.weights), abs=1e-12)
    # THE SET'S OWN RULE: exact to degree 2m - 1 (science § 5.1), nothing
    # said of frames placed for another motion -- the state named alone.
    assert got.exact_to_degree == 2 * m - 1, got
    assert got.said == ("the thermal state at the set's temperature, 300 K "
                        "(its temperature_k row)",), got
    given = weights_for_set(rows, temperature_k=300.0)
    assert given.said == () and given.exact_to_degree == 2 * m - 1, given


def test_an_even_rules_set_at_its_own_state_says_nothing_of_another():
    """§ 7: an even rule's set -- frame 0, the relaxed geometry, prepended at
    weight 0 (§ 5) -- re-weighted to the state it was written for is that
    state's own rule: its written weights, and nothing of frames placed for
    another motion said.  The silent failure: a set's own average said to be
    extrapolated beyond its frames."""
    from molbuilder.frameset import read
    from molbuilder.structure import Structure
    from molbuilder.vibration import state
    from molbuilder.vibration_frames import weights_for_set
    own = state(_QZP, _NU, temperature_k=300.0).frames(4)
    q, w = own.q_amu12_ang[:, 0], own.weights
    s = Structure.from_xyz(_H2)
    s = s.with_frames([s.positions] * len(q))
    s.set_customized("frequency_cm1", _NU, unit="cm-1")
    s.set_customized("zero_point_amplitude_amu12_ang", _QZP,
                     unit="amu^1/2 angstrom")
    s.set_customized("temperature_k", 300.0, unit="K")
    for f in range(len(q)):
        s.set_customized("q_amu12_ang", float(q[f]), frame=f)
        s.set_customized("weight", float(w[f]), unit="1", frame=f)
    got = weights_for_set(read(s), temperature_k=300.0)
    assert got.weights == pytest.approx(list(w), abs=1e-12)
    assert got.said == (), got.said


def test_a_state_beyond_the_frames_is_said_not_refused():
    """§ 9, § 7: a state reaching past the outermost frames -- a set written
    at 0 K re-weighted to 300 K -- is averaged and the share beyond said,
    never refused.  Three frames' blindness to the shape is said with it
    (science § 5.4).  The silent failure: a motion the frames never reached
    averaged from values extrapolated past them, unsaid -- or a person's
    question refused where an answer with its caveat stands."""
    from molbuilder.vibration_frames import weights_for_set
    got = weights_for_set(_mode_set(3, temperature_k=0.0),
                          temperature_k=300.0)
    assert got.beyond > 0.0, got
    assert math.fsum(got.weights) == pytest.approx(1.0, abs=1e-12)
    assert any("beyond the outermost frames" in s for s in got.said), got
    assert any("blind to the state's shape" in s for s in got.said), got


def test_a_state_given_no_temperature_takes_the_sets_and_says_so():
    """§ 2's table, § 7: a state that takes a temperature, given none, is
    taken at the set's own ``temperature_k`` row, said first; one given a
    temperature says nothing of it.  The silent failure: a temperature the
    person never named used unsaid."""
    from molbuilder.vibration_frames import weights_for_set
    thermal = weights_for_set(_mode_set(3), classical=True)
    assert thermal.said[0] == ("the classical state at the set's "
                               "temperature, 300 K (its temperature_k row)")
    given = weights_for_set(_mode_set(3), temperature_k=300.0)
    assert not any("temperature_k row" in s for s in given.said), given.said


@pytest.mark.parametrize("rows, ask, refused", [
    ("snapshots", {"temperature_k": 300.0}, "snapshot set is not re-weighted"),
    ("no mode", {"temperature_k": 300.0}, "no mode's frequency"),
    ("a frame without q", {"temperature_k": 300.0}, "no position along"),
    ("no temperature", {}, "name a temperature: the set states none"),
    ("no temperature", {"classical": True}, "the set states none"),
    ("none", {"temperature_k": 300.0}, "one structure, or frames"),
    ("a path", {"temperature_k": 300.0}, "not a str"),
], ids=["a snapshot set", "a set of no mode", "a frame stating no q",
        "a thermal state, the set stating no temperature",
        "a classical state, the set stating no temperature", "no rows",
        "a path, not rows"])
def test_a_set_the_state_cannot_weight_is_refused_by_name(rows, ask,
                                                          refused):
    """§ 7: refused by name -- a snapshot set (its frames have no single
    mode's coordinate and its weights are equal), a set of no mode, a frame
    stating no coordinate, a state that takes a temperature given none of
    a set stating none (naming the temperature to give), what is no set's
    rows (named by its kind, never printed whole).  A state that takes none
    -- n quanta -- needs none.  The silent failure:
    weights for a motion the frames cannot carry, or a temperature the
    person never named taken from nowhere."""
    from molbuilder.vibration_frames import weights_for_set
    if rows == "snapshots":
        got = _mode_set(extra=[("samples", 2)])
    elif rows == "no mode":
        got = _mode_set(without=("frequency_cm1",))
    elif rows == "a frame without q":
        got = _mode_set(no_q_at=0)
    elif rows == "no temperature":
        got = _mode_set(with_temperature=False)
    elif rows == "a path":
        got = "F.transport.nc" * 1000
    else:
        got = None
    with pytest.raises(ValueError, match=refused) as said:
        weights_for_set(got, **ask)
    # A REFUSAL NAMES WHAT IT WAS GIVEN, never prints it whole.
    assert len(str(said.value)) < 400, str(said.value)
    if rows == "no temperature":
        assert len(weights_for_set(got, quanta=1).weights) == 3


# --------------------------------------------------------------------- #
#  The one writer and its reader (§ 5)                                   #
# --------------------------------------------------------------------- #

def _model(*, held=False):
    """The model molecule's vibration (`support.model_vibration`, the one
    builder) -- with ``held``, a fourth atom held still."""
    from support.model_vibration import model_vibration
    return model_vibration(held=held)


#: THE RECIPES, each with the structure rows § 5.1 says its kind writes --
#: beside the provenance and assumptions every set states.
_RECIPES = {
    "thermal": (lambda v: v.mode(1).state(temperature_k=300.0).frames(3),
                {"sampling_rule": "gauss-hermite", "sampling_nodes": 3,
                 "quantum": 1, "mode_index_1based": 1, "temperature_k": 300.0},
                {"quanta", "result_frequency_cm1"}),
    "an even rule": (lambda v: v.mode(2).state(temperature_k=300.0).frames(2),
                     {"sampling_nodes": 2, "mode_index_1based": 2},
                     set()),
    "classical": (lambda v: v.mode(1).state(temperature_k=300.0,
                                            classical=True).frames(3),
                  {"quantum": 0, "temperature_k": 300.0}, {"mean_quanta"}),
    "n quanta": (lambda v: v.mode(1).state(quanta=2).frames(3),
                 {"quanta": 2, "sampling_rule": "gauss"},
                 {"temperature_k"}),
    "mean quanta": (lambda v: v.mode(3).state(mean_quanta=0.4).frames(3),
                    {"mean_quanta": 0.4}, {"temperature_k"}),
    "a scaled frequency": (lambda v: v.with_frequency_scale(0.96).mode(2)
                           .state(temperature_k=300.0).frames(3),
                           {"frequency_scale": 0.96,
                            "result_frequency_cm1": 700.0,
                            "frequency_cm1": 0.96 * 700.0}, set()),
    "snapshots": (lambda v: v.motion(300.0).with_mode(2, quanta=2)
                  .with_mode(1, temperature_k=77.0).with_mode(3, quanta=1)
                  .sample(4, seed=7, antithetic=True),
                  {"sampling_rule": "monte-carlo", "samples": 4, "seed": 7,
                   "start": 0, "antithetic": 1, "modes": "every real mode"},
                  # EVERY MODE SET APART: no mode keeps the background, so
                  # the set states neither its temperature nor whether it
                  # is quantum (§ 5.1) -- each mode's state is its own.
                  {"mode_index_1based", "temperature_k", "quantum"}),
    "snapshots of some modes": (lambda v: v.motion(300.0, modes=[2, 3])
                                .sample(4, seed=3, antithetic=False),
                                {"modes": "2, 3", "antithetic": 0},
                                {"set_apart_modes"}),
    "classical snapshots": (lambda v: v.motion(300.0, classical=True)
                            .sample(4, seed=5, antithetic=True),
                            {"quantum": 0, "temperature_k": 300.0},
                            {"set_apart_modes"}),
    # THE MODES CHOSEN IN STEPS, the record's temperature noted: a window,
    # one mode added, one dropped (§ 3.1).
    "modes chosen in steps": (lambda v: v.motion(modes=v.mode_selection()
                                                 .window(500.0, None)
                                                 .add([1]).drop([3]))
                              .sample(4, seed=11, antithetic=True),
                              {"modes": "1, 2", "quantum": 1,
                               "temperature_k": 300.0}, set()),
    "scaled snapshots": (lambda v: v.with_frequency_scale(0.96).motion(300.0)
                         .sample(4, seed=13, antithetic=False),
                         {"frequency_scale": 0.96}, set()),
}

#: The recipes whose frames are snapshots (§ 3), each with its
#: info.sampling.
_SNAPSHOT_KINDS = {"snapshots", "snapshots of some modes",
                   "classical snapshots", "modes chosen in steps",
                   "scaled snapshots"}


#: EACH MODE SET'S STATE IN ITS ASSUMPTIONS ROW, in the words of § 2's table
#: (`engines/vibration-frames.md` § 2: the thermal state's bell curve and
#: temperature, the n-th standing wave, the classical bell curve).
_STATE_WORDS = {
    "thermal": ("thermal", "300 K", "sqrt(2n + 1)"),
    "an even rule": ("thermal", "300 K"),
    "classical": ("the classical bell curve", "300 K",
                  "no zero-point motion"),
    "n quanta": ("2 quanta", "standing wave"),
    "mean quanta": ("thermal", "0.4 mean quanta"),
    "a scaled frequency": ("thermal", "300 K"),
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
    # WHAT THE SET ASSUMES, in its own words: a mode's state as the toolkit
    # words it.
    assumptions = rows.value("assumptions")
    if getattr(frames, "state", None) is not None:
        assert all(w in assumptions for w in _STATE_WORDS[kind]), assumptions
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


@pytest.mark.parametrize("kind", list(_RECIPES))
def test_a_set_read_back_is_the_set_written(tmp_path, kind):
    """§ 5: a set read back is the set as written, every frame, frame 0
    included -- with its vibration given, its geometries, coordinates,
    weights and the state it was written for (a snapshot set's motion
    whole, its words the same); without it, its geometries, weights and
    largest moves still, and why it has no vibration said.  The silent
    failure: results paired with frames that are not the ones written, or
    a set re-read at another state than its own."""
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
        assert [(k, s.state) for k, s in back.motion.states.items()] == \
            [(k, s.state) for k, s in frames.motion.states.items()]
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
        from molbuilder.vibration import THE_SETS
        assert back.state.state == frames.state.state
        assert back.exact_to_degree == frames.exact_to_degree
        if frames.state.temperature_k is not None:
            assert back.state.temperature_from == THE_SETS
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
    assert (first_s.info or {}).get("sampling") == \
        (again_s.info or {}).get("sampling")


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
    ("frames of a bare state", "have no geometries"),
    ("a frame row named weight", "the writer's own"),
    ("a weight short", "one weight per frame"),
    ("no vibration", "with the vibration they are of"),
    ("a recipe given weights", "bring their own vibration"),
    ("a row named mass_source", "the writer's own"),
    ("a set read back without its vibration", "once its vibration is given"),
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
    (a calculation citing it is pinned to it), frames of a bare state, a
    person's row -- in rows, frame_rows or frame0_rows -- that is no
    mapping or is named like the writer's own, weights not one per frame,
    a person's frames without their vibration, a recipe's frames given what
    it brings, a set read back that is written again otherwise, a set of
    one frame.  The silent failure: a cited pair overwritten, or a frame's
    weight replaced by the person's row of that name."""
    from molbuilder.vibration import state
    from molbuilder.vibration_frames import FrameSetError, write_frame_set
    v = _model()
    path = tmp_path / "set.xyz"
    f = v.mode(1).state(temperature_k=300.0).frames(3)
    grid = [v.displaced({}), v.displaced({1: 0.1})]
    if call == "a pair already there":
        write_frame_set(path, f)
    with pytest.raises(FrameSetError, match=refused):
        if call in ("a pair already there", "a recipe given weights"):
            write_frame_set(path, f, *([[1.0, 0, 0]] if call ==
                                      "a recipe given weights" else []))
        elif call == "frames of a bare state":
            write_frame_set(path, state(0.1, 700.0, temperature_k=300.0)
                            .frames(3))
        elif call == "a frame row named weight":
            write_frame_set(path, grid, [0.5, 0.5], vibration=v,
                            frame_rows=[[{"name": "weight", "value": 0.5,
                                          "unit": "1"}]] * 2)
        elif call == "a weight short":
            write_frame_set(path, grid, [1.0], vibration=v)
        elif call == "a row named mass_source":
            write_frame_set(path, grid, [0.5, 0.5], vibration=v,
                            rows=[{"name": "mass_source", "value": "mine"}])
        elif call in ("a set read back without its vibration",
                      "snapshots read back without their vibration"):
            from molbuilder.vibration_frames import read_frame_set
            first = write_frame_set(tmp_path / "first.xyz", f if call ==
                                    "a set read back without its vibration"
                                    else v.motion(300.0).sample(
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
            write_frame_set(path, v.mode(1).state(temperature_k=300.0)
                            .frames(1))
        else:
            write_frame_set(path, grid, [0.5, 0.5])
    assert (path.exists() and call == "a pair already there") \
        or not path.exists()


def test_the_writer_says_what_it_writes_as_given(tmp_path):
    """§ 5: written and said, never refused -- weights not summing to 1, a
    person's frames moving a held atom.  The silent failure: an average at
    weights that are not shares, or a held atom moved unseen."""
    from molbuilder.vibration_frames import write_frame_set
    v = _model(held=True)
    moved = v.displaced({1: 0.1}).copy()
    moved[3] += 0.05
    got = write_frame_set(tmp_path / "set.xyz", [v.displaced({}), moved],
                          [0.5, 0.6], vibration=v,
                          rows=[{"name": "assumptions", "value": "mine"}])
    assert any("sum to 1.1" in w for w in got.said), got.said
    assert any("frame 2 of 2 moves a held atom" in w for w in got.said), \
        got.said


@pytest.mark.parametrize("asked", [dict(temperature_k=300.0), dict(quanta=2),
                                   dict(temperature_k=300.0, classical=True)],
                         ids=["thermal", "n quanta", "classical"])
@pytest.mark.parametrize("nodes", [3, 4])
def test_the_outermost_move_is_the_written_sets_for_any_state(tmp_path, asked,
                                                              nodes):
    """§ 4, § 5.1: a state's outermost move for its rule -- before a set is
    written -- is the largest atom's move at the outermost frames of the
    set written of it, its ``max_displacement_ang`` row exactly.  The
    silent failure: the page's readout and the set's row disagreeing."""
    from molbuilder.frameset import read
    from molbuilder.vibration_frames import outermost_move, write_frame_set
    from molbuilder.workingcopy_structure import StructureCodec
    v = _model()
    s = v.mode(1).state(**asked)
    f = s.frames(nodes)
    move = outermost_move(s, nodes)
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
        scaled).mode(2).state(temperature_k=300.0).frames(3)).path
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
    ("no rows", "FrameSetError", "states no frame set's rows"),
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
    set's), a person's set too, or that is no vibration; a structure that
    states no frame set's rows; a folder that holds no run of ours, a path
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
    snaps = v.motion(300.0).sample(4, seed=7, antithetic=True)
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
        elif call == "no rows":
            lone = StructureCodec().write(Structure.from_xyz(
                "2\nH2\nH 0 0 0\nH 0 0 0.74\n"), tmp_path / "lone.xyz")
            read_frame_set(lone)
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


#: EACH MODE SET'S FREQUENCY, SPREAD AND MEAN QUANTA, by science § 2b.3:
#: a quantum state's spread ``Q_zp sqrt(2 <n> + 1)``, its mean quanta the
#: thermal state's or a number state's own; a classical bell curve's
#: ``sqrt(k_B T) / omega``, counting no quanta.
_CLOSED_FORMS = {
    "thermal": (120.0, lambda q: (_n_th(120.0, 300.0), None)),
    "an even rule": (700.0, lambda q: (_n_th(700.0, 300.0), None)),
    "classical": (120.0, lambda q: (None, _classical_spread(120.0, 300.0))),
    "n quanta": (120.0, lambda q: (2.0, None)),
    "mean quanta": (1600.0, lambda q: (0.4, None)),
    "a scaled frequency": (0.96 * 700.0,
                           lambda q: (_n_th(0.96 * 700.0, 300.0), None)),
}


@pytest.mark.parametrize("kind", list(_CLOSED_FORMS))
def test_a_mode_sets_derived_rows_are_the_sciences(tmp_path, kind):
    """§ 5.1, science § 2b.3: a mode's own set states its frequency, its
    zero-point amplitude, its state's spread and mean quanta (none for a
    classical state), each displaced frame's coordinate in units of the
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
    mean, classical = forms(q_zp)
    spread = (classical if classical is not None
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
    (dict(mode=1.5, temperature_k=300.0), "a whole number"),
    (dict(mode=True, temperature_k=300.0), "a whole number"),
    (dict(mode=9, temperature_k=300.0), "numbered 1 to 3"),
    (dict(mode=1, nodes=0, temperature_k=300.0), "nodes is a whole number"),
    (dict(mode=1, temperature_k=300.0, imaginary=True), None),
    (dict(mode=1, temperature_k=300.0), "three nodes when unsaid"),
], ids=["a mode of 1.5", "a mode of True", "no such mode", "no nodes",
        "an imaginary mode", "the default rule"])
def test_the_outermost_move_of_a_mode_by_its_number(asked, answer):
    """§ 4: ``outermost_move_of(v, mode, *, nodes=None, **state)`` -- a mode
    named by its whole number (1.5 is no mode 1, True no mode at all), a
    rule refused by name (``FrameSetError``), an imaginary mode's ``None``,
    and three nodes when unsaid.  The silent failure: a move read for
    another mode than the one named, or for another rule."""
    from molbuilder.vibration_frames import (FrameSetError, outermost_move,
                                             outermost_move_of)
    asked = dict(asked)
    v = (_with_imaginary_mode_1(_model()) if asked.pop("imaginary", False)
         else _model())
    mode = asked.pop("mode")
    if answer is None:
        assert outermost_move_of(v, mode, **asked) is None
    elif answer == "three nodes when unsaid":
        s = v.mode(mode).state(**asked)
        assert outermost_move_of(v, mode, **asked) == outermost_move(s) \
            == outermost_move(s, 3)
    else:
        with pytest.raises(FrameSetError, match=answer):
            outermost_move_of(v, mode, **asked)


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
    said = write_frame_set(tmp_path / "set.xyz", v.mode(1).state(
        temperature_k=300.0).frames(3)).said
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
    frames = (v.motion(300.0).sample(2, seed=1, antithetic=True) if snapshots
              else v.mode(1).state(temperature_k=300.0).frames(3))
    path = write_frame_set(tmp_path / "set.xyz", frames).path
    back = read_frame_set(path)
    assert getattr(back, "state" if not snapshots else "motion") is None
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
        v.motion(300.0).sample(2, seed=1, antithetic=True)).path,
        frames=True))
    assert rows.value("modes") == "every real mode"
    assert ("every mode not included stays at the relaxed geometry "
            "(imaginary, no harmonic state: 1)") in rows.value("assumptions")
    assert rows.value("temperature_k") == 300.0
    # MODE 1 GIVEN A MEASURED FREQUENCY (§ 3.1) moves: no longer said still,
    # mode 3, left out, still said.
    measured = v.mode(1).with_frequency(100.0).state(temperature_k=300.0)
    added = read(StructureCodec().load(write_frame_set(
        tmp_path / "added.xyz", v.motion(300.0, modes=[2]).with_mode(
            1, state=measured).sample(2, seed=1, antithetic=True)).path,
        frames=True)).value("assumptions")
    assert "every mode not included stays at the relaxed geometry" in added
    assert "imaginary" not in added, added


# --------------------------------------------------------------------- #
#  The doors' state and what it means before a set is written (§ 6)     #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("asked, words, mean", [
    ({"temperature_k": 600.0}, "the thermal state at 600 K,",
     lambda: _n_th(120.0, 600.0)),
    ({"mean_quanta": 1.0}, "a thermal state of 1 mean quantum,", lambda: 1.0),
    ({"quanta": 2}, "2 quanta exactly,", lambda: 2.0),
], ids=["thermal", "one mean quantum", "n quanta"])
def test_what_a_state_means_for_a_mode_before_its_set_is_written(
        asked, words, mean):
    """`web/spectra.md` § 4.3's table, science § 2b.3: the readout under the
    dialog for each state -- the state in the set's words (§ 5.1), the mean
    quanta, the spread against the zero-point one (``sqrt(2 <n> + 1)``),
    the outermost move for each frame count the doors offer.  The silent
    failure: a dialog whose numbers or words are another state's than the
    frames written."""
    from molbuilder.vibration_frames import (OFFERED_NODES, outermost_move,
                                             readout)
    m = _model().mode(1)
    said = readout(m, asked)
    assert said["state"].startswith(words), said["state"]
    n = mean()
    assert said["mean_quanta"] == pytest.approx(n, rel=1e-6)
    ratio = math.sqrt(2.0 * n + 1.0)
    assert said["spread_over_zero_point"] == pytest.approx(ratio, rel=1e-6)
    assert said["spread_amu12_ang"] == pytest.approx(
        ratio * _q_zp(120.0), rel=1e-6)
    assert said["outermost_move_ang"] == {
        str(k): outermost_move(m.state(**asked), k) for k in OFFERED_NODES}
    assert readout(_with_imaginary_mode_1(_model()).mode(1), asked) is None


@pytest.mark.parametrize("asked, refused", [
    ({}, "asked every time, never assumed"),
    ({"temperature_k": "warm"}, "'warm' is not one"),
    ({"temperature_k": -5}, "-5 is not one"),
    ({"quanta": 2.5}, "quanta is a whole number"),
    ({"quanta": 2, "temperature_k": 300.0},
     "a temperature and a number of quanta name no one state"),
    ({"mean_quanta": -0.5}, "a mean number of quanta is 0 or more"),
], ids=["no state", "a temperature in words", "below zero",
        "half a quantum", "two states",
        "mean quanta below zero"])
def test_the_doors_ask_the_state_every_time(asked, refused):
    """§ 2's table, § 6: `frameset from-mode` and *Write frame set…* ask the
    state every time -- none named refused, never the record's temperature
    assumed; a temperature by the rule every door shares; two states at once
    refused in the doors' words, a value by the table's own rule
    (`vibration.checked_state`) -- each before any run is read, and the
    readout asking the same rule.  The silent failure: frames sampled at a
    state the person never stated."""
    from molbuilder.vibration_frames import StateRefused, doors_state, readout
    with pytest.raises(StateRefused, match=refused):
        doors_state(**asked)
    with pytest.raises(StateRefused, match=refused):
        readout(_model().mode(1), asked)


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
    said = readout(m, {"quanta": quanta})
    assert said["mean_quanta"] == quanta
    if warned:
        assert said["beyond_harmonic"].startswith(
            f"mode 3 holds {quanta} quanta in this state, {energy:.3g} eV")
    else:
        assert said["beyond_harmonic"] is None

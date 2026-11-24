"""A vibration's frame sets on the host: what a page says of a cited set's
rows, where the set came from, and the weights its rows give for a state
(`engines/vibration-frames.md` § 5.1, § 6, § 7).

API-level, as a person's script and the web layer ask: each set is a pair on
disk, made through the Structure API and the structure's one writer, and
read back by `vibration_frames.said_of` from its path; a set's rows are the
structure's own reader's (`frameset.read`), as `postprocess.frame_rows`
hands them.  Each expectation is the contract's: § 5.1's words and rule, § 6's
provenance and changed-pair words, § 7's refusals, and the Gauss-Hermite rule
from numpy (``hermgauss``) for a set's own weights.
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
    ("frames", "displacement_ang", "angstrom", "Displacement"),
    ("frames", "displacement_ang", None, "Displacement"),
    ("frames", "displacement_ang", "nm", None),
    ("frames", "weight", "1", "Weight"),
    ("frames", "weight", None, "Weight"),
    ("structure", "frequency_cm1", "cm-1", "Frequency"),
    ("structure", "frequency_cm1", "Hz", None),
    ("structure", "sigma_amu12_ang", "amu^1/2 angstrom", "Spread σ"),
    ("structure", "bond", None, None),
], ids=["its unit", "no unit stated", "another unit", "the dimensionless "
        "unit", "a unitless name, none stated", "a structure row",
        "a structure row in another unit", "a name written before 2.81.0",
        "a name it does not know"])
def test_a_page_says_a_row_in_the_vocabularys_words_where_its_unit_is_the_vocabularys(
        tmp_path, where, name, unit, words):
    """§ 5.1: a row the vocabulary names is said in its words where its unit
    is the vocabulary's or it states none -- the name is a claim about the
    unit -- an older set's name read the same; any other row by its name
    and unit as written (no words given).  The silent failure: a row
    written in nm labelled on the page with the vocabulary's angstrom."""
    from molbuilder.vibration_frames import said_of
    row = (name, 0.5, unit, "what it is")
    path = _pair(tmp_path, rows=[row] if where == "structure" else (),
                 frame_rows=[row] if where == "frames" else ())
    got = said_of(path)[where].get(name)
    assert (got and got["words"]) == words, got


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
    path = _pair(tmp_path, frame_rows=[("weight", 0.5, None, note)], n=3)
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
    path = _pair(tmp_path, rows=[("mode_index_1based", 6, None, None),
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
    ([("mode_index_1based", 6, None, None)], "frames of one mode"),
    ([("samples", 64, None, None)],
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
    beside it; a person's driven set said to be thermal."""
    from molbuilder.vibration_frames import (assumptions_of_a_mode, said_of,
                                             thermal_state_words)
    rows = [("mode_index_1based", 6, None, None),
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
        # (§ 6): the generator writes its row by these same two calls.
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
              extra=(), without=(), q_unit="amu^1/2 angstrom",
              no_q_at=None):
    """A mode's own set's rows as `frameset.read` hands them, made through
    the Structure API as a script makes one: its frames at the m-point
    Gauss-Hermite rule's nodes for the thermal spread at ``temperature_k``,
    each weighted by the rule -- the set's own weights; ``extra`` structure
    rows ``(name, value)`` added, the structure rows named in ``without``
    left out, the coordinate written in ``q_unit``, frame ``no_q_at``
    stating none."""
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
    stated = [("mode_index_1based", 7, None), ("frequency_cm1", _NU, "cm-1"),
              ("zero_point_amplitude_amu12_ang", _QZP, "amu^1/2 angstrom")]
    if with_temperature:
        stated.append(("temperature_k", temperature_k, "K"))
    for name, value, unit in [*stated, *((n, v, None) for n, v in extra)]:
        if name not in without:
            s.set_customized(name, value, unit=unit)
    for f in range(m):
        if f != no_q_at:
            s.set_customized("q_amu12_ang", math.sqrt(2.0) * sigma * x[f],
                             unit=q_unit, frame=f)
        s.set_customized("weight", w[f] / math.sqrt(math.pi), frame=f)
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
        s.set_customized("weight", float(w[f]), frame=f)
    got = weights_for_set(read(s), temperature_k=300.0)
    assert got.weights == pytest.approx(list(w), abs=1e-12)
    assert got.said == (), got.said


def test_a_swing_beyond_the_frames_is_said_not_refused():
    """§ 9, § 7: a state reaching past the outermost frames is averaged and
    the share beyond said -- never refused.  Three frames' blindness to the
    shape is said with it (science § 5.4).  The silent failure: a swing the
    frames never reached averaged from values extrapolated past them,
    unsaid -- or a person's question refused where an answer with its
    caveat stands."""
    from molbuilder.vibration_frames import weights_for_set
    got = weights_for_set(_mode_set(3, temperature_k=0.0), swing_quanta=9)
    assert got.beyond > 0.0, got
    assert math.fsum(got.weights) == pytest.approx(1.0, abs=1e-12)
    assert any("beyond the outermost frames" in s for s in got.said), got
    assert any("blind to the state's shape" in s for s in got.said), got


def test_a_swing_named_alone_on_a_warm_set_says_its_temperature_is_not_used():
    """§ 2's table, science § 6: a swing named by its quanta alone is the
    cold coherent state, whatever the set's temperature -- said beside the
    weights, with the way to the warm one, so the cold and the warm are
    never confused silently."""
    from molbuilder.vibration_frames import weights_for_set
    got = weights_for_set(_mode_set(3), swing_quanta=1)
    assert got.said[0] == (
        "the cold coherent state, 1 swing quantum: the set's temperature, "
        "300 K, is not used -- temperature_k names the environment whose "
        "thermal width the swing carries"), got.said
    classical = weights_for_set(_mode_set(3), swing_quanta=1, classical=True)
    assert classical.said[0].startswith("the classical swing alone, "), \
        classical.said
    warm = weights_for_set(_mode_set(3), swing_quanta=1, temperature_k=300.0)
    cold_set = weights_for_set(_mode_set(3, temperature_k=0.0),
                               swing_quanta=1)
    for quiet in (warm, cold_set):
        assert not any("is not used" in s for s in quiet.said), quiet.said
    thermal = weights_for_set(_mode_set(3), classical=True)
    assert thermal.said[0] == ("the classical state at the set's "
                               "temperature, 300 K (its temperature_k row)")


def test_a_coordinate_in_another_unit_is_read_as_written_and_said():
    """§ 5.1, the product's stance (foreign input gets a warning): a set's
    coordinate written in another unit than its name says is read as the
    name's unit -- the weights given -- and said.  The silent failure:
    weights computed from coordinates a hundred times off, unsaid."""
    from molbuilder.vibration_frames import weights_for_set
    got = weights_for_set(_mode_set(3, q_unit="nm"))
    assert any(s.startswith("q_amu12_ang is written in another unit")
               for s in got.said), got.said


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
    -- n quanta, a cold driven mode -- needs none.  The silent failure:
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
        assert len(weights_for_set(got, swing_quanta=1).weights) == 3

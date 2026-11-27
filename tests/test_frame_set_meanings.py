"""A frame set's rows are carried as written wherever a transport
calculation cites them, and the one row it reads is each frame's weight.

`model/structure.md` § 2.2f; `engines/transport.md` § 2a.12, § 3.8.1: the
calculation reads each frame's ``weight`` and carries every other row whole
-- name, value, unit and meaning -- never checking it, and a page is
answered each by its stored name, value and unit, none interpreted (the
user, 2026-11-26: "transport does transport calculation and that's it");
what is said beside the average is the weights' and transport's own
assumptions, said of frames, and of a calculation's citation its pair no
longer as pinned.  API-level: the frame set is made through the Structure
API, as a person's script makes it, and each of transport's readers is
asked -- the page's answer, the record's average and citation,
`summarize`'s lines, the data file's frame part and the citation's words.

The silent failures it guards: a person's own row refused by a calculation,
or relabelled with words it never had; a maker's note drawn on a transport
page; a harmonic mode's assumptions stated beside frames a person built.
"""
import math
import re

import pytest

from molbuilder.transport.stages import same_volts

_OWN = "a Boltzmann weight at 300 K over the frames' harmonic energies"


def _frames(weights=(0.5, 0.5), note=_OWN):
    """Frames of H2, each stating its ``weight`` (``None``: none) with
    ``note``, and a stretch of its maker's own in nm; the structure a mode's
    number, a frequency at full precision -- its unit in its name, its
    ``unit`` blank (`model/structure.md` § 2.2d) -- and a row of its
    maker's own."""
    from molbuilder.structure import Structure
    s = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
    s = s.with_frames([s.positions] * len(weights))
    for f, w in enumerate(weights):
        if w is not None:
            s.set_customized("weight", w, unit="1", frame=f, note=note)
        s.set_customized("bond_stretch", 0.05 * f, unit="nm", frame=f)
    s.set_customized("mode_index_1based", 6, unit="1")
    s.set_customized("frequency_cm1", 2548.365564447995,
                     note="the mode's wavenumber")
    s.set_customized("seed", 20261123, unit="1")
    s.set_customized("sampling_nodes", 3, unit="1")
    s.set_customized("bond", "H-H", note="the bond the frames stretch")
    return s


def test_the_page_is_answered_the_rows_as_the_set_stores_them():
    """§ 3.8.1: the describe door's frames are the rows as the set stores
    them -- each name, value and unit -- a person's own row in its own unit
    carried, none interpreted: no words of transport's own and no meaning
    drawn, the maker's notes staying in its file.  The silent failures: a
    row of the person's relabelled, or the citation refused for it; a
    maker's note answered for a transport page to draw."""
    from molbuilder.frame_average import answer
    got = answer(_frames())
    assert set(got) == {"n_frames", "definition", "said"}, got
    assert got["said"] == [], got
    definition = got["definition"]
    assert definition["structure"] == [
        {"name": "mode_index_1based", "value": 6, "unit": "1"},
        {"name": "frequency_cm1", "value": 2548.365564447995, "unit": None},
        {"name": "seed", "value": 20261123, "unit": "1"},
        {"name": "sampling_nodes", "value": 3, "unit": "1"},
        {"name": "bond", "value": "H-H", "unit": None}]
    assert definition["columns"] == [
        {"name": "weight", "unit": "1"},
        {"name": "bond_stretch", "unit": "nm"}]
    assert definition["frames"] == [[0.5, 0.0], [0.5, 0.05]]
    assert set(definition) == {"structure", "columns", "frames", "weights",
                               "weight_sum", "warnings"}, definition


@pytest.mark.parametrize("weights, average, warned", [
    ((0.5, 0.5), True, 0),
    ((0.6, 0.6), True, 1),
    ((0.5, None), False, 1),
], ids=["weights summing to 1", "weights off 1", "a frame stating none"])
def test_the_record_reads_the_weights_alone_and_says_its_own_assumptions(
        weights, average, warned):
    """§ 2a.12: the average is taken at each frame's stated weight where
    every frame states one, a sum off 1 averaged as written and warned of;
    what is said is the weights' alone -- never a row of the maker's -- and
    transport's own five assumptions, said of frames, stand beside an
    average and only beside one.  The silent failure: a harmonic mode's
    assumption stated beside frames a person built."""
    from molbuilder.frame_average import ASSUMPTIONS, average_block
    from molbuilder.frameset import read
    block = average_block(2, read(_frames(weights)), [], [],
                          same_volts)
    assert (block["weights"] is not None) is average, block
    assert len(block["warnings"]) == warned, block
    assert not any("bond_stretch" in w for w in block["warnings"])
    assert block["assumptions"] == (list(ASSUMPTIONS) if average else [])
    # TRANSPORT'S FIVE (science § 7, transport's own) hold for ANY frames --
    # a fixed geometry, one set of weights at every voltage, each frame its
    # own ground state, DFT's alignment, the electrodes unmoved -- and say
    # nothing of how the frames were made: a vibration's are its maker's
    # own row (the user, 2026-11-26).
    assert len(ASSUMPTIONS) == 5
    vibration_words = re.compile(
        r"\b(vibrations?|hbar|omega|polarons?|inelastic|heat\w*|modes?|"
        r"quant\w*|motions?)\b")
    assert [a for a in ASSUMPTIONS if vibration_words.search(a.lower())] \
        == [], ASSUMPTIONS


def test_a_rows_meaning_is_read_as_written():
    """`model/structure.md` § 2.2f: the one reader carries each row's note as
    its meaning -- the structure's row's own, a frame row's the first
    meaning any frame writes -- and ``None`` where none is written, as a
    person's script asks it of `postprocess.frame_rows`.  The silent
    failure: a meaning drawn from another row, or a blank note answered as
    one."""
    from molbuilder.frameset import read
    rows = read(_frames())
    assert rows.meaning("frequency_cm1") == "the mode's wavenumber"
    assert rows.meaning("weight", of_frames=True) == _OWN
    assert rows.meaning("weight") is None
    assert rows.meaning("seed") is None
    assert rows.meaning("bond_stretch", of_frames=True) is None


def test_summarize_prints_each_row_as_written():
    """§ 3.8.1: `summarize` prints the set's rows by name, value and unit as
    written -- a whole number whole, a number at its full precision.  The
    silent failure: a seed printed to six figures (20261100), a different
    set than the one written."""
    from molbuilder.frame_average import (average_block, citation_block,
                                          summary_lines)
    from molbuilder.frameset import read
    s = _frames()
    lines = summary_lines(average_block(2, read(s), [], [],
                                        same_volts),
                          citation_block(s, {}))
    rows = lines[lines.index("  the frame set's rows, as written:") + 1:]
    assert rows == ["    mode_index_1based 6",
                    "    frequency_cm1 2548.365564447995",
                    "    seed 20261123", "    sampling_nodes 3",
                    "    bond H-H"], lines


def test_frames_with_no_average_say_why():
    """§ 2a.12: frames that state no rows are a family with no average, said
    so -- told apart from a composed junction that does not read, whose
    frames are counted but not read.  The silent failure: a family's
    calculation told its junction is broken."""
    from molbuilder.frame_average import average_block
    family = average_block(3, None, [], [], same_volts)
    unread = average_block(3, None, [], [], same_volts,
                           unread="its sidecar does not read")
    assert "family" in family["why"] and "does not read" not in \
        family["why"], family["why"]
    assert "does not read" in unread["why"], unread["why"]
    for block in (family, unread):
        assert block["at"] == [] and block["assumptions"] == [], block


@pytest.mark.parametrize("page", ["the citation", "the Results report"])
def test_a_page_opens_whatever_rows_a_set_writes(page):
    """§ 3.8.1, `model/structure.md` § 2.2f: a calculation refuses no row --
    a set with a person's own row in its own unit has its citation's frames
    answered as written, that row among them, and its Results report's
    average taken at its weights.  The silent failure: the
    Transport tab's picker, Task setup's folder or the Results tab answering
    an error, or no average, for a person's own row."""
    from molbuilder.frame_average import answer, average_said
    from molbuilder.frameset import read
    s = _frames()
    if page == "the citation":
        columns = answer(s)["definition"]["columns"]
        assert {"name": "bond_stretch", "unit": "nm"} in columns, columns
        return
    said = average_said(2, read(s), [], [], same_volts)
    assert said["why"] is None and said["weights"] == [0.5, 0.5], said


@pytest.mark.parametrize("frames", [1, 2], ids=["one structure", "a set"])
def test_the_record_cites_a_frame_set_and_only_a_frame_set(frames):
    """§ 2a.12: the record's ``frame_set`` is the pair cited -- its path and
    its files' sha256 as the citation pinned them -- and the set's rows as
    written; null where the citation is one structure.  Nothing of its
    maker's ``info`` or channels: those are the pair's.  The silent failure:
    a script reading ``record["frame_set"]`` as "a frame set was cited" for
    one structure."""
    from molbuilder.frame_average import citation_block
    from molbuilder.structure import Structure
    s = (_frames() if frames == 2
         else Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n"))
    block = citation_block(s, {"citation": "P/structure/h2.xyz",
                               "files": {"h2.xyz": "ab" * 32}})
    if frames == 1:
        assert block is None, block
        return
    assert block == {"citation": "P/structure/h2.xyz",
                     "files": {"h2.xyz": "ab" * 32},
                     "rows": s.customized_rows()}, block


def test_the_data_file_writes_each_frames_weight_and_no_other_row():
    """`model/datafiles.md` § 9.1 (@14): the frames' part is each frame's
    ``weight`` -- the fill value where a frame states none -- their sum and
    tolerance and the average; every other row is the junction's
    ``structure_json``'s, never written a second time.  The silent failure:
    a maker's row laid out by transport's own table, under its words."""
    from molbuilder.frame_average import (average_block, data_file_variables,
                                          fill_data_file)
    from molbuilder.frameset import read
    from molbuilder.ncdata import empty
    s = _frames((0.5, None))
    average = average_block(2, read(s), [], [], same_volts)
    sizes = {"frame": 2, "bias_v": 0, "energy_ev": 0}
    values = {v.name: empty(v, sizes) for v in data_file_variables()}
    attrs = {}
    fill_data_file(values, attrs, s, average, [], lambda v, vs: 0)
    assert values["weight"][0] == 0.5 and math.isnan(values["weight"][1])
    assert {v.name for v in data_file_variables()} == {
        "weight", "weight_sum", "weight_tolerance", "average_transmission",
        "average_delta_transmission", "average_conductance_g0",
        "average_conductance_change_percent", "average_why",
        "average_conductance_why"}
    assert attrs["frame_warnings"] == average["warnings"], attrs


@pytest.mark.parametrize("edited, changed, unchanged", [
    ("the .xyz", ["set.xyz"], "set.molstruct.json"),
    ("the sidecar", ["set.molstruct.json"], "set.xyz"),
    ("nothing", [], None),
])
def test_a_calculations_cited_pair_changed_since_is_said(tmp_path, edited,
                                                         changed, unchanged):
    """§ 3.8.1, § 3.1: a calculation's cited pair either of whose files no
    longer holds the bytes its record pinned is said, naming the file, with
    its way on -- transport's own word of its citation, the frames shown
    being the ones the calculation ran; a pair as pinned says nothing.  The
    silent failure: a Results tab showing a pair's rows as the ones the
    calculation ran after a person edited it."""
    from molbuilder.persist import sha256_of
    from molbuilder.transport.compose import pair_changed_said
    from molbuilder.workingcopy_structure import StructureCodec
    path = StructureCodec().write(_frames(), tmp_path / "set.xyz")
    sidecar = path.with_name("set.molstruct.json")
    pinned = {path.name: sha256_of(path), sidecar.name: sha256_of(sidecar)}
    # THE PERSON EDITS ONE FILE after the record pinned it, as an editor
    # saves it: its bytes another.
    target = {"the .xyz": path, "the sidecar": sidecar}.get(edited)
    if target is not None:
        target.write_text(target.read_text() + "\n")
    said = pair_changed_said(path, pinned)
    if not changed:
        assert said is None, said
        return
    assert said.startswith("the cited pair has changed since this "
                           "calculation cited it"), said
    assert all(name in said for name in changed), said
    assert unchanged not in said, said
    assert "the frames shown are the ones the calculation ran" in said
    assert said.endswith("cite the pair as it is now in a new calculation, "
                         "or put it back"), said

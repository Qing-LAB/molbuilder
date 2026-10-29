"""A frame set's rows keep what they mean wherever they are cited.

`model/structure.md` § 2.2f: every row of the vocabulary is written with its
meaning as its ``note``; a citation carries the row whole, and a page and
the data file say the meaning the set writes -- the vocabulary's where it
writes none -- and a vocabulary row written without one is said beside the
frames (`model/datafiles.md` § 2, `engines/transport.md` § 2a.12, § 3.8.1).
API-level: the frame set is made through the Structure API, as a person's
script makes it, and each reader is asked.

The silent failure it guards: a script's frames weighted by its own rule --
a Boltzmann weight at 300 K, say -- cited into a data file whose ``weight``
definition still claims the Gauss-Hermite rule's share, or a page whose
hover says so: a number labelled with a meaning it does not have.
"""
import pytest

_OWN = "a Boltzmann weight at 300 K over the frames' harmonic energies"


def _frames(note=None):
    """Two frames of H2, each stating its ``weight`` with ``note`` as its
    meaning and its displacement with the vocabulary's; the structure its
    ``temperature_k`` with ``note``, and a row the vocabulary does not
    know."""
    from molbuilder.frame_average import ROWS
    from molbuilder.structure import Structure
    means = {r.name: r.means for r in ROWS}
    s = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
    s = s.with_frames([s.positions, s.positions])
    for f in range(2):
        s.set_customized("weight", 0.5, frame=f, note=note)
        s.set_customized("displacement_ang", 0.1 * f, unit="angstrom",
                         frame=f, note=means["displacement_ang"])
    s.set_customized("temperature_k", 300.0, unit="K", note=note)
    s.set_customized("bond", "H-H", note="the bond the frames stretch")
    return s


def _said(reader, s, name):
    """What ``reader`` says row ``name`` means."""
    from molbuilder.frame_average import answer, with_meanings
    from molbuilder.transport.datafile import VARIABLES
    if reader == "data file":
        return next(v.definition for v in with_meanings(VARIABLES, s)
                    if v.name == name)
    definition = answer(s)["definition"]
    return next(r["means"] for r in definition["structure"]
                + definition["columns"] if r["name"] == name)


@pytest.mark.parametrize("row", ["weight", "temperature_k"],
                         ids=["a frame row", "a structure row"])
@pytest.mark.parametrize("reader", ["data file", "page"])
@pytest.mark.parametrize("note", [_OWN, None], ids=["the set's", "none"])
def test_a_cited_row_says_the_meaning_the_set_writes(row, reader, note):
    from molbuilder.frame_average import ROWS
    vocabulary = next(r.means for r in ROWS if r.name == row)
    said = _said(reader, _frames(note), row)
    assert (note or vocabulary) in said, said
    if note:
        assert vocabulary not in said, said


def test_a_row_the_vocabulary_does_not_know_keeps_its_own_meaning():
    assert _said("page", _frames(), "bond") == "the bond the frames stretch"


@pytest.mark.parametrize("reader", ["record", "data file", "page"])
def test_a_vocabulary_row_without_its_meaning_is_said_once_by_name(reader):
    """...beside the frames, at every reader, never refused: one line naming
    the row, however many frames write it without a note."""
    import json
    from molbuilder.frame_average import (answer, average_block,
                                          data_file_variables, fill_data_file)
    from molbuilder.frameset import read
    from molbuilder.ncdata import Var, empty
    s = _frames(_OWN)
    for f in range(2):
        s.set_customized("displacement_ang", 0.1 * f, unit="angstrom",
                         frame=f)
    average = average_block(2, read(s), [], [], lambda a, b: a == b)
    if reader == "record":
        said = average["warnings"]
    elif reader == "data file":
        attrs = {}
        # THE FRAME PART'S VALUES as the writer starts them, from its table.
        sizes = {"frame": 2, "atom": 2, "bias_v": 0, "energy_ev": 0}
        values = {v.name: empty(v, sizes) for v in data_file_variables(Var)}
        fill_data_file(values, attrs, s, average, [], lambda v, vs: 0)
        said = json.loads(attrs["frame_warnings"])
    else:
        said = answer(s)["definition"]["warnings"]
    assert len(said) == 1 and said[0].count("displacement_ang") == 1, said
    assert "weight" not in said[0], said


def test_summarize_says_a_sets_rows_as_the_page_does():
    """`model/structure.md` § 2.2f: a set's rows are said in the
    vocabulary's words and page units on every surface -- the line
    `summarize` prints as the frame card does (`engines/transport.md`
    § 3.8.1, *Mode 6 · Frequency 2548.37 cm⁻¹*).

    THE SILENT FAILURE: the terminal printing the stored names
    (``mode_index_1based 6, frequency_cm1 2548.365564447995 cm-1``) where
    the page says them in words."""
    from molbuilder.frame_average import ROWS, average_block, summary_lines
    from molbuilder.frameset import read
    means = {r.name: r.means for r in ROWS}
    s = _frames(_OWN)
    s.set_customized("mode_index_1based", 6,
                     note=means["mode_index_1based"])
    s.set_customized("frequency_cm1", 2548.365564447995, unit="cm-1",
                     note=means["frequency_cm1"])
    head = summary_lines(average_block(2, read(s), [], [],
                                       lambda a, b: a == b))[0]
    assert "Mode 6" in head and "Frequency 2548.37 cm⁻¹" in head, head
    assert "mode_index_1based" not in head, head


@pytest.mark.parametrize("frames", [1, 2], ids=["one structure", "a set"])
def test_the_record_cites_a_frame_set_and_only_a_frame_set(frames):
    """`engines/transport.md` § 2a.12: the record's ``frame_set`` is the
    frame set the average is taken over, cited whole -- its rows with their
    meanings -- and null where the citation is one structure.

    THE SILENT FAILURE: a script reading ``record["frame_set"]`` as "a frame
    set was cited" for a calculation of one structure, its block holding
    the run's citation under a frame set's name."""
    from molbuilder.frame_average import citation_block
    from molbuilder.structure import Structure
    s = (_frames(_OWN) if frames == 2
         else Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n"))
    block = citation_block(s, {"citation": "P/structure/h2.xyz"})
    if frames == 1:
        assert block is None, block
    else:
        assert block["citation"] == "P/structure/h2.xyz", block
        assert block["rows"] == s.customized_rows(), block

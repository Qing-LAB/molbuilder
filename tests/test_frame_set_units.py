"""A frame set's vocabulary rows keep their unit at every reader.

`model/structure.md` § 2.2f: a row the vocabulary names carries its unit in
its name, and one written with another unit is refused by name wherever the
rows are read -- for the transport record's average, for its data file, and
for a page.  API-level: the frame set is made through the Structure API, as a
person's script makes it, and each reader is asked.

The silent failure it guards: ``displacement_ang`` written in nm by a script,
then averaged and written into the data file under the vocabulary's
"angstrom" -- a number ten times off, labelled as right.
"""
import pytest


def _frames_with_a_row_in_nm():
    from molbuilder.structure import Structure
    s = Structure.from_xyz("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
    s = s.with_frames([s.positions, s.positions])
    s.set_customized("weight", 0.5, frame=0)
    s.set_customized("weight", 0.5, frame=1)
    s.set_customized("displacement_ang", 0.005, unit="nm", frame=0)
    return s


@pytest.mark.parametrize("reader", ["record", "data file", "page"])
def test_a_vocabulary_row_in_another_unit_is_refused_by_name(reader):
    from molbuilder.frame_average import (AverageError, answer,
                                          average_block, fill_data_file)
    from molbuilder.frameset import read
    s = _frames_with_a_row_in_nm()
    with pytest.raises(AverageError, match="displacement_ang.*'nm'"):
        if reader == "record":
            average_block(2, read(s), [], [], lambda a, b: a == b)
        elif reader == "data file":
            fill_data_file({}, {}, s, {}, [], lambda v, vs: 0)
        else:
            answer(s)

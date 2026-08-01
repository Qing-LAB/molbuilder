"""``Structure.to_extxyz`` -- the one writer for a structure that has a cell, or
more than one frame.

Derived from ``docs/web/molview.md`` § 11.3 (what a Trajectory export produces)
and § 11.7 (the server is the only thing that writes a file), plus the
same-atoms rule every frame model rests on.

WHY IT EXISTS BESIDE ``to_xyz``.  A plain ``.xyz`` has one comment line per frame
and nowhere to put a cell, so a periodic structure written that way loses its
box -- and a trajectory written that way loses it on every frame.  A trajectory
that has lost its cell is not the thing that was computed.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure


def _isolated() -> Structure:
    return Structure.from_xyz(
        "3\nwater\nO 0 0 0\nH 0.957 0 0\nH -0.239 0.927 0\n")


def _periodic() -> Structure:
    s = Structure.from_xyz("2\nAu2\nAu 0 0 0\nAu 2 2 0\n")
    s.cell = [[8, 0, 0], [0, 8, 0], [0, 0, 8]]
    s.axis_kind = ("periodic", "periodic", "isolated")
    s.__post_init__()
    return s


def _blocks(text: str):
    """Split extended XYZ into (count_line, comment_line, atom_lines) blocks."""
    lines = text.splitlines()
    out, at = [], 0
    while at < len(lines):
        n = int(lines[at])
        out.append((n, lines[at + 1], lines[at + 2:at + 2 + n]))
        at += 2 + n
    return out


# --------------------------------------------------------------------- #
#  One frame, or many                                                    #
# --------------------------------------------------------------------- #

def test_one_frame_writes_one_block_and_many_write_one_each_in_order():
    """§ 11.3: a range writes "every frame as one extended-XYZ document"."""
    s = _periodic()
    shifted = [s.positions + np.array([d, 0.0, 0.0]) for d in (0.0, 0.1, 0.2)]

    assert len(_blocks(s.to_extxyz())) == 1, "no frames given -> this structure"

    blocks = _blocks(s.to_extxyz(frames=shifted))
    assert len(blocks) == 3, f"one block per frame: {len(blocks)}"
    # IN ORDER, and each carrying its own coordinates -- a trajectory whose
    # frames arrive shuffled is a different trajectory.
    first_x = [float(b[2][0].split()[1]) for b in blocks]
    assert first_x == pytest.approx([0.0, 0.1, 0.2]), (
        f"the frames are not in the order they were given: {first_x}")


def test_every_frame_carries_the_same_atoms_or_nothing_is_written():
    """The same-atoms rule: the elements and the cell are written from the
    structure and shared by every block, which is what makes this ONE trajectory
    rather than a pile of structures. A frame that does not fit is refused
    rather than padded, truncated or guessed into fitting."""
    s = _periodic()
    with pytest.raises(ValueError, match="frame 1 has 1 atoms"):
        s.to_extxyz(frames=[s.positions, [[0, 0, 0]]])
    with pytest.raises(ValueError, match="at least one frame"):
        s.to_extxyz(frames=[])


# --------------------------------------------------------------------- #
#  The cell, and the honesty of it                                       #
# --------------------------------------------------------------------- #

def test_the_lattice_is_the_cell_as_it_will_actually_be_used():
    """The box written is ``resolve_cell()`` -- the same one MolView draws and
    the Cell page reports (molview.md § 9.3). A file and the viewer it came from
    must not describe different systems."""
    s = _periodic()
    line = _blocks(s.to_extxyz())[0][1]
    assert 'Lattice="' in line
    written = [float(v) for v in line.split('Lattice="')[1].split('"')[0].split()]
    assert written == pytest.approx(
        [v for row in np.asarray(s.resolve_cell()) for v in row])
    assert "Properties=species:S:1:pos:R:3" in line, (
        "the Properties key is what tells a reader how to parse the columns")


def test_an_isolated_structure_says_so_even_though_it_has_a_box():
    """The honesty this pairing exists for.

    An isolated molecule still HAS a resolved box -- its bounding box plus
    vacuum -- and writing that Lattice alone would tell a reader the system
    repeats when it does not. The ``pbc`` flags are what keep it truthful, and
    they come from the structure's own axis kinds.
    """
    isolated = _blocks(_isolated().to_extxyz())[0][1]
    assert 'pbc="F F F"' in isolated, (
        f"an isolated structure was written as periodic: {isolated}")
    assert 'Lattice="' in isolated, (
        "the box it will actually be computed in is still worth writing")

    mixed = _blocks(_periodic().to_extxyz())[0][1]
    assert 'pbc="T T F"' in mixed, (
        f"the periodic axes are not reported per axis: {mixed}")


# --------------------------------------------------------------------- #
#  It must not drift from to_xyz                                         #
# --------------------------------------------------------------------- #

def test_the_atom_lines_are_byte_identical_to_the_plain_writer():
    """Two writers of one thing drift -- "a title line here, a decimal place
    there" (molview.md § 11.7). The coordinate rows are the part both emit, so
    they are compared directly rather than trusted to match.
    """
    s = _isolated()
    plain = s.to_xyz().splitlines()[2:]
    ext = _blocks(s.to_extxyz())[0][2]
    assert ext == plain, (
        "the two writers format the same atoms differently, which is how the "
        f"same structure comes to produce two different files:\n{ext}\n{plain}")


def test_the_title_survives_and_the_metadata_rides_beside_it():
    s = _isolated()
    comment = _blocks(s.to_extxyz())[0][1]
    assert comment.startswith("water "), (
        f"the title must still be readable by a human first: {comment}")
    assert _blocks(s.to_extxyz(comment="chosen"))[0][1].startswith("chosen ")


def test_it_writes_utf8_regardless_of_the_platform_locale(tmp_path):
    """The rule ``to_xyz`` already follows: an explicit encoding, never the
    platform's, or a non-ASCII title is silently corrupted on cp1252."""
    s = _isolated()
    s.title = "wasser — Ångström"
    path = tmp_path / "t.extxyz"
    returned = s.to_extxyz(str(path), frames=[s.positions, s.positions])
    assert path.read_text(encoding="utf-8") == returned
    assert "Ångström" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
#  It reads back                                                         #
# --------------------------------------------------------------------- #

def test_what_it_writes_the_plain_reader_can_still_open():
    """Extended XYZ is XYZ with a richer comment line, so a reader that ignores
    the comment still gets the geometry. A format only this project can read
    would be a worse answer than the plain ``.xyz`` it replaces."""
    s = _periodic()
    back = Structure.from_xyz(s.to_extxyz())
    assert list(back.elements) == list(s.elements)
    assert back.positions == pytest.approx(s.positions)

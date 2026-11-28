"""MOLBUILDER'S EXAMPLE NOTEBOOKS IN THE PROJECTS TREE -- `molbuilder
examples` (`execution/project-layout.md` § 2.6; `ops/installation.md`
§ 2.1; the user, 2026-11-27: "all installation will cleanly put the correct
version there", and of an example a person edited, "Replace it").

Through the command as a person types it, in a projects tree of the test's
own: every notebook molbuilder ships (its package's `data/examples/`) lands in
the tree's `notebooks/examples/` as the shipped bytes, each said; a file there
molbuilder does not ship is left alone; and nothing else is written into the
tree -- the run folder an example reads stays in the package, where the tree
would show it as a calculation.  The installation's own call of the same door
is `tests/test_seeding_a_fresh_machine.py`'s.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner


def _shipped():
    """Every example notebook molbuilder ships, by name: its bytes."""
    from molbuilder import data_dir
    found = {p.name: p.read_bytes()
             for p in sorted((data_dir() / "examples").glob("*.ipynb"))}
    assert found, "molbuilder ships no example notebook"
    return found


#: WHAT THE FOLDER HOLDS OF EACH SHIPPED NOTEBOOK BEFORE, and the word the
#: command says of it (`project-layout.md` § 2.6: "each said as written,
#: replaced or kept").
_ROWS = {
    "a tree with no examples yet": (None, "written"),
    "an example the person changed": (b'{"cells": []}', "replaced"),
    "an example already the shipped one": ("shipped", "kept"),
}


@pytest.mark.parametrize("row", list(_ROWS))
def test_molbuilder_examples_writes_the_shipped_notebooks_into_the_tree(
        row, isolated_projects_root):
    """`project-layout.md` § 2.6: `molbuilder examples` writes each notebook
    molbuilder ships into `<projects>/notebooks/examples/`, the current
    version in place of the one there, and says which it did; a file there
    molbuilder does not ship is left as it is; nothing else lands in the
    tree.  THE SILENT FAILURES: a person's Jupyter tab opening an example an
    older molbuilder wrote, a notebook of theirs beside it written over, and
    the example's run folder showing in their tree as a calculation."""
    from molbuilder.cli import cli
    before, word = _ROWS[row]
    shipped = _shipped()
    examples = isolated_projects_root / "notebooks" / "examples"
    mine = examples / "mine.ipynb"
    examples.mkdir(parents=True)
    mine.write_bytes(b'{"mine": true}')
    if before is not None:
        for name, data in shipped.items():
            (examples / name).write_bytes(data if before == "shipped"
                                          else before)
    r = CliRunner().invoke(cli, ["examples"])
    assert r.exit_code == 0, r.output
    for name, data in shipped.items():
        assert (examples / name).read_bytes() == data, name
        said = [line for line in r.output.splitlines() if name in line]
        assert len(said) == 1 and word in said[0], (name, r.output)
    assert mine.read_bytes() == b'{"mine": true}'
    assert "mine.ipynb" not in r.output, r.output
    written = sorted(p.relative_to(isolated_projects_root).as_posix()
                     for p in isolated_projects_root.rglob("*") if p.is_file())
    assert written == sorted([f"notebooks/examples/{n}" for n in shipped]
                             + ["notebooks/examples/mine.ipynb"]), written


def test_a_tree_that_cannot_be_written_is_refused_in_words(
        isolated_projects_root):
    """`project-layout.md` § 2.6: `molbuilder examples` refuses a projects
    tree it cannot write into -- in words, naming the folder, exit 1, no
    traceback.  THE SILENT FAILURE: a person asking for the examples and
    meeting a Python traceback."""
    from molbuilder.cli import cli
    isolated_projects_root.chmod(0o555)
    try:
        r = CliRunner().invoke(cli, ["examples"])
    finally:
        isolated_projects_root.chmod(0o755)
    assert r.exit_code == 1, (r.output, r.exception)
    assert isinstance(r.exception, SystemExit), r.exception
    assert "could not be written" in r.output, r.output
    assert str(isolated_projects_root / "notebooks" / "examples") in r.output

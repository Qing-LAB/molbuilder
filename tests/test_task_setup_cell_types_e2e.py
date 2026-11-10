"""Typing a k-grid into the stage table writes a k-grid.

The failure this closes (reported live against Au-BDT-Au, 2026-08-25)
====================================================================

``prep bench`` refused a stage with::

    [config.kgrid] Field metadata for ``kgrid`` declares ``range = (1, 64)``
    but the value '4,4,1' is not comparable with it.  This is a programmer
    bug: either the range or the field's type is wrong.
    [config.kgrid] kgrid must be a 3-tuple of ints; got '4,4,1'

It was not a bug in the metadata.  The description on disk held
``"kgrid": "4,4,1"`` — a JSON **string**, sitting beside
``"relax_steps": 100``, a JSON **number**, in the same ``overrides`` map.

``setCell`` asked the catalogue's declared type for exactly one case,
``bool``, and stored the raw text for anything that did not parse as a single
number.  ``kgrid`` is declared ``int3`` and ``Number("4,4,1")`` is ``NaN``,
so the k-grid spelling ``--kgrid`` itself accepts became text.  Four columns
had it: ``kgrid``, ``kgrid_displacement``, ``species_order``, ``ecp_atoms``.

**Why this test is an e2e and not a unit.**  Four gates could each have
stopped it and none did.  Three are pinned where they live —
``test_task_preflight.py`` (§ 6.6's declared-type row),
``test_web.py`` (the form's coercion) and ``test_stage_resolution.py`` (the
⊕ shaping).  The fourth is a keystroke, and no unit test can see one: the
value that reached the file was produced by typing into a widget, and in
seven months nothing had ever typed into it.
"""
from __future__ import annotations

import json

import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from support.road import (described, description_with_overrides, read_request,
                          road_tree, save_both)
from support.sidebar import committed
from support.task_setup_page import edited_by_hand, file_text, task_setup_on


def _described_h2(root, name, *, calculation="optimization"):
    """An H2 SIESTA calculation, ``name``, in the project ``root`` of the
    projects tree: `molbuilder describe` (`support.road`), its one stage --
    `coarse` for an optimization, `freq` for a vibration.  The tree is the
    module's, so describing moves nothing for the test."""
    from molbuilder.kind_table import kind
    with pytest.MonkeyPatch.context() as local:
        return described(road_tree(root.parents[1], local),
                         f"{root.name}/{kind(calculation).topic}/{name}",
                         name=name, calculation=calculation,
                         stage_strategy="")


def _saved(d, task):
    """``task`` through Task setup's Save, the job request as it lies."""
    r = save_both(d, task, read_request(d))
    assert r.status_code == 200, r.get_json()


def _varying(d, column):
    """``d``'s description with ``column`` its one varying parameter and no
    stage overriding it -- a column every cell of which is blank -- through
    Task setup's Save (`stages.md` § 6.2)."""
    task = json.loads((d / "task.json").read_text())
    task["varies"] = [column]
    _saved(d, task)
    return d


@pytest.fixture(scope="module")
def calc_dir(isolated_projects_root_module):
    """A calculation the tab can open, as `molbuilder describe` leaves one,
    whose one column is the one that broke: `kgrid`, through Save.

    Built under the configured projects root, because the picker refuses a
    path outside it — that is the roots guard working, not a test problem.
    """
    return _varying(_described_h2(isolated_projects_root_module
                                  / "cell_types", "probe"), "kgrid")


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


#: The one stage's k-grid cell, the column that broke.
_KGRID = 'input[aria-label="coarse k-point grid"]'


def _overrides(page):
    """The one stage's overrides in the description as the page would write
    it -- read from the editor the tab shows (`support.task_setup_page.
    file_text`), once the cell's write has landed: a cell is committed as a
    person commits it, typed, then left (`support.sidebar.committed`)."""
    return json.loads(file_text(page, "task"))["stages"][0]["overrides"]


@pytest.mark.parametrize("typed", ["4,4,1", "4x4x1", "4 4 1"])
def test_a_typed_k_grid_lands_as_three_numbers(page, flask_server, calc_dir,
                                               typed):
    """All three spellings, because a person writes a k-grid all three ways
    and the field means the same thing each time."""
    task_setup_on(page, flask_server, calc_dir)
    committed(page, _KGRID, typed)
    got = _overrides(page)["kgrid"]
    assert got == [4, 4, 1], f"typed {typed!r}, description carries {got!r}"


def test_an_emptied_cell_still_removes_the_override(page, flask_server,
                                                    calc_dir):
    """Absent means *this stage uses the template's value* (`stages.md`
    § 6.2)."""
    task_setup_on(page, flask_server, calc_dir)
    committed(page, _KGRID, "4,4,1")
    assert "kgrid" in _overrides(page)
    committed(page, _KGRID, "")
    assert "kgrid" not in _overrides(page)


def test_text_that_is_not_a_k_grid_is_kept_as_typed(page, flask_server,
                                                    calc_dir):
    """A half-parsed value would be the quiet version of the same bug, so a
    cell that cannot be read keeps what was typed — and the save door refuses
    it BY NAME (§ 6.6's declared-type row, pinned in
    ``test_task_preflight.py``).  Two counts is not a k-grid.  The setting
    door says so as it is typed, in the item's words (§ 5.2).

    MUTATION THIS MUST FAIL AGAINST: the door's refusal not said."""
    task_setup_on(page, flask_server, calc_dir)
    committed(page, _KGRID, "4,4")
    assert _overrides(page)["kgrid"] == "4,4"
    said = page.evaluate(
        "() => document.getElementById('ts-state').getAttribute('data-state')"
        " + ' | ' + document.getElementById('ts-state-body').textContent")
    assert said.startswith("refuse | kgrid = '4,4'"), said


# --------------------------------------------------------------------- #
#  The other half of the same rule: the WIDGET, not just the reader      #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bool_column_dir(isolated_projects_root_module):
    """The same calculation, but the column that varies is a **bool**.

    Separate from `calc_dir` rather than added to it: that fixture is
    module-scoped and shared by the three tests above, and widening its
    `varies` would change what they open.
    """
    return _varying(_described_h2(isolated_projects_root_module
                                  / "cell_bool", "probe"), "write_coor_xmol")


def test_a_bool_column_is_a_chooser_not_a_box(page, flask_server,
                                              bool_column_dir):
    """**A value's look must never pick the widget** *(user, 2026-08-20)*.

    The tests above are the READING half of that rule — text typed into a
    cell is parsed by the column's declared type.  This is the OFFERING half:
    `write_coor_xmol` is declared `bool`, so the catalogue already knows its
    only two answers and the cell must present them rather than invite a
    person to spell one.
    """
    task_setup_on(page, flask_server, bool_column_dir)
    cell = page.locator('[aria-label="coarse Write XMOL .xyz"]').evaluate(
        "n => ({tag: n.tagName,"
        "       options: [...(n.options || [])].map(o => o.value)})")
    assert cell["tag"] == "SELECT", (
        f"`write_coor_xmol` is declared bool, and its cell came up as a "
        f"<{cell['tag']}>.  The catalogue knows the only two "
        f"answers, so the cell must offer them: a text box invites 'True', "
        f"'yes', '1' and each is a different kind of wrong, discovered at "
        f"run time.")
    # ...and it must offer BOTH of them.  A <select> proves the widget rule
    # ran; only the options prove it got an answer, and an empty chooser is
    # a worse box than a box -- there is nothing a person can even pick.
    assert {"true", "false"} <= {str(v).lower() for v in cell["options"]}, (
        f"the chooser offers {cell['options']} -- a bool's two answers are "
        f"not both there, so `legalValues()` returned something other than "
        f"the pair, or the option list was built from the wrong source.")


@pytest.fixture
def vibration_dir(isolated_projects_root_module):
    """A vibration, as `molbuilder describe` leaves one: its one stage
    `freq`, overriding nothing."""
    return _described_h2(isolated_projects_root_module / "cell_offered",
                         "probe", calculation="vibration")


def test_a_stored_value_the_kind_does_not_offer_is_shown_as_itself(
        page, flask_server, vibration_dir):
    """The cell's menu offers what a vibration's relaxation may take --
    CG, Broyden, FIRE -- and still shows the override the description
    holds, marked, rather than a blank cell over it; prep refuses it by
    name (`engines/template.md` § 6.3a).  `task-setup.md` § 5.2: *"A stored
    value this kind does not offer is shown as itself, marked —
    `none (not offered here)` — beside the offered choices, never a blank
    cell over an override the description still holds"*.

    THE VALUE ARRIVES AS IT CAN: by a person's hand edit of the
    description.  The description's own check refuses a stage override
    outside the kind's set when the description is written (gate ③,
    `engines/template.md` § 6.3a), so Save never writes one; the page's
    editor takes a hand edit, and reads it back into the stage table
    (`task-setup.md` § 9a) -- the cell is where the person then picks an
    offered one.

    MUTATION THIS MUST FAIL AGAINST: the viewer without the branch (the
    cell shows nothing selected)."""
    task_setup_on(page, flask_server, vibration_dir)
    edited_by_hand(page, "task", json.dumps(description_with_overrides(
        vibration_dir, {"freq": {"relax_type": "none"}}), indent=2))
    # THE CELL THE HAND EDIT DRAWS, once the page has read it back.
    cell = page.locator('[aria-label="freq Relaxation algorithm"]').evaluate(
        "n => ({value: n.value, text: n.selectedIndex < 0 ? null"
        "       : n.options[n.selectedIndex].text,"
        "       options: [...n.options].map(o => o.value)})")
    assert cell["value"] == "none", cell
    assert cell["text"] == "none (not offered here)", cell
    assert {"CG", "Broyden", "FIRE"} <= set(cell["options"]), cell

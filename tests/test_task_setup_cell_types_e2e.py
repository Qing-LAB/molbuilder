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
import shutil

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def calc_dir():
    """A calculation the tab can open: a template, a structure, and a
    description whose one column is the one that broke.

    Built under the configured projects root, because the picker refuses a
    path outside it — that is the roots guard working, not a test problem.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.task import (Stage, StructureRef, Task, derive_run,
                                 write_task)
    from molbuilder.template import template_with_values

    d = ROOT / "projects/_t_cell_types/optimization/probe"
    if d.exists():
        shutil.rmtree(d.parents[1])
    d.mkdir(parents=True)
    try:
        (d / "probe.source.xyz").write_text(
            "2\nprobe\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
        (d / "probe.template.toml").write_text(
            template_with_values(SiestaConfig(system_label="probe"),
                                 engine="siesta",
                                 calculation="optimization"),
            encoding="utf-8")
        write_task(d / "task.json", Task(
            engine="siesta", shape="flat", run=derive_run("probe"),
            structure=StructureRef(source="probe.source.xyz"),
            varies=("kgrid",),
            stages=(Stage(name="tight", overrides={}),)))
        yield d
    finally:
        shutil.rmtree(d.parents[1], ignore_errors=True)


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, base, calc):
    """Land the tab on the folder the way the app itself does.

    `projects.md` § 5's hand-off writes the TARGET page's own selection slot
    and the sidebar restores from it at init.  Calling ``navigateTo`` after
    load instead is a race the sidebar wins: ``restoreSelection`` runs
    second and puts the tree back at the projects root, which is how this
    test first "opened a folder with no description in it".
    """
    slot = json.dumps(str(calc))
    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir.task-setup', {slot});"
        f" sessionStorage.setItem('molbuilder.current_dir', {slot});"
        "} catch (_) {}")
    page.goto(f"{base}/task-setup")
    page.wait_for_selector('input[aria-label="tight kgrid"]', timeout=20000)
    # The editor arrives LAST -- CodeMirror is fetched lazily, so the stage
    # table can be up while `.CodeMirror` is still null.  Everything below
    # reads the description through the editor, so this is part of "open".
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return !!(n && n.CodeMirror); }", timeout=20000)


def _described(page):
    """The description as the page would write it — read from the editor the
    tab shows, not from any internal the test would have to know about."""
    return json.loads(page.evaluate(
        "() => document.querySelector('.CodeMirror').CodeMirror.getValue()"))


def _type(page, cell, text):
    """Commit a value into one cell, and WAIT FOR THE MODEL TO TAKE IT.

    A fixed pause here was flaky in both directions -- the assertion read
    the editor either before ``syncFromModel`` had repainted it or after,
    depending on the machine.  The repaint IS the signal: the editor's
    bytes changing means the keystroke reached the description.
    """
    sel = f'input[aria-label="{cell}"]'
    before = page.evaluate(
        "() => document.querySelector('.CodeMirror').CodeMirror.getValue()")
    page.fill(sel, text)
    page.dispatch_event(sel, "change")
    page.wait_for_function(
        "b => document.querySelector('.CodeMirror').CodeMirror.getValue() !== b",
        arg=before, timeout=5000)


@pytest.mark.parametrize("typed", ["4,4,1", "4x4x1", "4 4 1"])
def test_a_typed_k_grid_lands_as_three_numbers(page, flask_server, calc_dir,
                                               typed):
    """All three spellings, because all three are what ``--kgrid`` takes
    (`cli.KGridParam`) — what works in the terminal works in the table."""
    _open(page, flask_server, calc_dir)
    _type(page, "tight kgrid", typed)
    got = _described(page)["stages"][0]["overrides"]["kgrid"]
    assert got == [4, 4, 1], f"typed {typed!r}, description carries {got!r}"


def test_the_k_grid_is_not_stored_as_text(page, flask_server, calc_dir):
    """The defect in its own words.  A ``[4, 4, 1]`` assertion alone would
    still pass if the value were the STRING ``"[4, 4, 1]"``, and the whole
    bug was a string that looked right."""
    _open(page, flask_server, calc_dir)
    _type(page, "tight kgrid", "4,4,1")
    got = _described(page)["stages"][0]["overrides"]["kgrid"]
    assert not isinstance(got, str), (
        f"the k-grid is stored as text ({got!r}) -- the description now says "
        f"a string where the config declares Tuple[int, int, int]")


def test_an_emptied_cell_still_removes_the_override(page, flask_server,
                                                    calc_dir):
    """Absent means *this stage uses the template's value* (`stages.md`
    § 6.2), and the reader rewrite must not have cost that branch."""
    _open(page, flask_server, calc_dir)
    _type(page, "tight kgrid", "4,4,1")
    assert "kgrid" in _described(page)["stages"][0]["overrides"]
    _type(page, "tight kgrid", "")
    assert "kgrid" not in _described(page)["stages"][0]["overrides"]


def test_text_that_is_not_a_k_grid_is_kept_as_typed(page, flask_server,
                                                    calc_dir):
    """A half-parsed value would be the quiet version of the same bug, so a
    cell that cannot be read keeps what was typed — and the save door refuses
    it BY NAME (§ 6.6's declared-type row, pinned in
    ``test_task_preflight.py``).  Two counts is not a k-grid."""
    _open(page, flask_server, calc_dir)
    _type(page, "tight kgrid", "4,4")
    assert _described(page)["stages"][0]["overrides"]["kgrid"] == "4,4"

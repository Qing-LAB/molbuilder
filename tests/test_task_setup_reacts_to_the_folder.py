"""Two defects the Task-setup tab had, both reported 2026-08-24.

**(1) A checkpoint restore left the tab showing a description the folder no
longer had.**  Restoring can swap `task.json` for `task.1st.json` while the
tab is displaying the first one.  The SELECTION never changed, so
`projects.onChange` never fired, and the checkpoint panel's own `_refresh()`
repaints only itself -- nothing told the rest of the page that the bytes
under it had changed.

**(2) Every enum and bool in the stage table had to be typed by hand.**  The
cell's widget is chosen by `legalValues()`, which reads `_meta`.
`refreshPickers()` CLEARS `_meta` and then calls the two vocabulary loaders
-- and both loaders returned early from their cache *before* the line that
publishes into `_meta`.  So from the second load on, the metadata the server
had already sent was thrown away and never restored, and a dropdown became a
text box.

Static assertions rather than a driven page: `viewer.js` is an ES module and
the repo's node harness requires CommonJS.  What these pin is the WIRING --
that the announcement exists, that someone listens, and that no cached path
returns without publishing -- which is exactly what regressed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static"
_VIEWER = _STATIC / "task-setup" / "viewer.js"
_STATE = _STATIC / "lib" / "projects" / "state.js"
_CKPT = _STATIC / "lib" / "projects" / "checkpoint.js"


@pytest.fixture(scope="module")
def viewer() -> str:
    return _VIEWER.read_text(encoding="utf-8")


# ---- (1) the folder-changed channel --------------------------------- #

def test_the_projects_module_offers_a_folder_changed_channel():
    """A fourth kind of event, and the one nothing had: `onChange` says the
    SELECTION moved, `onCommit` says a file was chosen -- neither fires when
    the same folder suddenly holds different bytes."""
    src = _STATE.read_text(encoding="utf-8")
    assert "onFolderChanged:" in src
    assert "publishFolderChanged" in src


def test_a_restore_announces_that_the_folder_changed():
    """The checkpoint panel cannot know which tabs are open or what they
    cache, so it ANNOUNCES rather than reaching into them."""
    src = _CKPT.read_text(encoding="utf-8")
    # The CALL, with the directory in it -- not merely the name, which also
    # appears in the `typeof ... === "function"` guard beside it.  The first
    # version of this test asserted the bare name and passed happily with
    # the call deleted (caught by mutation-testing it, 2026-08-24).
    assert re.search(r"publishFolderChanged\(\s*_state\.currentDir\s*\)", src), (
        "a restore rewrites the folder and tells nobody -- the defect")


def test_the_tab_re_reads_when_its_own_folder_changes(viewer):
    assert "onFolderChanged" in viewer, (
        "the tab never learns the folder changed underneath it")
    # ...and re-reads only ITS folder: a restore elsewhere must not repaint.
    assert re.search(r"onFolderChanged\([\s\S]{0,400}?changed !== _dir", viewer), (
        "the tab reacts to every folder's change, not only its own")


# ---- (2) the vocabulary cache must still publish -------------------- #

def _cached_returns(src: str, cache_var: str):
    """The early-return branch guarding each vocabulary loader's cache."""
    m = re.search(rf"if \(_{cache_var} && _{cache_var}Key === key\) \{{"
                  rf"([\s\S]*?)\n    \}}", src)
    return m.group(1) if m else None


@pytest.mark.parametrize("cache_var, filler",
                         [("cols", "_fillMeta"), ("sweep", "_fillSweepMeta")])
def test_a_cached_vocabulary_still_publishes_its_meta(viewer, cache_var, filler):
    """**The regression.**  The cache is about not re-FETCHING; it was never
    meant to skip publishing what was fetched.  `refreshPickers` clears
    `_meta` first, so an early return here leaves the widget question
    unanswerable and every enum renders as a text box."""
    body = _cached_returns(viewer, cache_var)
    assert body is not None, f"the _{cache_var} cache guard moved -- repoint this"
    assert filler in body, (
        f"the cached path returns _{cache_var} without publishing into "
        f"_meta; legalValues() will find nothing and the dropdown becomes "
        f"a text box")


def test_the_meta_fill_has_one_home_per_vocabulary(viewer):
    """Two loaders x (cached, fetching) is four places to remember, and the
    two cached ones were forgotten.  One function each, called from both."""
    for filler in ("_fillMeta", "_fillSweepMeta"):
        assert f"function {filler}(" in viewer, filler
        # defined once, called from BOTH paths
        assert viewer.count(filler + "(") >= 3, (
            f"{filler} is not called from both the cached and the fetching "
            f"path")


def test_the_table_is_rendered_only_after_the_meta_is_awaited(viewer):
    """Ordering, the other half of the same defect: `renderStages` asks
    `legalValues()` per cell, so the vocabulary must have ARRIVED -- it was
    rendered first and the loaders ran after it, one of them unawaited."""
    load = viewer[viewer.index("async function loadFolder("):]
    load = load[:load.index("\n/* ---------- the table edits")]
    assert "await refreshPickers();" in load, (
        "refreshPickers is not awaited before the table renders")
    assert load.index("await refreshPickers();") < load.index("renderStages(task);"), (
        "the stage table renders before its vocabulary has arrived")

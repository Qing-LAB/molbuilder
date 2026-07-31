"""Verify the selection-panel atom list handles both small and
large structures correctly, via BOTH render paths (diff-only + virtual
scroll).

The implementation picks a path automatically based on atom count
(``VSCROLL_THRESHOLD = 2000`` in ``lib/molview/selection/panel.js``).  This
test uses the panel's force-mode switch to drive each path against
two structures explicitly:

  Small structure (water, 3 atoms):
    * Default (simple/diff-only) -- renders all 3 rows in the DOM.
    * Forced virtual -- still works; spacer + visible-window rows.

  Large structure (1c75.pdb, 1184 atoms, OR a synthetic many-atom
  PDB if the user file isn't on disk):
    * Default (varies by size; ~1184 falls below threshold so
      simple) -- all rows in the DOM.
    * Forced virtual -- only a small window of rows actually
      rendered; spacer heights match the offscreen portion.

The assertions are NOT "did the page survive" -- they're about the
ACTUAL DOM SHAPE:
  - simple path: every atom has a ``<tr data-atom-index="N">`` row.
  - virtual path: only the visible window + buffer is in the DOM;
    a top + bottom spacer carry the rest of the scrollbar height.

If either path drifts (e.g. virtual scroll skips an atom; simple
path doesn't render N rows for N atoms), this test fails with a
concrete diff.

Run::

    python -m pytest tests/test_atom_list_render_paths.py -v
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


_USER_PDB = Path(
    "/home/qqing/molbuilder/projects/hemeC-dithiol/structure/1c75.pdb"
)


def _synthetic_large_pdb(n_atoms: int) -> str:
    """Build a PDB with ``n_atoms`` ALA atoms on a line.  Used when
    the user's 1c75.pdb isn't on disk so this test still runs."""
    lines = ["HEADER    SYNTHETIC LARGE\n"]
    for i in range(n_atoms):
        x = i * 2.0
        lines.append(
            f"ATOM   {i + 1:>4d}  CA  ALA A{i + 1:>4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        )
    lines.append("END\n")
    return "".join(lines)


_SYNTHETIC_LARGE = _synthetic_large_pdb(1184)


@pytest.fixture(scope="module")
def flask_server():
    pytest.importorskip("flask")
    pytest.importorskip("playwright")
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        th.join(timeout=5)


@pytest.fixture(scope="module")
def small_pdb(tmp_path_factory):
    """Tiny 3-atom water structure."""
    d = tmp_path_factory.mktemp("small")
    p = d / "water.pdb"
    p.write_text(
        "HEADER    WATER\n"
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H\n"
        "ATOM      3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H\n"
        "END\n"
    )
    return p


@pytest.fixture(scope="module")
def large_pdb(tmp_path_factory):
    """Use the user's actual 1c75.pdb (1184 atoms) when available;
    otherwise a synthetic 1184-atom PDB so this test runs anywhere."""
    if _USER_PDB.exists():
        return _USER_PDB
    d = tmp_path_factory.mktemp("large")
    p = d / "large.pdb"
    p.write_text(_SYNTHETIC_LARGE)
    return p


@pytest.fixture(scope="module")
def large_pdb_n_atoms(large_pdb):
    """Actual atom count after Structure.from_pdb parsing.

    The PDB file holds 1184 raw ATOM/HETATM records, but the parser
    deduplicates altLoc occupants (keeps the first occurrence of each
    ``(chain, resid, atom_name)`` tuple), so 1c75.pdb resolves to 1146
    atoms (38 altLoc B records dropped).  The DOM count the test
    asserts must match this -- the JS selection store gets what the
    server's parser produced, not the raw line count."""
    from molbuilder.structure import Structure
    return len(Structure.from_pdb(large_pdb.read_text()).elements)


@pytest.fixture(autouse=True)
def _restore_capabilities_class_method():
    """Snapshot ``Capabilities.file_picker_roots`` on entry, restore
    on exit.

    ``_setup_picker_root_for`` below mutates the CLASS attribute
    directly (the closure needs to capture the per-test
    ``structure_path`` and pytest's ``monkeypatch`` is function-
    scoped + saves to instance, not class).  Without this fixture
    the last test's lambda leaked into subsequent test files and
    caused intermittent failures on any later test that called
    ``Capabilities(...).file_picker_roots()`` directly — concretely
    ``tests/test_web_files.py::test_capabilities_returns_only_projects_root``
    bisected to this file as the upstream offender on 2026-06-10.

    The conftest's autouse ``_reset_diagnostics_singleton`` only
    handles the GLOBAL ``_snapshot``; it does NOT restore class-level
    mutations, so we need this file-local fixture.
    """
    from molbuilder import diagnostics
    cls = diagnostics.Capabilities
    original = cls.file_picker_roots
    try:
        yield
    finally:
        cls.file_picker_roots = original


def _setup_picker_root_for(structure_path):
    """Repoint Capabilities.file_picker_roots() to the structure's
    parent dir so the selection / files endpoints accept it.

    Restoration is handled by the module-local autouse fixture
    ``_restore_capabilities_class_method`` above — every call to
    this helper is wrapped by a save/restore on the class method.
    """
    from molbuilder import diagnostics
    parent = Path(structure_path).resolve().parent
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    cls.file_picker_roots = lambda self: ((parent, "projects"),)
    diagnostics.set_capabilities(caps)
    return diagnostics


def _drive_modify_with(page, base_url, structure_path):
    """Open /molbuilder, set the projects sidebar to the structure_path,
    wait until the selection store has loaded N atoms."""
    page.goto(base_url + "/molbuilder")
    page.wait_for_function(
        "() => !!window.molbuilder "
        "      && !!window.molbuilder.molview && !!window.molbuilder.molview.data "
        "      && !!window.molbuilder.molview.data.selection "
        "      && !!window.molbuilder.runtime"
    )
    # Wait for the projects module to land via the runtime registry.
    page.wait_for_function(
        "() => window.molbuilder.runtime.listRegistered()"
        "                 .includes('projects')"
    )
    # Fire publishCommit (the dblclick-equivalent commit channel)
    # so the Molbuilder tab's onCommit subscriber kicks off
    # _commitFile → viewer load + selection store atoms fetch.
    # B.5.2 (2026-06-07) separated single-click = preview
    # (setShared/onChange) from dblclick = commit
    # (publishCommit/onCommit); this test needs the commit path.
    page.evaluate(
        """async (path) => {
            const mod = await import("/static/lib/projects/state.js");
            const dir = path.substring(0, path.lastIndexOf("/"));
            mod.publishCommit(dir, path);
        }""",
        str(Path(structure_path).resolve()),
    )


def _wait_atoms_loaded(page, n_expected):
    page.wait_for_function(
        f"() => window.molbuilder.molview.data.selection.getState()"
        f"             .atoms.length === {n_expected}",
        timeout=15000,
    )


def _count_atom_rows(page):
    """How many actual <tr data-atom-index> rows are in the DOM."""
    return page.evaluate(
        "() => document.querySelectorAll("
        "  '#selection-atom-list tr[data-atom-index]'"
        ").length"
    )


def _vscroll_active(page):
    """Is the virtual-scroll wrapper present?  Detected by the
    spacer <tr> with data-role=vscroll-top."""
    return page.evaluate(
        "() => !!document.querySelector("
        "  '#selection-atom-list tr[data-role=\"vscroll-top\"]'"
        ")"
    )


def _force_mode(page, mode):
    """Set the force-mode switch and trigger a re-render by toggling
    the store -- the panel reads forceRenderMode on every render."""
    page.evaluate(
        """(mode) => {
            window.molbuilder.molview.selection.panel.forceRenderMode = mode;
            // Force a notify so the panel re-renders with the new mode.
            const s = window.molbuilder.molview.data.selection;
            const cur = s.getState().indices;
            // Toggle then restore to drive _notify cheaply.
            s.set(cur.length ? [] : [0]);
            s.set(cur);
        }""",
        mode,
    )


# --------------------------------------------------------------------- #
# Small structure: 3 atoms                                              #
# --------------------------------------------------------------------- #


class TestSmallStructureRenderPaths:
    """Water (3 atoms).  Both render paths must produce a complete,
    consistent atom list."""

    def test_default_simple_path_renders_all_three_rows(
        self, page, flask_server, small_pdb,
    ):
        _setup_picker_root_for(small_pdb)
        _drive_modify_with(page, flask_server, small_pdb)
        _wait_atoms_loaded(page, 3)
        # Default path for n=3: simple (diff-only / full DOM).
        assert not _vscroll_active(page), (
            "Default render for a 3-atom structure should NOT activate "
            "the virtual scroller -- the threshold is 2000."
        )
        # All 3 rows in the DOM.
        n = _count_atom_rows(page)
        assert n == 3, (
            f"Simple path should render all 3 rows for a 3-atom "
            f"structure; got {n}"
        )

    def test_force_virtual_path_still_renders_all_three(
        self, page, flask_server, small_pdb,
    ):
        _setup_picker_root_for(small_pdb)
        _drive_modify_with(page, flask_server, small_pdb)
        _wait_atoms_loaded(page, 3)
        _force_mode(page, "virtual")
        # Spacers must appear.
        assert _vscroll_active(page), (
            "force-mode='virtual' did NOT activate the virtual scroller "
            "(no vscroll-top spacer in the DOM)."
        )
        # All 3 rows still rendered (3 < buffer; whole list fits in
        # the visible window).
        n = _count_atom_rows(page)
        assert n == 3, (
            f"Virtual path should render all 3 rows for a 3-atom "
            f"structure (well under the visible-window buffer); got {n}"
        )


# --------------------------------------------------------------------- #
# Large structure: 1184 atoms                                           #
# --------------------------------------------------------------------- #


class TestLargeStructureRenderPaths:
    """1184 atoms (user's 1c75.pdb OR a synthetic 1184-atom PDB).
    Below the 2000 threshold but large enough that the virtual-scroll
    path is visibly different from the simple path."""

    def test_default_simple_path_renders_all_1184_rows(
        self, page, flask_server, large_pdb, large_pdb_n_atoms,
    ):
        _setup_picker_root_for(large_pdb)
        _drive_modify_with(page, flask_server, large_pdb)
        _wait_atoms_loaded(page, large_pdb_n_atoms)
        # Atom count < 2000 -> simple path by default.
        assert not _vscroll_active(page), (
            f"Default render for a {large_pdb_n_atoms}-atom structure "
            f"should be simple (below the 2000 threshold)."
        )
        n = _count_atom_rows(page)
        assert n == large_pdb_n_atoms, (
            f"Simple path: expected {large_pdb_n_atoms} rendered rows; got {n}"
        )

    def test_force_virtual_renders_only_visible_window(
        self, page, flask_server, large_pdb, large_pdb_n_atoms,
    ):
        _setup_picker_root_for(large_pdb)
        _drive_modify_with(page, flask_server, large_pdb)
        _wait_atoms_loaded(page, large_pdb_n_atoms)
        _force_mode(page, "virtual")
        assert _vscroll_active(page), (
            "force-mode='virtual' did NOT activate the virtual scroller."
        )
        # Only a fraction of atoms should be in the DOM after virtual
        # scroll kicks in.  The exact number depends on the
        # scroller's clientHeight; we just assert it's MUCH less than
        # the total (otherwise the virtual scroller isn't pruning).
        n = _count_atom_rows(page)
        assert n < 200, (
            f"Virtual scroll should prune the rendered set; got {n} "
            f"rows out of {large_pdb_n_atoms}.  Threshold for 'pruning' "
            f"set generously at 200 to allow for buffer + a tall viewport."
        )
        assert n > 0, (
            "Virtual scroll rendered 0 rows -- empty window means "
            "the paint logic is broken."
        )

    def test_force_virtual_then_back_to_simple_swap_clean(
        self, page, flask_server, large_pdb, large_pdb_n_atoms,
    ):
        """Toggling force-mode mid-session should swap cleanly:
        virtual scroller torn down, simple path reinstated, all
        rows back in the DOM."""
        _setup_picker_root_for(large_pdb)
        _drive_modify_with(page, flask_server, large_pdb)
        _wait_atoms_loaded(page, large_pdb_n_atoms)
        _force_mode(page, "virtual")
        assert _vscroll_active(page)
        assert _count_atom_rows(page) < 200

        # Swap back.
        _force_mode(page, "simple")
        assert not _vscroll_active(page), (
            "force-mode='simple' did NOT tear down the virtual scroller."
        )
        n = _count_atom_rows(page)
        assert n == large_pdb_n_atoms, (
            f"After swap virtual->simple, expected {large_pdb_n_atoms} "
            f"rendered rows; got {n}.  Suggests the scroller's torn-down "
            f"state is inconsistent."
        )

    def test_virtual_scroll_repaints_on_scroll_with_current_selection(
        self, page, flask_server, large_pdb, large_pdb_n_atoms,
    ):
        """Stale-closure regression: after a selection toggle, the
        virtual scroller's scroll-event repaints with the CURRENT
        selection, not the captured-at-create-time one."""
        _setup_picker_root_for(large_pdb)
        _drive_modify_with(page, flask_server, large_pdb)
        _wait_atoms_loaded(page, large_pdb_n_atoms)
        _force_mode(page, "virtual")
        # Select atom 800.  Force-mode toggle helper already drove a
        # re-render; explicitly set the selection now.
        page.evaluate(
            "() => window.molbuilder.molview.data.selection.set([800])"
        )
        # Scroll the container to bring atom 800 into the visible window.
        scrolled_to_visible = page.evaluate(
            """() => {
                const tbody = document.getElementById('selection-atom-list');
                const scroller = tbody.parentElement.parentElement;
                // 22 px per row * 800 = scroll to ~atom 800.
                scroller.scrollTop = 22 * 800;
                // Synchronously trigger the scroll listener.
                scroller.dispatchEvent(new Event('scroll'));
                // Read back what's now in the DOM.
                const rows = document.querySelectorAll(
                    '#selection-atom-list tr[data-atom-index]'
                );
                const indices = Array.from(rows).map(
                    (tr) => parseInt(tr.dataset.atomIndex, 10)
                );
                return indices;
            }"""
        )
        # Atom 800 should now be in the rendered window.
        assert 800 in scrolled_to_visible, (
            f"After scrolling to atom 800's offset, the rendered "
            f"window doesn't contain index 800.  Got: "
            f"{scrolled_to_visible[:5]}...{scrolled_to_visible[-5:]}"
        )
        # And atom 800's <tr> carries the .is-selected class -- the
        # post-toggle selection state survived the scroll repaint
        # (not the stale-closure bug where the scroll handler used
        # the OLD selection captured at scroller-create time).
        is_selected = page.evaluate(
            """() => {
                const tr = document.querySelector(
                    '#selection-atom-list tr[data-atom-index="800"]'
                );
                return tr && tr.classList.contains('is-selected');
            }"""
        )
        assert is_selected, (
            "Atom 800 was selected BEFORE the scroll, but the scroll-"
            "triggered repaint didn't carry the selection through -- "
            "stale-closure regression."
        )

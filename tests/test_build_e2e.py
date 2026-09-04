"""End-to-end Playwright tests for /structure-optimization — the describing tab.

Audit task #190 (2026-06-02) opened this file when the tab had no
browser coverage at all; the tab has been rebuilt twice since, and this
docstring describes what the tests cover NOW, not the world of #190:

  * The two parameter forms (SIESTA + PySCF) populate from the CATALOGUE
    schema door and carry engine_key badges — a silent renderer break
    leaves the "Loading from schema..." placeholder (the failure class
    the 2026-08-22 span-cut regression demonstrated: the file parsed,
    every text pin was green, and only this lane knew the form was dead).
  * Tab switching between the two engine panels.
  * The sidebar contract: setShared alone must NOT load (candidate-only);
    publishCommit is the load; a second commit replaces the first
    (the #51 `enforce` rule); the form-dirty gate warns before a commit
    discards typed edits.
  * Second-visit persistence: the tab keeps its own structure across
    navigation, and only an explicit Load re-reads changed disk bytes.
  * Live preflight findings render beside their own control and clear
    when the value is fixed.
  * Send to Task setup — the tab's PRIMARY loop since script generation
    left it (#295, 2026-08-15): the button writes the structure pair,
    the parameter template and the hand-over into the selected folder
    through the one shared door (lib/task-handover.js), and the
    one-job-per-folder guard fails CLOSED.

NOT covered here (intentional scope split): the HTTP layer is
test_web.py's; Task setup's own save door is test_task_setup_tab.py's.
There is no Generate button and no Build form on this tab any more —
decks are rendered by `prep` on the machine that runs them.
"""
from __future__ import annotations


import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Pin ``tmp_path`` as the only Capabilities picker root so the
    files blueprint accepts files written under it.  Mirrors the
    helper in ``test_molbuilder_e2e.py`` / ``test_source_inspector_e2e.py``.
    """
    from molbuilder import diagnostics
    _orig = diagnostics.get_capabilities()
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None,
        conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    monkeypatch.setattr(diagnostics, "_snapshot", _orig)


@pytest.fixture
def water_xyz_file(tmp_path, monkeypatch):
    """Minimal real XYZ on disk for the sidebar-driven load test."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    p = tmp_path / "water.xyz"
    p.write_text(
        "3\nwater\n"
        "O 0.000  0.000 0.000\n"
        "H 0.957  0.000 0.000\n"
        "H -0.239 0.927 0.000\n"
    )
    return str(p)


@pytest.fixture
def two_xyz_files(tmp_path, monkeypatch):
    """TWO structures with DIFFERENT atom counts, in one picker root.

    Different counts on purpose: the atom-count line is what tells the two
    apart, so a viewer that ignored the second file would show 3 where the test
    demands 5.  Same-sized structures would let that pass.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    first = tmp_path / "water.xyz"
    first.write_text(
        "3\nwater\n"
        "O 0.000  0.000 0.000\n"
        "H 0.957  0.000 0.000\n"
        "H -0.239 0.927 0.000\n"
    )
    second = tmp_path / "methane.xyz"
    second.write_text(
        "5\nmethane\n"
        "C  0.000  0.000  0.000\n"
        "H  0.629  0.629  0.629\n"
        "H -0.629 -0.629  0.629\n"
        "H -0.629  0.629 -0.629\n"
        "H  0.629 -0.629 -0.629\n"
    )
    return str(first), str(second)


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


_BOOT_TIMEOUT_MS = 5000


def _open_build(page, base_url):
    """Navigate to / and wait for the viewer.js wiring to land.

    Capture JS errors so a regression that throws during init() (the
    same failure mode as the 2026-05 Spectra parse bug) surfaces
    here with a clear page-name attribution.
    """
    errors = []
    page.on("pageerror",
            lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    # ARRIVE AT A TAB THAT REMEMBERS NOTHING.
    #
    # This tab keeps the structure it was showing (workspace.md § 4: several
    # savers on one page, each deciding what and when), so "open the tab" is not
    # the same as "open an empty tab" once anything has been loaded -- and these
    # tests share one server and one workspace, so without this each one
    # inherits whatever the previous one left on the canvas.
    #
    # Cleared through the tab's own door rather than by deleting files: the id
    # is the workspace's to compute, and a test that hard-codes a filename is
    # pinning a layout the workspace is free to change.  `above_index = -1`
    # clears the whole tag.
    page.request.post(
        f"{base_url}/api/workspace-storage/prune",
        data={"workspace_id": "ws-structure-opt", "above_index": -1},
    )
    page.goto(f"{base_url}/structure-optimization")
    # The "Load from sidebar selection" button is server-rendered;
    # waiting for it proves the HTML reached the browser.  The form-
    # container fields are added by JS after the schema fetch lands.
    # (Was ``#build-btn`` before the 2026-06-08 Build-form retirement.)
    page.wait_for_selector("#load-from-sidebar-btn", timeout=_BOOT_TIMEOUT_MS)
    return errors


# --------------------------------------------------------------------- #
#  Form schemas populate from the server                                #
# --------------------------------------------------------------------- #


class TestFormSchemasRender:
    """Both engine forms fetch their schema and render fields.  A
    regression in dataclass_to_form_schema or in form-schema.js'
    renderer would leave the containers empty -- silent UI break."""

    def test_siesta_form_renders_fields_after_init(
            self, page, flask_server):
        errors = _open_build(page, flask_server)
        # The form renderer mounts inputs/selects into
        # #siesta-form-container after the schema fetch resolves.
        page.wait_for_selector(
            "#siesta-form-container input, "
            "#siesta-form-container select",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # The Loading-from-schema placeholder must be gone.
        container_text = page.locator(
            "#siesta-form-container"
        ).inner_text()
        assert "Loading from schema" not in container_text
        assert errors == [], f"JS errors during /build init: {errors}"

    @pytest.mark.parametrize("engine", ["siesta", "pyscf"])
    def test_no_field_renders_loose_outside_a_card(
            self, page, flask_server, engine):
        """Every field the catalogue asks for lands INSIDE a card.

        `form-schema.js` draws one card per name in its own
        ``WORKFLOW_GROUP_ORDER`` and renders a field whose group is not in
        that list bare, appended straight to the container — the same
        invisible outcome as no group at all.  So adding a group to the
        catalogue without adding it to the renderer looks exactly like the
        bug the grouping was introduced to fix.

        *Replaces `test_catalogue_form_schema.py::
        test_the_renderer_knows_every_card_the_form_actually_asks_for`,
        retired 2026-09-03.*  That test regex-extracted `WORKFLOW_GROUP_ORDER`
        from the renderer's source and compared it to the catalogue's groups
        as SETS — and its own docstring conceded "only a browser would show
        it" (`process/testing.md` § 3a.1).  It could not see the two ways the
        comparison passes while the page is still wrong: a role listed in
        ``WORKFLOW_GROUP_ORDER`` but missing from ``WORKFLOW_GROUP_META``
        fails the `if` that admits a field to a card, and a card whose
        section map comes out empty is skipped entirely.  Here the question
        is asked of the rendered DOM, where the answer is the same one the
        person gets.
        """
        _open_build(page, flask_server)
        sel = f"#{engine}-form-container"
        page.wait_for_selector(f"{sel} input, {sel} select",
                               state="attached", timeout=_BOOT_TIMEOUT_MS)
        loose = page.evaluate("""(sel) => {
            const root = document.querySelector(sel);
            if (!root) return {error: "no container"};
            const all = [...root.querySelectorAll("fieldset.schema-section")];
            const bare = all.filter(fs => !fs.closest("section.workflow-group"));
            return {
                total: all.length,
                bare:  bare.map(fs => (fs.querySelector("legend") || {})
                                        .textContent || "(no legend)"),
            };
        }""", sel)
        assert not loose.get("error"), loose
        assert loose["total"] > 0, (
            f"{engine}: no field groups rendered at all, so 'none of them is "
            f"loose' would be true of an empty page")
        assert loose["bare"] == [], (
            f"{engine}: section(s) {loose['bare']} rendered below the cards "
            f"instead of inside one.  Their workflow_group is not among the "
            f"roles form-schema.js draws, so the catalogue asks for a card "
            f"the renderer does not know about -- add it to "
            f"WORKFLOW_GROUP_ORDER *and* WORKFLOW_GROUP_META.")

    def test_pyscf_form_renders_fields_after_init(
            self, page, flask_server):
        _open_build(page, flask_server)
        # The PySCF tab panel starts ``hidden`` (SIESTA is the
        # default-active tab), so its rendered inputs are NOT visible
        # by default.  Use ``state="attached"`` so Playwright waits
        # for the elements to exist in the DOM rather than waiting
        # for them to become visible -- form-schema.js builds both
        # panels at init() time regardless of tab visibility, and
        # that's the property we want to pin.
        page.wait_for_selector(
            "#pyscf-form-container input, "
            "#pyscf-form-container select",
            state="attached",
            timeout=_BOOT_TIMEOUT_MS,
        )
        container_text = page.locator(
            "#pyscf-form-container"
        ).inner_text()
        assert "Loading from schema" not in container_text

    def test_engine_key_badges_present_after_schema_render(
            self, page, flask_server):
        """Per the 2026-05-26 source-of-truth contract every form
        field carries an ``engine_key`` metadata entry which the
        renderer surfaces as a ``<code class="schema-engine-key">``
        badge.  Absence here means the schema -> form binding lost
        the metadata, OR a future field was added without an
        engine_key (which would also fail the Python-side coverage
        test in ``test_web.py::test_engine_key_present_on_every_*``,
        so this test triangulates the contract from the JS side).
        """
        _open_build(page, flask_server)
        page.wait_for_selector(
            "#siesta-form-container .schema-engine-key",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # At least one badge in each container.  Existence proves
        # the renderer reached the badge branch; the text contents
        # are validated by the Python-side coverage test.
        siesta_badges = page.locator(
            "#siesta-form-container .schema-engine-key"
        ).count()
        pyscf_badges = page.locator(
            "#pyscf-form-container .schema-engine-key"
        ).count()
        assert siesta_badges > 0, \
            "SIESTA form rendered without engine_key badges"
        assert pyscf_badges > 0, \
            "PySCF form rendered without engine_key badges"


# --------------------------------------------------------------------- #
#  Tab switching                                                        #
# --------------------------------------------------------------------- #


class TestTabSwitching:
    """Clicking the SIESTA / PySCF tabs reveals the matching panel."""

    def test_clicking_pyscf_tab_shows_pyscf_panel(
            self, page, flask_server):
        _open_build(page, flask_server)
        # Wait for the SIESTA panel (default active, visible) so we
        # know the schema render landed before driving the tab click.
        page.wait_for_selector(
            "#siesta-form-container input, "
            "#siesta-form-container select",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # The PySCF tab activator: matched by role + name.  Tab
        # button text is "PySCF script"; ``has_text`` does substring
        # matching so this still works if the wording shifts slightly.
        page.locator("[role='tab']").filter(has_text="PySCF").click()
        # After the click the PySCF panel must be visible (not
        # ``hidden`` attribute).
        page.wait_for_selector(
            "#tab-pyscf:not([hidden])", timeout=_BOOT_TIMEOUT_MS,
        )

    def test_clicking_siesta_tab_returns_to_siesta_panel(
            self, page, flask_server):
        _open_build(page, flask_server)
        page.wait_for_selector(
            "#siesta-form-container input, "
            "#siesta-form-container select",
            timeout=_BOOT_TIMEOUT_MS,
        )
        page.locator("[role='tab']").filter(has_text="PySCF").click()
        page.locator("[role='tab']").filter(has_text="SIESTA").click()
        page.wait_for_selector(
            "#tab-siesta:not([hidden])", timeout=_BOOT_TIMEOUT_MS,
        )


# --------------------------------------------------------------------- #
#  Build a peptide and check the post-build viewer / button state       #
# --------------------------------------------------------------------- #


class TestSidebarStructureFlow:
    """The sidebar is the ONLY way a structure reaches this tab
    (the typed Build form retired with #295): a commit loads, a bare
    pick does not, a second commit replaces the first, and typed
    parameter edits are guarded before a commit discards them.

    (This class was ``TestPeptideBuild`` from the #190 era — no test
    here has typed a sequence or clicked Build since that form left.)
    """

    def test_sidebar_load_updates_info_atoms(
            self, page, flask_server, water_xyz_file):
        """Post-2026-06-08 (task #295): the Build/Load form is
        retired; structures come from the project sidebar.  A
        commit on a sidebar pick rebuilds the structure section
        and the #info-atoms span flips from the em-dash
        placeholder to a numeric atom count."""
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder "
            "&& window.molbuilder.projects "
            "&& typeof window.molbuilder.projects.publishCommit "
            "       === 'function'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            """(ctx) => window.molbuilder.projects.publishCommit(
                ctx.dir, ctx.file)""",
            {"dir": parent, "file": p},
        )
        page.wait_for_function(
            "() => /^\\d+$/.test("
            "document.querySelector('#info-atoms').textContent.trim())",
            timeout=_BOOT_TIMEOUT_MS,
        )
        atom_count = int(
            page.locator("#info-atoms").inner_text().strip()
        )
        assert atom_count == 3   # water.xyz

    def test_commit_mounts_molview_card(
            self, page, flask_server, water_xyz_file):
        """structure-optimization migration: committing a structure mounts the
        FULL concealed MolView component (mode:"readonly" — the same rich card
        Modify uses, read-only) into #viewer-host, and the structure lives in
        molview.data as the single source of truth (which Generate reads)."""
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => !!(window.molbuilder && window.molbuilder.projects"
            "         && document.getElementById('viewer-host'))",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        page.evaluate(
            "(ctx) => window.molbuilder.projects.publishCommit(ctx.dir, ctx.file)",
            {"dir": str(Path(p).parent), "file": p},
        )
        # The full fused card + 3Dmol canvas mount into the host.
        page.wait_for_function(
            "() => { const h = document.getElementById('viewer-host');"
            "  return !!h && !!h.querySelector('.molviewer-card')"
            "         && !!h.querySelector('canvas'); }",
            timeout=15_000,
        )
        # The viewer holds the committed structure, and the panel's own count
        # line is where that is visible.  Asked through the DOM because there is
        # nowhere else to ask: a viewer belongs to whoever mounted it and there
        # is no registry (molview.md § 5.6).
        count = page.locator(".molviewer-selection-count").inner_text()
        assert count.split(" of ")[1].split()[0] == "3"   # water

    # ``test_the_form_learns_the_structure_after_sidebar_load`` was retired
    # 2026-08-16.  Its witness was the block-size placeholder rendering
    # ``auto (<N>, n=<atoms>)`` once a structure was loaded -- and the user
    # deleted exactly that display: "nobody knows what this means and nobody
    # can predict what the auto value would be ... i'd rather just leave it as
    # 'auto'".  ``autoBlockSize()`` went with it, so the test pinned an
    # interaction the tab no longer has.
    #
    # The property it stood for -- the form REACTS to a structure arriving --
    # still deserves a witness, and it needs a control whose display depends
    # on the structure.  There is no such control today; when one exists, the
    # test comes back pointed at that instead of at a hint that was removed
    # on purpose.

    def test_setShared_alone_does_NOT_load_structure(
            self, page, flask_server, water_xyz_file):
        """Pins the candidate-only contract for Build: a single
        sidebar click sets the global pick but MUST NOT auto-
        rebuild the structure section.  Browse-clicking through
        the project tree no longer silently switches the
        molecule the form is configured for."""
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder "
            "&& window.molbuilder.projects "
            "&& typeof window.molbuilder.projects.setShared "
            "       === 'function'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            """(ctx) => window.molbuilder.projects.setShared(
                ctx.dir, ctx.file)""",
            {"dir": parent, "file": p},
        )
        # Brief settle so an erroneous auto-load would land in
        # #info-atoms before the assertion.
        page.wait_for_timeout(500)
        n = page.evaluate(
            "() => document.querySelector('#info-atoms').textContent.trim()"
        )
        assert n in ("—", "", "0"), (
            f"setShared must NOT auto-load; #info-atoms is {n!r}"
        )

    def test_publishCommit_loads_water_xyz_into_viewer(
            self, page, flask_server, water_xyz_file):
        """Commit (dblclick equivalent) rebuilds the structure
        section — the canonical "use this file in this tab"
        action."""
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder "
            "&& window.molbuilder.projects "
            "&& typeof window.molbuilder.projects.publishCommit "
            "       === 'function'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            """(ctx) => window.molbuilder.projects.publishCommit(
                ctx.dir, ctx.file)""",
            {"dir": parent, "file": p},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

    def test_a_second_file_replaces_the_first_in_one_visit(
            self, page, flask_server, two_xyz_files):
        """PICK ONE FILE, THEN ANOTHER, WITHOUT LEAVING THE PAGE.

        THE BUG THIS EXISTS FOR (#51).  This tab mounts MolView read-only, and
        `installMolecule` refuses to REPLACE what a read-only viewer already
        holds unless the caller says it means to (§ 9.4).  `openMolecule` is the
        user pressing Load, so it passes `enforce: true` -- and until it did,
        every read-only surface (this tab, spectra, transport, the results
        inspector) would load ONE file per page and then silently ignore every
        later pick.  The viewer answered null, nothing threw, and the old
        structure stayed on screen.

        NOTHING GUARDED IT.  Every commit test here commits a single file, and
        the second-visit class covers navigating away and back, or the SAME file
        changed on disk -- neither is a second pick in one visit.  Delete the
        word `enforce` from projects/parser.js today and the whole suite still
        passes.

        The two fixtures have different atom counts because that is the only
        thing on screen that can tell them apart.
        """
        first, second = two_xyz_files
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder "
            "&& window.molbuilder.projects "
            "&& typeof window.molbuilder.projects.publishCommit "
            "       === 'function'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        from pathlib import Path

        def commit(path):
            p = str(Path(path).resolve())
            page.evaluate(
                """(ctx) => window.molbuilder.projects.publishCommit(
                    ctx.dir, ctx.file)""",
                {"dir": str(Path(p).parent), "file": p},
            )

        commit(first)
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

        commit(second)
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '5'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        assert page.evaluate(
            "() => document.querySelector('#info-atoms').textContent.trim()"
        ) == "5", (
            "the second file did not replace the first -- a read-only viewer "
            "refused the install and said nothing, which is exactly what "
            "`enforce` exists to prevent"
        )

    def test_form_edits_followed_by_commit_fires_warning(
            self, page, flask_server, tmp_path, monkeypatch):
        """The form-dirty gate: a user who typed a SIESTA / PySCF
        parameter edit since the last commit must see a warning
        before the next commit silently rebuilds the structure
        section.  Cancel = preserve edits; Discard = proceed."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        # Two distinct structure files so the second commit isn't
        # short-circuited by the same-file guard.
        water = tmp_path / "water.xyz"
        water.write_text(
            "3\nwater\n"
            "O 0.000  0.000 0.000\n"
            "H 0.957  0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        methane = tmp_path / "methane.xyz"
        methane.write_text(
            "5\nmethane\n"
            "C 0.000  0.000 0.000\n"
            "H 0.629  0.629 0.629\n"
            "H -0.629 -0.629 0.629\n"
            "H -0.629  0.629 -0.629\n"
            "H 0.629 -0.629 -0.629\n"
        )
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder"
            "       && typeof window.molbuilder.warningModal"
            "          === 'object'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # Initial commit on water.
        page.evaluate(
            """(c) => window.molbuilder.projects.publishCommit(
                c.dir, c.file)""",
            {"dir": str(tmp_path), "file": str(water)},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # Make the form dirty by typing into a SIESTA param input.
        page.wait_for_selector(
            "#siesta-form-container input", timeout=_BOOT_TIMEOUT_MS)
        siesta_input = page.locator(
            "#siesta-form-container input").first
        siesta_input.focus()
        # A DIGIT, not a letter.  This pressed "a" until 2026-08-15, which
        # worked only because the first control happened to be a text input.
        # The form is now ordered by the shared category vocabulary
        # (`web/form-schema.md` § 1), so the first control is a NUMBER input --
        # and a number input silently swallows a letter, firing no `input`
        # event, leaving the form not-dirty and this test waiting on a modal
        # that was never going to open.  A digit is a real edit in either.
        siesta_input.press("9")
        # Commit a DIFFERENT structure (methane).  Form-dirty gate
        # fires the warning before discarding the parameter edits.
        page.evaluate(
            """(c) => window.molbuilder.projects.publishCommit(
                c.dir, c.file)""",
            {"dir": str(tmp_path), "file": str(methane)},
        )
        page.wait_for_selector(
            "dialog.molbuilder-warning-modal",
            state="attached", timeout=3000)
        # Cancel → structure stays at 3 atoms (water); the rebuild
        # didn't run.
        page.locator(
            'dialog.molbuilder-warning-modal [data-action="cancel"]'
        ).click()
        page.wait_for_selector(
            "dialog.molbuilder-warning-modal",
            state="detached", timeout=2000)
        n = page.evaluate(
            "() => document.querySelector('#info-atoms').textContent.trim()"
        )
        assert n == "3", (
            f"Cancel must preserve the current structure; "
            f"#info-atoms is {n!r} (expected 3 for water)"
        )


# --------------------------------------------------------------------- #
#  Second-visit + external-change pattern (#195, audit follow-up to    #
#  the 2026-06-02 /results stale-dropdown bug).  Per                   #
#  docs/process/testing.md, every tab whose UI       #
#  is driven by a subscriber-on-state-change needs at least one       #
#  test exercising the "user navigated away, external state          #
#  changed, returned" workflow.                                       #
# --------------------------------------------------------------------- #


class TestBuildSecondVisitExternalChange:
    """Audit follow-up: pin the second-visit refresh contract for
    /build so a future regression that breaks the sidebar-pick →
    viewer-update wiring on bfcache restore / tab re-entry fails
    loudly.  The /results file-picker shipped a bug of this exact
    shape (#192) that no single-page-load test could catch."""

    def test_revisiting_build_with_existing_selection_reloads_viewer(
            self, page, flask_server, water_xyz_file):
        """User opens /build, picks water.xyz, navigates to /modify
        (the canonical "go look at the structure" flow), comes back
        to /build.  The viewer MUST still show the structure +
        atom-count line, even though the page just bootstrapped
        fresh.  Pre-fix the bug class: sessionStorage holds the
        file, the sidebar onChange subscriber fires with the same
        value as last time, picker-style "bails on same key"
        suppresses the load -> viewer is empty."""
        _open_build(page, flask_server)
        # Drive the sidebar to a real file (same path as
        # TestSidebarPickLoad).
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        parent = str(Path(p).parent)
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": parent, "file": p},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

        # Navigate to /modify -- sessionStorage carries dir + file
        # over (cross-tab handoff).
        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#projects-sidebar", timeout=_BOOT_TIMEOUT_MS)

        # Come back to /build.  The viewer MUST reload from the
        # persisted selection without a sidebar click.
        page.goto(f"{flask_server}/structure-optimization")
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

    def test_external_xyz_replacement_reloads_on_explicit_load(
            self, page, flask_server, tmp_path, monkeypatch):
        """User picks water.xyz on /build, leaves the tab, the file
        content is REPLACED on disk (different atom count), user
        comes back to /build.

        NEW contract (2026-07-22, persistency wins / explicit load):
        the revisit KEEPS the tab's own loaded data -- the tab does
        NOT auto-reload the sidebar file, so it still shows the
        3-atom water it held.  The externally-changed file is picked
        up only when the user EXPLICITLY clicks Load, which forces a
        fresh re-read from disk (bypassing the same-path dedup) and
        surfaces the new 5-atom structure.  This pins both halves:
        (a) no silent auto-swap on revisit, (b) explicit Load always
        re-reads current disk bytes."""
        # Pin tmp_path as picker root + write the initial 3-atom XYZ.
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        xyz_path = tmp_path / "structure.xyz"
        xyz_path.write_text(
            "3\nwater\n"
            "O 0.000  0.000 0.000\n"
            "H 0.957  0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        _open_build(page, flask_server)
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": str(tmp_path), "file": str(xyz_path)},
        )
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )

        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#projects-sidebar", timeout=_BOOT_TIMEOUT_MS)

        # Replace the file on disk with a different structure
        # (5 atoms, methane-ish).
        import time
        xyz_path.write_text(
            "5\nmethane\n"
            "C 0.000  0.000 0.000\n"
            "H 0.629  0.629 0.629\n"
            "H -0.629 -0.629 0.629\n"
            "H -0.629  0.629 -0.629\n"
            "H 0.629 -0.629 -0.629\n"
        )
        time.sleep(0.5)

        page.goto(f"{flask_server}/structure-optimization")
        # Persistency wins: the tab KEEPS its own 3-atom data on revisit;
        # the externally-changed file is NOT auto-reloaded.
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS,
        )
        # The sidebar still points at the (now-changed) file, so an EXPLICIT
        # Load re-reads it from disk -- the new 5-atom structure appears.
        page.locator("#load-from-sidebar-btn").click()
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '5'",
            timeout=_BOOT_TIMEOUT_MS,
        )



class TestFindingsSitBesideTheirField:
    """A finding belongs next to the control it is about (user, 2026-08-15).

    It landed in the CARD's list until then — the right neighbourhood and the
    wrong address, since a card holds twenty controls — and a finding whose
    field had no card fell all the way to the residual panel at the bottom of
    the page. That is where the ECP warning was: as far from the ECP box as
    the layout allows.

    Asserted in a browser because nothing below one can see it. The placement
    depends on the rendered DOM (the ``.schema-field`` wrapper), on the schema
    the page fetched (which supplies the field's id), and on the live preflight
    round-trip. A unit test can check any one of those and still be looking at
    a page where the warning is somewhere else.
    """

    def test_an_out_of_range_value_warns_beside_its_own_control(
            self, page, flask_server, water_xyz_file):
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder && window.molbuilder.projects"
            "   && typeof window.molbuilder.projects.publishCommit === 'function'",
            timeout=_BOOT_TIMEOUT_MS)
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": str(Path(p).parent), "file": p})
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent.trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS)

        # 5 Ry is far below the recommended floor -> one warn on mesh_cutoff.
        page.fill("#p-mesh-cutoff", "5")
        page.dispatch_event("#p-mesh-cutoff", "change")

        # The warning must appear INSIDE the mesh-cutoff control's own wrapper.
        page.wait_for_function(
            "() => { const i = document.querySelector('#p-mesh-cutoff');"
            "  const w = i && i.closest('.schema-field');"
            "  const u = w && w.querySelector('.field-issues .issue-item');"
            "  return !!u; }",
            timeout=10_000)
        text = page.evaluate(
            "() => document.querySelector('#p-mesh-cutoff')"
            "        .closest('.schema-field')"
            "        .querySelector('.field-issues .issue-item').textContent")
        # Both halves: the meaning, and the keyword you can grep the .fdf for.
        assert "Real-space grid cutoff" in text, text
        assert "MeshCutoff" in text, text

        # And it is not ALSO duplicated into the card list.
        in_card = page.evaluate(
            "() => document.querySelectorAll("
            "  '.card-issues[data-workflow-group] .issue-item').length")
        assert in_card == 0, f"{in_card} finding(s) also sitting in a card list"

    def test_the_warning_clears_from_the_field_when_the_value_is_fixed(
            self, page, flask_server, water_xyz_file):
        """The half a placement change breaks quietest.

        Per-field lists are created on demand, so clearing has to REMOVE them,
        not empty them — an emptied one leaves a gap under every field that
        ever had a finding, and a list that is never cleared shows a warning
        about a value the user already corrected.
        """
        _open_build(page, flask_server)
        page.wait_for_function(
            "() => window.molbuilder && window.molbuilder.projects"
            "   && typeof window.molbuilder.projects.publishCommit === 'function'",
            timeout=_BOOT_TIMEOUT_MS)
        from pathlib import Path
        p = str(Path(water_xyz_file).resolve())
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": str(Path(p).parent), "file": p})
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent.trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS)

        page.fill("#p-mesh-cutoff", "5")
        page.dispatch_event("#p-mesh-cutoff", "change")
        page.wait_for_function(
            "() => !!document.querySelector('#p-mesh-cutoff')"
            "        .closest('.schema-field').querySelector('.field-issues')",
            timeout=10_000)

        page.fill("#p-mesh-cutoff", "300")
        page.dispatch_event("#p-mesh-cutoff", "change")
        page.wait_for_function(
            "() => !document.querySelector('#p-mesh-cutoff')"
            "        .closest('.schema-field').querySelector('.field-issues')",
            timeout=10_000)


class TestSendToTaskSetup:
    """The tab's PRIMARY loop, witnessed in a browser for the first time
    (U6 close, 2026-08-22): every layer below this one was green while
    the hand-over ran — the endpoint in test_task_setup_tab.py, the
    guards in source pins — but nothing ever clicked the button.  That
    is the exact blindness the 2026-08-22 span-cut regression proved
    this lane exists for.

    Two halves: a legal send writes the CLI's own files into the
    selected folder, and a folder already holding a description refuses
    with nothing overwritten (the one-job-per-folder guard, which since
    2026-08-22 also fails CLOSED on a read error)."""

    def _calc_dir(self, tmp_path, monkeypatch):
        """projects/<project>/<topic>/<calc> — the depth the send guard
        demands (a calculation lives under a topic)."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        calc = tmp_path / "proj" / "opt" / "water-run"
        calc.mkdir(parents=True)
        xyz = calc / "water.xyz"
        xyz.write_text(
            "3\nwater\n"
            "O 0.000  0.000 0.000\n"
            "H 0.957  0.000 0.000\n"
            "H -0.239 0.927 0.000\n"
        )
        return calc, xyz

    def _load_and_send(self, page, base_url, calc, xyz):
        _open_build(page, base_url)
        page.wait_for_function(
            "() => window.molbuilder && window.molbuilder.projects"
            "  && typeof window.molbuilder.projects.publishCommit"
            "     === 'function'"
            "  && !!(window.molbuilder.taskHandover)",
            timeout=_BOOT_TIMEOUT_MS)
        page.evaluate(
            "(c) => window.molbuilder.projects.publishCommit(c.dir, c.file)",
            {"dir": str(calc), "file": str(xyz)})
        page.wait_for_function(
            "() => document.querySelector('#info-atoms')"
            ".textContent.trim() === '3'",
            timeout=_BOOT_TIMEOUT_MS)
        # The form must be rendered before collectParams reads it.
        page.wait_for_selector("#siesta-form-container input",
                               timeout=_BOOT_TIMEOUT_MS)
        page.locator("#send-to-task-setup").click()

    def test_send_writes_the_handover_into_the_selected_folder(
            self, page, flask_server, tmp_path, monkeypatch):
        calc, xyz = self._calc_dir(tmp_path, monkeypatch)
        self._load_and_send(page, flask_server, calc, xyz)
        # Success either navigates to /task-setup or (with cell notices)
        # stays put with the written-files status — both mean the files
        # are on disk, which is the contract that matters.
        page.wait_for_function(
            "() => window.location.pathname === '/task-setup'"
            " || (document.querySelector('#handover-status') || {})"
            "      .textContent.includes('Wrote')",
            timeout=10_000)
        assert (calc / "task.1st.json").is_file(), (
            "the hand-over never landed")
        templates = list(calc.glob("*.template.toml"))
        assert templates, "the parameter template never landed"
        sources = list(calc.glob("*.source.xyz"))
        assert sources, "the travelling structure copy never landed"
        import json as _json
        over = _json.loads((calc / "task.1st.json").read_text())
        assert over["engine"]["name"] == "siesta"
        assert "Structure-optimization" in over["_what"], (
            "the hand-over's provenance does not name this tab (E-B9)")

    def test_send_refuses_a_folder_that_is_already_described(
            self, page, flask_server, tmp_path, monkeypatch):
        calc, xyz = self._calc_dir(tmp_path, monkeypatch)
        marker = '{"note": "another calculation lives here"}'
        (calc / "task.json").write_text(marker)
        self._load_and_send(page, flask_server, calc, xyz)
        page.wait_for_function(
            "() => (document.querySelector('#handover-status') || {})"
            "      .textContent.includes('one job per folder')",
            timeout=10_000)
        assert (calc / "task.json").read_text() == marker, (
            "the refusal still overwrote the existing description")
        assert not (calc / "task.1st.json").exists(), (
            "the refusal still wrote the hand-over")

"""End-to-end Playwright tests for /structure-optimization — the describing tab.

What the tests cover:

  * The two parameter forms (SIESTA + PySCF) populate from the CATALOGUE
    schema door, every field inside a card, with engine_key badges and no
    JS error -- a silent renderer break leaves the containers empty (the
    failure class the 2026-08-22 span-cut regression demonstrated: the
    file parsed, every text pin was green, and only this lane knew the
    form was dead).
  * A field that waits on another (`active_when`) greys out and comes
    back as the catalogue declares, over a reload too.
  * The sidebar contract, by its rows: a single click alone must NOT load
    (candidate-only); a double click is the load; a second one replaces the
    first (the #51 `enforce` rule).
  * Second-visit persistence: the tab keeps its own structure across
    navigation, and only an explicit Load re-reads changed disk bytes.
  * Live preflight findings render beside their own control and clear
    when the value is fixed.
  * Send to Task setup — the tab's PRIMARY loop: the button writes the structure pair,
    the parameter template and the hand-over into the selected folder
    through the one shared door (lib/task-handover.js), carrying the
    structure you loaded with its held atoms after a click elsewhere, at
    the frame on screen; a folder already holding a description is
    refused, nothing overwritten (one calculation per folder).

NOT covered here (intentional scope split): the HTTP layer is
test_web.py's; Task setup's own save door is test_task_setup_tab.py's.
Decks are rendered by `prep` on the machine that runs them.
"""
from __future__ import annotations


import pytest


pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from playwright.sync_api import expect  # noqa: E402

from molbuilder.projects import ensure_structure_dir  # noqa: E402
from molbuilder.structure import Structure  # noqa: E402
from molbuilder.workingcopy_structure import StructureCodec  # noqa: E402
from support.sidebar import WAIT_MS, committed, walk_to  # noqa: E402


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #

#: The structures these tests load, each made once and written through the
#: pair door (`StructureCodec.write`, `model/structure.md` § 2.4): water;
#: methane, told apart from it by its atom count; and the water in an 18 Å
#: box it states -- a cell lives in the sidecar (§ 2.3).
_WATER = Structure(elements=["O", "H", "H"],
                   positions=[[0.000, 0.000, 0.000], [0.957, 0.000, 0.000],
                              [-0.239, 0.927, 0.000]])
_METHANE = Structure(elements=["C", "H", "H", "H", "H"],
                     positions=[[0.000, 0.000, 0.000],
                                [0.629, 0.629, 0.629],
                                [-0.629, -0.629, 0.629],
                                [-0.629, 0.629, -0.629],
                                [0.629, -0.629, -0.629]])
_BOXED = _WATER.replace(positions=_WATER.positions + 9.0,
                        cell=[[18.0, 0, 0], [0, 18.0, 0], [0, 0, 18.0]])


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _structures(root):
    """Where a person keeps structures: ``P/structure`` under the projects
    root the server lists (`isolated_projects_root`)."""
    folder = root.resolve() / "P" / "structure"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@pytest.fixture
def water_xyz_file(isolated_projects_root):
    """The water, where the sidebar lists it."""
    return StructureCodec().write(_WATER,
                                  _structures(isolated_projects_root)
                                  / "water.xyz")


@pytest.fixture
def two_xyz_files(isolated_projects_root):
    """TWO structures with DIFFERENT atom counts, in one folder.

    Different counts on purpose: the atom-count line is what tells the two
    apart, so a viewer that ignored the second file would show 3 where the test
    demands 5.  Same-sized structures would let that pass.
    """
    folder = _structures(isolated_projects_root)
    return (StructureCodec().write(_WATER, folder / "water.xyz"),
            StructureCodec().write(_METHANE, folder / "methane.xyz"))


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _js_errors(page) -> list:
    """Every error the page throws or logs from here on, collected as it
    comes -- a throw during init() (the 2026-05 Spectra parse bug) can
    leave a page that still draws."""
    errors = []
    page.on("pageerror",
            lambda exc: errors.append(("pageerror", str(exc))))
    page.on("console", lambda msg: (
        errors.append(("console.error", msg.text))
        if msg.type == "error" else None
    ))
    return errors


def _open_build(page, base_url):
    """The tab opened and LISTENING to the sidebar: its load bar's readout
    written, which the tab does in the same synchronous step it subscribes
    to the sidebar's picks and uses (`support.sidebar.picked`) -- a pick or a
    double click made before that is not heard.  The Load button alone is
    in the server's HTML from the start and says nothing of it.  The form's
    fields are added by JS after the schema fetch lands.  Each test arrives
    at a tab that remembers nothing: its own workspace store
    (`conftest._isolated_workspace_store`) and a fresh browser context."""
    page.goto(f"{base_url}/structure-optimization")
    page.wait_for_function(
        "() => (document.getElementById('load-source-readout') || {})"
        "      .textContent", timeout=WAIT_MS)


def _use(page, path, atoms):
    """``path`` used -- its row double-clicked, *use it*
    (`web/projects.md`) -- and the tab showing its ``atoms``, its form drawn
    for what comes next to read."""
    walk_to(page, path, use=True)
    expect(page.locator("#info-atoms")).to_have_text(str(atoms),
                                                     timeout=WAIT_MS)
    page.wait_for_selector("#siesta-form-container input", timeout=WAIT_MS)


# --------------------------------------------------------------------- #
#  Form schemas populate from the server                                #
# --------------------------------------------------------------------- #


class TestFormSchemasRender:
    """Both engine forms fetch their schema and render fields.  A
    regression in catalogue_to_form_schema or in form-schema.js'
    renderer would leave the containers empty -- silent UI break."""

    @pytest.mark.parametrize("engine", ["siesta", "pyscf"])
    def test_no_field_renders_loose_outside_a_card(
            self, page, flask_server, engine):
        """Every field the catalogue asks for lands INSIDE a card, and the
        page boots without a JS error.

        `form-schema.js` draws the cards the schema lists (``cards``) and
        renders a field whose ``card`` is not among them bare, appended
        straight to the container -- the same invisible outcome as no card
        at all (`web/form-schema.md` § 1a rule 1).  Asked of the rendered
        DOM, where the answer is the same one the person gets.  A throw
        during init() -- the 2026-05 Spectra parse bug's failure mode --
        can leave a page that still draws its fields, so the error list is
        asked too.
        """
        errors = _js_errors(page)
        _open_build(page, flask_server)
        sel = f"#{engine}-form-container"
        page.wait_for_selector(f"{sel} input, {sel} select",
                               state="attached", timeout=WAIT_MS)
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
            f"instead of inside one: their card is not among the schema's "
            f"`cards` (`template.cards`), so the renderer has nowhere to "
            f"draw them.")
        assert errors == [], f"JS errors during the tab's init: {errors}"

    def test_engine_key_badges_present_after_schema_render(
            self, page, flask_server):
        """Each engine's form draws the control's engine-key badge
        (`web/form-schema.md` § 1a: `engine_key` is the control's badge).

        THE SILENT FAILURE: the badges gone from a form -- the served schema
        reaching the page without its ``engine_key`` entries, or the
        renderer no longer drawing them -- while every field still draws.
        What this cannot see is one field without its badge: a field
        lacking its ``engine_key`` is the schema's, asked of every field in
        ``test_web.py::test_engine_key_present_on_every_form_field``, and
        every field kind passes the renderer's one badge call
        (`form-schema.js`), which draws none only where the label already
        is the keyword.
        """
        _open_build(page, flask_server)
        for engine in ("siesta", "pyscf"):
            expect(page.locator(
                f"#{engine}-form-container .schema-engine-key").first,
                f"the {engine} form drew no engine-key badge").to_be_attached(
                    timeout=WAIT_MS)


# --------------------------------------------------------------------- #
#  The sidebar structure flow                                           #
# --------------------------------------------------------------------- #


class TestSidebarStructureFlow:
    """The sidebar is the ONLY way a structure reaches this tab: a commit
    loads, a bare pick does not, a second commit replaces the first.
    """

    def test_commit_mounts_molview_card(
            self, page, flask_server, water_xyz_file):
        """structure-optimization migration: committing a structure mounts the
        FULL concealed MolView component (mode:"readonly" — the same rich card
        Modify uses, read-only) into #viewer-host, and the structure lives in
        molview.data as the single source of truth."""
        _open_build(page, flask_server)
        walk_to(page, water_xyz_file, use=True)
        # The full fused card + 3Dmol canvas mount into the host -- AND the
        # count line has been filled.
        #
        # WAITING FOR THE CARD IS NOT ENOUGH, and that is what made this test
        # flaky.  A viewer MOUNTS BEFORE IT HAS A STRUCTURE (molview.md § 8;
        # `structure-optimization/viewer.js` says so at the call), and the
        # count element is created during card construction
        # (`lib/molview/ui.js`) while its text is written only by `drawList`
        # when atoms arrive.  So `.molviewer-card` + `canvas` can all exist
        # with the count still the EMPTY STRING, and the read below then did
        # `"".split(" of ")[1]` -> `undefined` -> `IndexError`.
        #
        # The structure's own arrival is the thing to wait for, so wait for
        # the line that reports it.  Test-only: nothing about the product
        # changes, and the mount-before-load order is deliberate.
        page.wait_for_function(
            "() => { const h = document.getElementById('viewer-host');"
            "  if (!h || !h.querySelector('.molviewer-card')"
            "         || !h.querySelector('canvas')) return false;"
            "  const c = h.querySelector('.molviewer-selection-count');"
            "  return !!c && c.textContent.includes(' of '); }",
            timeout=WAIT_MS,
        )
        # The viewer holds the committed structure, and the panel's own count
        # line is where that is visible.  Asked through the DOM because there is
        # nowhere else to ask: a viewer belongs to whoever mounted it and there
        # is no registry (molview.md § 5.6).
        count = page.locator(".molviewer-selection-count").inner_text()
        assert count.split(" of ")[1].split()[0] == "3"   # water


    def test_a_single_click_alone_does_NOT_load_structure(
            self, page, flask_server, water_xyz_file):
        """Pins the candidate-only contract for this tab: a single
        sidebar click sets the global pick but MUST NOT auto-
        rebuild the structure section.  Browse-clicking through
        the project tree no longer silently switches the
        molecule the form is configured for."""
        _open_build(page, flask_server)
        walk_to(page, water_xyz_file)
        # THE PAGE TOOK THE PICK -- its load bar names the file as
        # selected -- and a load it began would say so at once: a commit's
        # first act is "Loading <file>…" (`_commitStructure`).
        page.wait_for_function(
            "(n) => (document.getElementById('load-source-readout') || {})"
            "  .textContent === 'Selected: ' + n", arg=water_xyz_file.name,
            timeout=WAIT_MS)
        status = page.locator("#load-status").inner_text()
        assert water_xyz_file.name not in status, (
            f"a single click began a load: the load bar says {status!r}")

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
        structure stayed on screen.  Every other commit test here commits a
        single file, and the second-visit class covers navigating away and
        back, or the SAME file changed on disk -- neither is a second pick in
        one visit.

        The two fixtures have different atom counts because that is the only
        thing on screen that can tell them apart.
        """
        first, second = two_xyz_files
        _open_build(page, flask_server)
        _use(page, first, 3)
        _use(page, second, 5)


# --------------------------------------------------------------------- #
#  Second visit: the tab keeps what it held; the file changed on disk    #
#  is read again only on an explicit Load.                               #
# --------------------------------------------------------------------- #


class TestBuildSecondVisitExternalChange:
    """The second-visit contract on /structure-optimization: a regression
    that breaks the sidebar-pick → viewer-update wiring on tab re-entry
    fails loudly.  The /results file-picker shipped a bug of this exact
    shape (#192) that no single-page-load test could catch."""

    def test_external_xyz_replacement_reloads_on_explicit_load(
            self, page, flask_server, isolated_projects_root):
        """The person loads water.xyz on /structure-optimization, leaves the
        tab, the file is REPLACED on disk (a different atom count), and
        comes back.

        The contract (persistency wins / explicit load):
        the revisit KEEPS the tab's own loaded data -- the tab does
        NOT auto-reload the sidebar file, so it still shows the
        3-atom water it held.  The externally-changed file is picked
        up only when the user EXPLICITLY clicks Load, which forces a
        fresh re-read from disk (bypassing the same-path dedup) and
        surfaces the new 5-atom structure.  This pins both halves:
        (a) no silent auto-swap on revisit, (b) explicit Load always
        re-reads current disk bytes."""
        # The water, where the sidebar lists it.
        xyz_path = StructureCodec().write(
            _WATER, _structures(isolated_projects_root) / "structure.xyz")
        _open_build(page, flask_server)
        _use(page, xyz_path, 3)

        page.goto(f"{flask_server}/molbuilder")
        page.wait_for_selector("#projects-sidebar", timeout=WAIT_MS)

        # Replaced on disk by another structure: methane, 5 atoms.
        StructureCodec().write(_METHANE, xyz_path)

        page.goto(f"{flask_server}/structure-optimization")
        # Persistency wins: the tab KEEPS its own 3-atom data on revisit;
        # the externally-changed file is NOT auto-reloaded.
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '3'",
            timeout=WAIT_MS,
        )
        # The sidebar still points at the (now-changed) file, so an EXPLICIT
        # Load re-reads it from disk -- the new 5-atom structure appears.
        page.locator("#load-from-sidebar-btn").click()
        page.wait_for_function(
            "() => document.querySelector('#info-atoms').textContent"
            ".trim() === '5'",
            timeout=WAIT_MS,
        )


class TestFindingsSitBesideTheirField:
    """A finding belongs next to the control it is about (user, 2026-08-15).

    Asserted in a browser because nothing below one can see it. The placement
    depends on the rendered DOM (the ``.schema-field`` wrapper), on the schema
    the page fetched (which supplies the field's id), and on the live preflight
    round-trip. A unit test can check any one of those and still be looking at
    a page where the warning is somewhere else.
    """

    def test_an_out_of_range_value_warns_beside_its_own_control(
            self, page, flask_server, water_xyz_file):
        _open_build(page, flask_server)
        _use(page, water_xyz_file, 3)

        # 5 Ry is far below the recommended floor -> one warn on mesh_cutoff.
        committed(page, "#p-mesh-cutoff", "5")

        # The warning must appear INSIDE the mesh-cutoff control's own wrapper.
        page.wait_for_function(
            "() => { const i = document.querySelector('#p-mesh-cutoff');"
            "  const w = i && i.closest('.schema-field');"
            "  const u = w && w.querySelector('.field-issues .issue-item');"
            "  return !!u; }",
            timeout=WAIT_MS)
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
            "  '.card-issues[data-card] .issue-item').length")
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
        _use(page, water_xyz_file, 3)

        committed(page, "#p-mesh-cutoff", "5")
        page.wait_for_function(
            "() => !!document.querySelector('#p-mesh-cutoff')"
            "        .closest('.schema-field').querySelector('.field-issues')",
            timeout=WAIT_MS)

        committed(page, "#p-mesh-cutoff", "300")
        page.wait_for_function(
            "() => !document.querySelector('#p-mesh-cutoff')"
            "        .closest('.schema-field').querySelector('.field-issues')",
            timeout=WAIT_MS)


    def test_a_finding_about_the_mesh_sits_beside_the_mesh(
            self, page, flask_server, water_xyz_file):
        """A triple is one field, and its finding sits beside it like any
        other's (`web/form-schema.md` § 1.1: the settings gate's refusal
        lands beside the field it names; the K3 review): the field's id was
        on its three cells alone, so the finding about `kgrid` -- placed by
        that id -- fell back to the card."""
        _open_build(page, flask_server)
        _use(page, water_xyz_file, 3)

        # A count of 0 is past the count's limit: one finding, on kgrid.
        for lab, v in zip("xyz", ("0", "4", "4")):
            page.fill(f"#p-kgrid-{lab}", v)
        page.wait_for_function(
            "() => { const w = document.querySelector('#p-kgrid-x')"
            "          .closest('.schema-field');"
            "  return !!(w && w.querySelector('.field-issues .issue-item')); }",
            timeout=WAIT_MS)
        in_card = page.evaluate(
            "() => document.querySelectorAll("
            "  '.card-issues[data-card] .issue-item').length")
        assert in_card == 0, f"{in_card} finding(s) also sitting in a card list"


def test_every_finding_the_live_check_returns_is_on_the_page(
        page, flask_server, water_xyz_file):
    """Every finding each engine's live check returns is drawn on the tab, a
    row each, in that engine's panel and form -- beside its control, on its
    card, or in the panel (`science/validation.md` § 4.1 R2, R3).

    THE SILENT FAILURE: a finding lost on its way to the page -- the tab
    drawing only those about a field, or none -- so a structure the check
    has something to say about reads as a clean one.

    Water loaded, then SIESTA's mesh cutoff set far below its floor: each
    engine's answer read off the page's own request -- PySCF's to the load,
    SIESTA's to the edit, which speaks of a field and of the structure.

    MUTATIONS THIS MUST FAIL AGAINST: the page without the renderer's global
    entry point; the tab drawing only the findings about a field, or only
    the others; a finding drawn twice."""
    from support.findings import assert_every_finding_drawn

    def answer_for(engine):
        return lambda r: ("/api/build/preflight" in r.url
                          and (r.request.post_data_json or {}).get("engine")
                          == engine)
    _open_build(page, flask_server)
    with page.expect_response(answer_for("siesta")), \
            page.expect_response(answer_for("pyscf")) as loaded:
        walk_to(page, water_xyz_file, use=True)
    assert_every_finding_drawn(page, ["#pyscf-issues", "#pyscf-form-container"],
                               loaded.value.json()["issues"])
    with page.expect_response(answer_for("siesta")) as edited:
        committed(page, "#p-mesh-cutoff", "5")
    said = edited.value.json()["issues"]
    wheres = [i.get("where") or "" for i in said]
    assert "config.mesh_cutoff" in wheres, said
    assert any(not w.startswith("config.") for w in wheres), said
    assert_every_finding_drawn(page, ["#fdf-issues", "#siesta-form-container"],
                               said)


class TestSendToTaskSetup:
    """The tab's PRIMARY loop, witnessed in a browser: every layer below
    this one -- the endpoint in test_task_setup_tab.py -- can be green while
    nothing clicks the button.

    Two halves: a legal send writes the CLI's own files into the
    selected folder, and a folder already holding a description refuses
    with nothing overwritten (the one-calculation-per-folder guard,
    `describe.refuse_described`: the description's file is there or it is
    not, and the guard reads nothing of it)."""

    def _calc_dir(self, *, boxed=False, pseudos=True):
        """A calculation's folder, ``<project>/optimization/<calc>`` -- under
        a topic, the depth the Send demands -- made by the projects door
        (`projects.ensure_structure_dir`, which takes only a canonical
        topic, as a person's tree has: `web/projects.md` § 2) under the
        projects root the server lists, the test's own
        (`isolated_projects_root`); the water in it: ``(calc, xyz)``.

        ``boxed``: the water in the box it states (``_BOXED``).  Without
        one the preflight raises `cell.vacuum_defaulted` and the send
        answers with notices; with one it answers clean
        (`web/handover-procedure.md` § 2.1).  ``pseudos``: the
        pseudopotentials beside the calculation, as a person leaves them --
        the Send refuses a SIESTA calculation without them (§ 2.2)."""
        calc = ensure_structure_dir("proj", "optimization",
                                    "water-run").resolve()
        xyz = StructureCodec().write(_BOXED if boxed else _WATER,
                                     calc / "water.xyz")
        if pseudos:
            from conftest import write_pseudos
            write_pseudos(calc, ["O", "H"])
        return calc, xyz

    def _open_on(self, page, base_url, folder):
        """The tab, its sidebar walked to ``folder`` by its rows -- where
        the Send writes (`web/projects.md` § 4)."""
        _open_build(page, base_url)
        page.wait_for_function(
            "() => !!(window.molbuilder && window.molbuilder.taskHandover)",
            timeout=WAIT_MS)
        walk_to(page, folder)

    def _send(self, page, says="Wrote"):
        """Send to Task setup pressed, and the tab's status saying
        ``says``."""
        page.locator("#send-to-task-setup").click()
        page.wait_for_function(
            "(w) => (document.querySelector('#handover-status')?.textContent"
            "        || '').includes(w)", arg=says, timeout=WAIT_MS)

    @staticmethod
    def _handed_over(dest, *, frames=False):
        """The structure the hand-over written into ``dest`` names, read
        back through the codec -- every frame it holds with ``frames``."""
        import json as _json
        over = _json.loads((dest / "task.1st.json").read_text())
        return StructureCodec().load(dest / over["structure"]["source"],
                                     frames=frames)

    @pytest.mark.parametrize("boxed", [False, True],
                             ids=["notices", "clean"])
    def test_a_send_writes_the_handover_and_stays(
            self, page, flask_server, isolated_projects_root, boxed):
        """A Send writes the structure's pair, the parameter template and
        `task.1st.json` into the folder selected in the sidebar, for the
        engine on screen, and the page stays where it is, its status naming
        Task setup as the road after it (`web/handover-procedure.md` § 2,
        steps 1-2; § 3: the hand-over says what it is in its `_what` line)
        -- on both of the door's answers: with notices (a structure stating
        no cell, given the default vacuum) and clean (a box the structure
        states; § 2.1), each row's branch asked of the answer the page
        received.

        This tab's job ends when the files are on disk; opening Task setup
        is the person's move (user, 2026-09-19: *"we just skip the fancy tab
        to tab jump connection to avoid implicit coupling"*).

        THE SILENT FAILURE: the button clicked and nothing written -- every
        route below it green -- or written and the page gone elsewhere, so
        the person edits whatever folder another tab last showed.  WHAT THE
        JUMP COST: it relied on the sidebar's selection being shared, but
        sessionStorage slots are PER PAGE and a page's own slot SHADOWS the
        shared fallback (`projects/state.js::handOffSelection`), so Task
        setup opened whatever folder IT last showed.  Measured 2026-09-19:
        the hand-over landed correctly in `optimization/junction6-hier` and
        Task setup opened `spectrum/bridge-hier`, ready to edit the wrong
        calculation."""
        calc, xyz = self._calc_dir(boxed=boxed)
        self._open_on(page, flask_server, calc)
        _use(page, xyz, 3)
        with page.expect_response(
                lambda r: "/api/task-setup/handover" in r.url,
                timeout=WAIT_MS) as answered:
            self._send(page)
        notices = answered.value.json()["notices"]
        assert bool(notices) is not boxed, (
            "this row must reach its own branch of the door's answer, or "
            f"the assertions below are about the other; it answered {notices}")
        assert (calc / "task.1st.json").is_file(), (
            "the hand-over never landed")

        # AND IT STAYS PUT (user, 2026-09-19).
        assert page.evaluate("() => window.location.pathname") == (
            "/structure-optimization"), (
            "the hand-over navigated away; it writes files and stops")
        status = page.evaluate(
            "() => document.querySelector('#handover-status').textContent")
        assert "Task setup" in status, (
            "nothing opens Task setup for you, so the status has to say to; "
            "got: " + status)
        templates = list(calc.glob("*.template.toml"))
        assert templates, "the parameter template never landed"
        sources = list(calc.glob("*.source.xyz"))
        assert sources, "the travelling structure copy never landed"
        import json as _json
        over = _json.loads((calc / "task.1st.json").read_text())
        assert over["engine"]["name"] == "siesta"
        from molbuilder.web.tabs import TABS
        tab = next(t["label"] for t in TABS
                   if t["key"] == "structure-optimization")
        assert f"from the {tab} tab" in over["_what"], (
            "the hand-over's provenance does not name this tab (E-B9)")

    def test_send_keeps_what_the_live_check_says(
            self, page, flask_server, isolated_projects_root):
        """A Send adds its findings to the live check's and clears none
        (`handover-procedure.md` § 2.1): the live check's are still true of
        the form it sent.

        MUTATION THIS MUST FAIL AGAINST: the send handing the panel its own
        notices alone (the live warning is gone)."""
        calc, xyz = self._calc_dir()
        beside = ("() => { const w = document.querySelector('#p-mesh-cutoff')"
                  "  .closest('.schema-field');"
                  "  return !!w.querySelector('.field-issues .issue-item'); }")
        self._open_on(page, flask_server, calc)
        _use(page, xyz, 3)
        # 5 Ry is far below the recommended floor: one warning, beside the
        # control.
        committed(page, "#p-mesh-cutoff", "5")
        page.wait_for_function(beside, timeout=WAIT_MS)

        self._send(page)
        # The send's own notice -- the default vacuum this box was given --
        # came back into the panel...
        page.wait_for_function(
            "() => [...document.querySelectorAll('#fdf-issues .issue-item')]"
            "      .some(r => r.textContent.includes('No vacuum set'))",
            timeout=WAIT_MS)
        # ...beside the live warning, which is still there.
        assert page.evaluate(beside), (
            "the send cleared the live check's warning from the form")

    def test_send_waits_for_the_pseudopotentials(self, page, flask_server,
                                                  isolated_projects_root):
        """The pseudopotential folder is settled before the files are
        written (`web/handover-procedure.md` § 2.2, plan § 5w K20; user,
        2026-09-30: *"it seems user easily misses this in the first setup and
        only finds out after the script is generated"*).  On the SIESTA form
        the field is drawn required and red while empty, and the live check
        names the tree's `pseudopotential` folder, which covers the
        structure; a Send with no folder and no files beside the calculation
        is refused -- the finding beside the field, brought into view,
        nothing written; set to the folder, the Send writes.

        MUTATIONS THIS MUST FAIL AGAINST: the Send not asking the check (the
        files are written); the field not drawn required; the suggestion not
        made; the refusal not brought into view."""
        calc, xyz = self._calc_dir(pseudos=False)
        from conftest import write_pseudos
        library = isolated_projects_root / "pseudopotential"
        library.mkdir()
        write_pseudos(library, ["O", "H"])
        field = ("document.querySelector('#p-psml-lib')"
                 ".closest('.schema-field')")

        self._open_on(page, flask_server, calc)
        _use(page, xyz, 3)
        page.wait_for_function(
            f"() => {{ const w = {field};"
            "  return w.classList.contains('is-required')"
            "    && w.classList.contains('is-empty')"
            "    && [...w.querySelectorAll('.field-issues .issue-item')]"
            "       .some(r => r.textContent.includes("
            "           'folder covers all 2 species')); }",
            timeout=WAIT_MS)
        self._send(page, says="pseudopotentials are not settled")
        assert not (calc / "task.1st.json").exists(), (
            "the refusal still wrote the hand-over")
        seen = page.evaluate(
            f"() => {{ const r = {field}.querySelector("
            "    \".field-issues .issue-item[data-severity='error']\");"
            "  if (!r) return null;"
            "  const b = r.getBoundingClientRect();"
            "  return {top: b.top, bottom: b.bottom,"
            "          height: window.innerHeight}; }")
        assert seen is not None, "the refusal is not beside the field"
        assert page.evaluate(
            f"() => {field}.querySelectorAll('.field-issues .issue-item').length"
        ) == 1, "one fact said twice beside the field, in two severities"
        assert 0 <= seen["top"] and seen["bottom"] <= seen["height"], seen

        page.fill("#p-psml-lib", "pseudopotential")
        assert not page.evaluate(
            f"() => {field}.classList.contains('is-empty')")
        self._send(page)
        assert (calc / "task.1st.json").is_file()

    def test_send_carries_the_held_atoms_of_the_file_you_loaded(
            self, page, flask_server, isolated_projects_root):
        """A Send writes the structure you loaded, with the atoms it holds
        still, into the folder you clicked -- not the file you clicked last.

        THE SILENT FAILURE (2026-06-14).  A pair whose sidecar held 25 gold
        atoms still was loaded; the person then clicked into a sibling folder
        to write the calculation there, and the request was built from the
        file clicked last, which had no sidecar.  The deck held nothing still,
        SIESTA relaxed the gold surface, and the run looked like any other.

        THE CONTRACT: a single click is a preview, a double click is *use
        it* (`web/projects.md`); a request carries the structure on screen,
        its held atoms read with its coordinates (`web/molview.md` § 9.3a);
        the hand-over's pair keeps them (`handover-procedure.md`'s round
        trip).  Driven through the sidebar's own rows: a double click on the
        pair, a click on the sibling folder, a click on a copy in it that
        holds nothing still.

        MUTATIONS THIS MUST FAIL AGAINST: a single click loading the file;
        the load dropping the sidecar; `exportFile` dropping the held atoms;
        the Send building its structure from the sidebar's file."""
        from conftest import write_pseudos
        import numpy as np
        loaded_dir, dest = (
            ensure_structure_dir("proj", "optimization", calc).resolve()
            for calc in ("relaxed", "next-run"))
        held = StructureCodec().write(_BOXED.replace(frozen_atoms=[0]),
                                      loaded_dir / "water.xyz")
        # THE FILE CLICKED LAST: the same atoms a little apart, nothing held.
        plain = StructureCodec().write(
            _BOXED.replace(positions=_BOXED.positions + [0.3, 0.0, 0.0]),
            dest / "water.xyz")
        write_pseudos(dest, ["O", "H"])

        self._open_on(page, flask_server, loaded_dir)
        _use(page, held, 3)
        walk_to(page, plain)
        # Whatever the clicks began has ended before the Send asks the
        # viewer: a load a single click started would be on screen by now.
        expect(page.locator("#load-status")).not_to_contain_text(
            "Loading", timeout=WAIT_MS)

        self._send(page)
        sent = self._handed_over(dest)
        assert sent.frozen_atoms == [0], (
            "the Send dropped the atoms the loaded pair holds still: "
            f"{sent.frozen_atoms}")
        assert np.allclose(sent.positions, _BOXED.positions, atol=1e-5), (
            "the Send wrote another structure than the one loaded")

    def test_send_writes_the_frame_on_screen(
            self, page, flask_server, isolated_projects_root):
        """A Send writes the frame you are looking at, after the frame bar
        has moved away from the one the file opened on.

        THE SILENT FAILURE.  This tab once copied the geometry when a file
        loaded and read the labels and the cell fresh at each request, so a
        request carried coordinates the viewer no longer showed: the server
        judged, and the calculation would start from, a structure on nobody's
        screen (`science/validation.md` § 4.1, F1).

        THE CONTRACT: a request's structure is read from the viewer at the
        moment it is sent, in one read, for the frame on screen (F1;
        `web/molview.md` § 9.3a); a file opens on its first frame
        (§ 10.8).  Driven through the frame bar's own buttons, on a
        relaxation's three frames -- a frame set, made through its door
        (`Structure.with_frames`, `model/structure.md` § 2.2e) -- to the
        MIDDLE one: neither the first nor the last, the two frames a
        shortcut would answer.

        MUTATIONS THIS MUST FAIL AGAINST: the structure copied once, when
        the file loaded, and sent from the copy; `exportFile` answering the
        first frame, or the last, whatever is on screen; or every frame from
        the one on screen to the last."""
        from conftest import write_pseudos
        import numpy as np
        calc = ensure_structure_dir("proj", "optimization", "scan").resolve()
        step = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, -0.1, 0.0]])
        frames = [_BOXED.positions + k * step for k in range(3)]
        trajectory = StructureCodec().write(_BOXED.with_frames(frames),
                                            calc / "relax.xyz")
        write_pseudos(calc, ["O", "H"])

        self._open_on(page, flask_server, calc)
        _use(page, trajectory, 3)
        counter = page.locator(".molviewer-frames-counter")
        expect(counter).to_have_text("1 / 3", timeout=WAIT_MS)
        page.get_by_role("button", name="Next frame").click()
        expect(counter).to_have_text("2 / 3", timeout=WAIT_MS)

        self._send(page)
        sent = self._handed_over(calc, frames=True)
        assert sent.n_frames == 1, (
            f"the Send wrote {sent.n_frames} frames; the one on screen is one")
        assert np.allclose(sent.positions, frames[1], atol=1e-5), (
            "the Send wrote another frame than the one on screen: "
            f"{sent.positions.tolist()}")

    def test_send_refuses_a_folder_that_is_already_described(
            self, page, flask_server, tmp_path, monkeypatch):
        """A Send into a folder that already holds a calculation is refused,
        naming the folder by its address from the projects root, as every
        door that describes names it, and nothing in the folder changes (one
        calculation per folder, `describe.refuse_described`).

        The folder is described on the road -- `molbuilder describe`
        (`support.road`) -- and its own structure is the one loaded: the
        person double-clicks the pair the description travels with, sets the
        tree's pseudopotential folder, and sends into the folder it came
        from.

        THE SILENT FAILURE: a Send overwriting the description in place --
        a second calculation's files over the first's.  MUTATION THIS MUST
        FAIL AGAINST: the hand-over naming the folder by the full path the
        page sent, where the command line names it by its address."""
        from support.road import described, road_tree
        calc = described(road_tree(tmp_path, monkeypatch)).resolve()
        before = {f.name: f.read_bytes() for f in calc.iterdir()
                  if f.is_file()}
        xyz = next(calc.glob("*.source.xyz"))
        self._open_on(page, flask_server, calc)
        _use(page, xyz, 2)
        page.fill("#p-psml-lib", "pseudopotential")
        # The address the road described it at (`support.road`), and not
        # the full path the page sent, which ends in the same words.
        self._send(page, says=f"P/optimization/{calc.name} already holds a "
                              f"calculation")
        said = page.locator("#handover-status").text_content()
        assert str(calc) not in said, said
        after = {f.name: f.read_bytes() for f in calc.iterdir()
                 if f.is_file()}
        assert after == before, (
            "the refusal still changed the folder: "
            f"{sorted(set(after) ^ set(before))} came or went, "
            f"{sorted(n for n in after if n in before and after[n] != before[n])}"
            " were rewritten")


# --------------------------------------------------------------------- #
#  The recommended-value panel, on both engines' forms                  #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("engine", ["siesta", "pyscf"])
def test_a_field_off_its_recommended_value_raises_the_panel(
        page, flask_server, engine):
    """Move one field; the panel appears and names it.  Reset; it goes.

    WHY IN A BROWSER.  The string `id="siesta-recommend"` in index.html and
    the host ids before `mountRecommended(engine, host` in viewer.js are all
    true of a page where `mountRecommended` returns at its first line -- it
    does exactly that when `formSchema.diffFromDefaults` is missing, and it
    is a `return`, not a throw, so nothing anywhere says so.

    Driven per engine because "both forms carry it" is the claim, and the
    pyscf panel is behind a tab a person has to click.  The field is
    discovered from the rendered form rather than named here: which field is
    first is the schema's business, and hard-coding one would make this fail
    the day the schema is reordered.
    """
    _open_build(page, flask_server)
    page.click(f'.tab-btn[data-tab="{engine}"]')
    page.wait_for_function(
        "(e) => document.querySelectorAll("
        "  '#' + e + '-form-container input, #' + e + '-form-container select'"
        ").length > 0", arg=engine, timeout=WAIT_MS)

    moved = page.evaluate("""(engine) => {
        const box = document.getElementById(engine + "-form-container");
        // A field nobody chose is BLANK, the kind's recommendation its hint
        // (form-schema.md § 1.1) -- so the field is found by a numeric hint.
        // One with no recommendation has nothing to be off:
        // diffFromDefaults skips `f.default === null | undefined`, which is
        // how the first pass of this test picked pyscf's `net_charge` and
        // concluded the panel was broken.
        const num = [...box.querySelectorAll('input[type="number"]')]
            .find((n) => n.value === "" && n.placeholder !== ""
                         && !isNaN(parseFloat(n.placeholder)));
        if (!num) return {error: "no number field with a recommended hint in "
                                 + "the " + engine + " form"};
        // The panel tags each row's checkbox with the SCHEMA field name,
        // which is the input's id minus the engine prefix, dashes to
        // underscores -- `p-mesh-cutoff` -> `mesh_cutoff`, `py-basis` ->
        // `basis`.  Matching on that ties the row to the field that moved
        // without depending on how the label happens to be marked up.
        return {was: num.placeholder, id: num.id,
                name: num.id.replace(/^p(y)?-/, "").replace(/-/g, "_")};
    }""", engine)
    assert "error" not in moved, moved.get("error")
    # TYPED, which is `input` ALONE on the way up -- `change` alone on the
    # way back down (see the reset below).  Each listener is proved
    # separately because they answer different writers: a person types
    # (input), while the session restore writes the form without a
    # keystroke and fires change.  A panel wired to only one of them shows
    # a stale count exactly when it matters.
    page.fill(f"#{moved['id']}", f"{float(moved['was']) + 137:g}")

    panel = page.locator(f"#{engine}-recommend")
    panel.wait_for(state="visible", timeout=WAIT_MS)
    rows = page.eval_on_selector_all(
        f"#{engine}-recommend .rec-diff-list li input[type=checkbox]",
        "els => els.map(e => e.value)")
    assert rows == [moved["name"]], (
        f"the panel is up but lists {rows!r}; the one field that moved was "
        f"{moved['name']!r}, and a panel that names the wrong parameter -- "
        f"or names extras nobody touched -- sends the reset to the wrong box")

    count = page.locator(f"#{engine}-recommend .rec-diff-count").inner_text()
    # ...in the word every caption uses for it (`form-schema.md` § 3.1,
    # 2.68.1): a default, not molbuilder's recommendation.
    assert count == "1 setting is not at its default", count

    # And its Reset puts the field back to BLANK -- not chosen, the
    # recommendation what applies, never the recommended number typed in,
    # which the template would record as the person's (form-schema.md
    # § 3.1) -- and the panel goes: one that only ever appears is a banner.
    page.click(f"#{engine}-recommend > summary")        # it opens folded
    page.check(f"#{engine}-recommend .rec-diff-list input[type=checkbox]")
    page.click(f"#{engine}-recommend .rec-diff-reset")
    panel.wait_for(state="hidden", timeout=WAIT_MS)
    assert page.evaluate(
        "(id) => document.getElementById(id).value", moved["id"]) == ""


def test_the_form_holds_only_what_the_person_gave_and_keeps_it_over_a_reload(
        page, flask_server):
    """`web/form-schema.md` § 1.1 (plan § 5w K7), on the real
    Structure-optimization page: a new calculation's form holds nothing
    until the person gives it something -- a field is blank, not chosen,
    the kind's recommendation its hint, and a box nobody answered is
    indeterminate.  What the person gave survives a reload, and so does
    what they did not: the restore writes through the form's own writer,
    so a box nobody answered does not come back "off".

    The words are the served schema's -- a source said in `source_words`,
    the recommendation the kind's `default` in its unit (§ 1.1) -- read off
    the answer the page received."""
    with page.expect_response(
            lambda r: "/api/build/schema/siesta?" in r.url) as served:
        _open_build(page, flask_server)
    schema = served.value.json()["schema"]
    words = schema["source_words"]
    mesh = next(f for section in schema["sections"]
                for f in section["fields"] if f["name"] == "mesh_cutoff")
    page.wait_for_selector("#p-mesh-cutoff", timeout=WAIT_MS)

    def state():
        return page.evaluate("""() => {
            const m = document.getElementById("p-mesh-cutoff");
            const b = document.getElementById("p-write-hs");
            return {mesh: m.value, hint: m.placeholder, box: b.indeterminate,
                    said: m.closest(".schema-field")
                           .querySelector(".schema-source").textContent};
        }""")

    recommended = f"{mesh['default']:g}"
    assert state() == {
        "mesh": "", "hint": recommended, "box": True,
        # ...led by where the default comes from, the served words
        # (`form-schema.md` § 1.1, 2.68.1).
        "said": f"{words['default']} \u00b7 {mesh['default_words']}: "
                f"{recommended} {mesh['unit']}"}, (state(), mesh)
    page.fill("#p-mesh-cutoff", "450")
    assert state()["said"] == words["person"]
    # A half-typed mesh beside it does not cost the form its save: the field
    # that will not read keeps what it was last saved with (the K7 review --
    # the whole engine's save was skipped, and the reload lost the 450).
    page.fill("#p-kgrid-x", "4.5")

    page.reload()
    page.wait_for_function(
        "() => (document.getElementById('p-mesh-cutoff') || {}).value === '450'",
        timeout=WAIT_MS)
    after = state()
    assert after["box"] is True, "the box nobody answered came back answered"
    assert after["said"] == words["person"], after


#: The tab's two locks, the catalogue's (`active_when`, `web/form-schema.md`
#: § 1.3): a relaxation's step settings wait on Relaxation type being a
#: relaxer, the PCM model on a solvent being chosen.
_WAITING = ("p-relax-steps", "p-relax-force-tol", "p-relax-max-displ",
            "py-solvent-method")


def test_a_field_waits_on_the_field_the_catalogue_names(page, flask_server):
    """`web/form-schema.md` § 1.3: *"a relaxation's step settings wait on
    Relaxation type being a relaxer (`relax_type = ["CG", "Broyden",
    "FIRE"]`), and the solvent model on a solvent being chosen (`solvent =
    "set"`, an optional field holding any value) -- the
    Structure-optimization tab's two locks, its own code until 2.65.0"*.
    The renderer draws a waiting field *"greyed out while any named field
    holds another value -- its words muted, its control disabled, and a
    line beside it saying what it waits on (`active_words`) -- and follows
    each change"*; a blank applies the kind's recommendation (§ 1.1), CG
    here, so the step settings are live at load and the PCM model, with no
    solvent, is not.  After a reload the restore writes the held values
    through the form's writer, and the greying follows them.

    THE SILENT FAILURE: 2.65.0 deleted the tab's own lock code for the
    renderer's one rule, which only a browser runs -- a lock lost on the way
    leaves the step settings editable under `none`, where they enter
    nothing, or the PCM model greyed for good, and the form still sends
    both, so nothing says so.

    MUTATIONS THIS MUST FAIL AGAINST: the `"set"` branch dropped (the PCM
    model never live); a blank read as no value rather than the default
    (the step settings greyed at load); the lock applied at render only (a
    change, or the restore's write, leaves it as it was)."""
    errors = _js_errors(page)
    _open_build(page, flask_server)
    for sel in ("#p-relax-type", "#py-solvent", *(f"#{i}" for i in _WAITING)):
        page.wait_for_selector(sel, state="attached",
                               timeout=WAIT_MS)

    def states():
        # A FIELD IS LIVE WHEN ITS CONTROLS TAKE AN EDIT and nothing beside
        # it says what it waits on; greyed out when they take none and the
        # line beside it says.
        got = page.evaluate("""(ids) => ids.map((i) => {
            const w = document.getElementById(i).closest('.schema-field');
            const why = w.querySelector(':scope > .lock-reason');
            return [[...w.querySelectorAll('input, select, textarea')]
                        .every((c) => !c.disabled),
                    !!why && !why.hidden];
        })""", list(_WAITING))
        return ["live" if on and not said else
                "greyed" if said and not on else
                f"mixed (live={on}, said={said})" for on, said in got]

    seen = {"at load": states()}
    page.select_option("#p-relax-type", value="none")
    seen["no relaxation"] = states()
    page.select_option("#p-relax-type", value="Broyden")
    seen["a relaxer"] = states()
    page.click('.tab-btn[data-tab="pyscf"]')
    page.fill("#py-solvent", "water")
    seen["a solvent"] = states()
    page.click('.tab-btn[data-tab="siesta"]')
    page.select_option("#p-relax-type", value="none")

    page.reload()
    page.wait_for_function(
        "() => (document.getElementById('p-relax-type') || {}).value === 'none'"
        "   && (document.getElementById('py-solvent') || {}).value === 'water'",
        timeout=WAIT_MS)
    seen["after a reload"] = states()

    steps, pcm = 3, 1
    assert seen == {
        "at load":        ["live"] * steps + ["greyed"] * pcm,
        "no relaxation":  ["greyed"] * steps + ["greyed"] * pcm,
        "a relaxer":      ["live"] * steps + ["greyed"] * pcm,
        "a solvent":      ["live"] * steps + ["live"] * pcm,
        "after a reload": ["greyed"] * steps + ["live"] * pcm,
    }, seen
    assert errors == [], f"JS errors on the tab: {errors}"

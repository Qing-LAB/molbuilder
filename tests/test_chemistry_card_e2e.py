"""The chemistry card, in the browser: the charge and spin a calculation will
carry, and why (``docs/science/chemistry-correctness.md`` §§ 2.5, 2a.5).

PINS the forms' half of the electronic state (plan § 5s.3, P3's
done-condition): loading a structure shows, for each engine form, the state the
one class resolves for exactly what that form says -- each value with its
source -- and **a blank field stays blank**: nothing is filled in.  Typing a
charge changes the card.  The chip on each form's Profile card says the same.

Driven the way a person drives it: the file picked in the sidebar, the Load
button, a field typed into; on the Transport tab, a cited junction restored
the way a returning person's tab restores it.  Headless chromium through
pytest-playwright; no engine runs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


def _ready_for_a_pick(page):
    """The tab is LISTENING to the sidebar.  The sidebar's door existing is
    not enough: each tab subscribes to its changes inside
    ``whenReady("projects")``, right after it writes the load bar's readout
    in the same synchronous step -- so a written readout means a pick is
    heard.  Waiting only for the door let a warm second page pick before the
    tab subscribed, and the Load button never woke (measured 2026-09-29: the
    second test of this file, passing alone)."""
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.projects "
        "      && typeof window.molbuilder.projects.setShared === 'function'"
        "      && document.getElementById('load-source-readout')"
        "             .textContent.length > 0",
        timeout=10000)


def _load(page, base, tab, target):
    """Open ``tab``, pick ``target`` in the sidebar, press Load -- and wait
    for the card to have answered."""
    page.goto(f"{base}/{tab}", wait_until="domcontentloaded")
    _ready_for_a_pick(page)
    page.evaluate(
        "(p) => window.molbuilder.projects.setShared("
        "  p.substring(0, p.lastIndexOf('/')), p)", str(target))
    page.wait_for_function(
        "() => !document.getElementById('load-from-sidebar-btn').disabled",
        timeout=5000)
    page.locator("#load-from-sidebar-btn").click()
    _answered(page)


def _answered(page, needle=""):
    page.wait_for_function(
        "(n) => { const p = document.getElementById('chemistry-panel');"
        "  const s = document.getElementById('chemistry-state');"
        "  return p && !p.hidden && s && s.textContent.length > 0"
        "    && s.textContent.includes(n); }",
        arg=needle, timeout=15000)


def _card(page) -> str:
    return page.evaluate(
        "() => document.getElementById('chemistry-state').textContent")


def _options(page, *ids) -> dict:
    """What each select offers, by id -- the values, blank included."""
    return page.evaluate(
        "(ids) => Object.fromEntries(ids.map(id => [id, Array.from("
        "  document.getElementById(id).options).map(o => o.value)]))",
        list(ids))


def _blocks(page) -> dict:
    """Each engine's block of the card, by its heading."""
    return page.evaluate(
        "() => Object.fromEntries(Array.from(document.querySelectorAll("
        "  '#chemistry-state .chemistry-engine')).map(b => ["
        "    b.querySelector('h3').textContent, b.textContent]))")


# --------------------------------------------------------------------- #
#  The Build tab                                                         #
# --------------------------------------------------------------------- #

def test_the_card_answers_for_each_form_and_fills_nothing(
        page, flask_server, tmp_path, monkeypatch):
    """Fe with every field blank: both forms' answers are unrestricted, 2S =
    2, detected from an open-d metal and saying to verify it; the charge
    fields stay blank; each form's chip carries its own answer."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    (tmp_path / "p").mkdir()
    target = tmp_path / "p" / "fe.xyz"
    target.write_text("1\nFe atom\nFe 0 0 0\n")
    _load(page, flask_server, "structure-optimization", target)
    _answered(page, "PySCF")
    card = _card(page)
    assert "SIESTA" in card and "PySCF" in card, card
    assert card.count("unrestricted, 2S = 2") == 2, card
    assert "detected: Fe is an open-d metal" in card, card
    assert "verify it against experiment" in card, card
    # Each form's own block.  SIESTA's method is a rule -- it is a
    # density-functional code -- so it is no choice anybody made and has no
    # line; PySCF's is the form's.
    blocks = _blocks(page)
    assert "Method" not in blocks["SIESTA"], blocks["SIESTA"]
    assert "DFT — stated" in blocks["PySCF"], blocks["PySCF"]
    # NOTHING FILLED IN: a blank stays blank -- the instruction, answered.
    assert page.evaluate(
        "() => [document.getElementById('p-net-charge').value,"
        "       document.getElementById('py-net-charge').value,"
        "       document.getElementById('py-unpaired-electrons').value]"
    ) == ["", "", ""]
    chips = page.evaluate(
        "() => Array.from(document.querySelectorAll("
        "  '.workflow-group--profile .workflow-detection-chip'))"
        "  .map(c => c.textContent)")
    assert chips and all("unrestricted, 2S = 2" in c for c in chips), chips
    # ES4 / ES6, the form half: each form offers only what its engine runs
    # for this kind -- SIESTA has no restricted-open, PySCF is collinear and
    # cannot float a moment.
    offered = _options(page, "p-spin-treatment", "p-unpaired-electrons",
                       "py-spin-treatment", "py-unpaired-electrons")
    assert "restricted-open" not in offered["p-spin-treatment"]
    assert "free" in offered["p-unpaired-electrons"]
    assert not {"non-collinear", "spin-orbit"} & set(
        offered["py-spin-treatment"]), offered["py-spin-treatment"]
    assert "restricted-open" in offered["py-spin-treatment"]
    assert "free" not in offered["py-unpaired-electrons"]


def test_a_typed_charge_changes_the_answer(page, flask_server, tmp_path,
                                           monkeypatch):
    """Formate, blank: 23 electrons at the detected charge 0, so unrestricted.
    Typed -1 on the PySCF form: that form's answer turns closed-shell at 24
    electrons, stated -- and the SIESTA form, still blank, does not move."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    (tmp_path / "p").mkdir()
    target = tmp_path / "p" / "formate.xyz"
    target.write_text("4\nformate\nC 0 0 0\nO 1.26 0 0\nO -0.63 1.09 0\n"
                      "H -0.55 -0.95 0\n")
    _load(page, flask_server, "structure-optimization", target)
    _answered(page, "PySCF")
    assert _card(page).count("unrestricted, 2S = 1") == 2, _card(page)

    page.locator('.tab-btn[data-tab="pyscf"]').click()
    field = page.locator("#py-net-charge")
    field.fill("-1")
    field.press("Tab")                       # the change the card listens to
    _answered(page, "-1")
    card = _card(page)
    assert "-1 — stated" in card, card
    assert "restricted (closed shell, 2S = 0)" in card, card
    assert "unrestricted, 2S = 1" in card, "the blank SIESTA form keeps its own"


# --------------------------------------------------------------------- #
#  The Spectrum tab                                                      #
# --------------------------------------------------------------------- #

def test_the_spectrum_tab_answers_for_its_vibration_forms(
        page, flask_server, tmp_path, monkeypatch):
    """The same card on the Spectrum tab, for the vibration kind: both
    engines' forms, answered as they stand."""
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    (tmp_path / "p").mkdir()
    target = tmp_path / "p" / "fe.xyz"
    target.write_text("1\nFe atom\nFe 0 0 0\n")
    _load(page, flask_server, "spectrum-calculation", target)
    _answered(page, "SIESTA")
    card = _card(page)
    assert "PySCF" in card and "SIESTA" in card, card
    assert "unrestricted, 2S = 2" in card, card
    # PySCF's vibration takes the analytic Hessian, which ROHF/ROKS lack
    # (ES4): its form does not offer restricted-open.
    assert "restricted-open" not in _options(
        page, "py-spin-treatment")["py-spin-treatment"]


def test_a_periodic_structure_moves_the_spectrum_strip_to_siesta(
        page, flask_server, tmp_path, monkeypatch):
    """`web/spectra.md` § 5: a structure that repeats along an axis switches
    the Spectrum tab's engine strip to SIESTA and says why, because PySCF's
    gate refuses a periodic structure.  Driven the way a person drives it:
    the pair picked in the sidebar, the Load button, the strip read back."""
    import numpy as np

    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    proj = tmp_path / "strip_proj"
    proj.mkdir()
    target = proj / "chain.xyz"
    StructureCodec().write(
        Structure(elements=["C", "C"],
                  positions=np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 1.3]]),
                  cell=np.diag([10.0, 10.0, 2.6]),
                  axis_kind=("isolated", "isolated", "periodic")),
        target)

    page.goto(f"{flask_server}/spectrum-calculation",
              wait_until="domcontentloaded")
    _ready_for_a_pick(page)
    page.wait_for_selector(
        "#spectra-form-container input, #spectra-form-container select",
        timeout=15000)
    assert page.evaluate(
        "() => document.querySelector('#spectra-engine-strip .tab-btn.active').dataset.tab"
    ) == "pyscf"
    page.evaluate(
        "(p) => window.molbuilder.projects.setShared("
        "  p.substring(0, p.lastIndexOf('/')), p)", str(target))
    page.wait_for_function(
        "() => !document.getElementById('load-from-sidebar-btn').disabled",
        timeout=5000)
    page.locator("#load-from-sidebar-btn").click()
    page.wait_for_function(
        "() => document.querySelector('#spectra-engine-strip .tab-btn.active')"
        "        .dataset.tab === 'siesta'",
        timeout=15000)
    assert not page.evaluate(
        "() => document.getElementById('spectra-engine-note').hidden")
    assert page.evaluate(
        "() => document.getElementById('spectra-tab-siesta').hidden") is False
    assert page.evaluate(
        "() => document.getElementById('spectra-tab-pyscf').hidden") is True
    # ...and the card is about THAT structure, sidecar and all: the pair's
    # periodicity reached the class through the envelope the viewer holds.
    _answered(page, "a repeating cell")
    assert "a repeating cell" in _blocks(page)["SIESTA"]


# --------------------------------------------------------------------- #
#  The Transport tab                                                     #
# --------------------------------------------------------------------- #

def test_the_transport_card_answers_for_the_cited_junction(
        page, flask_server, isolated_projects_root):
    """The Transport tab's card: the junction the tab cited, the SHARED
    panel's spin, the charge 0 by rule.  The citation records no spin, so the
    panel's is blank and the junction decides it -- metallic gold in a
    repeating cell, a closed shell.  Stating unrestricted on the panel moves
    the card, and the count floats: a repeating cell has no count to pin.

    Cited the way a returning person's tab cites: the tab's session note,
    restored through the same describe-and-adopt path a pick takes
    (`lib/transport/core.js` `_restoreSession`) -- the tree picker's pop-out
    is not what this pins."""
    import numpy as np

    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    junction = Structure(
        elements=["Au", "Au", "S", "Au", "Au"],
        positions=np.array([[0., 0, z] for z in (0, 2.4, 5.0, 7.6, 10.0)]),
        cell=np.diag([12., 12., 14.4]),
        axis_kind=("periodic", "periodic", "transport"),
        regions={"L-electrode": [0, 1], "bridge": [2],
                 "R-electrode": [3, 4]})
    folder = isolated_projects_root / "J" / "structure" / "cite"
    folder.mkdir(parents=True)
    StructureCodec().write(junction, folder / "junction.xyz")

    page.goto(f"{flask_server}/transport-calculation",
              wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.workspace "
        "      && typeof window.molbuilder.workspace.persist === 'function'",
        timeout=10000)
    assert page.evaluate(
        """async (j) => {
             const ws = window.molbuilder.workspace;
             const id = {workspace_id: ws.workspaceId('transport:panel'),
                         state_index: 0};
             ws.persist('transport:panel', {v: 3, junction: j}, id);
             for (let i = 0; i < 50; i++) {
               const n = await ws.readState(id);
               if (n && n.junction === j) return true;
               await new Promise(r => setTimeout(r, 100));
             }
             return false;
           }""", "J/structure/cite"), "the session note never landed"
    page.reload(wait_until="domcontentloaded")
    _answered(page, "SIESTA")
    blocks = _blocks(page)
    assert list(blocks) == ["SIESTA"], "transport runs on SIESTA alone"
    card = blocks["SIESTA"]
    assert "0 — rule: a transport junction is neutral" in card, card
    assert "restricted (closed shell, 2S = 0)" in card, card
    assert "detected: metallic Au in a repeating cell" in card, card

    page.select_option("#t-spin-treatment", "unrestricted")
    _answered(page, "the moment floats")
    card = _blocks(page)["SIESTA"]
    assert "unrestricted, the moment floats (free) — stated" in card, card

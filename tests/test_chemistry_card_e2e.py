"""The chemistry card, in the browser: the charge and spin a calculation will
carry, and why (``docs/science/chemistry-correctness.md`` §§ 2.5, 2a.5).

PINS the forms' half of the electronic state (``chemistry-correctness.md``
§ 2a.5): loading a structure shows, for each engine form, the state the
one class resolves for exactly what that form says -- each value with its
source -- and **a blank field stays blank**: nothing is filled in.  Typing a
charge changes the card.  The chip on each form's Profile card says the same.

Driven the way a person drives it: the file picked in the sidebar by its
rows (`support.sidebar`), the Load button, a field typed into -- in a
projects tree of the test's own, the one door pointed at it.  Headless chromium through pytest-playwright; no
engine runs.  The Transport tab's card answers for a cited junction, which
only a real relaxation makes: it is asked on the real junction's road
(`test_transport_on_a_real_junction_e2e.py`).
"""
from __future__ import annotations

import pytest

from support.chemistry_card import card_answered, card_blocks

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server(isolated_projects_root_module):
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _load(page, base, tab, target):
    """Open ``tab``, pick ``target`` in the sidebar by its rows, press Load
    -- and wait for the card to have answered."""
    from support.sidebar import picked
    picked(page, base, tab, target)
    page.locator("#load-from-sidebar-btn").click()
    card_answered(page)


def _card(page) -> str:
    return page.evaluate(
        "() => document.getElementById('chemistry-state').textContent")


def _options(page, *ids) -> dict:
    """What each select offers, by id -- the values, blank included."""
    return page.evaluate(
        "(ids) => Object.fromEntries(ids.map(id => [id, Array.from("
        "  document.getElementById(id).options).map(o => o.value)]))",
        list(ids))


# --------------------------------------------------------------------- #
#  The Build tab                                                         #
# --------------------------------------------------------------------- #

def test_the_card_answers_for_each_form_and_fills_nothing(
        page, flask_server, isolated_projects_root):
    """Fe with every field blank: both forms' answers are unrestricted, 2S =
    2, detected from an open-d metal and saying to verify it (§ 2.5); the
    PySCF form's Method, left blank, is the catalogue's DFT said *not
    chosen* (§ 2a.1a: *"a method the person left blank is the catalogue's
    and the card says so ... never stated"*; ``web/form-schema.md`` § 1.1,
    `template.SOURCE_WORDS`); the charge fields stay blank; each
    form's chip carries its own answer.

    THE SILENT FAILURE on the Method line: the card presenting the
    catalogue's value as the person's choice -- *stated* where nobody
    stated it."""
    (isolated_projects_root / "p").mkdir()
    target = isolated_projects_root / "p" / "fe.xyz"
    target.write_text("1\nFe atom\nFe 0 0 0\n")
    _load(page, flask_server, "structure-optimization", target)
    card_answered(page, "PySCF")
    card = _card(page)
    assert "SIESTA" in card and "PySCF" in card, card
    assert card.count("unrestricted, 2S = 2") == 2, card
    assert "detected: Fe is an open-d metal" in card, card
    assert "verify it against experiment" in card, card
    # Each form's own block.  SIESTA's method is a rule -- it is a
    # density-functional code -- so it is no choice anybody made and has no
    # line; PySCF's is the form's, blank here, so the catalogue's default.
    blocks = card_blocks(page)
    assert "Method" not in blocks["SIESTA"], blocks["SIESTA"]
    assert "DFT — " in blocks["PySCF"], blocks["PySCF"]
    assert "not chosen" in blocks["PySCF"], blocks["PySCF"]
    assert "stated" not in blocks["PySCF"].split("Method", 1)[1], \
        blocks["PySCF"]
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


def test_a_typed_charge_changes_the_answer(page, flask_server,
                                           isolated_projects_root):
    """Formate, blank: 23 electrons at the detected charge 0, so unrestricted.
    Typed -1 on the PySCF form: that form's answer turns closed-shell at 24
    electrons, stated -- and the SIESTA form, still blank, does not move."""
    (isolated_projects_root / "p").mkdir()
    target = isolated_projects_root / "p" / "formate.xyz"
    target.write_text("4\nformate\nC 0 0 0\nO 1.26 0 0\nO -0.63 1.09 0\n"
                      "H -0.55 -0.95 0\n")
    _load(page, flask_server, "structure-optimization", target)
    card_answered(page, "PySCF")
    assert _card(page).count("unrestricted, 2S = 1") == 2, _card(page)

    from support.sidebar import committed
    page.locator('.tab-btn[data-tab="pyscf"]').click()
    # The change the card listens to: the field left by Tab.
    committed(page, "#py-net-charge", "-1")
    card_answered(page, "-1")
    card = _card(page)
    assert "-1 — stated" in card, card
    assert "restricted (closed shell, 2S = 0)" in card, card
    assert "unrestricted, 2S = 1" in card, "the blank SIESTA form keeps its own"


# --------------------------------------------------------------------- #
#  The Spectrum tab                                                      #
# --------------------------------------------------------------------- #

def test_the_spectrum_tab_answers_for_its_vibration_forms(
        page, flask_server, isolated_projects_root):
    """The same card on the Spectrum tab, for the vibration kind: both
    engines' forms, answered as they stand."""
    (isolated_projects_root / "p").mkdir()
    target = isolated_projects_root / "p" / "fe.xyz"
    target.write_text("1\nFe atom\nFe 0 0 0\n")
    _load(page, flask_server, "spectrum-calculation", target)
    card_answered(page, "SIESTA")
    card = _card(page)
    assert "PySCF" in card and "SIESTA" in card, card
    assert "unrestricted, 2S = 2" in card, card
    # The spin is drawn once, for both engines, offering every choice either
    # takes for a vibration (`engines/template.md` § 6.2a): restricted-open
    # is neither's -- PySCF's vibration takes the analytic Hessian, which
    # ROHF/ROKS lack (ES4), and SIESTA has no such formalism.
    assert "restricted-open" not in _options(
        page, "common-spin-treatment")["common-spin-treatment"]


def test_a_periodic_structure_moves_the_spectrum_strip_to_siesta(
        page, flask_server, isolated_projects_root):
    """`web/spectra.md` § 5 step 4: a structure that repeats along an axis
    switches the Spectrum tab's engine strip to SIESTA and says why, because
    PySCF computes a repeating structure as an isolated cluster -- a note,
    not a refusal.  Driven the way a person drives it: the pair picked in
    the sidebar, the Load button, the strip read back."""
    import numpy as np

    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    from support.sidebar import picked

    proj = isolated_projects_root / "strip_proj"
    proj.mkdir()
    target = proj / "chain.xyz"
    StructureCodec().write(
        Structure(elements=["C", "C"],
                  positions=np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 1.3]]),
                  cell=np.diag([10.0, 10.0, 2.6]),
                  axis_kind=("isolated", "isolated", "periodic")),
        target)

    picked(page, flask_server, "spectrum-calculation", target)
    page.wait_for_selector(
        "#spectra-form-container input, #spectra-form-container select",
        timeout=15000)
    # PICKED, NOT YET LOADED: the strip still on its default.
    assert page.evaluate(
        "() => document.querySelector('#spectra-engine-strip .tab-btn[aria-selected=\"true\"]').dataset.tab"
    ) == "pyscf"
    page.locator("#load-from-sidebar-btn").click()
    page.wait_for_function(
        "() => document.querySelector('#spectra-engine-strip .tab-btn[aria-selected=\"true\"]')"
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
    card_answered(page, "a repeating cell")
    assert "a repeating cell" in card_blocks(page)["SIESTA"]


"""**A SIESTA vibration, calculated and then looked at**: the Results tab shows
the result as its modes, not a spectrum.

*(User, 2026-09-28: "there should be no width or spectrum for siesta result.
it is just mode of frequency, and animation of the mode ... the options and
presentation just need to be clear about that.")*

The road first -- ``jobset init --engine siesta --calculation vibration``,
``prep``, ``launch --mode direct``, the box ticked so ``freq`` runs alone at
the relaxed bond of H2 with one atom held (`test_siesta_vibration_e2e.py`
walks every state of that road; this file walks it once, to have a result).
Then the result is opened the way a person opens it, through the Results
tab's dropdown, and the page is read against `web/spectra.md`:

* **mode positions, not a spectrum**: one line of one height at each mode,
  no curve, no height numbers, no width control, under the heading *Mode
  positions* and one sentence saying the route computes no intensities
  (§ 2; `spectrumchart.md` § 6.2 -- user, 2026-09-28: "just bar/line to show
  where the mode is");
* the modes table without the columns the route cannot compute -- infrared,
  Raman, the per-mode orbitals -- and the run summary saying *not computed
  on this route* for each (§ 9b.3);
* no phase dot for Raman or the probe (§ 3);
* the electronic-structure tab naming the planned projected density of
  states, never PySCF's ``es_mode_selection`` (§ 9b.3);
* the thermochemistry labelled as vibrational contributions, with no
  pressure, and its bars the file's own numbers -- summing to ``F_vib``
  (§ 3; `engines/vibration.md` § 4.7).

Every hiding is read from the COMPUTED style, not the attribute or the
class: a ``display`` rule of the element's own beats ``[hidden]`` in the
author cascade, which is how the dots stayed visible under ``hidden`` until
``.phase[hidden]`` was added.

MUTATIONS THIS MUST FAIL AGAINST: the width control shown whatever the
strengths (``drawn`` forced true); the ``.phase[hidden]`` guard
removed (the Raman dot stays on screen); the electronic-structure notice
without its route branch (a SIESTA reader is then told about a selection
their run could not make); the full-RRHO labels forced on every result.

Needs the ``molbuilder-siesta`` env, a conda hook and Playwright; one
launch of seven H2 single points, about 20 s here.
"""
from __future__ import annotations

import json

import pytest

from _road import conda_hook, env_available
from test_siesta_vibration_e2e import (_describe, _jobset,
                                       _the_result, _tick_already_relaxed)

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

pytestmark = [
    pytest.mark.e2e, pytest.mark.slow, pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

#: named: a hidden Plotly node resized on load logs this, and nothing else
#: may.
_PLOTLY_HIDDEN_RESIZE = "Resize must be passed a displayed plot div"


def _shown(page, selector):
    """Whether ``selector`` takes up space on the page -- its computed
    display, up its ancestors, never its attribute."""
    return page.evaluate(
        "(sel) => { const e = document.querySelector(sel);"
        "  return !!e && e.getClientRects().length > 0; }", selector)


def test_the_results_tab_shows_a_siesta_vibration_as_its_modes(
        page, tmp_path, monkeypatch):
    from support.live_server import serve

    tree = tmp_path / "projects"
    bundle = _describe(tree, monkeypatch,
                       [[5.0, 5.0, 5.77446], [5.0, 5.0, 5.0]])
    task = json.loads((bundle / "task.json").read_text())
    task["stages"] = [s for s in task["stages"] if s["name"] == "freq"]
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    _tick_already_relaxed(bundle)
    r = _jobset("prep", "run", "freq", "--bundle", str(bundle),
                "--target", "this")
    assert r.exit_code == 0, r.output
    r = _jobset("launch", "run", "freq", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output
    attempt = bundle / "01_freq" / "run-0"
    d = _the_result(attempt, "01_freq")
    result = attempt / "H2.spectra.json"

    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))
    with serve() as base:
        page.add_init_script(
            "try { sessionStorage.setItem('molbuilder.current_dir', "
            f"{json.dumps(str(attempt))}); }} catch (_) {{}}")
        page.goto(f"{base}/results")
        page.wait_for_function(
            "(want) => [...document.querySelectorAll("
            "  '#results-file-picker-select option')]"
            "  .some(o => o.value === want)", arg=str(result), timeout=20000)
        page.select_option("#results-file-picker-select", value=str(result))
        # the rows are drawn by the same call that decides the spectrum
        # section (core.js renderResults), so once they are on the page
        # that decision is too
        page.wait_for_selector("#modes-tbody tr", state="attached",
                               timeout=30000)

        # MODE POSITIONS, NOT A SPECTRUM (§ 2): one line of one height at
        # each mode, no curve and no height numbers, no width control, and
        # the one sentence saying why there are no heights.
        assert _shown(page, "#spectrum-chart")
        assert page.inner_text("#spectrum-heading") == "Mode positions"
        assert not _shown(page, "#broadening-fwhm")
        assert not _shown(page, "#display-floor")
        absent = page.inner_text("#spectrum-absent")
        assert "not infrared or Raman intensities" in absent, absent
        assert page.locator("#modes-tbody tr").count() == len(d["modes"]) == 1
        drawn = page.wait_for_function(
            "() => { const g = document.querySelector("
            "  '#spectrum-chart .js-plotly-plot');"
            "  if (!g || !g.data) return null;"
            "  return {traces: g.data.map(t => ({type: t.type,"
            "    mode: t.mode || '', x: t.x, y: t.y})),"
            "    ticks: g.layout.yaxis.showticklabels}; }",
            timeout=20000).json_value()
        bars = [t for t in drawn["traces"] if t["type"] == "bar"]
        assert len(bars) == 1, drawn
        freqs = [m["frequency_cm1"] for m in d["modes"]]
        assert bars[0]["x"] == pytest.approx(freqs)
        assert len(set(bars[0]["y"])) == 1, bars[0]["y"]
        assert not [t for t in drawn["traces"]
                    if t["type"] == "scatter" and "lines" in t["mode"]], drawn
        assert drawn["ticks"] is False, drawn

        # THE TABLE AND THE SUMMARY BY ROUTE (§ 9b.3): what SIESTA cannot
        # compute has no column, and the summary says so for each.
        for col in ("raman_activity_a4_amu", "ir_intensity_km_mol",
                    "has_es", "gap_eq_ev"):
            assert not _shown(page, f'#modes-table th[data-col="{col}"]'), col
        assert _shown(page, '#modes-table th[data-col="frequency_cm1"]')
        summary = page.evaluate("""() => {
            const dl = document.getElementById("results-summary-list");
            const out = {};
            dl.querySelectorAll("dt").forEach(dt => {
                out[dt.textContent] = dt.nextElementSibling.textContent; });
            return out; }""")
        for line in ("IR intensities", "Raman activities",
                     "Per-mode orbital energies"):
            assert summary[line].startswith("not computed on this route"), \
                (line, summary[line])

        # NO DOT FOR A PHASE THE ROUTE DOES NOT HAVE (§ 3).
        assert _shown(page, '.phase-dot[data-phase="frequencies"]')
        assert not _shown(page, '.phase-dot[data-phase="raman"]')
        assert not _shown(page, '.phase-dot[data-phase="es"]')
        # the Infrared dot exists in the partial and, SIESTA computing no
        # intensities, is not shown (web/spectra.md § 3)
        assert page.locator('.phase-dot[data-phase="ir"]').count() == 1
        assert not _shown(page, '.phase-dot[data-phase="ir"]')

        # THE ELECTRONS' RESPONSE: planned, and named as what it will be.
        page.click("#mode-tabbtn-es")
        es = page.inner_text("#es-bar-diagram")
        assert "projected density of states" in es, es
        assert "es_mode_selection" not in es and "Mode selection" not in es, es

        # THE THERMOCHEMISTRY BY ITS REGIME (§ 3): vibrational
        # contributions, and no pressure -- none entered them.
        assert d["thermo"]["regime"] == "vibrational-only"
        page.click("#mode-tabbtn-thermo")
        note = page.inner_text("#thermo-note")
        assert "vibrational contributions only" in note, note
        assert "F_vib" in note and "S_vib" in note, note
        assert "P =" not in note and "atm" not in note, note
        # ...and its bars are the file's: SIESTA carries no electronic
        # energy, so the reference is zero, the parts add up to the last
        # bar and the last bar is the file's own F_vib (§ 3).
        from molbuilder import constants as C
        kcal = C.HARTREE_EV / C.KCAL_MOL_EV
        bars = page.wait_for_function(
            "() => { const d = document.getElementById('thermo-decomp');"
            "  return d && d.data && d.data[0]"
            "  ? {x: d.data[0].x, y: d.data[0].y} : null; }",
            timeout=20000).json_value()
        assert bars["x"] == ["ZPE", "U_vib", "−T·S_vib", "F_vib"], bars["x"]
        y = bars["y"]
        assert d["equilibrium"]["scf_energy_eh"] is None
        assert sum(y[:-1]) == pytest.approx(y[-1], abs=1e-9), y
        assert y[-1] == pytest.approx(d["thermo"]["g_eh"] * kcal, abs=1e-6)
        assert y[0] == pytest.approx(d["thermo"]["zpe_eh"] * kcal, abs=1e-6)

    unexpected = [e for e in errors if _PLOTLY_HIDDEN_RESIZE not in e]
    assert unexpected == [], f"the page reported JS errors: {unexpected}"

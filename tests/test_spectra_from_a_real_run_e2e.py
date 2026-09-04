"""**Calculate CO2, then look at it.** The spectra chain, with nothing faked.

*(User ruling, 2026-09-03: "e2e means you can fucking calculate a CO2 molecule
within 5 min and use that output to do this. why the fuck do you need anything
copied? that would be true e2e")*

Every other spectra test in this suite points at a `job.spectra.json` that does
not exist, takes the 404, and stops — so the viewer's load path had **no
coverage at all**, and the two doors it was rebuilt around on 2026-09-03 could
not be exercised.  The gap was not a missing fixture.  A fixture is a copy of
somebody's old answer; the question is whether THIS code, today, can produce a
result and then read its own result back.

So this walks it:

  1. build a CO2 molecule — three atoms, C=O at 1.16 Å;
  2. render the vibration deck through the production door
     (`pyscf.input.spec_for` → `script_emit.prepare_deck`, the same pair the
     Build tab calls);
  3. run it in the env molbuilder itself routes PySCF to
     (`Capabilities.env_for_category("pyscf")`, the four-env model's own
     routing, and dispatched with `conda run -n <name>` as molbuilder does);
  4. load the `.spectra.json` that came out into the real page, through the
     page's own "Load once" button;
  5. read the modes off the screen.

**2.3 seconds of compute.**  RHF/STO-3G on three atoms — the same level of
theory `test_pyscf_smoke.py` uses as its reference for the same reason: it is
the most exhaustively documented case in the literature, so a wrong answer is
our bug and not numerical weather.

**Why CO2 and not something smaller.**  A diatomic has one mode and would pass
on almost any wiring.  CO2 is the smallest molecule whose spectrum has a shape
worth checking: 3N−5 = 4 modes for a linear triatomic, of which the two bends
are DEGENERATE.  A pipeline that drops a mode, or that loses the degeneracy by
mangling coordinates, fails here and passes on water.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]

#: Experiment, for the record: the bend is 667 cm-1 (doubly degenerate), the
#: symmetric stretch 1333, the antisymmetric 2349.  RHF/STO-3G reproduces the
#: SHAPE and not the numbers -- it has no correlation and a minimal basis, so
#: it overshoots the stretches and undershoots the bend.  The assertions below
#: are therefore about ordering and degeneracy, which is what this level of
#: theory is entitled to get right, with a wide band on the values so the test
#: reports a broken pipeline and not a change of basis.
_CO2_MODES = 4


def _pyscf_env():
    """The env molbuilder itself routes PySCF to, if it exists here.

    `env_for_category`, not `routed_env`: PySCF is a CATEGORY in the
    four-env model (`diagnostics.DEFAULT_ENV_NAMES`), not an executable.
    `TOOL_TO_CATEGORY` maps binaries -- `siesta`, `mpirun` -- and answers
    None for a category name, which is a quiet skip rather than an error.

    And `detect()`, not `Capabilities()`: the dataclass defaults to an
    empty env set, so a bare constructor reports that nothing exists.
    """
    from molbuilder.diagnostics import detect
    try:
        caps = detect()
        env = caps.env_for_category("pyscf")
        return env if env and caps.env_available(env) else None
    except Exception:
        return None


def _run_one(env, d, label, bond):
    """Compute one CO2 frequency job at the given C=O distance.

    Two runs at different bond lengths give two results whose numbers are
    unmistakably different -- which is what a file-switch test needs, and
    what makes "the view still shows the previous file" impossible to miss.
    """
    import numpy as np

    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.input import spec_for
    from molbuilder.script_emit import prepare_deck
    from molbuilder.structure import Structure

    d.mkdir(parents=True, exist_ok=True)
    struct = Structure(
        elements=["C", "O", "O"],
        positions=np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, bond],
                            [0.0, 0.0, -bond]]))
    cfg = PySCFConfig(job_name=label, method="RHF", basis="STO-3G",
                      compute_ir=True, compute_raman=False,
                      optimize=False, already_relaxed=True)
    deck = d / f"{label}_vib.py"
    prepare_deck(spec_for(struct, cfg, calculation="vibration"),
                 struct, cfg, deck, verbose=False)
    proc = subprocess.run(["conda", "run", "-n", env, "python", deck.name],
                          cwd=str(d), capture_output=True, text=True,
                          timeout=600)
    out = d / f"{label}.spectra.json"
    assert out.exists(), (
        f"the deck ran (exit {proc.returncode}) but wrote no {out.name}.\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}")
    return out


@pytest.fixture(scope="module")
def co2_pair():
    """TWO real runs, at 1.16 A and 1.24 A.

    The second exists for the file-switch test: a stretched CO2 has
    visibly softer stretches, so "the numbers on screen belong to the file
    you picked" has an answer you can read rather than infer.
    """
    env = _pyscf_env()
    if env is None:
        pytest.skip("no conda env routes PySCF on this machine")
    root = ROOT / "projects/_t_co2_pair_e2e"
    if root.exists():
        shutil.rmtree(root)
    try:
        # ONE folder, two results.  The picker lists the result files in
        # the CURRENT folder (`results.md` § 2.1), so two runs in two
        # folders cannot be switched between without also moving the
        # sidebar -- and moving the sidebar re-scopes and auto-picks, which
        # is a different gesture from "pick the other result".
        d = root / "spectrum" / "co2pair"
        a = _run_one(env, d, "co2a", 1.16)
        b = _run_one(env, d, "co2b", 1.24)
        yield a, b
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def co2_run():
    """Run CO2 and yield the `.spectra.json` it produced.

    Under the projects root, because `/api/spectra/load` resolves through
    `_resolve_within_roots` and refuses anything outside the picker roots --
    which is also simply where a calculation lives.
    """
    env = _pyscf_env()
    if env is None:
        pytest.skip("no conda env routes PySCF on this machine")

    import numpy as np

    from molbuilder.config.pyscf import PySCFConfig
    from molbuilder.pyscf.input import spec_for
    from molbuilder.script_emit import prepare_deck
    from molbuilder.structure import Structure

    root = ROOT / "projects/_t_co2_e2e"
    if root.exists():
        shutil.rmtree(root)
    d = root / "spectrum" / "co2"
    d.mkdir(parents=True)
    try:
        struct = Structure(
            elements=["C", "O", "O"],
            positions=np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.16],
                                [0.0, 0.0, -1.16]]))
        cfg = PySCFConfig(job_name="co2", method="RHF", basis="STO-3G",
                          compute_ir=True, compute_raman=False,
                          optimize=False, already_relaxed=True)
        deck = d / "co2_vib.py"
        # THE PRODUCTION DOOR, not a hand-assembled script: prepare_deck
        # runs validate -> render -> write -> check, so a deck this fixture
        # produces is one the Build tab would have produced.
        prepare_deck(spec_for(struct, cfg, calculation="vibration"),
                     struct, cfg, deck, verbose=False)

        # ...and molbuilder's own dispatch to run it.
        proc = subprocess.run(
            ["conda", "run", "-n", env, "python", deck.name],
            cwd=str(d), capture_output=True, text=True, timeout=600)
        results = d / "co2.spectra.json"
        assert results.exists(), (
            f"the deck ran (exit {proc.returncode}) but wrote no "
            f"co2.spectra.json.\n--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}")
        yield results
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_the_run_produces_the_spectrum_co2_actually_has(co2_run):
    """Four modes, the two bends degenerate, in the right order.

    This is the half that would catch a pipeline producing confident
    nonsense -- a coordinate mix-up, a mass-weighting error, a dropped
    mode.  None of those raise; they just give you a wrong spectrum, and
    only the physics says so.
    """
    doc = json.loads(co2_run.read_text(encoding="utf-8"))
    modes = doc["modes"]
    assert len(modes) == _CO2_MODES, (
        f"a linear triatomic has 3N-5 = 4 vibrational modes; this run "
        f"reported {len(modes)}.  Too few means modes were dropped with the "
        f"translations/rotations; too many means some were kept.")

    freqs = sorted(m["frequency_cm1"] for m in modes)
    assert not any(m["has_imag"] for m in modes), (
        f"an imaginary frequency means the geometry is not a minimum; CO2 at "
        f"1.16 A is.  Frequencies: {freqs}")

    bend_a, bend_b, sym, asym = freqs
    assert bend_a == pytest.approx(bend_b, rel=1e-4), (
        f"CO2's two bending modes are degenerate by symmetry -- they bend in "
        f"perpendicular planes of a linear molecule.  Got {bend_a} and "
        f"{bend_b}, which means the two directions are no longer equivalent: "
        f"suspect the coordinates or the mass weighting.")
    assert bend_a < sym < asym, (
        f"the order is bend < symmetric stretch < antisymmetric stretch for "
        f"CO2 at every level of theory.  Got {freqs}.")
    # A wide band: RHF/STO-3G is entitled to be wrong about the values (it
    # has no correlation and a minimal basis) and is not entitled to be
    # wrong by an order of magnitude.
    assert 200 < bend_a < 900, f"bend far outside any plausible range: {bend_a}"
    assert 1000 < sym < 2200, f"symmetric stretch implausible: {sym}"
    assert 2000 < asym < 3600, f"antisymmetric stretch implausible: {asym}"


def test_the_viewer_reads_a_run_this_suite_just_computed(
        page, flask_server, co2_run):
    """The other end of the chain, in a browser.

    The spectra viewer is not the `/spectra` page -- that one GENERATES a
    calculation.  The viewer is `_spectra_inspector.html`, mounted on
    `/results` by the inspector registry, which dispatches on the filename:
    `*.spectra.json` picks the spectra adapter, and mounting with a file
    loads it.

    So this is the real surface, reached the real way, on a file this suite
    computed sixty lines above.  Until 2026-09-03 nothing reached it: every
    spectra e2e mounted a `job.spectra.json` that did not exist, took the
    404 and stopped in ERROR -- which is why the load path could be rebuilt
    that day with only half of it provable.
    """
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))
    # SEED THE SELECTION BEFORE THE PAGE LOADS.
    #
    # `test_task_setup_cell_types_e2e.py` records why: the sidebar restores
    # its selection at init, so setting one afterwards is a race the sidebar
    # wins -- `restoreSelection` runs second and puts the tree back where it
    # was.  The app's own hand-off writes the target page's slots and lets
    # init pick them up, which is what this does.
    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir', {json.dumps(str(co2_run.parent))});"
        f" sessionStorage.setItem('molbuilder.current_file', {json.dumps(str(co2_run))});"
        "} catch (_) {}")
    page.goto(f"{flask_server}/results")
    page.wait_for_selector("#inspector-host", timeout=20000)
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.inspectors "
        "&& window.molbuilder.inspectors.list().length >= 4", timeout=20000)

    # ...and nudge the selection too, for the case where init got there
    # first.  setShared is what a sidebar click does.
    #
    # Calling `inspectors.mount(host, path, ctx)` directly does put the
    # viewer in the host -- and then the page's own dispatcher, which
    # watches the sidebar selection and sees nothing picked, replaces it
    # with the "Pick a file to inspect" fallback.  The host ends up holding
    # `#results-fallback` and every selector below looks broken, while the
    # server log cheerfully shows `POST /api/spectra/load 200`.
    #
    # `projects.setShared(dir, file)` is what a click in the sidebar does,
    # so the dispatcher mounts the inspector itself and then leaves it
    # alone.  It is also the honest walk: a person opens Results and picks
    # the file.
    page.evaluate(
        "([d, f]) => window.molbuilder.projects.setShared(d, f)",
        [str(co2_run.parent), str(co2_run)])

    # Wait on THE ROWS -- what a person is looking at.
    page.wait_for_selector("#modes-tbody tr", state="attached", timeout=30000)

    shown = page.evaluate("""() => {
        const host = document.getElementById("inspector-host");
        const sum  = host.querySelector("#results-summary");
        return {
            rows: host.querySelectorAll("#modes-tbody tr").length,
            first: (host.querySelector("#modes-tbody tr td:nth-child(2)")
                    || {}).textContent,
            summaryShown: !!(sum && !sum.hidden),
        };
    }""")

    # ONE exclusion, named rather than blanket.  `_watchEsWidth` installs a
    # ResizeObserver that calls `Plotly.Plots.resize(node)` whenever the
    # node's box changes -- including while it is hidden, which Plotly
    # refuses with "Resize must be passed a displayed plot div element."
    # The call is already wrapped in try/catch, so the author knew it could
    # fail and chose to swallow the throw; what the catch cannot swallow is
    # Plotly's own console.error.  Nothing breaks, so this is left as a
    # finding rather than fixed under a test: the fix is a visibility check
    # before the call, and whether to add one is a UI call, not a defect
    # against any written rule.
    _PLOTLY_HIDDEN_RESIZE = "Resize must be passed a displayed plot div"
    unexpected = [e for e in errors if _PLOTLY_HIDDEN_RESIZE not in e]
    assert unexpected == [], f"the page reported JS errors: {unexpected}"

    assert shown["summaryShown"], (
        "the results summary stayed hidden, so the data arrived and nothing "
        "drew it")
    assert shown["rows"] == _CO2_MODES, (
        f"the run produced {_CO2_MODES} modes and the table shows "
        f"{shown['rows']}.  The numbers reached the viewer; the table "
        f"disagrees with them.")


def test_switching_files_replaces_the_view_it_does_not_merge(
        page, flask_server, co2_pair):
    """**`results.md` § 4: the parsed file is replaced whole, never patched.**

    This is the contract the 23 regex tests in
    `test_results_state_contract_spectra_js.py` were reaching for — the four
    state buckets and the one `transition()` exist so that picking a
    different file cannot leave you looking at the previous one's numbers.
    Every one of those searched `spectra/core.js` for the *shape* that
    implements it; none could see whether it works.

    Two real runs, 1.16 Å and 1.24 Å.  A stretched CO2 has visibly softer
    stretches, so the numbers say which file is on screen.  Switch, and the
    old ones must be gone — not merged, not appended, not stale.
    """
    a, b = co2_pair
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))

    def _freqs(what):
        # Bounded, and the failure is SWALLOWED so the assertion below is
        # what a person reads.  Letting the wait raise reports "Timeout
        # 30000ms exceeded" and nothing about which file failed to appear.
        try:
            page.wait_for_selector("#modes-tbody tr", state="attached",
                                   timeout=30000)
        except Exception:                  # noqa: BLE001 - see above
            pass
        rows = page.evaluate(
            "() => [...document.querySelectorAll('#modes-tbody tr')]"
            "        .map(r => r.cells[1].textContent.trim())")
        assert rows, (
            f"no modes are on screen after picking {what}.  The dropdown "
            f"dictates the display (`results.md` § 2.1) -- if the selection "
            f"changed and the panel did not, the dispatcher is not "
            f"re-mounting on a new file and a person is looking at whichever "
            f"result they opened first.")
        return rows

    def _pick(path):
        """Choose a result the way the page says you choose one.

        `results.md` § 2.1: "The dropdown dictates the display... There is
        no second route to a mounted viewer."  Writing the selection slot
        and reloading does NOT work and is worth recording: the sidebar
        restores its own remembered selection during init and overwrites
        it, so the page came back showing the previous file with its slot
        rewritten to match -- which looks exactly like a stale view and is
        not one.
        """
        page.select_option("#results-file-picker-select", value=str(path))
        page.wait_for_function(
            "(want) => { const s = document.querySelector("
            "  '#results-file-picker-select'); return s && s.value === want; }",
            arg=str(path), timeout=10000)

    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir', {json.dumps(str(a.parent))});"
        "} catch (_) {}")
    page.goto(f"{flask_server}/results")
    page.wait_for_selector("#results-file-picker-select", timeout=20000)
    page.wait_for_function(
        "() => document.querySelectorAll("
        "  '#results-file-picker-select option').length >= 2", timeout=20000)

    _pick(a)
    freqs_a = _freqs("the 1.16 A run")
    _pick(b)
    try:
        page.wait_for_function(
            "(prev) => { const r = [...document.querySelectorAll('#modes-tbody tr')];"
            "  return r.length && r[0].cells[1].textContent.trim() !== prev; }",
            arg=freqs_a[0], timeout=30000)
    except Exception:                      # noqa: BLE001
        pass
    freqs_b = _freqs("the 1.24 A run")

    assert len(freqs_a) == _CO2_MODES and len(freqs_b) == _CO2_MODES, (
        f"expected {_CO2_MODES} modes from each run; got {len(freqs_a)} and "
        f"{len(freqs_b)}")
    # The anti-vacuity guard: if the two runs happened to agree, "the view
    # changed" could not be told from "the view is stale".
    assert freqs_a != freqs_b, (
        f"the two runs produced identical frequencies ({freqs_a}), so this "
        f"test cannot tell a replaced view from a stale one.  Stretching the "
        f"bond should soften the stretches -- check the fixture actually "
        f"ran two different geometries.")
    assert not (set(freqs_a) & set(freqs_b)), (
        f"after switching files the table still shows {sorted(set(freqs_a) & set(freqs_b))} "
        f"from the previous run.  `results.md` § 4 says the parsed file is "
        f"replaced WHOLE -- a row that survives a switch is the merge that "
        f"rule forbids, and a person is reading two calculations at once.")
    assert errors == [], f"the page reported JS errors: {errors}"

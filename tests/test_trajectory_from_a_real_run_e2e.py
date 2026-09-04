"""**Optimise CO2, then watch it relax.** The trajectory chain, nothing faked.

The sibling of `test_spectra_from_a_real_run_e2e.py`, and for the same reason
*(user, 2026-09-03: run the molecule, don't copy somebody's old answer)*.

The trajectory viewer had a fixture problem of its own: the multi-frame
`*_geom_optim.xyz` the timer test drives was **hand-written by me** on
2026-09-03 — frames from `Structure.to_xyz()` with a comment line I guessed at.
The guess happened to be close, which is worse than being wrong: a fixture
nobody checked against a real optimiser is a description of what I expected
geomeTRIC to emit, and the test built on it proves my expectation, not the
program.

So this runs one:

  1. CO2 **stretched to 1.30 Å** — the RHF/STO-3G minimum is near 1.19, so the
     optimiser has real work and writes a real multi-step trajectory.  Starting
     at the minimum would converge in one step and prove nothing about frames;
  2. the optimisation deck through the production door (`spec_for` +
     `prepare_deck`, the pair the Build tab calls) — `optimize`,
     `optimizer="geometric"` and `write_trajectory` are all ON by default, so
     this asks for nothing special;
  3. run in the env the four-env model routes PySCF to;
  4. open `co2opt_geom_optim.xyz` on `/results` and read the energy curve.

**4.4 seconds of compute**, and the answer is a real relaxation: five frames,
1.3000 → 1.1775 → 1.1925 → 1.1881 → 1.1879 Å, with the energy falling
monotonically.  The overshoot on the first step and the small corrections
after it are what a quasi-Newton optimiser does; a fixture I wrote by hand
would not have had them, because I would not have thought to put them there.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]

#: Where CO2 starts, and roughly where RHF/STO-3G puts it.  Experiment is
#: 1.162 Å; a minimal basis with no correlation lands near 1.19, which is the
#: kind of wrong this level of theory is entitled to be.
_START_ANG = 1.30
_EXPECTED_MIN = 1.19


def _pyscf_env():
    """The env molbuilder routes PySCF to, if it exists here.

    `env_for_category`, not `routed_env`: PySCF is a CATEGORY in the four-env
    model, and `TOOL_TO_CATEGORY` maps executables.  And `detect()`, not the
    bare `Capabilities()`, whose env set defaults to empty — that pair cost an
    hour of quiet skips on a machine where the env was right there.
    """
    from molbuilder.diagnostics import detect
    try:
        caps = detect()
        env = caps.env_for_category("pyscf")
        return env if env and caps.env_available(env) else None
    except Exception:
        return None


def _frames(xyz_text):
    """Parse a multi-frame XYZ into [(comment, positions)].

    Hand-rolled rather than routed through the project's reader on purpose:
    this is the test's own independent look at the bytes geomeTRIC wrote, so
    a reader bug cannot hide inside the thing being checked.
    """
    import numpy as np

    lines = xyz_text.strip().splitlines()
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i].strip())
        comment = lines[i + 1].strip()
        pos = [list(map(float, ln.split()[1:4]))
               for ln in lines[i + 2:i + 2 + n]]
        out.append((comment, np.array(pos)))
        i += 2 + n
    return out


@pytest.fixture(scope="module")
def co2_optimization():
    """Run a real geometry optimisation; yield the trajectory it wrote.

    Under the projects root, because `/api/watch/load` resolves through
    `_resolve_within_roots` and refuses anything outside the picker roots —
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

    root = ROOT / "projects/_t_co2opt_e2e"
    if root.exists():
        shutil.rmtree(root)
    d = root / "optimization" / "co2opt"
    d.mkdir(parents=True)
    try:
        struct = Structure(
            elements=["C", "O", "O"],
            positions=np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, _START_ANG],
                                [0.0, 0.0, -_START_ANG]]))
        cfg = PySCFConfig(job_name="co2opt", method="RHF", basis="STO-3G")
        deck = d / "co2opt.py"
        prepare_deck(spec_for(struct, cfg, calculation="optimization"),
                     struct, cfg, deck, verbose=False)
        proc = subprocess.run(
            ["conda", "run", "-n", env, "python", deck.name],
            cwd=str(d), capture_output=True, text=True, timeout=900)
        traj = d / "co2opt_geom_optim.xyz"
        assert traj.exists(), (
            f"the optimisation ran (exit {proc.returncode}) but wrote no "
            f"co2opt_geom_optim.xyz.\n--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}")
        yield traj
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def test_the_optimisation_actually_relaxes_the_molecule(co2_optimization):
    """It moved, it went downhill, and it stopped somewhere sensible.

    The failure this catches is a trajectory that *looks* fine — right file,
    right frame count, parseable — and is not a relaxation: an optimiser
    wired to the wrong gradient walks, and every frame after the first is
    then confident nonsense.  Only the energy and the geometry say so.
    """
    import numpy as np

    frames = _frames(co2_optimization.read_text(encoding="utf-8"))
    assert len(frames) >= 3, (
        f"a relaxation from {_START_ANG} A should take several steps; this "
        f"trajectory has {len(frames)} frame(s).  One frame means the "
        f"optimiser never ran, or the trajectory writer only kept the last.")

    bonds = [float(np.linalg.norm(p[1] - p[0])) for _, p in frames]
    assert bonds[0] == pytest.approx(_START_ANG, abs=1e-3), (
        f"the first frame should be the geometry we handed in "
        f"({_START_ANG} A); it is {bonds[0]:.4f}.  The trajectory does not "
        f"start where the input did.")
    assert bonds[-1] == pytest.approx(_EXPECTED_MIN, abs=0.05), (
        f"CO2 relaxed to {bonds[-1]:.4f} A at RHF/STO-3G; expected about "
        f"{_EXPECTED_MIN}.  Experiment is 1.162 -- a minimal basis with no "
        f"correlation lands a little long, and this band is wide enough for "
        f"that and narrow enough to catch a walk in the wrong direction.")
    assert abs(bonds[-1] - _START_ANG) > 0.05, (
        "the geometry barely moved, so this trajectory shows no relaxation "
        "and every frame assertion below it would pass on a stationary run")

    # Energy is the honest monotone: geometry can overshoot and come back
    # (this run does -- 1.1775 then 1.1925), a line search cannot go uphill.
    energies = []
    for comment, _ in frames:
        parts = comment.replace("=", " ").split()
        for j, tok in enumerate(parts):
            if tok.lower().startswith("energy") and j + 1 < len(parts):
                energies.append(float(parts[j + 1]))
                break
    assert len(energies) == len(frames), (
        f"parsed {len(energies)} energies from {len(frames)} frame comments; "
        f"geomeTRIC writes 'Iteration N Energy E' on each.  First comment: "
        f"{frames[0][0]!r}")
    assert energies[-1] < energies[0], (
        f"the optimisation ended higher than it started "
        f"({energies[0]} -> {energies[-1]}).  That is not a relaxation.")
    assert all(b <= a + 1e-6 for a, b in zip(energies, energies[1:])), (
        f"the energy went uphill between steps: {energies}.  A line search "
        f"that accepts an uphill step is taking a gradient it should not.")


def _log_steps(text):
    """How many steps the molwatch log records, read independently.

    The log's own marker, not a parse through the project's reader: this is
    the test's separate look at the bytes, so a reader bug cannot hide
    inside the thing being checked.
    """
    return len(re.findall(r"^==== molwatch step \d+ begin ====$",
                          text, re.M))


def test_a_run_is_one_entry_in_the_picker_and_it_is_the_log(
        page, flask_server, co2_optimization):
    """**A run is one result, not a pile of files** (`results.md` § 2.3).

    THE CONTRACT, from `lib/inspectors/trajectory.js`:

    * it CLAIMS four shapes -- `*.molwatch.log`, `*.out`, `*_optim.xyz`,
      `*_geom_optim.xyz` -- and is registered before the structure
      inspector so the `_optim.xyz` claim beats structure's generic `.xyz`;
    * it ABSORBS, and only when the master is a `.molwatch.log`, in the
      same folder: `<stem>_initial.xyz`, `<stem>_optimized.xyz` and
      `<stem>_geom_*_optim.xyz`.  A SIESTA `.out` absorbs nothing, on
      purpose, until that is checked against a real staged SIESTA run.

    So this run -- which writes an initial xyz, an optimised xyz, a
    geomeTRIC stream and the log -- must appear as ONE line in the picker,
    and that line must be the log.  Before absorption landed it was five
    (2026-08-04).

    Absorption narrows the MENU, not what can be opened: `/api/watch/load`
    returns whichever file you ask for, and the physics test above reads
    the `.xyz` directly for exactly that reason.
    """
    d = co2_optimization.parent
    log = d / "co2opt.molwatch.log"
    assert log.exists(), (
        f"the run wrote no molwatch log; the files present are "
        f"{sorted(p.name for p in d.iterdir())}")

    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir', {json.dumps(str(d))});"
        "} catch (_) {}")
    page.goto(f"{flask_server}/results")
    page.wait_for_selector("#results-file-picker-select", timeout=20000)
    page.wait_for_function(
        "() => [...document.querySelectorAll("
        "  '#results-file-picker-select option')]"
        "  .filter(o => o.value).length > 0", timeout=20000)

    offered = page.evaluate(
        "() => [...document.querySelectorAll("
        "  '#results-file-picker-select option')]"
        "  .map(o => o.value).filter(v => v)")

    on_disk = sorted(p.name for p in d.iterdir() if p.is_file())
    assert offered == [str(log)], (
        f"the picker offers {[p.split('/')[-1] for p in offered]} for one "
        f"relaxation.  The folder holds {on_disk}; `absorbs()` should fold "
        f"the initial xyz, the optimised xyz and the geomeTRIC stream into "
        f"the .molwatch.log master, leaving exactly one entry "
        f"(`results.md` § 2.3).")


def test_the_viewer_draws_the_run_this_suite_just_optimised(
        page, flask_server, co2_optimization):
    """The other end, in a browser: the energy curve is the run's own steps.

    **The file shown is the `.molwatch.log`**, because that is the one entry
    absorption leaves in the picker (see the test above).  So the curve is
    compared against the LOG's step count, not the `.xyz`'s -- they differ
    by one on purpose, and comparing against the wrong file is what made an
    earlier version of this test fail on correct code twice.

    The log records `step_index: 0, kind: initial_preview` -- the geometry
    as handed in, before any SCF -- so its first point carries no energy and
    plots as a null.  The energies are compared across the points that have
    one.

    **Picked through the dropdown**, because `results/viewer.js` retired
    sidebar-driven dispatch on 2026-06-09 (task #301): the page listens for
    `molbuilder:results:fileSelected`, which only
    `#results-file-picker-select` dispatches.
    """
    d = co2_optimization.parent
    log = d / "co2opt.molwatch.log"
    steps = _log_steps(log.read_text(encoding="utf-8", errors="replace"))
    assert steps >= 3, (
        f"the molwatch log records {steps} steps; a relaxation from 1.30 A "
        f"takes several, so the fixture or the log emitter changed")

    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))

    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir', {json.dumps(str(d))});"
        "} catch (_) {}")
    page.goto(f"{flask_server}/results")
    page.wait_for_selector("#results-file-picker-select", timeout=20000)
    page.wait_for_function(
        "(want) => [...document.querySelectorAll("
        "  '#results-file-picker-select option')].some(o => o.value === want)",
        arg=str(log), timeout=20000)
    page.select_option("#results-file-picker-select", value=str(log))

    # Read at the moment the condition holds: the inspector re-renders on
    # its poll and replaces the plot node, so waiting and then evaluating
    # separately raced -- `#energy-plot` came back null a tick after the
    # wait had just seen it holding data.
    drawn = page.wait_for_function(
        """() => {
            const el = document.querySelector("#energy-plot");
            const y = el && el.data && el.data[0] && el.data[0].y;
            if (!y || !y.length) return null;
            const e = y.filter(v => typeof v === "number");
            if (e.length < 2) return null;
            return {points: y.length, energies: e.length,
                    first: e[0], last: e[e.length - 1]};
        }""",
        timeout=30000, polling=250).json_value()

    assert drawn["points"] == steps, (
        f"the log records {steps} steps and the energy curve plots "
        f"{drawn['points']} points.  The viewer is showing this file "
        f"(absorption leaves no other entry), so these must agree.")
    assert drawn["energies"] == steps - 1, (
        f"{drawn['energies']} of {drawn['points']} points carry an energy. "
        f"Exactly one -- the `initial_preview` -- has none by nature; any "
        f"other missing energy is a step whose result did not reach the "
        f"plot.")
    assert drawn["last"] < drawn["first"], (
        f"the plotted curve rises ({drawn['first']} -> {drawn['last']}) "
        f"while the run falls.  The viewer is drawing the relaxation "
        f"backwards, which would tell a person their optimisation "
        f"diverged.")
    assert errors == [], f"the page reported JS errors: {errors}"

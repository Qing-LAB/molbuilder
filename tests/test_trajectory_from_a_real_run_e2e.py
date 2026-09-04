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


def test_the_viewer_draws_the_run_this_suite_just_optimised(
        page, flask_server, co2_optimization):
    """The other end, in a browser: the energy curve has one point per step.

    Seeded before `goto`, not selected after: the sidebar restores its
    selection at init, so setting one afterwards is a race the sidebar wins.
    That is written down in `test_task_setup_cell_types_e2e.py` and it cost
    an hour to rediscover on the spectra sibling.
    """
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: (errors.append(m.text)
                                  if m.type == "error" else None))

    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir', {json.dumps(str(co2_optimization.parent))});"
        f" sessionStorage.setItem('molbuilder.current_file', {json.dumps(str(co2_optimization))});"
        "} catch (_) {}")
    page.goto(f"{flask_server}/results")
    page.wait_for_selector("#inspector-host", timeout=20000)
    page.wait_for_selector("#energy-plot", state="attached", timeout=30000)
    # Plotly hangs the drawn traces off the div itself, so this waits for the
    # DATA rather than for the container that will eventually hold it.
    # READ AT THE MOMENT THE CONDITION HOLDS, in one step.  Waiting and
    # then evaluating separately raced: the inspector re-renders on its poll
    # and replaces the plot node, so `#energy-plot` was null in the read a
    # tick after the wait had just seen it holding data.
    drawn = page.wait_for_function(
        """() => {
            const d = document.querySelector("#energy-plot");
            const y = d && d.data && d.data[0] && d.data[0].y;
            if (!y || !y.length) return null;
            // The molwatch log's step 0 is `kind: initial_preview` -- the
            // geometry as handed in, BEFORE any SCF -- so it carries no
            // energy and plots as a null.  Compare across the points that
            // have one; a null is a real part of this series, not a glitch.
            const e = y.filter(v => typeof v === "number");
            if (e.length < 2) return null;
            return {points: y.length, energies: e.length,
                    first: e[0], last: e[e.length - 1]};
        }""",
        timeout=30000, polling=250).json_value()

    steps = len(_frames(co2_optimization.read_text(encoding="utf-8")))

    # WHY THIS IS NOT AN EQUALITY, and it is the thing this test taught me.
    #
    # The viewer does not read the `.xyz` when a run directory holds a
    # `.molwatch.log` -- the discovery chain prefers the richer per-step log,
    # and you can see it has: the runtime line on the page ("CPU · 20T
    # BLAS=1 · 3.9 GB") exists only in that log's header.  The log opens with
    # `step_index: 0, kind: initial_preview` -- the geometry as handed in,
    # before the optimiser has taken a step -- so it carries ONE MORE point
    # than geomeTRIC wrote frames.
    #
    # Both counts are right for what they are, and asserting the xyz's count
    # here failed on correct code.  A hand-built fixture would never have
    # shown this, because the one I wrote on 2026-09-03 was a bare xyz with
    # no log beside it: the viewer's actual file-preference was invisible to
    # it.  So the assertion is the RELATIONSHIP, which holds either way.
    assert drawn["points"] >= steps, (
        f"the optimiser took {steps} steps and the energy curve plots only "
        f"{drawn['points']} points -- steps are being dropped on the way to "
        f"the plot.")
    assert drawn["points"] <= steps + 1, (
        f"the curve plots {drawn['points']} points for {steps} steps.  One "
        f"extra is the molwatch log's `initial_preview`; more than that "
        f"means points are being invented or a series is being concatenated "
        f"with itself.")
    assert drawn["energies"] >= steps - 1, (
        f"only {drawn['energies']} of {drawn['points']} plotted points carry "
        f"an energy, for {steps} optimiser steps.  The preview point has "
        f"none by nature; the rest must.")
    assert drawn["last"] < drawn["first"], (
        f"the plotted curve rises ({drawn['first']} -> {drawn['last']}) "
        f"while the run falls.  The viewer is drawing the relaxation "
        f"backwards, which would tell a person their optimisation diverged.")
    assert errors == [], f"the page reported JS errors: {errors}"

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


def _trajectory(path):
    """Read a run through **the project's own parser**, never a private one.

    `molbuilder.parse.registry.parse()` is the single door the app itself
    goes through -- `/api/watch/load` lands in the same parsers -- so a test
    that reads a run any other way is a second implementation that drifts
    from the contract the moment a format grows a field.

    This file used to carry two: a hand-rolled multi-frame XYZ splitter and
    a regex counting `==== molwatch step N begin ====`.  Both worked, and
    both were second answers to a question the project already answers.
    `parse()` dispatches on content, returns a `TrajectoryResult`, and its
    `Frame`s carry `.structure`, `.energy` and `.step_index` -- which is
    also what the viewer is drawing, so the comparison is like for like.
    """
    from molbuilder.parse.registry import parse
    return parse(Path(path))


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
        assert proc.returncode == 0, (
            f"the deck exited {proc.returncode}.  A run that writes a good "
            f"trajectory and then dies still leaves the file, so checking "
            f"only for the file lets a broken run through.\n"
            f"--- stderr ---\n{proc.stderr[-1500:]}")
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

    Read through `parse()` -- the same door the app uses -- on the `.xyz`
    directly.  Absorption hides that file from the PICKER; it does not stop
    anything opening it, and here the question is what the optimiser did,
    not what the menu offers.

    The failure this catches is a trajectory that *looks* fine -- right
    file, right frame count, parseable -- and is not a relaxation: an
    optimiser wired to the wrong gradient walks, and every frame after the
    first is then confident nonsense.  Only the energy and the geometry say
    so.
    """
    import numpy as np

    traj = _trajectory(co2_optimization)
    frames = traj.frames
    assert len(frames) >= 3, (
        f"a relaxation from {_START_ANG} A should take several steps; this "
        f"trajectory has {len(frames)} frame(s).  One frame means the "
        f"optimiser never ran, or the trajectory writer only kept the last.")

    bonds = [float(np.linalg.norm(np.asarray(f.structure.positions)[1]
                                  - np.asarray(f.structure.positions)[0]))
             for f in frames]
    assert bonds[0] == pytest.approx(_START_ANG, abs=1e-3), (
        f"the first frame should be the geometry we handed in "
        f"({_START_ANG} A); it is {bonds[0]:.4f}.  The trajectory does not "
        f"start where the input did.")
    assert bonds[-1] == pytest.approx(_EXPECTED_MIN, abs=0.05), (
        f"CO2 relaxed to {bonds[-1]:.4f} A at RHF/STO-3G; expected about "
        f"{_EXPECTED_MIN}.  Experiment is 1.162 -- a minimal basis with no "
        f"correlation lands a little long, and this band is wide enough for "
        f"that and narrow enough to catch a walk in the wrong direction.")
    # (No "did it move" assertion here: `bonds[-1] == approx(1.19, abs=0.05)`
    # above already forces a move of at least 0.06 A from the 1.30 A start,
    # so a separate `abs(...) > 0.05` could never fire.  It read as an
    # anti-vacuity guard and was arithmetic already implied -- the kind of
    # assertion that makes a test look stronger than it is.)

    # Energy is the honest monotone: geometry can overshoot and come back
    # (this run does), a line search cannot go uphill.
    energies = [f.energy for f in frames]
    assert all(e is not None for e in energies), (
        f"the .xyz's frames should each carry an energy; got {energies}")
    assert energies[-1] < energies[0], (
        f"the optimisation ended higher than it started "
        f"({energies[0]} -> {energies[-1]}).  That is not a relaxation.")
    assert all(b <= a + 1e-6 for a, b in zip(energies, energies[1:])), (
        f"the energy went uphill between steps: {energies}.  A line search "
        f"that accepts an uphill step is taking a gradient it should not.")


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

    # THE SATELLITES MUST ACTUALLY BE THERE, or "one entry" proves nothing:
    # a run that wrote no absorbable file leaves one entry with `absorbs()`
    # deleted.  These three are exactly the shapes `trajectory.js::absorbs`
    # folds, one per rule, so the assertion below exercises all three
    # rather than whichever the emitter happened to write.
    absorbable = {
        "co2opt_initial.xyz":    "rule 1 (<stem>_initial.xyz)",
        "co2opt_optimized.xyz":  "rule 2 (<stem>_optimized.xyz)",
        "co2opt_geom_optim.xyz": "rule 3 (<stem>_geom_*_optim.xyz)",
    }
    absent = sorted(n for n in absorbable if n not in on_disk)
    assert not absent, (
        f"the run wrote no {absent}, so this test would report 'one entry' "
        f"with absorbs() deleted -- {', '.join(absorbable[n] for n in absent)} "
        f"would go unexercised.  Folder: {on_disk}")

    assert offered == [str(log)], (
        f"the picker offers {[p.split('/')[-1] for p in offered]} for one "
        f"relaxation.  The folder holds {on_disk}, of which "
        f"{sorted(absorbable)} are the master's satellites; `absorbs()` "
        f"should fold all three into the .molwatch.log, leaving exactly one "
        f"entry (`results.md` § 2.3).  Before absorption landed this was "
        f"five (2026-08-04).")


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
    # THROUGH THE PROJECT'S PARSER, which is the same one `/api/watch/load`
    # runs -- so "what the plot shows" is compared against "what the app
    # served", not against a second reading of the bytes.
    served = _trajectory(log)
    steps = len(served.frames)
    with_energy = sum(1 for f in served.frames if f.energy is not None)
    assert steps >= 3, (
        f"the molwatch log parses to {steps} frames; a relaxation from "
        f"1.30 A takes several, so the fixture or the log emitter changed")
    assert with_energy == steps - 1, (
        f"{with_energy} of {steps} parsed frames carry an energy.  Exactly "
        f"one -- the `initial_preview` the log opens with -- has none by "
        f"nature; if that changed, the plot's null point changed with it "
        f"and the assertion below is measuring something else.")

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
    assert drawn["energies"] == with_energy, (
        f"{drawn['energies']} of {drawn['points']} plotted points carry an "
        f"energy; the parser found {with_energy} of {steps}.  The plot and "
        f"the parse disagree about which steps have a result.")
    assert drawn["last"] < drawn["first"], (
        f"the plotted curve rises ({drawn['first']} -> {drawn['last']}) "
        f"while the run falls.  The viewer is drawing the relaxation "
        f"backwards, which would tell a person their optimisation "
        f"diverged.")
    assert errors == [], f"the page reported JS errors: {errors}"


def test_the_run_this_suite_just_generated_declares_its_own_engine(
        co2_optimization):
    """The whole chain on a REAL run: writer -> disk -> reader -> field.

    Everything else pinning the engine declaration builds its input.
    This one does not: the deck above was rendered by the production
    seam, a real PySCF process ran it, and the directory is whatever
    that left behind. So it is the only test that can fail because the
    WRITER stopped writing.

    That gap was real. Until 2026-09-04 every test of this mechanism
    handed the reader a hand-typed PROVENANCE block, so an adversarial
    review deleted `engine=` from BOTH emitters and measured 853 tests
    still green -- the feature entirely gone, the suite entirely happy.

    The third assertion is the anti-vacuity guard and the point of the
    test. Without it this passes on the file-cluster sniff alone
    (`.chk` and `_geom_optim.xyz` are in `pyscf/warm-files.toml`), which
    is exactly the fallback that exists for directories molbuilder did
    NOT write -- so it would go green with the declaration missing,
    which is the failure this test exists to catch.
    """
    from molbuilder.parse.contract import _declared_in_provenance, engine_of
    from molbuilder.parse.scripts.provenance import _extract_provenance_dict

    run_dir = co2_optimization.parent

    deck = run_dir / "co2opt.py"
    block = _extract_provenance_dict(deck.read_text(encoding="utf-8"))
    assert (block or {}).get("engine") == "pyscf", (
        f"the deck this suite generated carries engine="
        f"{(block or {}).get('engine')!r} in its PROVENANCE block. That "
        f"line IS the declaration; without it a run directory falls back "
        f"to guessing from file shapes.")

    assert engine_of(run_dir) == "pyscf"

    assert _declared_in_provenance(run_dir) == {"pyscf"}, (
        "the DECLARATION must be what answered, not the file sniff. If "
        "this set is empty the run resolved from its .chk and "
        "_geom_optim.xyz -- the fallback meant for directories molbuilder "
        "did not write -- and the declaration is not being written at all")

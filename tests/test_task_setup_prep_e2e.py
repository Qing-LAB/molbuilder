"""The buttons produce a working folder — driven, not described.

`task-setup.md` § 7a is the chain this proves, end to end and in one test,
because **every defect this tab had on 2026-09-02 was a break BETWEEN two
steps** and each step looked fine from the one either side of it:

  * the checkpoint tick shipped `checked`, and a ticked checkpoint with no
    note disables Save — so Save was **dead on arrival on every folder** and
    nothing the page wrote ever reached disk;
  * a clean save said nothing at all (success was reported only when the
    preflight had notes), so a dead button and a working one looked alike;
  * the confirmation sat after a folder reload that could throw, and did;
  * one button changed meaning between clicks, keeping that meaning in a
    closure every repaint destroyed — so "Write it" ran a fresh preview;
  * the A13 block reported `-c` as unset while the header carried the config
    default, which is the lie A13 exists to forbid.

Unit tests saw none of it.  Each piece was correct on its own; the chain was
not.  So this test does what a person does — type, Save, Preview, Prep — and
then reads the `.sbatch` that came out.
"""
from __future__ import annotations

import json
import shutil

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def calc_dir():
    """A described calculation, with a probed machine record beside it.

    The record is not scenery: since 2026-09-02 a rank count is read from one
    and nowhere else, so a folder without it cannot be prepped at all
    (`running-a-job.md` § 3.1)."""
    import numpy as np

    from conftest import write_pseudos
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.scheduler import Environment, Topology
    from molbuilder.structure import Structure
    from molbuilder.task import Stage

    root = ROOT / "projects/_t_prep_e2e"
    if root.exists():
        shutil.rmtree(root)
    src_dir = root / "structure"
    src_dir.mkdir(parents=True)
    d = root / "optimization" / "probe"
    try:
        struct = Structure(elements=["H", "H"],
                           positions=np.array([[0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.74]]),
                           vacuum=(10.0, 10.0, 10.0))
        src = src_dir / "probe.xyz"
        src.write_text(struct.to_xyz(), encoding="utf-8")

        # THE DESCRIPTION IS BUILT BY THE PRODUCTION DOOR, not assembled by
        # hand.  A hand-built `StructureRef` carries the DEFAULT witness --
        # formula "" and 0 atoms -- and prep refuses a description whose
        # witness disagrees with the file beside it ("the structure has
        # changed since this calculation was described").  So the first
        # version of this fixture could not be prepped at all, and the
        # refusal it earned was the fixture's own, not the tab's.
        D.write_description(
            D.build_description(struct,
                                SiestaConfig(system_label="probe"),
                                [Stage(name="coarse",
                                       overrides={"mesh_cutoff": 200})],
                                engine="siesta", shape="hierarchical",
                                name="probe", source=str(src)),
            d, struct=struct)
        write_pseudos(d, ["H"])

        # The probe's answer, pre-seeded: a rank count is read from a machine
        # record and nowhere else since 2026-09-02, so a folder without one
        # cannot be prepped at all (`running-a-job.md` § 3.1).
        # `script_generation` rides ON THE RECORD, not in the local config:
        # a wrapper generated here would otherwise carry THIS machine's way
        # into its environment -- a path that need not exist on the target
        # (the refusal names it, and prep is right to refuse).
        (d / "environment.json").write_text(
            Environment(scheduler="slurm",
                        topology=Topology(sockets=2, cores_per_socket=32),
                        script_generation={"preamble": "true",
                                           "activation": "conda activate"},
                        ).to_json() + "\n", encoding="utf-8")
        # A SCHEDULER, or there is no `.sbatch` to read: `prep_jobset` emits
        # one only when the bundle names a queue system, and the whole point
        # of this test is the header that comes out.
        (d / ".molbuilder.json").write_text(json.dumps({
            "scheduler": {
                "kind": "slurm",
                "directives": {"partition": "public", "qos": "public"},
                "defaults": {"time": "0-04:00:00", "cpus_per_task": None,
                             "mem": None},
            },
        }), encoding="utf-8")
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _open(page, base, calc):
    slot = json.dumps(str(calc))
    page.add_init_script(
        "try {"
        f" sessionStorage.setItem('molbuilder.current_dir.task-setup', {slot});"
        f" sessionStorage.setItem('molbuilder.current_dir', {slot});"
        "} catch (_) {}")
    page.goto(f"{base}/task-setup")
    page.wait_for_function(
        "() => { const n = document.querySelector('.CodeMirror');"
        " return !!(n && n.CodeMirror); }", timeout=20000)


def test_the_buttons_produce_a_folder_that_carries_what_the_card_asked_for(
        page, flask_server, calc_dir):
    """Type · Save · Preview · Prep — then read the `.sbatch`."""
    _open(page, flask_server, calc_dir)

    # ── the machine, because prep refuses without one ──────────────────
    page.wait_for_selector("#ts-target-card button", timeout=20000)
    page.evaluate(
        "() => { for (const b of document.querySelectorAll('button'))"
        "  if ((b.textContent||'').trim().startsWith('(this machine)'))"
        "    { b.click(); return; } }")

    # ── SAVE IS NOT DISABLED ON ARRIVAL ────────────────────────────────
    # The checkpoint offer must not block the thing it is attached to.
    # The editor and its Save live in a fold (§ 9a.1); open it the way a
    # person does rather than reaching past it.
    page.evaluate(
        "() => { const d = document.getElementById('ts-editor-card');"
        "        if (d && 'open' in d) d.open = true; }")
    page.wait_for_selector("#ts-save", state="attached", timeout=20000)
    assert page.evaluate("() => document.getElementById('ts-ckpt').checked") \
        is False, "the checkpoint offer ships ticked again -- Save is dead"

    # ── add a rank count to the run card, then type it ─────────────────
    # A run needs no benchmark (stages.md § 6.8d), so this description has
    # none and the card starts with no rows -- "+ Add setting" is how a
    # person puts one there.
    page.wait_for_selector(".ts-runcard select.ts-pick", timeout=20000)
    page.evaluate(
        "() => { const cards = document.querySelectorAll('.ts-runcard');"
        "  const c = cards[0];"
        "  const sel = c.querySelector('select.ts-pick');"
        "  sel.value = 'mpi_np';"
        "  sel.dispatchEvent(new Event('change', {bubbles: true}));"
        "  for (const b of c.querySelectorAll('button'))"
        "    if (/Add setting/.test(b.textContent)) { b.click(); return; } }")
    sel = '.ts-runcard input[aria-label="value for mpi_np"]'
    page.wait_for_selector(sel, timeout=20000)
    page.fill(sel, "8")
    page.dispatch_event(sel, "change")
    page.wait_for_function(
        "() => { const cm = document.querySelector('.CodeMirror');"
        " return cm && /\"mpi_np\"/.test(cm.CodeMirror.getValue()); }",
        timeout=10000)

    # ── SAVE SAYS WHAT IT DID ──────────────────────────────────────────
    page.click("#ts-save")
    page.wait_for_function(
        "() => { const n = document.getElementById('ts-save-said');"
        " return n && !n.hidden && /Saved/.test(n.textContent); }",
        timeout=20000)

    # ...and it reached disk, which is the only bridge to prep.
    on_disk = json.loads((calc_dir / "task.json").read_text())
    assert on_disk["stages"][0].get("execution", {}).get("mpi_np") == 8, (
        "Save said it wrote, and the file does not carry it: "
        + json.dumps(on_disk["stages"][0]))

    # ── PREVIEW ENABLES PREP, AND NOT THE OTHER WAY ROUND ──────────────
    panel = "[id^=ts-steppanel]"
    prep_sel = (f"{panel} .ts-prep button:has-text('Prep run here')")
    assert page.locator(prep_sel).first.is_disabled(), (
        "Prep was live before anything had been previewed")
    page.locator(f"{panel} .ts-prep button:has-text('Preview run')") \
        .first.click()
    page.wait_for_selector(f"{panel} .ts-emitted-row", timeout=30000)
    assert not page.locator(prep_sel).first.is_disabled(), (
        "a successful preview did not enable Prep")

    # ── A13: what the block says is what the header carries ────────────
    rows = page.evaluate(
        "() => Array.from(document.querySelectorAll('.ts-emitted-row'))"
        " .map(r => [r.querySelector('.ts-emitted-flag').textContent,"
        "            r.querySelector('.ts-emitted-val').textContent])")
    said = {flag: val for flag, val in rows}
    assert said.get("-n") == "8", f"the preview lost the rank count: {said}"

    # ── PREP, and read what it produced ────────────────────────────────
    page.locator(prep_sel).first.click()
    page.wait_for_function(
        "() => Array.from(document.querySelectorAll('.ts-prep-say'))"
        " .some(n => /Prepared for/.test(n.textContent))", timeout=60000)

    sbatch = list((calc_dir / "01_coarse").glob("*.sbatch"))
    assert sbatch, ("the button reported success and wrote no .sbatch: "
                    + repr(sorted(p.name for p in
                                  (calc_dir / "01_coarse").iterdir())))
    header = dict(
        (parts[1], " ".join(parts[2:]))
        for parts in (ln.split() for ln in sbatch[0].read_text().splitlines()
                      if ln.startswith("#SBATCH"))
        if len(parts) > 2)

    assert header.get("-n") == "8", (
        f"the card asked for 8 ranks; the header carries {header.get('-n')!r}")
    # EVERY NUMBER THE PREVIEW SHOWED IS THE ONE THE HEADER CARRIES.  The
    # preview said `-c` was unset while the header took the config default,
    # which is the lie A13 forbids -- and only comparing the two catches it.
    for flag, shown in said.items():
        if shown in ("—", "— cannot size"):
            continue
        if flag in header:
            assert header[flag] == shown, (
                f"the preview said {flag}={shown!r} and the .sbatch carries "
                f"{header[flag]!r}")

    # the deck and the wrapper came out too -- a folder you can actually run
    d = calc_dir / "01_coarse"
    assert list(d.glob("*.fdf")), "no deck"
    assert list(d.glob("*.run.sh")), "no wrapper"


@pytest.fixture(scope="module")
def filled_dir():
    """A calculation that ALREADY states a run condition, at both levels.

    The point of the fixture is that nothing here is typed by the test: the
    values are on disk before the page opens, which is the state a person is
    in every time they come back to a folder.
    """
    import numpy as np

    from conftest import write_pseudos
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.scheduler import Environment, Topology
    from molbuilder.structure import Structure
    from molbuilder.task import Allocation, Stage, read_task, write_task

    root = ROOT / "projects/_t_prep_filled"
    if root.exists():
        shutil.rmtree(root)
    src_dir = root / "structure"
    src_dir.mkdir(parents=True)
    d = root / "optimization" / "filled"
    try:
        struct = Structure(elements=["H", "H"],
                           positions=np.array([[0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.74]]),
                           vacuum=(10.0, 10.0, 10.0))
        src = src_dir / "filled.xyz"
        src.write_text(struct.to_xyz(), encoding="utf-8")
        D.write_description(
            D.build_description(struct,
                                SiestaConfig(system_label="filled"),
                                [Stage(name="coarse",
                                       overrides={"mesh_cutoff": 200})],
                                engine="siesta", shape="hierarchical",
                                name="filled", source=str(src)),
            d, struct=struct)
        write_pseudos(d, ["H"])
        (d / "environment.json").write_text(
            Environment(scheduler="slurm",
                        topology=Topology(sockets=2, cores_per_socket=32),
                        script_generation={"preamble": "true",
                                           "activation": "conda activate"},
                        ).to_json() + "\n", encoding="utf-8")

        # THE THREE ROW STATES, one of each, arranged on purpose
        # (`task-setup.md` § 6.2b):
        #   `omp_threads`  chosen    -- this rung says 4
        #   `mpi_np`       inherited -- only the calculation-wide block says 16
        #   `time`         chosen    -- a LANE ASK, this run's own (§ 6.8e)
        # and `allocation` beside them, which is a different block for a
        # different question (§ 7).
        t = read_task(d / "task.json")
        write_task(d / "task.json", type(t)(
            **{**{f.name: getattr(t, f.name)
                  for f in __import__("dataclasses").fields(t)},
               "execution": {"mpi_np": 16},
               "allocation": Allocation(time="0-04:00:00", mem="16G",
                                        domain="debug"),
               "stages": (Stage(name="coarse",
                                overrides={"mesh_cutoff": 200},
                                execution={"omp_threads": 4,
                                           "time": "2-00:00:00"}),)}))
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _rows(page):
    """Every run-card row, as {name: {kind, value, placeholder}}."""
    return page.evaluate(
        "() => { const out = {};"
        " for (const r of document.querySelectorAll('.ts-runcard .ts-row')) {"
        "   const n = r.querySelector('.ts-row-name');"
        "   const i = r.querySelector('.ts-runval');"
        "   if (!n || !i) continue;"
        "   const name = (n.childNodes[0] || {}).textContent || '';"
        "   out[name.trim()] = { kind: r.getAttribute('data-kind'),"
        "     value: i.value, placeholder: i.placeholder || '',"
        "     note: (r.querySelector('.ts-row-note') || {}).textContent || '' };"
        " } return out; }")


def test_a_folder_opens_showing_what_it_already_states(
        page, flask_server, filled_dir):
    """**Coming back to a folder is a LOAD, and it shows what the file says**
    (`task-setup.md` § 2.1, § 9's *open a folder* row).

    The page holds no state of its own, so everything on screen has to come
    back out of `task.json` -- and § 6.2b's three row states are how the card
    says WHICH of the two `execution` blocks answered each row.

    **The empty field on an inherited row is the load-bearing half.** Filling
    it with the merged value would make blank and `×` unreachable: both write
    this rung's own block, so with nothing there they write nothing and the
    field snaps back.
    """
    _open(page, flask_server, filled_dir)
    page.wait_for_selector(".ts-runcard .ts-row", timeout=20000)
    rows = _rows(page)

    assert set(rows) >= {"mpi_np", "omp_threads", "time"}, (
        "a stated value produced no row -- the card cannot show what the "
        f"file says: {rows}")

    # CHOSEN: this rung's own value, in the field.
    assert rows["omp_threads"]["kind"] == "chosen", rows["omp_threads"]
    assert rows["omp_threads"]["value"] == "4", rows["omp_threads"]

    # INHERITED: field EMPTY, the calculation-wide value in the placeholder,
    # marked so you can tell it from something you typed.
    assert rows["mpi_np"]["kind"] == "inherited", rows["mpi_np"]
    assert rows["mpi_np"]["value"] == "", (
        "an inherited row filled its field, so blank and × cannot be "
        f"reached: {rows['mpi_np']}")
    assert "16" in rows["mpi_np"]["placeholder"], rows["mpi_np"]
    assert "every stage" in rows["mpi_np"]["placeholder"], rows["mpi_np"]

    # A LANE ASK is an ordinary chosen row -- `time` is the RUN's own, and
    # `allocation.time` beside it is the calculation's and the bench's
    # (`stages.md` § 6.8e).  They are different fields and both are stated.
    assert rows["time"]["kind"] == "chosen", rows["time"]
    assert rows["time"]["value"] == "2-00:00:00", rows["time"]
    ask = page.evaluate(
        "() => (document.getElementById('ts-ask-time') || {}).value || ''")
    assert ask == "0-04:00:00", (
        "the queue card lost the calculation's own wall clock, or read the "
        f"run's: {ask!r}")

    # `mem` HAS NO SECOND HOME and must not be offerable on the run card.
    offered = page.evaluate(
        "() => Array.from("
        "  document.querySelectorAll('.ts-runcard select.ts-pick option'))"
        " .map(o => o.value).filter(Boolean)")
    assert "mem" not in offered, (
        "the run card offered `mem`, which § 6.8e keeps out of `execution` "
        "on purpose -- a trial and a run hold about the same memory")


def test_what_you_typed_is_still_there_when_you_come_back(
        page, flask_server, filled_dir):
    """**Persistency is `task.json`, and nothing else** (§ 7a).

    There is no page-level store to remember a form: the value is written to
    the file by Save, and the file is what the page reads on the way back in.
    So this walks all three ways of coming back -- another tab, a full
    reload, and re-opening the folder -- and the value has to survive every
    one of them.
    """
    _open(page, flask_server, filled_dir)
    page.evaluate(
        "() => { const d = document.getElementById('ts-editor-card');"
        "        if (d && 'open' in d) d.open = true; }")
    sel = '.ts-runcard input[aria-label="value for omp_threads"]'
    page.wait_for_selector(sel, timeout=20000)

    page.fill(sel, "6")
    page.dispatch_event(sel, "change")
    page.click("#ts-save")
    page.wait_for_function(
        "() => { const n = document.getElementById('ts-save-said');"
        " return n && !n.hidden && /Saved/.test(n.textContent); }",
        timeout=20000)

    on_disk = json.loads((filled_dir / "task.json").read_text())
    assert on_disk["stages"][0]["execution"]["omp_threads"] == 6, on_disk

    # 1 -- away to another tab, and back.
    page.goto(f"{flask_server}/molbuilder")
    page.wait_for_load_state("domcontentloaded")
    _open(page, flask_server, filled_dir)
    page.wait_for_selector(".ts-runcard .ts-row", timeout=20000)
    assert _rows(page)["omp_threads"]["value"] == "6", (
        "the value did not survive leaving the tab and coming back")

    # 2 -- a full reload of this tab.
    page.reload()
    page.wait_for_selector(".ts-runcard .ts-row", timeout=20000)
    assert _rows(page)["omp_threads"]["value"] == "6", (
        "the value did not survive a page reload")


# NOT tested here: that a bool still renders as a chooser on the SECOND load
# (the 2026-08-24 defect).  A test for it was written and then deleted, and
# the reason is worth keeping.
#
# The defect was a vocabulary loader returning from its cache before it
# published into `_meta`.  Both loaders now refill on BOTH paths, and because
# `_fillSweepMeta` fills any name `_meta` is missing, EITHER ONE alone
# restores what the other dropped.  Reverting the documented fix in
# `loadColumnChoices` -- the literal original bug -- leaves every dropdown a
# dropdown.  Only mutating both publishers at once reproduces it, and that is
# two regressions, not one.
#
# So there is no single break for such a test to catch: the invariant is held
# by the shape of the code, not by a check.  A test that cannot fail for the
# reason it names is the thing this whole sweep is retiring, and writing a new
# one would have been the same mistake in the other direction.
def test_a_row_added_on_one_folder_does_not_follow_you_to_the_next(
        page, flask_server, filled_dir, calc_dir):
    """**§ 2.1: the page holds no state of its own** -- *"no in-progress
    buffer that outlives a directory change"*, and § 9: opening a folder is a
    **load, not a merge**.

    A row put on the card by *+ Add setting* is the sharpest case, because it
    is the one thing on this card that is NOT in the file yet: it exists only
    as a row waiting for a value.  Carried across a folder change it would
    show the next calculation asking for something nobody asked it for.
    """
    _open(page, flask_server, filled_dir)
    page.wait_for_selector(".ts-runcard select.ts-pick", timeout=20000)
    page.evaluate(
        "() => { const c = document.querySelector('.ts-runcard');"
        "  const s = c.querySelector('select.ts-pick');"
        "  s.value = 'block_size';"
        "  s.dispatchEvent(new Event('change', {bubbles: true}));"
        "  for (const b of c.querySelectorAll('button'))"
        "    if (/Add setting/.test(b.textContent)) { b.click(); return; } }")
    page.wait_for_function(
        "() => !!document.querySelector("
        "  '.ts-runcard [aria-label=\"value for block_size\"]')",
        timeout=10000)

    # ...now switch folders THE WAY THE SIDEBAR DOES, without a reload --
    # a reload would rebuild the page and prove nothing about state that
    # outlives a directory change.
    page.evaluate(
        "(dir) => window.molbuilder.projects.setShared(dir, '')",
        str(calc_dir))   # one arg, positional -- evaluate's own signature
    # WAIT ON A POSITIVE SIGNAL -- the editor showing the OTHER
    # calculation's id -- not on the absence of the row, which is the thing
    # being asserted.  The old card stays in the DOM until the new folder's
    # description has been read, so a bare `.ts-runcard` selector matches
    # the card that is on its way out.
    page.wait_for_function(
        "() => { const cm = document.querySelector('.CodeMirror');"
        " return cm && /\"probe_H2\"/.test(cm.CodeMirror.getValue()); }",
        timeout=20000)
    page.wait_for_selector(".ts-runcard", timeout=20000)

    assert "block_size" not in _rows(page), (
        "a row added on the previous folder followed the page into this "
        "one -- § 2.1's 'no in-progress buffer that outlives a directory "
        "change'")


def test_a_parameter_write_and_a_save_BLOCK_THE_WINDOW(
        page, flask_server, filled_dir):
    """`ui-contract.md` § 10 — **one cover for every heavy click**, and a
    parameter write is one.

    A card edit is three writes that must land together: this rung's
    `execution` in the model, the editor buffer that Save posts, and the
    panels rendered from both. While that was merely *fast*, a second edit
    arriving inside it interleaved — type a memory value, blur, click
    "+ Add stage" straight away, and the block the first was writing was
    gone.

    **How fast is fast enough is the wrong question.** The framework already
    has a barrier that covers the whole window, so the write takes it and
    the question does not arise *(user, 2026-09-02: "before the parameter is
    ready or persistently saved, the user can't do anything … what the fuck
    do you care about half a second")*.

    Recorded rather than raced: catching the cover mid-flight is a
    stopwatch test and would be flaky. What is asserted is the property §
    10 states — the operation **claims**, and it **releases in a finally**,
    so the window is never left covered.
    """
    _open(page, flask_server, filled_dir)
    page.evaluate(
        "() => { const d = document.getElementById('ts-editor-card');"
        "        if (d && 'open' in d) d.open = true; }")
    sel = '.ts-runcard input[aria-label="value for omp_threads"]'
    page.wait_for_selector(sel, timeout=20000)

    # Record every claim/release without changing what they do.
    page.evaluate(
        "() => { const f = window.molbuilder.pageBusy;"
        "  window.__fence = [];"
        "  const c = f.claim.bind(f), r = f.release.bind(f);"
        "  f.claim = (reason, cx) => { window.__fence.push(['claim', reason]);"
        "                              return c(reason, cx); };"
        "  f.release = () => { window.__fence.push(['release', null]);"
        "                      return r(); }; }")

    page.fill(sel, "3")
    page.dispatch_event(sel, "change")
    page.wait_for_function(
        "() => window.__fence.some(e => e[0] === 'release')", timeout=20000)
    after_edit = page.evaluate("() => window.__fence.slice()")

    page.evaluate("() => { window.__fence.length = 0; }")
    page.click("#ts-save")
    # WAIT FOR THE FENCE, not for the message.  The confirmation is said
    # BEFORE the folder is re-read (`task-setup.md` § 7a, rule 3: a message
    # about the write belongs to the write), and the re-read is inside the
    # operation -- so "Saved" appears while the window is still, correctly,
    # covered.  Sampling on the message caught the save mid-flight and read
    # as a stuck cover.
    page.wait_for_function(
        "() => window.__fence.some(e => e[0] === 'release')", timeout=30000)
    after_save = page.evaluate("() => window.__fence.slice()")
    assert page.evaluate(
        "() => { const n = document.getElementById('ts-save-said');"
        " return !!n && !n.hidden && /Saved/.test(n.textContent); }"), (
        "the save finished without saying so")

    for what, log in (("a parameter write", after_edit), ("a save", after_save)):
        kinds = [e[0] for e in log]
        assert "claim" in kinds, (
            f"{what} did not cover the window -- everything else on the page "
            f"stayed clickable while it ran: {log}")
        assert kinds.count("claim") == kinds.count("release"), (
            f"{what} left the window covered: {log}")
        reason = next(e[1] for e in log if e[0] == "claim")
        assert reason and reason.strip(), (
            f"{what} covered the window with no reason shown: {log}")

    # And the fence is DOWN afterwards -- no stuck cover.
    assert page.evaluate("() => window.molbuilder.pageBusy.isClaimed()") is False

    # The value still reached disk, which is the point of blocking at all.
    on_disk = json.loads((filled_dir / "task.json").read_text())
    assert on_disk["stages"][0]["execution"]["omp_threads"] == 3, on_disk


def test_a_check_that_cannot_run_SAYS_SO_rather_than_going_blank(
        page, flask_server, filled_dir):
    """**"I don't know" is an answer; a blank space is not.**

    Both fit panels used to set `hidden` on any failure, on the reasoning
    that the rows above are the substance and the card should not break.
    That produces the worst of the three readings: an empty space where a
    verdict belongs is indistinguishable from *everything fits* and from
    *this feature is gone*, and the one thing it never says is the true one
    — that the check could not run *(user, 2026-09-02: "we can't have a fit
    on. I just said, I don't know. Lack of information.")*.

    Driven by making the door fail, which is the only honest way to reach
    the branch: the server is what decides it cannot answer.
    """
    _open(page, flask_server, filled_dir)
    page.wait_for_selector(".ts-runcard .ts-row", timeout=20000)

    # The door refuses, as it does on a folder whose machine has no record.
    page.route("**/api/task-setup/bench-grid",
               lambda route: route.fulfill(
                   status=400, content_type="application/json",
                   body=json.dumps({"ok": False,
                                    "error": "no machine record for 'sol'"})))

    # Nudge a value so the panel re-asks.
    sel = '.ts-runcard input[aria-label="value for omp_threads"]'
    page.fill(sel, "5")
    page.dispatch_event(sel, "change")

    page.wait_for_function(
        "() => Array.from(document.querySelectorAll('.ts-fit'))"
        " .some(n => !n.hidden && /Cannot say/.test(n.textContent))",
        timeout=20000)

    said = page.evaluate(
        "() => Array.from(document.querySelectorAll('.ts-fit'))"
        " .filter(n => !n.hidden).map(n => n.textContent).join(' | ')")
    assert "Cannot say whether this fits" in said, said
    # AND IT CARRIES THE SERVER'S REASON, not a paraphrase of our own.
    assert "no machine record" in said, (
        "the panel said it could not answer but not why: " + said)


# ---------------------------------------------------------------------------
#  What the card TELLS YOU TO RUN  (plans/plan.md § 5h, cluster 4)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_stage_dir():
    """A described calculation with TWO stages and a declared bench axis.

    Two, because the claim under test is *per stage, not only the first* --
    the card once offered `prep bench` for `enabled[0]` alone, which a
    one-stage fixture cannot tell apart from correct.  The bench block,
    because the Measure half renders only when the description declares axes
    to measure (`viewer.js`: `if (benchKeys.length)`).
    """
    import json as _json

    import numpy as np

    from conftest import write_pseudos
    from molbuilder import describe as D
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.scheduler import Environment, Topology
    from molbuilder.structure import Structure
    from molbuilder.task import Stage

    # UNDER THE PROJECTS ROOT, like `filled_dir` and for the same reason: the
    # page opens a folder through the picker, so a tmpdir outside the root is
    # never reached and the tab silently shows the root instead -- which is
    # what the first version of this fixture did, and it read as "the stage
    # blocks do not render".
    root = ROOT / "projects/_t_two_stage"
    if root.exists():
        shutil.rmtree(root)
    src_dir = root / "structure"
    src_dir.mkdir(parents=True)
    d = root / "optimization" / "ladder"
    try:
        struct = Structure(elements=["H", "H"],
                           positions=np.array([[0.0, 0.0, 0.0],
                                               [0.0, 0.0, 0.74]]),
                           vacuum=(10.0, 10.0, 10.0))
        src = src_dir / "ladder.xyz"
        src.write_text(struct.to_xyz(), encoding="utf-8")
        D.write_description(
            D.build_description(struct, SiestaConfig(system_label="ladder"),
                                [Stage(name="coarse",
                                       overrides={"mesh_cutoff": 200}),
                                 Stage(name="tight",
                                       overrides={"mesh_cutoff": 400})],
                                engine="siesta", shape="hierarchical",
                                name="ladder", source=str(src)),
            d, struct=struct)
        write_pseudos(d, ["H"])

        # The axes to measure, declared once for the calculation -- which is
        # exactly why every enabled stage can be measured.
        task = _json.loads((d / "task.json").read_text())
        task["bench"] = {"mpi_np": [1, 2]}
        (d / "task.json").write_text(_json.dumps(task, indent=2),
                                     encoding="utf-8")

        (d / "environment.json").write_text(
            Environment(scheduler="",
                        topology=Topology(sockets=2, cores_per_socket=32),
                        script_generation={"preamble": "true",
                                           "activation": "conda activate"},
                        ).to_json() + "\n", encoding="utf-8")
        yield d
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_commands_the_card_hands_over(page, flask_server, two_stage_dir):
    """The commands a person copies, READ FROM THE PAGE.

    CONVERTED 2026-09-06 (`plans/plan.md` § 5h, cluster 4).  Nine assertions
    in `test_task_setup_tab.py` read `viewer.js` as text and checked for
    concatenation expressions -- `'prep bench " + name + _bundleArg() +
    _targetArg()' in src`.  That pins one spelling of one line: reorder the
    operands harmlessly and it fails; render the block for the wrong stage,
    or in the wrong order, and it passes.  This reads what the card renders.

    The claims, all of them about what a person is TOLD:
      * every enabled stage offers both things you can do with it -- the card
        once hardwired `prep bench` to `enabled[0]`, a guess dressed as an
        answer;
      * each command names ITS OWN stage;
      * the bench order is shown in order, because skipping `summarize` does
        not fail -- it quietly preps a run with no measured verdict behind it;
      * `launch` never carries `--target`, because launching happens ON the
        machine and there is nothing to target.
    """
    _open(page, flask_server, two_stage_dir)
    # `state="attached"`: the stage blocks are TABS -- every stage renders
    # its own and all but the selected one carry `hidden`, so waiting for a
    # VISIBLE one would wait for the tab a person has not clicked.
    page.wait_for_selector("pre.ts-cmd", state="attached", timeout=20000)

    blocks = page.eval_on_selector_all(
        "pre.ts-cmd", "els => els.map(e => e.textContent)")
    joined = "\n".join(blocks)

    # ── every stage, not only the first ────────────────────────────────
    for stage in ("coarse", "tight"):
        assert f"prep bench {stage}" in joined, (
            f"the card offers no bench command for {stage!r} -- the axes are "
            "declared once for the calculation, so every enabled stage can "
            "be measured")
        assert f"prep run {stage}" in joined, (
            f"the card offers no run command for {stage!r}")

    # ── the order is load-bearing, and is shown ────────────────────────
    bench = next(b for b in blocks if "prep bench coarse" in b)
    assert (bench.index("prep bench") < bench.index("launch bench")
            < bench.index("summarize bench")), (
        "the bench order is not shown in order -- skipping summarize does "
        "not fail, it preps a run with no measured verdict behind it")

    # NOT ASSERTED HERE, and the reason is measured: *launch never carries
    # --target*.  `_targetArg()` returns "" unless a NAMED machine is chosen,
    # and this page can only be driven to "(this machine)" without a named
    # record in the server's config root -- so a check for the absence of
    # `--target` passes no matter what the code does.  Adding `_targetArg()`
    # to the launch line leaves this test GREEN (verified 2026-09-06).  A
    # vacuous assertion is the thing this conversion exists to remove, so the
    # claim stays a source pin in `test_task_setup_tab.py` until a fixture
    # can supply a named target.

    # ── and each half says what it is for, in the page's own component ──
    hints = page.eval_on_selector_all(
        "p.hint", "els => els.map(e => e.textContent).join('\\n')")
    assert "Measure it" in hints and "Run it" in hints, (
        "the per-stage blocks do not explain themselves")
    assert "--np / --omp / --time" in hints, (
        "a person who filled the card is not told a flag still overrides it")


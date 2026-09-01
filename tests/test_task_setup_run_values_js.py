"""The run's own values, and the check beside them — `task-setup.md` § 6.2b.

**The concealment this card was rebuilt to end** *(user, 2026-09-01)*: the run
took `run-config.toml`'s verdict for every field the description had not
stated, and the tab gave no way to state one. So the recommendation was the
only input, what it chose was never shown, and a run needed a benchmark to
have been *executed*.

These drive the REAL functions out of `task-setup/viewer.js` under Node, for
the reason `test_task_setup_cell_readers_js.py` gives at length: a test that
does not run the source can only check that names exist, and a stub returning
the wrong thing passes that.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "molbuilder/web/static/task-setup/viewer.js"


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)].rstrip()


def _run(js: str, fns: str) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    harness = f"""
        const _els = {{}};
        function el(tag, attrs, ...kids) {{
            const n = {{ tag, attrs: attrs || {{}}, kids: [], textContent: "",
                        hidden: false,
                        appendChild(c) {{ this.kids.push(c); }},
                        set text(v) {{ this.textContent = v; }} }};
            for (const k of kids) if (k) n.kids.push(k);
            if (typeof kids[0] === "string") n.textContent = kids[0];
            return n;
        }}
        function $(id) {{ return _els[id] || null; }}
{fns}
{js}
    """
    proc = subprocess.run([node, "--input-type=commonjs", "-e", harness],
                          capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        pytest.fail(f"node exited {proc.returncode}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _painter() -> str:
    src = VIEWER.read_text(encoding="utf-8")
    return _slice(src, "function paintRunFit(host, body)",
                  "function renderMachine(task)")


def _flat(node) -> str:
    """Every string in a painted tree, joined — what a person would read."""
    out = [node.get("textContent") or ""]
    for k in node.get("kids") or []:
        out.append(_flat(k) if isinstance(k, dict) else str(k))
    return " ".join(x for x in out if x)


class TestTheCheckIsBesideTheField:

    def test_a_fitting_ask_names_the_queues_that_would_take_it(self):
        """The shape and where it would go — the same two facts the grid
        block shows for a measurement, for the one point a run is."""
        out = _run(
            """
            const host = el("div", null);
            paintRunFit(host, { ok: true, stated: true, fits: true,
              cell: { shape: "8 rank(s) x 2 core(s)",
                      fits: ["public", "bench", "gpu"], why: [] } });
            console.log(JSON.stringify({ hidden: host.hidden, host }));
            """, _painter())
        assert out["hidden"] is False
        text = _flat(out["host"])
        assert "8 rank(s) x 2 core(s)" in text
        assert "public" in text and "bench" in text

    def test_an_impossible_ask_is_struck_WITH_THE_NUMBERS(self):
        """`scheduler.md` R4: the reason says what to change. A verdict word
        would send a person to the CLI to find out which number was wrong —
        the failure this whole lane was rebuilt to remove."""
        why = "needs 100000 cores but gpu's largest machine has 48"
        out = _run(
            f"""
            const host = el("div", null);
            paintRunFit(host, {{ ok: true, stated: true, fits: false,
              cell: {{ shape: "100000 rank(s) x 1 core(s)", fits: [],
                      why: [{json.dumps(why)}] }} }});
            console.log(JSON.stringify({{ hidden: host.hidden, host }}));
            """, _painter())
        assert out["hidden"] is False
        text = _flat(out["host"])
        assert why in text, text
        assert "100000" in text

    def test_an_ask_no_queue_holds_says_what_to_do(self):
        out = _run(
            """
            const host = el("div", null);
            paintRunFit(host, { ok: true, stated: true, cell: null });
            console.log(JSON.stringify({ hidden: host.hidden, host }));
            """, _painter())
        assert out["hidden"] is False
        text = _flat(out["host"]).lower()
        assert "no queue" in text
        assert "different machine" in text or "lower" in text


def test_the_page_computes_no_verdict_of_its_own():
    """One admission path (`generator.md` § 4.3a). A grid computed in the
    page would be a second decider, free to say a run is fine where `launch`
    refuses it."""
    src = VIEWER.read_text(encoding="utf-8")
    fn = _slice(src, "function paintRunFit(host, body)",
                "function renderMachine(task)")
    for invented in ("max_cores", "largest machine", "> 48", "cores >"):
        assert invented not in fn, f"the painter decides {invented} itself"
    assert "/api/task-setup/run-fit" in src, "the page must ask the door"


def test_an_unstated_row_is_never_written_into_the_description():
    """A row you added but have not filled is the page's state, not the
    calculation's. Writing `"mpi_np": ""` would make the reader invent a
    meaning for it — absent-is-a-state, all the way down."""
    src = VIEWER.read_text(encoding="utf-8")
    assert "_extraRunRows" in src, "the unstated rows have no home"
    # DRIVEN, not grepped: a source that merely CONTAINS the delete passes a
    # substring check with the branch disabled, which is exactly what a
    # mutation showed on 2026-09-01.
    fns = "\n\n".join([
        _slice(src, "function coercePoint(raw)", "function benchOf()"),
        _slice(src, "function runValuesOf()", "/** The parameters"),
        _slice(src, "function setRunValue(name, raw)", "function dropRunValue"),
    ])
    out = _run(
        """
        let _task = { allocation: { domain: "htc", mpi_np: 8 } };
        function syncFromModel() {}
        setRunValue("mpi_np", "");
        const afterBlank = JSON.parse(JSON.stringify(_task));
        setRunValue("mpi_np", "16");
        console.log(JSON.stringify({ afterBlank, afterTyped: _task }));
        """, fns)
    assert "mpi_np" not in out["afterBlank"]["allocation"], (
        f"a blank was stored rather than removed: {out['afterBlank']}")
    assert out["afterBlank"]["allocation"] == {"domain": "htc"}
    assert out["afterTyped"]["allocation"]["mpi_np"] == 16, (
        "and a typed value must still land, as a NUMBER")


# --------------------------------------------------------------------- #
#  Measuring is not running — stages.md § 6.8c                           #
# --------------------------------------------------------------------- #

def _bench_alloc(domain, time_):
    src = VIEWER.read_text(encoding="utf-8")
    fns = _slice(src, "function applyBenchAllocToDoc()",
                 "/* ---------- when this run should tell you something")
    return _run(
        f"""
        _els["ts-bench-domain"] = {{ value: {json.dumps(domain)} }};
        _els["ts-bench-time"]   = {{ value: {json.dumps(time_)} }};
        let _task = {{ allocation: {{ domain: "htc", time: "2-00:00:00" }} }};
        function syncFromModel() {{}}
        applyBenchAllocToDoc();
        console.log(JSON.stringify({{ task: _task }}));
        """, fns)


class TestTheBenchMayAskForItsOwnWall:

    def test_what_is_typed_lands_in_its_own_block(self):
        """A benchmark is short by construction and a run is not.  One wall
        serving both queues a thirty-second job behind a two-day
        reservation, or kills the calculation."""
        out = _bench_alloc("general", "30m")
        assert out["task"]["bench_allocation"] == {"domain": "general",
                                                   "time": "30m"}
        # and it does not disturb the RUN's, which is the whole point
        assert out["task"]["allocation"] == {"domain": "htc",
                                             "time": "2-00:00:00"}

    def test_an_empty_block_writes_no_key(self):
        """Absent-is-a-state: absent means "use the run's", so a description
        whose bench asks for what the run asks for round-trips untouched —
        which is every description written before 2026-09-01."""
        out = _bench_alloc("", "")
        assert "bench_allocation" not in out["task"], out["task"]

    def test_one_field_alone_is_a_real_answer(self):
        """Field by field, like everything else here: a shorter wall on the
        run's own queue is the ordinary case."""
        out = _bench_alloc("", "30m")
        assert out["task"]["bench_allocation"] == {"time": "30m"}

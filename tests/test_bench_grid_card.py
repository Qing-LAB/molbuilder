"""The bench grid, live in the card that sets the axes.

User, 2026-08-30: *"can't this list be just updated in the same card where
the parameters are set? this update is trivial when target and parameter
list is known and can be updated in real time. this does not need to be a
message with a window."*

So `/api/task-setup/bench-grid` serves the report `_bench_inputs` already
computes — **the one enumerator**, handed the axes as they are being
edited.  The browser paints it; it never enumerates a grid of its own,
because a second enumerator is exactly the drifting decider that let a
cell look fine in one place and be refused in another.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VIEWER = REPO / "molbuilder" / "web" / "static" / "task-setup" / "viewer.js"

_DECK_TASK = {
    "schema": "molbuilder/task@1",
    "label": "probe",
    "engine": "siesta",
    "calculation": "relaxation",
    "structure": "probe.xyz",
    "stages": [{"name": "coarse", "values": {}}],
    "bench": {"mpi_np": [2, 4], "omp_threads": [1]},
}


@pytest.fixture()
def bundle(tmp_path, monkeypatch):
    """A description whose bench axes can be resolved, with tmp registered
    as a picker root so the door may read it."""
    from molbuilder import diagnostics
    src = REPO / "projects" / "Au-BDT-Au" / "optimization" / "AuBDTAu-slabcorrected"
    if not src.is_dir():
        pytest.skip("the worked-example bundle is not in this checkout")
    import shutil
    dst = tmp_path / "bundle"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        ".git", ".binsnapshots", "01_coarse"))
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(type(caps), "file_picker_roots",
                        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)
    return dst


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


def _post(client, dest, bench, target="(this machine)"):
    return client.post("/api/task-setup/bench-grid",
                       json={"dest": str(dest), "target": target,
                             "bench": bench}).get_json()


class TestTheDoorServesTheOneEnumerator:

    def test_it_answers_the_cells_and_where_each_would_go(self, client, bundle):
        # A CPU-family grid states no device counts: declaring `gpu_count`
        # beside `use_gpu: [false]` is refused by name upstream, because
        # the counts would be silently ignored.
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False]})
        assert d["ok"] is True
        assert d["cells"], "a resolvable grid must answer its cells"
        one = d["cells"][0]
        assert set(one) >= {"label", "shape", "family", "ranks",
                            "cores_each", "gpus", "gpu_type", "fits", "why"}
        assert one["fits"], "this bundle's queues take a 48-rank CPU cell"

    def test_the_axes_SENT_win_over_the_axes_on_disk(self, client, bundle):
        """The card's edits live in the browser's model until the person
        saves, so a list read from `task.json` would describe the previous
        state.  The point of the door is that it does not."""
        on_disk = json.loads((bundle / "task.json").read_text())["bench"]
        assert on_disk["mpi_np"] == [48], on_disk

        d = _post(client, bundle, {**on_disk, "mpi_np": [48, 128]})
        labels = {c["label"] for c in d["cells"]}
        assert any("K128" in ell for ell in labels), (
            f"the in-flight 128 must reach the grid; got {sorted(labels)}")

    def test_a_cell_no_queue_takes_is_returned_struck_not_dropped(
            self, client, bundle):
        """R4 -- the struck row carries the numbers to change.  Dropping it
        would leave the person guessing why their point vanished."""
        d = _post(client, bundle, {"mpi_np": [48, 128], "omp_threads": [1],
                                   "use_gpu": [True], "gpu_count": [4]})
        struck = [c for c in d["cells"] if c["why"]]
        assert struck, "a 128-rank GPU cell fits no queue on this record"
        assert any(ch.isdigit() for ch in struck[0]["why"][0])

    def test_nothing_surviving_is_a_result_not_an_error(self, client, bundle):
        """*Nothing here fits* is the answer, and the crossed-out rows are
        how the person sees why.  A 400 would throw them away."""
        d = _post(client, bundle, {"mpi_np": [999999], "omp_threads": [1],
                                   "use_gpu": [False]})
        assert d["ok"] is True and d["kept"] == 0
        assert d["cells"] and all(c["why"] for c in d["cells"])

    def test_a_bench_that_is_not_an_object_is_refused(self, client, bundle):
        r = client.post("/api/task-setup/bench-grid",
                        json={"dest": str(bundle), "bench": [1, 2]})
        assert r.status_code == 400
        assert "bench" in r.get_json()["error"]

    def test_the_folder_must_be_inside_a_picker_root(self, client):
        r = client.post("/api/task-setup/bench-grid",
                        json={"dest": "/etc", "bench": {}})
        assert r.status_code >= 400


    def test_an_unresolvable_declaration_answers_the_readers_own_words(
            self, client, bundle):
        """A grid that cannot be resolved at all is a 400 carrying the
        refusal the terminal gives -- not a browser-side paraphrase, which
        would be a second account of the same rule, free to drift."""
        d = _post(client, bundle, {"mpi_np": [48], "omp_threads": [1],
                                   "use_gpu": [False], "gpu_count": [2]})
        assert d["ok"] is False
        assert "gpu_count" in d["error"] and "use_gpu" in d["error"]


class TestTheBrowserDoesNotEnumerate:
    """Source pins.  The browser must ASK for the grid, never derive it."""

    def _src(self) -> str:
        src = VIEWER.read_text()
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        return re.sub(r"^\s*//.*$", "", src, flags=re.M)

    def test_the_card_asks_the_one_door(self):
        assert "/api/task-setup/bench-grid" in self._src()

    def test_it_sends_the_axes_THE_ROWS_WERE_PAINTED_FROM(self):
        """One source for both, or they disagree.

        `renderMachine` takes the task as an ARGUMENT and is called with
        the handover object in handover mode; reading the module's `_task`
        instead described a different object than the rows above it.  And
        the axes must be the in-memory model's, not the file's, so the
        list tracks typing rather than the last save.
        """
        src = self._src()
        rm = src.split("function renderMachine", 1)[1]
        assert "scheduleFitRefresh(bench)" in rm, (
            "the list must be refreshed from the same `bench` the rows "
            "were painted from, not from a module global")
        body = src.split("async function refreshFit", 1)[1]
        assert "_fitBench" in body

    def test_it_stays_quiet_where_there_is_no_description(self):
        """Handover and empty modes have no `task.json`, and the door
        reads one -- so a request there could only 400."""
        src = self._src()
        body = src.split("async function refreshFit", 1)[1].split(
            "function paintFit", 1)[0]
        assert '_mode !== "description"' in body

    def test_a_stale_answer_is_dropped(self):
        """Typing outruns the network; an earlier reply landing after a
        later one would paint a grid for axes that no longer exist."""
        src = self._src()
        assert "_fitSeq" in src and "seq !== _fitSeq" in src

    def test_the_request_is_debounced(self):
        """Every keystroke repaints the rows; the server answer is worth
        one request per PAUSE.

        Asserting the name ``_fitTimer`` appears was the first version of
        this pin, and it survived deleting the `setTimeout` -- the name
        still occurs in its own declaration.  A presence check cannot tell
        a working debounce from a dead one, which is the same fault that
        let `payload.info` stay green over a bug.  So this reads the
        scheduler's BODY.
        """
        src = self._src()
        # To the NEXT function, not to the first `}` -- the body holds a
        # `{}` literal, and splitting on a brace truncated before the line
        # under test, so the pin reported on its own arithmetic.
        body = src.split("function scheduleFitRefresh", 1)[1].split(
            "async function refreshFit", 1)[0]
        assert "setTimeout(refreshFit" in body, (
            "scheduleFitRefresh must DEFER the call, not make it")
        assert "clearTimeout" in body, (
            "and cancel the pending one, or every keystroke still fires")

    def test_a_failed_fetch_hides_the_list_and_keeps_the_card(self):
        """`loadSweepChoices`' recorded lesson: a card that cannot get a
        nicety shows what it has.  Only a card that cannot get its
        SUBSTANCE may refuse -- and this list is not the substance."""
        src = self._src()
        tail = src.split("async function refreshFit", 1)[1].split(
            "function paintFit", 1)[0]
        assert tail.count("host.hidden = true") >= 2, (
            "a failed or unparseable answer must hide the list, never "
            "throw out of renderMachine and strand the card")

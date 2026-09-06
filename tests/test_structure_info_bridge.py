"""The metadata bridge — a run's ``info`` store, end to end.

`archive/2026-09-01-structure-info-plan.md` § 5, settled 2026-08-30.  ``info`` is a
structure's free store (`web/molview.md` § 8.4a): a dict of key -> value
that DESCRIBES a structure without being part of it.  § 8.4a states it
"rides ``installMolecule`` in and ``exportFile`` out"; this file pins the
chain that makes that true, at every link:

  1. **The composer** — ``parse.dirs.run_info.run_info_for_dir``: one
     answer to *what does this run directory say about itself*, so the
     two doors that ask cannot come to disagree.
  2. **The load door** — ``/api/build/load`` applies a stated ``info``
     on the text branch (the one shape with no document behind it), and
     answers the store inside the canonical ``structure`` envelope on
     every branch.
  3. **The results adapter** — ``/api/watch/load`` answers the block from
     all THREE of its builders, upload included.
  4. **The browser** — ``structureFromServer`` reads the store from the
     canonical envelope, ``requestBodyFor`` sends it, and the trajectory
     page holds it across rebuilds and hands it back on every one.

WHY LINK 4 IS PINNED BY SOURCE AND NOT BY PRESENCE.  The read on the way
in asked for a FLAT ``payload.info``, which no route has ever sent, so
every structure arrived with an empty store at HTTP 200 — and the pin
that was supposed to catch it asserted the string ``payload.info``
appeared in the file, which the broken line satisfied perfectly.  The
pins below name the envelope the value actually arrives in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests._node_esm import run_node

_LIB = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static" / "lib"

_DECK = """SystemLabel Relax
MeshCutoff 250.0 Ry
PAO.BasisSize DZP
XC.functional GGA
XC.authors PBE
"""

_XYZ_3 = "".join(
    "3\n"
    f"Iteration {i} Energy {-76.41 + i * 0.001:.6f}\n"
    f"O 0 0 {i * 0.01:.4f}\nH 0.957 0 0\nH -0.239 0.927 0\n"
    for i in range(3)
)


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Watch's JSON-path mode constrains reads to the picker roots; point
    them at tmp so the test's run dir is loadable (same helper shape as
    ``test_atom_metadata_results_bridge``)."""
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset())
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),))
    diagnostics.set_capabilities(caps)


# --------------------------------------------------------------------- #
#  1. The composer -- one answer to "what does this directory say"       #
# --------------------------------------------------------------------- #

class TestTheComposer:

    def test_a_deck_becomes_the_calculation_key(self, tmp_path):
        from molbuilder.parse.dirs.run_info import run_info_for_dir
        (tmp_path / "Relax.fdf").write_text(_DECK)
        info = run_info_for_dir(tmp_path)
        assert set(info) == {"calculation"}, (
            "today the block has exactly one key; a new metadata category "
            "is a new key HERE and nowhere else")
        assert info["calculation"]["contract"]["basis_size"] == "DZP"
        assert info["calculation"]["source"] == "Relax.fdf"

    def test_nothing_to_say_is_none_not_an_empty_dict(self, tmp_path):
        """``None`` reads like its two siblings on the same response
        (``atom_metadata``, ``periodicity``): absent when there is nothing
        to say.  An empty dict would be a store the viewer then holds."""
        from molbuilder.parse.dirs.run_info import run_info_for_dir
        assert run_info_for_dir(tmp_path) is None
        assert run_info_for_dir(None) is None

    def test_the_results_door_asks_the_composer_not_the_extractor(
            self, client, tmp_path, monkeypatch):
        """``/api/results/contract`` answers the composer's ``calculation``
        key, so the structure inspector's door and the trajectory load
        door cannot disagree about what a directory records."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        run = tmp_path / "run"
        run.mkdir()
        (run / "Relax.fdf").write_text(_DECK)
        (run / "Relax.xyz").write_text("1\n\nH 0 0 0\n")

        d = client.get("/api/results/contract",
                       query_string={"path": str(run / "Relax.xyz")}).get_json()
        assert d["ok"] is True

        from molbuilder.parse.dirs.run_info import run_info_for_dir
        assert d["calculation"] == run_info_for_dir(run)["calculation"]


# --------------------------------------------------------------------- #
#  2. The load door -- the store rides installMolecule IN                #
# --------------------------------------------------------------------- #

class TestTheLoadDoorTakesIt:

    def test_a_stated_store_lands_on_the_text_branch(self, client):
        """A text load parses a file, and a file being parsed carries no
        store -- so a host that knows one states it, and it comes back on
        the structure."""
        d = client.post("/api/build/load", json={
            "text": "2\n\nH 0 0 0\nH 0 0 0.74\n",
            "filename": "probe.xyz",
            "info": {"calculation": {"engine": "siesta"}},
        }).get_json()
        assert d["ok"] is True
        assert d["structure"]["info"] == {"calculation": {"engine": "siesta"}}

    def test_a_non_dict_store_is_refused_not_dropped(self, client):
        """Dropping it silently is how the labels went missing for months
        at HTTP 200 (`_shared._struct_from_envelope`'s own note)."""
        r = client.post("/api/build/load", json={
            "text": "2\n\nH 0 0 0\nH 0 0 0.74\n",
            "filename": "probe.xyz",
            "info": ["not", "a", "dict"],
        })
        assert r.status_code == 400
        assert "info" in r.get_json()["error"]

    def test_a_pair_on_disk_brings_its_store_back(
            self, client, tmp_path, monkeypatch):
        """THE ROUND TRIP § 8.4a claims.  A saved pair carries the store in
        its ``.molstruct.json``; re-opening it must answer the same store,
        inside the canonical envelope -- which is where the browser reads
        it from."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec

        s = Structure(elements=["H", "H"],
                      positions=[[0, 0, 0], [0, 0, 0.74]])
        s.info = {"calculation": {"engine": "siesta",
                                  "contract": {"basis_size": "DZP"}}}
        StructureCodec().write(s, tmp_path / "pair.xyz")
        side = json.loads((tmp_path / "pair.molstruct.json").read_text())
        assert side["info"] == s.info, "the store must reach the sidecar"

        d = client.post("/api/build/load",
                        json={"path": str(tmp_path / "pair.xyz")}).get_json()
        assert d["ok"] is True, d
        assert d["structure"]["info"] == s.info


# --------------------------------------------------------------------- #
#  3. The results adapter -- one composer, three builders                #
# --------------------------------------------------------------------- #

class TestWatchLoadAnswersTheBlock:

    def test_a_run_directory_answers_its_recorded_contract(
            self, client, tmp_path, monkeypatch):
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        run = tmp_path / "run"
        run.mkdir()
        (run / "Relax.fdf").write_text(_DECK)
        (run / "run_geom_optim.xyz").write_text(_XYZ_3)

        d = client.post("/api/watch/load",
                        json={"path": str(run)}).get_json()
        assert d["ok"] is True
        assert d["info"]["calculation"]["contract"]["basis_size"] == "DZP"

    def test_a_run_with_no_deck_says_so(self, client, tmp_path, monkeypatch):
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        run = tmp_path / "bare"
        run.mkdir()
        (run / "run_geom_optim.xyz").write_text(_XYZ_3)

        d = client.post("/api/watch/load",
                        json={"path": str(run)}).get_json()
        assert d["ok"] is True
        assert "info" in d and d["info"] is None

    def test_pointing_at_the_log_itself_finds_the_deck_beside_it(
            self, client, tmp_path, monkeypatch):
        """The single-FILE builder searches the file's own directory --
        the same rule the directory builder uses, now spelled once."""
        _register_tmp_as_picker_root(tmp_path, monkeypatch)
        run = tmp_path / "single"
        run.mkdir()
        (run / "Relax.fdf").write_text(_DECK)
        log = run / "run_geom_optim.xyz"
        log.write_text(_XYZ_3)

        d = client.post("/api/watch/load", json={"path": str(log)}).get_json()
        assert d["ok"] is True
        assert d["info"]["calculation"]["contract"]["basis_size"] == "DZP"

    def test_an_upload_states_that_it_has_nothing_to_say(self, client):
        """One route, one response shape: the upload builder has no run
        directory, and answers so in the same three fields the other two
        answer -- rather than leaving a caller to notice they are missing.
        (Omission means KEEP on this route: the browser's APPLY rule is
        keep-on-undefined, which is what lets the 200 ms poll re-send the
        frames without re-sending the metadata.)"""
        import io
        d = client.post("/api/watch/load", data={
            "file": (io.BytesIO(_XYZ_3.encode()), "run.xyz"),
        }, content_type="multipart/form-data").get_json()
        assert d["ok"] is True
        for field in ("info", "atom_metadata", "periodicity"):
            assert field in d, f"the upload builder must answer {field}"
            assert d[field] is None


# --------------------------------------------------------------------- #
#  4. The browser -- the store rides in, and survives a rebuild          #
# --------------------------------------------------------------------- #

def _src(rel: str) -> str:
    """The module's CODE, comments removed.  A pin that reads comments passes
    on the strength of a note describing the bug it guards against -- which is
    how two pins here were once written, and both survived the mutation that
    put the bug back."""
    src = (_LIB / rel).read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


REPO = Path(__file__).resolve().parents[1]
JOBS = _LIB / "molview" / "model-jobs.js"

#: `installMolecule` POSTs through `postJson`, so a stubbed `fetch` is what
#: makes the REQUEST BODY inspectable -- the thing link 4 is actually about.
#: It records every call and answers with a minimal valid structure.
_FETCH_STUB = """
globalThis.__sent = [];
globalThis.fetch = async (route, opts) => {
    globalThis.__sent.push({ route, body: JSON.parse(opts.body) });
    return { ok: true, status: 200,
             json: async () => ({ atoms: [{ element: "O", xyz: [0,0,0] }],
                                  structure: { info: { calculation: "relax" } } }) };
};
"""

_PRELUDE = f"""
const JOBS = await import({json.dumps(JOBS.resolve().as_uri())});
"""


def _run(snippet: str):
    return run_node([], _PRELUDE + snippet, globals_js=_FETCH_STUB)


class TestTheBrowserSide:
    """Link 4, RUN.

    CONVERTED 2026-09-06 (`plans/plan.md` § 5h, cluster 3).  These read the
    module as TEXT until today, and this file's own header records why that
    was never enough: the read on the way in asked for a flat `payload.info`,
    which no route has ever sent, so every structure arrived with an empty
    store at HTTP 200 -- *and the pin that was supposed to catch it asserted
    the string `payload.info` appeared in the file, which the broken line
    satisfied perfectly.*

    The answer to a pin that missed a bug is not a narrower pin.  One of them
    had reached `src.count('state.fileState.info         = null;') == 2` --
    an exact line, nine embedded spaces included, counted twice.  A reformat
    breaks it; the defect it guards walks past it.  These call the functions.
    """

    def test_the_store_arrives_from_the_envelope_and_only_from_there(self):
        """``payload.structure`` IS the structure's own dict, and ``info`` is
        a field of a Structure, so it arrives there and nowhere else.

        MUTATION THIS MUST FAIL AGAINST: read `payload.info` instead -- which
        is the ORIGINAL BUG, and which the retired pin passed through.
        """
        out = _run("""
        console.log(JSON.stringify({
            envelope: JOBS.structureFromServer(
                { atoms: [{ element: "O", xyz: [0,0,0] }],
                  structure: { info: { calculation: "relax" } } }).structure.info,
            flat: JOBS.structureFromServer(
                { atoms: [{ element: "O", xyz: [0,0,0] }],
                  info: { calculation: "SHOULD-BE-IGNORED" } }).structure.info,
        }));""")
        assert out["envelope"] == {"calculation": "relax"}, (
            "the store must be read from the canonical envelope")
        assert out["flat"] == {}, (
            "a FLAT payload.info is a key no route sends -- reading it is the "
            "bug that shipped, and it must stay unread")

    def test_a_stated_store_rides_the_request_out(self):
        """§ 8.4a's *'it rides installMolecule in'* -- asked of the request
        the door actually posts, not of the line that builds it.

        The TEXT shape, because that is the one this field exists for: a
        `path` load reads the store out of the pair's `.molstruct.json` and a
        structure put back carries it in its envelope, so text is the only
        shape with no document behind it and the host must state it.
        """
        out = _run("""
        // `handed` is the viewer the door installs INTO -- stubbed to the
        // three calls the load path makes, so the REQUEST is what is tested.
        const handed = { put() {}, recordFirstState() {}, announce() {} };
        const install = JOBS.createLoad(handed);
        await install({ text: "1\\nx\\nO 0 0 0\\n", filename: "a.xyz",
                        info: { calculation: "vibration" } });
        await install({ text: "1\\nx\\nO 0 0 0\\n", filename: "b.xyz",
                        info: {} });
        console.log(JSON.stringify({
            route: globalThis.__sent[0].route,
            info:  globalThis.__sent[0].body.info,
            emptyIsAbsent: "info" in globalThis.__sent[1].body,
        }));""")
        assert out["route"] == "/api/build/load"
        assert out["info"] == {"calculation": "vibration"}, (
            "installMolecule dropped the store the host stated, so the server "
            "never learns what describes the structure it is being handed")
        assert out["emptyIsAbsent"] is False, (
            "an empty store must not be sent -- absent and 'described with "
            "nothing' are different answers")

    # ---------------------------------------------------------------- #
    #  NOT CONVERTED, and why -- `plans/plan.md` § 5h                    #
    # ---------------------------------------------------------------- #

    def test_the_trajectory_holds_the_store_across_rebuilds(self):
        """The page rebuilds its viewer on every poll, so the store lives in
        ``fileState`` beside its two neighbours and is handed back on every
        rebuild -- not attached to the viewer once after a load.

        **STILL A SOURCE PIN, deliberately, and it is BROWSER work not node
        work** (2026-09-06).  The other three claims in this class became node
        tests because their functions are exported and pure.  These four are
        not: the aliasing runs inside `mountInspector` through
        `inspectorLifecycle.alias`, and the resets and the APPLY branch are in
        `transition()`, a reducer that only exists once a viewer is mounted.
        No harness mounts one headless, and inventing one to reach four lines
        would cost more than the Playwright test that is the real answer.
        Reclassified from *node* to *browser* in
        `tools/classify_source_reads.py`, so the work list says so.

        What DID change: the two assertions that measured whitespace now
        match on structure.  `src.count('state.fileState.info         = null;')`
        counted an exact line, nine embedded spaces included -- it fired on a
        reformat and stayed green through the defect.  Same coverage, one less
        way to be wrong for no reason.
        """
        src = _src("trajectory/core.js")
        assert re.search(r'alias\(\s*"info"\s*,\s*"fileState"\s*\)', src), (
            "the store is per-file state, like atomMetadata/periodicity")
        assert len(re.findall(r'state\.fileState\.info\s*=\s*null', src)) == 2, (
            "both resets (LOADING and IDLE) must clear it, beside the two "
            "fields they already clear")
        assert re.search(r'if\s*\(\s*payload\.info\s*!==\s*undefined\s*\)', src), (
            "APPLY must KEEP on undefined -- the 200 ms poll omits the block, "
            "and an always-present one would clear it every tick")
        install = src.split("_mvdata().installMolecule({")[1].split("});")[0]
        assert "info:" in install, (
            "the store must ride the ONE entrance, so the history anchor is "
            "recorded with it and a rebuild cannot drop it")


    def test_an_export_carries_the_store_out(self):
        """The inverse: what the Metadata pane shows is what the pair carries.

        Also pins the absence rule -- an EMPTY store is not written at all,
        rather than written as `{}`, which is what keeps a structure that was
        never described from claiming it was described with nothing.
        """
        out = _run("""
        const withInfo = { elements: ["O"], annotations: [{labels: []}],
                           periodicity: null, info: { calculation: "relax" } };
        const without  = { elements: ["O"], annotations: [{labels: []}],
                           periodicity: null, info: {} };
        console.log(JSON.stringify({
            carried: JOBS.structureForServer(withInfo, [[0,0,0]]).info,
            emptyIsAbsent: "info" in JOBS.structureForServer(without, [[0,0,0]]),
        }));""")
        assert out["carried"] == {"calculation": "relax"}, (
            "an exported pair would lose what the Metadata pane shows")
        assert out["emptyIsAbsent"] is False, (
            "an empty store must be ABSENT, not written as {}")

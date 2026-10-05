"""The metadata bridge — a run's ``info`` store, end to end.

`archive/2026-09-01-structure-info-plan.md` § 5, settled 2026-08-30.  ``info`` is a
structure's free store (`web/molview.md` § 8.4a): a dict of key -> value
that DESCRIBES a structure without being part of it.  § 8.4a states it
"rides ``installMolecule`` in and ``exportFile`` out"; this file pins the
chain that makes that true, at every link:

  1. **The composer** — ``parse.dirs.run_info.run_info``: one
     answer to *what does this run directory say about itself*, so the
     two doors that ask cannot come to disagree.
  2. **The load door** — ``/api/build/load`` answers the store inside the
     canonical ``structure`` envelope, off the pair's ``.molstruct.json``.
     (A stated ``info`` beside a text went with the text branch's side
     blocks on 2026-09-25 -- nothing sent one; plan § 5q D15.)
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
    them at the test's tree so its run folder is loadable."""
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

    # `test_a_deck_becomes_the_calculation_key` retired 2026-10-04 (W56
    # review, ruling 1): it laid a deck in a bare folder for the composer to
    # find; the composer is handed the run's own deck now, and both its keys
    # are pinned on the measured runs below.

    def test_a_finished_run_answers_both_keys_at_both_doors(
            self, client, monkeypatch):
        """A run directory says two things about the structure it left
        (`model/parse.md` § 5b, § 5b.1): the level of theory its deck
        stated, and what the run did to the geometry -- read from the file
        the viewer has open, so the two doors that ask answer one record:
        the trajectory load, of the file it opened, and the structure
        inspector's, of the file the Results tab opens in the structure's
        folder (`runs.openable`).

        WHY API-LEVEL: a measured fixture, read where it was measured --
        the H2 relaxation under `tests/fixtures/siesta_relax`; the road
        that produced it is the SIESTA e2e test."""
        from molbuilder.parse.contract import contract_of
        fixtures = Path(__file__).resolve().parent / "fixtures"
        _register_tmp_as_picker_root(fixtures, monkeypatch)
        run = fixtures / "siesta_relax" / "01_relax" / "run-0"

        loaded = client.post("/api/watch/load",
                             json={"path": str(run)}).get_json()
        assert loaded["ok"] is True, loaded
        asked = client.get("/api/results/contract",
                           query_string={"path": str(run / "H2.XV")}
                           ).get_json()
        assert asked["ok"] is True, asked

        record = loaded["info"]["relaxation"]
        assert record["source"] == Path(loaded["path"]).name, (
            "the record is of the file on screen", record)
        assert record["converged"] is True, record
        # THE ATOM IT HELD, as the run's own output states it -- SIESTA's
        # constraints echo, `position 1` (`model/parse.md` § 5.3).
        assert record["held_atom_idxs"] == [0], record
        assert asked["relaxation"] == record
        assert (asked["calculation"] == loaded["info"]["calculation"]
                == contract_of(run / "H2_01_relax.fdf"))

    def test_nothing_to_say_is_none_not_an_empty_dict(self, tmp_path):
        """``None`` reads like its two siblings on the same response
        (``atom_metadata``, ``periodicity``): absent when there is nothing
        to say.  An empty dict would be a store the viewer then holds."""
        from molbuilder.parse.dirs.run_info import run_info
        assert run_info() is None
        assert run_info(deck=tmp_path / "absent.fdf") is None

    # `test_the_results_door_asks_the_composer_not_the_extractor` retired
    # 2026-10-04 (W56 review): it laid a deck and a structure in a bare
    # folder and compared the route with the very function the route
    # calls; both doors answering one record of one run is
    # `test_a_finished_run_answers_both_keys_at_both_doors`.


# --------------------------------------------------------------------- #
#  2. The load door -- the store rides installMolecule IN                #
# --------------------------------------------------------------------- #

class TestTheLoadDoorTakesIt:

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

    # Retired 2026-10-04 (plan B11/B12): a run folder a test laid by hand -- a
    # deck and a trajectory written beside it -- loaded as a directory.  A
    # folder no calculation claims holds no run of ours; what a run of ours
    # declared reaches the viewer through the run door, on the road
    # (`process/testing.md` § 6).

    def test_a_flat_run_reads_its_own_deck_and_nothing_else(
            self, client, monkeypatch):
        """Each run of a flat calculation takes what it declared -- its labels
        and its box -- from ITS OWN deck, both from that one deck: the Results
        load of the flat H2 run shows atom 0 held in the isolated 10 Å box,
        and each of the folder's two stages reads its own `.fdf` (user,
        2026-10-04: *"make sure that it does make each run sees its own .fdf
        and take information from there rather than mixing"*).

        WHY API-LEVEL: a measured fixture, read where it was measured --
        `tests/fixtures/siesta_flat_h2` (its README)."""
        import numpy as np

        from molbuilder.runs import declared, run_of
        fixtures = Path(__file__).resolve().parent / "fixtures"
        _register_tmp_as_picker_root(fixtures, monkeypatch)
        flat = fixtures / "siesta_flat_h2"

        d = client.post("/api/watch/load", json={"path": str(flat)}).get_json()
        assert d["ok"] is True, d
        held = json.loads(d["atom_metadata"])["regions"]["frozen_atoms"]
        assert held == [0], d["atom_metadata"]
        box = d["periodicity"]
        assert box["axis_kind"] == ["isolated"] * 3, box
        assert box["engine_offset"] == [0.0, 0.0, 0.0], box
        np.testing.assert_allclose(box["cell"], np.eye(3) * 10.0, atol=1e-4)
        # ITS OWN OUTPUT NAMES ITS COMPANIONS AND ITS HELD ATOMS: the history
        # SIESTA wrote under the label the `.out` states, and the atom its
        # echo says the engine held (`model/parse.md` § 5.3).
        rt = d["data"]["runtime_info"]
        assert rt.get("mdnc_source") == "H2.MD.nc", rt
        assert rt.get("frozen_atoms") == [0], rt
        # ITS OWN DECK STATES ITS CONTRACT, though the folder holds two decks
        # (ruling 1, 2026-10-04: a folder search found two and said none).
        calc = d["info"]["calculation"]
        assert calc["source"] == "H2_01_coarse.fdf", calc
        assert calc["contract"]["basis_size"] == "DZP", calc

        # EACH RUN, ITS OWN DECK: the coarse run that wrote the output, and
        # the medium stage prepped beside it in the same folder.
        assert (declared(run_of(flat / "H2_01_coarse-run0.out")).deck.name
                == "H2_01_coarse.fdf")
        assert (declared(run_of(flat, stage="02_medium")).deck.name
                == "H2_02_medium.fdf")

    # `test_pointing_at_the_log_itself_finds_the_deck_beside_it` retired
    # 2026-10-04 (W56 review): a deck and a trajectory written into a bare
    # folder -- the state the note above retires.  A run of ours, loaded by
    # its own file, is the flat H2 test above.

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

    def test_the_store_arrives_on_the_structure(self):
        """§ 8.4a's *"it rides installMolecule in"* -- asked of what the viewer
        is HANDED, not of the key that used to carry it.

        The store travels INSIDE the envelope (`payload.structure.info`), which
        is where the read has always looked -- this file's header records the
        bug from asking for a flat `payload.info` that no route ever sent. As
        of 2026-09-07 that is the only way in: the `info` request key existed
        for the TEXT branch, whose last real caller (the Results trajectory
        tab) now hands over an envelope the server assembled, so there is no
        longer a shape with a structure and no room for its store.

        The assertion is the one that survives a mechanism change: the viewer
        ends up holding what the server said, and the request carried nothing
        beside the structure.
        """
        out = _run("""
        let installed = null;
        const handed = { put(structure) { installed = structure; },
                         recordFirstState() {}, announce() {} };
        const install = JOBS.createLoad(handed);
        await install({ structure: { elements: ["O"], positions: [[0, 0, 0]],
                                     info: { calculation: "vibration" } } });
        console.log(JSON.stringify({
            route:     globalThis.__sent[0].route,
            sentKeys:  Object.keys(globalThis.__sent[0].body).sort(),
            heldInfo:  installed && installed.info,
        }));""")
        assert out["route"] == "/api/build/load"
        assert out["sentKeys"] == ["structure"], (
            "the envelope goes over alone; a side-block beside it is the shape "
            "that let a structure and its store disagree")
        assert out["heldInfo"] == {"calculation": "relax"}, (
            "the viewer must end up holding the store the SERVER answered "
            "with -- read off `payload.structure.info`, the one place it lives")

    # `test_the_trajectory_holds_the_store_across_rebuilds` STOOD HERE and is
    # retired (2026-09-07).  It was the last source pin in the repo: it grepped
    # `trajectory/core.js` for an `alias("info", ...)` call, counted
    # `state.fileState.info = null` twice, and asserted the literal `info:`
    # appeared inside the `installMolecule({...})` call.
    #
    # Its subject is gone.  The tab no longer passes `info` to
    # `installMolecule` at all -- the server assembles frame 0 as an envelope
    # with the store already on it -- so the pin now fails on a change that
    # made the code MORE correct, which is exactly the failure mode a pin has.
    # The behaviour it reached for is covered above, against what the viewer
    # actually ends up holding.


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

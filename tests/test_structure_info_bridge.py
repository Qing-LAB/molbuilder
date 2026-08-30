"""The metadata bridge — a run's ``info`` store, end to end.

`plans/structure-info-plan.md` § 5, settled 2026-08-30.  ``info`` is a
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
    """The module's CODE, comments removed.  A pin that reads comments
    passes on the strength of a note describing the bug it is guarding
    against -- which is how the first two pins here were written, and
    they both survived the mutation that put the bug back."""
    src = (_LIB / rel).read_text()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


class TestTheBrowserSide:

    def test_the_store_is_read_from_the_canonical_envelope(self):
        """``payload.structure`` IS the structure's own dict, and ``info``
        is a field of a Structure, so it arrives there and nowhere else --
        the same rule this reader already applies to ``title``."""
        src = _src("molview/model-jobs.js")
        assert "payload.structure.info" in src, (
            "structureFromServer must read the store from the canonical "
            "envelope; a flat payload.info is a key no route sends")
        assert not re.search(r"payload\.info\b", src), (
            "the flat read is the bug: it satisfied a presence pin while "
            "every structure arrived with an empty store")

    def test_the_load_door_sends_a_stated_store(self):
        src = _src("molview/model-jobs.js")
        assert "body.info = input.info" in src, (
            "installMolecule must forward the store the host stated -- "
            "§ 8.4a's 'it rides installMolecule in'")

    def test_an_export_still_carries_the_store_out(self):
        assert "out.info = structure.info" in _src("molview/model-jobs.js"), (
            "an exported pair would lose what the Metadata pane shows")

    def test_the_trajectory_holds_the_store_across_rebuilds(self):
        """The page rebuilds its viewer on every poll, so the store lives
        in ``fileState`` beside its two neighbours and is handed back on
        every rebuild -- not attached to the viewer once after a load."""
        src = _src("trajectory/core.js")
        assert 'alias("info",         "fileState");' in src, (
            "the store is per-file state, like atomMetadata/periodicity")
        assert src.count("state.fileState.info         = null;") == 2, (
            "both resets (LOADING and IDLE) must clear it, beside the two "
            "fields they already clear")
        assert "if (payload.info !== undefined)" in src, (
            "APPLY must KEEP on undefined -- the 200 ms poll omits the "
            "block, and an always-present one would clear it every tick")
        install = src.split("_mvdata().installMolecule({")[1].split("});")[0]
        assert "info:" in install, (
            "the store must ride the ONE entrance, so the history anchor "
            "is recorded with it and a rebuild cannot drop it")

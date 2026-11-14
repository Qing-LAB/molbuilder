"""The metadata bridge — a run's ``info`` store, end to end.

`archive/2026-09-01-structure-info-plan.md` § 5, settled 2026-08-30.  ``info`` is a
structure's free store (`web/molview.md` § 8.4a): a dict of key -> value
that DESCRIBES a structure without being part of it.  § 8.4a states it
"rides ``installMolecule`` in and ``exportFile`` out"; this file pins two
links of the chain that makes that true:

  1. **The composer** — ``parse.dirs.run_info.run_info``: one
     answer to *what does this run directory say about itself*, so the
     two doors that ask cannot come to disagree.
  4. **The browser** — a saved structure's store, read by the server from
     its sidecar, arrives inside the structure the viewer holds; the
     viewer's export carries it out; a structure put back keeps it.

WHY LINK 4 IS PINNED BY RUNNING IT.  The read on the way in asked for a
FLAT ``payload.info``, which no route has ever sent, so every structure
arrived with an empty store at HTTP 200 — and a pin asserting the string
``payload.info`` appeared in the file was satisfied by the broken line.
So it runs on the app's own routes: a pair opened by its path, as a tab
opens one, and what the server answered is what the viewer must hold.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served


# --------------------------------------------------------------------- #
#  1. The composer -- one answer to "what does this directory say"       #
# --------------------------------------------------------------------- #

class TestTheComposer:


    def test_nothing_to_say_is_none_not_an_empty_dict(self, tmp_path):
        """``None`` when the run says nothing: absent when there is nothing
        to say.  An empty dict would be a store the viewer then holds."""
        from molbuilder.parse.dirs.run_info import run_info
        assert run_info() is None
        assert run_info(deck=tmp_path / "absent.fdf") is None


# --------------------------------------------------------------------- #
#  4. The browser -- the store rides in, and out                         #
# --------------------------------------------------------------------- #

#: The page that hosts MolView, as a test plays it: it hands the viewer the
#: one reader of a server's answer (`support/molview_host.mjs`).
HOST = Path(__file__).resolve().parent / "support" / "molview_host.mjs"

#: What a person's script wrote into the store (`Structure.set_info`, the
#: Python half of the host's `data.info.set`): one entry of its own.
NOTE = ("note", {"drawn": "by hand", "relaxed": False})


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the viewer (`support.live_server`)."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture(scope="module")
def pairs(isolated_projects_root_module):
    """The road's H2 saved as the road saves a structure
    (`support.road.structure_saved`), twice: once with :data:`NOTE` in its
    store -- the sidecar carries it -- and once never described."""
    from support.road import H2, road_structure, structure_saved
    root = isolated_projects_root_module
    described = road_structure(H2)
    described.set_info(*NOTE)
    return {
        "described": str(root / structure_saved(
            root, described, at="P/structure/described.xyz")),
        "never described": str(root / structure_saved(
            root, H2, at="P/structure/plain.xyz")),
    }


def _run(app_at, pairs, snippet: str):
    """The test's script on the page, the app's own routes answering
    (`_node_esm.served`), each request kept in ``__calls``."""
    return run_node([], f"""
const {{ createModel }} = await import({json.dumps(HOST.as_uri())});
const PAIRS = {json.dumps(pairs)};
""" + snippet, globals_js=served(app_at))


class TestTheBrowserSide:
    """Link 4, RUN on the app's own routes: a pair opened as a tab opens one,
    the viewer's own export, and a structure put back as a tab puts one."""

    @pytest.mark.parametrize("which", ["described", "never described"])
    def test_the_store_rides_in_with_the_file_and_out_with_its_export(
            self, app_at, pairs, which):
        """§ 8.4a: "It rides `installMolecule` in and `exportFile` out, so
        what the pane shows is exactly what the `.xyz + .molstruct.json`
        pair will carry".  A saved structure's store is its sidecar's, read
        by the server; the viewer holds what the server answered -- inside
        the structure, its one place -- and its export carries it out.  A
        store never written is not written out either: an export of it
        claims no description, not an empty one.

        MUTATION THIS MUST FAIL AGAINST: the store read beside the structure
        (a flat ``payload.info``, which no route sends) -- the ORIGINAL BUG.
        """
        out = _run(app_at, pairs, """
        const m = createModel({});
        await m.installMolecule({ path: PAIRS[__WHICH__] });
        const file = m.exportFile();
        console.log(JSON.stringify({
            held: m.getStructure().info,
            exported: "info" in file.structure ? file.structure.info : "absent",
        }));""".replace("__WHICH__", json.dumps(which)))
        if which == "described":
            assert out["held"] == dict([NOTE]), (
                "the viewer must hold the store the server read from the "
                f"sidecar: {out['held']}")
            assert out["exported"] == dict([NOTE]), (
                "an exported pair would lose what the Metadata pane shows")
        else:
            assert out["held"] == {}, out["held"]
            assert out["exported"] == "absent", (
                "an empty store must be ABSENT from the export, not written "
                "as {}")

    def test_a_structure_put_back_keeps_its_store(self, app_at, pairs):
        """`requestBodyFor`: a structure put back is "`exportFile`'s exact
        inverse" -- what a tab puts back, as the structure card puts back
        its draft -- so its store comes back with it.  (That the envelope
        goes over alone is `test_molview_model`'s.)"""
        out = _run(app_at, pairs, """
        const shown = createModel({});
        await shown.installMolecule({ path: PAIRS.described });
        const file = shown.exportFile();
        const back = createModel({});
        await back.installMolecule({ structure: file.structure });
        console.log(JSON.stringify({ held: back.getStructure().info }));""")
        assert out["held"] == dict([NOTE]), (
            f"a structure put back lost its store: {out['held']}")

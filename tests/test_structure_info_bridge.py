"""The metadata bridge — a run's ``info`` store, end to end.

`archive/2026-09-01-structure-info-plan.md` § 5, settled 2026-08-30.  ``info`` is a
structure's free store (`web/molview.md` § 8.4a): a dict of key -> value
that DESCRIBES a structure without being part of it.  § 8.4a states it
"rides ``installMolecule`` in and ``exportFile`` out"; this file pins two
links of the chain that makes that true:

  1. **The composer** — ``parse.dirs.run_info.run_info``: one
     answer to *what does this run directory say about itself*, so the
     two doors that ask cannot come to disagree.
  4. **The browser** — ``structureFromServer`` reads the store from the
     canonical envelope, ``requestBodyFor`` sends it, and the trajectory
     page holds it across rebuilds and hands it back on every one.

WHY LINK 4 IS PINNED BY RUNNING IT.  The read on the way in asked for a
FLAT ``payload.info``, which no route has ever sent, so every structure
arrived with an empty store at HTTP 200 — and a pin asserting the string
``payload.info`` appeared in the file was satisfied by the broken line.
The pins below name the envelope the value actually arrives in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


from tests._node_esm import run_node

_LIB = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static" / "lib"


# --------------------------------------------------------------------- #
#  1. The composer -- one answer to "what does this directory say"       #
# --------------------------------------------------------------------- #

class TestTheComposer:


    def test_nothing_to_say_is_none_not_an_empty_dict(self, tmp_path):
        """``None`` reads like its two siblings on the same response
        (``atom_metadata``, ``periodicity``): absent when there is nothing
        to say.  An empty dict would be a store the viewer then holds."""
        from molbuilder.parse.dirs.run_info import run_info
        assert run_info() is None
        assert run_info(deck=tmp_path / "absent.fdf") is None


# --------------------------------------------------------------------- #
#  4. The browser -- the store rides in, and survives a rebuild          #
# --------------------------------------------------------------------- #

def _src(rel: str) -> str:
    """The module's CODE, comments removed.  A pin that reads comments passes
    on the strength of a note describing the bug it guards against."""
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
    """Link 4, RUN: these call the functions.
    """

    def test_the_store_arrives_from_the_envelope_and_only_from_there(self):
        """``payload.structure`` IS the structure's own dict, and ``info`` is
        a field of a Structure, so it arrives there and nowhere else.

        MUTATION THIS MUST FAIL AGAINST: read `payload.info` instead -- which
        is the ORIGINAL BUG.
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
        is HANDED.

        The store travels INSIDE the envelope (`payload.structure.info`), the
        only way in.

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

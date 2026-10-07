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


from tests._node_esm import run_node

_LIB = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static" / "lib"


# --------------------------------------------------------------------- #
#  1. The composer -- one answer to "what does this directory say"       #
# --------------------------------------------------------------------- #

class TestTheComposer:

    # `test_a_deck_becomes_the_calculation_key` retired 2026-10-04 (W56
    # review, ruling 1): it laid a deck in a bare folder for the composer to
    # find; the composer is handed the run's own deck now.  Both its keys,
    # answered alike at both doors, are asked of a relaxation run on the road
    # with the real SIESTA (`tests/test_siesta_relax_run_e2e.py`, moved there
    # 2026-10-06; `process/testing.md` § 6).

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
    # `test_siesta_relax_run_e2e.py`'s.


# Retired 2026-10-04 (user: "any fucking faking tests should be retired"):
# 2 tests here uploaded a geomeTRIC trajectory typed by hand, or
# saved a structure carrying the record of a calculation nothing ran (`process/testing.md` § 6).


# --------------------------------------------------------------------- #
#  3. The results adapter -- one composer, three builders                #
# --------------------------------------------------------------------- #

# What the load answers for a run of ours -- each run its own deck's labels
# and box -- is asked of a flat run made on the road with the real SIESTA
# (`tests/test_siesta_flat_run_e2e.py`, moved there 2026-10-06;
# `process/testing.md` § 6).  A run folder a test laid by hand was retired
# 2026-10-04 (plan B11/B12): a folder no calculation claims holds no run of
# ours.


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

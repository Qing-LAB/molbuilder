"""The view context (molview.md § 11.2b) — run, not described.

The REAL model and the REAL lane execute together against the app's own
routes and the real workspace, in one browser tab on Node's own storage
(`_node_esm.served`, `tab_storage`; `molview.md` § 13.1).  The one stand-in
is the sealed layer's pose pair -- the drawing library needs WebGL, which
node cannot run; its real half is vendor 3Dmol, pinned in
test_molview_3dmol_embed.py.  What these pin:

  1. looking is never changing — a lane write raises no badge and rewrites
     no draft but its own (the user's rule, verbatim: "not a change of
     data");
  2. the round trip — switches, style, frame, the ruler's on/off and camera
     come back;
  3. the match guard — a different structure gets the preferences but
     never the pose, the frame, or the selection — the same molecule
     listed in another order among them;
  4. selection ownership — the truth lane owns it where one exists, the
     ui lane carries it only for read-only viewers;
  5. ordering — the context applies AFTER a draft restore's view.reset(),
     not under it.

Each structure is a pair on disk, written as the road writes one, opened by
path.  A viewer reopened is a second session in the same browser tab, as a
page reloaded is.
"""
from __future__ import annotations

import json

import pytest

from support.molview_lane import prelude
from tests._node_esm import run_node, served, tab_storage


@pytest.fixture(scope="module")
def app_at(isolated_projects_root_module):
    """The app, serving: what answers the viewer and the workspace."""
    from support.live_server import serve
    with serve() as base:
        yield base


@pytest.fixture(scope="module")
def pairs(isolated_projects_root_module):
    """The structures the tests open, by name -- the road's own
    (`support.road`), each saved as the road saves one: its H2 and its water,
    each also as a three-frame set so a frame has somewhere to be
    (`Structure.with_frames`, the door the road's own frame sets use -- theirs
    move a transport bridge, which these have none of), its CO2 (O O C), as
    a three-frame set too, and the same molecule listed C O O (the order
    `CO2_HELD_FIRST` names), saved as a molecule of its own in the user
    order: the lane's match reads its elements and its frame count, not
    which atom is held."""
    import numpy as np
    from support.road import (CO2, CO2_HELD_FIRST, H2, WATER, road_structure,
                              structure_saved)
    root = isolated_projects_root_module

    def saved(name, s):
        return root / structure_saved(root, s, at=f"P/structure/{name}.xyz")

    def framed(spec):
        s = road_structure(spec)
        return s.with_frames(np.array([s.positions + 0.01 * f
                                       for f in range(3)]))
    order = CO2_HELD_FIRST["sorted_to_original"]
    coo = {"elements": [CO2["elements"][i] for i in order],
           "positions": [CO2["positions"][i] for i in order],
           "cell": CO2["cell"]}
    return {"two": saved("h2", H2), "three": saved("water", WATER),
            "two framed": saved("h2-frames", framed(H2)),
            "three framed": saved("water-frames", framed(WATER)),
            "O O C": saved("ooc", CO2),
            "O O C framed": saved("ooc-frames", framed(CO2)),
            "C O O": saved("coo", coo)}


def _run(app_at, pairs, snippet: str):
    return run_node([], prelude(pairs) + snippet,
                    globals_js=tab_storage() + served(app_at), storage=True)


def test_looking_is_never_changing(app_at, pairs):
    """A lane write raises no badge and rewrites no draft but its own -- the
    timeline cannot even see it (the module holds no reference to the
    history): after a look changes, the one write on the wire is the lane's
    draft, which reads back as the look."""
    out = _run(app_at, pairs, """
        const one = await session();
        await opened(one, "two");
        await settled();
        const before = writes().length;

        one.model.selection.setSwitch("showCell", true);
        one.model.view.set("style", "sphere");
        await settled();

        const sent = writes().slice(before).map((c) => c.body.data);
        const lane = await ws.readDraft("s:ui");
        console.log(JSON.stringify({
            sent, lane,
            badge: one.model.uncommitted,
            at: one.model.state_index,
        }));
        """)
    assert len(out["sent"]) == 1, (
        f"a look change wrote {len(out['sent'])} times — the draft and the "
        f"points are the truth's, and looking is not changing")
    assert out["sent"][0] == out["lane"], (
        "the one write was not the lane's own draft")
    assert out["badge"] is False
    assert out["at"] == 0


def test_the_round_trip_restores_how_you_were_looking(app_at, pairs):
    out = _run(app_at, pairs, """
        // A FRAME SET OPENS READ-ONLY (an editable viewer refuses one,
        // § 9.4) -- the Results tab's road, where a frame has somewhere to be.
        const RO = { mode: "readonly" };
        const one = await session({ pose: [1, 2, 3, 4], model: RO });
        await opened(one, "three framed");
        await settled();
        one.model.view.set("style", "sphere");
        one.model.selection.setSwitch("showAxis", true);
        one.model.selection.setSwitch("forceScale", 2);
        one.model.setCurrentFrame(2);
        one.model.measurement.setActive(true);
        await settled();
        one.off();

        const two = await session({ model: RO });
        await opened(two, "three framed");
        console.log(JSON.stringify({
            style: two.model.view.get().style,
            axis: two.model.selection.switches().showAxis,
            scale: two.model.selection.switches().forceScale,
            frame: two.model.currentFrame(),
            measuring: two.model.measurement.getState().active,
            pointed: two.engine.pointed,
            badge: two.model.uncommitted,
        }));
        """)
    assert out["style"] == "sphere"
    assert out["axis"] is True and out["scale"] == 2
    assert out["frame"] == 2, "the displayed frame did not come back"
    assert out["measuring"] is True, "the ruler's on/off did not come back"
    assert out["pointed"] == [[1, 2, 3, 4]], (
        "the pose was not put back on a matching structure"
    )
    assert out["badge"] is False, "restoring a look raised the badge"


def test_a_different_structure_gets_the_preferences_never_the_pose(app_at,
                                                                   pairs):
    """§ 11.2b's guard: style and switches are the user's standing
    preferences; the camera, the frame and the selection belonged to a
    structure, and a stale pose can leave a molecule off-screen."""
    out = _run(app_at, pairs, """
        const RO = { mode: "readonly" };     // a frame set opens read-only
        const one = await session({ pose: [9, 9, 9], model: RO });
        await opened(one, "two framed");
        await settled();
        one.model.view.set("style", "sphere");
        one.model.setCurrentFrame(2);
        await settled();
        one.off();

        const two = await session({ model: RO });
        await opened(two, "three framed");
        console.log(JSON.stringify({
            style: two.model.view.get().style,
            pointed: two.engine.pointed,
            frame: two.model.currentFrame(),
        }));
        """)
    assert out["style"] == "sphere", "the standing preference must carry over"
    assert out["pointed"] == [], "a pose was applied onto a different structure"
    assert out["frame"] == 0


def test_selection_rides_the_lane_only_where_no_truth_lane_exists(app_at,
                                                                  pairs):
    out = _run(app_at, pairs, """
        // READ-ONLY: the lane carries the selection home.
        const ro1 = await session({ owner: "ro", model: { mode: "readonly" } });
        await opened(ro1, "two");
        await settled();
        ro1.model.selection.add([1]);
        await settled();
        ro1.off();
        const ro2 = await session({ owner: "ro", model: { mode: "readonly" } });
        await opened(ro2, "two");

        // EDITABLE: the draft owns the selection; the lane must not fight it.
        const ed1 = await session({ owner: "ed" });
        await opened(ed1, "two");
        await settled();
        ed1.model.selection.add([0, 1]);
        await settled();
        ed1.off();
        const uiSlot = await ws.readDraft("ed:ui");

        console.log(JSON.stringify({
            roSelection: ro2.model.selection.get(),
            editableLaneCarries: "selection" in (uiSlot || {}),
        }));
        """)
    assert out["roSelection"] == [1], (
        "a read-only viewer's selection did not survive — it has no other "
        "way home"
    )
    assert out["editableLaneCarries"] is False, (
        "the lane carries a selection beside a truth lane that owns it — "
        "two lanes restoring one fact is the drift § 5.2 forbids"
    )


def test_the_context_applies_after_a_draft_restore_not_under_it(app_at,
                                                                 pairs):
    """§ 11.2b's ordering sentence: a draft's own restore calls
    `view.reset()` after its settle announces; the context must land on
    top of that, or the style comes back only to be reset."""
    out = _run(app_at, pairs, """
        const one = await session();
        await opened(one, "two");
        await settled();
        one.model.view.set("style", "sphere");
        await settled();
        one.off();

        // A reopened page: a FRESH viewer adopts the draft (load(0) calls
        // view.reset()), and the context must still win the ordering.
        const two = await session();
        const adopted = await two.model.load(0);
        await __quiet();
        console.log(JSON.stringify({
            adopted: adopted !== null && adopted !== undefined,
            style: two.model.view.get().style,
        }));
        """)
    assert out["adopted"] is True
    assert out["style"] == "sphere", (
        "the draft's view.reset() landed AFTER the context — the ordering "
        "§ 11.2b states is broken"
    )


@pytest.mark.parametrize("now, restored", [
    pytest.param("O O C", [2], id="the structure it was made on: put back"),
    pytest.param("C O O", [],
                 id="the same molecule, its elements in another order"),
    pytest.param("O O C framed", [],
                 id="the same elements, as a frame set"),
])
def test_a_selection_is_put_back_only_on_the_structure_it_was_made_on(
        app_at, pairs, now, restored):
    """§ 11.2b's `match`: an atom index holds only against the structure it
    was made on (`model/overview.md` § 2.1), so the read-only viewer's
    selection comes back only where the elements, in order, and the order
    the structure is listed in are the ones it was made on.  The silent
    failure: a selection made on the user's pair put back onto a run's own
    file of the same molecule, as many atoms listed in the engine's order --
    other atoms selected, and the transport report asking their PDOS.  The
    match on the stated order alone -- the same elements in the same sequence,
    one listed in an engine's order -- needs a run's own file whose held-first
    order keeps the sequence (the road's CO2 with an oxygen held, O O C); it
    is owed (the finish-line review's O-TESTS)."""
    out = _run(app_at, pairs, f"""
        const one = await session({{ owner: "m", model: {{ mode: "readonly" }} }});
        await opened(one, "O O C");
        await settled();
        one.model.selection.add([2]);
        await settled();
        one.off();
        const two = await session({{ owner: "m", model: {{ mode: "readonly" }} }});
        await opened(two, {json.dumps(now)});
        console.log(JSON.stringify({{ selection: two.model.selection.get() }}));
        """)
    assert out["selection"] == restored, out

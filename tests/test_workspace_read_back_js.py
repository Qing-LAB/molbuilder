"""What a page saved, read back -- and a read that fails, said
(`web/workspace.md` § 5, ``readState``; `web/notifications.md` § 3).

The store answers its caller ``null`` both for nothing saved and for a read
that failed -- "nothing to put back" is the only safe reading of either --
and says the failure, never the miss, on its failure channel; the
notification bar words it as a row of its own, the server's reason in it.

THE FAILURE IS A REAL ONE: a state the store wrote through the real route,
its file made unreadable on disk, and the real route asked for it back.

THE SILENT FAILURE: through 2.68.5 a read that failed was taken for nothing
saved -- the reopened tab came up empty, and nothing told the person that
what they saved was there and could not be read.
"""
from __future__ import annotations

from pathlib import Path

from support import workspace_page

_LIB = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static" / "lib"
#: The header's notification bar, which words a failed read.
_BAR = _LIB / "app-notifications.js"

#: The page around the store and the bar: its event bus, Node's own
#: `EventTarget`, which the bar listens on -- the bar keeps its rows
#: (``notify.list()``) with no document to draw them in.  ONE browser tab's
#: storage, Node's own (`_node_esm.tab_storage`), the same
#: tab's on every opening, so a second opening is that tab reloaded and finds
#: its files (`workspace.md` § 4).
_PAGE = """
const __bus = new EventTarget();
globalThis.addEventListener = __bus.addEventListener.bind(__bus);
globalThis.dispatchEvent = __bus.dispatchEvent.bind(__bus);
"""

def _page(base, snippet, tab_storage=None):
    """``snippet`` on the page's workspace in one browser tab, the bar in
    its header (`support.workspace_page`) -- ``tab_storage`` what the tab
    kept in an earlier opening, so this one is that tab reloaded."""
    return workspace_page.run(base, snippet, tab=tab_storage,
                              header=[_BAR], globals_js=_PAGE)


def test_a_state_that_cannot_be_read_back_is_said_and_a_miss_is_not(app_at):
    from molbuilder.web.blueprints.workspace_storage import _state_dir
    states = _state_dir()
    saved = _page(app_at, """
        ws.persistDraft("modify", { v: 1, state: { atoms: 3 } });
        await __quiet();
        console.log(JSON.stringify({ back: await ws.readDraft("modify"),
                                     tab: tabStorage() }));
    """)
    assert saved["back"] == {"v": 1, "state": {"atoms": 3}}, saved
    # THE ONE FILE the draft was written to, as the server keeps it.
    (kept,) = states.glob("*.wc.json")
    kept.chmod(0)
    try:
        out = _page(app_at, """
            const heard = [];
            ws.onPersistError((d) => heard.push(d));
            // A point never saved: nothing to put back, and nothing to say.
            const missed = await ws.readState("modify", 7);
            const back = await ws.readDraft("modify");
            console.log(JSON.stringify({
                missed, back, heard, rows: window.molbuilder.notify.list(),
            }));
        """, tab_storage=saved["tab"])
    finally:
        kept.chmod(0o600)
    assert out["missed"] is None and out["back"] is None, out
    assert [(d["op"], d["status"]) for d in out["heard"]] == [
        ("read-state", 500)], out["heard"]
    reason = out["heard"][0]["error"]
    assert "could not read state" in reason, reason
    rows = out["rows"]
    assert [(r["id"], r["level"]) for r in rows] == [("read-error", "error")], rows
    assert reason in rows[0]["message"], rows

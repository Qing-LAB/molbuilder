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

import json
from pathlib import Path

import pytest

from tests._node_esm import run_node, served

_LIB = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "static" / "lib"
_STORE = _LIB / "workspace" / "dispatcher.js"
_BAR = _LIB / "app-notifications.js"
#: What a page keeps in this browser tab -- where the tab's own id lives
#: (`runtime.md` § 3), loaded by the page header before any page's own script.
_KEPT = _LIB / "page-prefs.js"

#: The page around the store and the bar: its event bus, a document with no
#: bar drawn in it -- the bar keeps its rows (``notify.list()``) either way --
#: and ONE browser tab's storage, the same tab's on every opening, so a second
#: opening is that tab reloaded and finds its files (`workspace.md` § 4).
_PAGE = """
const __tabStore = new Map(Object.entries(globalThis.__tabStorage || {}));
globalThis.tabStorage = () => Object.fromEntries(__tabStore);
globalThis.sessionStorage = {
    getItem: (k) => (__tabStore.has(k) ? __tabStore.get(k) : null),
    setItem: (k, v) => { __tabStore.set(k, String(v)); },
    removeItem: (k) => { __tabStore.delete(k); },
};
const __bus = new EventTarget();
globalThis.addEventListener = __bus.addEventListener.bind(__bus);
globalThis.dispatchEvent = __bus.dispatchEvent.bind(__bus);
globalThis.document = { getElementById: () => null, addEventListener() {} };
"""

@pytest.fixture
def app_at(isolated_projects_root):
    """The app, serving: what answers the store's requests."""
    from support.live_server import serve
    with serve() as base:
        yield base


def _page(base, snippet, tab_storage=None):
    """``snippet`` on the page in one browser tab -- ``tab_storage`` what the
    tab kept in an earlier opening, so this one is that tab reloaded."""
    seed = f"globalThis.__tabStorage = {json.dumps(tab_storage or {})};\n"
    return run_node([_KEPT, _BAR, _STORE], snippet,
                    globals_js=seed + _PAGE + served(base))


def test_a_state_that_cannot_be_read_back_is_said_and_a_miss_is_not(app_at):
    from molbuilder.web.blueprints.workspace_storage import _state_dir
    states = _state_dir()
    saved = _page(app_at, """
        const ws = window.molbuilder.workspace;
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
            const ws = window.molbuilder.workspace;
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

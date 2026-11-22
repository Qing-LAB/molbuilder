"""``lib/molbuilder-runtime.js``, the runtime registry (`web/runtime.md`
§ 2): ``window.molbuilder.runtime.register / whenReady / get /
listRegistered / listPending``, the load-order contract every other module
relies on.

Pure JS in node, no DOM (`process/testing.md` § 4): the registry is loaded
as a page loads it, first and through its own file (`_node_esm.run_node`),
and each test is a page of its own -- its own node run -- so the registry's
maps start empty.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests._node_esm import run_node

ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molbuilder-runtime.js"


def _run_node(snippet: str) -> object:
    """``snippet`` on a page that has loaded the registry; it ends with
    ``console.log(JSON.stringify(OUT))``."""
    return run_node([MODULE], snippet)


# --------------------------------------------------------------------- #
#  register()                                                            #
# --------------------------------------------------------------------- #


class TestRegister:

    def test_register_then_get_round_trip(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("foo", {hello: "world"});
            console.log(JSON.stringify(rt.get("foo")));
        ''')
        assert out == {"hello": "world"}

    @pytest.mark.parametrize("call", [
        'rt.register("", {x: 1})',
        'rt.register("x", null)',
        'rt.register("x", undefined)',
    ], ids=["an empty name", "a null api", "no api"])
    def test_register_refuses_a_name_or_an_api_it_cannot_hold(self, call):
        """§ 2: "Throws if the name is empty or the api is null" -- a
        programmer's mistake said at once, rather than a ``whenReady`` that
        later resolves to nothing."""
        out = _run_node(f'''
            const rt = window.molbuilder.runtime;
            let thrown = null;
            try {{ {call}; }} catch (e) {{ thrown = e.name; }}
            console.log(JSON.stringify({{ thrown,
                                          registered: rt.listRegistered() }}));
        ''')
        assert out == {"thrown": "TypeError", "registered": []}

    def test_register_accepts_falsy_but_non_null_api(self):
        """The api object is opaque to the registry; 0, "", false,
        and {} are all valid registrations (they're not null /
        undefined).  Pinning this stops a future "tighten the
        guard" PR from accidentally rejecting legitimate APIs."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("zero",  0);
            rt.register("empty", "");
            rt.register("false", false);
            rt.register("obj",   {});
            console.log(JSON.stringify({
                zero:  rt.get("zero"),
                empty: rt.get("empty"),
                false_: rt.get("false"),
                obj:   typeof rt.get("obj"),
            }));
        ''')
        assert out == {"zero": 0, "empty": "", "false_": False, "obj": "object"}

    def test_re_register_replaces_and_warns(self):
        """A second register() for the same name is a programmer
        bug (two modules claiming the same slot).  The registry
        warns to console but accepts the new value so the page
        doesn't break."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            const warnings = [];
            console.warn = (msg) => warnings.push(msg);
            rt.register("dup", {version: 1});
            rt.register("dup", {version: 2});
            console.log(JSON.stringify({
                current:    rt.get("dup"),
                warn_count: warnings.length,
                warn_msg:   warnings[0] || "",
            }));
        ''')
        assert out["current"] == {"version": 2}
        assert out["warn_count"] == 1
        assert "re-registered" in out["warn_msg"]
        assert "dup" in out["warn_msg"]


# --------------------------------------------------------------------- #
#  whenReady()                                                           #
# --------------------------------------------------------------------- #


class TestWhenReady:

    def test_resolves_immediately_if_already_registered(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("alpha", {x: 1});
            rt.whenReady("alpha").then((api) => {
                console.log(JSON.stringify(api));
            });
        ''')
        assert out == {"x": 1}

    def test_resolves_after_late_register(self):
        """The canonical "consumer is ready before producer" case:
        whenReady() must return a Promise that resolves the
        moment register() lands."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            let resolved = null;
            rt.whenReady("beta").then((api) => {
                resolved = api;
            });
            // At this point the waiter is queued -- listPending must
            // mention beta.
            const pending_before = rt.listPending();
            rt.register("beta", {y: 2});
            // The Promise resolution runs on the microtask queue --
            // use queueMicrotask to land AFTER it so we observe the
            // resolved value.
            queueMicrotask(() => {
                console.log(JSON.stringify({
                    pending_before:  pending_before,
                    pending_after:   rt.listPending(),
                    resolved_value:  resolved,
                }));
            });
        ''')
        assert out["pending_before"] == ["beta"]
        assert out["pending_after"]  == []
        assert out["resolved_value"] == {"y": 2}

    def test_multiple_consumers_all_resolved_by_one_register(self):
        """Three whenReady() callers for the same name -- all must
        resolve with the same api when register() fires."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            const results = [];
            rt.whenReady("triple").then((a) => results.push(["c1", a]));
            rt.whenReady("triple").then((a) => results.push(["c2", a]));
            rt.whenReady("triple").then((a) => results.push(["c3", a]));
            rt.register("triple", {id: 42});
            queueMicrotask(() => {
                console.log(JSON.stringify(results));
            });
        ''')
        # All three consumers see the same api object.
        assert len(out) == 3
        assert all(entry[1] == {"id": 42} for entry in out)
        assert {entry[0] for entry in out} == {"c1", "c2", "c3"}

    def test_rejects_empty_name(self):
        """Unlike register() (which throws synchronously),
        whenReady() rejects via the Promise so callers always get
        a Promise back -- no mixed sync/async error handling."""
        out = _run_node('''
            window.molbuilder.runtime.whenReady("").catch((e) => {
                console.log(JSON.stringify({
                    rejected: true,
                    name:     e.name,
                }));
            });
        ''')
        assert out == {"rejected": True, "name": "TypeError"}

    def test_consumer_throwing_does_not_block_other_consumers(self):
        """If one whenReady consumer's .then() handler throws, the
        other consumers MUST still receive the api.  Isolation
        comes from the Promise microtask queue (each consumer runs
        independently); the registry just has to make sure both
        Promises actually resolve.  The throwing consumer's error
        bubbles out as an unhandled rejection -- callers are
        expected to handle their own .catch()."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            const seen = [];
            // Catch the throw inside the handler so it does NOT
            // become an unhandled rejection that crashes node;
            // production code is expected to either not throw or
            // attach its own .catch().  The point of the test is:
            // does the OTHER consumer still see the api?
            rt.whenReady("err").then((api) => {
                seen.push(["1", api]);
                throw new Error("kaboom");
            }).catch(() => { /* silence the bubble */ });
            rt.whenReady("err").then((api) => {
                seen.push(["2", api]);
            });
            rt.register("err", {ok: 1});
            queueMicrotask(() => {
                console.log(JSON.stringify({ seen: seen }));
            });
        ''')
        # Both consumers received the api despite the first throwing.
        assert len(out["seen"]) == 2
        names = {entry[0] for entry in out["seen"]}
        assert names == {"1", "2"}
        assert all(entry[1] == {"ok": 1} for entry in out["seen"])


# --------------------------------------------------------------------- #
#  list* snapshots                                                       #
# --------------------------------------------------------------------- #


class TestListSnapshots:

    def test_listRegistered_returns_sorted_snapshot(self):
        """Names returned in lexical sort order regardless of
        registration order -- predictable for devtools."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("zebra", {});
            rt.register("apple", {});
            rt.register("mango", {});
            console.log(JSON.stringify(rt.listRegistered()));
        ''')
        assert out == ["apple", "mango", "zebra"]

    def test_listPending_only_has_unresolved_waiters(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.whenReady("notyet1").then(() => {});
            rt.whenReady("notyet2").then(() => {});
            rt.register("alreadyhere", {});
            // alreadyhere is NOT pending (no waiter); notyet1 +
            // notyet2 are pending (waiters but no registration).
            console.log(JSON.stringify({
                pending:    rt.listPending(),
                registered: rt.listRegistered(),
            }));
        ''')
        assert out["pending"]    == ["notyet1", "notyet2"]
        assert out["registered"] == ["alreadyhere"]

    def test_listRegistered_is_a_snapshot_not_a_live_view(self):
        """Mutating the returned array MUST NOT affect the registry's
        internal state.  Without this guarantee, devtools code that
        sorts / filters the result could corrupt the registry."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("a", {});
            rt.register("b", {});
            const snapshot = rt.listRegistered();
            snapshot.push("synthetic");
            snapshot.length = 0;
            console.log(JSON.stringify(rt.listRegistered()));
        ''')
        assert out == ["a", "b"]


# --------------------------------------------------------------------- #
#  get()                                                                 #
# --------------------------------------------------------------------- #


class TestGet:

    def test_get_returns_undefined_for_unregistered_name(self):
        """get() is synchronous; for a name the registry has never
        seen it returns undefined (not null, not a rejected
        promise)."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            console.log(JSON.stringify({
                missing: rt.get("nope") === undefined,
            }));
        ''')
        assert out == {"missing": True}


# --------------------------------------------------------------------- #
#  typed() -- molbuilder's command as a person types it                  #
# --------------------------------------------------------------------- #

def _head_holding(prog):
    """The page head as `_page_head.html` writes it: one
    ``<meta name="molbuilder-typed">`` -- stood in by a document holding
    that one element (a named need: this harness has no DOM)."""
    meta = ("null" if prog is None else
            "{ getAttribute: (n) => n === 'content' ? %s : null }"
            % __import__("json").dumps(prog))
    return ("globalThis.document = { querySelector: (s) => "
            "s === 'meta[name=\"molbuilder-typed\"]' ? " + meta + " : null };")


def test_a_page_spells_a_command_as_its_head_hands_it():
    """`job-system.md` § 5.3: a page says molbuilder's command as a person
    types it, the server's one spelling (`identity.PROG`), handed in its
    head -- never spelled by the page."""
    from support.road import TYPED
    out = run_node([MODULE], '''
        console.log(JSON.stringify(
            window.molbuilder.typed("serve", "status", "--port", 8888)));
    ''', globals_js=_head_holding(TYPED))
    assert out == TYPED + " serve status --port 8888"


def test_a_page_whose_head_hands_no_spelling_says_so():
    """No meta, no command spelled: a page whose head is not the shared one
    is said at once, never a bare `molbuilder`."""
    out = run_node([MODULE], '''
        let thrown = null;
        try { window.molbuilder.typed("describe"); }
        catch (e) { thrown = e.message; }
        console.log(JSON.stringify(thrown));
    ''', globals_js=_head_holding(None))
    assert "molbuilder-typed" in out

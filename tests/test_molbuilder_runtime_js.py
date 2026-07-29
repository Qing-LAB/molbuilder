"""Unit tests for ``lib/molbuilder-runtime.js`` driven via Node.

Task #191 (audit follow-up, 2026-06-02).  The runtime registry
(``window.molbuilder.runtime.register / whenReady / get /
listRegistered / listPending``) is the load-bearing init contract
every other module relies on -- but until this commit it had NO
direct unit tests.  Consumers (test_molbuilder_e2e.py and friends)
exercised it indirectly by virtue of the tab's other modules
self-registering, which means:

  * A regression in ``register`` (e.g. forgetting to drain
    ``_waiters``) would surface as "the next tab refresh hangs",
    not as a test failure with a clear name.
  * Edge cases (re-registration, error in a resolver, list*
    snapshot stability) were entirely untested.

This file is the lowest test layer for the registry per
``docs/process/testing.md`` § 1: pure JS unit tests in
node, no DOM.  The IIFE is loaded via ``-e`` with a stubbed
``window`` so each test starts from a clean slate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT   = Path(__file__).resolve().parents[1]
MODULE = ROOT / "molbuilder/web/static/lib/molbuilder-runtime.js"


def _run_node(snippet: str) -> object:
    """Run a JS snippet under Node with the runtime IIFE pre-loaded.

    The IIFE attaches to ``window.molbuilder.runtime``.  Each test
    runs in its own ``node -e`` invocation so the module's internal
    Maps start empty -- no cross-test pollution from the singleton.
    The snippet must end with ``console.log(JSON.stringify(OUT))``.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    bootstrap = """
        // The IIFE picks ``window`` or ``globalThis``; we pin it to
        // a per-invocation global so re-loading the module across
        // tests can't leak state (Node sandboxes each --input-type
        // invocation anyway, but we make the contract explicit).
        global.window = global;
    """
    full = bootstrap + "\n" + MODULE.read_text() + "\n" + snippet
    proc = subprocess.run(
        [node, "--input-type=commonjs", "-e", full],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout}"
        )
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)


# --------------------------------------------------------------------- #
#  API surface presence                                                  #
# --------------------------------------------------------------------- #


class TestSurfacePresence:

    def test_runtime_object_exposes_documented_methods(self):
        """The five public methods named in the module's docstring
        MUST all be callable functions on ``window.molbuilder.runtime``.
        """
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            console.log(JSON.stringify({
                register:       typeof rt.register,
                whenReady:      typeof rt.whenReady,
                get:            typeof rt.get,
                listRegistered: typeof rt.listRegistered,
                listPending:    typeof rt.listPending,
            }));
        ''')
        for fn in ("register", "whenReady", "get",
                   "listRegistered", "listPending"):
            assert out[fn] == "function", f"missing method: {fn}"

    def test_double_load_is_idempotent(self):
        """Loading the IIFE twice (e.g. a template double-includes
        the script) must keep the original runtime + ignore the
        second IIFE.  Concretely: a module registered BEFORE the
        second load must still resolve whenReady() AFTER it."""
        out = _run_node(f'''
            // First load happened in the bootstrap above; register
            // a marker module against THAT runtime.
            window.molbuilder.runtime.register("marker", {{value: 1}});
            // Now re-evaluate the IIFE source.  If the idempotence
            // guard is missing, this would clobber the registry.
            {json.dumps(MODULE.read_text())};
            eval({json.dumps(MODULE.read_text())});
            const api = window.molbuilder.runtime.get("marker");
            console.log(JSON.stringify({{
                marker_still_there: api && api.value === 1,
                list: window.molbuilder.runtime.listRegistered(),
            }}));
        ''')
        assert out["marker_still_there"] is True
        assert out["list"] == ["marker"]


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

    def test_register_lands_in_listRegistered(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("a", {x: 1});
            rt.register("b", {y: 2});
            rt.register("c", {z: 3});
            console.log(JSON.stringify(rt.listRegistered()));
        ''')
        # listRegistered is documented as sorted -- pin the contract.
        assert out == ["a", "b", "c"]

    def test_register_rejects_empty_name(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            try {
                rt.register("", {x: 1});
                console.log(JSON.stringify({threw: false}));
            } catch (e) {
                console.log(JSON.stringify({
                    threw: true,
                    name:  e.name,
                    msg:   e.message,
                }));
            }
        ''')
        assert out["threw"] is True
        assert out["name"] == "TypeError"
        assert "non-empty string" in out["msg"]

    def test_register_rejects_non_string_name(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            try {
                rt.register(42, {x: 1});
                console.log(JSON.stringify({threw: false}));
            } catch (e) {
                console.log(JSON.stringify({
                    threw: true,
                    name:  e.name,
                }));
            }
        ''')
        assert out["threw"] is True
        assert out["name"] == "TypeError"

    def test_register_rejects_null_api(self):
        """A null/undefined api is a programmer bug -- the registry
        rejects it loudly rather than letting whenReady() resolve to
        ``null`` and downstream callers NPE."""
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            const results = [];
            try { rt.register("x", null); }
            catch (e) { results.push("null:" + e.name); }
            try { rt.register("x", undefined); }
            catch (e) { results.push("undef:" + e.name); }
            console.log(JSON.stringify(results));
        ''')
        assert out == ["null:TypeError", "undef:TypeError"]

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

    def test_get_returns_registered_api(self):
        out = _run_node('''
            const rt = window.molbuilder.runtime;
            rt.register("g", {payload: [1, 2, 3]});
            console.log(JSON.stringify(rt.get("g")));
        ''')
        assert out == {"payload": [1, 2, 3]}

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

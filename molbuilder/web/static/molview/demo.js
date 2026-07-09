/* Standalone MolView component demo driver (served by templates/molview_demo.html).
 *
 * Mounts the FULL component via molview.mount's empty-host build path against the real
 * workspace, and wires the sample-load buttons.  External file (not inline) so it satisfies
 * the app CSP (script-src 'self').
 */
(function () {
    "use strict";

    var SAMPLES = {
        water:   "3\nwater\nO 0.000 0.000 0.000\nH 0.757 0.586 0.000\nH -0.757 0.586 0.000\n",
        benzene: "12\nbenzene\n"
               + "C 0.000 1.396 0\nC 1.209 0.698 0\nC 1.209 -0.698 0\nC 0.000 -1.396 0\n"
               + "C -1.209 -0.698 0\nC -1.209 0.698 0\nH 0.000 2.479 0\nH 2.147 1.240 0\n"
               + "H 2.147 -1.240 0\nH 0.000 -2.479 0\nH -2.147 -1.240 0\nH -2.147 1.240 0\n",
        auCell:  "4\nAu fcc\nAu 0 0 0\nAu 2.04 2.04 0\nAu 2.04 0 2.04\nAu 0 2.04 2.04\n",
    };

    function ready(fn) {
        if (document.readyState !== "loading") fn();
        else document.addEventListener("DOMContentLoaded", fn);
    }

    ready(function () {
        var ws   = window.molbuilder && window.molbuilder.workspace;
        var mv   = window.molbuilder && window.molbuilder.molview;
        var host = document.getElementById("molview-demo-host");
        var statusEl = document.getElementById("demo-status");
        function say(m) { if (statusEl) statusEl.textContent = m; }

        if (!ws || !mv || typeof mv.mount !== "function" || !host) {
            say("molview / workspace failed to load — check the console.");
            return;
        }

        function load(name) {
            return ws.loadFromText(SAMPLES[name], "demo-" + name + ".xyz")
                .then(function () { say("loaded " + name + " — panel + render updated."); })
                .catch(function (e) { say("load failed: " + (e && e.message)); });
        }

        // Load a first sample, THEN mount the full component (its render loop draws it on
        // onReady + re-draws whenever the workspace changes).
        load("water").then(function () {
            return mv.mount(host, ws, { mode: "modify", owner: "molview-demo" });
        }).then(function (handle) {
            window.__molview = handle;   // poke the §D API from the console
            say("Mounted. Try the panel (Selection ↔ Cell tabs), the view toggles, and the "
                + "sample buttons; the render reacts through ws.*.  __molview holds the handle.");
            document.getElementById("demo-water").addEventListener("click", function () { load("water"); });
            document.getElementById("demo-benzene").addEventListener("click", function () { load("benzene"); });
            document.getElementById("demo-au-cell").addEventListener("click", function () { load("auCell"); });
        }).catch(function (e) {
            say("mount failed: " + (e && e.message));
            if (window.console) window.console.error("[molview-demo]", e);
        });
    });
})();

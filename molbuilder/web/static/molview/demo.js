/* Standalone MolView component demo driver (served by templates/molview_demo.html).
 *
 * Mounts the FULL component via molview.mount's empty-host build path against the real
 * workspace, and wires the sample-load buttons.  External file (not inline) so it satisfies
 * the app CSP (script-src 'self').
 *
 * ES module: it IMPORTS { mount } from the concealed MolView module (the single-import contract)
 * instead of reading the transitional window.molbuilder.molview.mount global.
 */
import { mount, data as mvData } from "/static/lib/molview/index.js";

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

    // A 3-frame trajectory of water (workspace-contract.md §1.5) -- atom 0 (O) slides along +x
    // (0 -> 1 -> 2) so a frame swap is trivially verifiable in the viewer; per-frame forces on
    // O grow with the displacement.  Frame 0 IS the loaded structure.  Lets multi-frame be
    // exercised on /molview-demo alone, without touching the trajectory inspector or any tab.
    var TRAJECTORY = {
        text:   SAMPLES.water,
        frames: [
            [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
            [[1, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
            [[2, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
        ],
        forces: [
            [[0.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0.5, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1.0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
    };

    function ready(fn) {
        if (document.readyState !== "loading") fn();
        else document.addEventListener("DOMContentLoaded", fn);
    }

    ready(function () {
        var ws   = window.molbuilder && window.molbuilder.workspace;     // persistence layer (separate module, classic global)
        var data = mvData;                                               // the in-memory DATA model (imported from the molview door)
        var host = document.getElementById("molview-demo-host");
        var statusEl = document.getElementById("demo-status");
        function say(m) { if (statusEl) statusEl.textContent = m; }

        if (!ws || !data || typeof mount !== "function" || !host) {
            say("molview / workspace failed to load — check the console.");
            return;
        }

        function load(name) {
            return data.installMolecule({ text: SAMPLES[name], filename: "demo-" + name + ".xyz" })
                .then(function () { say("loaded " + name + " — panel + render updated."); })
                .catch(function (e) { say("load failed: " + (e && e.message)); });
        }

        // Load the structure (frame 0) then hand MolView's data model the full frame series
        // WITH its per-frame forces.  The render engine builds the force arrows itself from
        // ``forces`` (molview-render-streamline.md §2.4 / §11) -- the demo does NOT push arrows;
        // the "Overlay" View-menu toggle shows/hides what the engine baked.
        function loadTrajectory() {
            return data.installMolecule({ text: TRAJECTORY.text, filename: "demo-traj.xyz" }).then(function () {
                var n = data.reloadFrames(TRAJECTORY.frames, { forces: TRAJECTORY.forces });
                say("loaded a " + n + "-frame trajectory — play/scrub with the bar; turn on "
                    + "'Overlay' in the View menu to see per-frame force arrows.");
            }).catch(function (e) { say("trajectory load failed: " + (e && e.message)); });
        }

        // Load a first sample, THEN mount the full component (its render loop draws it on
        // onReady + re-draws whenever the workspace changes).
        load("water").then(function () {
            return mount(host, ws, { mode: "modify", owner: "molview-demo" });
        }).then(function (handle) {
            if (!handle || !handle.ok) {   // mount contract: failure -> {ok:false}
                say("Mount failed: " + ((handle && handle.error) || "unknown"));
                return;
            }
            window.__molview = handle;   // poke the §D API from the console
            say("Mounted. Try the panel (Selection ↔ Cell tabs), the view toggles, and the "
                + "sample buttons; the render reacts through molview.data.  __molview holds the handle.");
            document.getElementById("demo-water").addEventListener("click", function () { load("water"); });
            document.getElementById("demo-benzene").addEventListener("click", function () { load("benzene"); });
            document.getElementById("demo-au-cell").addEventListener("click", function () { load("auCell"); });
            var trajBtn = document.getElementById("demo-trajectory");
            if (trajBtn) trajBtn.addEventListener("click", loadTrajectory);
        }).catch(function (e) {
            say("mount failed: " + (e && e.message));
            if (window.console) window.console.error("[molview-demo]", e);
        });
    });
})();

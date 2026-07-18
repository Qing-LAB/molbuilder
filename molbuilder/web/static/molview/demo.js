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
        var ws   = window.molbuilder && window.molbuilder.workspace;     // persistence layer
        var mv   = window.molbuilder && window.molbuilder.molview;
        var data = mv && mv.data;                                        // the in-memory DATA model
        var host = document.getElementById("molview-demo-host");
        var statusEl = document.getElementById("demo-status");
        function say(m) { if (statusEl) statusEl.textContent = m; }

        if (!ws || !mv || !data || typeof mv.mount !== "function" || !host) {
            say("molview / workspace failed to load — check the console.");
            return;
        }

        var mvHandle   = null;    // the mounted handle (setArrows / currentFrame / onChange)
        var trajLoaded = false;   // draw force arrows only while a trajectory is loaded

        // CONSUMER-owned overlay: the demo builds force arrows from ITS OWN force data (with its
        // own scale) and hands them to MolView via handle.setArrows.  MolView draws exactly what
        // it's handed and never generates arrows (molview-module.md §14.5.1); the "Overlay"
        // View-menu toggle shows/hides them.
        function updateForceArrows() {
            if (!mvHandle || typeof mvHandle.setArrows !== "function") return;
            if (!trajLoaded) { mvHandle.setArrows([]); return; }
            var i = (typeof mvHandle.currentFrame === "function") ? mvHandle.currentFrame() : 0;
            var coords = TRAJECTORY.frames[i], forces = TRAJECTORY.forces[i], arrows = [];
            if (coords && forces) {
                for (var k = 0; k < coords.length; k++) {
                    var f = forces[k];
                    if (Math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) < 1e-9) continue;
                    var p = coords[k];
                    arrows.push({ start: [p[0], p[1], p[2]],
                                  end:   [p[0] + f[0], p[1] + f[1], p[2] + f[2]],
                                  color: "#f0a020", radius: 0.06 });
                }
            }
            mvHandle.setArrows(arrows);
        }

        function load(name) {
            trajLoaded = false;
            return data.installMolecule({ text: SAMPLES[name], filename: "demo-" + name + ".xyz" })
                .then(function () { updateForceArrows(); say("loaded " + name + " — panel + render updated."); })
                .catch(function (e) { say("load failed: " + (e && e.message)); });
        }

        // Load the structure (frame 0) then hand MolView's data model the full frame series.
        function loadTrajectory() {
            return data.installMolecule({ text: TRAJECTORY.text, filename: "demo-traj.xyz" }).then(function () {
                var n = data.reloadFrames(TRAJECTORY.frames, { forces: TRAJECTORY.forces });
                trajLoaded = true;
                updateForceArrows();
                say("loaded a " + n + "-frame trajectory — play/scrub with the bar; turn on "
                    + "'Overlay' in the View menu to see per-frame force arrows.");
            }).catch(function (e) { say("trajectory load failed: " + (e && e.message)); });
        }

        // Load a first sample, THEN mount the full component (its render loop draws it on
        // onReady + re-draws whenever the workspace changes).
        load("water").then(function () {
            return mv.mount(host, ws, { mode: "modify", owner: "molview-demo" });
        }).then(function (handle) {
            mvHandle = handle;
            window.__molview = handle;   // poke the §D API from the console
            handle.onChange(updateForceArrows);   // recompute per-frame force arrows on a frame change
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

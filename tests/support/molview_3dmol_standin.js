/* A stand-in for 3Dmol — the LIBRARY beneath MolView's sealed layer.
 *
 * docs/web/molview.md § 13.1: "A stand-in takes the place of a level, so it must
 * obey that level's rules from this document." This one stands in for the
 * library, so it obeys the library behaviour the contract records as the reason
 * things are built the way they are:
 *
 *   - addModelsAsFrames parses a multi-frame XYZ ONCE into frames, and setFrame
 *     swaps which frame is current with no restyle (§ 10.4);
 *   - addArrow returns a shape that itself carries .addArrow, so a whole overlay
 *     batches into one scene object (§ 10.7's measurement);
 *   - every add* returns a handle the matching remove* accepts;
 *   - it is COHERENT: it never reports frames while claiming no model, which
 *     § 13.1 names as the specific way a stand-in goes wrong.
 *
 * It records every call, so a test can assert what work a change actually cost
 * rather than only what it looked like afterwards.
 */

globalThis.__calls = [];
function __rec(name, args) { globalThis.__calls.push({ name: name, args: args }); }
globalThis.__callNames = function (only) {
    return globalThis.__calls
        .map(function (c) { return c.name; })
        .filter(function (n) { return !only || only.indexOf(n) >= 0; });
};
globalThis.__countCalls = function (name) {
    return globalThis.__calls.filter(function (c) { return c.name === name; }).length;
};
globalThis.__lastCall = function (name) {
    const hits = globalThis.__calls.filter(function (c) { return c.name === name; });
    return hits.length ? hits[hits.length - 1] : null;
};
globalThis.__resetCalls = function () { globalThis.__calls = []; };

// Parse the multi-frame XYZ the sealed layer builds. Deliberately strict: the
// format is the ONE fact the drawing commands own, so a test that feeds a bad
// one should see it fail here rather than be quietly tolerated.
function __parseMultiXyz(text) {
    const lines = String(text).split("\n");
    const frames = [];
    let i = 0;
    while (i < lines.length) {
        const head = lines[i].trim();
        if (head === "") { i++; continue; }
        const n = parseInt(head, 10);
        if (!Number.isFinite(n)) throw new Error("standin: bad atom count: " + head);
        i += 2;                                    // count line + comment line
        const atoms = [];
        for (let k = 0; k < n; k++, i++) {
            const parts = (lines[i] || "").trim().split(/\s+/);
            atoms.push({
                elem: parts[0],
                x: Number(parts[1]), y: Number(parts[2]), z: Number(parts[3]),
            });
        }
        frames.push(atoms);
    }
    return frames;
}

globalThis.$3Dmol = {
    elementColors: { Jmol: {} },

    createViewer: function (container, opts) {
        __rec("createViewer", [opts]);
        let handleSeq = 0;

        const v = {
            container: container,
            _frames: [],
            _frame: 0,

            addModelsAsFrames: function (text, fmt) {
                __rec("addModelsAsFrames", [fmt]);
                v._frames = __parseMultiXyz(text);
                v._frame = 0;
            },

            getModel: function () {
                if (!v._frames.length) return null;      // coherent: no frames, no model
                return {
                    selectedAtoms: function () {
                        return v._frames[v._frame].map(function (a, idx) {
                            return { index: idx, elem: a.elem, x: a.x, y: a.y, z: a.z };
                        });
                    },
                    getNumFrames: function () { return v._frames.length; },
                    addFrame: function (atoms) {
                        v._frames.push(atoms.map(function (a) {
                            return { elem: a.elem, x: a.x, y: a.y, z: a.z };
                        }));
                    },
                };
            },

            setFrame: function (i) {
                __rec("setFrame", [i]);
                if (i >= 0 && i < v._frames.length) v._frame = i;
            },

            setStyle:           function (sel, spec) { __rec("setStyle", [sel, spec]); },
            setBackgroundColor: function (c) { __rec("setBackgroundColor", [c]); },
            setProjection:      function (p) { __rec("setProjection", [p]); },
            setClickable:       function (sel, on, cb) {
                __rec("setClickable", [sel, on]);
                v._clickHandler = cb;
            },
            render:  function () { __rec("render", []); },
            zoomTo:  function () { __rec("zoomTo", []); },

            addSphere: function (s) {
                __rec("addSphere", [s]);
                return { __h: "sphere" + (handleSeq++) };
            },
            addCylinder: function (s) {
                __rec("addCylinder", [s]);
                return { __h: "cyl" + (handleSeq++) };
            },
            addLabel: function (text, o) {
                __rec("addLabel", [text, o]);
                return { __h: "label" + (handleSeq++) };
            },
            // The batching contract: the returned shape carries .addArrow, so
            // every arrow after the first appends to the same scene object.
            addArrow: function (s) {
                __rec("addArrow", [s]);
                return {
                    __h: "arrow" + (handleSeq++),
                    addArrow: function (more) { __rec("addArrow:batched", [more]); },
                };
            },

            removeShape:     function (h) { __rec("removeShape", [h && h.__h]); },
            removeLabel:     function (h) { __rec("removeLabel", [h && h.__h]); },
            removeAllModels: function () {
                __rec("removeAllModels", []);
                v._frames = []; v._frame = 0;
            },
            removeAllShapes: function () { __rec("removeAllShapes", []); },
            removeAllLabels: function () { __rec("removeAllLabels", []); },

            pngURI: function (w, h) {
                __rec("pngURI", [w, h]);
                return "data:image/png;base64,aGk=";
            },

            // The vendor's camera pair (GLViewer.getView/setView) -- the
            // sealed layer's pose read rides these (molview.md § 9.6).
            getView: function () {
                __rec("getView", []);
                return (v._view || [0, 0, 0, 40, 0, 0, 0, 1]).slice();
            },
            setView: function (arr) {
                __rec("setView", [arr]);
                v._view = Array.isArray(arr) ? arr.slice() : v._view;
            },
        };
        return v;
    },
};

// A host element that answers only what the sealed layer actually asks of it.
globalThis.__makeHost = function () {
    const busyMsg = { textContent: "" };
    const busy = { hidden: true, querySelector: function () { return busyMsg; } };
    return {
        clientWidth: 300,
        clientHeight: 200,
        _busy: busy,
        _busyMsg: busyMsg,
        querySelector: function (sel) {
            if (sel === ".molview-busy") return busy;
            if (sel === "canvas") return null;         // force the pngURI path
            return null;
        },
    };
};

globalThis.atob = globalThis.atob || function (b64) {
    return Buffer.from(b64, "base64").toString("binary");
};
globalThis.Blob = globalThis.Blob || class { constructor(parts, o) { this.type = o && o.type; } };

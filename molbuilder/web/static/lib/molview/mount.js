/* molview.mount -- the fully-concealed, embeddable MolView component (molview-module.md §18).
 *
 * ONE call assembles the whole molview -- the 3-D viewer + the selection/Cell panel + the
 * view toggles -- inside `hostEl`, reading and writing ONLY through the `workspace` you hand
 * it (the uniform ws.* data interface).  There are NO loader/embed/data hooks: molview holds
 * no data of its own; the workspace owns protection (defensive-copy accessors) and
 * persistence.  Pass the real workspace (edits persist) or a throwaway one (they don't) --
 * molview can't tell the difference and doesn't need to.
 *
 *   molview.mount(hostEl, workspace, { mode, owner }) -> Promise<handle>
 *   handle = { getStructure(), getSelection(), onChange(fn), dispose() }
 *     The READ + notify side of the §D API is built (B1): the owner reads the molecule +
 *     subscribes to changes THROUGH the handle, never storage.  The WRITE side -- load /
 *     save / undo -- is the remaining single-door build (B2).  The handle exposes NO
 *     internals: not the viewer, not the store, not DOM refs.
 *
 * OWNER (molview is aware of its user).  `owner` is this molview's identity -- the tab /
 * consumer it belongs to (e.g. "modify", "results:<id>").  molview forwards it to the
 * workspace so the workspace NAMESPACES its saving points by it (the sessionStorage snapshot
 * key + the server draft id gain the `<owner>` prefix/suffix).  That isolates one tab's
 * persisted data from another's -- two molviews never collide on the single global slot.
 * The namespacing itself lives in the WORKSPACE persistence layer (it owns saving points),
 * driven by this owner -- molview never keys storage itself.  Absent owner => the
 * workspace's default namespace (today's single global slot; unchanged for Modify).
 *
 * INCREMENTAL BUILD (2026-07-08, §18.5): step 4a wires the VIEW-CHROME -- the panel, the
 * view-controls bar, and the fold handle -- against the workspace's store.  The viewer embed
 * + k-grid + measurement (today in modify/viewer.js) fold in at step 4b; the DOM build +
 * Focus-molecule at 4c.  Until then the host must already carry the fused-card DOM
 * (fused-layout.css: .molview-card > .molview-body > viewer | fold | .molview-panel).
 */
(function (root) {
    "use strict";

    async function mount(hostEl, workspace, opts) {
        opts = opts || {};
        const mode   = opts.mode || "modify";
        const owner  = opts.owner || (workspace && workspace.owner) || null;
        const mb     = root.molbuilder || {};
        const selApi = mb.selection;
        const mvApi  = mb.molview;
        const store  = workspace && workspace.selection;

        if (!hostEl || !workspace || !store) return null;
        if (!selApi || typeof selApi.mountPanel !== "function") return null;

        // Tell the workspace who this molview belongs to, so IT namespaces its saving
        // points (sessionStorage snapshot + server draft) by `owner` -- isolating this
        // tab's persisted data from any other's.  Namespacing lives in the workspace
        // persistence layer, not here.  Feature-detected: a no-op until that layer
        // consumes it (workspace-contract persistence namespace work).
        if (owner && workspace && typeof workspace.useNamespace === "function") {
            workspace.useNamespace(owner);
        }

        // Resolve the fused card + its sub-hosts (fused-layout.css structure).
        const card = hostEl.classList && hostEl.classList.contains("molview-card")
            ? hostEl
            : ((hostEl.closest && hostEl.closest(".molview-card")) || hostEl);
        const panelHost = card.querySelector(".molview-panel") || hostEl;
        const vcHost    = card.querySelector(".viewer-toggles");
        const foldBtn   = card.querySelector(".molview-fold-btn");

        const cleanups = [];

        // Panel (atom selection + the Cell page), bound to the workspace's store.
        const panelMount = await selApi.mountPanel(panelHost, { store: store, mode: mode });
        if (!panelMount || !panelMount.panel) return null;   // mountPanel showed its own banner
        cleanups.push(function () { try { panelMount.dispose && panelMount.dispose(); } catch (_) {} });

        // View-controls bar (isolate / k-grid toggles) -- same store, no parallel state.
        if (vcHost && mvApi && typeof mvApi.mountViewControls === "function") {
            const vc = mvApi.mountViewControls(vcHost, store);
            cleanups.push(function () { try { vc.dispose && vc.dispose(); } catch (_) {} });
        }

        // Fold handle -- local layout state (fused-layout.css .is-folded), not store state.
        if (foldBtn) {
            const onFold = function () {
                const folded = card.classList.toggle("is-folded");
                foldBtn.setAttribute("aria-expanded", String(!folded));
            };
            foldBtn.addEventListener("click", onFold);
            cleanups.push(function () { foldBtn.removeEventListener("click", onFold); });
        }

        // ---- The owner-facing handle (§D) --------------------------------------------- //
        // The owner reads the molecule + reacts to changes THROUGH molview -- never storage
        // directly (§A single door).  Reads return copies (the workspace accessors
        // defensive-copy, workspace-contract §1.2.1).  `onChange` is the ONE change channel
        // (§E rule 4): the owner subscribes here instead of reaching for ws.subscribe /
        // store.subscribe itself.  (load / save / undo -- the WRITE side -- land in B2.)
        const _offs = [];   // onChange subscriptions, torn down on dispose
        return {
            getStructure: function () {
                return (typeof workspace.getStructure === "function")
                    ? workspace.getStructure() : null;
            },
            getSelection: function () {
                const s = store.getState();
                return (s && Array.isArray(s.indices)) ? s.indices.slice() : [];
            },
            onChange: function (fn) {
                if (typeof fn !== "function" || typeof workspace.subscribe !== "function") {
                    return function () {};
                }
                const off = workspace.subscribe(fn);
                _offs.push(off);
                return off;
            },
            dispose: function () {
                _offs.forEach(function (off) { try { off(); } catch (_) {} });
                cleanups.forEach(function (fn) { try { fn(); } catch (_) {} });
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mount = mount;
})(typeof window !== "undefined" ? window : this);

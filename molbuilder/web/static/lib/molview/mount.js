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
 *   handle = { load(fileOrText), save(), undo(),                       // WRITE
 *              setFrame(i), getFrame(i), frameCount(), currentFrame(), // FRAMES (§14.5)
 *              play(opts), pause(), isPlaying(),                       //   navigation + playback
 *              showForces(on), showIndices(on),                       //   frame overlays (§14.5.1)
 *              getStructure(), getSelection(),                         // READ
 *              onChange(fn), dispose() }                               // notify + teardown
 *     The full §D owner-facing API + the frame axis (§14.5).  Every call goes through the
 *     WORKSPACE (the single door); the render reacts to workspace changes (§18.2).  A frame's
 *     coord swap notifies the store, so the render redraws on its own; MolView owns only the
 *     playback timer.  The handle exposes NO internals: not the viewer, not the store, not DOM.
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
 * ASSEMBLY -- two paths, chosen by the host you pass:
 *  - EMPTY host  -> molview BUILDS the fused card, EMBEDS the viewer, and OWNS the render
 *    loop (molview.mountRender): the full component.  This is the target every consumer
 *    converges to.
 *  - PRE-BUILT card (a host already carrying .molview-card > .molview-panel, e.g. Modify's
 *    template today) -> molview only wires the existing panel / toggles / fold; that host
 *    still owns its own viewer + render.  Transitional -- retired when the consumer passes an
 *    empty host and lets molview own everything.
 */
(function (root) {
    "use strict";

    function _el(tag, cls) {
        const e = root.document.createElement(tag);
        if (cls) e.className = cls;
        return e;
    }

    // Build the fused-card DOM (fused-layout.css) into hostEl.  Returns the sub-hosts by
    // DIRECT reference -- molview created them, so no querySelector is needed.
    function _buildCard(hostEl) {
        const card       = _el("div", "molview-card");
        const body       = _el("div", "molview-body");
        const viewerCol  = _el("div", "molview-viewer");
        const wrap       = _el("div", "viewer-wrap");
        const viewerHost = _el("div", "viewer");
        wrap.appendChild(viewerHost);
        const controls   = _el("div", "viewer-controls");
        const vcHost     = _el("span", "viewer-toggles");
        controls.appendChild(vcHost);
        viewerCol.appendChild(wrap);
        viewerCol.appendChild(controls);
        const foldBtn    = _el("button", "molview-fold-btn");
        foldBtn.setAttribute("type", "button");
        foldBtn.setAttribute("aria-expanded", "true");
        const chevron    = _el("span", "molview-fold-chevron");
        chevron.textContent = "›";   // ›
        foldBtn.appendChild(chevron);
        const panelHost  = _el("div", "molview-panel");
        body.appendChild(viewerCol);
        body.appendChild(foldBtn);
        body.appendChild(panelHost);
        card.appendChild(body);
        hostEl.appendChild(card);
        return { card: card, panelHost: panelHost, vcHost: vcHost,
                 foldBtn: foldBtn, viewerHost: viewerHost };
    }

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

        const cleanups = [];

        // Frame-overlay toggles (§14.5.1) -- molview-local view state (like isolate/k-grid);
        // the frame-overlay controller reads them + the handle's showForces/showIndices flip them.
        let _showForces = false, _showIndices = false, _frameOverlays = null;

        // Resolve the fused card + its sub-hosts.  PRE-BUILT card (Modify's template) -> wire
        // the existing panel/toggles/fold; that host owns its own viewer + render.  EMPTY
        // host -> molview BUILDS the card, EMBEDS the viewer, and OWNS the render loop.
        let card = hostEl.classList && hostEl.classList.contains("molview-card")
            ? hostEl
            : ((hostEl.closest && hostEl.closest(".molview-card")) || null);
        let panelHost = card && card.querySelector(".molview-panel");
        let vcHost    = card && card.querySelector(".viewer-toggles");
        let foldBtn   = card && card.querySelector(".molview-fold-btn");

        if (!panelHost) {
            // EMPTY host: build the DOM, embed the viewer, own the render (the full component).
            const built = _buildCard(hostEl);
            card = built.card; panelHost = built.panelHost;
            vcHost = built.vcHost; foldBtn = built.foldBtn;
            const viewer = mb.viewer;
            if (viewer && typeof viewer.embed === "function") {
                // Wire the render loop once the viewer handle is ready; molview owns it.
                viewer.embed(built.viewerHost, {
                    onReady: function (h) {
                        // Test hook: expose the viewer handle so e2e can read what was drawn
                        // (the owner never sees it -- it's not on the returned handle).
                        built.viewerHost.__molview_test_handle = h;
                        if (mvApi && typeof mvApi.mountRender === "function") {
                            const rc = mvApi.mountRender(h, workspace, store,
                                                         { viewerHost: built.viewerHost });
                            cleanups.push(function () { try { rc.dispose(); } catch (_) {} });
                        }
                        // Viewer-adapter: selection halos + isolate opacity + click-to-select
                        // reach the viewer (the same attach mountPanel does, §13.2).
                        const adapter = selApi && selApi.viewerAdapter;
                        if (adapter && typeof adapter.attach === "function") {
                            const ah = adapter.attach(h, { store: store, mode: mode });
                            cleanups.push(function () {
                                try { ah && ah.dispose && ah.dispose(); } catch (_) {}
                            });
                        }
                        // Frame-scoped overlays (§14.5.1): force arrows (per-frame) + atom-index
                        // labels for the current frame; gated by the showForces/showIndices flags.
                        if (mvApi && typeof mvApi.mountFrameOverlays === "function") {
                            _frameOverlays = mvApi.mountFrameOverlays(h, workspace, store, {
                                getShowForces:  function () { return _showForces; },
                                getShowIndices: function () { return _showIndices; },
                            });
                            cleanups.push(function () {
                                try { _frameOverlays && _frameOverlays.dispose(); } catch (_) {}
                            });
                        }
                    },
                });
            }
        }

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
        // Frame playback timer -- MolView owns it; navigation delegates to the workspace.
        let _playTimer = null;
        function _stopPlay() {
            if (_playTimer != null) { root.clearInterval(_playTimer); _playTimer = null; }
        }
        const _offs = [];   // onChange subscriptions, torn down on dispose
        return {
            // WRITE side (§D): the owner asks molview to load / save / undo; molview asks the
            // WORKSPACE (the single door) and the render reacts to the resulting workspace
            // change (§18.2) -- the owner never touches storage or triggers a redraw itself.
            load: function (fileOrText) {
                // "Load this molecule."  A path STRING -> the file loader; raw structure text
                // as { text, filename } -> the text loader.
                if (fileOrText && typeof fileOrText === "object"
                        && typeof fileOrText.text === "string") {
                    return (typeof workspace.loadFromText === "function")
                        ? workspace.loadFromText(fileOrText.text, fileOrText.filename)
                        : Promise.reject(new Error("molview.load: workspace.loadFromText missing"));
                }
                if (typeof fileOrText === "string" && fileOrText) {
                    return (typeof workspace.loadFromFile === "function")
                        ? workspace.loadFromFile(fileOrText)
                        : Promise.reject(new Error("molview.load: workspace.loadFromFile missing"));
                }
                return Promise.reject(new TypeError(
                    "molview.load(fileOrText): pass a path string or { text, filename }"));
            },
            save: function () {
                return (typeof workspace.save === "function")
                    ? workspace.save()
                    : Promise.reject(new Error("molview.save: workspace.save missing"));
            },
            undo: function () {
                return (typeof workspace.undo === "function")
                    ? workspace.undo()
                    : Promise.reject(new Error("molview.undo: workspace.undo missing"));
            },
            // ---- Frames -- the coordinate time axis (workspace §1.5, molview §14.5) -------- //
            // Navigation delegates to the workspace (the data owner); MolView owns the playback
            // timer.  A frame's coord swap notifies the store, so the render redraws on its own.
            setFrame: function (i) {
                return (typeof workspace.setFrame === "function") ? workspace.setFrame(i) : undefined;
            },
            getFrame: function (i) {
                return (typeof workspace.getFrame === "function") ? workspace.getFrame(i) : null;
            },
            frameCount: function () {
                return (typeof workspace.frameCount === "function") ? workspace.frameCount() : 0;
            },
            currentFrame: function () {
                return (typeof workspace.currentFrame === "function") ? workspace.currentFrame() : 0;
            },
            play: function (opts) {
                opts = opts || {};
                if (typeof workspace.frameCount !== "function" || workspace.frameCount() <= 1) return;
                const fps = (typeof opts.fps === "number" && opts.fps > 0) ? opts.fps : 10;
                _stopPlay();
                _playTimer = root.setInterval(function () {
                    const n = workspace.frameCount();
                    if (n <= 1) { _stopPlay(); return; }
                    workspace.setFrame((workspace.currentFrame() + 1) % n);
                }, 1000 / fps);
            },
            pause: function () { _stopPlay(); },
            isPlaying: function () { return _playTimer != null; },
            // Frame-overlay view toggles (§14.5.1) -- force arrows + atom-index labels.
            showForces: function (on) {
                _showForces = !!on;
                if (_frameOverlays) _frameOverlays.refresh();
            },
            showIndices: function (on) {
                _showIndices = !!on;
                if (_frameOverlays) _frameOverlays.refresh();
            },
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
                _stopPlay();
                _offs.forEach(function (off) { try { off(); } catch (_) {} });
                cleanups.forEach(function (fn) { try { fn(); } catch (_) {} });
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mount = mount;
})(typeof window !== "undefined" ? window : this);

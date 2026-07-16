/* molview.mount -- the fully-concealed, embeddable MolView component (molview-module.md §18).
 *
 * ONE call assembles the whole molview -- the 3-D viewer + the selection/Cell panel + the
 * view toggles -- inside `hostEl`, reading and writing ONLY through the `workspace` you hand
 * it (the uniform ws.* data interface).  There are NO loader/embed/data hooks: molview holds
 * no data of its own; the workspace owns protection (defensive-copy accessors) and
 * persistence.  Pass the real workspace: every consumer persists its SESSION STATE through
 * it (Modify + read-only Results alike).  "read-only" (opts.mode) is about edit controls,
 * not persistence.
 *
 *   molview.mount(hostEl, workspace, { mode, owner }) -> Promise<handle>
 *   handle = { openMolecule(fileOrText), exportFile(), undo(),         // WRITE
 *              setFrame(i), getFrame(i), frameCount(), currentFrame(), // FRAMES (§14.5)
 *              play(opts), pause(), isPlaying(),                       //   navigation + playback
 *              setArrows(arrows), setLabels(labels),                  //   overlays (§14.5.1)
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
        // The viewer chrome -- the isolate toggle + the trajectory bar -- is PLACED INTO the
        // embed's knob bar (the View/Export row) after the viewer embeds (see onReady), so it
        // lines up on ONE row with View/Export, within the viewer width.  Created here, parented
        // there.  Gated: shown ONLY for a trajectory (frameCount > 1).
        const vcHost     = _el("span", "viewer-toggles");
        const fcHost     = _el("div", "molview-frame-controls");   // trajectory bar (§14.5)
        fcHost.hidden = true;   // shown by mountFrameControls only when a trajectory is loaded
        viewerCol.appendChild(wrap);
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
        return { card: card, panelHost: panelHost, vcHost: vcHost, fcHost: fcHost,
                 foldBtn: foldBtn, viewerHost: viewerHost };
    }

    async function mount(hostEl, workspace, opts) {
        opts = opts || {};
        const mode   = opts.mode || "modify";
        const owner  = opts.owner || (workspace && workspace.owner) || null;
        const mb     = root.molbuilder || {};
        const selApi = mb.selection;
        const mvApi  = mb.molview;
        // DATA comes from MolView's own data model; `workspace` is only the persistence layer.
        const data   = (mb.molview && mb.molview.data) || workspace;
        const store  = data && data.selection;

        if (!hostEl || !workspace || !store) return null;
        if (!selApi || typeof selApi.mountPanel !== "function") return null;

        // Tell the workspace who this molview belongs to, so IT namespaces its saving
        // points (sessionStorage mirror key + the on-disk state-file id) by `owner` --
        // isolating this consumer's persisted session from any other's.  Namespacing lives
        // in the workspace persistence layer, not here (dispatcher.useNamespace).  Feature-
        // detected so a workspace without the method still mounts (unnamespaced).
        if (owner && workspace && typeof workspace.useNamespace === "function") {
            workspace.useNamespace(owner);
        }

        const cleanups = [];

        // Overlay controller (§14.5.1): MolView DRAWS overlays the consumer hands it (arrows /
        // labels, via the handle's setArrows/setLabels) -- it never generates them.
        let _overlays = null;
        // Frame playback state -- declared HERE (before mountFrameControls, which reads _loop via
        // its getLoop callback during setup) to avoid a temporal-dead-zone on _loop.
        let _playTimer = null, _loop = true;   // _loop: playback + single-step wrap at the ends

        // Resolve the fused card + its sub-hosts.  PRE-BUILT card (Modify's template) -> wire
        // the existing panel/toggles/fold; that host owns its own viewer + render.  EMPTY
        // host -> molview BUILDS the card, EMBEDS the viewer, and OWNS the render loop.
        let card = hostEl.classList && hostEl.classList.contains("molview-card")
            ? hostEl
            : ((hostEl.closest && hostEl.closest(".molview-card")) || null);
        let panelHost = card && card.querySelector(".molview-panel");
        let vcHost    = card && card.querySelector(".viewer-toggles");
        let fcHost    = card && card.querySelector(".molview-frame-controls");
        let foldBtn   = card && card.querySelector(".molview-fold-btn");

        if (!panelHost) {
            // EMPTY host: build the DOM, embed the viewer, own the render (the full component).
            const built = _buildCard(hostEl);
            card = built.card; panelHost = built.panelHost;
            vcHost = built.vcHost; fcHost = built.fcHost; foldBtn = built.foldBtn;
            const viewer = mb.viewer;
            if (viewer && typeof viewer.embed === "function") {
                // Wire the render loop once the viewer handle is ready; molview owns it.
                viewer.embed(built.viewerHost, {
                    onReady: function (h) {
                        // Test hook: expose the viewer handle so e2e can read what was drawn
                        // (the owner never sees it -- it's not on the returned handle).
                        built.viewerHost.__molview_test_handle = h;
                        // Register the embed as molview.data's view-state target (§20): the
                        // model reads THIS handle (not the retired ``molbuilder.modify.handle``
                        // global), so camera / style / axes / labels persist + restore per
                        // owner-namespace across a tab round-trip.  Applies any view state a
                        // pre-embed session restore stashed.
                        if (data && typeof data.attachViewHandle === "function") {
                            data.attachViewHandle(h);
                            cleanups.push(function () {
                                try { data.detachViewHandle && data.detachViewHandle(h); }
                                catch (_) {}
                            });
                        }
                        if (mvApi && typeof mvApi.mountRender === "function") {
                            const rc = mvApi.mountRender(h, data, store,
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
                        // Overlay controller (§14.5.1): MolView draws the arrows/labels the
                        // CONSUMER hands it (handle.setArrows / setLabels) and re-applies them
                        // across per-frame redraws -- it never generates or normalizes them.
                        if (mvApi && typeof mvApi.mountOverlays === "function") {
                            _overlays = mvApi.mountOverlays(h, store);
                            cleanups.push(function () {
                                try { _overlays && _overlays.dispose(); } catch (_) {}
                            });
                        }
                        // the isolate toggle joins the embed's untitled toggle group in the
                        // View menu (one group with Show axes/labels/overlay/unit cell).  The
                        // trajectory bar goes on the knob-bar ROW next to View/Export (one line),
                        // shown only for a trajectory (frame-controls gates itself on frameCount).
                        const knobs = built.viewerHost.querySelector(".mol-viewer-knobs");
                        const toggleGroup = built.viewerHost.querySelector(
                            ".mol-viewer-menu-view .mol-viewer-menu-toggles");
                        const viewMenuBody = built.viewerHost.querySelector(
                            ".mol-viewer-menu-view .mol-viewer-menu-body");
                        if (toggleGroup)       { toggleGroup.appendChild(vcHost); }
                        else if (viewMenuBody) { viewMenuBody.appendChild(vcHost); }
                        else if (knobs)        { knobs.appendChild(vcHost); }   // fallback: no View menu
                        if (knobs) { knobs.appendChild(fcHost); }
                    },
                });
            }
        }

        // Panel (atom selection + the Cell page), bound to the workspace's store.
        const panelMount = await selApi.mountPanel(panelHost, { store: store, mode: mode });
        if (!panelMount || !panelMount.panel) return null;   // mountPanel showed its own banner
        cleanups.push(function () { try { panelMount.dispose && panelMount.dispose(); } catch (_) {} });

        // View-controls bar (the isolate toggle) -- same store, no parallel state.
        if (vcHost && mvApi && typeof mvApi.mountViewControls === "function") {
            const vc = mvApi.mountViewControls(vcHost, store);
            cleanups.push(function () { try { vc.dispose && vc.dispose(); } catch (_) {} });
        }

        // Frame controls bar (§14.5) -- play/pause + slider + counter.  MolView renders it (like
        // the view-toggles); hidden until a trajectory is loaded (frameCount > 1).  Overlays are
        // NOT here: they are the consumer's (handle.setArrows / setLabels), not a viewer toggle.
        if (fcHost && mvApi && typeof mvApi.mountFrameControls === "function") {
            const fc = mvApi.mountFrameControls(fcHost, {
                setFrame:     function (i) { return data.setFrame(i); },
                frameCount:   function () {
                    return (typeof data.frameCount === "function") ? data.frameCount() : 0;
                },
                currentFrame: function () {
                    return (typeof data.currentFrame === "function") ? data.currentFrame() : 0;
                },
                play:      _play,
                pause:     _stopPlay,
                isPlaying: function () { return _playTimer != null; },
                getLoop:   function () { return _loop; },
                setLoop:   function (on) { _loop = !!on; },
            }, store);
            cleanups.push(function () { try { fc.dispose && fc.dispose(); } catch (_) {} });
        }

        // §20 view flush: persistence is push-only, and a pure view change (camera rotate, menu
        // toggle) has no push trigger -- so mirror the live view state on navigation away so it
        // survives a tab round-trip.  pagehide covers navigating to another page; visibilitychange
        // ('hidden') covers backgrounding the tab.  View-ONLY (data.flushViewState patches just the
        // mirror's view slot), so it never touches structure persistence.
        const _flushView = function () {
            try { if (data && typeof data.flushViewState === "function") data.flushViewState(); }
            catch (_) {}
        };
        const _onVisibility = function () {
            if (root.document && root.document.visibilityState === "hidden") _flushView();
        };
        // Guarded: the Node mount-test harness stubs `root` without a DOM event API.
        if (typeof root.addEventListener === "function") {
            root.addEventListener("pagehide", _flushView);
            const _doc = root.document;
            if (_doc && typeof _doc.addEventListener === "function") {
                _doc.addEventListener("visibilitychange", _onVisibility);
            }
            cleanups.push(function () {
                root.removeEventListener("pagehide", _flushView);
                if (_doc && typeof _doc.removeEventListener === "function") {
                    _doc.removeEventListener("visibilitychange", _onVisibility);
                }
            });
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
        // NOTE: the panel's height matching the viewer square is done PURELY in CSS
        // (fused-layout.css: the viewer and panel share `--viewer-extent`) -- no JS/
        // ResizeObserver.  See the fused card's §7.1 system in ui-design-contract.md.

        // ---- The owner-facing handle (§D) --------------------------------------------- //
        // The owner reads the molecule + reacts to changes THROUGH molview -- never storage
        // directly (§A single door).  Reads return copies (the workspace accessors
        // defensive-copy, workspace-contract §1.2.1).  `onChange` is the ONE change channel
        // (§E rule 4): the owner subscribes here instead of reaching for ws.subscribe /
        // store.subscribe itself.  (load / save / undo -- the WRITE side -- land in B2.)
        // Frame playback + overlay-toggle helpers -- MolView owns them; both the returned
        // handle AND the frame-controls bar (mountFrameControls) drive them.
        function _stopPlay() {
            if (_playTimer != null) { root.clearInterval(_playTimer); _playTimer = null; }
        }
        function _play(opts) {
            opts = opts || {};
            if (typeof data.frameCount !== "function" || data.frameCount() <= 1) return;
            const fps = (typeof opts.fps === "number" && opts.fps > 0) ? opts.fps : 10;
            _stopPlay();
            _playTimer = root.setInterval(function () {
                const n = data.frameCount();
                if (n <= 1) { _stopPlay(); return; }
                const next = data.currentFrame() + 1;
                if (next < n) { data.setFrame(next); }
                else if (_loop) { data.setFrame(0); }            // loop: wrap to the start
                else { data.setFrame(n - 1); _stopPlay(); }      // no loop: stop at the last frame
            }, 1000 / fps);
        }
        const _offs = [];   // onChange subscriptions, torn down on dispose
        return {
            // WRITE side (§D): the owner asks molview to load / save / undo; molview asks the
            // WORKSPACE (the single door) and the render reacts to the resulting workspace
            // change (§18.2) -- the owner never touches storage or triggers a redraw itself.
            openMolecule: function (fileOrText) {
                // "Open this molecule."  ONE door: data.openMolecule dispatches
                // { text, filename } vs a project-file path string, and atomically
                // replaces the whole model (and resets the session-state timeline).
                // (Named to MATCH data.openMolecule -- the handle's `save`/`load`
                // would collide with the data model's timeline save/load(delta).)
                return (typeof data.openMolecule === "function")
                    ? data.openMolecule(fileOrText)
                    : Promise.reject(new Error("molview.openMolecule: data.openMolecule missing"));
            },
            exportFile: function () {
                // "Serialize this molecule" -> {xyz, sidecar} bytes (openMolecule's inverse).
                return (typeof data.exportFile === "function")
                    ? Promise.resolve(data.exportFile())
                    : Promise.reject(new Error("molview.exportFile: data.exportFile missing"));
            },
            undo: function () {
                // Retract one session-state checkpoint (= data.load(-1); §19.5).
                return (typeof data.undo === "function")
                    ? data.undo()
                    : Promise.reject(new Error("molview.undo: data.undo missing"));
            },
            // ---- Frames -- the coordinate time axis (workspace §1.5, molview §14.5) -------- //
            // Navigation delegates to the workspace (the data owner); MolView owns the playback
            // timer.  A frame's coord swap notifies the store, so the render redraws on its own.
            setFrame: function (i) {
                return (typeof data.setFrame === "function") ? data.setFrame(i) : undefined;
            },
            getFrame: function (i) {
                return (typeof data.getFrame === "function") ? data.getFrame(i) : null;
            },
            frameCount: function () {
                return (typeof data.frameCount === "function") ? data.frameCount() : 0;
            },
            currentFrame: function () {
                return (typeof data.currentFrame === "function") ? data.currentFrame() : 0;
            },
            play:  function (opts) { _play(opts); },
            pause: function () { _stopPlay(); },
            isPlaying: function () { return _playTimer != null; },
            // Overlay API (§14.5.1) -- the consumer hands MolView what to DRAW; MolView draws it
            // and re-applies it across per-frame redraws (it never generates arrows/labels).
            // `arrows` = [{start,end,color,radius}, …]; `labels` = a setLabels spec (or false).
            setArrows: function (arrows) { if (_overlays) _overlays.setArrows(arrows); },
            setLabels: function (labels) { if (_overlays) _overlays.setLabels(labels); },
            getStructure: function () {
                return (typeof data.getStructure === "function")
                    ? data.getStructure() : null;
            },
            getSelection: function () {
                const s = store.getState();
                return (s && Array.isArray(s.indices)) ? s.indices.slice() : [];
            },
            onChange: function (fn) {
                if (typeof fn !== "function" || typeof data.subscribe !== "function") {
                    return function () {};
                }
                const off = data.subscribe(fn);
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

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
 *   handle = { installMolecule({text}), exportFile(), undo(),          // WRITE
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
 *    loop (the render engine, engine/engine.js): the full component.  This is the target
 *    every consumer converges to.
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
        // The trajectory bar is PLACED INTO the embed's knob bar (the View/Export row)
        // after the viewer embeds (see onReady), so it lines up on ONE row with
        // View/Export, within the viewer width.  Created here, parented there.  Gated:
        // shown ONLY for a trajectory (frameCount > 1).  (The isolate toggle is added to
        // the embed's own toggle path via handle.addViewToggle -- no host element here.)
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
        return { card: card, panelHost: panelHost, fcHost: fcHost,
                 foldBtn: foldBtn, viewerHost: viewerHost };
    }

    // Uniform mount contract: mount() ALWAYS resolves to a handle carrying
    // ``dispose`` (never a sentinel null).  A failure returns a no-op-disposer
    // handle with ``ok:false`` + a reason, so a consumer branches on ``handle.ok``
    // and can still call ``handle.dispose()`` unconditionally.  Success handles
    // carry ``ok:true`` (see the return block below).
    function _failMount(msg) {
        if (root.console) root.console.warn("[molview.mount] " + msg);
        return { ok: false, error: msg, dispose: function () {} };
    }

    async function mount(hostEl, workspace, opts) {
        opts = opts || {};
        const mode   = opts.mode || "modify";
        const owner  = opts.owner || (workspace && workspace.owner) || null;
        const mb     = root.molbuilder || {};
        const selApi = mb.molview && mb.molview.selection;
        const mvApi  = mb.molview;
        // DATA comes from MolView's own data model; `workspace` is only the persistence layer.
        const data   = (mb.molview && mb.molview.data) || workspace;
        const store  = data && data.selection;

        if (!hostEl || !workspace || !store)
            return _failMount("missing host / workspace / data store");
        if (!selApi || typeof selApi.mountPanel !== "function")
            return _failMount("selection.mountPanel unavailable (load order?)");

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
        let fcHost    = card && card.querySelector(".molview-frame-controls");
        let foldBtn   = card && card.querySelector(".molview-fold-btn");

        if (!panelHost) {
            // EMPTY host: build the DOM, embed the viewer, own the render (the full component).
            const built = _buildCard(hostEl);
            card = built.card; panelHost = built.panelHost;
            fcHost = built.fcHost; foldBtn = built.foldBtn;
            const viewer = mb.viewer;
            if (viewer && typeof viewer.embed === "function") {
                // Wire the render loop once the viewer handle is ready; molview owns it.
                viewer.embed(built.viewerHost, {
                    // The fused card sizes its viewer square itself (fused-layout.css
                    // --viewer-extent), so the embed must FILL that square, not fall back to
                    // its standalone default height (clamp(360px,52vh,500px)) which overflows a
                    // small square and clips the molecule.  "100%" = "fill the host you're
                    // given" -- the square is the single source of truth, no magic number here.
                    card: { height: "100%" },
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
                        // The render ENGINE (molview-render-streamline.md) is the SINGLE render
                        // place: data.attachEngine feeds it StructureData; it reads the view flags
                        // from the store and draws through embedIo. The engine (engine/engine.js)
                        // is now the single render place; the old pre-engine render controllers
                        // are deleted.
                        if (mvApi && mvApi.engine && typeof mvApi.engine.create === "function") {
                            const engine = mvApi.engine.create(h, { store: store });
                            if (data && typeof data.attachEngine === "function") {
                                data.attachEngine(engine);
                            }
                            cleanups.push(function () {
                                try { if (data && data.detachEngine) data.detachEngine(engine); } catch (_) {}
                                try { engine && engine.dispose && engine.dispose(); } catch (_) {}
                            });
                        }
                        // Viewer-adapter: click-to-select ONLY. Halos are the engine's job now, so
                        // the adapter does NOT paint overlays (paintHalos:false); it just routes
                        // in-window picks to the store (and stands down under isolate itself).
                        const adapter = selApi && selApi.viewerAdapter;
                        if (adapter && typeof adapter.attach === "function") {
                            const ah = adapter.attach(h, { store: store, mode: mode, paintHalos: false });
                            cleanups.push(function () {
                                try { ah && ah.dispose && ah.dispose(); } catch (_) {}
                            });
                        }
                        // isolate ("show selected only") toggle -- store-backed (writes
                        // store.setIsolate; the ENGINE reads store.isolate and renders it as a
                        // structural regen, so a trajectory SURVIVES isolate). The other view
                        // toggles (axes/labels/cell/overlay) remain the embed's built-ins for now;
                        // rerouting them to store flags is the follow-up (task #64).
                        const adapterMod = selApi && selApi.viewerAdapter;
                        if (adapterMod && typeof adapterMod.isolateToggle === "function"
                            && typeof h.addViewToggle === "function") {
                            const iso = h.addViewToggle(adapterMod.isolateToggle(store));
                            cleanups.push(function () {
                                try { iso && iso.dispose && iso.dispose(); } catch (_) {}
                            });
                        }
                        // Measurement readout -- a SEPARATE interaction layer (§2.4), not the render
                        // machine. It reads the panel selection + the CURRENT frame's clean coords
                        // (data.getFrame -> engine.getFrame) and repaints on selection OR frame
                        // change. It owns its own overlay in the viewer window.
                        if (mvApi && typeof mvApi.mountMeasurementOverlay === "function") {
                            const meas = mvApi.mountMeasurementOverlay(built.viewerHost, {
                                store: store,
                                coordsProvider: function () {
                                    return (data && typeof data.getFrame === "function")
                                        ? (data.getFrame(data.currentFrame()) || []) : [];
                                },
                            });
                            const measFrameUnsub = (data && typeof data.onFrameChange === "function")
                                ? data.onFrameChange(function () { try { meas.render && meas.render(); } catch (_) {} })
                                : function () {};
                            cleanups.push(function () {
                                try { measFrameUnsub(); } catch (_) {}
                                try { meas && meas.dispose && meas.dispose(); } catch (_) {}
                            });
                        }
                        const knobs = built.viewerHost.querySelector(".mol-viewer-knobs");
                        if (knobs) { knobs.appendChild(fcHost); }
                    },
                });
            }
        }

        // Panel (atom selection + the Cell page), bound to the workspace's store.
        const panelMount = await selApi.mountPanel(panelHost, { store: store, mode: mode });
        if (!panelMount || !panelMount.panel)
            return _failMount("selection panel failed to mount (see banner)");
        cleanups.push(function () { try { panelMount.dispose && panelMount.dispose(); } catch (_) {} });

        // (isolate "show selected only" is added as view toggle #5 via the embed's
        // addViewToggle in the onReady above -- no separate control bar.)

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
            // Subscribe the bar to the DATA model's dedicated frame-change channel --
            // NOT the selection store.  A native frame swap fires this channel only, so
            // the slider/counter track the shown frame WITHOUT re-rendering the
            // selection panel (which would steal focus from a filter input mid-play).
            }, { subscribe: (typeof data.onFrameChange === "function")
                    ? data.onFrameChange
                    : (store && store.subscribe && store.subscribe.bind(store)) });
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
            ok: true,   // uniform mount contract: a live handle -> ok:true (see _failMount)
            // (No owner-facing setBusy: busy is gated SOLELY by the render streamline
            // -- an owner changes state, the render line turns the scrim on + recovers.
            // There is one busy path, and consumers do not touch it.  §18.6.)
            // WRITE side (§D): the owner asks molview to load / save / undo; molview asks the
            // WORKSPACE (the single door) and the render reacts to the resulting workspace
            // change (§18.2) -- the owner never touches storage or triggers a redraw itself.
            installMolecule: function (input) {
                // "Install this molecule from TEXT into the model" ({ text, filename };
                // generators / demos).  Atomically replaces the whole model + resets the
                // session-state timeline.  A project-file PATH load goes through the
                // format-aware sidebar door (projects.parser.openMolecule), NOT the
                // mounted handle -- files are the projects package's job
                // (structure-load-save-contract.md).
                return (typeof data.installMolecule === "function")
                    ? data.installMolecule(input)
                    : Promise.reject(new Error("molview.installMolecule: data.installMolecule missing"));
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
            // Per-FRAME force arrows for a trajectory (§14.5.1): the consumer builds the whole
            // arrowsPerFrame array ONCE (on an overlay-option change) and hands it here; MolView
            // bakes it into the native animation so playback draws arrows[frame] with no
            // per-frame synthesis.  `arrowsPerFrame` = [[{start,end,color,radius}…] per frame].
            setFrameArrows: function (arrowsPerFrame) {
                if (typeof data.setFrameArrows === "function") data.setFrameArrows(arrowsPerFrame);
            },
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
    // Module-init contract (molbuilder-runtime.js): register so consumers can
    // ``whenReady("molview")`` instead of racing on the namespace, like every
    // other module (projects / inspectors / selection.* / molview.data).
    if (root.molbuilder.runtime
        && typeof root.molbuilder.runtime.register === "function") {
        root.molbuilder.runtime.register("molview", root.molbuilder.molview);
    }
})(typeof window !== "undefined" ? window : this);

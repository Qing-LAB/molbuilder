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
 *              setCurrentFrame(i), getFrameAllAtoms(i),                        // FRAMES (§14.5)
 *              frameCount(), currentFrame(),                            //   (all atoms, pre-isolate)
 *              play(opts), pause(), isPlaying(),                       //   navigation + playback
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
"use strict";

// The concealed viewer-overlay framework -- one consistent, token-styled corner overlay for the
// whole module (import, not a global, so the dependency is legible).
// The 3Dmol viewer embed -- now an ES module in the graph (was the classic `mb.viewer` global).
import { viewer as _embedViewer } from "./_seal.js";

const root = (typeof window !== "undefined") ? window : globalThis;

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

    // Run a teardown stack LIFO (reverse of registration) -- attachments detach before the
    // embed disposes, and the card DOM (registered first) is removed last.  Every entry is
    // isolated: one throwing cleanup can't strand the rest.
    function _drain(cleanups) {
        for (var i = cleanups.length - 1; i >= 0; i--) {
            try { cleanups[i](); } catch (_) {}
        }
        cleanups.length = 0;
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
        if (!selApi || !selApi.panel || typeof selApi.panel.mount !== "function")
            return _failMount("selection panel unavailable (load order?)");

        // Tell the workspace who this molview belongs to, so IT namespaces its saving
        // points (sessionStorage mirror key + the on-disk state-file id) by `owner` --
        // isolating this consumer's persisted session from any other's.  Namespacing lives
        // in the workspace persistence layer, not here (dispatcher.useNamespace).  Feature-
        // detected so a workspace without the method still mounts (unnamespaced).
        if (owner && workspace && typeof workspace.useNamespace === "function") {
            workspace.useNamespace(owner);
        }

        const cleanups = [];

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

        let _deferredEmbed = null;
        if (!panelHost) {
            // A prior too-narrow attempt may have left its error card in this host
            // (a consumer that never disposes a failed handle). Remove it before
            // building, so remount-after-resize can never stack error cards.
            const stale = hostEl.querySelectorAll
                ? hostEl.querySelectorAll(".molview-card--unmountable") : [];
            for (let i = 0; i < stale.length; i++) {
                try { stale[i].parentNode.removeChild(stale[i]); } catch (_) {}
            }
            // EMPTY host: BUILD the card now; DEFER the viewer embed until AFTER the width
            // contract check below, so a too-narrow host can never orphan a started WebGL embed.
            const built = _buildCard(hostEl);
            card = built.card; panelHost = built.panelHost;
            fcHost = built.fcHost; foldBtn = built.foldBtn;
            _deferredEmbed = built;
            // molview BUILT this card, so dispose must REMOVE it (a pre-built template card is
            // host-owned and stays).  Registered FIRST so the LIFO teardown removes the DOM
            // LAST -- after the embed and every attachment let go of it.  Without this a
            // mount/dispose cycle (the Results inspector re-mounts per file view) stacked a
            // dead card per view.
            cleanups.push(function () {
                try { if (built.card.parentNode) built.card.parentNode.removeChild(built.card); }
                catch (_) {}
            });
        }

        // SIZING CONTRACT (embedded module, fused-layout.css --molview-min-width): the card needs at
        // least (viewer square min + fold + panel min) to render.  If the PARENT hands it LESS, that
        // is a broken embed -- render a blank window + a clear message instead of overflowing (the
        // module owns its minimum; the parent owns the actual width).  Guarded on getComputedStyle
        // (absent in the node test) + a laid-out host (width 0 == display:none/detached -> skip, not
        // an error).
        if (card && typeof root.getComputedStyle === "function" && card.parentElement
                && typeof card.parentElement.getBoundingClientRect === "function") {
            const _need = parseFloat(root.getComputedStyle(card).minWidth) || 0;
            const _have = card.parentElement.getBoundingClientRect().width;
            if (_need > 0 && _have > 0 && _have < _need - 1) {
                while (card.firstChild) card.removeChild(card.firstChild);
                // Drop the min-width so the error message itself fits the too-small host.
                if (card.classList) card.classList.add("molview-card--unmountable");
                const emsg = _el("div", "molview-embed-error");
                emsg.setAttribute("role", "alert");
                emsg.textContent = "MolView needs at least " + Math.round(_need)
                    + " px of width to show the structure; this area is only "
                    + Math.round(_have) + " px. Widen the window or container.";
                card.appendChild(emsg);
                const fh = _failMount("host too narrow (" + Math.round(_have)
                    + "px < " + Math.round(_need) + "px min)");
                // The error card deliberately STAYS visible (it is the message), but
                // the handle's dispose removes it -- honoring the "always call
                // handle.dispose()" contract instead of the no-op disposer.
                fh.dispose = function () { _drain(cleanups); };
                return fh;
            }
        }

        // Contract passed -> embed the viewer NOW (deferred from the build above so a failed
        // width check can never leak a started embed / dangling engine+adapter attachments).
        if (_deferredEmbed) {
            const built = _deferredEmbed;
            const viewer = _embedViewer;
            if (viewer && typeof viewer.embed === "function") {
                // Wire the render loop once the viewer handle is ready; molview owns it.
                // The embed handle is returned SYNCHRONOUSLY (onReady fires on the next
                // microtask); capture it so dispose can tear the embed down -- without this,
                // every mount/dispose cycle leaked a WebGL context (browsers cap ~8-16, then
                // force-lose the oldest -> black viewers after enough Results inspections).
                const embedHandle = viewer.embed(built.viewerHost, {
                    // Host-driven view toggles: molview owns the view flags in the selection
                    // store, so the embed must NOT build its own stateful axes/labels/overlay/
                    // cell toggles (that second copy was the desync source).  molview injects
                    // store-backed specs below; the embed keeps only its stateless reset.
                    storeToggles: true,
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
                        // place: data.attachRenderEngine feeds it StructureData; it reads the view flags
                        // from the store and draws through embedIo. The engine (engine/engine.js)
                        // is now the single render place; the old pre-engine render controllers
                        // are deleted.
                        // NOTE the namespace: `renderEngine`, never bare `engine` -- across this
                        // project an *engine* is a CALCULATION backend (SIESTA, PySCF).  This
                        // lookup read `mvApi.engine` until 2026-07-30 and silently found nothing
                        // once the module renamed its publish, so the viewer mounted and never
                        // drew: the guard below turns a missing factory into a no-op.
                        if (mvApi && mvApi.renderEngine
                            && typeof mvApi.renderEngine.create === "function") {
                            const engine = mvApi.renderEngine.create(h, { store: store });
                            if (data && typeof data.attachRenderEngine === "function") {
                                data.attachRenderEngine(engine);
                            }
                            cleanups.push(function () {
                                try { if (data && data.detachRenderEngine) data.detachRenderEngine(engine); } catch (_) {}
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
                        // View toggles -- ALL store-backed (ONE authority: the button writes a
                        // store flag, the ENGINE reads it and draws; no embed-local copy).
                        //   * axes/labels/overlay/cell -> setViewFlag (overlay-refresh render)
                        //   * isolate -> setIsolate (structural regen, so a trajectory SURVIVES it)
                        // The embed builds none of these itself (storeToggles above); it renders
                        // each injected spec as a rail button through addViewToggle.
                        const adapterMod = selApi && selApi.viewerAdapter;
                        if (adapterMod && typeof h.addViewToggle === "function") {
                            const specs = [];
                            if (typeof adapterMod.viewFlagToggles === "function") {
                                specs.push.apply(specs, adapterMod.viewFlagToggles(store));
                            }
                            if (typeof adapterMod.isolateToggle === "function") {
                                specs.push(adapterMod.isolateToggle(store));
                            }
                            specs.forEach(function (spec) {
                                const t = h.addViewToggle(spec);
                                cleanups.push(function () {
                                    try { t && t.dispose && t.dispose(); } catch (_) {}
                                });
                            });
                        }
                        // Measurement readout -- a SEPARATE interaction layer (§2.4), not the render
                        // machine. It reads the panel selection + the CURRENT frame's clean coords
                        // (data.getFrameAllAtoms: every atom, original order) and repaints on selection OR frame
                        // change. It owns its own overlay in the viewer window.
                        if (mvApi && typeof mvApi.mountMeasurementOverlay === "function") {
                            const meas = mvApi.mountMeasurementOverlay(built.viewerHost, {
                                store: store,
                                coordsProvider: function () {
                                    return (data && typeof data.getFrameAllAtoms === "function")
                                        ? (data.getFrameAllAtoms(data.currentFrame()) || []) : [];
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
                // Tear the embed itself down on dispose (WebGL context + its DOM + listeners).
                // Registered here -- AFTER the card-removal cleanup, BEFORE the onReady
                // attachments -- so the LIFO teardown runs: attachments detach -> embed
                // disposes -> card DOM is removed.
                cleanups.push(function () {
                    try { embedHandle && embedHandle.dispose && embedHandle.dispose(); }
                    catch (_) {}
                });
            }
        }

        // Panel (atom selection + the Cell page), bound to the workspace's store.
        // The await is guarded: a REJECTED mountPanel must drain exactly like the
        // falsy-return failure below -- otherwise the already-running embed +
        // attachments leak (WebGL context, engine subscription) on the throw path.
        let panelMount = null;
        try {
            panelMount = await mountPanel(panelHost, { store: store, mode: mode });
        } catch (e) {
            _drain(cleanups);
            return _failMount("selection panel mount threw: "
                + ((e && e.message) || String(e)));
        }
        if (!panelMount || !panelMount.panel) {
            // The embed + attachments are already running -- drain the teardown stack so a
            // failed mount can NEVER orphan a started WebGL embed (the same guarantee the
            // width check gives the too-narrow path).
            _drain(cleanups);
            return _failMount("selection panel failed to mount (see banner)");
        }
        cleanups.push(function () { try { panelMount.dispose && panelMount.dispose(); } catch (_) {} });

        // §19.5 PERSISTENCE INDICATOR: reflect the suspend gate so the user SEES when saves are
        // deferred (a consumer bracketing a multi-step data operation via suspendPersist).  Hidden
        // unless suspended; a synchronous internal bracket (a modify op) toggles it 0->1->0 with no
        // paint between, so it only becomes visible during a genuine async deferral window.
        if (card && data && typeof data.onPersistStateChange === "function") {
            // The overlay FRAMEWORK owns placement + token styling; mount only binds it to the
            // persist-suspend signal.  Anchor to the VIEWER (always visible), NOT the card: when the
            // panel folds the card keeps its width but the panel area goes empty, so a card anchor
            // would strand the pill in that gap; the viewer square shows in every fold state.
            const _pindAnchor = (card.querySelector
                && (card.querySelector(".viewer-wrap") || card.querySelector(".molview-viewer")))
                || card;
            const persistOverlay = createViewerOverlay(_pindAnchor, {
                corner: "top-right", kind: "warn", role: "status", live: "polite",
                text: "⏸ Saving paused",
                title: "Saving is paused while a multi-step edit is in progress; "
                     + "your changes are saved when it finishes.",
            });
            try {
                persistOverlay.toggle(typeof data.isPersistSuspended === "function"
                                      && data.isPersistSuspended());
            } catch (_) {}
            const _offPersist = data.onPersistStateChange(function (on) { persistOverlay.toggle(on); });
            cleanups.push(function () {
                try { _offPersist && _offPersist(); } catch (_) {}
                persistOverlay.dispose();
            });
        }

        // (isolate "show selected only" is added as view toggle #5 via the embed's
        // addViewToggle in the onReady above -- no separate control bar.)

        // The frame surface (§14.5), defined ONCE.  Both consumers read frames through THIS
        // object: the frame-controls bar MolView renders itself, and the handle it returns to
        // the tab.  They used to carry two independently-written delegate sets over the same
        // molview.data doors -- two places for the same answer, which is how a frame count and
        // the frames a viewer can actually show drift apart (#35).  Navigation and the reads
        // belong to molview.data (the data owner); the playback TIMER is mount's own.
        const _frames = {
            setCurrentFrame: function (i) {
                return (typeof data.setCurrentFrame === "function")
                    ? data.setCurrentFrame(i) : undefined;
            },
            getFrameAllAtoms: function (i) {
                return (typeof data.getFrameAllAtoms === "function") ? data.getFrameAllAtoms(i) : null;
            },
            frameCount: function () {
                return (typeof data.frameCount === "function") ? data.frameCount() : 0;
            },
            currentFrame: function () {
                return (typeof data.currentFrame === "function") ? data.currentFrame() : 0;
            },
            play:      function (opts) { _play(opts); },
            pause:     function () { _stopPlay(); },
            isPlaying: function () { return _playTimer != null; },
        };

        // Frame controls bar (§14.5) -- play/pause + slider + counter.  MolView renders it (like
        // the view-toggles); hidden until a trajectory is loaded (frameCount > 1).  Force/label
        // overlays are NOT here: the render engine bakes them from the data's per-frame forces
        // (molview-render-streamline.md §2.4), not a consumer push or a viewer toggle.
        if (fcHost && mvApi && typeof mvApi.mountFrameControls === "function") {
            // The bar gets the shared frame surface plus the loop flag it alone owns (loop is
            // bar UI, deliberately not on the tab-facing handle's 15 keys).
            const fc = mvApi.mountFrameControls(fcHost, Object.assign({}, _frames, {
                getLoop:   function () { return _loop; },
                setLoop:   function (on) { _loop = !!on; },
            // Subscribe the bar to the DATA model's dedicated frame-change channel --
            // NOT the selection store.  A native frame swap fires this channel only, so
            // the slider/counter track the shown frame WITHOUT re-rendering the
            // selection panel (which would steal focus from a filter input mid-play).
            }), { subscribe: (typeof data.onFrameChange === "function")
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
        // Frame playback helpers -- MolView owns them; both the returned handle AND the
        // frame-controls bar (mountFrameControls) drive them.
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
                if (next < n) { data.setCurrentFrame(next); }
                else if (_loop) { data.setCurrentFrame(0); }            // loop: wrap to the start
                else { data.setCurrentFrame(n - 1); _stopPlay(); }      // no loop: stop at the last frame
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
            // The SAME _frames surface the frame-controls bar drives (defined once, above):
            // navigation and the reads belong to molview.data (the data owner), the playback
            // timer to MolView.  A frame's coord swap notifies the store, so the render redraws
            // on its own.
            setCurrentFrame: _frames.setCurrentFrame,
            getFrameAllAtoms:     _frames.getFrameAllAtoms,
            frameCount:   _frames.frameCount,
            currentFrame: _frames.currentFrame,
            play:         _frames.play,
            pause:        _frames.pause,
            isPlaying:    _frames.isPlaying,
            // (No consumer overlay API: the render engine bakes force arrows / labels from the
            // data's per-frame forces itself -- molview-render-streamline.md §2.4. The old
            // consumer-push doors -- setArrows / setLabels / setFrameArrows -- are gone.)
            getStructure: function () {
                return (typeof data.getStructure === "function")
                    ? data.getStructure() : null;
            },
            getSelection: function () {
                // ONE name, ONE shape: delegate to data.getSelection (the rich snapshot
                // {indices, pickOrder, mode, filters, combinator, isolate, view-flags}).
                // (Used to return a bare number[] built separately off the store -- the same
                // name with a second shape; zero consumers relied on it, so unified 2026-07.)
                return (typeof data.getSelection === "function") ? data.getSelection()
                    : { indices: [] };
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
                _drain(cleanups);   // LIFO: attachments -> embed -> card DOM
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

export { mount };

/* ── Composing the panel beside the viewer (§ 8) — was selection/mount-panel.js  */

const DEFAULT_PARTIAL = "/partials/selection-panel";

function _mb() { return root.molbuilder || {}; }

function _renderFailure(host, message) {
    const banner = document.createElement("div");
    banner.className = "selection-bootstrap-error";
    banner.setAttribute("role", "alert");
    banner.textContent = "Selection panel failed to load: " + message;
    host.innerHTML = "";
    host.appendChild(banner);
}

async function _fetchPartial(host, url) {
    try {
        const r = await fetch(url);
        if (!r.ok) {
            _renderFailure(host, "HTTP " + r.status + " from " + url);
            return false;
        }
        host.innerHTML = await r.text();
        return true;
    } catch (e) {
        _renderFailure(host, (e && e.message) ? e.message : String(e));
        return false;
    }
}

async function _resolveHandle(opts) {
    if (opts.viewerHandle) return opts.viewerHandle;
    const key = opts.viewerHandleKey;
    if (!key) return null;
    const rt = _mb().runtime;
    if (!rt || typeof rt.whenReady !== "function") return null;
    try { return await rt.whenReady(key); }
    catch (_) { return null; }
}

export async function mountPanel(host, opts) {
    opts = opts || {};
    if (!host) throw new Error("selection.mountPanel: host is required");
    const mb = _mb();
    const store = opts.store || (mb.molview && mb.molview.data && mb.molview.data.selection);

    // 1. fetch + insert the partial.
    const ok = await _fetchPartial(host, opts.partialUrl || DEFAULT_PARTIAL);
    if (!ok) return { panel: null, adapterHandle: null };

    // 2. mount the panel against the store.  The panel + adapter share ALL
    // state through the store (selection, filters, isolate), so the panel
    // needs no reference to the adapter -- no handle threading.
    const _panelMod = mb.molview && mb.molview.selection && mb.molview.selection.panel;
    if (!_panelMod || typeof _panelMod.mount !== "function") {
        _renderFailure(host, "selectionPanel module missing");
        return { panel: null, adapterHandle: null };
    }
    const panel = _panelMod.mount(host, { store: store, mode: opts.mode });

    // 3. attach the viewer-adapter to the viewer handle.
    let adapterHandle = null;
    const adapter = mb.molview && mb.molview.selection && mb.molview.selection.viewerAdapter;
    const handle = await _resolveHandle(opts);
    if (adapter && typeof adapter.attach === "function" && handle) {
        adapterHandle = adapter.attach(handle, { store: store, mode: opts.mode });
    }
    return { panel: panel, adapterHandle: adapterHandle };
}

// ── Transitional global (removed once every consumer imports this module) ──
if (typeof window !== "undefined") {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.molview = window.molbuilder.molview || {};
    window.molbuilder.molview.selection = window.molbuilder.molview.selection || {};
    window.molbuilder.molview.selection.mountPanel = mountPanel;
}

/* ── A badge in a corner of the 3D window — was _viewer-overlay.js ───────────── */

const CORNERS = { "top-right": 1, "top-left": 1, "bottom-right": 1, "bottom-left": 1 };

export function createViewerOverlay(anchorEl, opts) {
    opts = opts || {};
    // DOM-only (browser).  In a document-less context (node mount-test stub) return an inert handle
    // so callers need no guard of their own.
    if (typeof document === "undefined" || !anchorEl || typeof anchorEl.appendChild !== "function") {
        return { el: null, show: function () {}, hide: function () {},
                 toggle: function () {}, setText: function () {}, dispose: function () {} };
    }
    const corner = CORNERS[opts.corner] ? opts.corner : "top-right";
    const el = document.createElement("div");
    el.className = "molview-overlay molview-overlay--" + corner
                 + (opts.kind ? " molview-overlay--" + opts.kind : "");
    el.hidden = true;
    if (opts.role)  el.setAttribute("role", opts.role);
    if (opts.live)  el.setAttribute("aria-live", opts.live);
    if (opts.title) el.title = opts.title;
    if (opts.text != null) el.textContent = opts.text;
    anchorEl.appendChild(el);

    return {
        el: el,
        show:    function () { el.hidden = false; },
        hide:    function () { el.hidden = true; },
        toggle:  function (on) { el.hidden = !on; },
        setText: function (t) { el.textContent = t; },
        dispose: function () { try { el.remove(); } catch (_) { /* already detached */ } },
    };
}

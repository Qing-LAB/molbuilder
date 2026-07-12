/* Reusable selection-panel mount -- the fused molview+selection composition
 * (Phase 5, docs/protocols/atom-annotations.md § 6).
 *
 * Every molview tab puts a selection panel next to a viewer the same way:
 *   1. fetch the panel partial + insert it into `host`
 *   2. selectionPanel.mount(host, { store })
 *   3. viewerAdapter.attach(<viewer handle>, { store })
 *
 * modify/selection-bootstrap.js still owns the Modify-specific glue (candidate
 * state, Load button, mount-restore); this module is the shared CORE so the
 * Results inspectors mount selection with a few lines + an EPHEMERAL store
 * (selection.createEphemeralStore()) that never touches the workspace.
 *
 *   window.molbuilder.selection.mountPanel(host, opts) -> Promise<handle>
 *
 * opts:
 *   store           : store the panel + adapter drive.  DEFAULT = the workspace
 *                     singleton (Modify).  A readonly inspector passes
 *                     selection.createEphemeralStore() for an isolated selection.
 *   viewerHandle    : the embed handle to attach the adapter to (inspectors pass
 *                     their embed return value directly); OR
 *   viewerHandleKey : a runtime-registry key to await for the handle.  Ignored when
 *                     viewerHandle is given.  (Not used by Modify anymore -- molview.mount's
 *                     built-card path embeds the viewer + attaches the adapter itself.)
 *   mode            : "modify" | "readonly" (forwarded to mount/attach; reserved
 *                     for per-mode UI -- adapter behaviour is identical today).
 *   partialUrl      : override the panel partial URL (default below).
 *
 * Resolves to { panel, adapterHandle }.  adapterHandle is null when no viewer
 * handle is available.  On partial-fetch failure it renders an inline banner
 * (as the bootstrap does) and resolves { panel: null, adapterHandle: null }.
 */
(function (root) {
    "use strict";

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

    async function mountPanel(host, opts) {
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
        if (!mb.selectionPanel || typeof mb.selectionPanel.mount !== "function") {
            _renderFailure(host, "selectionPanel module missing");
            return { panel: null, adapterHandle: null };
        }
        const panel = mb.selectionPanel.mount(host, { store: store, mode: opts.mode });

        // 3. attach the viewer-adapter to the viewer handle.
        let adapterHandle = null;
        const adapter = mb.selection && mb.selection.viewerAdapter;
        const handle = await _resolveHandle(opts);
        if (adapter && typeof adapter.attach === "function" && handle) {
            adapterHandle = adapter.attach(handle, { store: store, mode: opts.mode });
        }
        return { panel: panel, adapterHandle: adapterHandle };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.selection = root.molbuilder.selection || {};
    root.molbuilder.selection.mountPanel = mountPanel;
})(typeof window !== "undefined" ? window : globalThis);

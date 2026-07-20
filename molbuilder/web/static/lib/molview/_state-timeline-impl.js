/* MolView state timeline — Save-state / Retract (undo) mechanics.  MolView-internal submodule.
 *
 * MODULE: molview (lib/molview/).  A concealed submodule of the MolView data model
 *   (data-model.js).  The factory mounts at ``window.molbuilder.molview._createStateTimeline``
 *   (browser global) AND exports ``createStateTimeline`` under CommonJS (node tests).  NOT a
 *   public surface — the data model owns the public ``save()`` / ``load()`` / ``undo()`` /
 *   ``state_index`` and delegates them here.  Sibling-module precedent: ``_frame-series.js``
 *   (``_createFrameSeries``).
 *
 * ROLE: the push-only INDEXED history mechanics (workspace-contract §4.7 / molview-module §19.5)
 *   — owns the timeline position (``stateIndex``), the ``uncommitted`` flag, the ordered
 *   save/load chain (``_pushPopChain``), and the persist / prune / Retract-navigation
 *   orchestration.  It is format-BLIND: the data model injects ``serialise`` (model -> snapshot)
 *   and ``applySnapshot`` (snapshot -> model); this submodule only decides WHEN / WHERE bytes are
 *   written or read, through the injected workspace accessor ``getWorkspace``.
 *
 * USED BY: lib/molview/data-model.js ONLY.  It builds ONE instance via
 *   ``_createStateTimeline({serialise, applySnapshot, getWorkspace, trace})`` and delegates the
 *   public save/load/undo/anchor + the live ``state_index`` / ``uncommitted`` getters to it; the
 *   model calls ``markUncommitted()`` on every data change (guarded by the model's own
 *   ``_applying`` flag).  The workspace persistence layer (lib/workspace/dispatcher.js,
 *   POST /api/state-timeline/*) is reached ONLY through the injected ``getWorkspace`` accessor.
 */
(function (root) {
    "use strict";

    // deps: { serialise: () -> snapshot,
    //         applySnapshot: (snap) -> void,
    //         getWorkspace: () -> workspace-layer | null,
    //         trace: (ev, extra) -> void }
    function createStateTimeline(deps) {
        deps = deps || {};
        var serialise     = deps.serialise;
        var applySnapshot = deps.applySnapshot;
        var getWorkspace  = deps.getWorkspace;
        var trace         = deps.trace || function () {};

        // Position in the tab's operation history; `_uncommitted` = the model changed since the
        // last `save` (what a `load(-1)` would discard).  save/load are SERIALIZED through
        // `_pushPopChain` so the index advances/retreats only after each workspace round-trip
        // resolves.
        var _stateIndex   = 0;
        var _uncommitted  = false;
        var _pushPopChain = Promise.resolve();
        // Timeline generation.  `anchor()` (opening a NEW molecule) bumps this to INVALIDATE any
        // save/load of the PREVIOUS molecule still queued or mid-round-trip: anchor prunes the old
        // {wid}.* files and resets the index off-chain, so a stale save/load must NOT move the
        // index or apply an old snapshot over the freshly-opened structure.  Each op captures the
        // generation when it is REQUESTED and aborts if it changed by the time it runs / resolves.
        var _generation   = 0;

        // The model calls this on every data change (guarded by its own `_applying` flag so a
        // snapshot restore doesn't spuriously mark the model uncommitted).
        function markUncommitted() { _uncommitted = true; }

        // The draft key a Save must drop (b1) — the stable sourceless-workspace id.
        function draftIdentity() {
            return { workspace_id: (getWorkspace() ? getWorkspace().workspaceId() : null) };
        }

        // Serialize save/load through _pushPopChain so state_index advances/retreats only after
        // each workspace round-trip resolves.  A rejected op must NOT poison the chain -- keep an
        // error-swallowing tail as the next op's predecessor, but return the real op promise so
        // the caller sees the rejection.
        function _enqueue(fn) {
            var op = _pushPopChain.then(fn);
            _pushPopChain = op.catch(function () { /* keep the chain alive */ });
            return op;
        }

        // save(delta=0) (§19.5): session-state save, index-delta parameterized.  Serialize the
        // current snapshot and persist it (MIRROR + disk) at state_index+delta; ON SUCCESS move
        // the index to that target and clear `uncommitted`.  delta>0 (a new checkpoint, e.g.
        // save(1)="Save state") additionally tail-deletes every abandoned index above the target
        // (the divergent tail left after a load(-1)).  delta=0 re-saves the current index in
        // place.  A rejected persist does NOT move the index.
        function save(delta) {
            delta = delta || 0;
            var gen = _generation;                       // capture at request time
            return _enqueue(function () {
                if (gen !== _generation) return;         // a new molecule superseded this save
                var ws = getWorkspace();
                if (!ws || typeof ws.persist !== "function") return;
                var wid    = ws.workspaceId();
                var target = _stateIndex + delta;
                var snap   = serialise();
                // serialise() stamps `state_index: stateIndex` (the CURRENT index),
                // but this snapshot is FILED at `target` (= _stateIndex + delta) and
                // becomes the session mirror.  Stamp it with `target` so the file's
                // self-reported index matches where it lives -- otherwise every saved
                // snapshot reports one-too-low, and a reload's load(0) (which reads
                // the mirror's internal state_index) restores _stateIndex off by one,
                // so the next Retract skips a saved state (the "retract loses states
                // after loading back" bug).
                snap.state.state_index = target;
                return Promise.resolve(
                    ws.persist(snap, snap, { workspace_id: wid, state_index: target })
                ).then(function () {
                    if (gen !== _generation) return;     // superseded during the round-trip
                    _stateIndex  = target;
                    _uncommitted = false;
                    if (delta > 0 && typeof ws.pruneStatesAbove === "function") {
                        // Fire-and-forget: this prunes indices strictly ABOVE the
                        // just-written `target`, so unlike the anchor case there is no
                        // delete-vs-write race to order (it never touches `target` or
                        // below).  Best-effort, like the write itself.
                        ws.pruneStatesAbove(wid, target);   // drop the abandoned tail
                    }
                });
            });
        }

        // load(delta=0) (§19.5): session-state restore, index-delta parameterized.
        //   delta=0  -> RELOAD / mount-restore: read the sessionStorage MIRROR (the current
        //              committed snapshot + its state_index) and apply it.  No index move beyond
        //              adopting the mirror's own state_index.  Synchronous mirror read.
        //   delta!=0 -> navigate the on-disk timeline: read {workspace_id, state_index+delta},
        //              APPLY that snapshot to the whole model, move the index to the target, and
        //              clear `uncommitted`.  load(-1)=Retract/undo.  No-op if the target < 0 or
        //              the snapshot is missing.  Re-mirrors the applied snapshot (MIRROR ONLY, no
        //              disk write) so a later reload restores THIS index.  Discards uncommitted
        //              changes.  Undo-only (a save after a load(-1) overwrites the abandoned tail).
        function load(delta) {
            delta = delta || 0;
            var gen = _generation;                       // capture at request time
            return _enqueue(function () {
                if (gen !== _generation) return;         // a new molecule superseded this load
                var ws = getWorkspace();
                if (!ws) return;
                if (delta === 0) {
                    var m = (typeof ws.readPersistedSnapshot === "function")
                        ? ws.readPersistedSnapshot() : null;
                    if (m) {
                        applySnapshot(m);
                        _stateIndex = (m.state && typeof m.state.state_index === "number")
                            ? m.state.state_index : _stateIndex;
                        _uncommitted = false;
                    }
                    return;
                }
                var target = _stateIndex + delta;
                // Retract semantics (delta < 0): UNCOMMITTED edits are a working state
                // sitting ABOVE the current committed checkpoint, so the FIRST step of a
                // Retract reverts them to that checkpoint -- it must NOT skip past a SAVED
                // checkpoint.  An uncommitted state therefore consumes the first unit of a
                // retract: from [checkpoint N + uncommitted edit], load(-1) restores N
                // (discards the edit) and only a SECOND load(-1) steps to N-1.  (This also
                // lets you undo uncommitted work sitting on the index-0 anchor, which used
                // to be a no-op because target went negative.)
                if (delta < 0 && _uncommitted) {
                    target += 1;
                }
                if (target < 0) return;                    // can't go below the anchor
                if (typeof ws.readState !== "function") return;
                var wid = ws.workspaceId();
                return Promise.resolve(
                    ws.readState({ workspace_id: wid, state_index: target })
                ).then(function (snap) {
                    if (gen !== _generation) return;   // superseded during the round-trip
                    if (snap == null) return;   // missing history file -> no-op, index unchanged
                    applySnapshot(snap);
                    _stateIndex  = target;
                    _uncommitted = false;
                    if (typeof ws.persist === "function") {
                        // MIRROR ONLY (snapshotBlob=null) -- so a later reload restores THIS index.
                        return ws.persist(snap, null, { workspace_id: wid, state_index: target });
                    }
                });
            });
        }

        // Undo IS `load(-1)` (§19.5) -- kept as an alias so the mount handle's `undo()` works.
        function undo() { return load(-1); }

        // `openMolecule` anchors a FRESH timeline (§19.5): prune every existing {workspace_id}.* state
        // file, reset state_index to 0, and write the loaded structure as the index-0 anchor.
        // This is the ONE automatic write; everything after is explicit save(delta).
        function anchor() {
            // Supersede any save/load of the PREVIOUS molecule still queued or mid-round-trip:
            // bumping the generation makes each of them abort (it checks `gen !== _generation`)
            // instead of moving the index or applying a stale snapshot over the new structure.
            _generation++;
            var ws = getWorkspace();
            _stateIndex  = 0;
            _uncommitted = false;
            if (!ws || typeof ws.persist !== "function") return Promise.resolve();
            var wid = ws.workspaceId();
            var snap = serialise();
            // ORDER MATTERS: pruneStatesAbove(-1) DELETES every {wid}.* state file
            // (the old timeline).  It MUST fully complete BEFORE the index-0 anchor
            // write is ISSUED -- otherwise the two HTTP calls race and a late-landing
            // delete-all unlinks the just-written anchor, so a later Retract to index
            // 0 reads a missing file and no-ops (the intermittent "retract never
            // returns to the opened state" hang).  ``pruneStatesAbove`` returns its
            // fetch promise, so awaiting it before calling ``persist`` is sufficient;
            // the anchor write itself stays best-effort/fire-and-forget (persist is
            // crash-safety, not a blocking write -- workspace-contract), and by the
            // time any Retract reads index 0 it has long since landed.
            trace("anchor:prune:issue", { wid: wid });
            var prune = (typeof ws.pruneStatesAbove === "function")
                ? Promise.resolve(ws.pruneStatesAbove(wid, -1))
                : Promise.resolve();
            return prune.then(function () {
                trace("anchor:prune:resolve -> persist:issue");
                ws.persist(snap, snap, { workspace_id: wid, state_index: 0 });
                trace("anchor:persist:issued");
            });
        }

        var timeline = {
            save:            save,
            load:            load,
            undo:            undo,
            anchor:          anchor,
            enqueue:         _enqueue,
            draftIdentity:   draftIdentity,
            markUncommitted: markUncommitted,
        };
        // LIVE reads (§19.5): a plain value would freeze at 0 / false.  The data model re-exposes
        // these as its own live getters (state_index / uncommitted) reading through here.
        Object.defineProperty(timeline, "stateIndex", {
            get: function () { return _stateIndex; }, enumerable: true, configurable: true,
        });
        Object.defineProperty(timeline, "uncommitted", {
            get: function () { return _uncommitted; }, enumerable: true, configurable: true,
        });
        return timeline;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview._createStateTimeline = createStateTimeline;

    if (typeof module !== "undefined" && module.exports) {
        module.exports = { createStateTimeline: createStateTimeline };
    }
})(typeof window !== "undefined" ? window : globalThis);

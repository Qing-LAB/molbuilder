/* MolView data model — INSTALL / codec write-back (bytes -> model).  MolView-internal submodule.
 *
 * MODULE: molview (lib/molview/).  Extracted from data-model.js (the god-hub split) as an
 *   INJECTED FACTORY, like _operations.js / _serialise.js.  No global shim -- data-model.js is the
 *   sole consumer and IMPORTS `createInstall` directly (the 4a real import graph).
 *
 * ROLE: the bytes -> model direction (the inverse of _serialise.js):
 *   - `applyWorkspacePayload(payload, opts)` is THE single atomic cross-store sync point (§19.3.1):
 *      canvas replaceContent/setStructure + selection-store adoptAtoms + clearSelection + notify +
 *      engine push, all inside ONE suspend/resume persistence bracket so no transient
 *      geometry<->labels desync is ever published.  DATA-SAFETY critical.
 *   - `installMolecule({text[,sidecar,source,periodicity,annotations]})` -> `loadText` -> parse via
 *      /api/build/load, apply atomically, then anchor a fresh timeline (§19.5).  The one MODEL
 *      primitive for bringing a new molecule in from TEXT (generators + projects.parser.openMolecule).
 *   - `generate(kind, input, opts)` -> dispatch to the loaded structure-generator module.
 *
 * USED BY: lib/molview/data-model.js ONLY.  It builds ONE instance via createInstall({...}) and
 *   exposes generate/installMolecule on molview.data + hands `applyWorkspacePayload` to
 *   _operations.js (op responses) and to the hub's `_applySnapshot` (timeline restore).  The shared
 *   `_applying` flag STAYS in the hub (set here via the injected `setApplying` seam) because the
 *   hub's notify/markDirty/frame writers also read it.
 */
"use strict";

const root = (typeof window !== "undefined") ? window : globalThis;

// deps (aliased below to the names the moved bodies already use, keeping them near-verbatim):
//   { getStore, getCanvas, suspendPersist, resumePersist, notify, pushToEngine, trace, missing,
//     setApplying: (bool) -> void, anchor: () -> Promise }
export function createInstall(deps) {
    deps = deps || {};
    const _store          = deps.getStore;
    const _canvas         = deps.getCanvas;
    const suspendPersist  = deps.suspendPersist;
    const resumePersist   = deps.resumePersist;
    const _notify         = deps.notify;
    const _pushToEngine   = deps.pushToEngine;
    const _trace          = deps.trace;
    const _missing        = deps.missing;
    const _setApplying    = deps.setApplying;
    const _anchor         = deps.anchor;

    /**
     * Apply a server-returned workspace payload to every store atomically.  The single
     * cross-store sync point.  Side effects, in order:
     *   1. canvas-state.replaceContent(text) when ``touchCanvas`` (modifier-op flow; flips dirty);
     *      or setStructure on a fresh load (``installSource``, dirty=false).  Generator/sidebar
     *      flows pass ``touchCanvas:false`` because canvas-state was already set upstream.
     *   3. ``opts.resetSelection`` clears the selection (§19.3.2) on any atom-count change / load.
     *   4. Dispatcher subscribers are notified; then the settled structure is pushed to the engine.
     */
    function applyWorkspacePayload(payload, opts) {
        opts = opts || {};
        var touchCanvas    = opts.touchCanvas !== false;
        var resetSelection = !!opts.resetSelection;
        var text = payload && (payload.text || payload.xyz);

        var st = _store();

        // ATOMIC load/replace (§19.3.1 + F4): suspend persistence across the WHOLE
        // multi-store write so the intermediate steps (canvas install/replace, THEN
        // adoptAtoms) never publish a transient geometry<->labels atom-count desync.
        // Exactly one coherent persist fires on resume.  try/finally so a mid-write
        // throw can never leave persistence wedged suspended.
        suspendPersist();
        try {

        // 1. Canvas-state — text + periodicity + dirty bit.  A modifier op that
        //    recaptured a cell (e.g. add-electrodes) carries `periodicity` in the
        //    payload; passing it keeps the store's periodicity in step with the
        //    new geometry (workspace-contract.md §4.0).  Omitted -> kept as-is.
        var cs = _canvas();
        var fmt = payload && payload.source_format;
        if (touchCanvas && cs && text
                && typeof cs.replaceContent === "function") {
            cs.replaceContent(text, payload && payload.periodicity,
                              payload && payload.annotations);
        } else if (!touchCanvas && cs && text && opts.installSource
                && (fmt === "xyz" || fmt === "pdb")
                && typeof cs.setStructure === "function") {
            // A FRESH LOAD (`installSource` set): install the WHOLE structure into
            // the canvas -- text + periodicity, dirty=false -- REPLACING whatever was
            // there.  This is what makes loading ONE atomic operation: the SAME sync
            // point that adopts the atoms below (step 3) also (re)writes the canvas
            // here, so after a single call getStructure(), getUnitCell(), and
            // getAtoms() ALL reflect the just-loaded structure and stay coherent
            // across re-loads (load water, then benzene -> the canvas is benzene, not
            // stuck on water).  There is no second "load into canvas" door.  Only
            // loadFromText sets `installSource`; generator/sidebar flows install the
            // canvas via their own path (loadIntoCanvas/adoptSession) and never reach
            // here, so their source provenance is untouched.  A modifier op takes the
            // replaceContent (dirty=true) branch above.
            cs.setStructure(
                { source_format: fmt, text: text,
                  periodicity:   payload.periodicity || null,
                  annotations:   payload.annotations || null },
                opts.installSource);
        }

        // 2. (removed) The old modify-tab ``applyStructure`` hook is GONE.  The module no
        //    longer calls OUT to a consumer to mirror the structure into a local state.*
        //    copy -- consumers read the single source through the unified API and react to
        //    the subscription fired below.  Keeping the module free of consumer callbacks is
        //    the point of the concealed data model.

        // 3. Selection store atoms — the BOMB-0 cross-store sync.
        //    Single source of truth: this is the ONLY place the
        //    dispatcher consults ``payload.atoms``.
        if (st && Array.isArray(payload.atoms)
                && typeof st.adoptAtoms === "function") {
            // Distribute the payload's TOP-LEVEL metadata arrays onto each atom before it
            // enters the store.  /api/build/load returns atom_names / residue_ids /
            // residue_names / chain_ids as PARALLEL ARRAYS, not per-atom, so without this the
            // store -- and thus getStructure() -- would drop them, forcing consumers to keep a
            // parallel state.* mirror.  Distributing here makes molview.data the COMPLETE
            // single source.  Values can be 0/"" legitimately -> guard with != null, and never
            // overwrite an atom that already carries the field (adoptSession's per-atom shape).
            var _META = [["residue_ids", "residue_id"], ["atom_names", "atom_name"],
                         ["residue_names", "residue_name"], ["chain_ids", "chain_id"]];
            for (var _mi = 0; _mi < _META.length; _mi++) {
                var _arr = payload && payload[_META[_mi][0]];
                if (!Array.isArray(_arr) || _arr.length !== payload.atoms.length) continue;
                var _key = _META[_mi][1];
                for (var _ai = 0; _ai < payload.atoms.length; _ai++) {
                    var _at = payload.atoms[_ai];
                    if (_at && _at[_key] == null && _arr[_ai] != null) {
                        _at[_key] = _arr[_ai];
                    }
                }
            }
            // sourceFile rides IN on the SAME synchronous adopt when this is a file
            // open (installSource.kind === "file"): atoms + source land together, so
            // after this one write the store is fully settled -- no trailing
            // adoptSession is needed to name the file (the old second write that
            // clobbered a just-made selection).  A generator (kind !== "file") has no
            // source path, so sourceFile stays as-is (null for a fresh generate).
            var _srcFile = (opts.installSource
                            && opts.installSource.kind === "file")
                ? (opts.installSource.file || null) : undefined;
            st.adoptAtoms(payload.atoms, _srcFile);
        }

        // 4. §19.3.2 atom-count selection rule: any count-changing mutation (grow/shrink)
        //    -- and a load/generate -- passes resetSelection and CLEARS the selection.  A
        //    cleared selection can never mis-point at a shifted index, so there is no server
        //    ``selection_remap`` (retired).  Count-preserving transforms pass
        //    resetSelection:false and leave the selection untouched.
        if (resetSelection && st && typeof st.clearSelection === "function") {
            st.clearSelection();
        }

        // 5. Notify dispatcher subscribers.
        _notify();

        } finally {
            // One coherent persist for the whole atomic load (no-op if a caller
            // nested us inside its own suspend bracket -- the counter handles it).
            resumePersist();
        }
        // The atoms are settled -> hand the single-frame structure to the render engine
        // (it re-renders from this clean data). A trajectory caller follows with reloadFrames.
        _pushToEngine();
    }

    // ---- THE installMolecule model primitive (molview-module.md §19.3) --------- //
    // Bring a NEW molecule IN from TEXT: ONE atomic operation that parses the text (+ an
    // optional paired sidecar) via /api/build/load and replaces the WHOLE model AND resets the
    // session timeline (§19.5), landing on the single sync point (applyWorkspacePayload).  A
    // MODEL primitive -- it does NOT read files; the format-aware project-file DOOR that reads a
    // path lives in projects.parser.openMolecule, which hands { text, sidecar } here.
    //
    // THE LOAD CONTRACT (why ONE write): everything the loaded molecule needs -- geometry text,
    // and (server-applied from the sidecar) periodicity/annotations/regions/frozen -- rides IN on
    // this single call and is installed by ONE synchronous applyWorkspacePayload pass BEFORE the
    // load resolves.  A caller must NEVER install and then do a SECOND store write: the "ready"
    // signal fires at the single write, so any second write lands AFTER a consumer may have acted
    // on the settled structure and silently clobbers it (the 2026-07 selection-wipe).
    function installMolecule(input) {
        // FILE-ONLY project-file load (structure-authority.md): pass a ``path`` and
        // the SERVER reads the .xyz + paired .molstruct.json via StructureCodec.read
        // (Python owns the file access + pairing).  The response comes back enriched
        // (atoms + periodicity + annotations), so no text/sidecar/periodicity ride here.
        if (input && typeof input === "object"
                && typeof input.path === "string" && input.path) {
            return _loadText("", input.path, {
                path:   input.path,
                source: input.source || { kind: "file", file: input.path,
                                          generator_input: null },
            });
        }
        if (input && typeof input === "object" && typeof input.text === "string") {
            return _loadText(input.text, input.filename, {
                source:      input.source || null,
                periodicity: input.periodicity || null,
                annotations: input.annotations || null,
                atoms:       Array.isArray(input.atoms) ? input.atoms : null,
                format:      input.format || null,
                // The paired .molstruct.json CONTENT a project-file open read off disk;
                // the server (/api/build/load) applies it.  Omitted for generators.
                sidecar:     (typeof input.sidecar === "string") ? input.sidecar : null,
            });
        }
        return Promise.reject(new TypeError(
            "molview.data.installMolecule(input): pass { text, filename"
          + "[, sidecar, source, periodicity, annotations] }"));
    }

    async function _loadText(text, filename, opts) {
        opts = opts || {};
        _trace("loadText:start", { filename: filename, len: (text || "").length });
        // ``format`` is EXPLICIT when the caller knows the text's format regardless of
        // the filename -- the parser door (projects.parser.openMolecule) hands us the
        // canonical XYZ while the source file may be ``.pdb``; without this,
        // /api/build/load auto-detects "pdb" from the ``.pdb`` filename and parses the
        // XYZ text as PDB -> 400.  Omitted -> the server auto-detects (filename ext,
        // then content sniff), the raw-text/generator behaviour.
        // FILE-ONLY load (structure-authority.md): a project-file open passes a
        // ``path`` and the SERVER reads the .xyz + paired .molstruct.json through
        // StructureCodec.read -- Python owns the file access + the pairing, so the
        // browser sends NO raw text and does NOT derive the sidecar path.  The
        // text/sidecar body below is only for a raw-geometry IMPORT (paste/upload,
        // no persisted file yet) + generators.
        const _body = opts.path
            ? { path: opts.path }
            : { text: text, filename: filename };
        if (!opts.path) {
            if (opts.format) _body.format = opts.format;
            // The paired .molstruct.json CONTENT (raw JSON string) a project-file open
            // read through the projects package.  The server applies it (regions/frozen/
            // cell/axis_kind/vacuum/annotations) so the response's atoms + periodicity +
            // annotations come back ENRICHED -- the sidecar schema stays server-side.
            if (opts.sidecar) _body.sidecar = opts.sidecar;
        }
        const resp = await root.fetch("/api/build/load", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(_body),
        });
        const r = await resp.json();
        _trace("loadText:build-load:done", { ok: r.ok, n_atoms: r.n_atoms });
        if (!r.ok) {
            throw new Error(r.error || "Load failed.");
        }
        // Loading is ONE operation: hand the payload to the single sync point, which
        // populates the WHOLE model (canvas text + periodicity AND atoms) coherently.
        // Caller-supplied provenance + sidecar overrides ride on the load door itself (no
        // side-channel, §19.3): a generator's `source` (kind/generator_input), and the
        // sidecar `periodicity`/`annotations` a file-open carries but /api/build/load can't
        // re-derive, are applied OVER the server-parsed payload.  `installSource` names where
        // these bytes came from so the sync point seeds the canvas; `_applying` is held true
        // so the driven canvas signal does not mark the fresh load uncommitted.
        if (opts.periodicity) r.periodicity = opts.periodicity;
        if (opts.annotations) r.annotations = opts.annotations;
        // Sidecar-enriched atoms (regions/frozen the .molstruct.json carries, which
        // /api/build/load cannot re-derive) REPLACE the plain parsed atoms so the ONE
        // install below is the FINAL per-atom state -- no second store write to overlay
        // them afterwards.  This is what closes the load-order gap (see openMolecule).
        if (opts.atoms) r.atoms = opts.atoms;
        var installSource = opts.source
            || { kind: "file", file: filename || null, generator_input: null };
        _setApplying(true);
        try {
            applyWorkspacePayload(r, {
                touchCanvas:    false,
                resetSelection: true,
                installSource:  installSource,
            });
        } finally {
            _setApplying(false);
        }
        _trace("loadText:applyPayload:done");
        // Anchor a fresh timeline (§19.5): prune the previous molecule's state files, reset
        // state_index to 0, and write this loaded structure as the index-0 anchor.
        _trace("loadText:anchor:begin");
        await _anchor();
        _trace("loadText:anchor:awaited -> return");
        return r;
    }

    var _GENERATOR_MODULE_BY_KIND = {
        smiles:  "structureSmiles",
        name:    "structureName",
        dna:     "structureDna",
        rna:     "structureRna",
        peptide: "structurePeptide",
        file:    "structureFile",
    };

    // Dispatch to the loaded structure-generator module's generate(input, opts).
    function generate(kind, input, opts) {
        var key = String(kind || "").toLowerCase();
        var moduleName = _GENERATOR_MODULE_BY_KIND[key];
        if (!moduleName) {
            return Promise.reject(new Error(
                "workspace.generate: unknown kind " + JSON.stringify(kind)
                + "; expected one of "
                + Object.keys(_GENERATOR_MODULE_BY_KIND).join(", ")));
        }
        var mod = root.molbuilder && root.molbuilder[moduleName];
        if (!mod || typeof mod.generate !== "function") {
            return Promise.reject(_missing(
                moduleName + " (generator not loaded on this page)"));
        }
        return mod.generate(input, opts || {});
    }

    return {
        applyWorkspacePayload: applyWorkspacePayload,
        installMolecule:       installMolecule,
        generate:              generate,
    };
}

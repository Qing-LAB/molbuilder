/* MolView data model — modifier OPERATIONS (server round-trips).  MolView-internal submodule.
 *
 * MODULE: molview (lib/molview/).  Extracted from data-model.js (the god-hub split) as an
 *   INJECTED FACTORY — the same pattern _state-timeline-impl.js uses.  It has NO global shim:
 *   data-model.js is its sole consumer and IMPORTS `createOperations` directly (the 4a real
 *   import graph), so there is no transitional `window.molbuilder.*` mount to retire.
 *
 * ROLE: the `applyOp` pipeline (molview-module.md §19.3.2) — the op registry (ops-as-data), the
 *   empty-policy + arity gate, the whole-structure vs SUBSET-transform dispatch, the POST to
 *   /api/modify/<op>, the defensive parse, the atom-count invariant, and the atomic apply of the
 *   server response.  It owns the `_mutating` serialize flag (at most one mutation in flight).
 *
 * USED BY: lib/molview/data-model.js ONLY.  It builds ONE instance via
 *   `createOperations({ getStructure, getCoordinates, getElements, structureBody,
 *   applyWorkspacePayload, getStore })` and exposes the result's `applyOp` on `molview.data`.
 *   The two codec seams are injected (they stay in the hub, shared with install + the timeline):
 *     - `structureBody()`        builds the COMPLETE op-request body (§19.3.2)
 *     - `applyWorkspacePayload(payload, opts)` is THE atomic cross-store sync point
 *   and the accessors (`getStructure`/`getCoordinates`/`getElements`) + `getStore()` (for the
 *   current selection) read the live model.  This submodule holds NO model state of its own
 *   besides the in-flight `_mutating` guard.
 */
"use strict";

const root = (typeof window !== "undefined") ? window : globalThis;

// deps: { getStructure: () -> Structure, getCoordinates: () -> Vec3[], getElements: () -> str[],
//         structureBody: () -> opRequestBody, applyWorkspacePayload: (payload, opts) -> void,
//         getStore: () -> selectionStore }
export function createOperations(deps) {
    deps = deps || {};
    const getStructure          = deps.getStructure;
    const getCoordinates        = deps.getCoordinates;
    const getElements           = deps.getElements;
    const structureBody         = deps.structureBody;
    const applyWorkspacePayload = deps.applyWorkspacePayload;
    const getStore              = deps.getStore;

    // §19.3.2 op registry -- ops are DATA (keyed by canonical name = server route).
    //   role       "subject" (atoms that change) | "anchor" (reference atoms)
    //   empty      "all" (empty group -> every atom) | "reject" | "canonical" (0 allowed)
    //   arity      null (any) | int | [min,max] -- required RESOLVED group size
    //   groupField body key the resolved group is written to (null = whole-structure transform)
    //   scalar     true -> groupField takes a single int (group[0]), not an array
    //   shape      "transform" (count kept, selection KEPT) | "grow" | "shrink" (count changes,
    //              selection CLEARED) -- drives the count invariant + the atom-count selection rule
    //   wholeOnly  true -> ALWAYS the whole-structure path, never the subset path, even with a
    //              partial selection (an op that has no per-subset meaning, e.g. calibrate)
    //   mapGroup   optional (group)->indices op-specific ordering (electrode top/bottom by z)
    var _OP_REGISTRY = {
        "translate": { role: "subject", empty: "all",    arity: null,   groupField: null,           shape: "transform" },
        "rotate":    { role: "subject", empty: "all",    arity: null,   groupField: null,           shape: "transform" },
        "orient":    { role: "anchor",  empty: "reject", arity: 2,      groupField: "anchors",      shape: "transform" },
        "add_atom":  { role: "anchor",  empty: "reject", arity: 1,      groupField: "anchor_index", scalar: true, shape: "grow" },
        "electrode": { role: "anchor",  empty: "canonical", arity: null, groupField: "center_indices", shape: "grow" },
        // Junction CENTRE = the centroid of the selected atom group (any count):
        // 1 atom -> that atom, 2 -> midpoint, N -> centroid.  Empty selection ->
        // "canonical" (the field is omitted, server centres on the origin).  No
        // ordering needed -- a centroid is order-independent.
        "symmetric_electrodes": {
            role: "anchor", empty: "canonical", arity: null, groupField: "center_indices", shape: "grow",
        },
        "delete":    { role: "subject", empty: "reject", arity: null,   groupField: "indices",      shape: "shrink" },
        // § 3c: rigid whole-structure calibrate -> atoms into [0,cell), cell at origin.
        // Count-preserving (selection kept); no group.  wholeOnly: calibrate moves ALL
        // atoms + clears cell_origin -- there is no per-subset meaning, so it ALWAYS
        // takes the whole-structure path even when a partial selection is active.
        "calibrate": { role: "subject", empty: "all",    arity: null,   groupField: null,           shape: "transform", wholeOnly: true },
    };
    var _mutating = false;   // §19.3.2 serialize: at most ONE structure mutation in flight.

    function _selectionIndices() {
        var st = getStore();
        var s = (st && typeof st.getState === "function") ? st.getState() : null;
        return (s && Array.isArray(s.selection)) ? s.selection.slice() : [];
    }
    function _range(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; }

    // The fetch + defensive parse + (for whole-structure ops) count invariant + atomic apply.
    // ``onOk(env.r)`` handles the applied result -- a whole-structure apply, or the subset map-back.
    function _postOp(op, body, onOk) {
        return root.fetch("/api/modify/" + op, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        }).then(function (resp) {
            // TEXT-first parse: a 500/404 HTML page would make resp.json() throw the cryptic
            // "Unexpected token '<'..."; parse defensively so the real error surfaces.
            return resp.text().then(function (txt) {
                var r = null; try { r = txt ? JSON.parse(txt) : null; } catch (_) { r = null; }
                return { httpOk: resp.ok, status: resp.status, r: r };
            });
        }).then(function (env) {
            if (!env.httpOk || !env.r || !env.r.ok) {
                throw new Error((env.r && env.r.error)
                    || ("modify/" + op + " failed (HTTP " + env.status + ")"));
            }
            return onOk(env.r);
        });
    }

    // Whole-structure path: complete body + group->groupField + POST + count invariant + apply.
    function _wholeStructureOp(op, desc, group, opParams, oldCount) {
        var body = structureBody();
        if (desc.groupField) {
            var placed = desc.mapGroup ? desc.mapGroup(group) : group;
            // An empty group under a "canonical" empty-policy means OMIT the field
            // entirely so the server applies its canonical default (e.g. origin-centred
            // electrode slabs).  Writing center_indices:[] would be an empty selection.
            if (placed.length) body[desc.groupField] = desc.scalar ? placed[0] : placed;
        }
        for (var k in opParams) { if (k !== "indices" && k !== undefined) body[k] = opParams[k]; }
        return _postOp(op, body, function (r) {
            var newCount = Array.isArray(r.atoms) ? r.atoms.length
                         : (typeof r.n_atoms === "number" ? r.n_atoms : oldCount);
            if (desc.shape === "transform" && newCount !== oldCount) {
                throw new Error("applyOp(" + op + "): transform changed the atom count ("
                    + oldCount + " -> " + newCount + ")");
            }
            if (desc.shape === "grow" && !(newCount > oldCount)) {
                throw new Error("applyOp(" + op + "): grow did not add atoms ("
                    + oldCount + " -> " + newCount + ")");
            }
            if (desc.shape === "shrink" && !(newCount < oldCount)) {
                throw new Error("applyOp(" + op + "): shrink did not remove atoms ("
                    + oldCount + " -> " + newCount + ")");
            }
            // §19.3.2 selection rule: count change (grow/shrink) CLEARS; transform KEEPS.
            applyWorkspacePayload(r, {
                touchCanvas:    true,
                resetSelection: (desc.shape === "grow" || desc.shape === "shrink"),
            });
            return r;
        });
    }

    // §19.3.2 SUBSET transform: extract the subject atoms -> run the SAME order-preserving
    // rotate/translate route -> map the transformed coords BACK into the full structure at the
    // subject indices.  Untouched atoms + ALL per-atom metadata stay put (a pure coord write).
    function _subsetTransform(op, group, opParams) {
        var s = getStructure();
        var atoms = (s && Array.isArray(s.atoms)) ? s.atoms : [];
        var subXyz = group.length + "\nsubset\n";
        for (var i = 0; i < group.length; i++) {
            var a = atoms[group[i]];
            subXyz += (a.element || "X") + " " + a.x + " " + a.y + " " + a.z + "\n";
        }
        var body = { xyz: subXyz };
        for (var k in opParams) { if (k !== "indices") body[k] = opParams[k]; }
        return _postOp(op, body, function (r) {
            var sub = Array.isArray(r.atoms) ? r.atoms : [];
            // Order-preservation re-check: the returned sub-structure MUST match what we sent,
            // atom-for-atom, or the map-back would mis-assign coordinates.
            if (sub.length !== group.length) {
                throw new Error("applyOp(" + op + "): subset transform changed the atom count");
            }
            for (var j = 0; j < group.length; j++) {
                if (sub[j].element !== atoms[group[j]].element) {
                    throw new Error("applyOp(" + op
                        + "): subset transform reordered atoms (element mismatch)");
                }
            }
            // Build the full new coords (only the subject moved) and re-apply as a
            // count-preserving, metadata-preserving transform (selection KEPT).
            var coords = getCoordinates();               // fresh copy of the current full coords
            var subByOrig = {};
            for (var m = 0; m < group.length; m++) {
                coords[group[m]] = [sub[m].x, sub[m].y, sub[m].z];
                subByOrig[group[m]] = true;
            }
            var title = (s && s.title) || "";
            var text = atoms.length + "\n" + title + "\n";
            var fullAtoms = [];
            for (var p = 0; p < atoms.length; p++) {
                var at = atoms[p];
                text += (at.element || "X") + " " + coords[p][0] + " "
                      + coords[p][1] + " " + coords[p][2] + "\n";
                fullAtoms.push({
                    index: p, element: at.element,
                    x: coords[p][0], y: coords[p][1], z: coords[p][2],
                    regions: at.labels || [], is_frozen: !!at.isFrozen,
                    atom_name: at.atomName, residue_name: at.residueName,
                    chain_id: at.chainId, residue_id: at.residueId,
                });
            }
            applyWorkspacePayload({
                text: text, source_format: (s && s.source_format) || "xyz",
                atoms: fullAtoms, periodicity: (s && s.periodicity) || null,
                annotations: (s && s.annotations) || null,
            }, { touchCanvas: true });   // transform -> selection KEPT (no reset)
            return r;
        });
    }

    function applyOp(op, args) {
        args = args || {};
        var desc = _OP_REGISTRY[op];
        if (!desc) {
            return Promise.reject(new Error(
                "molview.data.applyOp: unknown op '" + op + "'"));
        }
        if (_mutating) {
            return Promise.reject(new Error(
                "molview.data.applyOp: a structure mutation is already in flight"));
        }
        var nAll = getElements().length;
        // Resolve the group -- explicit indices, else the current selection -- for the op's role.
        var group = Array.isArray(args.indices) ? args.indices.slice() : _selectionIndices();
        // Enforce the empty policy, then the arity, BEFORE any fetch.
        if (group.length === 0) {
            if (desc.empty === "all") group = _range(nAll);
            else if (desc.empty === "reject") {
                return Promise.reject(new Error("molview.data.applyOp(" + op
                    + "): a non-empty selection is required"));
            }
            // "canonical": proceed with an empty group.
        }
        if (desc.arity != null) {
            var lo, hi;
            if (Array.isArray(desc.arity)) { lo = desc.arity[0]; hi = desc.arity[1]; }
            else { lo = hi = desc.arity; }
            if (group.length < lo || group.length > hi) {
                return Promise.reject(new Error("molview.data.applyOp(" + op + "): needs "
                    + (lo === hi ? lo : lo + "-" + hi) + " atom(s), got " + group.length));
            }
        }
        _mutating = true;
        var run;
        // Dispatch: a transform on a SUBSET (subject smaller than all) uses the subset
        // orchestration so ONLY the selected atoms move + the box stays put; everything
        // else -- selection none/all, a wholeOnly op (calibrate), grow/shrink -- takes
        // the whole-structure path (the box moves WITH the atoms, § 3c).
        if (desc.shape === "transform" && desc.role === "subject" && !desc.wholeOnly
                && group.length > 0 && group.length < nAll) {
            run = _subsetTransform(op, group, args);
        } else {
            run = _wholeStructureOp(op, desc, group, args, nAll);
        }
        return run.then(function (r) { _mutating = false; return r; },
                        function (e) { _mutating = false; throw e; });
    }

    return { applyOp: applyOp };
}

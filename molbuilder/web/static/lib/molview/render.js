/* molview render loop -- "molview owns the render" (molview-module.md §14).
 *
 * Given a viewer HANDLE + the dataModel + the selection store, this draws the current
 * structure once and re-draws on every dataModel change -- reading EVERYTHING through ws.*
 * (no local structure copy; that is the data-unification the design mandates, §14.0 "always
 * recompute from the CLEAN unit cell" = the data model).
 *
 * The render is ONE coordinate pipeline -> ONE 3dmol draw (§14.0).  It is a READ-ONLY view
 * derivation: it GENERATES the coordinate list 3dmol draws from the stored atoms; it never
 * writes the data (isolate changes the render list, never the dataset).
 *   ws / store atoms       -> the clean unit-cell atoms (the source of truth)
 *   drawBase()             -> the plain unit cell (when isolate is off)
 *   mountIsolateRender(..) -> REPLACES it with the derived list when isolate is on (filtered
 *                             to the selected atoms) -- one setStructure, the derived list
 *                             instead of the base, NOT a second draw on top (§14.2)
 * plus the measurement overlay (mountMeasurementOverlay, §15), coords read from the handle.
 *
 *   molview.mountRender(handle, dataModel, store, { viewerHost }) -> { refresh, dispose }
 *
 * It composes the existing molview pieces (mountIsolateRender, mountMeasurementOverlay) -- it
 * does not reimplement them.  mount.js assembles this with the viewer embed to make the
 * full component.
 */
(function (root) {
    "use strict";

    function _noop() {}

    function mountRender(handle, dataModel, store, opts) {
        opts = opts || {};
        var mv = (root.molbuilder && root.molbuilder.molview) || {};
        if (!handle || !dataModel || !store) {
            return { refresh: _noop, dispose: _noop };
        }

        // ---- Everything read through ws.* -- never a local copy (§14.0) ---------------- //

        // Build an xyz body from the atoms THEMSELVES -- never a canvas text that a
        // store-only load (loadFromText, touchCanvas:false) leaves stale.
        function _buildXyz(elements, coords) {
            var lines = [String(coords.length), "molview"];
            for (var i = 0; i < coords.length; i++) {
                var p = coords[i];
                lines.push((elements[i] || "X") + " " + p[0] + " " + p[1] + " " + p[2]);
            }
            return lines.join("\n");
        }
        // The clean unit cell, read from the STORE -- the SAME atoms the panel lists, so the
        // viewer and the panel can never diverge (the data-consistency the design mandates).
        function getUnit() {
            var st = store.getState();
            var atoms = (st && Array.isArray(st.atoms)) ? st.atoms : [];
            if (!atoms.length) return null;
            var coords   = atoms.map(function (a) { return [a.x, a.y, a.z]; });
            var elements = atoms.map(function (a) { return a.element; });
            return { coords: coords, elements: elements, xyz: _buildXyz(elements, coords) };
        }
        // The RESOLVED lattice (explicit cell, OR the bbox+vacuum default) -- used to detect
        // cell changes in the structure signature, never for the base draw.
        function getCell() {
            if (typeof dataModel.getUnitCellInfo === "function") {
                var info = dataModel.getUnitCellInfo();
                if (info && info.value) return info.value;
            }
            return (typeof dataModel.getUnitCell === "function") ? dataModel.getUnitCell() : null;
        }
        // The EXPLICIT cell ONLY (raw, null for a cell-less molecule) -- what the BASE draw
        // passes as its lattice.  A user/imported cell draws a wireframe box + cell-scaled
        // axes; a fresh cell-less molecule draws NO box and Cartesian FIXED-length axes.  Do
        // NOT feed the resolved bbox here -- that would draw a spurious box + extent-scaled
        // axes for every cell-less molecule (the bbox belongs to tiling, via getCell).
        function getExplicitCell() {
            return (typeof dataModel.getUnitCell === "function" && dataModel.getUnitCell())
                || null;
        }

        // ---- The pipeline -> one draw ------------------------------------------------- //

        // Base draw: the plain current structure (all atoms, from the store).  When isolate
        // is on, mountIsolateRender REPLACES this with the derived list (§14.2).
        // The unit-cell BOX channel (§3a): the RESOLVED cell (explicit, else bbox+vacuum)
        // plus the corner it's anchored at, so the wireframe box WRAPS the atoms instead
        // of starting at the world origin (structure-periodicity.md Issue #2).  This is
        // SEPARATE from `lattice` (the explicit cell, which drives the a/b/c axes) so a
        // cell-less molecule still shows Cartesian xyz axes but gets a bbox+vacuum box.
        function getCellBox() {
            var resolved = (typeof dataModel.getUnitCellInfo === "function")
                ? (dataModel.getUnitCellInfo().value || null) : null;
            if (!resolved) return null;
            var origin = (typeof dataModel.getUnitCellOrigin === "function")
                ? dataModel.getUnitCellOrigin() : null;
            return { lattice: resolved, origin: origin };
        }
        function drawBase() {
            var u = getUnit();
            if (u && typeof handle.setStructure === "function") {
                // AXES: EXPLICIT cell only (not the resolved bbox) -- see getExplicitCell.
                // Pass the raw getExplicitCell() (a cell, or NULL when there's none) -- NOT
                // `|| undefined`: the embed reads `lattice: undefined` as "keep the current
                // lattice" and `lattice: null` as "CLEAR it".  Coercing the no-cell null to
                // undefined left a stale wireframe drawn after a cell -> no-cell transition
                // (add electrodes = hexagonal cell, then Retract to the cell-less molecule --
                // the box stayed hexagonal).  null clears it.
                // BOX: the resolved cell + anchor, so the box wraps the atoms (Issue #2).
                handle.setStructure({
                    xyz:     u.xyz,
                    lattice: getExplicitCell(),
                    cellBox: getCellBox(),
                });
            }
        }

        // Isolate controller: REPLACES the base draw with the selected-only list when the
        // store's isolate toggle is on (§14.2); restores the base draw when it goes off.
        var iso = (typeof mv.mountIsolateRender === "function")
            ? mv.mountIsolateRender(handle, store, {
                  getUnit: getUnit, getCell: getExplicitCell,
                  // Restore path (isolate off) -> the plain base draw.
                  drawBase: drawBase,
              })
            : { refresh: _noop, dispose: _noop };

        // Measurement overlay: the ONE overlay (§15); coords come from the viewer handle,
        // never a held copy.  Skipped if the host didn't give a viewerHost to paint into.
        var meas = (typeof mv.mountMeasurementOverlay === "function" && opts.viewerHost)
            ? mv.mountMeasurementOverlay(opts.viewerHost, {
                  store: store,
                  coordsProvider: function () {
                      return (typeof handle.getAtomCoords === "function") ? handle.getAtomCoords() : [];
                  },
              })
            : { dispose: _noop };

        // Re-draw the BASE only when the STRUCTURE (atoms or cell) changes -- NOT on every
        // selection click.  Selection/isolate overlays are the viewer-adapter's job.  A
        // store-only load fires store.subscribe; a periodicity edit fires ws.subscribe; a
        // plain click fires store.subscribe but leaves the signature the same.
        var _prevSig = null;
        function _structureSig() {
            var st = store.getState();
            var atoms = (st && st.atoms) || [];
            var sig = String(atoms.length);
            for (var i = 0; i < atoms.length; i++) {
                var a = atoms[i];
                sig += "|" + a.element + "," + a.x + "," + a.y + "," + a.z;
            }
            var cell = getCell();
            return sig + "|cell:" + (cell ? JSON.stringify(cell) : "none");
        }

        // Busy<->ready: the ONE surface is handle.setBusy(msg|null) (embed §6.x).  The
        // render controller drives it -- a file LOAD in flight or a big-system REDRAW
        // shows the scrim; a small/fast redraw doesn't (avoids flicker).  For a heavy
        // redraw we YIELD a paint (double requestAnimationFrame) so the scrim actually
        // appears BEFORE the blocking 3Dmol swap freezes the thread; cancel on re-entry
        // so only the latest state draws (drawBase reads the live store).
        var BUSY_ATOM_THRESHOLD = 1200;   // atoms above which a swap is slow enough to scrim
        var _rafId = null;
        function _setBusy(msg) {
            if (handle && typeof handle.setBusy === "function") handle.setBusy(msg);
        }
        function _atomCount() {
            var st = store.getState();
            return (st && st.atoms) ? st.atoms.length : 0;
        }
        // Isolate state signature (isolate flag + the selected indices, since the derived
        // "show selected only" list is those atoms).  A change here means the isolate view
        // must re-render even when the STRUCTURE is unchanged (a selection click while
        // isolating, or the toggle flipping).
        var _prevIso = null;
        function _isoState() {
            var st = store.getState() || {};
            var on = !!st.isolate
                && (Array.isArray(st.indices) ? st.indices.length : 0) > 0;
            return on ? ("on:" + (st.indices || []).join(",")) : "off";
        }
        function _renderNow(structChanged) {
            if (structChanged) drawBase();                 // full base draw
            // iso.refresh() renders the DERIVED list (isolate on) or restores drawBase
            // (isolate just turned off).  It is driven ONLY from here now -- mountIsolate-
            // Render holds no store subscription of its own (two subscriptions raced: its
            // derived draw + render's full drawBase left the viewer showing the FULL
            // structure after an edit while isolating).
            if (iso && typeof iso.refresh === "function") iso.refresh();
            // A redraw (setStructure) cleared the embed's overlays -- re-apply the consumer's
            // overlays HERE, inside the render streamline (the overlay controller has no store
            // subscription of its own; that used to clobber the view toggles).
            if (typeof opts.afterRedraw === "function") opts.afterRedraw();
        }

        // THE render streamline == the ONE place setBusy is gated.  Every store/data change
        // that needs a re-render (structure edit, isolate on/off, selection while isolating)
        // routes through here: it turns the busy scrim ON, YIELDS a paint, renders, then
        // clears -- so a big-system re-render never freezes the tab with no busy block.  No
        // consumer sets busy; they change state and this line gates the render + recovery.
        function onStoreChange() {
            var st = store.getState();
            if (st && st.loading) { _setBusy("Loading…"); return; }    // load in flight
            var sig = _structureSig();
            var isoNow = _isoState();
            var structChanged = (sig !== _prevSig);
            var isoChanged = (isoNow !== _prevIso);
            if (!structChanged && !isoChanged) { _setBusy(null); return; }
            var wasOn = (_prevIso || "").slice(0, 3) === "on:";
            var nowOn = isoNow.slice(0, 3) === "on:";
            _prevSig = sig; _prevIso = isoNow;
            // How big is the NEXT draw?  A FULL base draw (structure change, or isolate just
            // turned OFF -> drawBase) is the whole structure; an isolate-ON draw is only the
            // SELECTED atoms.  Scrim only when THAT draw is big enough to freeze the thread.
            var fullDraw = structChanged || (wasOn && !nowOn);
            var idxN = Array.isArray(st && st.indices) ? st.indices.length : 0;
            var renderSize = fullDraw ? _atomCount() : (nowOn ? idxN : 0);
            if (renderSize >= BUSY_ATOM_THRESHOLD) {
                _setBusy("Updating view…");
                if (_rafId) cancelAnimationFrame(_rafId);
                _rafId = requestAnimationFrame(function () {
                    _rafId = requestAnimationFrame(function () {
                        _rafId = null;
                        _renderNow(structChanged);
                        _setBusy(null);
                    });
                });
            } else {
                _renderNow(structChanged);
                _setBusy(null);
            }
        }
        onStoreChange();   // initial draw

        var storeUnsub = (typeof store.subscribe === "function")
            ? store.subscribe(onStoreChange) : _noop;
        var wsUnsub = (typeof dataModel.subscribe === "function")
            ? dataModel.subscribe(onStoreChange) : _noop;

        return {
            refresh: onStoreChange,
            dispose: function () {
                if (_rafId) { try { cancelAnimationFrame(_rafId); } catch (_) {} _rafId = null; }
                _setBusy(null);
                try { storeUnsub(); } catch (_) {}
                try { wsUnsub(); } catch (_) {}
                try { iso.dispose(); } catch (_) {}
                try { meas.dispose(); } catch (_) {}
            },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.molview = root.molbuilder.molview || {};
    root.molbuilder.molview.mountRender = mountRender;
    if (typeof module !== "undefined" && module.exports) {
        module.exports = { mountRender: mountRender };
    }
})(typeof window !== "undefined" ? window : this);

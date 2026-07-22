/* MolView — the single ES-module entry (the MolView ESM migration).
 *
 * THE unified embedding contract: a page embeds MolView with ONE module import instead of a
 * hand-maintained classic <script> stack --
 *
 *   <script type="module">
 *     import { mount } from "/static/lib/molview/index.js";
 *     mount(document.querySelector(".molview-host"), ws, { mode, owner });
 *   </script>
 *
 * Dependency order is enforced by the import graph here, not remembered per page.
 *
 * MIGRATION STATE: converting bottom-up (leaf -> data/state/store -> embed -> engine -> panels
 * -> mount). Converted modules are imported here (running their transitional side-effect global
 * publish + exposing exports); the rest still load as classic <script> globals until they flip.
 * When `mount` and its whole subtree are modules, the classic stack + globals go away.
 */
// The 3Dmol viewer embed (mol-*.js) -- ES modules in the graph.  mol-viewer-embed imports
// mol-viewer (the shared `viewer` object) and imports mol-style/mol-format/mol-axes directly
// (pure stateless helpers -- no globals; the embed `import`s spec/formula/axes).  Loaded here so
// the whole embed is in the module graph (no classic <script> stack); mount.js imports `viewer`
// from mol-viewer-embed.
import "../mol-style.js";         // spec() -- imported by the embed; no global (pure helper)
import "../mol-format.js";        // formula() -- re-exported below; no global (pure helper, imported)
import "../mol-axes.js";          // axes.draw() -- imported by the embed; no global (pure helper)
import "../mol-viewer.js";        // publishes window.molbuilder.viewer (base: .create)
import "../mol-viewer-embed.js";  // extends the viewer with .embed
import "./_atom-index.js";        // publishes window.molbuilder.atomIndexModel (transitional)
import "./_atom-channels.js";     // publishes window.molbuilder.atomChannelModel (transitional)
import "./engine/process.js";     // publishes window.molbuilder.molview.engine.process (transitional)
import "./engine/embed-io.js";    // publishes window.molbuilder.molview.engine.embedIo (transitional)
import "./engine/engine.js";      // publishes window.molbuilder.molview.engine.create (transitional)
import "./selection/measurements.js";  // publishes ...molview.selection.measurements (transitional)
import "./selection/panel.js";         // publishes ...molview.selection.panel (transitional)
import "./selection/viewer-adapter.js";// publishes ...molview.selection.viewerAdapter (transitional)
import "./selection/mount-panel.js";   // publishes ...molview.selection.mountPanel (transitional)
// The store/state layer.  data-model.js IMPORTS createStore/canvasState/createStateTimeline
// directly (real ES edges), so ES resolution -- not this list's order -- guarantees they evaluate
// first; these entries remain so the transitional shims (the node-test injection seam) still publish.
import "./_selection-store-impl.js";   // publishes ...molview.selection._createStore (test seam)
import "./_canvas-state-impl.js";      // publishes ...molview._canvasState (test seam)
import "./_state-timeline-impl.js";    // publishes ...molview._createStateTimeline (test seam)
import "./data-model.js";              // publishes ...molview.data (transitional)
import "./measurement-overlay.js";     // publishes ...molview.mountMeasurementOverlay (transitional)
import "./frame-controls.js";          // publishes ...molview.mountFrameControls (transitional)
// mount.js reads its deps (molview.data / .engine / .selection + the viewer embed) at mount()
// CALL time (runtime), never at module body -- so it loads LAST, and its transitional
// window.molbuilder.molview.mount shim + `export { mount }` make the single-import entry work.
import "./mount.js";                   // publishes ...molview.mount (transitional)

export * from "./_atom-index.js";
export * from "./_atom-channels.js";
export * from "./engine/process.js";
export * from "./engine/embed-io.js";
export * from "./engine/engine.js";
export * from "./selection/measurements.js";
export * from "./selection/panel.js";
export * from "./selection/viewer-adapter.js";
export * from "./selection/mount-panel.js";
export * from "./_selection-store-impl.js";
export * from "./_canvas-state-impl.js";
export * from "./_state-timeline-impl.js";
export * from "./data-model.js";
export * from "./measurement-overlay.js";
export * from "./frame-controls.js";
export { mount } from "./mount.js";
export { formula } from "../mol-format.js";   // Hill chemical-formula formatter (consumers import this)

/* MolView — the single ES-module entry.
 *
 * Contract: docs/web/molview.md § 4, § 9.1.  A page embeds MolView with ONE import:
 *
 *   <script type="module">
 *     import { mount } from "/static/lib/molview/index.js";
 *     mount(document.querySelector(".molview-host"), ws, { mode, owner });
 *   </script>
 *
 * Dependency order is enforced by the import graph here, not remembered per page.
 *
 * WHERE THIS IS GOING (plan Phase 7, docs/web/molview-rework-plan.md).  § 4 says the whole
 * surface is `mount` and `formula` and that "every other file in the module is internal —
 * a consumer that imports any of them directly has broken the module, not found a
 * shortcut".  The `export *` block below is therefore temporary: it is how the tabs and the
 * old tests still reach inside, and it goes when they are repointed, together with the
 * transitional `window.molbuilder.*` publishes each module still makes on load.
 */

// The sealed layer (§ 9.9).  Imported for its side effect only — it publishes the
// transitional `molbuilder.viewer` global, and mount.js imports `viewer` from it directly.
// Deliberately NOT re-exported: no consumer names the sealed layer's file (§ 15).
import "./_seal.js";

import "./_atom.js";                   // atomIndexModel + atomChannelModel (transitional globals)
import "./render-engine/embed-io.js";  // ...molview.renderEngine.embedIo
import "./render-engine/engine.js";    // ...molview.renderEngine.create + .process
import "./measurement.js";             // ...molview.selection.measurements + .mountMeasurementOverlay
import "./selection.js";               // ...molview.selection.panel + .viewerAdapter
// The store/state layer.  data-model.js IMPORTS these directly (real ES edges), so ES
// resolution — not this list's order — guarantees they evaluate first; the entries remain
// so the transitional shims (the node-test injection seam) still publish.
import "./_selection-store.js";        // ...molview.selection._createStore (test seam)
import "./_canvas-state-impl.js";      // ...molview._canvasState (test seam)
import "./_history.js";                // ...molview._createStateTimeline (test seam)
import "./data-model.js";              // ...molview.data
import "./controls.js";                // ...molview.mountFrameControls
// mount.js reads its deps (molview.data / .selection + the seal) at mount() CALL time, never
// at module body — so it loads LAST, and its transitional `molview.mount` shim plus the
// `export { mount }` below make the single-import entry work.
import "./mount.js";                   // ...molview.mount + ...molview.selection.mountPanel

export * from "./_atom.js";
export * from "./render-engine/embed-io.js";
export * from "./render-engine/engine.js";
export * from "./measurement.js";
export * from "./selection.js";
export * from "./_selection-store.js";
export * from "./_canvas-state-impl.js";
export * from "./_history.js";
export * from "./data-model.js";
export * from "./controls.js";

// The two the contract keeps (§ 4).
export { mount, mountPanel } from "./mount.js";
export { formula } from "./_formula.js";

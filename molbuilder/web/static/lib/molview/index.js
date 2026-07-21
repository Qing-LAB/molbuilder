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
import "./_atom-index.js";        // publishes window.molbuilder.atomIndexModel (transitional)
import "./_atom-channels.js";     // publishes window.molbuilder.atomChannelModel (transitional)
import "./engine/process.js";     // publishes window.molbuilder.molview.engine.process (transitional)
import "./engine/embed-io.js";    // publishes window.molbuilder.molview.engine.embedIo (transitional)
import "./engine/engine.js";      // publishes window.molbuilder.molview.engine.create (transitional)
import "./selection/measurements.js";  // publishes ...molview.selection.measurements (transitional)
import "./selection/panel.js";         // publishes ...molview.selection.panel (transitional)
import "./selection/viewer-adapter.js";// publishes ...molview.selection.viewerAdapter (transitional)
import "./selection/mount-panel.js";   // publishes ...molview.selection.mountPanel (transitional)
import "./measurement-overlay.js";     // publishes ...molview.mountMeasurementOverlay (transitional)
import "./frame-controls.js";          // publishes ...molview.mountFrameControls (transitional)

export * from "./_atom-index.js";
export * from "./_atom-channels.js";
export * from "./engine/process.js";
export * from "./engine/embed-io.js";
export * from "./engine/engine.js";
export * from "./selection/measurements.js";
export * from "./selection/panel.js";
export * from "./selection/viewer-adapter.js";
export * from "./selection/mount-panel.js";
export * from "./measurement-overlay.js";
export * from "./frame-controls.js";

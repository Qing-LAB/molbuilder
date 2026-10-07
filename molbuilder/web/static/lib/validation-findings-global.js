/* The namespace entry point for `lib/validation-findings.js`.
 *
 * ONE JOB: publish `molbuilder.validationFindings` for the classic scripts
 * that cannot `import` -- `lib/spectra/core.js`, `lib/transport/core.js`
 * and `structure-optimization/viewer.js`, which read it at call time
 * inside a handler.
 *
 * WHY IT IS A SEPARATE FILE.  `lib/molview/ui.js` imports the renderer, and
 * `web/molview.md` § 4 forbids mounting a viewer from publishing a name, so
 * only the page that asks for the namespace gets one.
 *
 * A page loads THIS; a module imports the other.
 */
import { render, clear } from "./validation-findings.js";

/* A PLAIN OBJECT, not the module namespace: the classic callers read
 * `{render, clear}`, and naming the two exports here means a
 * third one does not reach the namespace by accident. */
const root = typeof window !== "undefined" ? window : globalThis;
root.molbuilder = root.molbuilder || {};
root.molbuilder.validationFindings = { render, clear };

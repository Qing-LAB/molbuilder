/* The namespace entry point for `lib/validation-findings.js`.
 *
 * ONE JOB: publish `molbuilder.validationFindings` for the classic scripts
 * that cannot `import` -- `lib/spectra/core.js` and
 * `structure-optimization/viewer.js`, both of which read it at call time
 * inside a handler.
 *
 * WHY IT IS A SEPARATE FILE (2026-09-11).  The renderer used to publish
 * itself, so importing it published too -- and `lib/molview/ui.js` imports it
 * (a viewer reaching down to a presentation module, which is the dependency
 * going the right way).  That made mounting a viewer add a name to the app's
 * namespace, which `web/molview.md` § 4 forbids: *nothing it needs comes from
 * a global*, and mounting publishes nothing.  Splitting the registration out
 * lets both be true -- one implementation, two delivery forms, and only the
 * page that asks for the namespace gets one.
 *
 * A page loads THIS; a module imports the other.
 */
import { render, clear } from "./validation-findings.js";

/* A PLAIN OBJECT, not the module namespace.  `import * as ns` would publish an
 * exotic, frozen namespace object -- a different thing from the `{render,
 * clear}` the classic callers have always had, and the published surface
 * should be exactly what it was.  Naming the two exports here also means a
 * third one does not reach the namespace by accident. */
const root = typeof window !== "undefined" ? window : globalThis;
root.molbuilder = root.molbuilder || {};
root.molbuilder.validationFindings = { render, clear };

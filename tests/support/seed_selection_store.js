/* Test-support shim (MolView dispatcher node tests).
 *
 * Loaded through the ES-module harness BETWEEN the selection-store impl and the data model, it
 * creates the process-wide selection-store singleton on ``molview.selection.store`` -- exactly
 * the ordering the old require()-based harness had (require STORE -> create singleton -> require
 * DATA_MODEL).  The data model's ``_store()`` ADOPTS a pre-mounted ``selection.store`` (its
 * import-time ``_ensureSubscribed`` subscribes to it), so seeding here makes the store the test
 * drives the SAME instance the data model reads.  Without this, the data model would subscribe
 * to its own private store at import and the test's later seed would fork a divergent instance.
 *
 * This lives in the TEST only: production keeps the raw selection store concealed behind the
 * data model's curated ``data.selection.*`` surface (no public ``selection.store`` mount).
 */
globalThis.molbuilder.molview.selection.store =
    globalThis.molbuilder.molview.selection._createStore();

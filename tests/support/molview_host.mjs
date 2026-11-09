/* The page that hosts MolView, as a test plays it.
 *
 * A page that mounts MolView hands it the app's one reader of a server's
 * answer -- `{ask}`, `lib/server.js` -- because MolView reaches no module of
 * the app by name (molview.md § 4; ui-contract.md § 12).  A test that builds
 * a viewer stands where that page stands, so it hands the same reader in: the
 * real one, asking whatever stand-in server the test put on `fetch`.
 */
const LIB = new URL("../../molbuilder/web/static/lib/", import.meta.url);
const { ask } = await import(new URL("server.js", LIB).href);
const MODEL = await import(new URL("molview/model.js", LIB).href);
const MV = await import(new URL("molview/index.js", LIB).href);

export { ask };

/** A MolView model, handed the reader as `mount` hands it (`mount.js`). */
export function createModel(opts) {
    return MODEL.createModel(Object.assign({ ask }, opts || {}));
}

/** MolView mounted as a page mounts it: `opts.server` is the reader. */
export function mount(hostEl, workspace, opts) {
    return MV.mount(hostEl, workspace,
                    Object.assign({ server: { ask } }, opts || {}));
}

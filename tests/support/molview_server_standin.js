/* A stand-in server. It answers the routes of molview.md § 11.1 and
 * records what it was sent, so a test can assert what actually left the
 * browser (the model's tests, `tests/test_molview_model.py`, and the view
 * context's, `tests/test_molview_ui_context.py`). */
globalThis.__requests = [];
globalThis.__serverFails = false;
globalThis.__nextPayload = null;

function atomRow(i, element, x, opts) {
    /* The labels the atom carries, and nothing beside them: a reserved label
     * is IN `regions` (§ 6.6). The row used to carry an `is_frozen` flag too,
     * which is what the server sent while it kept a second store. */
    return Object.assign({ index: i, element, x, y: 0, z: 0, regions: [] },
                         opts || {});
}
globalThis.__atomRow = atomRow;

globalThis.__payload = function (atoms, extra) {
    return Object.assign({ atoms }, extra || {});
};

/* THE STAND-IN SPEAKS THE SERVER'S NAMES, not the module's (§ 13.1: a stand-in
 * "must obey THAT LEVEL's rules"). Its periodicity block is `{cell, engine_offset,
 * axis_kind, vacuum}` and, beside them, what the server resolved (`resolved_cell`,
 * `box_corner`, `resolved_vacuum`) — what /api/build/load actually sends. It used to carry
 * `{lattice, origin}`, which is what the MODULE calls them, so the module's
 * inbound translation was tested against its own output and the cell was silently
 * null against the real server for every structure ever loaded. */

globalThis.fetch = async function (route, init) {
    const body = JSON.parse(init.body);
    globalThis.__requests.push({ route, body });
    if (globalThis.__serverFails) {
        return { ok: false, status: 500, json: async () => ({}) };
    }
    const payload = globalThis.__nextPayload || globalThis.__payload([
        atomRow(0, "C", 0), atomRow(1, "O", 1),
    ]);
    return { ok: true, status: 200, json: async () => payload };
};

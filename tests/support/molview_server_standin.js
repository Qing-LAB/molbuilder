/* A stand-in server. It answers the routes of molview.md § 11.1 and
 * records what it was sent, so a test can assert what actually left the
 * browser (the model's tests, `tests/test_molview_model.py`, and the view
 * context's, `tests/test_molview_ui_context.py`). */
globalThis.__requests = [];
globalThis.__serverFails = false;
globalThis.__nextPayload = null;

/* One atom, as a test states it: its element, where it is, and the labels it
 * carries.  A test's convenience, never sent -- `__payload` puts each fact
 * where the server's answer keeps it. */
function atomRow(i, element, x, opts) {
    return Object.assign({ index: i, element, x, y: 0, z: 0, regions: [] },
                         opts || {});
}
globalThis.__atomRow = atomRow;

/* THE STAND-IN SPEAKS THE SERVER'S NAMES, not the module's (§ 13.1: a stand-in
 * "must obey THAT LEVEL's rules").  Its answer is the server's one shape
 * (`_shared.structure_answer`, `web/web-api.md` § 1): the structure's own dict
 * (`Structure.to_dict`) once, beside the box the server resolved
 * (`resolved_box`), `atom_order_said` and the kind table's `labels`.  It used
 * to carry the cell as `{lattice, origin}`, which is what the MODULE calls
 * them, so the module's inbound translation was tested against its own output
 * and the cell was silently null against the real server for every structure
 * ever loaded.
 *
 * Composed from `atoms`: the elements and the positions as the structure's
 * columns, and each atom's labels into the one label store
 * (`metadata.regions`, label -> its atoms, ascending, an index a row lists
 * twice listed twice -- what a hand-written store could hold).  What a test
 * does not state is filled by THIS STAND-IN'S OWN DEFAULTS, NOT THE SERVER'S:
 * the server fills the identity columns with placeholders, resolves a box
 * and sends the kind table's labels, where these leave the columns empty,
 * the resolved box null and the labels empty.  Feeding the stand-in from the
 * server's composer instead is planned work on the page-side test doors.
 * `extra` speaks the server's names too: `structure` (its fields over these,
 * its `metadata` key by key), `resolved_box`, `notices`, `labels`,
 * `n_frames`, ... */
globalThis.__payload = function (atoms, extra) {
    const rest = Object.assign({}, extra || {});
    const given = rest.structure || {};
    delete rest.structure;
    const regions = {};
    atoms.forEach((a, i) => {
        for (const name of a.regions || []) {
            (regions[name] = regions[name] || []).push(i);
        }
    });
    for (const name of Object.keys(regions)) {
        regions[name].sort((p, q) => p - q);
    }
    const structure = Object.assign({
        title: "",
        elements: atoms.map((a) => a.element),
        positions: atoms.map((a) => [a.x, a.y, a.z]),
        atom_names: [], residue_ids: [], residue_names: [], chain_ids: [],
        info: {},
    }, given, {
        metadata: Object.assign({
            regions, cell: null, engine_offset: null, offset_by: null,
            axis_kind: ["isolated", "isolated", "isolated"], vacuum: null,
            annotations: {}, customized: null, atom_order: null,
            order_unstated: false,
        }, given.metadata || {}),
    });
    // A frame set states `frames` IN PLACE OF `positions` (`to_dict`).
    if (given.frames) delete structure.positions;
    const resolved = rest.resolved_box || {};
    delete rest.resolved_box;
    return Object.assign({
        ok: true,
        structure,
        resolved_box: Object.assign({ resolved_cell: null, box_corner: null,
                                      resolved_vacuum: null }, resolved),
        atom_order_said: null,
        labels: [],
    }, rest);
};

/* Each answer as a browser hands it, for the page's one reader
 * (`lib/server.js`). */
globalThis.fetch = async function (route, init) {
    const body = JSON.parse(init.body);
    globalThis.__requests.push({ route, body });
    if (globalThis.__serverFails) {
        return new Response(JSON.stringify({}), { status: 500 });
    }
    const payload = globalThis.__nextPayload || globalThis.__payload([
        atomRow(0, "C", 0), atomRow(1, "O", 1),
    ]);
    return new Response(JSON.stringify(payload), { status: 200 });
};

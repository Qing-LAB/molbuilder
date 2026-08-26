/* task-setup/cell-readers.js -- what a stage-table cell's TEXT means.
 *
 * Contract: `docs/web/task-setup.md` § 5.2.  One reader per member of
 * `template.TYPES`, because the alternative is what shipped until
 * 2026-08-25: a branch for `bool`, `Number(text)` for anything that parsed
 * as a number, and the RAW STRING for everything else.  `kgrid` is declared
 * `int3`; `Number("4,4,1")` is NaN; so typing the k-grid spelling the CLI
 * itself accepts wrote `"kgrid": "4,4,1"` into the description -- a string
 * where the config declares `Tuple[int, int, int]`.  It saved clean and
 * died four steps later inside a range check, as "this is a programmer
 * bug", naming neither the stage nor the key.  Four columns had it:
 * kgrid, kgrid_displacement, species_order, ecp_atoms.
 *
 * A reader returns `undefined` when the text is not that type.  The text is
 * then kept AS TYPED and the save door refuses it by name (`stages.md`
 * § 6.6's declared-type row).  Storing a half-parsed value would be the
 * quiet version of the same bug.
 *
 * ITS OWN FILE, and not four more lines in a 2500-line page controller,
 * for one reason: a page controller cannot be imported without a DOM, so
 * while these lived in `viewer.js` the only thing a test could check was
 * that the KEYS existed -- `int3: (t) => t` would have passed.  The
 * behaviour was verified by hand, once, which is the assurance this
 * codebase has been burned by before.
 */

//: `4,4,1` / `4x4x1` / `4 4 1` -- the three spellings `--kgrid` takes
//: (`cli.KGridParam`), so what works in the terminal works in the table.
const _gridPieces = (t) => t.split(/[,\sx]+/).filter(Boolean);
//: A LIST separates on commas and spaces only: `x` is a grid spelling, and
//: splitting `species_order` on it would cut a name in half.
const _listPieces = (t) => t.split(/[,\s]+/).filter(Boolean);
//: A WHOLE number, by the same rule the preflight applies to the value it
//: ends up checking: `4.0` IS an integer and `4.7` is not (`validation/
//: task.py::_scalar_complaint`).  A stricter test here would refuse a
//: value the save door accepts, which is the two ends disagreeing about
//: one rule -- the thing this whole change exists to stop.
const _whole  = (t) => {
    const n = Number(t);
    return (t !== "" && Number.isInteger(n)) ? n : undefined;
};
const _number = (t) => {
    const n = Number(t);
    return (t !== "" && Number.isFinite(n)) ? n : undefined;
};
const _allOf = (split, read, n) => (t) => {
    const v = split(t).map(read);
    return (v.length && v.every((x) => x !== undefined)
            && (!n || v.length === n)) ? v : undefined;
};

/* THREE OF THESE HAVE NO CATALOGUE ITEM YET -- READ THIS BEFORE ADDING ONE.
 *
 * A column's type comes from the SHIPPED catalogue, so today no cell can
 * carry `pow2`, `text` or `intlist`: nothing in `catalogue.template.toml`
 * declares one for either engine.  They have readers anyway, and the
 * readers are the answer already decided -- so when such an item does
 * land, THE STAGE TABLE ALREADY HANDLES IT and there is nothing to build:
 *
 *   pow2     read as a whole number.  Snapping to a power of two is NOT
 *            done here -- `template._shape` owns that, and doing it in the
 *            cell would mean two places snap and they would drift.
 *   text     kept verbatim.  `template.TYPES` defines it as engine text to
 *            be COPIED, not interpreted, so interpreting it here would be
 *            the one thing its type forbids.
 *   intlist  whole numbers separated by commas or spaces.  NOT the range
 *            syntax (`0-35, 100`) the Build form's own control accepts --
 *            that belongs to `_parse_int_list_with_ranges` on the server
 *            and is a different control's contract, not this one's.
 *
 * They are here rather than left out because the failure without a reader
 * is SILENT: the lookup misses, `setCell` stores the raw text, and the
 * description carries a string where the config declares a list.  That is
 * this file's entire reason for existing, and a new item is exactly when
 * nobody would think to check.  `test_task_setup_cell_readers_js.py`
 * exercises all eleven, so these three are covered, not merely present.
 */
export const CELL_READERS = {
    bool:    (t) => t === "true",
    enum:    (t) => t,
    str:     (t) => t,
    text:    (t) => t,
    int:     _whole,
    pow2:    _whole,
    float:   _number,
    int3:    _allOf(_gridPieces, _whole, 3),
    float3:  _allOf(_gridPieces, _number, 3),
    intlist: _allOf(_listPieces, _whole),
    strlist: _listPieces,
};

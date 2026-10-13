/* lib/dialog.js -- the ONE modal component's behaviour, beside its look
 * (`lib/dialog.css`; `ui-contract.md` § 1, § 8).
 *
 * What every in-page dialog that ASKS for something needs, said once:
 *
 *   openDialog(dialog, cancelValue)  show it modally, ONE AT A TIME: opening
 *       a second settles the first to its cancel value (no stacking -- one
 *       modal keeps focus and Esc sane); Esc, the native close and a click
 *       off the surface (outside its box) settle to the cancel value.
 *       Resolves with what settleDialog is given.
 *   settleDialog(value)              close the open dialog with an answer.
 *   newDialog / dialogTitle / dialogHint / dialogField / dialogError /
 *   dialogActions                    the parts, in the vocabulary dialog.css
 *       styles (`mb-dialog-*`), so a dialog cannot be half-designed.
 *   askFields(opts)                  a form dialog: one or more labelled
 *       inputs, checked as typed, resolving to their values or null.
 *
 * The ONE CONFIRM, `warningModal.confirm` (`lib/warning-modal.js`, which the
 * page header loads on every page), is built on this module: a yes/no
 * question has one implementation, and the same rules as every dialog here.
 *
 * No HTTP here, and no knowledge of what the answer is for: the caller acts
 * on it.  `ui-contract.md` § 11: an in-page dialog, never `prompt()` /
 * `confirm()` / `alert()` -- a native one blocks the page and the browser's
 * automation alike.
 */

let _active = null;          // { dialog, resolve, cancelValue }

/** Close the open dialog, answering ``value``.  Idempotent. */
export function settleDialog(value) {
  if (!_active) return;
  const { dialog, resolve } = _active;
  _active = null;
  try { dialog.close(); } catch (_) { /* already closed */ }
  try { dialog.remove(); } catch (_) { /* already gone */ }
  resolve(value);
}

/** Show ``dialog`` modally and resolve with its answer -- ``cancelValue``
 *  on Esc, on its native close, and on a click off its surface.
 *
 *  The listeners are IDENTITY-GUARDED: ``dialog.close()`` fires ``close``
 *  on a queued task, so when one dialog settles and its caller at once opens
 *  the next (Copy: the destination picker, then the name), the first one's
 *  late ``close`` must not settle the second.
 *
 *  OFF THE SURFACE MEANS OUTSIDE ITS BOX, pressed and released there: a
 *  click on the dialog's own padding has the dialog for its target too, and
 *  so does a text selection begun in a field and released over the backdrop
 *  -- neither is a person leaving the question. */
export function openDialog(dialog, cancelValue = null) {
  if (_active) settleDialog(_active.cancelValue);
  document.body.appendChild(dialog);
  const mine = () => _active && _active.dialog === dialog;
  const outside = (ev) => {
    if (ev.target !== dialog) return false;
    const box = dialog.getBoundingClientRect();
    return ev.clientX < box.left || ev.clientX > box.right
        || ev.clientY < box.top || ev.clientY > box.bottom;
  };
  let pressedOutside = false;
  dialog.addEventListener("cancel", (ev) => {        // Esc
    // A file field's picker dismissed fires its own `cancel`, which
    // bubbles here: that is not the question being cancelled.
    if (ev.target !== dialog) return;
    ev.preventDefault();
    if (mine()) settleDialog(cancelValue);
  });
  dialog.addEventListener("close", () => {
    if (mine()) settleDialog(cancelValue);
  });
  dialog.addEventListener("pointerdown", (ev) => {
    pressedOutside = outside(ev);
  });
  dialog.addEventListener("click", (ev) => {         // off the surface
    if (pressedOutside && outside(ev) && mine()) settleDialog(cancelValue);
    pressedOutside = false;
  });
  return new Promise((resolve) => {
    _active = { dialog, resolve, cancelValue };
    try {
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
    } catch (_) { dialog.setAttribute("open", ""); }
  });
}

// ─── The parts ──────────────────────────────────────────────────────── //

function _node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

/** A ``<dialog>`` wearing the component, plus ``extra`` classes that name
 *  which dialog it is (for tests and for its own layout rules). */
export function newDialog(extra) {
  return _node("dialog", "mb-dialog" + (extra ? " " + extra : ""));
}
export function dialogTitle(text) { return _node("h2", "mb-dialog-title", text); }
export function dialogHint(text) { return _node("p", "mb-dialog-hint", text); }

/** A labelled input, stacked: ``{wrap, input}``. */
export function dialogField(label, spec) {
  spec = spec || {};
  const wrap = _node("label", "mb-dialog-field");
  wrap.appendChild(_node("span", null, label));
  const input = document.createElement("input");
  input.type = spec.type || "text";
  if (spec.value !== undefined && spec.value !== null) {
    input.value = String(spec.value);
  }
  if (spec.placeholder) input.placeholder = spec.placeholder;
  input.setAttribute("autocomplete", "off");
  input.setAttribute("spellcheck", "false");
  for (const [k, v] of Object.entries(spec.attrs || {})) {
    input.setAttribute(k, String(v));
  }
  if (spec.name) input.setAttribute("data-role", spec.name);
  wrap.appendChild(input);
  return { wrap, input };
}

/** The refusal line -- it reserves its line (dialog.css), so a refusal
 *  appearing under the pointer never moves the buttons. */
export function dialogError() {
  const p = _node("p", "mb-dialog-error");
  p.setAttribute("data-role", "error");
  return p;
}

/** The buttons, in the given order (cancel first: the platform order puts
 *  the answer last).  Each: ``{label, action, kind, onClick}`` -- ``kind``
 *  "primary" or "danger" (page-shell's button variants). */
export function dialogActions(buttons) {
  const row = _node("div", "mb-dialog-actions");
  for (const b of buttons) {
    const btn = _node("button", b.kind || null, b.label);
    btn.type = "button";
    btn.setAttribute("data-action", b.action);
    btn.addEventListener("click", b.onClick);
    row.appendChild(btn);
  }
  return row;
}

// ─── The form dialog ────────────────────────────────────────────────── //

/** Ask for one or more values.
 *
 *  ``opts``: ``{title, hint?, fields: [{name, label, value?, type?,
 *  placeholder?, attrs?}], validate?, confirmLabel?, danger?, extra?}``.
 *  ``validate(values)`` returns ``null`` when the answer can stand, else
 *  the refusal -- ``""`` refuses without words (an empty name).  The confirm
 *  button is enabled only while it can stand; Enter confirms likewise.
 *  ``danger`` marks the confirm as the costly choice (`.danger`) and puts
 *  the first focus on the field, never on the button.
 *
 *  Resolves to ``{name: value}`` -- a text field's value as typed, a file
 *  field's chosen File (or null) -- or ``null`` on cancel. */
export function askFields(opts) {
  opts = opts || {};
  const dialog = newDialog(opts.extra);
  dialog.appendChild(dialogTitle(opts.title || ""));
  if (opts.hint) dialog.appendChild(dialogHint(opts.hint));
  const inputs = [];
  for (const f of opts.fields || []) {
    const { wrap, input } = dialogField(f.label, f);
    inputs.push([f.name, input]);
    dialog.appendChild(wrap);
  }
  const err = dialogError();
  dialog.appendChild(err);

  const read = () => {
    const out = {};
    for (const [name, input] of inputs) {
      out[name] = input.type === "file"
        ? ((input.files && input.files[0]) || null) : input.value;
    }
    return out;
  };
  let confirmBtn = null;
  const check = () => {
    const values = read();
    const why = typeof opts.validate === "function"
      ? opts.validate(values) : null;
    err.textContent = why || "";
    if (confirmBtn) confirmBtn.disabled = why !== null;
    return why === null ? values : null;
  };
  const answer = () => {
    const values = check();
    if (values) settleDialog(values);
  };

  const actions = dialogActions([
    { label: opts.cancelLabel || "Cancel", action: "cancel",
      onClick: () => settleDialog(null) },
    { label: opts.confirmLabel || "OK", action: "confirm",
      kind: opts.danger ? "danger" : "primary", onClick: answer },
  ]);
  confirmBtn = actions.querySelector('[data-action="confirm"]');
  dialog.appendChild(actions);

  for (const [, input] of inputs) {
    input.addEventListener(input.type === "file" ? "change" : "input", check);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && input.type !== "file") {
        e.preventDefault();
        answer();
      }
    });
  }

  const p = openDialog(dialog, null);
  const first = inputs.length ? inputs[0][1] : null;
  try {
    if (first) {
      first.focus();
      if (first.type !== "file" && typeof first.select === "function") {
        first.select();
      }
    }
  } catch (_) { /* a stub without focus */ }
  check();
  return p;
}

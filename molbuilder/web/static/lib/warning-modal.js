/* lib/warning-modal.js -- THE ONE CONFIRM (`ui-contract.md` § 11).
 *
 * Every yes/no question the app asks: the canvas's "discard unsaved
 * modifications?", the sidebar's overwrite and delete, the preview editor's
 * unsaved edits, the server restart.  Built on the one modal component
 * (`lib/dialog.js`), so a confirm keeps its rules: ONE dialog at a time -- a
 * dialog opened over it answers it no -- and Esc, the native close and a
 * click off it answer no.
 *
 * Surface (`window.molbuilder.warningModal`, read at the click by the
 * classic scripts):
 *
 *   const go = await warningModal.confirm({title, body, confirmLabel,
 *                                          cancelLabel});
 *
 * ``body`` is one paragraph, or a list of them -- each its own line of the
 * question (the folder download's size, what it leaves out, the line that
 * copies it instead).
 *   const proceed = await warningModal.confirmDiscardUnsaved();
 *
 * Each resolves true on the costly answer -- marked `.danger`, and last --
 * and false on Cancel, which comes first and takes the focus, so the key
 * that closes the question keeps what you had.
 *
 * With no text given, the question is the discard-unsaved one, whose four
 * strings are fixed copy (`runtime.md` § 3).
 */

import {
  dialogActions, dialogHint, dialogTitle, newDialog, openDialog, settleDialog,
} from "./dialog.js";

const TITLE   = "Unsaved modifications";
const BODY    = "You have unsaved changes to the current canvas. "
              + "Continuing will discard them.";
const CANCEL  = "Cancel";
const DISCARD = "Discard and continue";

/** Ask ``{title, body, confirmLabel, cancelLabel}`` -- each defaulting to
 *  the discard-unsaved question's, ``body`` a paragraph or a list of them
 *  -- and resolve true or false. */
function confirm(opts) {
  opts = opts || {};
  const dialog = newDialog("molbuilder-warning-modal");
  dialog.setAttribute("aria-labelledby", "molbuilder-warning-title");
  const title = dialogTitle(opts.title || TITLE);
  title.id = "molbuilder-warning-title";
  dialog.appendChild(title);
  // EACH PARAGRAPH ITS OWN LINE OF THE QUESTION, the dialog described by
  // all of them, in order.
  const paras = Array.isArray(opts.body) ? opts.body : [opts.body || BODY];
  const ids = paras.map((text, i) => {
    const p = dialogHint(text);
    p.id = "molbuilder-warning-body" + (i ? "-" + (i + 1) : "");
    dialog.appendChild(p);
    return p.id;
  });
  dialog.setAttribute("aria-describedby", ids.join(" "));
  dialog.appendChild(dialogActions([
    { label: opts.cancelLabel || CANCEL, action: "cancel",
      onClick: () => settleDialog(false) },
    { label: opts.confirmLabel || DISCARD, action: "discard", kind: "danger",
      onClick: () => settleDialog(true) },
  ]));
  const answer = openDialog(dialog, false);
  const cancel = dialog.querySelector('[data-action="cancel"]');
  if (cancel && typeof cancel.focus === "function") cancel.focus();
  return answer;
}

export const warningModal = {
  confirm: confirm,
  confirmDiscardUnsaved: confirm,
};

window.molbuilder = window.molbuilder || {};
window.molbuilder.warningModal = warningModal;
if (window.molbuilder.runtime
    && typeof window.molbuilder.runtime.register === "function") {
  window.molbuilder.runtime.register("structure.warningModal", warningModal);
}

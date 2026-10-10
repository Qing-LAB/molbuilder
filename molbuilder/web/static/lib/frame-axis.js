/* frame-axis.js — the one control a chart offers for a calculation's frames.
 *
 * Contract: docs/web/results.md § 2.5 ("The frame family — the frame an axis
 *           of every result"): every chart card of a frame set carries it —
 *           *one frame*, the frame bar's, or *all frames*, a legend entry
 *           each. The frame bar (MolView's model) owns which frame; this
 *           control owns only how many a chart draws.
 * Owns:     lib/frame-axis.css.
 */

/** The two views a chart draws a frame set in. */
export const ONE = "one";
export const ALL = "all";

/**
 * The control, in ``mode``; ``onChange(mode)`` when it is switched. A pair of
 * buttons, the pressed one the view on screen (ui-contract.md § 4.1).
 */
export function frameAxisControl(doc, mode, onChange) {
    const box = doc.createElement("div");
    box.className = "frame-axis";
    box.setAttribute("role", "group");
    box.setAttribute("aria-label", "frames drawn");
    const buttons = [[ONE, "One frame"], [ALL, "All frames"]].map(([m, text]) => {
        const b = doc.createElement("button");
        b.type = "button";
        b.className = "toggle-button frame-axis-btn";
        b.textContent = text;
        b.dataset.mode = m;
        b.setAttribute("aria-pressed", m === mode ? "true" : "false");
        b.addEventListener("click", () => {
            buttons.forEach((o) => o.setAttribute(
                "aria-pressed", o === b ? "true" : "false"));
            onChange(m);
        });
        box.appendChild(b);
        return b;
    });
    return box;
}

/* THE TRANSMISSION WINDOW, FITTED TO A FINISHED GRID (engines/transport.md
 * § 2a.16) -- offered beside the window's fields of a calculation that
 * reuses a finished one, on the Transport tab and on Task setup: the range
 * asked (and how many times finer, an odd number), the fitted ends and count
 * the server answers (`/api/transport/fit`, `transport.window.fitted`), and
 * *use these*, which hands them to the page that writes them into its
 * fields.  A window typed value by value stays as typed; nothing is fitted
 * unasked.
 *
 * One builder for both pages, so the two cannot offer two fits.  The row
 * is made of the shared components both pages load (`form-components.css`
 * `.fdf-actions`; the shell's `.status` and `button`).
 */

/** A row offering the fit for the calculation `reuses` names -- `name`,
 *  as the server names it (`Extension.name`); `use` takes
 *  `{transmission_emin_ev, transmission_emax_ev, transmission_n_points}`. */
export function fitRow({ reuses, name, use, doc = document }) {
    const el = (tag, attrs, ...kids) => {
        const n = doc.createElement(tag);
        for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
        for (const k of kids) n.append(k);
        return n;
    };
    const num = (value, label, step) => {
        const i = el("input", { type: "number", step: step, value: value,
                                "aria-label": label });
        return [el("label", null, label + " ", i), i];
    };
    const [loLabel, lo] = num("-3", "from", "any");
    const [hiLabel, hi] = num("3", "to", "any");
    const [finerLabel, finer] = num("1", "times finer", "2");
    const ask = el("button", { type: "button", class: "btn" }, "Fit");
    const said = el("span", { class: "status", role: "status" });
    // THE ONE STATUS WRITER (`lib/status.js`): a refusal in the error colour.
    const say = (msg, kind) => window.molbuilder.status.set(said, msg,
                                                            kind || null);
    const take = el("button", { type: "button", class: "btn", hidden: "" },
                    "Use these");
    let fitted = null;
    // A RANGE CHANGED is another question: the fit it had answered is gone.
    for (const i of [lo, hi, finer]) {
        i.addEventListener("input", () => {
            fitted = null;
            take.hidden = true;
            say("");
        });
    }
    ask.addEventListener("click", async () => {
        fitted = null;
        take.hidden = true;
        say("Fitting…");
        const q = new URLSearchParams({ reuses: reuses, lo: lo.value,
                                        hi: hi.value, finer: finer.value });
        try {
            const r = await fetch("/api/transport/fit?" + q);
            const b = await r.json();
            if (!b.ok) { say(b.error || "not fitted", "error"); return; }
            fitted = b;
            say(b.words);
            take.hidden = false;
        } catch (e) {
            say(String((e && e.message) || e), "error");
        }
    });
    take.addEventListener("click", () => {
        if (!fitted) return;
        use({ transmission_emin_ev: fitted.emin,
              transmission_emax_ev: fitted.emax,
              transmission_n_points: fitted.points });
    });
    return el("div", { class: "fdf-actions" },
              el("strong", null, "The window, fitted to the grid of "
                 + name + ":"),
              loLabel, hiLabel, finerLabel, ask, said, take);
}

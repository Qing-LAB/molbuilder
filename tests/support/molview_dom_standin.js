/* A stand-in DOM — enough of one for mount.js and the controls.
 *
 * § 13.1: a stand-in takes the place of a level, so it must obey that level's
 * rules. This one stands in for the browser, so it behaves the way the browser
 * does in the ways this module depends on:
 *
 *   - an element built but not appended has no width, and `clientWidth` reports
 *     whatever the host was told to be — which is what § 8.2's sizing contract
 *     is checked against;
 *   - `getComputedStyle` answers custom properties from the nearest ancestor
 *     that set one, because that is how `--molviewer-size-card-min-width` reaches mount.js;
 *   - `classList.toggle` returns whether the class is now present, which is what
 *     the fold reads.
 *
 * It deliberately does NOT implement layout. Nothing here should ever be used to
 * check that something *looks* right — that is § 13.2's third level, in a real
 * page.
 */

function makeElement(doc, tag) {
    const node = {
        tagName: String(tag).toUpperCase(),
        ownerDocument: doc,
        children: [],
        parentNode: null,
        style: {},
        attributes: {},
        _text: "",
        hidden: false,
        _classes: new Set(),
        _listeners: {},
        _customProps: {},

        get className() { return Array.from(node._classes).join(" "); },
        set className(v) {
            node._classes = new Set(String(v).split(/\s+/).filter(Boolean));
        },
        classList: {
            add(c)    { node._classes.add(c); },
            remove(c) { node._classes.delete(c); },
            contains(c) { return node._classes.has(c); },
            /* `force` is honoured, as the real DOM does: `toggle(c, cond)` SETS
             * the class to `cond` rather than flipping it. Without it, the very
             * common "light exactly the chosen one" loop —
             * `for (x of all) x.classList.toggle("molviewer-is-active", x === chosen)` —
             * turned every class on, so a stand-in test could not tell "one lit"
             * from "all lit". That is a stand-in describing something the DOM
             * does not do (§ 13.1). */
            toggle(c, force) {
                if (force !== undefined) {
                    if (force) node._classes.add(c); else node._classes.delete(c);
                    return !!force;
                }
                if (node._classes.has(c)) { node._classes.delete(c); return false; }
                node._classes.add(c); return true;
            },
        },

        get textContent() {
            if (node.children.length === 0) return node._text;
            return node.children.map((c) => c.textContent).join("");
        },
        set textContent(v) { node._text = String(v); node.children = []; },

        setAttribute(k, v) { node.attributes[k] = String(v); },
        getAttribute(k) { return k in node.attributes ? node.attributes[k] : null; },
        // Taking one off is half of the pair, and the half that was missing:
        // the module marks a tab when a page holds a message and unmarks it
        // when the page is opened, which a stand-in with only `set` cannot do.
        removeAttribute(k) { delete node.attributes[k]; },
        hasAttribute(k) { return k in node.attributes; },
        appendChild(child) {
            child.parentNode = node;
            node.children.push(child);
            return child;
        },
        remove() {
            if (!node.parentNode) return;
            const at = node.parentNode.children.indexOf(node);
            if (at >= 0) node.parentNode.children.splice(at, 1);
            node.parentNode = null;
        },
        addEventListener(type, fn) {
            (node._listeners[type] = node._listeners[type] || []).push(fn);
        },
        removeEventListener(type, fn) {
            const list = node._listeners[type] || [];
            const at = list.indexOf(fn);
            if (at >= 0) list.splice(at, 1);
        },
        // Fire a handler the way a user would. A click on a LABEL activates the
        // control inside it, which is how every choice in the panel is made: the
        // radio and the checkbox are styled away, and the label IS the control
        // the user sees. A stand-in whose label clicks did nothing would make a
        // dead tab look exactly like a live one.
        click() {
            if (node.tagName === "LABEL") {
                const input = node.children.find((c) => c.tagName === "INPUT");
                if (input && input.type === "radio") {
                    for (const peer of findAll(doc.body, "input")) {
                        if (peer.name === input.name) peer.checked = peer === input;
                    }
                    input.dispatch("change", { target: input });
                } else if (input && input.type === "checkbox") {
                    input.checked = !input.checked;
                    input.dispatch("change", { target: input });
                }
            }
            node.dispatch("click", { target: node });
        },
        /* A dispatched event carries `stopPropagation`, as a real one does. A
         * handler on a control inside a clickable row calls it so the row does
         * not act on the same gesture; a stand-in without it throws there, which
         * would make the module look broken for obeying the DOM. */
        dispatch(type, event) {
            const base = { target: node, stopPropagation() {}, preventDefault() {} };
            for (const fn of (node._listeners[type] || []).slice()) {
                fn(Object.assign(base, { target: node }, event || {}));
            }
        },

        querySelector(selector) { return find(node, selector); },
        querySelectorAll(selector) { return findAll(node, selector); },

        // "Is this node inside me" — what a click handler asks to tell a click
        // on its own menu from a click anywhere else.
        contains(other) {
            for (let at = other; at; at = at.parentNode) if (at === node) return true;
            return false;
        },

        /* Still no layout: this reports whatever the test SAID this element
         * occupies (`_rect`), exactly as `clientWidth` reports what the host was
         * told to be, and zeros when nothing was said. A popover fixed to the
         * viewport has to be placed against its trigger's rectangle, so a
         * stand-in that cannot be asked for one cannot stand in for the browser
         * here at all. */
        getBoundingClientRect() {
            const r = node._rect || {};
            const top = r.top || 0, left = r.left || 0;
            const width = r.width || 0, height = r.height || 0;
            return {
                top, left, width, height,
                right: r.right == null ? left + width : r.right,
                bottom: r.bottom == null ? top + height : r.bottom,
            };
        },

        // Width the host was told to be. Zero means "not laid out", which is how
        // a real element that was never appended reports.
        clientWidth: 0,
        clientHeight: 0,
    };

    /* `<details>` is the one element the module drives by PROPERTY: it opens and
     * closes its menus by assigning `open`, and the browser answers with a
     * `toggle` event — which is where the menu places its popover. A stand-in
     * that took the assignment silently would let a menu that never places
     * itself, and so shows nothing, pass every test. */
    if (node.tagName === "DETAILS") {
        let open = false;
        Object.defineProperty(node, "open", {
            get() { return open; },
            set(value) {
                open = !!value;
                node.dispatch("toggle", { target: node });
            },
        });
    }
    return node;
}

/* ONE step of a selector: a tag, a class, or a compound of the two —
 * `div`, `.molviewer-is-active`, `input`, `.molviewer-menu-background-swatch.molviewer-is-transparent`. */
function matches(node, step) {
    const parts = String(step).split(".");
    const tag = parts.shift();
    if (tag && node.tagName !== tag.toUpperCase()) return false;
    for (const cls of parts) if (!node._classes.has(cls)) return false;
    return true;
}

/* Descendant chains too — `.molviewer-menu-radius-row input`. A stand-in that only
 * understood one step made a test asking a real question ("is the slider inside
 * the row the stylesheet styles?") fail as though the control were missing,
 * which is the stand-in describing something the DOM does not (§ 13.1). */
function findAll(root, selector) {
    let current = [root];
    for (const step of String(selector).trim().split(/\s+/)) {
        const next = [];
        for (const from of current) {
            (function walk(n) {
                for (const c of n.children) {
                    if (matches(c, step) && next.indexOf(c) < 0) next.push(c);
                    walk(c);
                }
            })(from);
        }
        current = next;
    }
    return current;
}

function find(root, selector) {
    const all = findAll(root, selector);
    return all.length ? all[0] : null;
}

const doc = {
    createElement(tag) { return makeElement(doc, tag); },
    createTextNode(text) {
        const node = makeElement(doc, "#text");
        node._text = String(text);
        return node;
    },
    get defaultView() { return globalThis; },
};
doc.body = makeElement(doc, "body");
doc.documentElement = makeElement(doc, "html");

/* The DOCUMENT and the WINDOW take listeners too, and a module that hangs one on
 * either must be able to take it off again. Both are given the element's own
 * listener machinery rather than a second implementation of it — a stand-in with
 * two ideas of what a listener is would be a place for the module to look
 * correct while being wrong. */
function listenable(target) {
    const listeners = {};
    target.addEventListener = (type, fn) => {
        (listeners[type] = listeners[type] || []).push(fn);
    };
    target.removeEventListener = (type, fn) => {
        const list = listeners[type] || [];
        const at = list.indexOf(fn);
        if (at >= 0) list.splice(at, 1);
    };
    target.dispatch = (type, event) => {
        for (const fn of (listeners[type] || []).slice()) fn(event || {});
    };
    target.listenerCount = (type) => (listeners[type] || []).length;
}
listenable(doc);
listenable(globalThis);

// A window has a size. Zero would say "no viewport", which is not a state a
// browser is ever in and would make every placement clamp against nothing.
globalThis.innerWidth = 1200;
globalThis.innerHeight = 800;

globalThis.document = doc;

/* `--molviewer-size-card-min-width` is declared on the card by the stylesheet, so mount.js
 * reads it rather than writing the number itself (§ 8.2). The stand-in serves it
 * from whatever the test set on the nearest ancestor that has one. */
globalThis.getComputedStyle = function (node) {
    return {
        getPropertyValue(name) {
            let at = node;
            while (at) {
                if (at._customProps && name in at._customProps) {
                    return at._customProps[name];
                }
                at = at.parentNode;
            }
            return "";
        },
    };
};

globalThis.__makeHost = function (width, minWidth) {
    const host = makeElement(doc, "div");
    host.clientWidth = width == null ? 900 : width;
    // What the stylesheet would declare on the card.
    host._customProps = { "--molviewer-size-card-min-width": (minWidth == null ? 350 : minWidth) + "px" };
    doc.body.appendChild(host);
    return host;
};

globalThis.URL = globalThis.URL || {};
globalThis.URL.createObjectURL = () => "blob:stand-in";
globalThis.URL.revokeObjectURL = () => {};
globalThis.Blob = globalThis.Blob || class { constructor(p, o) { this.type = o && o.type; } };

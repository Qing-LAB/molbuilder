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
 *     that set one, because that is how `--molview-min-width` reaches mount.js;
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
            toggle(c) {
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
        // Fire a handler the way a user would.
        click() { node.dispatch("click", { target: node }); },
        dispatch(type, event) {
            for (const fn of (node._listeners[type] || []).slice()) {
                fn(Object.assign({ target: node }, event || {}));
            }
        },

        querySelector(selector) { return find(node, selector); },
        querySelectorAll(selector) { return findAll(node, selector); },

        // Width the host was told to be. Zero means "not laid out", which is how
        // a real element that was never appended reports.
        clientWidth: 0,
        clientHeight: 0,
    };
    return node;
}

function matches(node, selector) {
    if (selector.startsWith(".")) return node._classes.has(selector.slice(1));
    return node.tagName === selector.toUpperCase();
}

function findAll(root, selector) {
    const out = [];
    (function walk(n) {
        for (const c of n.children) {
            if (matches(c, selector)) out.push(c);
            walk(c);
        }
    })(root);
    return out;
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

globalThis.document = doc;

/* `--molview-min-width` is declared on the card by the stylesheet, so mount.js
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
    host._customProps = { "--molview-min-width": (minWidth == null ? 350 : minWidth) + "px" };
    doc.body.appendChild(host);
    return host;
};

globalThis.URL = globalThis.URL || {};
globalThis.URL.createObjectURL = () => "blob:stand-in";
globalThis.URL.revokeObjectURL = () => {};
globalThis.Blob = globalThis.Blob || class { constructor(p, o) { this.type = o && o.type; } };

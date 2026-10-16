"""What a page holds after the thing that put it there is gone -- for a test
that asks whether opening and closing a viewer again and again leaves on the
page what doing it once left."""

#: A JavaScript expression: a spy put on the page.  It pairs every add of a
#: listener with its remove on ``EventTarget`` itself, as the browser pairs
#: them -- by target, type, function and capture, and an add the browser
#: ignores (the same four again) is not counted -- each listener named with
#: what added it; and it keeps every WebGL context made.
#:
#: ``spy.state()``: what outlives -- the listeners on the window, the
#: document or an element still on the page, sorted, and the count of
#: contexts not lost.  ``spy.quietly(target, type, fn, options)`` adds a
#: listener the spy does not count (a test's own wait).  ``spy.restore()``
#: takes the spy off.
SPY = r"""(() => {
    const live = [];
    const contexts = [];
    const add = EventTarget.prototype.addEventListener;
    const remove = EventTarget.prototype.removeEventListener;
    const getContext = HTMLCanvasElement.prototype.getContext;
    const capture = (o) => typeof o === "boolean" ? o : !!(o && o.capture);
    const at = (t, type, fn, o) => live.findIndex(([lt, ly, lf, lc]) =>
        lt === t && ly === type && lf === fn && lc === capture(o));
    const name = (t) => t === window ? "window"
        : t === document ? "document"
        : (t && t.nodeName) ? t.nodeName + (t.id ? "#" + t.id : "")
        : String(t);
    EventTarget.prototype.addEventListener = function (type, fn, o) {
        if (fn && at(this, type, fn, o) < 0) {
            const by = String(new Error().stack || "").split("\n")
                .slice(2, 4).map((s) => s.trim()).join(" | ");
            live.push([this, type, fn, capture(o), by]);
        }
        return add.call(this, type, fn, o);
    };
    EventTarget.prototype.removeEventListener = function (type, fn, o) {
        const i = at(this, type, fn, o);
        if (i >= 0) live.splice(i, 1);
        return remove.call(this, type, fn, o);
    };
    HTMLCanvasElement.prototype.getContext = function (kind, ...a) {
        const g = getContext.call(this, kind, ...a);
        if (g && /webgl/.test(kind) && !contexts.includes(g)) contexts.push(g);
        return g;
    };
    const outlives = (t) => t === window || t === document
        || (t instanceof Node && t.isConnected);
    return {
        state: () => ({
            listeners: live.filter(([t]) => outlives(t))
                .map(([t, type, , c, by]) =>
                    name(t) + ":" + type + (c ? " (capture)" : "") + " <= " + by)
                .sort(),
            contexts: contexts.filter((g) => !g.isContextLost()).length }),
        quietly: (t, type, fn, o) => add.call(t, type, fn, o),
        restore: () => {
            EventTarget.prototype.addEventListener = add;
            EventTarget.prototype.removeEventListener = remove;
            HTMLCanvasElement.prototype.getContext = getContext;
        },
    };
})()"""


def grew(left: dict) -> list:
    """The listeners ``left["again"]`` holds more of than ``left["once"]``
    -- what opening again left that opening once did not."""
    once, again = left["once"]["listeners"], left["again"]["listeners"]
    return sorted({x for x in again if again.count(x) > once.count(x)})

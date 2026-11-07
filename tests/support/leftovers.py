"""What a page holds after the thing that put it there is gone -- for a test
that asks whether doing a thing again and again, a viewer opened and put
away, leaves on the page what doing it once left (:func:`what_grew`)."""

#: A JavaScript expression: a spy put on the page.  It pairs every add of a
#: listener with its remove on ``EventTarget`` itself, as the browser pairs
#: them -- by target, type, function and capture, and an add the browser
#: ignores (the same four again) is not counted -- each listener named with
#: what added it; and it keeps every WebGL context made.
#:
#: ``spy.state()``: what outlives -- the listeners on the window, the
#: document or an element still on the page, sorted, and the count of
#: contexts not lost.  ``spy.restore()`` takes the spy off.
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
        restore: () => {
            EventTarget.prototype.addEventListener = add;
            EventTarget.prototype.removeEventListener = remove;
            HTMLCanvasElement.prototype.getContext = getContext;
        },
    };
})()"""


def what_grew(page, act, *, first=None, times: int = 5) -> list:
    """What ``page`` holds after ``times`` rounds that it did not hold
    after one: ``first`` done (``act``, when no ``first`` is named), what
    the page holds read, ``act`` done ``times - 1`` times more, read again
    -- each listener held more often, named with what added it, and the
    live GL contexts when their count moved; ``[]`` when the page holds
    what it held.

    THE SPY GOES ON HERE, ON THE PAGE AS IT STANDS, as ``window.__spy``
    (:data:`SPY`), and comes off when the rounds end: both readings count
    from the same moment, so what the page did before -- its own start, the
    viewer it already shows -- is in neither, and a test's own listener
    (a wait on the page's word) goes on before this is called, uncounted.
    ``first`` may wait on ``window.__spy.state()`` for a drawing window made
    where nothing on the page shows it."""
    page.evaluate("() => { window.__spy = " + SPY + "; }")
    try:
        (first or act)()
        once = page.evaluate("() => window.__spy.state()")
        for _ in range(times - 1):
            act()
        again = page.evaluate("() => window.__spy.state()")
    finally:
        page.evaluate("() => window.__spy.restore()")
    held, more = once["listeners"], again["listeners"]
    grew = sorted({x for x in more if more.count(x) > held.count(x)})
    if again["contexts"] != once["contexts"]:
        grew.append(f"live GL contexts: {once['contexts']} after one round, "
                    f"{again['contexts']} after {times}")
    return grew

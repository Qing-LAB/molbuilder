"""Static assets are revalidated, never served stale from the browser's cache.

WHAT THIS PREVENTS.  Flask's default caches a static file for 12 hours with no
check.  So a changed ``.js`` or ``.css`` keeps loading from the browser's copy
until that expires -- and the whole time the SERVER is serving the new bytes and
the browser never asks for them.  That is why a front-end change has looked like
it "needs a server restart": restarting changed nothing on the server side, it
just happened to coincide with someone hard-reloading.

It cost real time on 2026-08-02: a CSS namespace rename was served fresh while
the page still ran the old markup, and the tab's op sub-tabs stopped working in a
way that read exactly like a code bug.

WHY NOT A VERSION ON THE URL, which is the usual fix.  It only reaches URLs the
SERVER builds.  119 of this app's asset references come through
``url_for('static')`` -- but 51 more are ESM imports written inside the
JavaScript itself (``export { mount } from "./mount.js"``), which no template
sees and no ``url_for`` can rewrite.  Versioning the entry point would leave the
whole module graph behind it on cached copies, which is the half that actually
breaks.  Revalidation is a property of how a file is SERVED, so it covers every
reference without a build step.

WHY THIS IS FREE.  ``no-cache`` does not mean "do not store" -- the browser keeps
its copy and asks whether it is still good.  Unchanged files come back 304 with
no body, so the bandwidth saving that mattered is kept.

AND WHY IT DOES NOT WEAKEN THE RATE LIMITER: it raises requests per page load,
but ``rate_limit.py`` counts only 4xx (``if not (400 <= status_code < 500):
return``).  A 200 or a 304 is discarded before it reaches any buffer.  Pinned
below, because "revalidation is invisible to the limiter" is the assumption that
makes this change safe.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")


@pytest.fixture
def app():
    from molbuilder.web.app import create_app
    return create_app(config={})


def test_a_static_file_is_revalidated_not_blindly_cached(app):
    """The response tells the browser to ask before reusing its copy."""
    r = app.test_client().get("/static/modify/style.css")
    assert r.status_code == 200
    cache = r.headers.get("Cache-Control", "")
    assert "no-cache" in cache or "max-age=0" in cache, (
        f"a static file came back cacheable-without-checking: {cache!r}. "
        f"A changed .js/.css would then keep loading from the browser's copy "
        f"while the server serves the new bytes."
    )
    assert r.headers.get("ETag"), (
        "no ETag, so the browser has nothing to revalidate WITH and every "
        "check would re-download the whole file"
    )


def test_an_unchanged_file_costs_no_body(app):
    """Revalidation is cheap: unchanged means 304 and nothing resent.

    This is what makes `no-cache` affordable on a page with ~170 assets.
    """
    client = app.test_client()
    first = client.get("/static/modify/style.css")
    again = client.get("/static/modify/style.css",
                       headers={"If-None-Match": first.headers["ETag"]})
    assert again.status_code == 304, (
        f"an unchanged file came back {again.status_code}, not 304 — "
        f"every page load would re-download every asset"
    )
    assert not again.get_data(), "a 304 must carry no body"


def test_the_es_module_graph_is_covered_too(app):
    """The files reached ONLY by an import inside JavaScript revalidate as well.

    `molview/index.js` re-exports from `./mount.js`; nothing in any template
    names that file, so a URL-versioning scheme would never touch it.  This is
    the case that decided the approach.
    """
    r = app.test_client().get("/static/lib/molview/mount.js")
    assert r.status_code == 200
    assert "no-cache" in r.headers.get("Cache-Control", "")


def test_revalidation_is_invisible_to_the_rate_limiter():
    """Only 4xx feeds the abuse counter, so 200s and 304s cannot trip it.

    The limiter exists to catch someone probing for weak points -- rapid path
    enumeration, which generates 4xx.  Legitimate traffic passes freely, and
    that is what makes it safe to multiply the requests a page makes.

    Pinned against the source rather than by firing 200 requests at a test
    client: the predicate IS the guarantee, and a test that only fired traffic
    would still pass if someone widened it to 3xx.
    """
    import inspect
    from molbuilder.web import rate_limit

    src = inspect.getsource(rate_limit)
    assert "if not (400 <= status_code < 500):" in src, (
        "the rate limiter no longer counts 4xx-only. If it now counts 2xx or "
        "3xx, revalidating ~170 assets per page load will trip it on ordinary "
        "use -- see the DESIGN NOTE in rate_limit.py, which disabled the "
        "total-burst signal for exactly this reason."
    )

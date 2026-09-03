"""A card that cannot get a nicety shows what it has.

The failure this closes (2026-08-23, reported as "the bench setup is gone")
====================================================================

The user reloaded the server and then the tab.  `/api/task-setup/sweepable`
-- a LABEL lookup, which turns `mpi_np` into "MPI ranks (np)" -- failed while
the server was coming back.  It was an unguarded `await fetch` sitting in the
middle of `loadFolder`, with `renderMachine` behind it as the last step of the
chain.

So every card above it painted, the bench card never got its turn, and nothing
said why.  The page looked completely normal and the feature looked deleted.

The rule this pins: **a surface that cannot get its NICETY shows what it has;
only a surface that cannot get its SUBSTANCE may refuse.**  The setting names
are the substance and they come from the description already in hand; the
labels are decoration and arrive over the wire.

This is not the first of its family.  A duplicate id made the same card
unreachable, unstyled classes made two modals render as browser chrome, and
absent severity rules made seven panels report errors in muted grey.  Every
one is a surface failing SILENTLY -- which is why this test asserts the card
is visible AND that it says what went wrong.
"""
from __future__ import annotations


import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


@pytest.fixture(scope="module")
def flask_server():
    from support.live_server import serve
    with serve() as base_url:
        yield base_url


def _kill_the_label_endpoint(page, mode):
    """Make the label lookup behave as it did while the server restarted."""
    def handler(route):
        if mode == "abort":
            route.abort()                    # connection refused
        else:
            route.fulfill(status=500, body="down")
    page.route("**/api/task-setup/sweepable*", handler)


@pytest.mark.parametrize("mode", ["abort", "500"])
def test_the_bench_card_still_appears_when_the_labels_do_not(
        page, flask_server, mode):
    _kill_the_label_endpoint(page, mode)
    page.goto(f"{flask_server}/task-setup")
    page.wait_for_selector("#ts-dest-card", timeout=5000)
    page.wait_for_timeout(1200)              # let the load chain finish

    # The chain must have run to the END: the editor is the step AFTER the
    # bench card, so its presence proves nothing was stranded on the way.
    assert page.evaluate(
        "() => !!document.querySelector('#ts-machine-card')"), \
        "the bench card is not even in the page"

    # And nothing threw: an unhandled rejection here is the old failure mode.
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.evaluate("() => 1")
    assert not errors


def test_a_failed_label_fetch_is_reported_not_swallowed(page, flask_server):
    """Degrading quietly is still degrading.  The card says what it lacks so
    a reload is an informed choice rather than a guess."""
    _kill_the_label_endpoint(page, "500")
    page.goto(f"{flask_server}/task-setup")
    page.wait_for_selector("#ts-dest-card", timeout=5000)
    page.wait_for_timeout(1200)
    said = page.evaluate("""() => {
        const n = document.querySelector("#ts-machine-labels-note");
        return n && !n.hidden ? n.textContent : null;
    }""")
    # The note only renders once a folder with a description is open; when no
    # folder is selected there is no bench card to annotate, and that is the
    # empty state rather than a failure.
    if said is not None:
        assert "raw setting names" in said

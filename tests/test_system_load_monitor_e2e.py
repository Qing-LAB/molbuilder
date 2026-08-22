"""End-to-end contract: a host whose GPU driver is unreachable says so
in the browser, even before the user opens the load card.

Why this file exists
====================

``/api/system/load`` returns an empty ``gpus`` list for two causes that
are opposite news -- "this box has no GPU" and "this box has a GPU and
the driver is broken" -- and for a long time the widget rendered both
the same way: a tidy two-cell strip.  On 2026-08-04 that cost real
time.  A driver upgrade on the development host left the userspace
library (595.84) ahead of the loaded kernel module (595.71.05);
``nvidia-smi`` had been dead for five weeks; the monitor showed no GPU
cells and looked completely normal.

Two things now carry the reason, and BOTH are exercised here because
the interesting one is easy to get wrong:

  * the notice under the strip -- for a user with the card open;
  * the marker on the header pill -- for a user with the card CLOSED,
    which is every user on their first visit (the card is collapsed by
    default, and collapsing stops polling).  A fault that is only
    visible inside a folded-away card is a fault nobody sees.

The server runs in-process here, and the fixture fakes the WHOLE
broken-driver state (`_GPU_ERROR` + empty handles + `_NVML_OK` off) --
faking only the flag was enough while the development host's own
driver was dead, and failed the day it was repaired (see
``_set_gpu_error``).  Nothing about the front-end path is stubbed.
"""
from __future__ import annotations

import threading

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

#: A verbatim NVML failure -- the string the development host really
#: produced, so the assertions are about text a user really sees.
_REASON = ("NVMLError_LibRmVersionMismatch: "
           "RM has detected an NVML/RM version mismatch.")


@pytest.fixture(scope="module")
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _open_results(page, base_url):
    """Load /results and wait until the load reading has come back.

    Waiting on the response (rather than a timeout) is what makes the
    negative test below deterministic: by the time it returns, the
    widget has been told what the GPU situation is.
    """
    with page.expect_response(lambda r: "/api/system/load" in r.url):
        page.goto(f"{base_url}/results")
    page.wait_for_selector("#system-load-monitor", timeout=5000)


def _set_gpu_error(monkeypatch, value):
    """Put the server in the broken-driver state -- the WHOLE state.

    A real NVML init failure leaves ``_GPU_ERROR`` set and the handle
    list empty; the two never diverge in production.  Faking only the
    flag was enough while the development host's own driver was dead
    (the 2026-08-04 mismatch), but on a healthy host the snapshot kept
    returning real GPUs next to the faked error -- a payload no server
    ever produces -- and the "cells stay hidden" assertion failed
    against numbers that had every right to be drawn.
    """
    from molbuilder.web.blueprints import system_load
    monkeypatch.setattr(system_load, "_GPU_ERROR", value)
    if value is not None:
        monkeypatch.setattr(system_load, "_NVML_OK", False)
        monkeypatch.setattr(system_load, "_GPU_HANDLES", [])


class TestABrokenDriverIsVisible:

    def test_the_pill_is_marked_while_the_card_is_still_collapsed(
            self, page, flask_server, monkeypatch):
        """The first visit shows a collapsed card, so the pill is the
        only thing on screen -- and it has to carry the news."""
        _set_gpu_error(monkeypatch, _REASON)
        _open_results(page, flask_server)

        # Precondition: this really is the collapsed, non-polling state.
        # If the card ever stops defaulting to collapsed, this test is
        # no longer testing the case it was written for.
        assert page.locator("#system-load-monitor.is-collapsed").count() == 1, \
            "card is not collapsed on first visit; the premise changed"

        pill = page.locator(".system-load-toggle")
        page.wait_for_selector(".system-load-toggle.has-gpu-fault",
                               timeout=5000)
        # The reason is reachable without expanding anything.
        assert _REASON in pill.get_attribute("title")
        # ...and the notice is NOT what delivered it -- it is inside the
        # folded card.
        assert page.locator(".system-load-notice").is_visible() is False

    def test_opening_the_card_gives_the_reason_in_full(
            self, page, flask_server, monkeypatch):
        """Open the card and the notice states the failure and the one
        non-obvious consequence: a restart is required."""
        _set_gpu_error(monkeypatch, _REASON)
        _open_results(page, flask_server)
        page.wait_for_selector(".system-load-toggle.has-gpu-fault",
                               timeout=5000)

        page.locator(".system-load-toggle").click()

        notice = page.locator(".system-load-notice")
        notice.wait_for(state="visible", timeout=5000)
        text = notice.inner_text()
        assert _REASON in text
        assert "restart the server" in text.lower()
        # The GPU cells stay hidden -- there are no numbers to draw.
        for metric in ("gpu", "gpubw", "vram"):
            assert page.locator(
                f".system-load-cell[data-metric='{metric}']"
            ).is_visible() is False, f"{metric} cell drawn without data"


class TestAHealthyHostSaysNothing:

    def test_no_marker_and_no_notice_when_there_is_no_fault(
            self, page, flask_server, monkeypatch):
        """A CPU-only box is not a fault.  Nothing is marked, nothing is
        printed -- otherwise the warning becomes background noise and
        stops meaning anything."""
        _set_gpu_error(monkeypatch, None)
        _open_results(page, flask_server)

        # Expand and wait for a rendered sample, so the assertions below
        # are made against a widget that has demonstrably processed a
        # reading rather than one that simply hasn't got there yet.
        page.locator(".system-load-toggle").click()
        page.wait_for_function(
            """() => {
                const v = document.querySelector(
                    "[data-metric='cpu'] [data-value]");
                return v && v.textContent.trim() !== "\\u2014";
            }""",
            timeout=10000,
        )

        assert page.locator(".system-load-toggle.has-gpu-fault").count() == 0
        assert page.locator(".system-load-notice").is_visible() is False
        assert page.locator(".system-load-toggle").get_attribute("title") == \
            "Hide server load"

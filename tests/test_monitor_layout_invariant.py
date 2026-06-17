"""Pin the in-flow layout for the system-load monitor (PR 5,
2026-06-17 user-report-2).

Background
==========

PR 1 (2026-06-17) made the monitor ``position: fixed; bottom: 0``
with scroll-padding-bottom compensation on ``.results-main``.  Users
reported it STILL occupied screen real estate and the visual overlay
read as "blocking" content even with pointer-events: none on the
strip.

PR 5 moves the monitor INSIDE ``.results-main`` as a normal flow
element.  It scrolls with content; never overlaps plots, the file
picker, or inspector cards.  Foldable; collapse stops polling.

Invariants pinned here:

1. ``#system-load-monitor`` has NO ``position: fixed`` rule.
2. ``.results-main`` no longer reserves ``--monitor-height`` --
   the in-flow strip takes its own height naturally.
3. ``--monitor-height`` CSS var family is retired (no consumers).
4. The collapse-by-default behavior survives (sessionStorage key
   missing OR "1" -> collapsed; "0" opts in to expanded).
5. The HTML include lives INSIDE ``<main class="results-main">``,
   not in the global page header.
6. Collapsed state STILL stops polling (the JS calls stopTimer).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_STATIC = (Path(__file__).resolve().parent.parent
           / "molbuilder" / "web" / "static")
_TEMPLATES = (Path(__file__).resolve().parent.parent
              / "molbuilder" / "web" / "templates")
_LIB = _STATIC / "lib"
_RESULTS = _STATIC / "results"


def _strip_css_comments(body: str) -> str:
    """Strip /* ... */ comments so the assertion doesn't trip on
    historical-context strings inside comment blocks."""
    return re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)


# --------------------------------------------------------------------- #
#  CSS-side invariants                                                  #
# --------------------------------------------------------------------- #


class TestMonitorIsInFlow:
    """The monitor MUST NOT be ``position: fixed``.  It lives inside
    .results-main as a normal flex-column row."""

    @pytest.fixture(scope="class")
    def css_body(self):
        return (_LIB / "system-load-monitor.css").read_text()

    def test_no_position_fixed(self, css_body):
        cleaned = _strip_css_comments(css_body)
        m = re.search(
            r"#system-load-monitor\s*\{([^}]*)\}",
            cleaned, re.DOTALL,
        )
        assert m is not None, "monitor selector missing"
        body = m.group(1)
        assert "position: fixed" not in body, (
            "system-load-monitor.css has `position: fixed` again.  "
            "PR 5 made the monitor in-flow because the fixed overlay "
            "kept overlapping content even with pointer-events: none. "
            "Either revert PR 5 (and reinstate scroll-padding "
            "compensation) or remove the fixed positioning here.")

    def test_no_bottom_zero(self, css_body):
        cleaned = _strip_css_comments(css_body)
        m = re.search(
            r"#system-load-monitor\s*\{([^}]*)\}",
            cleaned, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "bottom: 0" not in body, (
            "system-load-monitor.css pins the strip to bottom: 0.  "
            "PR 5 moved it in-flow; this rule is obsolete.")


class TestResultsMainNoMonitorReservation:
    """``.results-main`` no longer reserves ``--monitor-height`` --
    the strip takes its own height naturally."""

    @pytest.fixture(scope="class")
    def results_css(self):
        return (_RESULTS / "style.css").read_text()

    def test_no_scroll_padding_bottom_monitor_height(self, results_css):
        cleaned = _strip_css_comments(results_css)
        m = re.search(
            r"\.results-main\s*\{([^}]*)\}",
            cleaned, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        # Either there's no scroll-padding-bottom rule, OR it doesn't
        # reference --monitor-height.
        bad = re.search(
            r"scroll-padding-bottom\s*:\s*var\(--monitor-height",
            body,
        )
        assert bad is None, (
            ".results-main still reserves --monitor-height in "
            "scroll-padding-bottom.  PR 5 made the monitor in-flow; "
            "this reservation is obsolete.")

    def test_no_padding_bottom_monitor_height(self, results_css):
        cleaned = _strip_css_comments(results_css)
        m = re.search(
            r"\.results-main\s*\{([^}]*)\}",
            cleaned, re.DOTALL,
        )
        assert m is not None
        body = m.group(1)
        assert "--monitor-height" not in body, (
            ".results-main still references --monitor-height.  "
            "PR 5 retired that CSS variable family; the in-flow "
            "monitor takes its own height naturally.")


class TestMonitorHeightVarRetired:
    """The --monitor-height token family was only useful for the
    scroll-padding reservation.  After PR 5 there are no consumers,
    so the JS no longer needs to write the var on toggle."""

    @pytest.fixture(scope="class")
    def js_body(self):
        return (_LIB / "system-load-monitor.js").read_text()

    def test_no_setProperty_monitor_height(self, js_body):
        assert not re.search(
            r"setProperty\s*\(\s*[\"']--monitor-height[\"']",
            js_body,
        ), ("system-load-monitor.js still writes --monitor-height to "
            ":root, but PR 5 retired the var (no CSS consumer).  "
            "Either remove the write or document who reads it now.")


# --------------------------------------------------------------------- #
#  Template invariant -- monitor lives INSIDE <main class="results-main">
# --------------------------------------------------------------------- #


class TestMonitorIncludedInsideResultsMain:
    """The {% include %} for _system_load_monitor.html MUST appear
    AFTER <main class="results-main"> opens and BEFORE it closes."""

    @pytest.fixture(scope="class")
    def results_html(self):
        return (_TEMPLATES / "results.html").read_text()

    def test_include_inside_results_main(self, results_html):
        # Find the open/close positions of <main class="results-main">
        # and the include line position.
        open_m = re.search(
            r'<main\s+class="results-main"\s*>',
            results_html,
        )
        close_m = re.search(r"</main>", results_html)
        include_m = re.search(
            r'\{%\s*include\s*"_system_load_monitor\.html"',
            results_html,
        )
        assert open_m is not None, "<main class='results-main'> not found"
        assert close_m is not None, "</main> not found"
        assert include_m is not None, (
            "_system_load_monitor.html is not included in results.html. "
            "Did the monitor get removed?")
        assert open_m.end() < include_m.start() < close_m.start(), (
            "_system_load_monitor.html include is NOT inside "
            "<main class='results-main'>.  PR 5 moved it inside so "
            "the monitor lays out as a normal flow element; "
            "regressing this re-introduces the overlay-overlap bug "
            "the user reported 2026-06-17.")


# --------------------------------------------------------------------- #
#  JS-side invariants (default-collapsed + collapse stops polling)      #
# --------------------------------------------------------------------- #


class TestJsDefaultsCollapsed:
    """First-visit default = collapsed.  sessionStorage key missing
    -> collapsed; explicit "0" opts user IN to expanded."""

    @pytest.fixture(scope="class")
    def js_body(self):
        return (_LIB / "system-load-monitor.js").read_text()

    def test_default_collapsed_when_key_missing(self, js_body):
        assert re.search(
            r"startCollapsed\s*=\s*\(\s*saved\s*===\s*null\s*\)\s*\?\s*true",
            js_body,
        ), ("system-load-monitor.js no longer defaults to collapsed "
            "when sessionStorage key is missing.")


class TestCollapseStopsPolling:
    """When the user collapses the strip, the 1 Hz poll timer MUST
    stop (and the in-flight fetch abort).  Polling resumes on
    expand."""

    @pytest.fixture(scope="class")
    def js_body(self):
        return (_LIB / "system-load-monitor.js").read_text()

    def test_applyCollapsed_calls_stopTimer(self, js_body):
        m = re.search(
            r"function\s+applyCollapsed\s*\([^)]*\)\s*\{(.+?)\n\s{8}\}",
            js_body, re.DOTALL,
        )
        assert m is not None, "applyCollapsed body shape changed"
        body = m.group(1)
        assert "stopTimer" in body, (
            "applyCollapsed no longer calls stopTimer() on collapse. "
            "The 1 Hz poll keeps running invisibly even when the "
            "user folded the strip away.")
        assert "startTimer" in body, (
            "applyCollapsed no longer calls startTimer() on expand. "
            "The expanded strip wouldn't update; sparklines would "
            "stay frozen at the moment of expand.")

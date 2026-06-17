"""Pin the layout invariant from results-state-contract.md § 9.

Background
==========

The system-load monitor lives at the bottom of every page via
``position: fixed; bottom: 0``.  Pre-2026-06-17 the monitor was
expanded by default and the results-tab scroll container had no
``scroll-padding-bottom`` reservation, so the bottom 48 px of plots
sat invisibly under the monitor strip.  Users reported "the monitor
panel at the bottom occupied space of other panels, it blocks the
plots in results".

The fix (PR 1) is purely CSS + a one-line JS default-flip:

1. ``:root`` exposes ``--monitor-height`` (collapsed default).
2. ``.results-main`` reserves that height via
   ``scroll-padding-bottom`` AND ``padding-bottom`` so the last
   plot is always reachable.  (The monitor mounts on /results
   only as of commit 88aba08; if it goes global again, add a
   ``body { scroll-padding-bottom: ... }`` rule in
   page-shell.css and a matching pin below.)
3. JS updates ``:root --monitor-height`` on collapse/expand toggle.
4. The strip is ``pointer-events: none`` so clicks pass through to
   the plot legends underneath.  The toggle stays clickable.
5. Default = collapsed.  Saved ``"0"`` opts the user IN to the
   expanded strip; missing key OR ``"1"`` stays collapsed.

Each rule below is one source-text pin.  A future "simplify" PR
that drops any of them re-introduces the bug class.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_STATIC = (Path(__file__).resolve().parent.parent
           / "molbuilder" / "web" / "static")
_LIB = _STATIC / "lib"
_RESULTS = _STATIC / "results"


# --------------------------------------------------------------------- #
#  CSS-side invariants                                                  #
# --------------------------------------------------------------------- #


class TestMonitorHeightVariable:
    """``:root`` MUST define both --monitor-height-collapsed,
    --monitor-height-expanded, AND --monitor-height (alias).  The
    scroll-padding rules depend on the alias resolving."""

    @pytest.fixture(scope="class")
    def css_body(self):
        return (_LIB / "system-load-monitor.css").read_text()

    def test_root_exports_collapsed_height_token(self, css_body):
        assert re.search(
            r":root\s*\{[^}]*--monitor-height-collapsed\s*:",
            css_body, re.DOTALL,
        ), ("system-load-monitor.css :root no longer exposes "
            "--monitor-height-collapsed; scroll-padding-bottom will "
            "fall back to 0 and the layout bug returns.")

    def test_root_exports_expanded_height_token(self, css_body):
        assert re.search(
            r":root\s*\{[^}]*--monitor-height-expanded\s*:",
            css_body, re.DOTALL,
        ), ("system-load-monitor.css :root no longer exposes "
            "--monitor-height-expanded; JS toggle handler has nothing "
            "to switch to and the strip + scroll-padding desync.")

    def test_root_exports_alias(self, css_body):
        """``--monitor-height`` is the consumer-facing alias.
        Scroll containers reference it; JS rewrites it on toggle."""
        assert re.search(
            r":root\s*\{[^}]*--monitor-height\s*:\s*var\(",
            css_body, re.DOTALL,
        ), ("system-load-monitor.css :root no longer aliases "
            "--monitor-height to one of the named-height tokens; "
            "scroll-padding-bottom resolves to 0 and the bug returns.")


class TestScrollContainersReserveSpace:
    """``.results-main`` MUST set both
    ``scroll-padding-bottom: var(--monitor-height)`` (for keyboard/
    scrollIntoView navigation) AND ``padding-bottom`` (for layout
    flow on pages shorter than the viewport) so plot content can
    scroll past the fixed strip."""

    @pytest.fixture(scope="class")
    def results_css_body(self):
        return (_RESULTS / "style.css").read_text()

    def test_results_main_has_scroll_padding(self, results_css_body):
        assert re.search(
            r"\.results-main\s*\{[^}]*scroll-padding-bottom\s*:\s*"
            r"var\(--monitor-height[^)]*\)",
            results_css_body, re.DOTALL,
        ), ("results/style.css no longer sets .results-main's "
            "scroll-padding-bottom; the results scroller's last "
            "plot is unreachable via keyboard or scrollIntoView.")

    def test_results_main_has_padding_bottom(self, results_css_body):
        """Belt-and-braces: padding-bottom (not just scroll-padding)
        because the page can be SHORTER than the viewport, in which
        case scroll-padding does nothing and the bottom of the page
        sits under the monitor.  Padding-bottom value must reference
        --monitor-height directly OR via a shorthand that resolves
        to it (e.g. ``padding: A B var(--monitor-height)``)."""
        # The .results-main rule.
        m = re.search(
            r"\.results-main\s*\{([^}]*)\}",
            results_css_body, re.DOTALL,
        )
        assert m is not None, ".results-main rule missing"
        body = m.group(1)
        # Either explicit padding-bottom referencing --monitor-height,
        # OR a shorthand `padding: A B C` where C references it.
        explicit = re.search(
            r"padding-bottom\s*:\s*var\(--monitor-height", body,
        )
        shorthand = re.search(
            r"padding\s*:\s*[^;]*var\(--monitor-height", body,
        )
        assert explicit or shorthand, (
            "results/style.css no longer reserves --monitor-height "
            "in .results-main padding-bottom; short pages stuff "
            "their last row under the monitor strip.")


class TestPointerEventsContract:
    """Strip and monitor wrapper must be pointer-events: none so
    clicks pass through to plots beneath; toggle stays clickable."""

    @pytest.fixture(scope="class")
    def css_body(self):
        return (_LIB / "system-load-monitor.css").read_text()

    def test_monitor_wrapper_is_pointer_transparent(self, css_body):
        """``#system-load-monitor`` rule must contain
        ``pointer-events: none``."""
        m = re.search(
            r"#system-load-monitor\s*\{([^}]*)\}",
            css_body, re.DOTALL,
        )
        assert m is not None, "monitor wrapper rule missing"
        assert "pointer-events: none" in m.group(1), (
            "system-load-monitor.css #system-load-monitor lost its "
            "``pointer-events: none``; clicks on plot legends in the "
            "bottom 48 px are eaten by the monitor again.")

    def test_strip_is_pointer_transparent(self, css_body):
        """``.system-load-strip`` (the cell container) is also
        pointer-transparent so clicks pass through the wide strip
        area, not just the edges."""
        # The strip rule may appear multiple times; the one with
        # pointer-events is the relevant one.  Grep any one.
        assert re.search(
            r"\.system-load-strip\s*\{[^}]*pointer-events\s*:\s*none",
            css_body, re.DOTALL,
        ), ("system-load-monitor.css .system-load-strip is no longer "
            "pointer-events: none; the strip cells block plot "
            "interaction even though the wrapper passes clicks "
            "through.")

    def test_toggle_stays_clickable(self, css_body):
        """The toggle MUST opt back into pointer-events: auto so the
        user can still close/open the strip."""
        m = re.search(
            r"\.system-load-toggle\s*\{([^}]*)\}",
            css_body, re.DOTALL,
        )
        assert m is not None, "toggle rule missing"
        assert "pointer-events: auto" in m.group(1), (
            "system-load-monitor.css .system-load-toggle lost its "
            "``pointer-events: auto`` opt-in; the wrapper's "
            "pointer-events: none now eats the toggle clicks too "
            "and the user can't collapse / expand the strip.")


# --------------------------------------------------------------------- #
#  JS-side invariants                                                   #
# --------------------------------------------------------------------- #


class TestJsDefaultsCollapsed:
    """Per results-state-contract.md § 9: collapsed BY DEFAULT on
    first visit.  Pre-fix the JS defaulted to ``saved || "0"`` which
    is "expanded if absent" — the wrong direction."""

    @pytest.fixture(scope="class")
    def js_body(self):
        return (_LIB / "system-load-monitor.js").read_text()

    def test_no_legacy_default_expanded_pattern(self, js_body):
        """The old shape was ``saved = sessionStorage.getItem(...) ||
        "0"`` which fell back to expanded.  That pattern is gone."""
        assert not re.search(
            r"sessionStorage\.getItem\([^)]*STORAGE_KEY_COLLAPSED[^)]*\)\s*\|\|\s*\"0\"",
            js_body,
        ), ("system-load-monitor.js still defaults `saved || \"0\"` "
            "to expanded.  Users get the bottom-overlay strip on "
            "every fresh visit — the layout bug returns.")

    def test_default_collapsed_when_key_missing(self, js_body):
        """The fix shape: explicit null-check on getItem, default to
        collapsed (startCollapsed=true) when key is absent."""
        assert re.search(
            r"startCollapsed\s*=\s*\(\s*saved\s*===\s*null\s*\)\s*\?\s*true",
            js_body,
        ), ("system-load-monitor.js no longer defaults to collapsed "
            "when the sessionStorage key is missing.  The strip will "
            "auto-expand on first visit and overlay plot content.")

    def test_monitor_height_var_written_on_toggle(self, js_body):
        """``applyCollapsed`` MUST update :root --monitor-height so
        the CSS scroll-padding follows the strip's actual height."""
        assert "--monitor-height" in js_body, (
            "system-load-monitor.js no longer writes --monitor-height "
            "to :root; CSS scroll-padding stays stuck on the "
            "collapsed value and the strip overlay returns when "
            "expanded.")
        assert re.search(
            r"setProperty\s*\(\s*[\"']--monitor-height[\"']",
            js_body,
        ), ("system-load-monitor.js no longer calls "
            "documentElement.style.setProperty('--monitor-height', "
            "...); the CSS variable doesn't track the strip state.")

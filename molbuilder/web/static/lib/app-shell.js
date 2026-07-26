/* App shell runtime — sets the two layout vars the shell CSS needs.
 * Spec: docs/protocols/ui-design-contract.md §4.1-4.3.
 *
 *   --app-header-h : height of header + tabs nav.  The fixed sidebar rail
 *                    starts BELOW this so the rail sits under the nav, not
 *                    behind it.
 *   --rail-w       : the rail's actual width (sum of its dock panels).  The
 *                    <body> left-padding matches it so content clears the rail
 *                    and header/nav are pulled back to full width by exactly
 *                    the rail width.
 *
 * No-op unless <body> has data-sidebars.  Re-measures on resize + whenever the
 * rail's size changes (a panel fold / width-drag), so the two vars stay true.
 */
(function () {
    "use strict";

    function measure() {
        var root = document.documentElement;
        var header = document.querySelector("body[data-sidebars] > header");
        var nav = document.querySelector("body[data-sidebars] > nav.app-tabs");
        var rail = document.querySelector("body[data-sidebars] .sidebar-rail");

        var h = (header ? header.offsetHeight : 0) + (nav ? nav.offsetHeight : 0);
        root.style.setProperty("--app-header-h", h + "px");

        // Rail width drives the body padding.  Measure the live rail so parallel
        // panels + width-drags stay in sync; fall back to --ps-w before the rail
        // has laid out.
        if (rail) {
            root.style.setProperty("--rail-w", rail.offsetWidth + "px");
        }
    }

    function init() {
        if (!document.body.hasAttribute("data-sidebars")) return;
        measure();
        window.addEventListener("resize", measure);
        // Re-measure when the rail resizes (panel fold / width-drag / a panel
        // mounting) so --rail-w + the body padding follow it.
        var rail = document.querySelector("body[data-sidebars] .sidebar-rail");
        if (rail && window.ResizeObserver) {
            new ResizeObserver(measure).observe(rail);
        }
        // The header height can change after web-fonts / notifications land;
        // one delayed re-measure catches that without polling.
        setTimeout(measure, 300);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();

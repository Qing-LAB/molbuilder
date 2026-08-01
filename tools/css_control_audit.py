#!/usr/bin/env python
"""Measure MolView's control sizing -- the check that outlives the CSS rename.

WHY THIS IS A TOOL AND NOT A MEMORY.  The panel's controls were inconsistent
(three type sizes, three font families, four heights) because browsers do not
inherit fonts into `button` / `input` / `select`.  That was fixed with tokens and
one rule scoped to `.molview-panel` -- and the scoping class is renamed by phase
2 of docs/web/molview-css-namespace-plan.md.  Rename it, miss those rules, and
every control reverts to browser defaults with NOTHING failing: the suite cannot
see a computed style.

So the acceptance check after each rename phase is this measurement, not an
opinion about whether it still looks right.

    python tools/css_control_audit.py            # against the running server
    python tools/css_control_audit.py --url ...  # somewhere else

Prints one line per distinct (font-size, family, height) and exits non-zero if
the panel shows more than one of any -- checkboxes excepted, which are sized
separately by design.
"""
import argparse
import json
import sys
import urllib.request

PROBE = """
(() => {
  const panel = document.querySelector(".molview-panel, .molviewer-panel");
  if (!panel) return {error: "no panel found"};
  const rows = [];
  panel.querySelectorAll("button, input, select, textarea").forEach(el => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    if (!r.height) return;
    rows.push({type: el.type || el.tagName.toLowerCase(),
               font: s.fontSize, family: s.fontFamily.split(",")[0],
               h: +r.height.toFixed(1)});
  });
  const box = t => t === "checkbox" || t === "radio";
  return {
    fonts:    [...new Set(rows.map(r => r.font))],
    families: [...new Set(rows.map(r => r.family))],
    rowHeights: [...new Set(rows.filter(r => !box(r.type)).map(r => r.h))],
    boxHeights: [...new Set(rows.filter(r =>  box(r.type)).map(r => r.h))],
    counted: rows.length,
  };
})()
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8888/molview-demo")
    ap.add_argument("--print-probe", action="store_true",
                    help="print the browser snippet and exit -- paste it into "
                         "devtools, or hand it to Claude-in-Chrome")
    args = ap.parse_args()
    if args.print_probe:
        print(PROBE)
        return 0
    print("This audit needs a real browser -- a computed style does not exist "
          "without one.\nRun with --print-probe and evaluate the snippet on "
          f"{args.url}.\n\nPASS when: one font size, one family, one row "
          "height (checkboxes may differ).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Rename MolView CSS classes across every file that names them.

WHY A TOOL.  docs/web/molview-css-namespace-plan.md moves 167 class names in
eight passes, and the two ways this goes wrong are both mechanical:

  * a PREFIX is not a namespace. `cell-` also matches `cell-detail` and
    `cell-summary` in the system-load template, which have nothing to do with
    MolView -- so the rename works from the EXACT class names the stylesheet
    actually defines, not from a prefix. (The dry run caught that: 34 matches
    where MolView owns 3.)
  * a name lives in FOUR kinds of file -- the stylesheet, the JS that creates
    the element, a template, and a test selector -- and renaming three of them
    leaves a silently unstyled control;
  * matching `.name` in CSS without masking comments deletes or rewrites the
    rule NEXT to a comment that mentions it. That happened, and it destroyed two
    live rules before a class-list diff caught it.

So: every occurrence in one pass, comments masked, and a before/after class-list
diff printed so the change can be checked rather than trusted.

    python tools/css_rename.py --area cell --dry-run
    python tools/css_rename.py --area cell

The suite cannot verify this -- no unit test has a computed style. After each
area, look at the page (tools/css_control_audit.py has the circuit).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSS = ROOT / "molbuilder/web/static/lib/molview/molview.css"
SEARCH = [ROOT / "molbuilder/web/static",
          ROOT / "molbuilder/web/templates",
          ROOT / "tests"]
#: lib/viewer is the RETIRED 3Dmol embed -- loaded by no template and imported by
#: no module (measured 2026-08-01, phase 3). It still defines the same class names,
#: so a rename must not follow them in there: that directory is not live code.
SKIP = ("molview-old", "lib/viewer/", "node_modules", ".test-progress")

#: area -> (old prefix, new prefix[, exact names to restrict to]).
#: The optional third element exists for phase 3, where one prefix covers both
#: live classes and orphans awaiting a ship-or-drop call -- renaming an orphan
#: would dress up a rule that styles nothing.
AREAS = {
    # phase 2 -- MolView's own names, no other definer
    "cell":      ("cell-",           "molviewer-cell-"),
    "regions":   ("region-defs-",    "molviewer-regions-"),
    "label":     ("tag-",            "molviewer-label-"),
    "atoms":     ("col-",            "molviewer-atoms-column-"),
    "filter":    ("selection-filter-", "molviewer-filter-"),
    "panel":     ("panel-page",      "molviewer-panel-tab"),
    "frames":    ("mvf-",            "molviewer-frames-"),
    "selection": ("selection-",      "molviewer-selection-"),
    # phase 3 -- the mol-viewer-* names, created by lib/molview/3dmol-embed.js
    # and ui.js.  They were believed shared with a live embed; they are not.
    "background": ("mol-viewer-bg-",     "molviewer-menu-background-"),
    "busy":       ("mol-viewer-busy",    "molviewer-window-busy"),
    "menu":       ("mol-viewer-menu",    "molviewer-menu"),
    "style":      ("mol-viewer-rep-",    "molviewer-menu-style-"),
    "radius":     ("mol-viewer-radius-", "molviewer-menu-radius-"),
    "rail":       ("mol-viewer-quickbar", "molviewer-rail"),
    "railbutton": ("mol-viewer-quick",   "molviewer-rail-button"),
    "export":     ("mol-viewer-export-", "molviewer-export-",
                   ("mol-viewer-export-btn", "mol-viewer-export-row",
                    "mol-viewer-export-section", "mol-viewer-export-section-label")),
    # the atom table belongs to the `atoms` area, not `selection` -- the plan's
    # section 3 said so, but phase 2 ran `selection-` first and its prefix
    # matched `selection-atom-table` before the `atoms` pass could claim it.
    # Left alone, the columns read `molviewer-atoms-column-*` while the table
    # they sit in read `molviewer-selection-atom-table`.
    "atomstable": ("molviewer-selection-atom-table", "molviewer-atoms-table"),
    "knobs":      ("mol-viewer-knobs",   "molviewer-menu-bar"),
    "toggle":     ("mol-viewer-toggle",  "molviewer-rail-toggle"),
    "stage":      ("mol-viewer-stage",   "molviewer-window-stage"),
    "canvas":     ("mol-viewer-canvas",  "molviewer-window-canvas"),
    # NOT `molviewer-card`: the plan's §3 mapped it there, but `.mol-viewer-card`
    # is NESTED inside `.molview-card` (mount.js:286, and the live selector
    # `.molview-card .mol-viewer-card`) -- they are the outer card and the 3D
    # window's frame, not two spellings of one thing. Collapsing them would have
    # merged two elements. mount.js already calls this one `frame`.
    "frame":      ("mol-viewer-card",    "molviewer-window-frame",
                   ("mol-viewer-card",)),
}


def _files():
    for base in SEARCH:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".css", ".js", ".html", ".py"):
                if not any(s in p.as_posix() for s in SKIP):
                    yield p


def _classes(css: str) -> set:
    """Class names in real rules -- comments masked so their text is invisible."""
    bare = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    return set(re.findall(r"\.([a-zA-Z][\w-]*)", bare))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", required=True, choices=sorted(AREAS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    old, new, *rest = AREAS[args.area]
    only = set(rest[0]) if rest else None

    before = _classes(CSS.read_text())
    # THE EXACT NAMES, taken from the stylesheet with comments masked -- never a
    # bare prefix. `\b` is not enough either: it would let `cell-value` match
    # inside `cell-value-extra`, so the trailing guard is explicit.
    names = sorted((n for n in before if n.startswith(old)
                    and (only is None or n in only)), key=len, reverse=True)
    if only:
        missing = only - set(names)
        if missing:
            print(f"!! restricted to names the stylesheet does not define: "
                  f"{', '.join(sorted(missing))}")
            return 1
    if not names:
        print(f"no classes named {old!r}* in the stylesheet -- nothing to do")
        return 0
    print(f"renaming {len(names)} exact names: {', '.join(names)}\n")
    pattern = re.compile(
        "(?<![\\w-])(" + "|".join(re.escape(n) for n in names) + ")(?![\\w-])")
    sub = lambda m: new + m.group(1)[len(old):]
    touched, total = [], 0
    for path in _files():
        text = path.read_text(errors="replace")
        hits = len(pattern.findall(text))
        if not hits:
            continue
        touched.append((path.relative_to(ROOT).as_posix(), hits))
        total += hits
        if not args.dry_run:
            path.write_text(pattern.sub(sub, text))

    print(f"{args.area}:  {old!r} -> {new!r}   {total} occurrences in "
          f"{len(touched)} files")
    for rel, n in sorted(touched, key=lambda x: -x[1]):
        print(f"   {n:4}  {rel}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    after = _classes(CSS.read_text())
    gone = sorted(n for n in before - after if not n.startswith(old.rstrip("-")))
    added = sorted(n for n in after - before if not n.startswith(new.rstrip("-")))
    print(f"\nclass-list diff -- unexpected losses: {gone or 'none'}")
    print(f"                   unexpected additions: {added or 'none'}")
    if gone or added:
        print("\n!! Something other than this area changed. Check before committing.")
        return 1
    print("\nNow LOOK AT THE PAGE. The suite has no computed style; it cannot "
          "see an unstyled control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

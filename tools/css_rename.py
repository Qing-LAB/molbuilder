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
SEARCH = [ROOT / "molbuilder/web/static/lib/molview",
          ROOT / "molbuilder/web/templates",
          ROOT / "tests"]
SKIP = ("molview-old", "node_modules", ".test-progress")

#: area -> (old prefix, new prefix). One entry per phase-2 pass.
AREAS = {
    "cell":      ("cell-",           "molviewer-cell-"),
    "regions":   ("region-defs-",    "molviewer-regions-"),
    "label":     ("tag-",            "molviewer-label-"),
    "atoms":     ("col-",            "molviewer-atoms-column-"),
    "filter":    ("selection-filter-", "molviewer-filter-"),
    "panel":     ("panel-page",      "molviewer-panel-tab"),
    "frames":    ("mvf-",            "molviewer-frames-"),
    "selection": ("selection-",      "molviewer-selection-"),
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
    old, new = AREAS[args.area]

    before = _classes(CSS.read_text())
    # THE EXACT NAMES, taken from the stylesheet with comments masked -- never a
    # bare prefix. `\b` is not enough either: it would let `cell-value` match
    # inside `cell-value-extra`, so the trailing guard is explicit.
    names = sorted((n for n in before if n.startswith(old)), key=len, reverse=True)
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

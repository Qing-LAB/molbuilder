"""Every class MolView's stylesheet defines is namespaced.

Phase 5 of docs/plans/molview-css-namespace-plan.md, and the reason the other
four phases stay done.

MolView's doctrine is concealment: one entry point, 3Dmol named in exactly one
file, nothing on ``window``.  The stylesheet obeyed none of it -- 167 class
names under 9 prefixes, published into a stylesheet shared with every other
page.  35% of them were also defined somewhere else, and whichever sheet loaded
last won.

A stylesheet rename is invisible to every other kind of test: no DOM stand-in
has a computed style, so the whole suite stays green while the UI is unstyled.
This guard is the one thing that fails.
"""
from __future__ import annotations

import pathlib
import re

import pytest

CSS = (pathlib.Path(__file__).resolve().parent.parent
       / "molbuilder/web/static/lib/molview/molview.css")

PREFIX = "molviewer-"

#: Names a browser defines, which MolView does not get to rename.  Nothing else
#: belongs here: an exemption is how a namespace comes apart one name at a time.
NOT_OURS_TO_RENAME: frozenset = frozenset()


def _defined_classes(css: str) -> set:
    """Class names in real rules.

    Comments are MASKED first.  A sweep that does not strip them counts names
    inside comments that record their own deletion -- four of those turned up
    while clearing phase 1c's orphans, and a matching pass that skipped the
    masking destroyed two live rules before a class-list diff caught it.
    """
    bare = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    return set(re.findall(r"\.([a-zA-Z][\w-]*)", bare))


def test_every_class_molview_defines_is_namespaced():
    stray = sorted(n for n in _defined_classes(CSS.read_text())
                   if not n.startswith(PREFIX) and n not in NOT_OURS_TO_RENAME)
    assert not stray, (
        "molview.css defines class names outside the `molviewer-` namespace:\n  "
        + "\n  ".join(stray)
        + "\n\nEvery class this stylesheet defines belongs to MolView and says so."
        "\nA bare name (`.card`, `.is-active`, `.viewer`) is one another page can"
        "\ndefine too, and then load order decides which one wins."
        "\nSee docs/plans/molview-css-namespace-plan.md."
    )


# RETIRED 2026-08-05 with the thing it watched:
# test_the_namespace_is_not_shared_with_the_retired_embed.
#
# It asserted that MolView defined no class name the retired 3-D embed also
# defined -- a real hazard while both sheets existed, since MolView had carried
# 44 of those names across with it.  The embed's stylesheet went first and the
# test began skipping; the rest of lib/viewer is deleted now (task #19), so there
# is no second definition left for MolView to collide with and nothing for this
# to check.  A test that can only skip is a test that reads as coverage and is
# not.
#
# The one-directional guard that still matters -- a PAGE sheet may not name a
# MODULE's classes -- lives in tests/test_css_module_boundary.py, where
# `mol-viewer-` remains on the forbidden-prefix list so the names cannot return.

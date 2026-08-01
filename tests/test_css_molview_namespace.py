"""Every class MolView's stylesheet defines is namespaced.

Phase 5 of docs/web/molview-css-namespace-plan.md, and the reason the other
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
        "\nSee docs/web/molview-css-namespace-plan.md."
    )


def test_the_namespace_is_not_shared_with_the_retired_embed():
    """lib/viewer is the retired 3Dmol embed: loaded by no page, imported by no
    module.  MolView carried 44 of its class names and 82 duplicate selectors
    with it, styling elements a dead file used to build.

    The directory stays until the VibrationView separation (task #104) removes
    it.  What must not come back is MolView sharing a single name with it.
    """
    embed = CSS.parent.parent / "viewer/mol-viewer-embed.css"
    if not embed.exists():          # deleted by #104 -- nothing left to share
        pytest.skip("lib/viewer/mol-viewer-embed.css is gone, as #104 intends")
    shared = _defined_classes(CSS.read_text()) & _defined_classes(embed.read_text())
    assert not shared, (
        "MolView is defining class names the retired embed also defines:\n  "
        + "\n  ".join(sorted(shared))
        + "\n\nThose elements are built by lib/molview/3dmol-embed.js now."
    )

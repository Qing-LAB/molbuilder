"""Inspectors must register in the order the template lists them.

``lib/inspectors/registry.js`` dispatches by walking its list and taking the
first ``match()`` that answers true, so REGISTRATION ORDER IS THE DISPATCH
RULE: "inspectors with more specific predicates (compound extensions like
``.molwatch.log``) MUST register before more general ones (``.log``)".

The template writes that order down, and it is easy to believe the order you
read is the order that happens.  It is not, if the tags are mixed: a plain
``<script src>`` executes during PARSING, while ``<script type="module">`` and
``<script defer>`` are held back and run afterwards, in their own document
order.  So one module among classic scripts silently moves to the END of the
group -- behind the catch-all viewers that are meant to get the last look.

THAT IS NOT HYPOTHETICAL.  Converting ``inspectors/spectra.js`` to a module on
2026-08-05 moved it from second to fourth, behind ``source.js`` (which matches
``.json``), and ``.spectra.json`` files stopped being recognised as results on
/results.  Every unit test stayed green: each inspector registered correctly,
matched correctly, and mounted correctly.  Only their ORDER was wrong, and
nothing was looking at it.

The rule this asserts is the simple one that makes the template honest: every
inspector script on a page runs in the same queue.  All classic, or all
deferred -- never a mix.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "templates"

#: A <script> tag naming an inspector, with whatever attributes it carries.
TAG = re.compile(r"<script([^>]*?)src=\"[^\"]*?(lib/inspectors/[A-Za-z0-9_.\-]+\.js)[^\"]*\"",
                 re.S)


def _inspector_scripts(html: str):
    """(filename, queue) for every inspector script, in document order."""
    out = []
    for attrs, path in TAG.findall(html):
        deferred = ("type=\"module\"" in attrs) or ("defer" in attrs)
        out.append((path.rsplit("/", 1)[-1], "deferred" if deferred else "parse"))
    return out


def _pages_with_inspectors():
    for p in sorted(TEMPLATES.glob("*.html")):
        html = p.read_text(encoding="utf-8")
        scripts = _inspector_scripts(html)
        # registry.js and the factory are machinery, not inspectors: they must
        # exist BEFORE any inspector runs, which is a different constraint.
        scripts = [s for s in scripts
                   if s[0] not in ("registry.js", "_partial_inspector_factory.js",
                                   # Same kind of thing: the mount/dispose
                                   # helpers both cores share.  It publishes a
                                   # global and registers nothing, so it is
                                   # bound by "must run first", not by "must
                                   # run in template order".
                                   "lifecycle.js")]
        if len(scripts) > 1:
            yield p.name, scripts


@pytest.mark.parametrize("page,scripts",
                         list(_pages_with_inspectors()),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_every_inspector_on_a_page_runs_in_the_same_queue(page, scripts):
    queues = {q for _, q in scripts}
    assert len(queues) == 1, (
        f"{page} mixes script kinds, so the inspectors do NOT register in the "
        f"order the template lists them — the deferred ones all move behind the "
        f"classic ones, and dispatch goes to whichever generic inspector now "
        f"registers first:\n  "
        + "\n  ".join(f"{name:<26} runs at {q}" for name, q in scripts)
        + "\n\nFix: give the classic ones `defer` so the whole group runs in "
          "document order (registry.js § dispatch order)."
    )


def test_the_catch_all_inspectors_are_listed_last():
    """The generic viewers must get the LAST look, which is what being listed
    last means once the queues agree.  Pinned by name because their whole job is
    to match broadly: source matches .json, and .spectra.json is a .json."""
    html = (TEMPLATES / "results.html").read_text(encoding="utf-8")
    order = [name for name, _ in _inspector_scripts(html)]
    for generic in ("markdown.js", "source.js"):
        assert generic in order, f"{generic} is not on /results at all"
    for specific in ("spectra.js", "structure.js", "trajectory.js"):
        assert order.index(specific) < order.index("source.js"), (
            f"{specific} is listed after source.js, which matches broadly "
            f"enough to claim its files first"
        )
    assert order.index("markdown.js") < order.index("source.js")

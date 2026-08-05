"""VibrationView is sealed at ``index.js`` — and this is what makes that true.

``docs/web/vibrationview.md`` § 4 says one entry point and nothing else
importable.  A directory is not a seal, though: everything under
``web/static/lib/`` is served, so ``/static/lib/vibrationview/_maths.js`` is a URL
any script could import.  Convention alone did not hold last time — the module's
predecessor reached OUT to a global for its drawing surface and published itself
back INTO one, and both leaks survived review.

So the boundary is asserted here instead, in one direction: **nothing outside
``lib/vibrationview/`` may name any path inside it except ``index.js``.**  The
module's own files are not read to judge them, because that is the reach this
exists to forbid.

The leading underscore on every internal file is the same statement in a form a
reader sees without running anything: an import of ``_maths.js`` is visibly wrong
at a glance, where an import of ``maths.js`` looks like ordinary code.

STYLESHEET: the module links its own.  No template names it — which is the point.
MolView's ``molview.css`` is linked by six templates, so a page that mounts a
viewer and forgets the ``<link>`` renders it unstyled, and nothing catches that
until someone looks at it.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "molbuilder" / "web"
PACKAGE = WEB / "static" / "lib" / "vibrationview"

#: The one name anything outside the package may write.
PUBLIC = "index.js"

#: Any reference to a file under the package directory.
REF = re.compile(r"lib/vibrationview/([A-Za-z0-9_.\-]+)")

#: Files outside the package that still name an internal path, each with the step
#: that removes it.  Shrink this list; never grow it without recording why.
KNOWN: dict[str, str] = {
    "molbuilder/web/templates/spectra.html": (
        "loads the PREDECESSOR module (vibrationview.js) as a classic <script>. "
        "It cannot mount — the global it reads is published by nothing — and the "
        "page has no #mode-viewer to put it in either.  Removed at step 7 of "
        "task #19, with the module's other dead entries."
    ),
    "molbuilder/web/templates/results.html": (
        "same predecessor <script> tag; this is the page that DOES show a mode "
        "viewer, so it is replaced rather than merely dropped — the page module "
        "imports index.js and hands `mount` to the inspector (§ 11).  Step 6-7 of "
        "task #19."
    ),
    "molbuilder/web/static/lib/spectra/core.js": (
        "a COMMENT at :2851-2853 pointing at the predecessor's internal "
        "mode-math.js, left where the scatter used to live.  Listed rather than "
        "waved through: a comment naming another module's private file is a "
        "reference that rots silently, and this one is already half-wrong (the "
        "global it cites is read by nothing).  It goes when core.js is rewired "
        "onto injection at step 6 of task #19."
    ),
}


def _sources():
    """Every file outside the package that could name one of its paths."""
    for pattern in ("**/*.js", "**/*.html", "**/*.css", "**/*.py"):
        for path in WEB.rglob(pattern):
            if PACKAGE in path.parents or path == PACKAGE:
                continue
            if "vendor" in path.parts:
                continue
            yield path


def test_nothing_outside_the_package_names_an_internal_file():
    """§ 4: "a consumer that imports any of them directly has broken the module,
    not found a shortcut"."""
    violations: list[str] = []
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in set(REF.findall(text)):
            if name == PUBLIC:
                continue
            if rel in KNOWN:
                continue
            violations.append(f"{rel} -> lib/vibrationview/{name}")
    assert not violations, (
        "these reach past the module's one door (docs/web/vibrationview.md § 4); "
        "import index.js, or have whoever mounts the viewer hand `mount` on:\n  "
        + "\n  ".join(sorted(violations))
    )


def test_every_internal_file_is_marked_as_internal():
    """A reader should not have to consult a document to tell the door from the
    rooms.  Every file in the package except the entry point is underscore-
    prefixed, so a wrong import looks wrong where it is written."""
    if not PACKAGE.is_dir():
        return
    unmarked = sorted(
        p.name for p in PACKAGE.iterdir()
        if p.is_file() and p.name != PUBLIC and not p.name.startswith("_")
        # The predecessor module, deleted at step 7 of task #19.
        and p.name not in {"vibrationview.js", "mode-math.js"}
    )
    assert not unmarked, (
        "internal files must be underscore-prefixed so an import of one reads as "
        "a violation at a glance: " + ", ".join(unmarked)
    )


#: Every ``import ... from "<specifier>"`` in a JS file.
IMPORT = re.compile(r"""\bfrom\s+["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']""")


def test_the_package_imports_nothing_outside_itself():
    """§ 4: "it reaches nothing else in the app by name".

    The seal is only half the claim.  A module that nothing can reach into, but
    which reaches out to three other modules, is not self-contained — it is
    coupled in the direction that is harder to see.  Every specifier here must be
    relative and stay inside the package.

    The predecessor failed exactly this way: it imported the shared XYZ writer
    from ``../xyz-io.js`` and read its drawing surface off a global.  The rewrite
    builds its own XYZ block, which is a dozen lines, rather than owing another
    module for them.
    """
    if not PACKAGE.is_dir():
        return
    escapes: list[str] = []
    for path in sorted(PACKAGE.glob("*.js")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # The predecessor, deleted at step 7 of task #19.
        if path.name in {"vibrationview.js", "mode-math.js"}:
            continue
        for m in IMPORT.finditer(text):
            spec = m.group(1) or m.group(2)
            if not spec.startswith("."):
                escapes.append(f"{path.name} -> {spec}  (not relative)")
                continue
            target = (path.parent / spec).resolve()
            if PACKAGE.resolve() not in target.parents:
                escapes.append(f"{path.name} -> {spec}  (leaves the package)")
    assert not escapes, (
        "the package must reach nothing outside itself by name (§ 4):\n  "
        + "\n  ".join(escapes)
    )


def test_only_the_sealed_layer_names_the_drawing_library():
    """§ 4: "the name 3Dmol occurs in exactly one file — the sealed layer — which
    is also the only place that fails with a clear error if the library is
    missing".  That is what makes "the graphics library is invisible" a property
    of the code rather than a habit."""
    if not PACKAGE.is_dir():
        return
    namers = sorted(
        p.name for p in PACKAGE.glob("*.js")
        if p.name not in {"vibrationview.js", "mode-math.js"}
        and re.search(r"3Dmol", p.read_text(encoding="utf-8", errors="ignore"))
    )
    assert namers == ["_seal.js"], (
        "exactly one file may name the drawing library; these do: "
        + ", ".join(namers)
    )


def test_no_template_links_the_module_stylesheet():
    """The module links its own sheet (§ 13).  A page that has to remember a
    <link> is a page that can forget one, and the module then mounts unstyled with
    nothing to catch it — which is the live defect in MolView's arrangement."""
    offenders = []
    for path in (WEB / "templates").rglob("*.html"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "lib/vibrationview/" in text and "stylesheet" in text:
            for line in text.splitlines():
                if "lib/vibrationview/" in line and "stylesheet" in line:
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}: {line.strip()}")
    assert not offenders, (
        "the module owns its stylesheet and injects its own link; no template "
        "should name it:\n  " + "\n  ".join(offenders)
    )

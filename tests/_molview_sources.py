"""What "the module" IS, computed rather than listed.

Several tests check a rule across MolView's source — that the drawing library is
named in one file, that the app's global namespace is never typed, that no file
handling is hand-rolled, that only the contract's routes are called. Every one of
them needs the same answer first: *which files are the module?*

**The directory is not the answer.** `lib/molview/` also holds the demo page,
which is a CONSUMER — it imports the entry point rather than being imported by
it — and the stylesheet, which is not code. Holding a consumer to the module's
internal rules is holding the wrong side of the boundary to them, and skipping it
by name is a list, which § 13.1 rules out: "a pinned list of names is a
transcription, not a contract."

**The import graph is the answer.** § 4 defines the module as what `index.js`
exposes: "every other file in the module is internal — a consumer that imports
any of them directly has broken the module, not found a shortcut." So the module
is exactly what is reachable from the entry point, and anything else in the
directory is either a consumer or not code.

That definition is structural, so it needs no maintenance: a new layer wired in
is found automatically, and a new consumer is excluded automatically — and a file
that is *supposed* to be a layer but was never wired up shows as absent rather
than passing silently.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "molbuilder" / "web" / "static" / "lib" / "molview"
ENTRY = MODULE_DIR / "index.js"

_IMPORT = re.compile(r"""(?:from|import)\s+["']([^"']+)["']""")


def _walk() -> tuple[dict[str, Path], dict[str, Path]]:
    """Walk the import graph from the entry point.

    Returns ``(layers, dependencies)`` -- reachable files INSIDE the module
    directory, and reachable files outside it.
    """
    layers: dict[str, Path] = {}
    deps: dict[str, Path] = {}
    seen: set[Path] = set()
    pending = [ENTRY]
    while pending:
        path = pending.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        inside = MODULE_DIR in path.parents
        # `setdefault`, so a name reached twice keeps its FIRST answer.  The
        # dict has always been keyed by filename; keying `seen` by full path
        # (which is what lets a dependency be recorded without being walked)
        # would otherwise let a later visit overwrite an earlier one.
        (layers if inside else deps).setdefault(path.name, path)
        # A DEPENDENCY IS NOT WALKED THROUGH.  What it imports is its own
        # module's business; following it would drag a third module's files
        # into MolView's answer and hold them to MolView's rules.
        if not inside:
            continue
        for spec in _IMPORT.findall(path.read_text()):
            if not spec.startswith("."):
                continue          # a bare or absolute specifier is not a layer
            target = (path.parent / spec).resolve()
            if target.suffix == ".js":
                pending.append(target)
    return layers, deps


def module_files() -> dict[str, Path]:
    """The module's LAYERS: files reachable from the entry point and living
    inside the module directory.

    **A reachable file outside the directory is a DEPENDENCY, not a layer**
    (added 2026-09-11, when `ui.js` began importing
    `lib/validation-findings.js` -- the one renderer for a finding, which
    MolView reuses rather than copying).  The rules these tests enforce are
    MolView's own: which drawing library is named, whose stylesheet defines a
    class, what may touch the app namespace.  A module MolView depends on
    answers to its OWN owner for all of those, and holding it to MolView's is
    the same category error as holding the demo page to them.

    See :func:`module_dependencies` for the other half.
    """
    return _walk()[0]


def module_dependencies() -> dict[str, Path]:
    """The modules MolView reaches DOWN to, by name — DIRECT ones only.

    A dependency is recorded and not walked through, so what *it* imports is
    absent here: that is its own module's business, and following it would
    hold a third module's files to MolView's rules. The cost is that a
    transitive dependency does not show up; the alternative was an answer
    that stops being about MolView at all.

    Kept visible on purpose: a dependency is a design decision, and one that
    arrives unnoticed is how a module stops being separable. Depending
    downward on a shared presentation module is reuse; a sideways reach into
    another viewer would not be, and this is what lets a test tell them
    apart.
    """
    return _walk()[1]


def module_code() -> dict[str, str]:
    """The module's layers with comments stripped.

    A rule about what the code DOES must not fire on prose explaining what was
    deleted and why — several headers describe the very things being banned.
    """
    out = {}
    for name, path in module_files().items():
        out[name] = "\n".join(
            line for line in path.read_text().splitlines()
            if not line.lstrip().startswith(("*", "//", "/*"))
        )
    return out


def consumers() -> dict[str, Path]:
    """Files in the directory that are NOT layers — the demo page today.

    They are held to § 4's single-import rule instead of to the module's internal
    ones, which is the opposite side of the same boundary.
    """
    layers = module_files()
    return {
        p.name: p for p in sorted(MODULE_DIR.glob("*.js"))
        if p.name not in layers
    }

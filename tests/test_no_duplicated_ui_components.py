"""Two UI functions with the same shape are one function written twice.

Why this file exists
====================

Three "full-text reviews" ran before this test did, and each swept for a
defect class already chosen -- rendered geometry, CSS values, CSS classes.
None of them looked for DUPLICATED LOGIC, so every duplicate surfaced one at a
time, as a bug report, and each looked unrelated to the last.  They were not
unrelated:

  * `configure()` and `_lazyResolve()` were byte-identical in five builder
    panels, comment and all -- eighty lines, five places to edit, five places
    for a fix to miss;
  * `_autoAnalyzeOnLoad()` was duplicated across two tabs *by an extraction
    that took the renderer and the protocol and left the caller*;
  * three memoised fetches in one file each spelled out their own `await
    fetch`, and none of the three caught anything -- which is how a restarting
    server made the bench card vanish with nothing said.

The last one is the argument for this test.  The duplication was not a tidiness
problem; it was the reason a fix applied to one copy did not reach the others.

How it works
============

Function bodies are normalised -- comments, whitespace, string CONTENTS and
identifier NAMES all removed -- so two functions that differ only in what they
are called and which URL they hit still collide.  That is deliberate: those are
exactly the differences a copy-paste leaves behind.
"""
from __future__ import annotations

import collections
import hashlib
import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "molbuilder/web/static"

#: Below this a shared shape is a coincidence, not a component.
_MIN_LINES = 5

#: Clusters that are parallel BY DESIGN, each with the reason it is not a
#: duplicate.  Adding to this is a decision; the list is meant to stay short.
_ALLOWED = {
    "rest-verb-wrappers":
        "lib/projects/api.js and state.js expose one thin function per REST "
        "verb (mkdir/rename/move/copy/list/stat).  They are parallel because "
        "the API is; folding them would hide which verbs exist behind an "
        "argument, which is the opposite of a readable door.",
    "runtime-registry-lists":
        "molbuilder-runtime.js listRegistered/listPending -- two one-line "
        "reads of two different maps.",
}

#: Clusters where the shape is shared but folding them would be WORSE, each
#: with the argument.  The test for a real duplicate is not "these look alike"
#: -- it is "a change to one of them should have reached the others".
_ALLOWED_FUNCS = {
    # Two callers of ONE shared API (`lib/auto-detect.js`).  What is left in
    # each is the binding: this page's sequence counter and this page's
    # wording.  Folding further would mean the module knowing which tab it is
    # serving, which is the thing it was extracted to stop.
    ("spectra/viewer.js", "_autoAnalyzeOnLoad"),
    ("structure-optimization/viewer.js", "_autoAnalyzeOnLoad"),
    # A CRC-32 is a fixed algorithm with a fixed polynomial: there is no
    # future change to one copy that should have reached the other.  Both
    # live in modules that deliberately carry no imports -- the vibration
    # seal (vibrationview.md § 15) and the zip store.
    ("lib/vibrationview/_export.js", "crc32"),
    ("lib/zip-store.js", "crc32"),
    # Same argument: a six-line read of a CSS custom property, inside two
    # SEALED embeddables whose whole point is having no dependencies.
    ("lib/molview/3dmol-embed.js", "readCssVar"),
    ("lib/vibrationview/_seal.js", "readCssVar"),
    # Coincidence, not a component: one reports a persistence failure, the
    # other forwards a panel change.  Same three lines, unrelated domains.
    ("lib/workspace/dispatcher.js", "onPersistError"),
    ("modify/structure/page.js", "onPanelChange"),
    ("lib/projects/api.js", "apiMove"), ("lib/projects/api.js", "apiCopy"),
    ("lib/projects/api.js", "apiMkdir"), ("lib/projects/api.js", "apiRename"),
    ("lib/projects/api.js", "apiList"), ("lib/projects/api.js", "apiStat"),
    ("lib/projects/state.js", "deleteEntry"), ("lib/projects/state.js", "rename"),
    ("lib/projects/state.js", "mkdir"), ("lib/projects/state.js", "copy"),
    ("lib/molbuilder-runtime.js", "listRegistered"),
    ("lib/molbuilder-runtime.js", "listPending"),
}

_FUNC = re.compile(r'(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{')


def _body(text, brace_idx):
    depth, i = 0, brace_idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx:i + 1]
        i += 1
    return ""


def _normalise(src):
    s = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    s = re.sub(r'//[^\n]*', '', s)
    s = re.sub(r'"[^"]*"|\'[^\']*\'|`[^`]*`', '"S"', s)
    s = re.sub(r'\b[A-Za-z_$][\w$]*\b', 'N', s)
    return re.sub(r'\s+', '', s)


def _clusters():
    seen = collections.defaultdict(list)
    for js in sorted(_STATIC.rglob("*.js")):
        if "vendor" in js.parts:
            continue
        rel = js.relative_to(_STATIC).as_posix()
        text = js.read_text(encoding="utf-8")
        for m in _FUNC.finditer(text):
            body = _body(text, m.end() - 1)
            if body.count("\n") < _MIN_LINES:
                continue
            key = hashlib.sha1(_normalise(body).encode()).hexdigest()
            seen[key].append((rel, m.group(1), text[:m.start()].count("\n") + 1))
    return {k: v for k, v in seen.items() if len(v) > 1}


def test_no_function_is_written_twice():
    offenders = []
    for members in _clusters().values():
        if all((f, n) in _ALLOWED_FUNCS for f, n, _ in members):
            continue
        offenders.append(members)
    assert not offenders, (
        "these functions have the same shape in more than one place — one "
        "function written twice is two places for a fix to miss:\n"
        + "\n".join(
            "  x%d  %s" % (len(m), ", ".join(f"{f}:{ln} {n}()" for f, n, ln in m))
            for m in offenders))


def test_every_allowance_states_why():
    """An allowlist without reasons is a list of things nobody re-examined."""
    for name, why in _ALLOWED.items():
        assert len(why) > 40, f"{name} is allowed but does not say why"

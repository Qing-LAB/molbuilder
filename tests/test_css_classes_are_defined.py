"""A class the JS writes is a class some stylesheet answers.

`form-components.css` has a comment on one of its rules that names this defect
exactly: a finding's message was *"written by every renderer and styled by
NONE"*.  It was not the only one.  On 2026-08-23 a sweep found 21 classes that
JavaScript assigns and no stylesheet defines, of which ten were the two modal
dialogs -- so every unsaved-changes prompt and every Save-as rendered in the
browser's own chrome: a white box with default buttons, in an app that has no
light theme at all.

Two guards:

  * every literal class name the JS writes is defined somewhere;
  * every ``<dialog>`` wears the shared component, because a dialog that
    misses it is exactly the failure above and is invisible until someone
    opens it.

Prefix fragments (``"phase-" + kind``) are not class names and are listed by
name below -- an allowlist that has to be edited is the point: adding to it is
a decision, not an accident.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_STATIC = _ROOT / "molbuilder/web/static"

#: Written as ``"<prefix>" + something`` -- a fragment, never a whole name.
_PREFIX_FRAGMENTS = {
    "app-notification--",   # + severity, app-notifications.js
    "phase-",               # + phase name, transport/core.js
}

#: MARKERS -- classes that carry meaning to code, not appearance to a reader.
#: Each needs a reason, because "it has no rule" is exactly what was wrong
#: with the ten modal classes: the difference between a marker and a
#: half-built widget is whether anybody meant it.
_MARKERS = {
    "is-clean":
        "the modify canvas has no unsaved edits; read by the dirty-gate, "
        "never drawn",
    "is-view-only":
        "the preview editor is read-only; queried by preview.js to decide "
        "whether to arm the editor, and CodeMirror owns the look",
    "error-card":
        "an inspector failed to mount; the card it is set on is already "
        "`.card`, and 15 tests name this to find the failure",
    "molbuilder-warning-modal":
        "the discard dialog, for tests to find; `.mb-dialog` dresses it",
    "molbuilder-save-name-modal":
        "the save-as dialog, same",
    "molbuilder-save-overwrite-modal":
        "the overwrite dialog, same",
    "molbuilder-projects-dest-root":
        "the root <ul> of the destination tree; the tree's own rules select "
        "through it rather than on it",
}
_ALLOWED_UNSTYLED = set(_MARKERS)


def _js_written_classes():
    """``{class: first file that writes it}`` for literal assignments."""
    out = {}
    for js in sorted(_STATIC.rglob("*.js")):
        if "vendor" in js.parts:
            continue
        text = js.read_text(encoding="utf-8")
        for m in re.finditer(
                r'className\s*=\s*"([^"{}]+)"|classList\.add\(\s*"([^"{}]+)"',
                text):
            for name in (m.group(1) or m.group(2) or "").split():
                if name and name not in _PREFIX_FRAGMENTS:
                    out.setdefault(name, js.relative_to(_STATIC).as_posix())
    return out


def _css_defined_classes():
    joined = "\n".join(
        p.read_text(encoding="utf-8")
        for p in _STATIC.rglob("*.css") if "vendor" not in p.parts)
    return set(re.findall(r'\.([a-zA-Z][\w-]*)', joined))


def test_every_class_the_js_writes_is_defined_somewhere():
    written = _js_written_classes()
    defined = _css_defined_classes()
    missing = {c: f for c, f in written.items()
               if c not in defined and c not in _ALLOWED_UNSTYLED}
    assert not missing, (
        "JavaScript writes these classes and no stylesheet answers them, so "
        "the browser paints its own chrome:\n"
        + "\n".join(f"    .{c:42} written by {f}"
                    for c, f in sorted(missing.items())))


def test_every_dialog_wears_the_shared_component():
    """A `<dialog>` without `.mb-dialog` is a white box in a dark app.

    There were three independent scaffolds (`web/audit-2026-08-05-tab-ui.md`
    § C8) and only one was styled at all.  This is checked at the point of
    construction because a modal is invisible until someone opens it -- the
    kind of surface a person meets on a bad day, mid-save.
    """
    offenders = []
    for js in sorted(_STATIC.rglob("*.js")):
        if "vendor" in js.parts:
            continue
        text = js.read_text(encoding="utf-8")
        for m in re.finditer(r'createElement\(\s*["\']dialog["\']\s*\)', text):
            after = text[m.end():m.end() + 400]
            if "mb-dialog" not in after and "molbuilder-projects-dialog" \
                    not in after:
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{js.relative_to(_STATIC).as_posix()}:{line}")
    assert not offenders, (
        "these <dialog>s are built without the shared component "
        "(`lib/dialog.css`, class `mb-dialog`): " + ", ".join(offenders))


def test_the_pages_that_can_open_a_dialog_load_the_sheet():
    """A component nobody links is a class nobody answers — the same defect
    one indirection out."""
    templates = _ROOT / "molbuilder/web/templates"
    missing = []
    for page in sorted(templates.glob("*.html")):
        if page.name.startswith("_"):
            continue                       # partials inherit their host's
        html = page.read_text(encoding="utf-8")
        if "page-shell.css" in html and "lib/dialog.css" not in html:
            missing.append(page.name)
    assert not missing, (
        f"these pages load the shell but not the dialog component: {missing}")


def test_every_marker_states_why_it_has_no_rule():
    """The allowlist is a set of decisions, not a set of names.

    A marker without a reason is indistinguishable from a class somebody
    forgot to style -- which is the whole defect this file exists for.
    """
    for name, reason in _MARKERS.items():
        assert reason and len(reason) > 20, (
            f".{name} is allowed to have no rule but does not say why")

"""CSS ``[hidden]``-attribute precedence audit — source-text invariant.

Pins the rule documented in ``docs/process/code-audit.md``:
when a class/id selector sets ``display: <not none>``, the rule ties
on specificity with the UA stylesheet's ``[hidden] { display: none }``
and **wins by source order**.  A JS ``element.hidden = true`` then
leaves the element visually present in the layout — the user sees a
clickable "ghost" control that does nothing on click.

The audit's question is: which CSS rules are at risk of this bug?
Answer: rules whose selector matches an element that JS actually
toggles via ``.hidden`` at runtime.  A rule on a purely-static
layout container (``.tab-grid``, ``.card-row``) is irrelevant — JS
never sets ``hidden`` on it.

This test takes the cross-reference approach:

  1. Build the set of element IDs that JS toggles via ``.hidden = ...``
     anywhere in ``lib/`` / ``modify/`` / ``spectra/`` / ``results/`` /
     ``transport/``.  Pattern: ``getElementById("X")`` or ``$("X")``
     within the same JS expression chain as ``.hidden = …``.
  2. For each such ID, find every CSS rule whose selector list
     contains ``#ID`` or matches an element with that ID via a
     descendant/ancestor selector.
  3. Each such rule that sets a non-``none`` ``display`` value MUST
     have a matching ``[hidden]`` guard, either on the same selector
     or on a parent that JS hides instead.

Precedents (each fixed in its own commit, audit was scoped narrowly
and missed siblings):
  * 2026-05-28 — ``.issues-panel`` (``style.css:357``)
  * 2026-06-14 — ``.ctab-panel label.check`` (``trajectory-inspector.css``,
                 commit ``e7651cb``)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.module

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "molbuilder/web/static"


# --------------------------------------------------------------------- #
#  Step 1 — collect IDs that JS toggles via .hidden                      #
# --------------------------------------------------------------------- #


# Match a hidden-assignment on an expression that includes an id
# literal.  Three common shapes covered:
#
#   document.getElementById("foo").hidden = true
#   $("foo").hidden = false
#   $("foo-row").hidden = !something
#
# Variable-tracked patterns (``const row = $("foo"); row.hidden = ...``)
# are picked up by a second pass that looks for the
# ``var = $("X")`` pairing within a small window before a
# ``var.hidden = …`` assignment.
_DIRECT_HIDDEN_ASSIGNMENT_RE = re.compile(
    r"""
    (?: getElementById | \$ | _\$ )
    \(\s* ["']([a-zA-Z0-9_\-]+)["'] \s*\)
    (?: \??\.[a-zA-Z_]\w* )*       # optional ?.foo.bar chain
    \.hidden \s* =
    """,
    re.VERBOSE,
)
_VAR_GET_RE = re.compile(
    r"""
    \b (?: const | let | var ) \s+ ([a-zA-Z_]\w*) \s* =
    \s* (?: getElementById | \$ | _\$ )
    \(\s* ["']([a-zA-Z0-9_\-]+)["'] \s*\)
    """,
    re.VERBOSE,
)
_VAR_HIDDEN_ASSIGNMENT_RE = re.compile(
    r"\b([a-zA-Z_]\w*)\.hidden\s*=",
)

# 2026-08-05: the pattern above wants a DECLARATION (``const row = $("x")``),
# and the tab controllers do not use one.  They resolve every element once into
# a table and reach it by property for the rest of the file:
#
#     els.animTemperatureRow = $("anim-temperature-row");
#     ...
#     els.animTemperatureRow.hidden = mode !== "thermal";
#
# ``spectra/core.js`` alone binds about forty elements that way, and NONE of
# them were visible to this audit -- so `.viewer-controls .inline-label`
# shipped with `display: inline-flex` and no guard, and the Temperature control
# sat in the layout under an amplitude mode that has no temperature.  Found by
# looking at the page, which is the failure mode this whole file exists to make
# unnecessary.
_PROP_GET_RE = re.compile(
    r"""
    \b [a-zA-Z_]\w* (?: \.[a-zA-Z_]\w* )* \. ([a-zA-Z_]\w*)   # els.someName
    \s* = \s*
    (?: getElementById | \$ | _\$ )
    \(\s* ["']([a-zA-Z0-9_\-]+)["'] \s*\)
    """,
    re.VERBOSE,
)

# 2026-08-20: a FOURTH idiom -- module-level ``let``s declared in a batch and
# bound later by BARE assignment inside an attach step:
#
#     let elPanel, elToggle, elActions, ...;
#     function _attach() {
#         elActions = document.getElementById("ps-checkpoint-actions");
#     }
#
# ``checkpoint.js`` binds its whole element table this way, and none of the
# three patterns above match it (no declaration keyword, no property chain) --
# so the audit was blind to the entire panel, and ``.ps-checkpoint-actions``
# shipped unguarded THROUGH this test (caught by the 2026-08-20 milestone
# review, not by this file).  The optional ``document.`` prefix also covers
# ``const x = document.getElementById(...)``, which _VAR_GET_RE never matched.
_BARE_ASSIGN_GET_RE = re.compile(
    r"""
    (?<! [.\w] ) ([a-zA-Z_]\w*) \s* = \s*
    (?: document \s* \. \s* )?
    (?: getElementById | \$ | _\$ )
    \(\s* ["']([a-zA-Z0-9_\-]+)["'] \s*\)
    """,
    re.VERBOSE,
)



def _scan_js_for_hidden_ids() -> set[str]:
    """Return the set of HTML element ids that some JS file assigns
    ``.hidden`` on (either ``true`` or ``false``).  These are the
    "dynamically hideable" elements the CSS rules must guard."""
    ids: set[str] = set()
    for js_path in STATIC.rglob("*.js"):
        rel = js_path.relative_to(STATIC).as_posix()
        if rel.startswith("vendor/") or rel.endswith(".min.js"):
            continue
        src = js_path.read_text(encoding="utf-8")
        # Direct: $("foo").hidden = ...
        for m in _DIRECT_HIDDEN_ASSIGNMENT_RE.finditer(src):
            ids.add(m.group(1))
        # Variable-tracked: const row = $("foo"); row.hidden = ...
        # Build a small lookup of var-name → id within a 5-line
        # window so cross-scope collisions are unlikely.
        var_to_id: dict[str, str] = {}
        for m in _VAR_GET_RE.finditer(src):
            var_to_id[m.group(1)] = m.group(2)
        # ...and the property-table form the tab controllers actually use.
        for m in _PROP_GET_RE.finditer(src):
            var_to_id.setdefault(m.group(1), m.group(2))
        for m in _BARE_ASSIGN_GET_RE.finditer(src):
            var_to_id.setdefault(m.group(1), m.group(2))
        for m in _VAR_HIDDEN_ASSIGNMENT_RE.finditer(src):
            name = m.group(1)
            if name in var_to_id:
                ids.add(var_to_id[name])
    return ids


# --------------------------------------------------------------------- #
#  Step 1b — the same question, for elements that have no id             #
# --------------------------------------------------------------------- #
#
# 2026-08-03: MolView builds its panel with `el("div", "molviewer-…")` and
# hides pieces of it by variable — there is no id anywhere, so the id-keyed
# scan above could not see it, and `.molviewer-selection-assign { display:
# flex }` shipped with no guard.  On every read-only viewer the label box was
# hidden by the module and drawn by the stylesheet: a chooser and three buttons
# that did nothing, on four pages.
#
# The shape it looks for is the one that module uses:
#
#     const assign = el("div", "molviewer-selection-assign");
#     ...
#     assign.hidden = model.mode === "readonly";

_VAR_CLASS_RE = re.compile(
    r"""(?:const|let|var)\s+(\w+)\s*=\s*el\(\s*["'][\w-]+["']\s*,\s*["']([\w\s-]+)["']""",
)


def _scan_js_for_hidden_classes() -> set[str]:
    """Class names on elements some JS assigns ``.hidden`` on."""
    names: set[str] = set()
    for js_path in STATIC.rglob("*.js"):
        rel = js_path.relative_to(STATIC).as_posix()
        if rel.startswith("vendor/") or rel.endswith(".min.js"):
            continue
        src = js_path.read_text(encoding="utf-8")
        var_to_classes: dict[str, list[str]] = {}
        for m in _VAR_CLASS_RE.finditer(src):
            var_to_classes[m.group(1)] = m.group(2).split()
        for m in _VAR_HIDDEN_ASSIGNMENT_RE.finditer(src):
            for cls in var_to_classes.get(m.group(1), ()):
                names.add(cls)
    return names


# --------------------------------------------------------------------- #
#  Step 2 — collect CSS rules + their guards                             #
# --------------------------------------------------------------------- #


_NON_NONE_DISPLAY_RE = re.compile(
    r"\bdisplay\s*:\s*"
    r"(flex|grid|block|inline|inline-block|inline-flex|"
    r"inline-grid|table|table-cell|table-row|list-item|"
    r"contents)\b",
)
_DISPLAY_NONE_RE = re.compile(
    r"\bdisplay\s*:\s*none\b(\s*!important)?",
)


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _split_rules(css: str) -> list[tuple[str, str]]:
    """Yield ``(selector_list, properties)`` pairs.  Descends into
    ``@media`` / ``@supports`` blocks so their nested rules are
    audited too."""
    css = _strip_comments(css)
    rules: list[tuple[str, str]] = []
    i, n = 0, len(css)
    while i < n:
        while i < n and css[i].isspace():
            i += 1
        if i >= n:
            break
        if css[i] == "@":
            j = i
            while j < n and css[j] not in "{;":
                j += 1
            if j >= n:
                break
            if css[j] == ";":
                i = j + 1
                continue
            depth, k = 1, j + 1
            while k < n and depth > 0:
                if css[k] == "{":
                    depth += 1
                elif css[k] == "}":
                    depth -= 1
                k += 1
            rules.extend(_split_rules(css[j + 1:k - 1]))
            i = k
            continue
        j = i
        while j < n and css[j] not in "{}":
            j += 1
        if j >= n or css[j] == "}":
            i = j + 1
            continue
        selector_list = css[i:j].strip()
        depth, k = 1, j + 1
        while k < n and depth > 0:
            if css[k] == "{":
                depth += 1
            elif css[k] == "}":
                depth -= 1
            k += 1
        if selector_list:
            rules.append((selector_list, css[j + 1:k - 1]))
        i = k
    return rules


def _selectors_in(selector_list: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(selector_list):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            s = selector_list[start:i].strip()
            if s:
                out.append(s)
            start = i + 1
    tail = selector_list[start:].strip()
    if tail:
        out.append(tail)
    return out


# --------------------------------------------------------------------- #
#  Step 3 — selectors that "could match" an id                           #
# --------------------------------------------------------------------- #


_ID_IN_SELECTOR_RE = re.compile(r"#([a-zA-Z0-9_\-]+)")


def _ids_targeted_by(selector: str) -> set[str]:
    """Ids that this CSS selector explicitly targets (via ``#id``).

    Selectors that don't name an id directly (pure class selectors
    like ``.ctab-panel label.check``) match by class.  Those can
    still be triggered by JS hiding an id-named ancestor whose
    class matches.  We pick those up via a separate "guarded
    ancestor" check in the test body.
    """
    return set(_ID_IN_SELECTOR_RE.findall(selector))


def _all_css_files() -> list[Path]:
    out: list[Path] = []
    for path in STATIC.rglob("*.css"):
        rel = path.relative_to(STATIC).as_posix()
        if rel.startswith("vendor/") or rel.endswith(".min.css"):
            continue
        out.append(path)
    return out


# --------------------------------------------------------------------- #
#  The actual audit                                                       #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def dynamically_hidden_ids() -> set[str]:
    """Ids that some JS file dynamically sets ``.hidden`` on."""
    return _scan_js_for_hidden_ids()


@pytest.fixture(scope="module")
def all_hidden_guards() -> set[str]:
    """Every ``selector[hidden]`` declaration that resolves to
    ``display: none`` across the project."""
    guards: set[str] = set()
    for path in _all_css_files():
        for sel_list, props in _split_rules(path.read_text()):
            if not _DISPLAY_NONE_RE.search(props):
                continue
            for sel in _selectors_in(sel_list):
                if sel.endswith("[hidden]"):
                    guards.add(sel)
    return guards


@pytest.fixture(scope="module")
def display_rules_by_id() -> dict[str, list[tuple[Path, str, str]]]:
    """Mapping from an html id → list of
    ``(css_file, selector, display_value)`` for every CSS rule
    that names that id AND sets a non-``none`` display."""
    out: dict[str, list[tuple[Path, str, str]]] = {}
    for path in _all_css_files():
        for sel_list, props in _split_rules(path.read_text()):
            m = _NON_NONE_DISPLAY_RE.search(props)
            if not m:
                continue
            display_val = m.group(1)
            for sel in _selectors_in(sel_list):
                if sel.endswith("[hidden]"):
                    continue
                for hit_id in _ids_targeted_by(sel):
                    out.setdefault(hit_id, []).append(
                        (path, sel, display_val))
    return out


def test_every_dynamically_hidden_id_has_a_guard(
    dynamically_hidden_ids, display_rules_by_id, all_hidden_guards,
):
    """For every element id that JS sets ``hidden = …`` on, any
    CSS rule that names that id (``#id`` or ``#id .foo``) and sets
    a non-``none`` display MUST have a matching ``[hidden]`` guard
    on the same selector.

    Per ``docs/process/code-audit.md`` — failure mode is
    the "ghost toggle" pattern the 2026-06-14 trajectory-inspector
    incident caught.  Fix by adding ``selector[hidden] {display:none;}``
    next to the offending rule in the same file.
    """
    failures: list[str] = []
    for el_id in sorted(dynamically_hidden_ids):
        rules = display_rules_by_id.get(el_id)
        if not rules:
            continue
        for css_path, selector, display_val in rules:
            if f"{selector}[hidden]" in all_hidden_guards:
                continue
            failures.append(
                f"  id #{el_id} is set by JS via .hidden, and CSS "
                f"rule `{selector} {{ display: {display_val} }}` "
                f"({css_path.relative_to(STATIC).as_posix()}) "
                f"would override that.  Add: "
                f"`{selector}[hidden] {{ display: none; }}`"
            )
    if failures:
        pytest.fail(
            f"CSS [hidden]-precedence violations "
            f"({len(failures)}; see code-audit.md § 3.1):\n"
            + "\n".join(failures)
        )


def test_every_dynamically_hidden_class_has_a_guard(all_hidden_guards):
    """The same rule, for elements that carry only a class.

    A class selector setting a non-``none`` display ties with the browser's own
    ``[hidden] { display: none }`` and wins on source order — exactly as an id
    selector does.  The element being built in JS rather than written in the
    template changes nothing about the cascade, and it is where this bug
    actually shipped: MolView's panel has no ids at all.
    """
    hidden_classes = _scan_js_for_hidden_classes()
    assert hidden_classes, (
        "the class scanner found nothing, so this test passes vacuously — "
        "the `el(tag, className)` pattern it keys on must have changed"
    )
    failures: list[str] = []
    for css_path in STATIC.rglob("*.css"):
        css = _strip_comments(css_path.read_text(encoding="utf-8"))
        for selector_list, props in _split_rules(css):
            if not _NON_NONE_DISPLAY_RE.search(props):
                continue
            for selector in (s.strip() for s in selector_list.split(",")):
                # The bare class, on its own: `.foo { display: flex }`.  A
                # descendant rule cannot be guarded by the element's own
                # attribute, so only the self-selector is checked.
                if not re.fullmatch(r"\.([\w-]+)", selector):
                    continue
                cls = selector[1:]
                if cls not in hidden_classes:
                    continue
                if f"{selector}[hidden]" in all_hidden_guards:
                    continue
                failures.append(
                    f"  .{cls} is hidden by JS, and `{selector} "
                    f"{{ display: … }}` "
                    f"({css_path.relative_to(STATIC).as_posix()}) overrides "
                    f"that.  Add: `{selector}[hidden] {{ display: none; }}`"
                )
    if failures:
        pytest.fail(
            "CSS [hidden]-precedence violations on class-named elements "
            "(the control is hidden in JS and drawn anyway):\n"
            + "\n".join(sorted(failures))
        )


def test_audit_actually_found_some_hidden_ids(dynamically_hidden_ids):
    """Sanity gate: if the JS-scanner returns an empty set, the
    main test passes vacuously and a future scanner regression
    (e.g., a regex that stops matching) goes unnoticed.  Assert
    that we found at least the canonical few ids we know JS hides.
    """
    # 2026-06-14: ``hide-frozen-row`` and ``phase-indicator`` were
    # removed -- neither is dynamically hidden by JS any more (the
    # hide-frozen row is template-owned + always-visible; the
    # phase-indicator's setHidden was refactored).  Use IDs the
    # current scanner actually finds.  This sanity gate's purpose
    # is to catch a regex regression in the scanner; any handful
    # of real ids satisfies that.
    expected_some_of = {
        "fdf-output",        # /structure-optimization
        "auto-detect-panel", # /structure-optimization
    }
    found = dynamically_hidden_ids & expected_some_of
    assert found, (
        "JS-scan returned no canonical hidden-ids.  The "
        "regex in _scan_js_for_hidden_ids is likely broken; "
        f"expected to find any of {expected_some_of}, got "
        f"{sorted(dynamically_hidden_ids)[:10]}."
    )


# --------------------------------------------------------------------- #
#  The blind spot the id-keyed scan above cannot see                    #
# --------------------------------------------------------------------- #
#
# Everything above keys on the ID: JS hides `#foo`, so a CSS rule naming `#foo`
# needs a guard.  That misses the commonest shape in this codebase, because the
# rule usually does NOT name the id -- it names a CLASS the element happens to
# carry:
#
#     <label class="inline-label" id="anim-temperature-row" hidden>
#     .viewer-controls .inline-label { display: inline-flex; }
#
# Nothing in the CSS mentions `anim-temperature-row`, so no amount of improving
# the JS scan connects them.  The link is in the TEMPLATE, and only reading it
# closes the loop.
#
# 2026-08-05: that is exactly how the Temperature control shipped visible under
# an amplitude mode that has no temperature.  It was found by looking at the
# rendered page.

_ID_CLASS_RE = re.compile(
    r"""<[a-zA-Z][^>]*?
        (?: id=["']([\w\-]+)["'][^>]*?class=["']([^"']+)["']
          | class=["']([^"']+)["'][^>]*?id=["']([\w\-]+)["'] )
    """,
    re.VERBOSE | re.S,
)

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "molbuilder" / "web" / "templates"


def _classes_of_hideable_ids() -> dict[str, set[str]]:
    """id -> the classes it carries, for ids some JS toggles via `.hidden`."""
    hideable = _scan_js_for_hidden_ids()
    out: dict[str, set[str]] = {}
    for tpl in TEMPLATES_DIR.rglob("*.html"):
        html = tpl.read_text(encoding="utf-8")
        for m in _ID_CLASS_RE.finditer(html):
            el_id = m.group(1) or m.group(4)
            classes = (m.group(2) or m.group(3) or "").split()
            if el_id in hideable:
                out.setdefault(el_id, set()).update(classes)
    return out


def test_a_class_on_a_hideable_element_does_not_defeat_the_hidden_attribute():
    """A rule that sets `display` on a class ties with the browser's own
    ``[hidden] { display: none }`` and wins on source order.

    So if JS hides an element by id, and a CSS rule styles it by CLASS, the
    element stays in the layout: a control the code believes is gone, sitting
    there for someone to click.  The rule needs its own ``[hidden]`` guard.
    """
    offenders: list[str] = []
    id_classes = _classes_of_hideable_ids()
    if not id_classes:
        pytest.skip("no hideable element carries a class in any template")

    for css_path in STATIC.rglob("*.css"):
        if any(x in css_path.as_posix() for x in ("vendor/", "codemirror")):
            continue
        text = css_path.read_text(encoding="utf-8")
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        for block in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
            selector, body = block.group(1).strip(), block.group(2)
            display = re.search(r"(?:^|;)\s*display\s*:\s*([a-zA-Z-]+)", body)
            if not display or display.group(1) == "none":
                continue
            if "[hidden]" in selector:
                continue
            # THE SUBJECT DECIDES (2026-08-20).  A display rule un-hides the
            # element only when its LAST compound targets it: `.bar {}` and
            # `.panel .bar {}` style the element itself, while
            # `.bar table {}` styles a DESCENDANT and `.bar::before {}` a
            # pseudo-element -- both live inside the element's box, so the
            # parent's `[hidden] -> display:none` already removes them and a
            # guard there would gate nothing.  Before this refinement the
            # matcher read class names anywhere in the selector, and closing
            # the id-scan's idiom gap (the bare-assignment fix that finally
            # collected checkpoint.js's and page.js's tables) surfaced four
            # such cascade-safe rules as false alarms beside the one real
            # `.docs-view-bar` bug.
            if "::" in selector:
                continue
            subject = re.split(r"[\s>+~]+", selector.strip())[-1]
            named = set(re.findall(r"\.([A-Za-z][\w-]*)", subject))
            for el_id, classes in id_classes.items():
                if not (named & classes):
                    continue
                # Guarded if the same sheet carries a [hidden] rule for any of
                # the classes this selector names.
                guard = any(
                    re.search(re.escape("." + c) + r"[^{,]*\[hidden\]", text)
                    for c in (named & classes)
                )
                if not guard:
                    offenders.append(
                        f"{css_path.relative_to(STATIC).as_posix()}: "
                        f"`{selector}` sets display:{display.group(1)} and matches "
                        f"#{el_id}, which JS hides via .hidden"
                    )
    assert not offenders, (
        "these elements are hidden by JS and un-hidden by CSS -- the control "
        "stays in the layout and the code believes it is gone:\n  "
        + "\n  ".join(sorted(set(offenders)))
        + "\n\nAdd a `[hidden] { display: none }` rule for the class."
    )

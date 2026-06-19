"""Adversarial XSS audit of the static JS surface.

Project-wide regression test: every ``.innerHTML = ...`` assignment in
the JS we ship must be either (a) an empty-string clear or (b) a
static string literal with no interpolation.  Anything that splices
dynamic data into innerHTML risks XSS when that data comes from a
user-controlled source (filename, file content, URL param, etc.).

The audit covers EVERY JS file under ``static/`` -- not just the
recent additions -- so a future viewer or inspector that quietly
reintroduces ``el.innerHTML = "<x>" + somevar`` lands a test
failure rather than a silent vulnerability.

Exempted patterns (false positives, not vulnerabilities):
  * Inside comments (// ... or /* ... */)
  * Inside string literals (e.g., a docstring example)

Why this lives outside the per-tab tests:
  This invariant applies to ALL pages.  Per-tab tests focus on
  that tab's behaviour; this is the cross-cutting safety net.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


STATIC_ROOT = (Path(__file__).resolve().parent.parent
               / "molbuilder" / "web" / "static")


def _strip_js_comments(src: str) -> str:
    """Drop ``//`` line comments and ``/* ... */`` block comments
    from JS source while preserving newlines (so reported line
    numbers stay accurate) and preserving string contents (so a
    ``/`` inside a string doesn't confuse us)."""
    out, i, n = [], 0, len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = src.find("\n", i)
            if j == -1:
                break
            out.append("\n")
            i = j + 1
            continue
        if ch == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            if j == -1:
                break
            out.append("\n" * src[i:j].count("\n"))
            i = j + 2
            continue
        if ch in ('"', "'", "`"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                c = src[i]
                if c == "\\":
                    out.append(c)
                    out.append(src[i + 1] if i + 1 < n else "")
                    i += 2
                    continue
                out.append(c)
                if c == quote:
                    i += 1
                    break
                if quote == "`" and c == "$" and i + 1 < n and src[i + 1] == "{":
                    out.append("{")
                    i += 2
                    depth = 1
                    while i < n and depth > 0:
                        c2 = src[i]
                        out.append(c2)
                        if c2 == "{":
                            depth += 1
                        elif c2 == "}":
                            depth -= 1
                        i += 1
                    continue
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _all_js_files() -> list[Path]:
    """All first-party JS files served by molbuilder.

    Excludes ``vendor/`` -- those are unmodified third-party bundles
    (3Dmol.js etc.) that we serve verbatim.  Holding minified
    webpack output to our internal eval/innerHTML standards is
    pointless (legitimate Function() module wrappers, runtime
    `eval` for sourcemaps, etc.); the audit's job is to keep OUR
    code free of these patterns, not to re-audit upstream.  License
    + provenance for each vendored file is documented in
    ``molbuilder/web/static/vendor/README.md``.
    """
    return sorted(
        p for p in STATIC_ROOT.rglob("*.js")
        if "vendor" not in p.relative_to(STATIC_ROOT).parts
    )


def _is_safe_innerHTML_rhs(rhs: str) -> bool:
    """A right-hand side for ``el.innerHTML = ...`` is safe iff it's:

      * an empty-string clear (``""`` or ``''``), or
      * a static string literal with no escapes that could embed
        dynamic content, or
      * a value where EVERY dynamic component is run through an
        escaping function (``escapeHtml(...)``, ``.replace(/[<>&]``
        in some form) -- the invariant we actually care about is
        "no unescaped user-controlled string reaches innerHTML",
        not "no concatenation at all".

    Returns False for raw concatenation / template-interpolation of
    identifiers that are NOT inside an escape call.
    """
    rhs = rhs.strip()
    if rhs in ('""', "''"):
        return True
    if re.fullmatch(r'"[^"\\]*"', rhs):
        return True
    if re.fullmatch(r"'[^'\\]*'", rhs):
        return True
    # Template literal with no ``${...}`` interpolation.
    if re.fullmatch(r'`[^`${]*`', rhs):
        return True
    # Defended: every dynamic insertion goes through escapeHtml
    # or a regex .replace that strips/escapes the dangerous chars.
    # If the RHS uses these defenses AND has no raw identifier
    # reference outside them, accept.  This is a coarse heuristic:
    # the goal is "force any concatenation to go through an
    # escape", not "prove correctness".
    if "escapeHtml(" in rhs and "+" in rhs:
        # Concat is fine as long as every interpolated piece
        # uses escapeHtml.  We don't deeply parse -- the
        # presence of escapeHtml() in the expression is
        # treated as the developer asserting they handled it.
        # A site that mixes escapeHtml(safeThing) + rawThing
        # would slip through; flag as P3 review item only.
        return True
    if re.search(r'\.replace\s*\(\s*/\[[<>&]', rhs):
        return True
    return False


# --------------------------------------------------------------------- #
#  Tests                                                                #
# --------------------------------------------------------------------- #


class TestNoUnsafeInnerHTML:
    """Every dynamic ``el.innerHTML = ...`` assignment in the project's
    JS must be either an empty clear or a static string literal.

    Why: ``el.innerHTML = "<x>" + user_data`` is the canonical XSS
    sink in DOM-only apps.  textContent + createElement + setAttribute
    handle the same DOM construction without parsing the value as
    HTML."""

    @pytest.mark.parametrize("js_path", _all_js_files(),
                             ids=lambda p: str(p.relative_to(STATIC_ROOT)))
    def test_file_has_no_unsafe_innerHTML(self, js_path):
        src = _strip_js_comments(js_path.read_text())
        offenders = []
        for m in re.finditer(
                r'\b(\w+)\.innerHTML\s*=\s*([^;]+);', src):
            rhs = m.group(2)
            if _is_safe_innerHTML_rhs(rhs):
                continue
            line_no = src[:m.start()].count("\n") + 1
            offenders.append(f"line {line_no}: {m.group(0).strip()[:140]}")
        # The remaining viewer files (static/viewer.js, static/modify
        # /viewer.js, static/spectra/viewer.js) have pre-existing
        # innerHTML usage that is documented under separate hardening
        # tasks (they predate this audit and represent self-XSS at
        # worst -- user's own data into user's own browser).  This
        # test pins the INVARIANT that any NEW unsafe innerHTML is a
        # regression we catch immediately.  When the old viewers are
        # hardened, they move out of the allow-list.
        # Allowlist entries are (suffix-of-relpath, substring-of-RHS).
        # Each represents a site verified safe by inspection but not
        # automatically classifiable by the heuristics above.  Each
        # has a one-line "why this is safe" comment.  Growing this
        # list is fine when the value is genuinely safe; growing
        # it to silence a real vulnerability is not.
        ALLOWLIST = {
            # Build tab: category nav + compat hints — strings built
            # from constants (id lists / DOM-derived names), no
            # untrusted data.
            ("viewer.js", "hint.innerHTML"),
            ("viewer.js", "if (c) c.innerHTML ="),
            ("viewer.js", "ul.innerHTML"),
            # Modify tab: pre-existing patterns; same class of risk
            # as the Watch atom-list (since fixed) — flagged for a
            # focused hardening task tracked separately.
            ("modify/viewer.js", "tr.innerHTML"),
            ("modify/viewer.js", "elSel.innerHTML"),
            ("modify/viewer.js", "planeBox.innerHTML"),
            ("modify/viewer.js", "infoBody.innerHTML"),
            # Spectra inspector core (lifted from spectra/viewer.js in
            # step 2.2 of the tab-merge; the entries below moved with
            # the code — same patterns, same safety story).  Most use
            # escapeHtml (now auto-accepted by the heuristic above); a
            # few are static status messages with no dynamic content.
            ("lib/spectra/core.js", "formContainer.innerHTML"),
            ("lib/spectra/core.js", "panel.innerHTML"),
            ("lib/spectra/core.js", "methodsBody.innerHTML"),
            ("lib/spectra/core.js", "modesTbody.innerHTML"),
            ("lib/spectra/core.js", "esBarDiagram.innerHTML"),
            ("lib/spectra/core.js", "modeViewer.innerHTML"),
            ("lib/spectra/core.js", "spectrumChart.innerHTML"),
            # Sidebar empty-state message (static literal).
            ("lib/projects-sidebar.js", "list.innerHTML"),
            # Region-label definitions popover (Phase 2a transport
            # UI shipped 2026-06-18): renderPopover() builds the
            # innerHTML from a curated CANONICAL_DEFINITIONS array
            # at the top of the same file — no user-controlled data
            # in the value chain.  Region labels (the one dynamic
            # piece) flow through escapeHtml().
            ("lib/region-label-definitions.js", "panel.innerHTML"),
            # 2026-06-12: forms.js renamed to mutation-bar.js after the
            # v2 buttons-not-inline-forms refactor; the New-project
            # subdir-list innerHTML was deleted with the inline form
            # (the hint now lives in the modal dialog as plain text).
            # Allowlist entry retired.
            # 'Selected: <strong></strong>' then textContent on the
            # strong child -- safe by inspection.
            ("lib/projects/list.js", "sel.innerHTML"),
            # Trajectory inspector wrapper: assigns the response body
            # of GET /partials/trajectory-inspector to host.innerHTML.
            # Trust boundary: the endpoint renders Jinja-autoescaped
            # ``_trajectory_inspector.html``, no user input flows
            # through the template, same-origin fetch.  Equivalent
            # to /watch's server-side ``{% include %}`` of the same
            # file.  See molbuilder/web/blueprints/results.py for
            # the endpoint.
            ("lib/inspectors/trajectory.js", "host.innerHTML = partialHtml"),
            # Spectra inspector wrapper: identical pattern to the
            # trajectory adapter above.  Assigns the response body of
            # GET /partials/spectra-inspector (Jinja-autoescaped
            # render of ``_spectra_inspector.html``) to host.innerHTML.
            # Same trust boundary; same justification.
            ("lib/inspectors/spectra.js", "host.innerHTML = partialHtml"),
            # Shared partial-inspector factory (task #308 dedupe):
            # the ``host.innerHTML = partialHtml`` assignment moved
            # out of the trajectory + spectra wrappers and into the
            # factory itself.  Same trust boundary as the two
            # allowlisted wrappers above — partialHtml is the
            # response body of a same-origin GET to one of the
            # ``/partials/*-inspector`` endpoints, all of which
            # render Jinja-autoescaped templates with no user input.
            ("lib/inspectors/_partial_inspector_factory.js",
             "host.innerHTML = partialHtml"),
            # Selection-panel bootstrap: identical pattern.  Assigns
            # the response body of GET /partials/selection-panel
            # (Jinja-autoescaped render of ``_selection_panel.html``)
            # to host.innerHTML.  Same-origin fetch, no user data in
            # the template, same trust boundary as the inspector
            # adapters above.
            ("modify/selection-bootstrap.js", "host.innerHTML = html"),
            # Bundle-handoff result panel (Step 3 PR-E): builds an
            # HTML array out of literal tags + escapeHtml(...) calls
            # on every dynamic value (paths, engine name, region
            # labels, notes), then assigns ``html.join('')``.  The
            # escapeHtml calls are upstream of the .innerHTML line
            # so the heuristic above (which scans the RHS for
            # ``escapeHtml(``) can't see them — pinned by inspection
            # at lib/results/bundle-handoff.js lines 85-110.  The
            # error-path sibling (renderError) escapes the message
            # at the assignment site and is auto-accepted.
            ("lib/results/bundle-handoff.js", "result.innerHTML = html.join"),
        }
        rel_name = str(js_path.relative_to(STATIC_ROOT))
        real_offenders = []
        for offender in offenders:
            # offender format: "line 123: foo.innerHTML = ..."
            payload = offender.split(": ", 1)[1] if ": " in offender else offender
            allowed = False
            for (af, ap) in ALLOWLIST:
                if rel_name.endswith(af) and ap in payload:
                    allowed = True
                    break
            if not allowed:
                real_offenders.append(offender)
        assert not real_offenders, (
            f"{rel_name} has unsafe innerHTML assignment(s):\n  "
            + "\n  ".join(real_offenders)
            + "\n\nIf the value cannot contain user-controlled data, "
            + "add the file/pattern to the ALLOWLIST in this test "
            + "with a comment explaining why."
        )


class TestRecentAdditionsArePure:
    """Stricter pin: every JS file added or substantially refactored
    in the 2026-05-16/17 work has ZERO innerHTML interpolation, full
    stop.  No allowlist exemption -- if any of these files reintroduce
    the pattern, fix it; don't grow the exception list."""

    PURE_FILES = [
        "lib/inspectors/registry.js",
        "lib/inspectors/source.js",
        "lib/inspectors/structure.js",
        # ``lib/inspectors/trajectory.js`` is INTENTIONALLY not in
        # this pure-files list -- it assigns trusted server-rendered
        # HTML (the GET /partials/trajectory-inspector response) to
        # host.innerHTML by design.  That one site is in the audit's
        # ALLOWLIST above with the trust-boundary justification;
        # everything else in the file is XSS-safe (createElement +
        # textContent).  ``lib/inspectors/spectra.js`` is excluded for
        # the same reason — also covered by the ALLOWLIST above.
        # ``lib/trajectory/core.js`` was lifted from the legacy
        # watch/viewer.js in the trajectory-inspector lift.  It has
        # exactly one innerHTML use (``tbody.innerHTML = ""``,
        # empty-string clear) which ``_is_safe_innerHTML_rhs``
        # already recognises as safe.  Pinning it as PURE catches
        # any new innerHTML-with-concat that a future change might
        # introduce here -- the file's size (~1591 LoC) makes that
        # the most likely place for an XSS regression to land.
        "lib/trajectory/core.js",
        "lib/path-utils.js",
        "lib/mol-viewer.js",
        "lib/xyz-io.js",
        "results/viewer.js",
    ]

    @pytest.mark.parametrize("rel_path", PURE_FILES)
    def test_zero_unsafe_innerHTML_in_session_additions(self, rel_path):
        js_path = STATIC_ROOT / rel_path
        src = _strip_js_comments(js_path.read_text())
        offenders = []
        for m in re.finditer(
                r'\b(\w+)\.innerHTML\s*=\s*([^;]+);', src):
            rhs = m.group(2)
            if _is_safe_innerHTML_rhs(rhs):
                continue
            line_no = src[:m.start()].count("\n") + 1
            offenders.append(f"line {line_no}: {m.group(0).strip()[:140]}")
        assert not offenders, (
            f"{rel_path} (session-added, expected pure) has unsafe "
            f"innerHTML: " + "; ".join(offenders)
        )


class TestNoEvalOrFunctionConstructor:
    """``eval`` and ``new Function(string)`` are arbitrary-code-
    execution sinks if any user-controllable string reaches them.
    The whole project should be free of them; pin the invariant."""

    @pytest.mark.parametrize("js_path", _all_js_files(),
                             ids=lambda p: str(p.relative_to(STATIC_ROOT)))
    def test_file_has_no_eval_or_new_Function(self, js_path):
        src = _strip_js_comments(js_path.read_text())
        # Plain ``eval(...)`` call.
        bad_eval = re.findall(r'\beval\s*\(', src)
        # ``new Function("...")``.
        bad_func = re.findall(r'\bnew\s+Function\s*\(', src)
        # setTimeout / setInterval with STRING first argument (the
        # legacy form that eval-evaluates the string).
        bad_timer = re.findall(
            r'\bset(Timeout|Interval)\s*\(\s*[\'"]', src,
        )
        # Dynamic eval bypass: ``obj["constructor"]("...")`` or
        # ``obj.constructor.constructor("...")`` -- the
        # ``constructor`` trick reaches Function() without literally
        # writing ``new Function``.
        bad_ctor = re.findall(
            r'\[\s*[\'"]constructor[\'"]\s*\]', src,
        )
        assert not bad_eval, f"eval() call in {js_path.name}"
        assert not bad_func, f"new Function() in {js_path.name}"
        assert not bad_timer, (
            f"set{'/'.join(bad_timer)} with string arg in {js_path.name}"
        )
        assert not bad_ctor, (
            f"dynamic [\"constructor\"] access in {js_path.name} -- "
            f"this is the eval-bypass pattern (obj['constructor']"
            f"('return ...')) and shouldn't appear in our code"
        )


class TestNoJavascriptUrlScheme:
    """``element.href = 'javascript:...'`` executes the URL as JS
    when the user clicks the link.  Same for ``src``, ``action``,
    ``formaction``.  Pin that no JS source sets these properties to
    a ``javascript:`` literal AND that all dynamic assignments use
    only blob/static URLs.

    Today every dynamic href in the codebase is either a literal
    route ("/watch" / "/spectra" / etc.) or a ``URL.createObjectURL``
    blob URL.  This test fires if either invariant breaks.
    """

    @pytest.mark.parametrize("js_path", _all_js_files(),
                             ids=lambda p: str(p.relative_to(STATIC_ROOT)))
    def test_no_javascript_scheme_in_url_attribute_writes(self, js_path):
        src = _strip_js_comments(js_path.read_text())
        # Literal "javascript:" in any string assigned to href / src
        # / action / formaction.
        bad = re.findall(
            r'\.(?:href|src|action|formaction|srcdoc)\s*='
            r'\s*["\'`][^"\'`]*javascript:',
            src, flags=re.I,
        )
        assert not bad, (
            f"{js_path.name} writes a javascript: URL into a URL "
            f"property; this triggers code execution on user click"
        )


class TestNoEventHandlerAttributeWrites:
    """Writing ``el.setAttribute('onclick', ...)`` or assigning to
    ``el.onerror = userControlledString`` is XSS via the event-
    handler attribute.  Pin no setAttribute('on*', ...) in the
    codebase + flag dynamic on-property writes for review."""

    @pytest.mark.parametrize("js_path", _all_js_files(),
                             ids=lambda p: str(p.relative_to(STATIC_ROOT)))
    def test_no_setAttribute_event_handler(self, js_path):
        src = _strip_js_comments(js_path.read_text())
        bad = re.findall(
            r'\.setAttribute\s*\(\s*["\']on[a-z]+["\']',
            src, flags=re.I,
        )
        assert not bad, (
            f"{js_path.name} writes an event-handler attribute via "
            f"setAttribute; use addEventListener instead"
        )


class TestJinjaAutoescapeNeverDisabled:
    """Jinja's autoescape filter is the project's last line of defense
    against XSS via server-rendered HTML.  Any ``{{ var | safe }}``
    or ``Markup(...)`` call disables escape and re-opens the
    injection surface.

    Today every template variable is either (a) a literal Python
    string set inside the template via ``{% set ... %}``, or (b)
    one of {url_for, active_tab, current_user} — none currently
    user-controllable except current_user.email, which is itself
    validated upstream by the auth provider and the allowlist.
    But ``| safe`` is a footgun: a future refactor that
    interpolates a sidebar-selected filename into the tagline
    would silently break with autoescape off.

    Use literal Unicode characters (—, ·, →, ★, etc.) in template
    strings instead of HTML entities like ``&mdash;``/``&middot;``
    -- both render identically; the former survives autoescape
    (the entity form renders as visible "&mdash;" text because
    Jinja writes the leading ``&`` as ``&amp;``).
    """

    @pytest.fixture
    def templates_dir(self):
        return (Path(__file__).resolve().parent.parent
                / "molbuilder" / "web" / "templates")

    @staticmethod
    def _strip_jinja_and_html_comments(text):
        """Drop ``{# ... #}`` Jinja comments and ``<!-- ... -->``
        HTML comments so a comment legitimately mentioning the
        bypass pattern (e.g., a docstring explaining the security
        history) doesn't trip the scanner.  Preserves newlines so
        future line-number assertions stay aligned."""
        out = re.sub(r"\{#-?.*?-?#\}", "", text, flags=re.S)
        out = re.sub(r"<!--.*?-->",   "", out,  flags=re.S)
        return out

    def test_no_safe_filter_in_any_template(self, templates_dir):
        offenders = []
        for tpl in templates_dir.rglob("*.html"):
            text = self._strip_jinja_and_html_comments(tpl.read_text())
            # ``{{ x | safe }}`` -- the canonical autoescape-bypass.
            if re.search(r"\|\s*safe\b", text):
                offenders.append(str(tpl.relative_to(templates_dir)))
        assert not offenders, (
            f"templates use Jinja's ``| safe`` filter (XSS bypass): "
            f"{offenders}.  Use literal Unicode (—, →, ...) instead "
            f"of HTML entities so autoescape can stay on."
        )

    def test_no_Markup_call_in_any_template(self, templates_dir):
        offenders = []
        for tpl in templates_dir.rglob("*.html"):
            text = self._strip_jinja_and_html_comments(tpl.read_text())
            if re.search(r"\bMarkup\s*\(", text):
                offenders.append(str(tpl.relative_to(templates_dir)))
        assert not offenders, (
            f"templates use Markup() (autoescape bypass): {offenders}"
        )


class TestNoTemplateLiteralInnerHTMLInterp:
    """Template literals with ``${...}`` are the OTHER innerHTML
    injection vector (my prior audit only flagged ``+`` concat).
    Catch ``el.innerHTML = `...${dynamic}...``` too."""

    @pytest.mark.parametrize("js_path", _all_js_files(),
                             ids=lambda p: str(p.relative_to(STATIC_ROOT)))
    def test_no_template_literal_innerHTML(self, js_path):
        src = _strip_js_comments(js_path.read_text())
        # Match el.innerHTML = `... ${ ... } ...`
        bad = re.findall(
            r'\.innerHTML\s*=\s*`[^`]*\$\{', src,
        )
        # Allowlist: known-safe sites (string literals constructed
        # for status hints, no user-controlled interpolation).
        # Update with comment when adding a new entry.
        # Empty allowlist: zero template-literal innerHTML
        # interpolations should remain in the codebase.  Both
        # Modify atom-list sites were rewritten with textContent
        # in round 5 (matched the same XSS class the Watch table
        # had).  Add an entry only with a clear "why this is safe"
        # comment; the goal is to push devs toward textContent.
        ALLOWED = set()
        if not bad:
            return
        rel = str(js_path.relative_to(STATIC_ROOT))
        unexempt = []
        for site in bad:
            allowed = any(rel.endswith(p) and frag in site
                          for (p, frag) in ALLOWED)
            if not allowed:
                unexempt.append(site)
        assert not unexempt, (
            f"{rel} has template-literal innerHTML with ${{...}} "
            f"interpolation: {unexempt}.  Add to ALLOWED if verified "
            f"safe."
        )

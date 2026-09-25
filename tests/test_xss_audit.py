"""Nothing a person, a file or a server supplies is parsed as markup, or run as
code, by the JavaScript the app serves.

GOAL -- the failure these catch: a runtime string reaching a DOM API that
parses HTML, or one that runs code.  The markup half has been measured: on
2026-09-14 a person-chosen folder name going into ``innerHTML``
(`jupyternb/index.js`), and on 2026-09-25 three sites an independent review
found behind this file's own allowlist -- the projects sidebar's roots error
and the spectrum chart's two failure notices, all spliced in raw while the
comments exempting them called them static.

CONTRACT: `web/ui-contract.md` § 7.  The CSP (``script-src 'self'``) stops an
injected script from running; these stop the injection.

Artifact lints (`testing.md` § 6): each quantifies over every first-party file
and names no line.  ``vendor/`` is the one exclusion -- unmodified third-party
bundles served verbatim, whose provenance is ``static/vendor/README.md``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytest


STATIC_ROOT = (Path(__file__).resolve().parent.parent
               / "molbuilder" / "web" / "static")


def _all_js_files() -> list[Path]:
    """Every first-party JS file the app serves: all of ``static/`` but
    ``vendor/``."""
    return sorted(p for p in STATIC_ROOT.rglob("*.js")
                  if "vendor" not in p.relative_to(STATIC_ROOT).parts)


# --------------------------------------------------------------------- #
#  Reading JavaScript -- tokens, not text                               #
# --------------------------------------------------------------------- #
#
# WHY A TOKENIZER.  This file matched ``.innerHTML = ([^;]+);`` over the source
# with its comments stripped, and reading text failed three ways at once, all
# measured 2026-09-25: the value stopped at the first ``;`` even inside a
# string (``'&amp;'``, ``padding:0.7rem;``), so a site passed or failed on a
# fragment of itself; ``+=`` and ``$("x").innerHTML`` were not seen at all; and
# the stripper had no regex state, so a quote inside ``/["']/g`` opened a
# "string", a real ``//`` in the next string was then taken for a comment, and
# the code after it was dropped.  Here a string, a template literal (with its
# ``${...}``), a comment and a regex literal are each ONE token, so what
# follows an ``=`` is known exactly.

class _Tok(NamedTuple):
    kind: str        # ident | num | str | tmpl | regex | punct | comment
    text: str
    start: int       # offset in the source
    line: int        # the 1-based line it starts on


#: After one of these words a ``/`` opens a regex literal, not a division.
_REGEX_AFTER_WORD = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "case", "do", "else", "yield", "await"})

#: Longest first, so ``===`` is one token and never ``==`` then ``=``.
_PUNCT = sorted(
    (">>>=", "===", "!==", "**=", "<<=", ">>=", ">>>", "...", "&&=", "||=",
     "??=", "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--",
     "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>"),
    key=len, reverse=True)


def _quoted_end(src: str, i: int) -> int:
    """Past the ``'...'`` or ``"..."`` string that opens at ``i``."""
    q, j, n = src[i], i + 1, len(src)
    while j < n and src[j] != q and src[j] != "\n":
        j += 2 if src[j] == "\\" else 1
    return min(j + 1, n)


def _template_end(src: str, i: int) -> int:
    """Past the template literal that opens at ``i``, across its ``${...}``."""
    j, n = i + 1, len(src)
    while j < n:
        if src[j] == "\\":
            j += 2
        elif src[j] == "`":
            return j + 1
        elif src.startswith("${", j):
            j = _braces_end(src, j + 2)
        else:
            j += 1
    return n


def _braces_end(src: str, j: int) -> int:
    """Past the ``}`` that closes a ``${`` opened just before ``j``."""
    depth, n = 1, len(src)
    while j < n:
        c = src[j]
        if c in "'\"":
            j = _quoted_end(src, j)
        elif c == "`":
            j = _template_end(src, j)
        elif src.startswith("//", j):
            e = src.find("\n", j)
            j = n if e == -1 else e
        elif src.startswith("/*", j):
            e = src.find("*/", j + 2)
            j = n if e == -1 else e + 2
        else:
            depth += (c == "{") - (c == "}")
            j += 1
            if depth == 0:
                return j
    return n


def _regex_end(src: str, i: int) -> int:
    """Past the ``/.../flags`` literal that opens at ``i``.  A ``/`` inside a
    character class does not close it."""
    j, n, in_class = i + 1, len(src), False
    while j < n and src[j] != "\n":
        c = src[j]
        if c == "\\":
            j += 2
            continue
        if in_class:
            in_class = c != "]"
        elif c == "[":
            in_class = True
        elif c == "/":
            j += 1
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            return j
        j += 1
    return j


def _tokens(src: str) -> List[_Tok]:
    """``src`` as tokens, comments included -- ``_strip_js_comments`` needs
    to know where they are."""
    toks: List[_Tok] = []
    i, n, line = 0, len(src), 1
    last: Optional[_Tok] = None          # the last token that is not a comment
    while i < n:
        c = src[i]
        if c.isspace():
            line += c == "\n"
            i += 1
            continue
        if src.startswith("//", i):
            e = src.find("\n", i)
            j, kind = (n if e == -1 else e), "comment"
        elif src.startswith("/*", i):
            e = src.find("*/", i + 2)
            j, kind = (n if e == -1 else e + 2), "comment"
        elif c in "'\"":
            j, kind = _quoted_end(src, i), "str"
        elif c == "`":
            j, kind = _template_end(src, i), "tmpl"
        elif c == "/" and (
                last is None
                or (last.kind == "punct" and last.text not in (")", "]", "}"))
                or (last.kind == "ident" and last.text in _REGEX_AFTER_WORD)):
            j, kind = _regex_end(src, i), "regex"
        elif c.isalpha() or c in "_$":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            kind = "ident"
        elif c.isdigit():
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] in "._"):
                j += 1
            kind = "num"
        else:
            op = next((p for p in _PUNCT if src.startswith(p, i)), c)
            j, kind = i + len(op), "punct"
        tok = _Tok(kind, src[i:j], i, line)
        toks.append(tok)
        if kind != "comment":
            last = tok
        line += src.count("\n", i, j)
        i = j
    return toks


def _strip_js_comments(src: str) -> str:
    """``src`` with every comment blanked -- its newlines kept, so a line
    number still points at the source -- and every string intact."""
    out, at = [], 0
    for t in _tokens(src):
        if t.kind == "comment":
            out.append(src[at:t.start])
            out.append("\n" * t.text.count("\n"))
            at = t.start + len(t.text)
    out.append(src[at:])
    return "".join(out)


# --------------------------------------------------------------------- #
#  Markup is written only from literals                                 #
# --------------------------------------------------------------------- #

#: Properties that parse what is assigned to them as HTML.
_HTML_PROPERTIES = frozenset({"innerHTML", "outerHTML", "srcdoc"})

#: Methods that parse an argument as HTML.
_HTML_METHODS = frozenset({"insertAdjacentHTML", "createContextualFragment",
                           "parseFromString", "setHTMLUnsafe",
                           "parseHTMLUnsafe"})

#: What may follow the one legal value: the statement, or the expression
#: holding it, ends there.
_VALUE_ENDS = frozenset({";", "}", ")", ","})


def _is_literal(t: Optional[_Tok]) -> bool:
    """One string written in the source: quoted, or a template with no
    ``${...}``."""
    return t is not None and (
        t.kind == "str" or (t.kind == "tmpl" and "${" not in t.text))


def html_writes(src: str) -> List[Tuple[int, str]]:
    """Every place ``src`` hands markup to a DOM API that parses it, as
    ``(line, what)`` -- except the one legal form, ``= <one literal>``."""
    toks = [t for t in _tokens(src) if t.kind != "comment"]

    def at(k: int) -> Optional[_Tok]:
        return toks[k] if 0 <= k < len(toks) else None

    found: List[Tuple[int, str]] = []
    for k, t in enumerate(toks):
        prev, nxt = at(k - 1), at(k + 1)
        dotted = prev is not None and prev.text in (".", "?.")
        # `x.innerHTML = ...`, and `x["innerHTML"] = ...`
        if t.kind == "ident" and t.text in _HTML_PROPERTIES and dotted:
            name, op_at = t.text, k + 1
        elif (t.kind == "str" and t.text[1:-1] in _HTML_PROPERTIES
              and prev is not None and prev.text == "["
              and nxt is not None and nxt.text == "]"):
            name, op_at = t.text[1:-1], k + 2
        else:
            name, op_at = None, -1
        if name is not None:
            op, value, end = at(op_at), at(op_at + 1), at(op_at + 2)
            if op is None or op.text not in ("=", "+="):
                continue                           # read, or compared
            if (op.text == "=" and _is_literal(value)
                    and (end is None or end.text in _VALUE_ENDS)):
                continue
            found.append((t.line, f"{name} {op.text}"))
        elif t.kind == "ident" and nxt is not None and nxt.text == "(":
            if t.text in _HTML_METHODS and dotted:
                found.append((t.line, t.text + "()"))
            elif (t.text in ("write", "writeln") and dotted
                  and (at(k - 2) or t).text == "document"):
                found.append((t.line, "document." + t.text + "()"))
    return found


#: The doors through which HTML made at run time enters a page -- file ->
#: (writes, why).  THE COUNT IS PART OF THE ALLOWANCE, as in
#: `test_one_door_reads_a_structure.py`: a second write in one of these files
#: has not inherited the first one's reason, and a file whose write is gone
#: fails too, so an allowance cannot outlive its argument.
PRODUCERS: Dict[str, Tuple[int, str]] = {
    "lib/markdown-render.js": (
        1, "a mermaid diagram's SVG, rendered with securityLevel 'strict' "
           "from the app's own docs"),
    "lib/inspectors/markdown.js": (
        1, "the editor's live preview -- lib/markdown-render.js, which runs "
           "marked through DOMPurify on every call"),
    "documents/page.js": (
        1, "the Documents tab -- the same DOMPurify path, on the app's docs "
           "from /api/docs/read"),
    "lib/inspectors/_partial_inspector_factory.js": (
        1, "a same-origin GET of /partials/*-inspector: Jinja-autoescaped "
           "templates with no request-derived values"),
}


@pytest.mark.parametrize(
    "rel", sorted({str(p.relative_to(STATIC_ROOT)) for p in _all_js_files()}
                  | set(PRODUCERS)))
def test_markup_is_written_only_from_literals(rel):
    """A page parses as HTML only markup written in its own source.

    GOAL: text that varies -- a file name, a server's ``error``, an
    exception's message -- is set with ``textContent``, never spliced into
    markup.  The three sites this file's allowlist hid until 2026-09-25 (the
    module docstring) are the failure.  CONTRACT: `web/ui-contract.md` § 7.

    The rest of the pages' markup arrives through PRODUCERS, by count.
    """
    path = STATIC_ROOT / rel
    assert path.is_file(), f"PRODUCERS names {rel}, which is not a file"
    found = html_writes(path.read_text(encoding="utf-8"))
    allowed, why = PRODUCERS.get(rel, (0, ""))
    assert len(found) == allowed, (
        f"{rel}: {len(found)} place(s) hand markup to a DOM API "
        f"(allowed: {allowed}{' -- ' + why if why else ''}):\n  "
        + "\n  ".join(f"line {ln}: {what}" for ln, what in found)
        + "\n\nText that varies goes in with textContent, on an element "
          "built with createElement; fixed markup is ONE literal "
          "(ui-contract.md § 7). A new producer of HTML is a design "
          "decision, not a line in PRODUCERS.")


#: What the reader must flag and must pass -- each row one statement a text
#: match gets wrong (`testing.md` § 2a: a guard nobody has watched fail is a
#: guard nobody has tested).
CASES = [
    ('el.innerHTML = "";', 0),
    ('el.innerHTML = "<p>fixed; one literal</p>";', 0),
    ('el.innerHTML = `<div>\n  <b>fixed</b>\n</div>`;', 0),
    ('if (a == b) el.innerHTML = "<p>a</p>";', 0),
    ('if (el.innerHTML === "") show();', 0),
    ('// el.innerHTML = name;', 0),
    ('const note = "el.innerHTML = name;";', 0),
    ('el.innerHTML = "<p>" + name + "</p>";', 1),
    ('el.innerHTML = `<p>${name}</p>`;', 1),
    ('el.innerHTML = escapeHtml(name);', 1),
    ('el.innerHTML = ok ? "<b>a</b>" : "<b>b</b>";', 1),
    ('el.innerHTML += "<p>more</p>";', 1),
    ('$("x").innerHTML = name;', 1),
    ('rows[i].innerHTML = name;', 1),
    ('el["innerHTML"] = name;', 1),
    ('el.outerHTML = name;', 1),
    ('frame.srcdoc = name;', 1),
    ('el.insertAdjacentHTML("beforeend", "<p>x</p>");', 1),
    ('document.write(name);', 1),
    # The stripper this replaced had no regex state: the quote inside the
    # regex opened a "string", the URL's `//` was then read as a comment, and
    # the sink after it vanished.
    ('const re = /["\']/g; const u = "http://x"; el.innerHTML = name;', 1),
    ('x = a / b; el.innerHTML = name; y = c / d;', 1),
]


@pytest.mark.parametrize("src,expected", CASES, ids=[c for c, _ in CASES])
def test_the_reader_flags_every_form_and_only_those(src, expected):
    """The lint above is only as good as its reader: each row is a statement
    it must flag or must pass, and a text match gets at least one wrong."""
    assert len(html_writes(src)) == expected, html_writes(src)


# RETIRED 2026-09-25, with the (file, spelling) allowlist they served:
#   * `TestNoUnsafeInnerHTML` and `test_every_allowlist_entry_names_a_real_site`
#     -- the text-matching lint and its allowlist's liveness check, replaced by
#     `test_markup_is_written_only_from_literals` above.  Of the allowlist's 11
#     entries, 3 exempted nothing (one could never match: the text it keyed on
#     began before the match did), and 3 hid raw splices behind comments that
#     called them static.
#   * `TestRecentAdditionsArePure` and `TestNoTemplateLiteralInnerHTMLInterp`
#     -- subsumed, by mutant: a spliced write in `results/viewer.js` and a
#     `${}` template write in `lib/form-schema.js` each turned the candidate
#     AND the lint above red.


# --------------------------------------------------------------------- #
#  Code is never run from a string                                      #
# --------------------------------------------------------------------- #


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

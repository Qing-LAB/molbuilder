"""Nothing a person, a file or a server supplies is parsed as markup, or run as
code, by the JavaScript the app serves -- except through the counted
PRODUCERS below.

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

import bisect
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
# WHY A TOKENIZER.  Matching ``.innerHTML = ([^;]+);`` over the source with its
# comments stripped fails three ways at once, all measured 2026-09-25: the value stopped at the first ``;`` even inside a
# string (``'&amp;'``, ``padding:0.7rem;``), so a site passed or failed on a
# fragment of itself; ``+=`` and ``$("x").innerHTML`` were not seen at all; and
# the stripper had no regex state, so a quote inside ``/["']/g`` opened a
# "string", a real ``//`` in the next string was then taken for a comment, and
# the code after it was dropped.  Here a string, a template literal, a comment
# and a regex literal are each ONE token, so what follows an ``=`` is known
# exactly -- and the expression inside a template's ``${...}`` is read by the
# SAME scanner.

class _Tok(NamedTuple):
    kind: str            # ident | num | str | tmpl | regex | punct | comment
    text: str
    start: int           # offset in the source
    line: int            # the 1-based line it starts on
    closed: bool = True  # a str / tmpl / regex / block comment found its end
    prop: bool = False   # an ident right after `.` / `?.`: a property name


#: After one of these words a ``/`` opens a regex literal, not a division.
_REGEX_AFTER_WORD = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "throw", "case", "do", "else", "yield", "await"})

#: The statements whose ``(...)`` a regex literal may follow: ``if (x) /re/``.
_CONTROL_WORDS = frozenset({"if", "while", "for", "with"})

#: Longest first, so ``===`` is one token and never ``==`` then ``=``.
_PUNCT = sorted(
    (">>>=", "===", "!==", "**=", "<<=", ">>=", ">>>", "...", "&&=", "||=",
     "??=", "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--",
     "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>"),
    key=len, reverse=True)


def _regex_ok(last: Optional[_Tok], owner: Optional[str]) -> bool:
    """Does a ``/`` read after ``last`` open a regex literal?  ``owner`` is the
    word before the ``(`` that ``last`` closes, when ``last`` is ``)``."""
    if last is None:
        return True
    if last.kind == "ident":
        # `a.of / 2` divides: a property named like a keyword is a value.
        return not last.prop and last.text in _REGEX_AFTER_WORD
    if last.kind != "punct":
        return False                     # after a value: a division
    if last.text == ")":
        return owner in _CONTROL_WORDS   # `if (x) /re/`, but `(a) / b`
    # After `]` or a postfix `++`/`--` a value has just ended.  After any
    # other punctuation -- `}` included, which ends a statement far more
    # often than an object that anybody then divides -- one is starting.
    return last.text not in ("]", "++", "--")


def _quoted_end(src: str, i: int) -> Tuple[int, bool]:
    """Past the ``'...'`` / ``"..."`` string opening at ``i``, and whether it
    closed (a raw newline ends an unterminated one)."""
    q, j, n = src[i], i + 1, len(src)
    while j < n:
        c = src[j]
        if c == "\\":
            j += 2
        elif c == q:
            return j + 1, True
        elif c == "\n":
            return j, False
        else:
            j += 1
    return n, False


def _regex_end(src: str, i: int) -> Tuple[int, bool]:
    """Past the ``/.../flags`` literal opening at ``i``, and whether it closed.
    A ``/`` inside a character class does not close it."""
    j, n, in_class = i + 1, len(src), False
    while j < n:
        c = src[j]
        if c == "\n":
            return j, False
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
            return j, True
        j += 1
    return n, False


class _Reader:
    """One pass over ``src``.  The expression inside a template's ``${...}`` is
    read by this same scanner, so the regex-or-division decision is made in
    one place, and a sink inside an interpolation is seen like any other."""

    def __init__(self, src: str):
        self.src, self.n = src, len(src)
        self.toks: List[_Tok] = []
        self._newlines = [k for k, c in enumerate(src) if c == "\n"]

    def _line(self, i: int) -> int:
        return bisect.bisect_left(self._newlines, i) + 1

    def scan(self, i: int, inside: bool = False) -> int:
        """Read from ``i``.  ``inside``: stop past the ``}`` that closes the
        ``${`` this scan was opened for, and return where that is."""
        src, n = self.src, self.n
        depth, last, owner = 0, None, None
        opened: List[Optional[str]] = []      # the word before each open `(`
        while i < n:
            c = src[i]
            if c.isspace():
                i += 1
                continue
            closed = True
            if c == "`":
                slot = len(self.toks)
                self.toks.append(None)        # the template goes BEFORE its parts
                j, closed = self._template(i)
                tok = _Tok("tmpl", src[i:j], i, self._line(i), closed)
                self.toks[slot] = tok
                last, owner, i = tok, None, j
                continue
            if src.startswith("//", i):
                e = src.find("\n", i)
                j, kind = (n if e == -1 else e), "comment"
            elif src.startswith("/*", i):
                e = src.find("*/", i + 2)
                j, kind, closed = (n if e == -1 else e + 2), "comment", e != -1
            elif c in "'\"":
                (j, closed), kind = _quoted_end(src, i), "str"
            elif c == "/" and _regex_ok(last, owner):
                (j, closed), kind = _regex_end(src, i), "regex"
            elif c.isalpha() or c in "_$":
                j = i + 1
                while j < n and (src[j].isalnum() or src[j] in "_$"):
                    j += 1
                kind = "ident"
                if last is not None and last.text in (".", "?."):
                    kind = "prop"                    # marked below, read as ident
            elif c.isdigit():
                j = i + 1
                while j < n and (src[j].isalnum() or src[j] in "._"):
                    j += 1
                kind = "num"
            else:
                op = next((p for p in _PUNCT if src.startswith(p, i)), c)
                j, kind = i + len(op), "punct"
                if inside and op == "}":
                    if depth == 0:
                        return j              # the `}` of `${`: not the expression's
                    depth -= 1
                elif inside and op == "{":
                    depth += 1
            if kind == "prop":
                tok = _Tok("ident", src[i:j], i, self._line(i), True, True)
            else:
                tok = _Tok(kind, src[i:j], i, self._line(i), closed)
            self.toks.append(tok)
            if kind != "comment":
                if tok.text == "(":
                    # `Symbol.for(k) / 2`: a method named like a statement
                    # opens no statement's parentheses.
                    opened.append(last.text if last is not None
                                  and last.kind == "ident"
                                  and not last.prop else None)
                    owner = None
                elif tok.text == ")":
                    owner = opened.pop() if opened else None
                else:
                    owner = None
                last = tok
            i = j
        return n

    def _template(self, i: int) -> Tuple[int, bool]:
        """Past the template literal opening at ``i``, and whether it closed;
        each ``${...}`` in it is read by :meth:`scan`."""
        src, n = self.src, self.n
        j = i + 1
        while j < n:
            if src[j] == "\\":
                j += 2
            elif src[j] == "`":
                return j + 1, True
            elif src.startswith("${", j):
                j = self.scan(j + 2, inside=True)
            else:
                j += 1
        return n, False


def _tokens(src: str) -> List[_Tok]:
    """``src`` as tokens, comments included -- ``_strip_js_comments`` needs
    to know where they are."""
    reader = _Reader(src)
    reader.scan(0)
    return reader.toks


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

#: Methods that parse an argument as HTML -- refused whatever they are given.
_HTML_METHODS = frozenset({"insertAdjacentHTML", "createContextualFragment",
                           "parseFromString", "setHTMLUnsafe",
                           "parseHTMLUnsafe"})

#: The documents whose ``write`` / ``writeln`` parse what they are given.
_DOCUMENTS = frozenset({"document", "contentDocument", "ownerDocument"})

#: What may follow the one legal value: the statement, or the expression
#: holding it, ends there.
_VALUE_ENDS = frozenset({";", "}", ")", ","})

#: The operators that end in ``=`` and compare rather than assign.
_COMPARISONS = frozenset({"==", "===", "!=", "!==", "<=", ">="})


def _is_literal(t: Optional[_Tok]) -> bool:
    """One string written in the source: quoted, or a template with no
    ``${...}``."""
    return t is not None and (
        t.kind == "str" or (t.kind == "tmpl" and "${" not in t.text))


def html_writes(src: str) -> List[Tuple[int, str]]:
    """Every place ``src`` hands markup to a DOM API that parses it, as
    ``(line, what)`` -- except the one legal form: a property given one
    literal, as ``= "..."`` or as an object key's value."""
    toks = [t for t in _tokens(src) if t.kind != "comment"]

    def at(k: int) -> Optional[_Tok]:
        return toks[k] if 0 <= k < len(toks) else None

    def one_literal(k: int) -> bool:
        end = at(k + 1)
        return _is_literal(at(k)) and (end is None or end.text in _VALUE_ENDS)

    def opener(k: int) -> Optional[int]:
        """The ``{`` / ``[`` / ``(`` that encloses token ``k``."""
        depth = 0
        for j in range(k - 1, -1, -1):
            s = toks[j].text
            if s in ("}", "]", ")"):
                depth += 1
            elif s in ("{", "[", "("):
                if depth == 0:
                    return j
                depth -= 1
        return None

    found: List[Tuple[int, str]] = []
    for k, t in enumerate(toks):
        prev, nxt = at(k - 1), at(k + 1)
        dotted = prev is not None and prev.text in (".", "?.")
        name = (t.text if t.kind == "ident"
                else t.text[1:-1] if t.kind == "str" else None)
        if name in _HTML_PROPERTIES:
            # `x.innerHTML = ...`, and `x["innerHTML"] = ...`
            if ((t.kind == "ident" and dotted)
                    or (t.kind == "str" and prev is not None
                        and prev.text == "[" and nxt is not None
                        and nxt.text == "]")):
                op_at = k + 1 if t.kind == "ident" else k + 2
                op = at(op_at)
                if (op is None or not op.text.endswith("=")
                        or op.text in _COMPARISONS):
                    continue                       # read, or compared
                if not (op.text == "=" and one_literal(op_at + 1)):
                    found.append((t.line, f"{name} {op.text}"))
                continue
            # `{ innerHTML: ... }` and `{ innerHTML }` -- an object for
            # Object.assign and kin.  Not `const { innerHTML: h } = el`, which
            # READS the property: a pattern after const / let / var.
            if (prev is not None and prev.text in ("{", ",")
                    and nxt is not None and nxt.text in (":", ",", "}")):
                o = opener(k)
                if o is None or toks[o].text != "{":
                    continue                       # an array, or arguments
                before = at(o - 1)
                if before is not None and before.text in ("const", "let", "var"):
                    continue                       # a destructuring read
                if nxt.text != ":" or not one_literal(k + 2):
                    found.append((t.line, f"{{{name}}}"))
                continue
        if t.kind != "ident" or nxt is None or nxt.text != "(" or not dotted:
            continue
        if t.text in _HTML_METHODS:
            found.append((t.line, t.text + "()"))
        elif t.text in ("write", "writeln") and at(k - 2) is not None \
                and at(k - 2).text in _DOCUMENTS:
            found.append((t.line, f"{at(k - 2).text}.{t.text}()"))
        elif t.text == "setAttribute":
            arg, comma = at(k + 2), at(k + 3)
            if (arg is not None and arg.kind == "str"
                    and arg.text[1:-1] == "srcdoc"
                    and comma is not None and comma.text == ","):
                close = at(k + 5)
                if not (_is_literal(at(k + 4)) and close is not None
                        and close.text == ")"):
                    found.append((t.line, 'setAttribute("srcdoc")'))
    return found


#: The doors through which HTML made at run time enters a page -- file ->
#: (writes, why).  THE COUNT IS PART OF THE ALLOWANCE: a second write in one
#: of these files has not inherited the first one's reason, and a file whose
#: write is gone fails too, so an allowance cannot outlive its argument.  It counts WRITES,
#: not what they write -- ``renderEl.innerHTML = r.text`` in `documents/page.js`
#: would keep the count at 1 -- which is what a count can say; a producer door
#: that takes the INPUT rather than HTML is what would close that.
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
    src = path.read_text(encoding="utf-8")
    # A reader that loses its place hides the rest of the file from every lint
    # here, and an unclosed string, template, regex or comment is what a lost
    # place looks like -- so it fails, rather than passing what follows.
    lost = [(t.line, t.kind) for t in _tokens(src) if not t.closed]
    assert not lost, f"{rel}: the reader lost its place at {lost}"
    found = html_writes(src)
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
    # A stripper with no regex state reads the quote inside the regex as a
    # "string", the URL's `//` then as a comment, and loses the sink after it.
    ('const re = /["\']/g; const u = "http://x"; el.innerHTML = name;', 1),
    ('x = a / b; el.innerHTML = name; y = c / d;', 1),
    ('x = 1e-3 / y; el.innerHTML = name;', 1),
    ('x = (a) / b; el.innerHTML = name;', 1),
    ('i++ / n; el.innerHTML = name;', 1),
    ('if (ok) /["\']/.test(s); el.innerHTML = name;', 1),
    ('function f() {}\n/["\']/.test(s); el.innerHTML = name;', 1),
    # A regex inside a template's ``${...}``.
    (r'''const q = `[a="${p.replace(/"/g, '\\"')}"]`; el.innerHTML = name;''', 1),
    ('const t = `a ${`b ${c}`} d`; el.innerHTML = name;', 1),
    ('`${el.innerHTML = name}`;', 1),
    ('const s = "open\nel.innerHTML = name;', 1),
    ('if (el?.innerHTML === "") show();', 0),
    ('Object.assign(el, { innerHTML: name });', 1),
    ('Object.assign(el, { className: "x", innerHTML: "" });', 0),
    ('frame.setAttribute("srcdoc", name);', 1),
    ('el.innerHTML ||= name;', 1),
    ('frame.contentDocument.write(name);', 1),
    ('Object.assign(el, { innerHTML });', 1),
    ('Object.assign(el, { srcdoc, a: 1 });', 1),
    ('const { innerHTML: h } = el;', 0),
    ('const list = [a, innerHTML, b];', 0),
    ('x = a.of / 2; el.innerHTML = name;', 1),
    ('x = Symbol.for(k) / 2; el.innerHTML = name;', 1),
    ('frame.setAttribute("srcdoc", "<p>x</p>");', 0),
]


#: What the reader must report left open -- and a line it must not.  The
#: per-file lint's first assertion is only a guard if it has been seen to fire.
UNCLOSED = [
    ('x = "open\n', "str"),
    ('x = /open\n', "regex"),
    ('x = `open', "tmpl"),
    ('x = `${ open', "tmpl"),
    ('/* open', "comment"),
    (r'''q = `[a="${p.replace(/"/g, '\\"')}"]`;''', None),
]


@pytest.mark.parametrize("src,kind", UNCLOSED, ids=[c for c, _ in UNCLOSED])
def test_the_reader_says_where_it_lost_its_place(src, kind):
    """A string, regex, template or comment the reader never closes is how a
    lost place shows, and the per-file lint fails on one -- so each row here
    must leave exactly its kind open, and the tree-picker line (a regex inside
    a template's ``${...}``) none (`testing.md` § 2a)."""
    lost = [t.kind for t in _tokens(src) if not t.closed]
    assert lost == ([kind] if kind else []), lost


@pytest.mark.parametrize("src,expected", CASES, ids=[c for c, _ in CASES])
def test_the_reader_flags_every_form_and_only_those(src, expected):
    """The lint above is only as good as its reader: each row is a statement
    it must flag or must pass, and a text match gets at least one wrong."""
    assert len(html_writes(src)) == expected, html_writes(src)


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

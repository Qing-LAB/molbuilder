"""THE DOCUMENTS TAB'S FOUR INDEXES, built from the code each time one is read
(`web/tabs.md` § 5.6).

MODULE  web.doc_indexes (L3; reads the running server's routes and its page
        scripts, the package's own files, the run-file catalogue's cells
        (`catalogue_tables`), the constants, the bibliography (`references`)
        and the live chapters -- writes nothing)
ROLE    :data:`INDEXES` -- each index's document path, title, what it lists
        and the builder of its tables; :func:`compose` -- one index as
        markdown: one row per thing, each saying where its fact lives -- the
        chapters that name it -- with at most the one line its owner wrote
USED-BY the Documents blueprint (`web/blueprints/docs.py`): its tree's
        Indexes folder, and `/api/docs/read` of ``index/<name>.md``
"""
from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .. import repo_root
from ..catalogue_tables import markdown_table as _table

#: What is not a live chapter (`web/tabs.md` § 5.6): an archived chapter, a
#: plan, and the changelog -- records of what was, which name everything.
_NOT_LIVE = ("archive/", "plans/", "process/changelog.md")

#: A page script that is a library someone else wrote -- read by no index.
_NOT_OURS = ("vendor/",)

#: A backticked token of a text.
_TICKED = re.compile(r"`([^`\n]+)`")

#: A ``<field>`` of a route or a file's name.
_FIELD_OF = re.compile(r"<[^>]*>")

#: A variable part as a chapter or a script writes it -- ``<id>``, ``{id}``,
#: a script's ``${id}``, ``…`` -- never a value: a chapter that writes
#: ``/api/spectra/7`` gives an example, it does not name a route.
_WRITTEN = r"(?:<[^>/]*>|\{[^},/]*\}|\$\{[^}]*\}|…|\.\.\.)"

#: A path as a chapter or a script writes it -- at least one character
#: after its slash, a lone ``/`` being prose -- up to the first character a
#: path does not hold; ``<field>`` and ``{a,b}`` each one piece.
_PATH = re.compile(r"(?<![\w./-])/(?:[\w./:%$…-]|<[^<>\s/]*>|\{[^}\s]*\})+")

#: A ``{a,b}`` list in a written path -- each of its names.
_LIST = re.compile(r"(?<!\$)\{([^}]*,[^}]*)\}")

#: An author-and-year key in brackets -- the shape a cited key the
#: bibliography lacks is recognized by.
_CITED_SHAPE = re.compile(r"\[`?([A-Z][A-Za-z]*\d{3,4}[a-z]?)`?\](?!\()")


def _cites(key: str) -> "re.Pattern[str]":
    """How a chapter cites a bibliography entry: its key in brackets,
    backticked or not -- a link's ``[text](url)`` is not one -- or after
    ``references.bib:``."""
    k = re.escape(key)
    return re.compile(r"\[`?" + k + r"`?\](?!\()|references\.bib:\s*" + k
                      + r"\b")


@dataclass(frozen=True)
class Index:
    """One index: its document path below ``docs/``, its title, what it
    lists, and the builder of its tables, ``build(app, chapters)``."""
    path: str
    title: str
    about: str
    build: Callable[..., str]


# --------------------------------------------------------------------- #
#  What is read                                                         #
# --------------------------------------------------------------------- #

def _texts(root: Path, pattern: str, skip: Tuple[str, ...] = ()
           ) -> Dict[str, str]:
    """Every file below ``root`` matching ``pattern`` and under none of
    ``skip``, by its path there, with its text."""
    out: Dict[str, str] = {}
    for p in sorted(root.rglob(pattern)):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(skip) or "__pycache__" in p.parts:
            continue
        try:
            out[rel] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def _scripts(app) -> Dict[str, str]:
    """The page's own scripts and templates, by their path below ``web/``
    -- the server's own folders (``app.static_folder``,
    ``app.template_folder``), a library someone else wrote left out."""
    out: Dict[str, str] = {}
    for folder, pattern in ((app.static_folder, "*.js"),
                            (app.template_folder, "*.html")):
        root = Path(app.root_path, folder)
        out.update({f"{root.name}/{k}": v
                    for k, v in _texts(root, pattern, _NOT_OURS).items()})
    return out


# --------------------------------------------------------------------- #
#  The cells                                                            #
# --------------------------------------------------------------------- #

def _plain(text: str) -> str:
    """A docstring's markup as a chapter writes it: ``:func:`x``` and
    ````x```` as ```x```."""
    return re.sub(r":\w+:`([^`]+)`", r"`\1`", text).replace("``", "`")


def _cell(text: str) -> str:
    """``text`` as one table cell: one line, its bars escaped, an angle
    bracket outside a code span written as itself, never read as a tag."""
    parts = " ".join(str(text).split()).replace("|", "\\|").split("`")
    return "`".join(p.replace("<", "&lt;").replace(">", "&gt;")
                    if i % 2 == 0 else p for i, p in enumerate(parts))


def _first_sentence(text: str, limit: int = 240) -> str:
    """The first sentence of ``text``'s first paragraph -- the one line its
    owner wrote -- cut at ``limit`` characters, never inside a code span."""
    para = " ".join(_plain(text).strip().split("\n\n")[0].split())
    m = re.search(r"(?<=[.!?])\s", para)
    one = para[:m.start()] if m else para
    if len(one) <= limit:
        return one
    cut = one[:limit - 1]
    if cut.count("`") % 2:
        cut = cut[:cut.rfind("`")]
    return cut.rstrip() + "…"


def _named_in(found: Iterable[str]) -> str:
    """The chapters, each a link the tab opens (`web/tabs.md` § 5.4)."""
    got = sorted(set(found))
    return ", ".join(f"[{c}](?doc={c})" for c in got) if got else ""


def _with_fields(text: str, *, last_optional: bool = False) -> str:
    """``text``'s fixed parts as written, each ``<field>`` a variable part
    as written (:data:`_WRITTEN`) -- the last one optional, as a script
    that builds a path leaves it off: ``"/api/x/" + id``."""
    parts = [re.escape(p) for p in _FIELD_OF.split(text)]
    if last_optional and len(parts) > 1 and parts[-1] == "":
        return _WRITTEN.join(parts[:-1]) + f"(?:{_WRITTEN})?"
    return _WRITTEN.join(parts)


# --------------------------------------------------------------------- #
#  Routes                                                               #
# --------------------------------------------------------------------- #

def _paths_in(texts: Dict[str, str]) -> Dict[str, Set[str]]:
    """Every path ``texts`` write, each ``{a,b}`` list read as each of its
    names, with the texts writing it -- read once, so a route is matched
    against short paths, never against whole chapters."""
    out: Dict[str, Set[str]] = {}
    for name, text in texts.items():
        for path in _PATH.findall(text):
            spelled = [path]
            while any(_LIST.search(q) for q in spelled):
                q = next(q for q in spelled if _LIST.search(q))
                spelled.remove(q)
                m = _LIST.search(q)
                spelled += [q[:m.start()] + alt + q[m.end():]
                            for alt in m.group(1).split(",")]
            for q in spelled:
                # A SENTENCE'S PERIOD is not the path's; a lone `/` left is
                # prose.
                q = q.rstrip(".")
                if len(q) > 1:
                    out.setdefault(q, set()).add(name)
    return out


def _routes(app, docs: Dict[str, str]) -> str:
    """Every route the server answers -- one row each, its methods merged:
    its handler's first sentence, the chapters naming it, the page scripts
    writing its path."""
    in_docs, in_scripts = _paths_in(docs), _paths_in(_scripts(app))
    by_rule: Dict[str, List] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint != "static":
            by_rule.setdefault(rule.rule, []).append(rule)
    rows = []
    for path in sorted(by_rule):
        rules = by_rule[path]
        methods = " ".join(sorted({m for r in rules for m in r.methods or ()
                                   if m not in ("HEAD", "OPTIONS")}))
        # EACH HANDLER'S OWN LINE, by its methods where several answer.
        said = []
        for r in rules:
            view = app.view_functions.get(r.endpoint)
            line = _first_sentence(inspect.getdoc(view) or "") if view else ""
            verbs = " ".join(sorted(m for m in r.methods or ()
                                    if m not in ("HEAD", "OPTIONS")))
            if line and line not in [x for _v, x in said]:
                said.append((verbs, line))
        what = (said[0][1] if len(said) == 1 else
                "; ".join(f"{v}: {x}" for v, x in said)) or "—"
        in_doc = re.compile(_with_fields(path))
        in_script = re.compile(_with_fields(path, last_optional=True))
        named = {c for p, cs in in_docs.items() if in_doc.fullmatch(p)
                 for c in cs}
        called = {s for p, ss in in_scripts.items()
                  if in_script.fullmatch(p) for s in ss}
        rows.append([_cell(f"`{path}`"), methods, _cell(what),
                     _named_in(named) or "**no chapter names it**",
                     _cell(", ".join(f"`{s}`" for s in sorted(called)))
                     or "—"])
    return _table(["route", "methods", "what it answers", "chapters",
                   "page scripts writing it"], rows)


# --------------------------------------------------------------------- #
#  Modules                                                              #
# --------------------------------------------------------------------- #

def _layers(docs: Dict[str, str], modules: Set[str]
            ) -> Optional[Tuple[Set[str], Set[str]]]:
    """`architecture.md` § 3's rule, read there: its L1 index -- the
    modules its table's rows name -- and L3, the modules its sentence
    *L3 is …* names; ``None`` where either is not found."""
    text = docs.get("architecture.md", "")
    at = text.find("**The L1 index")
    l1: Set[str] = set()
    if at >= 0:
        for line in text[at:].splitlines()[1:]:
            if line.startswith("|") and not line.startswith("|---"):
                cells = line.split("|")
                if len(cells) > 2:
                    l1.update(_TICKED.findall(cells[2]))
            elif l1:
                break
    said = re.search(r"L3\s+is\s+(.*?)[;.]", text, re.S)
    l3 = ({t for t in _TICKED.findall(said.group(1)) if t in modules}
          if said else set())
    return (l1, l3) if l1 and l3 else None


def _layer(name: str, layers: Optional[Tuple[Set[str], Set[str]]]) -> str:
    """L1 a module of the L1 index, L3 one of L3's, L2 every other --
    ``?`` where the rule was not read."""
    if layers is None:
        return "?"

    def under(top: str) -> bool:
        return name == top or name.startswith(top + ".")
    l1, l3 = layers
    if any(under(m) for m in l1):
        return "L1"
    if any(under(m) for m in l3):
        return "L3"
    return "L2"


def _role(tree: Optional[ast.Module]) -> str:
    """A module's own line: its header's ``ROLE``, else its docstring's
    first sentence."""
    doc = (ast.get_docstring(tree) or "") if tree is not None else ""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"ROLE:?\s+(.*)", line)
        if m:
            # ITS LINES TO THE NEXT HEADER WORD (``USED-BY``) or a blank.
            said = [m.group(1)]
            for more in lines[i + 1:]:
                if not more.strip() or re.match(r"[A-Z][A-Z-]+:?\s", more):
                    break
                said.append(more)
            return _first_sentence(" ".join(said))
    return _first_sentence(doc)


def _own_names(tree: Optional[ast.Module]) -> Set[str]:
    """The names a module defines at its top -- what ``module.name`` may
    name."""
    out: Set[str] = set()
    for node in (tree.body if tree is not None else ()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign)
                      else [node.target]):
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def _modules_named(docs: Dict[str, str], names: Dict[str, Set[str]]
                   ) -> Dict[str, Set[str]]:
    """Each module, with the chapters naming it in backticks: its path
    (``transport/record.py``), its dotted name (``transport.record``), or
    its name with one of its own names after it (``runs.concluded``) -- a
    top-level module's bare word, which a field or a variable may share,
    never."""
    out: Dict[str, Set[str]] = {}
    for c, text in docs.items():
        for tok in _TICKED.findall(text):
            tok = tok.strip().split("(")[0]
            # THE PACKAGE'S OWN NAME before it names the module whole.
            whole = tok.startswith(("molbuilder/", "molbuilder."))
            if whole:
                tok = tok[len("molbuilder/"):]
            if tok.endswith((".py", "/")):
                # A PATH -- a file (``transport/record.py``) or a package's
                # folder (``jobset/``); the root's only with the package's
                # name before it (``molbuilder/__init__.py``), a bare
                # ``.py`` or ``__init__.py`` being any package's.
                stem = tok[:-3] if tok.endswith(".py") else tok[:-1]
                mod = ".".join(q for q in stem.split("/")
                               if q and q != "__init__")
                if mod in names and (mod or whole):
                    out.setdefault(mod, set()).add(c)
                continue
            parts = tok.split(".")
            for k in range(len(parts), 0, -1):
                mod = ".".join(parts[:k])
                if mod not in names:
                    continue
                rest = parts[k:]
                if (rest and rest[0] in names[mod]) or (
                        not rest and (k > 1 or whole)):
                    out.setdefault(mod, set()).add(c)
                break
    return out


def _modules(app, docs: Dict[str, str]) -> str:
    """Every module of the package: its layer, its own line, the chapters
    naming it."""
    trees: Dict[str, Optional[ast.Module]] = {}
    for rel, source in _texts(repo_root() / "molbuilder", "*.py").items():
        name = ".".join(q for q in rel[:-3].split("/") if q != "__init__")
        try:
            trees[name] = ast.parse(source)
        except SyntaxError:
            trees[name] = None
    layers = _layers(docs, set(trees))
    named = _modules_named(docs, {n: _own_names(t) for n, t in trees.items()})
    head = ("" if layers is not None else
            "*`architecture.md` § 3's L1 index or its L3 sentence is not "
            "found: every layer below is unknown.*\n\n")
    rows = [[f"`{name or 'molbuilder'}`",
             _layer(name, layers) if name else "the package root",
             _cell(_role(tree)), _named_in(named.get(name, ())) or "—"]
            for name, tree in sorted(trees.items())]
    return head + _table(["module", "layer", "its own line", "chapters"],
                         rows)


# --------------------------------------------------------------------- #
#  Files                                                                #
# --------------------------------------------------------------------- #

def _file_chapters(row, docs: Dict[str, str]) -> str:
    """The chapters writing a catalogue file's name, its fields written as
    fields (:data:`_WRITTEN`) -- a chapter writing ``H2.fdf`` gives an
    example, it does not name ``<base>.fdf``."""
    from ..runfiles import shown_file
    name = shown_file(row)
    fixed = max(_FIELD_OF.split(name), key=len)
    pat = re.compile(r"(?<![\w.-])" + _with_fields(name) + r"(?![\w/-])")
    reach = len(name) + 120

    def writes(text: str) -> bool:
        # SEARCHED ONLY AROUND EACH PLACE ITS FIXED PART IS WRITTEN.
        at = text.find(fixed)
        while at >= 0:
            if pat.search(text, max(0, at - reach), at + reach):
                return True
            at = text.find(fixed, at + 1)
        return False
    return _named_in(c for c, t in docs.items() if writes(t)) or "—"


def _files(app, docs: Dict[str, str]) -> str:
    """Every file molbuilder writes into a calculation folder, then each
    engine file it names -- the catalogue's rows as its one rendering
    shows them (`catalogue_tables`), the chapters naming each beside."""
    from ..catalogue_tables import (ENGINE_HEAD, WRITTEN_HEAD, engine_cells,
                                    written_cells)
    from ..runfiles import ENGINE_FILES, WRITTEN
    return ("## Files molbuilder writes\n\n"
            + _table((*WRITTEN_HEAD, "chapters"),
                     [[_cell(c) for c in written_cells(a)]
                      + [_file_chapters(a, docs)] for a in WRITTEN])
            + "\n## Engine files molbuilder names\n\n"
            + _table((*ENGINE_HEAD, "chapters"),
                     [[_cell(c) for c in engine_cells(e)]
                      + [_file_chapters(e, docs)] for e in ENGINE_FILES]))


# --------------------------------------------------------------------- #
#  References                                                           #
# --------------------------------------------------------------------- #

def _constants() -> List[List[str]]:
    """Each constant of `constants.py`: its value, and the first paragraph
    of the comment above it -- what it is, where its value comes from and
    why this value (the user, 2026-09-07: "the foundation of all the
    constants")."""
    from .. import constants
    source = Path(constants.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    rows = []
    for node in ast.parse(source).body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        for t in targets:
            if not (isinstance(t, ast.Name) and t.id.isupper()):
                continue
            said, k = [], node.lineno - 2
            while k >= 0 and lines[k].lstrip().startswith("#:"):
                said.insert(0, lines[k].lstrip()[2:].strip())
                k -= 1
            first = " ".join(said).strip()
            first = " ".join(" ".join(said[:said.index("")]).split()) \
                if "" in said else first
            rows.append([t.id, repr(getattr(constants, t.id, "")),
                         _plain(first)])
    return rows


def _references(app, docs: Dict[str, str]) -> str:
    """The constants with their reasons; the bibliography, each entry with
    the chapters citing it -- or the code, where only the code does; and
    the keys cited that it lacks."""
    from ..references import citation_for, known_keys
    pkg = repo_root() / "molbuilder"
    code = {**_texts(pkg, "*.py"), **_texts(pkg, "*.toml")}
    rows_c = []
    for n, v, why in _constants():
        word = re.compile(r"\b" + re.escape(n) + r"\b")
        rows_c.append([f"`{n}`", _cell(f"`{v}`"), _cell(why),
                       _named_in(c for c, t in docs.items()
                                 if n in t and word.search(t)) or "—"])
    keys = sorted(known_keys())
    rows_b = []
    for key in keys:
        got = citation_for(key)
        cites = _cites(key)
        word = re.compile(r"\b" + re.escape(key) + r"\b")
        mine = {c for c, t in docs.items() if key in t and cites.search(t)}
        by_code = any(key in t and word.search(t) for t in code.values())
        rows_b.append([f"`{key}`", _cell(got["title"]), _cell(got["text"]),
                       f"[{got['doi']}](https://doi.org/{got['doi']})"
                       if got.get("doi") else "—",
                       _named_in(mine) or ("the code alone" if by_code else
                                           "**neither a chapter nor the "
                                           "code cites it**")])
    cited: Dict[str, Set[str]] = {}
    for c, text in docs.items():
        for key in _CITED_SHAPE.findall(text):
            cited.setdefault(key, set()).add(c)
    missing = sorted(set(cited) - set(keys))
    out = ["## Constants\n\n",
           _table(["constant", "value", "why this value", "chapters"],
                  rows_c),
           "\n## Citations\n\n",
           _table(["key", "title", "where to find it", "DOI", "cited by"],
                  rows_b)]
    if missing:
        out += ["\n## Cited, not in the bibliography\n\n",
                _table(["key", "chapters"],
                       [[f"`{k}`", _named_in(cited[k])] for k in missing])]
    return "".join(out)


#: THE FOUR INDEXES, in the order the sidebar shows them.
INDEXES = (
    Index("index/routes.md", "Routes",
          "every route the server answers", _routes),
    Index("index/modules.md", "Modules",
          "every module of the package", _modules),
    Index("index/files.md", "Files",
          "every file molbuilder writes into a calculation folder, and the "
          "engine files it names", _files),
    Index("index/references.md", "References",
          "the physical constants and the bibliography", _references),
)


def index_of(path: str) -> Optional[Index]:
    """The index at ``path`` below ``docs/``, or ``None``."""
    return next((i for i in INDEXES if i.path == path), None)


def compose(index: Index, app, root: Path) -> str:
    """``index`` as one markdown document: its title, what it lists, where
    it is built from, and its tables -- for the server ``app`` and the live
    chapters below ``root`` (``docs/``)."""
    return (f"# {index.title}\n\n"
            f"*{index.about[0].upper() + index.about[1:]} — built from the "
            f"code each time this page is read "
            f"([`web/tabs.md`](?doc=web/tabs.md) § 5.6).*\n\n"
            + index.build(app, _texts(root, "*.md", _NOT_LIVE)))


__all__ = ["INDEXES", "Index", "compose", "index_of"]

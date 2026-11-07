"""The directory's answers still typed in `tests/data/presenters.toml`: each
field a typed answer carries is one the server could give for its file.

`test_inspector_registry_e2e.py` asks the live page's registry each row with
the answer `/api/results/dir` gives about a file (`runs.folder_answer`,
`web/presenters.md` § 2) -- the route's own, on a real run, for every row but
those whose file no session run makes, which carry a typed ``meta``.  A
typed field no folder could produce asks the page a question it is never
asked, and its verdict means nothing.  A field the route gives and a row
leaves out is not asked for: the two transport rows carry ``role``,
``parser`` and ``engine``, and no ``label``, ``stage``, ``engine_words`` or
``noun``.

THE SILENT FAILURE: a role the catalogue dropped, a parser renamed, an
engine's words or a calculation's noun changed in their tables -- the row
still passes in the browser, on an answer the server no longer gives; or a
typed role that is not the one the file's name carries -- a sweep record
typed with its parser and no role, where the route answers it with both.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PRESENTERS = tomllib.loads(
    (Path(__file__).parent / "data" / "presenters.toml").read_text())


def test_each_field_a_typed_answer_carries_is_one_the_server_gives():
    """A typed answer stands for a file a run of ours writes in its
    calculation's folder, where the route reads the file's name back
    (`runs.file_row`, `runfiles.parse`): its role is the one its name
    carries (`runfiles.role_of`), left out only where the name carries
    none; its parser one the registry holds; its noun, when typed, one the
    kind table holds; its engine's words, when typed, the engine table's for
    the engine it names."""
    from molbuilder.engine_table import engine, engines
    from molbuilder.kind_table import kind, kinds
    from molbuilder.parse.registry import _FILE_PARSERS
    from molbuilder.runfiles import role_of
    parsers = {k.name for k in _FILE_PARSERS}
    nouns = {kind(k).noun for k in kinds()}
    wrong = []
    for tbl in ("pick", "result", "heading"):
        for r in PRESENTERS[tbl]:
            if "meta" not in r:
                continue
            meta = r["meta"]
            if (meta.get("role") != role_of(Path(r["file"]).name)
                    or meta.get("parser") not in parsers
                    or meta.get("noun") not in (None, *nouns)
                    or (meta.get("engine_words") is not None
                        and not (meta.get("engine") in engines()
                                 and engine(meta["engine"]).words
                                 == meta["engine_words"]))):
                wrong.append((tbl, r["name"], meta))
    assert not wrong, wrong

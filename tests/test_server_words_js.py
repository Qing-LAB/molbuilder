"""The server's words, drawn -- one renderer (`web/ui-contract.md` § 12.1).

A sentence molbuilder writes once, for the terminal and the page alike, is
drawn on a page by `codeWords` (`lib/dom.js`, loaded once by the shared
header): a command kept in backticks drawn as code, as typed; the terminal's
dash, ` -- `, drawn as a dash; and what it draws is what `wordsText` says it
draws, which the status line compares before it redraws.  The input is the
catalogue's own help for a setting, as the forms are served it.

THE SILENT FAILURES: through 2.69.3 every page showed the terminal's ` -- `
as two hyphens; and a renderer that turned every `--` into a dash would
break the flag of a command a person copies (`prep --cpus-per-task N`).
"""
from __future__ import annotations

import json
from pathlib import Path

from tests._node_esm import run_node

_WORDS = (Path(__file__).resolve().parents[1]
          / "molbuilder" / "web" / "static" / "lib" / "dom.js")

#: A document with what the renderer builds: elements, text and fragments.
_PAGE = """
const node = (kind, text) => ({ kind, textContent: text });
globalThis.document = {
  createElement: (tag) => ({ kind: tag, className: "", textContent: "" }),
  createTextNode: (t) => node("text", String(t)),
  createDocumentFragment: () => ({ children: [],
    appendChild(c) { this.children.push(c); return c; } }),
};
"""


def test_a_server_sentence_is_drawn_with_its_code_and_its_dashes():
    from molbuilder import template as T
    help_ = next(i.help for i in T.catalogue().items if i.name == "omp_threads")
    assert " -- " in help_ and "``prep --cpus-per-task N``" in help_, help_
    out = run_node([_WORDS], f"""
        const dom = window.molbuilder.dom;
        const said = {json.dumps(help_)};
        const drawn = dom.codeWords(said).children;
        console.log(JSON.stringify({{
            parts: drawn.map((n) => [n.kind, n.textContent]),
            text: dom.wordsText(said),
        }}));
    """, globals_js=_PAGE)
    code = [t for kind, t in out["parts"] if kind == "code"]
    words = "".join(t for kind, t in out["parts"] if kind == "text")
    assert "prep --cpus-per-task N" in code, out["parts"]
    assert " -- " not in words and words.count(" — ") == help_.count(" -- "), words
    assert out["text"] == "".join(t for _, t in out["parts"]), out

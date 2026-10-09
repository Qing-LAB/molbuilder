# Third-party browser assets and notices

This directory contains browser code served by molbuilder. The project itself is
BSD 3-Clause; that license does not replace the licenses below. Every bundled
asset keeps its upstream notice, and the wheel package-data rules include this
directory and its subdirectories.

## Inventory

Each row: the files, the release they are, where that release comes from, and
the license text that ships beside them. A file under this directory that no
row covers is refused by `tests/test_vendor_notices.py`.

| Component | Files | Version | Upstream | License and notice |
|---|---|---|---|---|
| 3Dmol.js | `3Dmol-min.js` | 2.5.2 | https://github.com/3dmol/3Dmol.js (npm `3dmol`) | BSD-3-Clause. [`LICENSE-3Dmol.txt`](LICENSE-3Dmol.txt) includes its GLmol, Three.js, and jQuery attributions; `3Dmol-min.js.LICENSE.txt` is the bundle sidecar. |
| gif.js | `gif.min.js`, `gif.worker.min.js` | 0.2.0 | https://github.com/jnordberg/gif.js | MIT. [`gif.min.js.LICENSE.txt`](gif.min.js.LICENSE.txt) contains the complete notice for both files. |
| CodeMirror | `codemirror/*` — core, the dialog/search/jump addons, and **eight language modes** (see below) | 5.65.16 | npm `codemirror@5.65.16`, minified by jsDelivr (each file's header names its original); https://codemirror.net/5/ | MIT. [`codemirror/LICENSE`](codemirror/LICENSE), which covers every file in that directory; the minified builds carry no per-file header, upstream's own included. |
| DOMPurify | `dompurify/purify.min.js` | 3.0.6 | https://github.com/cure53/DOMPurify (npm `dompurify`) | Apache-2.0 OR MPL-2.0. The complete dual-license text is in [`dompurify/LICENSE`](dompurify/LICENSE). |
| GitGraph | `gitgraph/gitgraph.umd.js` | 1.4.0 | npm `@gitgraph/js@1.4.0`, `lib/gitgraph.umd.js` — byte-identical, checked by SHA-256 on 2026-10-09; https://github.com/nicoespeon/gitgraph.js | MIT. [`gitgraph/LICENSE`](gitgraph/LICENSE). |
| KaTeX | `katex/katex.min.js`, `katex/katex.min.css`, `katex/fonts/*.woff2` | 0.16.22 | npm `katex@0.16.22`, its tarball checked against the registry's integrity hash (`sha512-XCHRdUw4…16zYg==`) on 2026-10-09; https://katex.org | MIT. [`katex/LICENSE`](katex/LICENSE). Only the WOFF2 fonts ship: the stylesheet lists WOFF and TTF after each, which a browser fetches only when it cannot read WOFF2. |
| Marked | `marked/marked.min.js` | 4.3.0 | https://github.com/markedjs/marked (npm `marked`) | MIT. [`marked/LICENSE`](marked/LICENSE). |
| Mermaid | `mermaid/mermaid.min.js` | 10.9.6 | https://github.com/mermaid-js/mermaid (npm `mermaid`) | MIT. [`mermaid/LICENSE`](mermaid/LICENSE). |
| Plotly.js | served by `/vendor/plotly.min.js` | plotly.js 3.7.0 in plotly Python 6.9.0 | https://github.com/plotly/plotly.js | MIT. The route serves the installed Python package resource; [`LICENSE-plotly.txt`](LICENSE-plotly.txt) preserves the current bundle notice. |

## Citations

How to cite each component, its copyright holders as its license names them.
3Dmol.js asks for a paper; the others are cited as software, at the release
shipped here.

- **3Dmol.js** — the citation requested by upstream:
  > Rego, N. & Koes, D. (2015). 3Dmol.js: molecular visualization with WebGL.
  > *Bioinformatics*, **31**(8), 1322-1324.
  > https://doi.org/10.1093/bioinformatics/btu829
- **gif.js** — Johan Nordberg. *gif.js*, version 0.2.0. https://github.com/jnordberg/gif.js
- **CodeMirror** — Marijn Haverbeke and others. *CodeMirror*, version 5.65.16. https://codemirror.net/5/
- **DOMPurify** — Mario Heiderich, Cure53, and other contributors. *DOMPurify*, version 3.0.6. https://github.com/cure53/DOMPurify
- **GitGraph** — Nicolas Carlo and Fabien Bernard. *GitGraph.js* (`@gitgraph/js`), version 1.4.0. https://github.com/nicoespeon/gitgraph.js
- **KaTeX** — Khan Academy and other contributors. *KaTeX*, version 0.16.22. https://katex.org
- **Marked** — Christopher Jeffrey and the MarkedJS contributors. *Marked*, version 4.3.0. https://github.com/markedjs/marked
- **Mermaid** — Knut Sveidqvist and contributors. *Mermaid*, version 10.9.6. https://github.com/mermaid-js/mermaid
- **Plotly.js** — Plotly Technologies Inc. *Plotly.js*, version 3.7.0. https://github.com/plotly/plotly.js

### CodeMirror language modes

Added 2026-08-16, all from the same 5.65.16 release. Highlighting is chosen
**from the file suffix**, and a mode file is fetched only when a file of that
kind is first opened — the map and the loader are
`static/lib/codemirror-load.js`, and both the projects-sidebar preview modal and
the Task setup editor read it, so there is one answer to "how is this file
highlighted".

| Mode file | Suffixes it serves |
|---|---|
| `javascript.min.js` | `.json` (as the JSON dialect — CodeMirror ships no separate json mode, so the spec is `{name: "javascript", json: true}`), `.js` |
| `python.min.js` | `.py` |
| `toml.min.js` | `.toml` — including `<label>.template.toml` |
| `shell.min.js` | `.sh`, `.bash`, `.sbatch` — so `.run.sh` wrappers highlight |
| `markdown.min.js` | `.md`, `.markdown` |
| `xml.min.js` | `.xml`, **and markdown requires it** (`require("../xml/xml")` in its module head). It was missing until 2026-08-16, so the markdown mode had been loading without its dependency |
| `css.min.js` | `.css` |
| `yaml.min.js` | `.yaml`, `.yml` |

**molbuilder's own formats get plain text on purpose** — `.fdf`, `.xyz`,
`.out`, `.log`, `.molwatch.log`, `.STRUCT_OUT`. CodeMirror has no upstream mode
for any of them, and asking for one it lacks yields plain text anyway, with a
misleading line of code left behind. `mode: null` is a real mode: line numbers,
editing, undo and the search addons all work.

## Updating a browser dependency

1. Download the upstream release and its complete license or notice text.
2. Replace the asset and notice together; preserve upstream copyright lines.
   A file type the repository's `.gitattributes` does not yet name as binary
   (a font format, an image) gets its `binary` line there; minified `.js` and
   `.css` anywhere under this directory are already kept out of diffs.
3. Update the inventory version and source information above. For a bundled
   dependency, record its release rather than relying only on a minified file.
4. Add or update the component's line under *Citations*, then run
   `tests/test_vendor_notices.py` and build a wheel to confirm the files ship.

The project deliberately serves these assets locally for offline use and a
strict Content Security Policy. Do not replace them with CDN references without
reviewing the security and notice implications.

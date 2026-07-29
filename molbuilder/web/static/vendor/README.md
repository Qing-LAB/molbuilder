# Third-party browser assets and notices

This directory contains browser code served by molbuilder. The project itself is
BSD 3-Clause; that license does not replace the licenses below. Every bundled
asset keeps its upstream notice, and the wheel package-data rules include this
directory and its subdirectories.

## Inventory

| Component | Files | Version | License and notice |
|---|---|---|---|
| 3Dmol.js | `3Dmol-min.js` | 2.5.2 | BSD-3-Clause. [`LICENSE-3Dmol.txt`](LICENSE-3Dmol.txt) includes its GLmol, Three.js, and jQuery attributions; `3Dmol-min.js.LICENSE.txt` is the bundle sidecar. |
| gif.js | `gif.min.js`, `gif.worker.min.js` | 0.2.0 | MIT. [`gif.min.js.LICENSE.txt`](gif.min.js.LICENSE.txt) contains the complete notice for both files. |
| CodeMirror | `codemirror/*` | 5.65.16 | MIT. [`codemirror/LICENSE`](codemirror/LICENSE). |
| DOMPurify | `dompurify/purify.min.js` | 3.0.6 | Apache-2.0 OR MPL-2.0. The complete dual-license text is in [`dompurify/LICENSE`](dompurify/LICENSE). |
| GitGraph | `gitgraph/gitgraph.umd.js` | unrecorded UMD build | MIT. [`gitgraph/LICENSE`](gitgraph/LICENSE). Record the upstream release when this bundle is next replaced. |
| Marked | `marked/marked.min.js` | 4.3.0 | MIT. [`marked/LICENSE`](marked/LICENSE). |
| Mermaid | `mermaid/mermaid.min.js` | 10.9.6 | MIT. [`mermaid/LICENSE`](mermaid/LICENSE). |
| Plotly.js | served by `/vendor/plotly.min.js` | plotly.js 3.7.0 in plotly Python 6.9.0 | MIT. The route serves the installed Python package resource; [`LICENSE-plotly.txt`](LICENSE-plotly.txt) preserves the current bundle notice. |

3Dmol.js citation requested by upstream:

> Rego, N. & Koes, D. (2015). 3Dmol.js: molecular visualization with WebGL.
> *Bioinformatics*, **31**(8), 1322-1324.
> https://doi.org/10.1093/bioinformatics/btu829

## Updating a browser dependency

1. Download the upstream release and its complete license or notice text.
2. Replace the asset and notice together; preserve upstream copyright lines.
3. Update the inventory version and source information above. For a bundled
   dependency, record its release rather than relying only on a minified file.
4. Run the vendor-notice test and build a wheel to confirm the files ship.

The project deliberately serves these assets locally for offline use and a
strict Content Security Policy. Do not replace them with CDN references without
reviewing the security and notice implications.

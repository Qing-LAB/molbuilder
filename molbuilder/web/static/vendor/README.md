# Vendored third-party JavaScript

Files in this directory are unmodified copies of upstream releases,
served locally by molbuilder so the web UI works offline + over
strict CSP (no `script-src` exemption for third-party CDNs).

When you upgrade a vendored file, also update its entry here.

---

## 3Dmol.js

| Field | Value |
|---|---|
| File | `3Dmol-min.js` (485 223 bytes) |
| Version | 2.5.2 |
| Upstream | https://github.com/3dmol/3Dmol.js |
| Distribution | https://3dmol.csb.pitt.edu/build/3Dmol-min.js |
| License | BSD-3-Clause — see `LICENSE-3Dmol.txt` (includes attributions for GLmol, Three.js, and jQuery code incorporated into the bundle) |
| Bundle banner sidecar | `3Dmol-min.js.LICENSE.txt` (the file the bundle's leading comment refers to) |

Citation (per upstream's request):

> Rego, N. & Koes, D. (2015). 3Dmol.js: molecular visualization with
> WebGL. *Bioinformatics*, **31**(8), 1322–1324.
> https://doi.org/10.1093/bioinformatics/btu829

### Upgrade procedure

```bash
curl -fsSL https://3dmol.csb.pitt.edu/build/3Dmol-min.js \
     -o molbuilder/web/static/vendor/3Dmol-min.js
curl -fsSL https://raw.githubusercontent.com/3dmol/3Dmol.js/master/LICENSE \
     -o molbuilder/web/static/vendor/LICENSE-3Dmol.txt
```

Then update the version + size in this README and re-run the
spectra / modify / watch tabs to confirm the molecular viewer still
renders.

---

## gif.js

| Field | Value |
|---|---|
| Files | `gif.min.js` (13 451 bytes), `gif.worker.min.js` (16 636 bytes) |
| Version | 0.2.0 |
| Upstream | https://github.com/jnordberg/gif.js |
| Distribution | https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/ |
| License | MIT — see banner in `gif.min.js.LICENSE.txt` |

Powers the GIF format of the embedded viewer's `handle.exportAnimation`
(see `docs/protocols/embedded-viewer.md` § 3.2). Loaded lazily on first
use via `/static/vendor/gif.min.js`; the main library spawns a Web
Worker that loads `/static/vendor/gif.worker.min.js` to do the LZW
encoding off the main thread. Both files must be present together
or the lazy-load surfaces `no_gif_encoder` per § 5.2.

### Upgrade procedure

```bash
cd molbuilder/web/static/vendor
curl -fsSL https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js \
     -o gif.min.js
curl -fsSL https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js \
     -o gif.worker.min.js
```

The cdnjs distribution names mirror the upstream `dist/` directory
exactly; the `.min` suffix is our local convention indicating that
these are production-ready bundles (gif.js v0.2.0 is already
minified by upstream).

---

## plotly.min.js

Plotly is **not** vendored as a file in this directory. The route
`/vendor/plotly.min.js` (defined in `molbuilder/web/app.py`) serves
the bundle directly from the installed `plotly` Python package's
`package_data/plotly.min.js` resource. The license travels with the
Python package (MIT for plotly.js itself; the Python package adds
the plotly Python wrapper under MIT as well).

This indirection means an upgrade of the `plotly` Python package
automatically picks up the new JS bundle — no manual file copy step
in this directory.

"""Shared Node harness for the MolView JS unit tests — ES-module capable.

The MolView ESM migration converts ~20 classic-IIFE global-mount files into native ES modules,
one layer at a time. During the transition the loaded set is MIXED (some classic, some ESM), so
the test harness must load BOTH:

  * a classic IIFE file ``(function(root){ ... root.molbuilder.X = api })(window)`` — running it
    sets its ``window.molbuilder.*`` global as a side effect; it has no ``export``.
  * a native ES module ``export function ...; window.molbuilder.X = api`` — running it publishes
    the same global (transitional side effect) AND exposes exports.

A dynamic ``import()`` in a MODULE context runs either kind: the classic IIFE executes its
side-effect, the ES module executes + exports. So ONE loader spans the whole migration. Tests
that read through the GLOBAL (``window.molbuilder.X``) therefore pass BEFORE and AFTER their
module converts — no per-module test churn at conversion time.

``run_node(files, snippet, *, globals_js="", timeout=20)``:
  * ``files``   : ordered list of Path/str module files to load (dependencies first). Each is
                  dynamically imported (side effects apply; a later file sees earlier globals).
  * ``snippet`` : JS run after the imports. Top-level ``await`` is allowed (module context). It
                  MUST ``console.log(JSON.stringify(...))`` its result as the LAST stdout line.
  * ``globals_js``: extra browser-stub JS injected BEFORE the imports (fetch/AbortController/…).
  * ``static_root``: when a file imports another module via the browser-absolute specifier
                     ``/static/…`` (the URL Flask serves it under), Node's ESM loader would try
                     the real filesystem root and fail. Pass the ``web/static`` dir here to install
                     a resolve hook mapping ``/static/…`` → that dir — mirroring the browser's
                     static root so consumer modules (e.g. ``import {mount} from
                     "/static/lib/molview/index.js"``) resolve in Node too.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def run_node(files, snippet: str, *, globals_js: str = "", static_root=None, timeout: int = 20):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    imports = "\n".join(
        f"await import({json.dumps(Path(f).resolve().as_uri())});" for f in files
    )
    hook = ""
    if static_root is not None:
        root_uri = Path(static_root).resolve().as_uri().rstrip("/") + "/"
        hook = (
            "import { registerHooks } from 'node:module';\n"
            f"const __STATIC_ROOT = {json.dumps(root_uri)};\n"
            "registerHooks({\n"
            "  resolve(spec, ctx, next) {\n"
            "    if (spec.startsWith('/static/')) {\n"
            "      return { url: new URL(spec.slice('/static/'.length), __STATIC_ROOT).href,\n"
            "               shortCircuit: true };\n"
            "    }\n"
            "    return next(spec, ctx);\n"
            "  }\n"
            "});\n"
        )
    program = (
        f"{hook}"
        "globalThis.window = globalThis;\n"
        "globalThis.molbuilder = globalThis.molbuilder || {};\n"
        f"{globals_js}\n"
        f"{imports}\n"
        f"{snippet}\n"
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", program],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])

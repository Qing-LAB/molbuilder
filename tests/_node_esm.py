"""Shared Node harness for the browser modules' JS unit tests — ES-module capable.

The loaded set can be MIXED (classic IIFE files under ``lib/``, native ES modules), so the
harness loads BOTH:

  * a classic IIFE file ``(function(root){ ... root.molbuilder.X = api })(window)`` — running it
    sets its ``window.molbuilder.*`` global as a side effect; it has no ``export``.
  * a native ES module ``export function ...; window.molbuilder.X = api`` — running it publishes
    the same global (transitional side effect) AND exposes exports.

A dynamic ``import()`` in a MODULE context runs either kind: the classic IIFE executes its
side-effect, the ES module executes + exports. So ONE loader serves both.

``lifted(module, start, end)``: a function or a literal of a shipped file, its text from
  ``start`` up to ``end``, for a harness that runs it against fakes.

``served(base)``: ``globals_js`` that answers the page's ``fetch`` from the app serving
  at ``base`` -- the real route's reply to the page's own request.

``run_node(files, snippet, *, globals_js="", timeout=20)``:
  * ``files``   : ordered list of Path/str module files to load (dependencies first). Each is
                  dynamically imported (side effects apply; a later file sees earlier globals).
                  A classic file that hands its api to ``module.exports`` when it sees
                  ``module`` (the builder panels) loads as CommonJS and publishes no global:
                  the snippet takes it as ``(await import(<its file URL>)).default``.
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


def lifted(module, start: str, end: str) -> str:
    """The text of the shipped file ``module`` from ``start`` up to ``end``
    -- a function or a literal lifted out for a harness to run against its
    fakes.  A harness that lifts by these two strings fails, naming them,
    when either moves."""
    src = Path(module).read_text()
    i = src.find(start)
    if i < 0:
        raise ValueError(f"{Path(module).name}: no {start!r} to lift from")
    j = src.find(end, i)
    if j < 0:
        raise ValueError(f"{Path(module).name}: no {end!r} after {start!r}")
    return src[i:j].rstrip()


def served(base: str) -> str:
    """Browser-stub JS for ``globals_js``: the page's ``fetch`` answered by
    the app serving at ``base`` (`support.live_server.serve`) -- each request
    the page makes, its path, method, body and signal as the page built them,
    sent to the real route, and the route's answer handed back as a browser
    hands it.  A page test's input is what the server answers
    (`process/testing.md` § 4).

    Each request is kept, in order, in ``globalThis.__calls``: ``url``,
    ``method``, ``body`` (the JSON it carries, parsed) and ``init``, the
    options as the page passed them (its signal, its cache); each answer,
    read, in ``globalThis.__answered``, in the order the answers came.

    THE ANSWER IS READ WHOLE before the page is handed it, so a request is
    on the wire until its body is in hand: ``await __quiet()`` -- until no
    request is on the wire, twice in a row, a moment apart -- then waits on
    the page's own work, which takes the body from memory within the same
    turn.  The body is handed over as a browser's is: the page's abort,
    until the page has read it, makes the read fail with the runtime's own
    ``AbortError``; an abort while the answer is still coming makes the
    request fail with it (node's ``fetch`` honours the signal).

    AN ORDER IS HELD, NEVER TIMED (`process/testing.md` § 5):
    ``__hold(match)`` keeps each request ``match(call)`` accepts -- ``call``
    as ``__calls`` keeps it -- off the wire, so ``__quiet()`` does not wait
    for it; ``__release()`` issues what it held, in the order the page made
    it, to the real route, and holds no more.  A held request the page
    aborted fails when it is released (node's ``fetch`` honours the
    signal) -- not at once, as a browser's would; one never released, or a
    ``match`` that throws, fails the run, never dropped unsaid."""
    return (
        "globalThis.__calls = [];\n"
        "globalThis.__answered = [];\n"
        "globalThis.__inflight = 0;\n"
        "globalThis.__quiet = async () => {\n"
        "  for (let calm = 0; calm < 2; ) {\n"
        "    await new Promise((r) => setTimeout(r, 25));\n"
        "    calm = globalThis.__inflight ? 0 : calm + 1;\n"
        "  }\n"
        "};\n"
        "{\n"
        f"  const __served = {json.dumps(base)};\n"
        "  const __fetch = globalThis.fetch;\n"
        "  const __noBody = [101, 103, 204, 205, 304];\n"
        "  let __holding = null;\n"
        "  const __parked = [];\n"
        "  globalThis.__hold = (match) => { __holding = match; };\n"
        "  globalThis.__release = () => {\n"
        "    __holding = null;\n"
        "    __parked.splice(0).forEach((go) => go());\n"
        "  };\n"
        "  process.on('beforeExit', () => {\n"
        "    if (!__parked.length) return;\n"
        "    console.error('served: ' + __parked.length\n"
        "                  + ' request(s) held and never released');\n"
        "    process.exitCode = 1;\n"
        "  });\n"
        "  globalThis.fetch = async (url, opts = {}) => {\n"
        "    let body = null;\n"
        "    try { body = opts.body ? JSON.parse(opts.body) : null; }\n"
        "    catch (_e) { body = opts.body; }\n"
        "    globalThis.__calls.push({ url: String(url),\n"
        "                              method: opts.method || 'GET', body,\n"
        "                              init: opts });\n"
        "    const __call = globalThis.__calls[globalThis.__calls.length - 1];\n"
        "    let __isHeld = false;\n"
        "    try { __isHeld = Boolean(__holding && __holding(__call)); }\n"
        "    catch (e) {\n"
        "      console.error('served: the hold\\'s match threw: ' + e);\n"
        "      process.exitCode = 1;\n"
        "    }\n"
        "    if (__isHeld) await new Promise((go) => __parked.push(go));\n"
        "    globalThis.__inflight++;\n"
        "    try {\n"
        "      const r = await __fetch(new URL(String(url), __served), opts);\n"
        "      const bytes = new Uint8Array(await r.arrayBuffer());\n"
        "      const text = new TextDecoder().decode(bytes);\n"
        "      let answer = text;\n"
        "      try { answer = JSON.parse(text); } catch (_e) { /* not JSON */ }\n"
        "      globalThis.__answered.push(answer);\n"
        "      const signal = opts.signal;\n"
        "      const held = __noBody.includes(r.status) ? null\n"
        "        : new ReadableStream({ start(c) {\n"
        "            c.enqueue(bytes); c.close();\n"
        "            if (!signal) return;\n"
        "            const cut = () => c.error(signal.reason);\n"
        "            if (signal.aborted) cut();\n"
        "            else signal.addEventListener('abort', cut, { once: true });\n"
        "          } });\n"
        "      return new Response(held, { status: r.status,\n"
        "                                  statusText: r.statusText,\n"
        "                                  headers: r.headers });\n"
        "    } finally {\n"
        "      globalThis.__inflight--;\n"
        "    }\n"
        "  };\n"
        "}\n"
    )


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

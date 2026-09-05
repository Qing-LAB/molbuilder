"""L2 Node test: ``_formatFetchError(e)`` distinguishes server-non-
JSON failures from real network failures.

R4-A audit finding #4 (2026-06-14)
==================================

Pre-fix the trajectory / Generate FDF / Generate PySCF / Auto-
detect chip catch blocks in ``viewer.js`` all wrote::

    setStatus(..., "Network error: " + e.message, "error");

When the server returned non-JSON (HTML 5xx error page, 501 stub,
proxy plain-text drop), ``r.json()`` threw ``SyntaxError`` and the
banner read ``Network error: Unexpected token < in JSON ...``.
Users read "network down" -- when actually the server itself
crashed.

The R4 fix introduces a shared ``_formatFetchError`` helper that
branches on ``e.name === "SyntaxError"`` and surfaces a clearer
message that tells the user to check the server log.

What this file pins
===================

* SyntaxError -> "Server returned non-JSON response ..."
* TypeError / generic Error -> "Network error: <msg>"
* The four user-visible call sites (load-status, fdf-status,
  pyscf-status, auto-detect-status) all route through the helper.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


# NOTE: the module marker is combined with the node skipif below --
# two `pytestmark =` assignments do not add up, the second REPLACES
# the first, and this file silently lost `-m module` selection.


_ROOT = Path(__file__).resolve().parents[1]
_VIEWER = _ROOT / "molbuilder/web/static/structure-optimization/viewer.js"
_MODULE = _ROOT / "molbuilder/web/static/lib/fetch-error.js"


def _have_node():
    return shutil.which("node") is not None


pytestmark = [
    pytest.mark.module,
    pytest.mark.skipif(
        not _have_node(),
        reason="node not installed; L2 Node-driven test",
    ),
]


def _extract_format_fetch_error_src():
    """The formatter's source — from its ONE home.

    Read `lib/fetch-error.js`, not a viewer: the rule moved there on
    2026-08-22 (roadmap 7.2) after a second copy appeared.
    """
    src = _MODULE.read_text(encoding="utf-8")
    needle = "    function format(e) {"
    start = src.find(needle)
    if start < 0:
        pytest.fail(
            "Could not find ``function format(e) {`` in "
            "lib/fetch-error.js -- helper renamed or moved.  Update marker."
        )
    open_brace = src.find("{", start)
    depth = 0
    i = open_brace
    while i < len(src):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start:i + 1]
        i += 1
    pytest.fail("Unbalanced braces in _formatFetchError")


def _run_with_error(error_ctor: str, msg: str) -> str:
    """Call ``_formatFetchError(new <error_ctor>('<msg>'))`` in Node
    and return the resulting string."""
    fn_src = _extract_format_fetch_error_src()
    script = (
        fn_src
        + f"\nconst e = new {error_ctor}({json.dumps(msg)});"
        + "\nconsole.log(format(e));"
    )
    proc = subprocess.run(
        ["node", "--input-type=commonjs", "-e", script],
        capture_output=True, text=True, timeout=10,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"node exited {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    return proc.stdout.strip()


# --------------------------------------------------------------------- #
#  SyntaxError -> non-JSON server message                                #
# --------------------------------------------------------------------- #


def test_syntax_error_routed_to_non_json_message():
    """When ``r.json()`` throws SyntaxError (server returned
    non-JSON), the formatted message MUST NOT include the raw
    'Network error' prefix that misleads the user.  It MUST hint
    that the SERVER returned non-JSON so the user knows to check
    server-side logs, not their network."""
    out = _run_with_error("SyntaxError",
                          "Unexpected token < in JSON at position 0")
    assert "non-JSON" in out, (
        f"SyntaxError must yield a 'non-JSON' message; got {out!r}"
    )
    assert "Network error" not in out, (
        f"SyntaxError must NOT surface as 'Network error'; got {out!r}"
    )
    assert "server" in out.lower(), (
        f"SyntaxError message must reference the server so the user "
        f"knows where to look; got {out!r}"
    )


# --------------------------------------------------------------------- #
#  Real network failures preserve the existing label                     #
# --------------------------------------------------------------------- #


def test_type_error_preserves_network_error_label():
    """TypeError 'Failed to fetch' (DNS, offline, CORS preflight
    reject) is a genuine network-level failure.  The "Network
    error" label is honest here -- the user should check their
    connection."""
    out = _run_with_error("TypeError", "Failed to fetch")
    assert out.startswith("Network error:"), (
        f"TypeError must yield a 'Network error:' message; got {out!r}"
    )
    assert "Failed to fetch" in out, (
        f"Underlying message must be preserved; got {out!r}"
    )


def test_generic_error_preserves_network_error_label():
    """An unknown Error subclass falls through to the same path
    as TypeError -- safe default."""
    out = _run_with_error("Error", "weird thing happened")
    assert out.startswith("Network error:"), (
        f"Generic Error must yield 'Network error:'; got {out!r}"
    )


# --------------------------------------------------------------------- #
#  Source-text guards: the helper is used by the 4 user-visible sites    #
# --------------------------------------------------------------------- #


def test_the_formatter_has_exactly_one_home():
    """No file spells the SyntaxError rule for itself.

    Replaces a test that grepped ONE viewer for its own call sites.  That
    shape passed while a second copy of the rule was being written in
    another file, and then failed for the wrong reason when the rule moved
    -- it was measuring a file, not the rule.

    A copy is what this catches: any active source that branches on
    ``e.name === "SyntaxError"`` to build a message is a second opinion
    about what a failed fetch means.  `lib/projects/api.js` is exempt by
    name -- ``_fetchEnvelope`` makes the same distinction while normalising
    a whole response into ``{ok, ...}``, which is an envelope contract
    rather than this sentence, and folding them together would put a
    response shape and a message in one function.
    """
    import re as _re
    static = _ROOT / "molbuilder/web/static"
    exempt = {"lib/fetch-error.js", "lib/projects/api.js"}
    offenders = []
    for js in sorted(static.rglob("*.js")):
        rel = js.relative_to(static).as_posix()
        if rel in exempt or "vendor/" in rel:
            continue
        body = "\n".join(l for l in js.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith(("*", "//", "/*")))
        if _re.search(r'name\s*===\s*"SyntaxError"', body):
            offenders.append(rel)
    assert not offenders, (
        "these files build their own non-JSON message instead of calling "
        "molbuilder.fetchError.format(): " + ", ".join(offenders))


def test_every_page_that_formats_a_fetch_error_loads_the_module():
    """A shared module nobody links is a ReferenceError at the worst moment."""
    templates = _ROOT / "molbuilder/web/templates"
    for page in ("index.html", "spectra.html"):
        html = (templates / page).read_text(encoding="utf-8")
        assert "lib/fetch-error.js" in html, (
            f"{page} uses the fetch-error formatter but never loads it")

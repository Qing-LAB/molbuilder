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


pytestmark = pytest.mark.module


_ROOT = Path(__file__).resolve().parents[1]
_VIEWER = _ROOT / "molbuilder/web/static/viewer.js"


def _have_node():
    return shutil.which("node") is not None


pytestmark = pytest.mark.skipif(
    not _have_node(),
    reason="node not installed; L2 Node-driven test",
)


def _extract_format_fetch_error_src():
    src = _VIEWER.read_text(encoding="utf-8")
    needle = "    function _formatFetchError(e) {"
    start = src.find(needle)
    if start < 0:
        pytest.fail(
            "Could not find ``function _formatFetchError(e) {`` in "
            "viewer.js -- helper renamed or moved.  Update marker."
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
        + "\nconsole.log(_formatFetchError(e));"
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


def test_user_visible_catches_route_through_formatter():
    """Pin that the 4 known user-visible status banners that
    formerly surfaced 'Network error: ' directly now route through
    ``_formatFetchError``.  A refactor that drops the call from any
    of these sites fails this loudly.
    """
    src = _VIEWER.read_text(encoding="utf-8")
    for status_id in [
        "load-status",
        "fdf-status",
        "pyscf-status",
        "auto-detect-status",
    ]:
        # Look for the setStatus call carrying this status id
        # somewhere within ~200 chars of a _formatFetchError call.
        # Cheap heuristic: every setStatus(<id>, _formatFetchError(...
        needle = f'setStatus("{status_id}",'
        idx = 0
        found_match = False
        while True:
            idx = src.find(needle, idx)
            if idx < 0:
                break
            window = src[idx:idx + 200]
            if "_formatFetchError" in window:
                found_match = True
                break
            idx += len(needle)
        assert found_match, (
            f"setStatus({status_id!r}, ...) is no longer routed "
            f"through _formatFetchError anywhere in viewer.js.  "
            f"A refactor may have re-introduced the raw "
            f"'Network error:' label that misleads the user on "
            f"server 5xx-with-HTML responses."
        )

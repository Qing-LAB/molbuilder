"""The serve log is a SECRET SINK, so it is 0600 in a 0700 directory.

Measured 2026-09-12: `~/.local/state/molbuilder/logs/` was 0775 and
`serve-6006.log` was 0664, 736 KB.  `LogRoll` created both with a bare
`mkdir()` and `open(path, "ab")`, neither taking a mode.

WHY THIS FILE IS A SECRET SINK, which is not obvious from its name: it
swallows the server child's stderr verbatim, and what the child writes there
on an auth failure is chosen ON PURPOSE to be the thing too sensitive for the
browser.  `web/auth_providers/oauth.py` routes a provider error to
`logging.exception` precisely because the `params` dict "on certain
misbehaving providers can include the client_secret", with the comment "don't
risk leaking that into the user-visible response".  So the response was
protected and this file was not -- the same inversion `run-reports.md` names
for the run-report records: "the KEY file was always 0600 and the DATA it
protects was not, which is the wrong way round on a shared server".
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from molbuilder.serve_daemon import LogRoll


def _fresh() -> Path:
    return Path(tempfile.mkdtemp()) / "state" / "logs"


def _mode(p: Path) -> str:
    return oct(p.stat().st_mode)[-3:]


def test_the_log_and_its_directory_are_private_from_creation():
    d = _fresh()
    roll = LogRoll(d / "serve-1.log", max_bytes=10_000, keep=2)
    try:
        roll.write(b"a line\n")
        assert _mode(d) == "700", "a listable directory names the file"
        assert _mode(d / "serve-1.log") == "600"
    finally:
        roll.close()


def test_rotation_keeps_both_the_live_file_and_the_archive_private():
    """The archive holds the same bytes, so it gets the same mode.  The
    rotation path created both with a bare `open(..., "wb")` and
    `gzip.open(dst, "wb")`."""
    d = _fresh()
    roll = LogRoll(d / "serve-2.log", max_bytes=100, keep=2)
    try:
        roll.write(b"x" * 50)
        roll.write(b"y" * 200)          # trips the rotation
    finally:
        roll.close()
    names = sorted(p.name for p in d.iterdir())
    assert any(n.endswith(".1.gz") for n in names), f"no archive made: {names}"
    for p in d.iterdir():
        assert _mode(p) == "600", f"{p.name} is {_mode(p)}"


def test_an_existing_loose_directory_is_tightened():
    """`mode=` on mkdir covers directories it CREATES; an existing one keeps
    what it had, which is exactly how the measured 0775 survived."""
    d = _fresh()
    d.mkdir(parents=True)
    d.chmod(0o775)
    roll = LogRoll(d / "serve-3.log", max_bytes=10_000, keep=1)
    try:
        assert _mode(d) == "700", "an already-loose directory must be fixed"
    finally:
        roll.close()


def test_an_existing_loose_logfile_is_tightened():
    """So a machine that already has the 0664 file is fixed by the next
    `serve` start, with nothing for the operator to remember."""
    d = _fresh()
    d.mkdir(parents=True)
    loose = d / "serve-4.log"
    loose.write_bytes(b"secret-ish already here\n")
    loose.chmod(0o664)
    roll = LogRoll(loose, max_bytes=10_000, keep=1)
    try:
        assert _mode(loose) == "600"
        assert b"already here" in loose.read_bytes(), "append, never truncate"
    finally:
        roll.close()

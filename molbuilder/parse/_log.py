"""Parse-activity log sidecar.

Every parser (SIESTA .out, .molwatch.log, .transport.json, ...)
writes a ``<input-stem>.parse.log`` file in the same directory as
its input, recording what it did and any problems it hit during
the scan.  The file is the on-disk counterpart of
:class:`Trajectory.parse_warnings` -- the in-memory list goes
to the browser inspector; the log file persists across browser
sessions and is greppable from CLI / SSH.

Default: ON.  Opt-out via environment variable
``MOLBUILDER_PARSE_LOG=0`` (any truthy value off-by-default
counterpart is "0" / "false" / "no" / case-insensitive).

Format: one line per event::

    2026-06-25T16:30:00Z INFO  scan started: <abs_path> (<bytes> B)
    2026-06-25T16:30:00Z INFO  scanned 1024 lines, 12 frames, run_state=ongoing
    2026-06-25T16:30:00Z WARN  line 408: SCF column parse: float() failed -- "******"
    2026-06-25T16:30:00Z INFO  scan finished in 0.034 s (2 warnings)

Each scan begins with a short banner so multiple appended runs are
visually delimited.  The log file is opened in append mode so re-
parses (e.g. /api/watch/data polls) accumulate history -- you can
``tail -f`` it during a live run.

If the log can't be written (read-only dir, no space, perms), the
parser DOES NOT raise -- a log-write failure is a side concern, not
the parse's job.  We swallow IOError + OSError and continue.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_ENV_VAR = "MOLBUILDER_PARSE_LOG"


def _enabled() -> bool:
    """Check the env-var opt-out.  Default ON; suppressed when the
    env var is set to one of {"0", "false", "no", "off"} (case-
    insensitive)."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with 'Z' suffix.  Compact form (no
    microseconds) since the per-event resolution we care about is
    seconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sidecar_path(input_path: str) -> Path:
    """Sidecar path: replace the final extension with ``.parse.log``.

    ``job.out``           -> ``job.parse.log``
    ``job.molwatch.log``  -> ``job.molwatch.parse.log``
    ``job.transport.json``-> ``job.transport.parse.log``
    """
    return Path(input_path).with_suffix(".parse.log")


class ParseLogger:
    """Per-scan logger writing to ``<input>.parse.log``.

    Use as a context manager so the summary line is written on exit
    even when the parser raises.  Methods return ``None`` and never
    raise -- a log-side I/O failure is silently dropped.

    Example::

        with ParseLogger(path, parser_name="siesta") as log:
            log.info(f"scan started: {path}")
            ...
            log.warn(line_no=408, message="float() failed", snippet=raw)
            ...
            # summary line auto-written by __exit__
    """

    def __init__(self, input_path: str, parser_name: str) -> None:
        self.input_path  = input_path
        self.parser_name = parser_name
        self.path        = _sidecar_path(input_path) if _enabled() else None
        self._fh         = None
        self._t0         = None
        self._warn_count = 0
        self._info_count = 0

    def __enter__(self) -> "ParseLogger":
        self._t0 = time.monotonic()
        if self.path is None:
            return self
        try:
            self._fh = open(self.path, "a", encoding="utf-8")
            try:
                size = os.path.getsize(self.input_path)
                size_str = f" ({size} B)"
            except OSError:
                size_str = ""
            banner = (
                f"# ----------------------------------------------------\n"
                f"# {_now_iso()} {self.parser_name} scan begin\n"
                f"# ----------------------------------------------------\n"
            )
            self._fh.write(banner)
            self.info(f"scan started: {self.input_path}{size_str}")
        except OSError:
            # Read-only dir / no perms / no space.  Log writes are a
            # side-concern; the parse must continue.
            self._fh = None
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            elapsed = (time.monotonic() - self._t0) if self._t0 else 0.0
            if exc is not None:
                self._write("ERROR",
                            f"scan aborted after {elapsed:.3f} s: "
                            f"{exc_type.__name__}: {exc}")
            else:
                self._write(
                    "INFO",
                    f"scan finished in {elapsed:.3f} s "
                    f"({self._warn_count} warnings)")
            self._fh.flush()
            self._fh.close()
        except OSError:
            pass
        self._fh = None

    # -- public surface ------------------------------------------- #

    def info(self, message: str) -> None:
        self._info_count += 1
        self._write("INFO", message)

    def warn(self, message: str, *,
             line_no: Optional[int] = None,
             snippet: Optional[str] = None,
             category: Optional[str] = None) -> None:
        """Log a parser warning.  Optional ``line_no`` / ``snippet`` /
        ``category`` mirror the :class:`ParseWarning` fields so the
        log file carries the same context the in-memory list does."""
        self._warn_count += 1
        parts = []
        if line_no is not None:
            parts.append(f"line {line_no}")
        if category:
            parts.append(f"[{category}]")
        parts.append(message)
        if snippet:
            # Single-line snippet only -- multi-line snippets get
            # collapsed so the log stays grep-friendly.
            one_line = snippet.replace("\n", " \\n ").strip()
            if len(one_line) > 160:
                one_line = one_line[:157] + "..."
            parts.append(f"-- {one_line!r}")
        self._write("WARN", " ".join(parts))

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    # -- internals ------------------------------------------------ #

    def _write(self, level: str, message: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(f"{_now_iso()} {level:<5} {message}\n")
        except OSError:
            pass


__all__ = ["ParseLogger"]

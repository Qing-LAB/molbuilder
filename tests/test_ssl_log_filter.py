"""K1: narrow log filter for client-disconnect SSL EOFs.

History
=======

The trajectory pollOnce (60s cadence with in-flight guard) +
spectra watchTick (2s cadence with in-flight guard) BOTH fire
AbortController.abort() on dispose / file-change supersede.  The
TLS connection is torn down mid-stream; werkzeug's dev server
sees ``ssl.SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]`` and
logs the full ssl.c traceback at ERROR level.

Under heavy live-poll use the user's log filled with these
benign tracebacks.  Round-3 R3-C security audit flagged that a
blanket SSL-error demote would silence real TLS-downgrade probes
+ cipher-fingerprinting; the right fix is a NARROW filter that
matches ONLY the ``UNEXPECTED_EOF_WHILE_READING`` symbol.

What this file pins
===================

* A werkzeug log record carrying the EOF symbol is DROPPED.
* A werkzeug log record about another SSL event (handshake
  failure, bad version, cert validation) is PRESERVED.
* A werkzeug log record with the EOF symbol in ``exc_info`` (not
  in the message body) is also DROPPED -- captures the case
  where werkzeug emits ``Exception in request handler`` with the
  traceback in ``exc_info``.
* An unrelated log record (no SSL involvement) is PRESERVED.

The filter is installed by ``create_app`` for the ``werkzeug``
+ ``http.server`` loggers.  A future refactor that drops the
``_install_client_disconnect_filter`` call (or weakens the
``_NEEDLE`` match) fails this file loudly.
"""
from __future__ import annotations

import logging
import ssl

import pytest


pytest.importorskip("flask")


from molbuilder.web.app import (
    _ClientDisconnectSSLFilter,
    _install_client_disconnect_filter,
)


_FILTER = _ClientDisconnectSSLFilter()


def _make_record(msg: str, exc_info=None) -> logging.LogRecord:
    """Build a synthetic LogRecord without going through a real
    logger."""
    rec = logging.LogRecord(
        name="werkzeug",
        level=logging.ERROR,
        pathname="<test>",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    return rec


# --------------------------------------------------------------------- #
#  Drop path: the EOF symbol matches                                     #
# --------------------------------------------------------------------- #


def test_drops_record_carrying_eof_symbol_in_message():
    rec = _make_record(
        "ssl.SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] "
        "unexpected eof while reading (_ssl.c:2580)"
    )
    assert _FILTER.filter(rec) is False, (
        "EOF-symbol record must be dropped"
    )


def test_drops_record_carrying_eof_symbol_in_exc_info():
    """werkzeug sometimes logs a short message + exception via the
    standard ``logger.error('...', exc_info=True)`` path.  The
    traceback formatting happens at handler time, so the filter
    must walk ``record.exc_info`` to find the symbol."""
    try:
        # Synthesize a real ssl.SSLError to attach.
        raise ssl.SSLError("[SSL: UNEXPECTED_EOF_WHILE_READING] "
                           "unexpected eof while reading "
                           "(_ssl.c:2580)")
    except ssl.SSLError:
        import sys
        exc_info = sys.exc_info()

    rec = _make_record(
        "Exception in request handler",  # message DOES NOT carry the symbol
        exc_info=exc_info,
    )
    assert _FILTER.filter(rec) is False, (
        "EOF-symbol-in-exc_info record must be dropped (filter "
        "walks the traceback for the needle)"
    )


# --------------------------------------------------------------------- #
#  Preserve path: real-attack signals + unrelated records                #
# --------------------------------------------------------------------- #


def test_preserves_record_about_other_ssl_event():
    """SSL handshake failures, bad-version probes, cert
    validation -- ALL of these surface SSL errors with DIFFERENT
    messages.  Filter must NOT match them."""
    for msg in [
        "ssl.SSLError: [SSL: WRONG_VERSION_NUMBER] wrong version",
        "ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] cert failed",
        "ssl.SSLError: [SSL: NO_SHARED_CIPHER] no shared cipher",
        "ssl.SSLError: [SSL: BAD_SIGNATURE] bad signature",
        "Exception while handling: connection reset by peer",
    ]:
        rec = _make_record(msg)
        assert _FILTER.filter(rec) is True, (
            f"Non-EOF SSL event must be preserved: {msg!r}"
        )


def test_preserves_unrelated_record():
    """A normal log record about something else (e.g. an HTTP
    request) must be untouched."""
    rec = _make_record(
        "127.0.0.1 - - [14/Jun/2026 22:33:11] "
        "\"POST /api/build/fdf HTTP/1.1\" 200 -"
    )
    assert _FILTER.filter(rec) is True


# --------------------------------------------------------------------- #
#  Install-side: the helper is called from create_app                    #
# --------------------------------------------------------------------- #


def test_install_attaches_filter_to_werkzeug_logger():
    """Sanity: ``_install_client_disconnect_filter`` attaches the
    filter to the ``werkzeug`` logger and is idempotent.
    """
    werkzeug_log = logging.getLogger("werkzeug")
    # Capture pre-state so we restore it afterwards.
    pre_filters = list(werkzeug_log.filters)
    try:
        # Clear any prior filter from earlier test runs.
        werkzeug_log.filters = []
        _install_client_disconnect_filter()
        assert any(
            isinstance(f, _ClientDisconnectSSLFilter)
            for f in werkzeug_log.filters
        ), "filter not installed on werkzeug logger"

        # Idempotent: calling again does NOT add a second copy.
        _install_client_disconnect_filter()
        n_filters = sum(
            1 for f in werkzeug_log.filters
            if isinstance(f, _ClientDisconnectSSLFilter)
        )
        assert n_filters == 1, (
            f"filter must be attached EXACTLY once; found "
            f"{n_filters} copies"
        )
    finally:
        werkzeug_log.filters = pre_filters


def test_create_app_installs_the_filter():
    """The filter MUST be wired from ``create_app`` so a deployed
    server gets it without an extra install step."""
    werkzeug_log = logging.getLogger("werkzeug")
    pre_filters = list(werkzeug_log.filters)
    werkzeug_log.filters = []
    try:
        from molbuilder.web.app import create_app
        create_app(config={})  # builds the Flask app
        assert any(
            isinstance(f, _ClientDisconnectSSLFilter)
            for f in werkzeug_log.filters
        ), "create_app did not install the client-disconnect filter"
    finally:
        werkzeug_log.filters = pre_filters

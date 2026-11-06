"""K1: narrow log filter for client-disconnect SSL EOFs.

Why
===

The trajectory pollOnce (60s cadence with in-flight guard) +
spectra watchTick (2s cadence with in-flight guard) BOTH fire
AbortController.abort() on dispose / file-change supersede.  The
TLS connection is torn down mid-stream; werkzeug's dev server
sees ``ssl.SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING]`` and
logs the full ssl.c traceback at ERROR level.

Under heavy live-poll use the user's log filled with these
benign tracebacks.  A blanket SSL-error demote would silence real
TLS-downgrade probes + cipher-fingerprinting, so the filter is NARROW:
it matches ONLY the ``UNEXPECTED_EOF_WHILE_READING`` symbol.

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


from molbuilder.web.app import _ClientDisconnectSSLFilter


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


# --------------------------------------------------------------------- #
#  Wired from create_app                                                 #
# --------------------------------------------------------------------- #


def test_create_app_installs_the_filter():
    """A deployed server gets the filter without an extra step: after
    ``create_app`` the werkzeug logger drops the EOF record and lets another
    SSL record through.

    THE SILENT FAILURE: the filter alive as a class and called by nothing, so
    the log floods again while every test of the class stays green."""
    werkzeug_log = logging.getLogger("werkzeug")
    pre_filters = list(werkzeug_log.filters)
    werkzeug_log.filters = []
    try:
        from molbuilder.web.app import create_app
        create_app(config={})  # builds the Flask app
        eof = _make_record(
            "ssl.SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] "
            "unexpected eof while reading (_ssl.c:2580)")
        other = _make_record(
            "ssl.SSLError: [SSL: WRONG_VERSION_NUMBER] wrong version")
        assert not werkzeug_log.filter(eof), (
            "create_app did not install the client-disconnect filter")
        assert werkzeug_log.filter(other), (
            "the filter on the logger dropped another SSL event")
    finally:
        werkzeug_log.filters = pre_filters

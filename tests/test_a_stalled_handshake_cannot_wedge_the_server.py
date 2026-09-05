"""One silent client must not stop the server for everyone.

**The outage this is for.** On 2026-09-05 the dev server on 8888 had been up
8h50m on **7 seconds of CPU**, with 52 connections queued and never accepted
and 52 more in CLOSE-WAIT from clients that had given up. Every write from
the browser failed with *"Couldn't save state to disk (write-state · Failed
to fetch)"*. `SIGUSR1` dumped the main thread parked in
`ssl.py:do_handshake`, reached from `socketserver.get_request` inside
`serve_forever` — the accept loop itself.

`TCPServer.get_request` is `return self.socket.accept()`, and under TLS that
socket is an `ssl.SSLSocket` whose `accept()` performs the handshake as part
of accepting. A client that finishes the TCP connection and then says nothing
holds that call open with no timeout, and nobody else is ever accepted.

It costs one TCP connection, needs no credentials, and **authentication
cannot reach it**: the handshake precedes all HTTP, so no request exists and
`web/rate_limit.py`'s IP blocklist — a `before_request` hook — never runs.
The connections holding the real server were internet scanners.

These tests run a REAL TLS server on a real port and park a real silent
client on it.
"""
from __future__ import annotations

import socket
import ssl
import threading
import time

import pytest

pytestmark = pytest.mark.module

cryptography = pytest.importorskip("cryptography")


def _self_signed(tmp_path):
    """A throwaway cert/key pair for 127.0.0.1."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))]),
                critical=False)
            .sign(key, hashes.SHA256()))

    cp, kp = tmp_path / "c.pem", tmp_path / "k.pem"
    cp.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    kp.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    return str(cp), str(kp)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"alive"]


def _get(port, timeout=5.0) -> str:
    """One real HTTPS request, cert checking off (it is self-signed).

    Reads to EOF: the answer comes back chunked, so a single ``recv`` returns
    the headers and none of the body — which failed this file's own first run
    with "the server does not serve at all" while the server had in fact
    answered ``200 OK``.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as raw:
        with ctx.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
            tls.settimeout(timeout)
            tls.sendall(b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            chunks = []
            while True:
                try:
                    b = tls.recv(4096)
                except (TimeoutError, ssl.SSLError, OSError):
                    break
                if not b:
                    break
                chunks.append(b)
            return b"".join(chunks).decode("latin-1")



def _serve(app, port, ssl_ctx):
    """Start the real Werkzeug TLS server, hardened the way `cmd_serve` does.

    `_harden_tls_accept_loop` patches `ThreadedWSGIServer` -- the class
    `make_server(threaded=True)` builds and `app.run` uses -- so this exercises
    exactly the shipped path rather than a parallel one.
    """
    from werkzeug.serving import make_server

    from molbuilder.cli import _harden_tls_accept_loop

    _harden_tls_accept_loop()
    srv = make_server("127.0.0.1", port, app, threaded=True, ssl_context=ssl_ctx)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    else:                                                # pragma: no cover
        pytest.fail("server never came up")
    return srv


def test_a_silent_client_does_not_stop_everyone_else(tmp_path):
    """A connection that never speaks must not block the accept loop.

    Mutation-checked by serving the same app through Werkzeug's own path
    (`app.run` / an un-overridden `get_request`): the request below then
    never completes, which is the outage.
    """
    port = _free_port()
    _serve(_app, port, _self_signed(tmp_path))

    assert "alive" in _get(port), "the server does not serve at all"

    # THE SILENT CLIENT: TCP completes, then not one byte of ClientHello.
    stallers = []
    try:
        # TWENTY of them. Three was enough to defeat the first version of
        # the fix (a bounded handshake still run in the accept loop); twenty
        # is well past any per-connection timeout budget, so this can only
        # pass if the handshake is off the loop entirely.
        for _ in range(20):
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            stallers.append(s)
        time.sleep(0.5)

        started = time.monotonic()
        body = _get(port, timeout=5.0)
        took = time.monotonic() - started

        assert "alive" in body, (
            "a client that sent nothing stopped the server answering anyone "
            "else — this is the 2026-09-05 outage")
        assert took < 5.0, f"the answer took {took:.1f}s behind a silent client"
    finally:
        for s in stallers:
            try:
                s.close()
            except OSError:                              # pragma: no cover
                pass


def test_the_handshake_deadline_is_not_left_on_the_request(tmp_path):
    """The deadline guards the HANDSHAKE only.

    If it stayed on the connection, a slow route or a long poll would be cut
    off mid-answer at `HANDSHAKE_TIMEOUT_S`, turning a denial-of-service fix
    into a denial of service.
    """
    from molbuilder.cli import HANDSHAKE_TIMEOUT_S

    slept = HANDSHAKE_TIMEOUT_S + 1.0

    def _slow(environ, start_response):
        time.sleep(slept)
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"slow-but-whole"]

    port = _free_port()
    _serve(_slow, port, _self_signed(tmp_path))

    body = _get(port, timeout=slept + 10.0)
    assert "slow-but-whole" in body, (
        f"a response that took longer than the {HANDSHAKE_TIMEOUT_S}s handshake "
        "deadline was cut off — the deadline leaked onto the request")

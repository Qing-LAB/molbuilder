"""CLI surface smoke tests.

Tests against the current argparse-based CLI in ``molbuilder/cli.py``.
A future click conversion will rewrite the implementation; these
tests are written to survive that rewrite by exercising the user-
visible interface only:

  * every subcommand has a ``--help``
  * every subcommand parses the documented happy-path invocation
  * subcommand routing (``main(["X", ...]) -> proper handler``) works
    for the build verbs without hitting heavy external deps

Heavy dispatches (smiles needs RDKit, name needs PubChem, fdf needs
ASE + a structure file, watch serve binds a port) are tested via
mocks where reasonable and via ``--help`` only otherwise.  An end-to-
end test that produces a real .fdf file lives in
``test_smiles_and_siesta.py``; this file stays light.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pytest

from molbuilder import cli


# --------------------------------------------------------------------- #
#  --help reachability for every subcommand                             #
# --------------------------------------------------------------------- #


_SUBCOMMANDS = [
    "peptide", "dna", "rna", "smiles", "name",
    "fdf", "pyscf",
    "serve", "watch",
]


@pytest.mark.parametrize("sub", _SUBCOMMANDS)
def test_subcommand_help_exits_cleanly(sub):
    """Every subcommand's --help must succeed (SystemExit code 0).
    Failure here means an argparse misconfiguration -- a typo in
    `add_argument`, a duplicate flag, etc."""
    with pytest.raises(SystemExit) as exc:
        cli.main([sub, "--help"])
    assert exc.value.code == 0


def test_top_level_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_no_subcommand_is_an_error():
    """Running `molbuilder` with no subcommand should fail with an
    argparse usage error, not an unhelpful crash."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    # argparse exits 2 on usage error.
    assert exc.value.code == 2


def test_unknown_subcommand_is_an_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["nonsense"])
    assert exc.value.code == 2


# --------------------------------------------------------------------- #
#  Build subcommand routing                                              #
#                                                                        #
#  We don't run the real builders here (they'd hit RDKit / PeptideBuilder/
#  PubChem); instead we monkeypatch the module-level builder funcs that  #
#  cli.main looks up, and assert it routed the right one.  This catches  #
#  argument-mapping bugs (e.g. --form vs --terminal swap).               #
# --------------------------------------------------------------------- #


def _make_capture(captured):
    """Return a fake builder that records (kind, sequence, kwargs)."""
    def _fake(seq, **kwargs):
        captured.append((seq, kwargs))
        # Return a tiny Structure so cli._emit() can call .to_xyz() / .summary().
        from molbuilder.structure import Structure
        return Structure(elements=["H"], positions=np.array([[0.0, 0.0, 0.0]]),
                         title="stub")
    return _fake


def test_build_peptide_routes_to_build_peptide(monkeypatch, capsys, tmp_path):
    captured = []
    monkeypatch.setattr("molbuilder.build_peptide", _make_capture(captured))
    rc = cli.main(["peptide", "ARNDC", "--out", str(tmp_path / "p.xyz")])
    assert rc == 0
    assert len(captured) == 1
    seq, kwargs = captured[0]
    assert seq == "ARNDC"
    # title kwarg is always passed (even if None).
    assert "title" in kwargs


def test_build_dna_passes_backend_and_form(monkeypatch, tmp_path):
    """DNA's --backend / --form / --terminal flags must reach the
    builder.  This catches the argparse->kwargs mapping; if a flag
    is dropped, the test fails -- a real bug."""
    captured = []
    monkeypatch.setattr("molbuilder.build_dna", _make_capture(captured))
    rc = cli.main([
        "dna", "ATGC",
        "--backend", "rdkit",
        "--form", "B",
        "--terminal", "OH",
        "--out", str(tmp_path / "d.xyz"),
    ])
    assert rc == 0
    seq, kwargs = captured[0]
    assert seq == "ATGC"
    assert kwargs.get("backend") == "rdkit"
    assert kwargs.get("form") == "B"
    assert kwargs.get("terminal") == "OH"
    # protonate_phosphates defaults True; --no-protonate-phosphates not given.
    assert kwargs.get("protonate_phosphates") is True


def test_build_dna_no_protonate_phosphates_flag(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("molbuilder.build_dna", _make_capture(captured))
    cli.main([
        "dna", "ATGC",
        "--no-protonate-phosphates",
        "--out", str(tmp_path / "d.xyz"),
    ])
    _seq, kwargs = captured[0]
    assert kwargs.get("protonate_phosphates") is False


def test_build_rna_default_form_is_a(monkeypatch, tmp_path):
    """RNA's default helix form is A; if the user doesn't pass --form,
    the builder must NOT receive a form kwarg (the builder library
    has its own RNA default)."""
    captured = []
    monkeypatch.setattr("molbuilder.build_rna", _make_capture(captured))
    cli.main(["rna", "AUGC", "--out", str(tmp_path / "r.xyz")])
    _seq, kwargs = captured[0]
    # When --form isn't passed, cli should not inject a default form
    # kwarg; the builder picks one.  See cli.py:_add_build_parser
    # form=None default + the dispatch's `if args.form is not None`.
    assert "form" not in kwargs


# --------------------------------------------------------------------- #
#  --pyscf-atom-block emission to stdout                                 #
# --------------------------------------------------------------------- #


def test_pyscf_atom_block_emits_to_stdout(monkeypatch, capsys, tmp_path):
    """The --pyscf-atom-block flag is a pipe-friendly stdout emitter
    documented in cli.py.  Verify it actually writes to stdout."""
    from molbuilder.structure import Structure
    monkeypatch.setattr(
        "molbuilder.build_peptide",
        lambda seq, **kwargs: Structure(
            elements=["C"], positions=np.array([[0.1, 0.2, 0.3]]),
            title="x",
        ),
    )
    rc = cli.main(["peptide", "A", "--pyscf-atom-block"])
    assert rc == 0
    out = capsys.readouterr().out
    # PySCF atom block format: "<element> <x> <y> <z>" per line.
    assert "C" in out
    assert "0.1" in out and "0.2" in out and "0.3" in out


# --------------------------------------------------------------------- #
#  watch serve subcommand wiring                                         #
# --------------------------------------------------------------------- #


def test_watch_serve_calls_create_app(monkeypatch):
    """`molbuilder watch serve` must instantiate the unified Flask app
    via create_app() and start it bound to the requested host/port."""
    captured = {}

    class _FakeApp:
        def run(self, host, port, debug, threaded):
            captured["host"] = host
            captured["port"] = port
            captured["debug"] = debug
            captured["threaded"] = threaded
    monkeypatch.setattr("molbuilder.web.app.create_app", lambda: _FakeApp())
    rc = cli.main(["watch", "serve", "--host", "127.0.0.1", "--port", "5050"])
    assert rc == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 5050
    assert captured["threaded"] is True

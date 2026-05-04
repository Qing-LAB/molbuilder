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
#  Phase 5b: stdin support (`fdf - out.fdf`, `pyscf - out.py`)          #
# --------------------------------------------------------------------- #


def test_fdf_reads_xyz_from_stdin(monkeypatch, tmp_path):
    """``molbuilder fdf - out.fdf`` reads stdin, sniffs XYZ vs PDB
    from the first non-blank line, writes to a temp file, and feeds
    that into the standard convert() pipeline.  Without this you
    can't pipe ``molbuilder dna ATGC | molbuilder fdf -``."""
    import io
    xyz = "2\nh2 stdin\nH 0 0 0\nH 0.74 0 0\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(xyz))
    out_fdf = tmp_path / "h2.fdf"
    rc = cli.main(["fdf", "-", str(out_fdf),
                   "--no-copy-psml", "--no-write-md-history"])
    assert rc == 0
    assert out_fdf.exists() and out_fdf.stat().st_size > 0
    text = out_fdf.read_text()
    assert "NumberOfAtoms" in text


def test_pyscf_reads_xyz_from_stdin(monkeypatch, tmp_path):
    """Same Unix-pipe support on the pyscf subcommand."""
    import io
    xyz = "2\nh2 stdin\nH 0 0 0\nH 0.74 0 0\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(xyz))
    out_py = tmp_path / "h2.py"
    rc = cli.main(["pyscf", "-", str(out_py), "--no-optimize",
                   "--no-density-fit"])
    assert rc == 0
    assert out_py.exists() and out_py.stat().st_size > 0


def test_stdin_pdb_sniffs_correctly(monkeypatch, tmp_path):
    """Stdin sniff: a first line that isn't an integer is treated as
    PDB (HEADER / TITLE / ATOM / HETATM all qualify)."""
    import io
    pdb = (
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000  1.00  0.00           H\n"
        "ATOM      2  H   MOL A   1       0.740   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(pdb))
    out_fdf = tmp_path / "h2.fdf"
    rc = cli.main(["fdf", "-", str(out_fdf),
                   "--no-copy-psml", "--no-write-md-history"])
    assert rc == 0


# --------------------------------------------------------------------- #
#  Phase 5c: validate subcommand (Issue JSON to stdout)                 #
# --------------------------------------------------------------------- #


def _write_xyz(path, text):
    path.write_text(text)
    return str(path)


def test_validate_clean_water_returns_no_errors(capsys, tmp_path):
    """Geometry-only validation on a clean structure: 0 errors, JSON
    payload with the expected shape, exit 0."""
    import json
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    rc = cli.main(["validate", _write_xyz(tmp_path / "h2o.xyz", xyz)])
    assert rc == 0
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["n_errors"] == 0
    # The h_ratio validator runs on every structure -- water at 2:1
    # H/heavy is well above the 0.3 warn threshold.
    assert not any(i["where"] == "geometry.h_ratio" for i in body["issues"])
    assert body["engine"] is None


def test_validate_skeleton_warns_on_h_ratio(capsys, tmp_path):
    """A heavy-atom skeleton (ratio ~ 0) must surface the h_ratio
    warn issue in the JSON payload."""
    import json
    xyz = "3\nskeleton\nC 0 0 0\nN 1.5 0 0\nO 3.0 0 0\n"
    rc = cli.main(["validate", _write_xyz(tmp_path / "sk.xyz", xyz)])
    assert rc == 0     # warnings don't stop the run
    body = json.loads(capsys.readouterr().out)
    h_warns = [i for i in body["issues"]
               if i["severity"] == "warn" and i["where"] == "geometry.h_ratio"]
    assert len(h_warns) == 1


def test_validate_exit_on_error_returns_2(monkeypatch, capsys, tmp_path):
    """--exit-on-error makes the command non-zero when any error-severity
    issue fires.  Synthesise a structure with two atoms < 0.3 A apart,
    which the min_distance check flags as error."""
    import json
    xyz = "2\nbroken\nO 0 0 0\nH 0.1 0 0\n"   # 0.1 A < 0.3 -> error
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", _write_xyz(tmp_path / "bad.xyz", xyz),
                  "--exit-on-error"])
    assert exc.value.code == 2
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["n_errors"] >= 1


def test_validate_engine_siesta_runs_config_checks(capsys, tmp_path):
    """--engine siesta runs the SIESTA-side validators (the same set
    render_fdf would run before emitting), not just geometry checks."""
    import json
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    rc = cli.main(["validate", _write_xyz(tmp_path / "h2o.xyz", xyz),
                   "--engine", "siesta"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["engine"] == "siesta"


def test_validate_pretty_json_indents(capsys, tmp_path):
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    rc = cli.main(["validate", _write_xyz(tmp_path / "h2o.xyz", xyz),
                   "--pretty"])
    assert rc == 0
    out = capsys.readouterr().out
    # Pretty-printed JSON uses newlines + 2-space indent.
    assert "\n  " in out


# --------------------------------------------------------------------- #
#  Phase 5d: watch parse / tail subcommands                             #
# --------------------------------------------------------------------- #


_MW_LOG = """\
# molwatch trajectory log v1
# engine: pyscf
# created: 2026-04-25T11:00:00

==== molwatch step 0 begin ====
step_index: 0
kind: initial_preview
wall_time: 1700000000.0
n_atoms: 2
coordinates (Ang):
   H  0.0  0.0  0.0
   H  0.74 0.0  0.0
energy (eV): None
forces (eV/Ang):
max_force (eV/Ang): None
scf_history begin
scf_history end
==== molwatch step 0 end ====

==== molwatch step 1 begin ====
step_index: 1
wall_time: 1700000005.0
n_atoms: 2
coordinates (Ang):
   H  0.0  0.0  0.0
   H  0.75 0.0  0.0
energy (eV): -32.5
forces (eV/Ang):
   H  0.0 0.0 0.0
   H  0.0 0.0 0.0
max_force (eV/Ang): 0.0
scf_history begin
scf_history end
==== molwatch step 1 end ====

# concluded: 2026-04-25T11:00:05
"""


def test_watch_parse_emits_full_trajectory_json(capsys, tmp_path):
    """`watch parse` reads a .molwatch.log and dumps the parsed
    trajectory as JSON: per-frame coords, energies, max_forces,
    wall_times, run_state."""
    import json
    p = tmp_path / "run.molwatch.log"
    p.write_text(_MW_LOG)
    rc = cli.main(["watch", "parse", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["source_format"] == "pyscf"
    assert body["run_state"]     == "finished"
    assert len(body["frames"])   == 2
    assert body["energies"]      == [None, -32.5]


def test_watch_parse_frames_only_drops_atom_arrays(capsys, tmp_path):
    """--frames-only emits the per-frame summary without the heavy
    coordinates / forces arrays.  Useful for piping a long trajectory
    into jq / grep without slurping megabytes of coordinates."""
    import json
    p = tmp_path / "run.molwatch.log"
    p.write_text(_MW_LOG)
    rc = cli.main(["watch", "parse", str(p), "--frames-only"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert "frames"  not in body
    assert "forces"  not in body
    assert body["energies"]   == [None, -32.5]
    assert body["wall_times"] == [1700000000.0, 1700000005.0]


def test_watch_tail_emits_ndjson_one_per_frame(capsys, tmp_path):
    """`watch tail` emits NDJSON: one JSON object per line, one line
    per new frame.  Stops when the run is concluded.  This file is
    already finished, so we get all 2 frames immediately."""
    import json
    p = tmp_path / "run.molwatch.log"
    p.write_text(_MW_LOG)
    rc = cli.main(["watch", "tail", str(p), "--poll-ms", "10",
                   "--max-frames", "10"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # NDJSON: one JSON object per line.
    lines = [json.loads(ln) for ln in out.splitlines() if ln]
    assert len(lines) == 2
    assert lines[0]["step"]   == 0
    assert lines[1]["step"]   == 1
    assert lines[1]["energy"] == -32.5


def test_watch_tail_rejects_stdin(capsys):
    """`watch tail` needs a real file to poll -- stdin can't be
    re-read.  Reject with an explicit error rather than hanging."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["watch", "tail", "-"])
    assert exc.value.code == 2


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

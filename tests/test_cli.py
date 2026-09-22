"""CLI surface smoke tests.

Tests against the click-based CLI in ``molbuilder/cli.py``.  Three
roles:

  * every subcommand has a ``--help``
  * every subcommand parses the documented happy-path invocation
  * subcommand routing (``main(["X", ...]) -> proper handler``) works
    for the build verbs without hitting heavy external deps
  * the dataclass -> click bridge (add_dataclass_options) wires every
    PySCFConfig field through to the right kwarg

Heavy dispatches (smiles needs RDKit, name needs PubChem, watch serve
binds a port) are tested via mocks where reasonable and via ``--help``
only otherwise.  ``molbuilder fdf`` was DELETED 2026-08-11 (C1+C2): a
deck is rendered by `jobset prep` from a description, so SiestaConfig
no longer meets click at all.  The rendered deck is owned by
``tests/test_prep_calculation.py``, value fidelity by the template
round-trip (``tests/test_template_roundtrip.py``).
"""

from __future__ import annotations

import contextlib
import sys

import numpy as np
import pytest

from molbuilder import cli


# --------------------------------------------------------------------- #
#  --help reachability for every subcommand                             #
# --------------------------------------------------------------------- #


_SUBCOMMANDS = [
    "peptide", "dna", "rna", "smiles", "name",
    # "fdf" left with the verb (C2, 2026-08-11), and "pyscf" followed it on
    # 2026-09-17 -- this line said it would stay "until decision 34 reworks the
    # emitted-script path the same way", and that is what happened: a deck is
    # written by `jobset prep` from a description, on both engines.
    "modify",
    "serve", "watch",
]


@pytest.mark.parametrize("sub", _SUBCOMMANDS)
def test_subcommand_help_exits_cleanly(sub):
    """Every subcommand's --help must succeed (SystemExit code 0).
    Failure here means a click misconfiguration -- a typo in a
    decorator, a duplicate flag name, an undefined kwarg, etc."""
    with pytest.raises(SystemExit) as exc:
        cli.main([sub, "--help"])
    assert exc.value.code == 0


def test_top_level_help_exits_cleanly():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_no_subcommand_is_an_error():
    """Running `molbuilder` with no subcommand should fail with a
    usage error, not an unhelpful crash."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    # click exits 2 on usage error.
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
                         title="stub", vacuum=(12.0, 12.0, 12.0))
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
    builder.  This catches the click-decorator -> kwargs mapping; if
    a flag is dropped, the test fails -- a real bug."""
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
            title="x", vacuum=(12.0, 12.0, 12.0)),
    )
    rc = cli.main(["peptide", "A", "--pyscf-atom-block"])
    assert rc == 0
    out = capsys.readouterr().out
    # PySCF atom block format: "<element> <x> <y> <z>" per line.
    assert "C" in out
    assert "0.1" in out and "0.2" in out and "0.3" in out


# --------------------------------------------------------------------- #
#  Phase 5b: stdin support (`pyscf - out.py`)                           #
# --------------------------------------------------------------------- #
#  The fdf twin of these tests went with the verb (C2, 2026-08-11).
#  The stdin helper is shared, so the sniff stays gated through pyscf.










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
#  Phase 5e: add_dataclass_options decorator                            #
# --------------------------------------------------------------------- #






# --------------------------------------------------------------------- #
#  Bridge coverage: every non-skip dataclass field is wired to the CLI  #
#                                                                       #
#  Safety net for the add_dataclass_options bridge under cmd_pyscf.     #
#  Three layers, each catching a different class of bug:                #
#                                                                       #
#    1. Bridge exposure -- every PySCFConfig field without              #
#       ``skip_cli=True`` must appear as a click option on the          #
#       subcommand's ``--help``.  Catches "bridge dropped a field"      #
#       when the metadata key is misspelled or a new field lands        #
#       without metadata.                                               #
#                                                                       #
#    2. CLI -> Config plumbing -- (flag, value) pairs invoked against   #
#       a patched ``convert`` that captures the constructed config.     #
#       Catches wrong-kwarg-name and type-coercion bugs.                #
#                                                                       #
#  The SiestaConfig half of every layer was RETIRED 2026-08-11 with     #
#  `molbuilder fdf` (C2): SiestaConfig no longer meets click at all,    #
#  so there is no bridge to guard.  What those tests really protected   #
#  -- a described value reaching the deck unchanged -- is owned by the  #
#  template round-trip (test_template_roundtrip.py, all 39 exposed      #
#  fields) and `prep`'s preflight declared-type row; the rendered deck  #
#  by test_prep_calculation.py.                                         #
# --------------------------------------------------------------------- #




def _h2_xyz_at(path):
    path.write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n")
    return str(path)


def _stub_pyscf_summary(out_path):
    return {"py": str(out_path), "n_atoms": 2, "charge": 0, "label": "h2"}






# The 20-case "each default renders in the FDF" sweep that sat here was
# retired 2026-08-19.  Every case re-ran the section walk the deck-runner
# tests already pin, over values that are DECLARED DATA in the catalogue;
# the failure it feared -- a field wired to the CLI that the generator
# ignores -- is caught, for every field including tomorrow's, by
# tests/test_every_form_field_reaches_the_deck.py, which fails NAMING the
# field whenever changing it cannot change the deck.

# ---- Modify electrode-spec parser is case-insensitive on key ----- #


# ---- Bridge: metadata['choices'] -> click.Choice ---------------- #




@pytest.mark.parametrize("subcommand,flag,bad_val", [
    # The two ("fdf", …) rows were RETIRED 2026-08-12: with the verb deleted
    # (C2), click's "No such command" also exits 2, so they passed while
    # asserting nothing about choice validation -- a vacuous green.  Choice
    # metadata for SiestaConfig is enforced on the described path by the
    # preflight's declared-type row instead.
    ("pyscf", "--method",          "UKKS"),
    ("pyscf", "--scf-init-guess",  "huckl"),
    ("pyscf", "--optimizer",       "geometric_v2"),
])
def test_real_subcommand_choice_validation_rejects_typos(
        subcommand, flag, bad_val, tmp_path):
    """The five config fields whose dataclass metadata carries a
    ``choices`` tuple keep their constraint after the bridge migration:
    a typo on the real ``fdf`` / ``pyscf`` subcommands fails at parse
    time with exit code 2 rather than producing a broken FDF / .py."""
    in_xyz = _h2_xyz_at(tmp_path / "h2.xyz")
    out_path = tmp_path / ("h2.fdf" if subcommand == "fdf" else "h2.py")
    with pytest.raises(SystemExit) as exc:
        cli.main([subcommand, in_xyz, str(out_path), flag, bad_val])
    assert exc.value.code == 2


# ---- Bridge: unknown types must error loudly (P3) --------------- #








def test_modify_electrode_spec_key_case_insensitive():
    """``@CONTACT=`` / ``@Contact=`` are accepted (R3 fix: the parser
    lowercases the key).  Catches a regression where the ``.lower()`` call in
    ``_parse_electrode_spec`` is dropped -- earlier mutation testing showed 0
    test failures when this was silently removed.

    The ``@GAP=`` half went with pair mode (redesign plan § 3.4); the
    lowercasing it also exercised is covered by the remaining key."""
    upper_contact = cli._parse_electrode_spec("Au:111:3x3x2@CONTACT=2.4:+z=3")
    assert upper_contact["mode"] == "single"
    assert upper_contact["contact_distance"] == 2.4

    mixed_contact = cli._parse_electrode_spec("Au:111:3x3x2@Contact=2.4:-z=7")
    assert mixed_contact["mode"] == "single"
    assert mixed_contact["contact_distance"] == 2.4


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
    # The .molwatch.log carries epochs, so the epoch series is the
    # one that is populated -- and the elapsed series is derived from
    # it once, in the payload builder (parse.md § 2a, P-T3).
    assert body["wall_clock_s"] == [1700000000.0, 1700000005.0]
    assert body["elapsed_s"]    == [0.0, 5.0]


@contextlib.contextmanager
def _must_return_within(seconds, what):
    """Turn a hang into a failure.

    `watch tail` polls until the run is concluded, so a bug in *that*
    decision does not fail the test -- it spins.  On 2026-08-25 it did:
    the § 2b rename retired the state names the loop compared against,
    and the lane stalled at 22% for eleven hours until the process was
    killed.  A test that cannot fail is worse than no test, because it
    takes the rest of the suite down with it.

    ``pytest-timeout`` is not installed; SIGALRM is stdlib and pytest
    runs tests on the main thread, which is all this needs.
    """
    import signal

    def _boom(signum, frame):
        raise AssertionError(f"{what} did not return within {seconds}s")

    prev = signal.signal(signal.SIGALRM, _boom)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def test_watch_tail_emits_ndjson_one_per_frame(capsys, tmp_path):
    """`watch tail` emits NDJSON: one JSON object per line, one line
    per new frame.  Stops when the run is concluded.  This file is
    already finished, so we get all 2 frames immediately.

    ``--max-frames`` is deliberately set ABOVE the frame count: the
    frame cap must not be what ends the loop, or the test would pass
    while `watch tail` had lost the ability to notice a finished run.
    Termination here is the run_state decision and nothing else.
    """
    import json
    p = tmp_path / "run.molwatch.log"
    p.write_text(_MW_LOG)
    with _must_return_within(10, "watch tail on a concluded run"):
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
# ``molbuilder watch serve`` removed 2026-05-19 along with the
# /watch page route; ``molbuilder serve`` is now the canonical
# entry point.  Test coverage for the unified entry point lives in
# tests/test_cli_tls.py (TLS precedence + readability) and in this
# file's bootstrap test that pins the registered CLI groups.


# --------------------------------------------------------------------- #
#  modify subcommand                                                     #
# --------------------------------------------------------------------- #


def _bdt_stub_xyz():
    """Tiny "BDT-like" 4-atom XYZ string -- two S anchors at the ends,
    two C in the middle.  Useful for end-to-end junction tests."""
    return (
        "4\n"
        "bdt-stub\n"
        "S  0.000  0.000  0.000\n"
        "C  1.500  0.000  0.000\n"
        "C  3.500  0.000  0.000\n"
        "S  5.000  0.000  0.000\n"
    )


def test_modify_no_op_is_error(tmp_path):
    """`molbuilder modify` with zero op flags is a UsageError."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    with pytest.raises(SystemExit):
        cli.main(["modify", str(inp), str(outp)])


def test_modify_two_op_types_is_error(tmp_path):
    """Mixing operation TYPES in one call is a UsageError; user must
    chain via stdin/stdout pipes for multi-step workflows."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    with pytest.raises(SystemExit):
        cli.main(["modify", str(inp), str(outp),
                  "--delete", "1",
                  "--orient-axis", "0,3"])


def test_modify_delete_only(tmp_path):
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    rc = cli.main(["modify", str(inp), str(outp), "--delete", "1,2"])
    assert rc == 0
    text = outp.read_text()
    assert text.startswith("2\n"), f"expected 2-atom output; got:\n{text}"
    assert text.count(" S ") == 2 or text.count("S ") >= 2


def test_modify_orient_only(tmp_path):
    """Default --center='midpoint' since the redesign."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    rc = cli.main(["modify", str(inp), str(outp),
                   "--orient-axis", "0,3"])
    assert rc == 0
    import re
    lines = outp.read_text().splitlines()[2:]
    s_atoms = [l for l in lines if l.lstrip().startswith("S")]
    assert len(s_atoms) == 2
    coords = [list(map(float, re.split(r"\s+", l.strip())[1:4])) for l in s_atoms]
    for c in coords:
        assert abs(c[0]) < 1e-6 and abs(c[1]) < 1e-6
    # Default midpoint => +z and -z symmetric
    assert abs(coords[0][2] + coords[1][2]) < 1e-6


def test_modify_rotate_only(tmp_path):
    """--rotate spins the structure around the named axis."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    rc = cli.main(["modify", str(inp), str(outp), "--rotate", "z:90"])
    assert rc == 0
    # After 90° around z, the original x-axis line of S atoms should
    # become a y-axis line.
    import re
    lines = outp.read_text().splitlines()[2:]
    s_atoms = [l for l in lines if l.lstrip().startswith("S")]
    s_pos = [list(map(float, re.split(r"\s+", l.strip())[1:4])) for l in s_atoms]
    for c in s_pos:
        assert abs(c[0]) < 1e-6   # x ~ 0 after 90° spin


def test_modify_rotate_rejects_duplicate_flags(tmp_path):
    """T2 (post-static-review): two --rotate flags in one call are a
    UsageError instead of click silently overriding to the last value."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    with pytest.raises(SystemExit):
        cli.main(["modify", str(inp), str(outp),
                  "--rotate", "z:90", "--rotate", "x:30"])


def test_modify_warns_when_orient_suboptions_unused(tmp_path, capsys):
    """D2 (post-static-review): --axis / --angle / --center used without
    --orient-axis emit a warning to stderr instead of being silently
    ignored."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    rc = cli.main(["modify", str(inp), str(outp),
                   "--delete", "1",
                   "--angle", "30"])     # --angle without --orient-axis
    assert rc == 0
    err = capsys.readouterr().err
    assert "warning" in err.lower()
    assert "--angle" in err


def test_modify_electrode_single_mode(tmp_path):
    """Single mode: @contact= and ±z=N -- one slab on the named side."""
    inp  = tmp_path / "in.xyz"
    oriented = tmp_path / "oriented.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    cli.main(["modify", str(inp), str(oriented), "--orient-axis", "0,3"])
    rc = cli.main(["modify", str(oriented), str(outp),
                   "--electrode", "Au:111:3x3x2@contact=2.4:+z=3"])
    assert rc == 0
    text = outp.read_text()
    n = int(text.splitlines()[0])
    # 4 molecule + 9 × 2 layers (one side only) = 4 + 18
    assert n == 4 + 18


def test_modify_electrode_bad_spec_raises(tmp_path):
    """Malformed --electrode value triggers a BadParameter."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    with pytest.raises(SystemExit):
        cli.main(["modify", str(inp), str(outp),
                  "--electrode", "garbage_no_colons"])


def test_modify_stdin_stdout_pipe(tmp_path, monkeypatch, capsys):
    """`-` for input/output enables piping.  Verify that writing to '-'
    produces XYZ on stdout, and reading from '-' parses it back."""
    import io
    inp = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    # Step 1: read file, write to stdout
    rc = cli.main(["modify", str(inp), "-", "--delete", "1"])
    assert rc == 0
    captured = capsys.readouterr()
    stdout_xyz = captured.out
    assert stdout_xyz.startswith("3\n"), (
        f"expected 3-atom XYZ on stdout; got:\n{stdout_xyz!r}"
    )
    # Step 2: feed that text in via stdin to a second invocation
    monkeypatch.setattr("sys.stdin", io.StringIO(stdout_xyz))
    rc = cli.main(["modify", "-", str(outp), "--orient-axis", "0,2"])
    assert rc == 0
    final = outp.read_text()
    assert final.startswith("3\n")


# --------------------------------------------------------------------- #
#  serve --no-auth: loopback-only guard (security-relevant)             #
# --------------------------------------------------------------------- #


def test_serve_no_auth_refuses_non_loopback_host():
    """--no-auth must REFUSE a non-loopback bind so an unauthenticated
    server is never exposed off the machine."""
    from click.testing import CliRunner
    res = CliRunner().invoke(
        cli.cli, ["serve", "foreground", "--no-auth", "--host", "0.0.0.0", "--port", "8099",
                  "--no-supervise"])
    assert res.exit_code != 0
    assert "loopback" in res.output.lower()


def test_serve_no_auth_loopback_uses_config_empty(monkeypatch):
    """On a loopback host, --no-auth builds the app via
    create_app(config={}) (the no-auth seam) and serves without TLS."""
    from click.testing import CliRunner
    import molbuilder.web.app as _appmod

    calls = {}

    class _FakeApp:
        # `config` because a Flask app has one, and `cmd_serve` now records
        # THIS PROCESS's port on it (`web.app.serve_port` -- the port used to
        # be parsed back out of the Host header, which is the wrong answer
        # behind a proxy).  A stub that stands in for a Flask app has to
        # carry the parts of a Flask app the caller uses.
        config: dict = {}

        def run(self, **kw):
            calls["run_kwargs"] = kw

    def _fake_create_app(*, config=None):
        calls["config"] = config
        return _FakeApp()

    monkeypatch.setattr(_appmod, "create_app", _fake_create_app)
    res = CliRunner().invoke(
        cli.cli, ["serve", "foreground", "--no-auth", "--host", "127.0.0.1", "--port", "8099",
                  "--no-supervise"])
    assert res.exit_code == 0, res.output
    assert calls["config"] == {}                 # no-auth seam
    assert calls["run_kwargs"].get("ssl_context") is None   # plain http


# ---------------------------------------------------------------------------
# `--electrode …@gap=` — retired with the pair
# ---------------------------------------------------------------------------

def test_electrode_gap_is_refused_by_name_not_as_a_typo():
    """Pairs are not built as one step any more (redesign plan § 3.4), and
    `gap` was the PAIR's parameter — the electrode-to-electrode distance,
    meaningless for one slab.

    Refused by name rather than falling into the generic "unknown key", which
    would read as a misspelling: this key existed, did something, and was
    removed, so the message has to say what to do instead.
    """
    import pytest as _pytest
    from molbuilder import cli
    with _pytest.raises(Exception, match="no longer supported"):
        cli._parse_electrode_spec("Au:111:3x3x2@gap=8.0:5,10")


def _run_electrode(tmp_path, st, spec):
    """Build `st`, run one `--electrode` flag over it, return the result."""
    from molbuilder.workingcopy_structure import StructureCodec
    inp = tmp_path / "in.xyz"
    st.to_xyz(inp)
    out = tmp_path / "out.xyz"
    assert cli.main(["modify", str(inp), str(out), "--electrode", spec]) == 0
    return StructureCodec().load(out)


def _closest_metal_z(struct, above):
    zs = [float(p[2]) for e, p in zip(struct.elements, struct.positions)
          if e == "Au"]
    return min(z for z in zs if z > above)


def test_the_centroid_rule_moved_here_with_the_placement(tmp_path):
    """**The rule survived its builder, so its test moved with it.**

    `--electrode ...:+z=I,J` centres on the CENTROID of the trailing index
    list — 1 index is that atom, 2 their midpoint, N their centroid. That
    arithmetic lived in `modify.add_electrode_slab` until 2026-09-01, when
    the second slab builder was deleted (`archive/2026-09-01-modify-redesign-plan.md` § 3.4b).
    It is the CLI's now, because the convenience is the CLI's; `add_slab`
    takes an absolute `start_z` and reads no selection at all.
    """
    from molbuilder.structure import Structure

    # four atoms at z = 0, 2, 4, 6
    st = Structure(elements=["S"] * 4,
                   positions=np.array([[0., 0., 0.], [0., 0., 2.],
                                       [0., 0., 4.], [0., 0., 6.]]))
    for idx, anchor_z in (("1", 2.0), ("0,2", 2.0), ("0,1,2,3", 3.0)):
        out = _run_electrode(tmp_path, st,
                             f"Au:111:2x2x1@contact=2.4:+z={idx}")
        got = _closest_metal_z(out, above=anchor_z)
        assert got == pytest.approx(anchor_z + 2.4, abs=1e-6), idx


def test_a_centre_index_off_the_end_is_refused_by_the_flag(tmp_path):
    """It was an `IndexError` out of the builder; with the builder gone the
    flag checks it, and says which index and how many atoms there are."""
    from molbuilder.structure import Structure
    st = Structure(elements=["S"], positions=np.array([[0., 0., 0.]]))
    inp = tmp_path / "in.xyz"
    st.to_xyz(inp)
    with pytest.raises(SystemExit):
        cli.main(["modify", str(inp), str(tmp_path / "o.xyz"),
                  "--electrode", "Au:111:2x2x1@contact=2.4:+z=5"])


def test_electrode_still_builds_one_slab_per_flag():
    from molbuilder import cli
    spec = cli._parse_electrode_spec("Au:111:3x3x2@contact=2.4:+z=3")
    assert spec["mode"] == "single" and spec["side"] == "+z"
    assert spec["contact_distance"] == 2.4


def test_electrode_registry_is_per_slab_and_reaches_the_builder(tmp_path):
    """The two sides of a junction need DIFFERENT stacking registries.

    `junction-cell.md` § 3.1a, measured 2026-09-15: `sequence` alone does
    not change the registry at the seam, so with both slabs left on 0 every
    layer count the § 3.1 table calls *continues* comes out eclipsed on
    (100)/(110) and twinned on (111).  `start_registry` is the control that
    does change it — the web slab card has always exposed it and the CLI had
    no spec field at all, so a junction built from the command line could
    not be asked for the right seam.

    Per-slab, not a uniform `--electrode-*` sub-option, precisely because
    the two sides must differ.
    """
    import numpy as np
    from molbuilder import cli
    from click.testing import CliRunner
    from molbuilder.workingcopy_structure import StructureCodec

    assert cli._parse_electrode_spec(
        "Au:111:3x3x2@contact=2.4:+z=3")["start_registry"] == 0, \
        "omitting it must keep what every existing command line does"
    for spelling, want in (("A", 0), ("b", 1), ("C", 2), ("1", 1)):
        got = cli._parse_electrode_spec(
            f"Au:111:3x3x2@contact=2.4:registry={spelling}:+z=3")
        assert got["start_registry"] == want, spelling
        assert got["side"] == "+z", "the side survived the extra key"

    src = tmp_path / "in.xyz"
    src.write_text("2\nm\nS 0 0 -1.0\nS 0 0 1.0\n")
    made = {}
    for tag, reg in (("a", ""), ("b", ":registry=B")):
        out = tmp_path / f"{tag}.xyz"
        res = CliRunner().invoke(cli.cli, [
            "modify", str(src), str(out),
            "--electrode", f"Au:111:3x3x3@contact=2.4{reg}:+z=1"])
        assert res.exit_code == 0, res.output
        p = np.asarray(StructureCodec().load(out).positions, dtype=float)
        up = p[p[:, 2] > 1.5]
        made[tag] = up[np.isclose(up[:, 2], up[:, 2].min())][:, :2].mean(axis=0)

    step = float(np.linalg.norm(made["a"] - made["b"]))
    assert step > 1.0, (
        f"the registry never reached the builder: the starting layer moved "
        f"{step:.4f} Å")


# --------------------------------------------------------------------- #
#  The trajectory verbs refuse what is not a trajectory                 #
# --------------------------------------------------------------------- #


#: `watch parse` and `runtime-info` are the SAME path -- detect, then read
#: `.frames` -- so one of them stands for both.  `watch tail` is here on its
#: own merits: it calls the guard inside a poll loop, where the failure is a
#: HANG rather than a traceback.
@pytest.mark.parametrize("verb", [["watch", "parse"], ["watch", "tail"]])
def test_the_trajectory_verbs_refuse_a_single_geometry(verb, tmp_path,
                                                       capsys):
    """These three read `.frames` after detection, so a file whose parser
    answers a `StructureResult` crashed with
    `AttributeError: 'StructureResult' object has no attribute 'frames'`.

    The guard written for this asked `is_dir()` -- which catches a
    directory and not this, the file PySCF writes at the end of every
    optimization.  `/api/watch/*` had the right rule a layer up
    (`_refuse_if_not_a_trajectory`); both now ask
    `parse.types.answers_a_trajectory`.

    `watch tail` is in the list deliberately: it calls the guard INSIDE a
    poll loop that retries `ParseError` and sleeps, so a refusal raised as
    one hangs for ever instead of printing.
    """
    p = tmp_path / "x_optimized.xyz"
    p.write_text("2\nfinal\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    with _must_return_within(15, f"{' '.join(verb)} on a single geometry"):
        with pytest.raises(SystemExit) as exc:
            cli.main(verb + [str(p)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not a trajectory" in err, err
    assert "pyscf-geom" in err, (
        "the refusal must name the parser that DID read the file, so the "
        f"reader knows it is not a detection failure: {err!r}")

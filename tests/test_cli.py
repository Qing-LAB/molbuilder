"""CLI surface smoke tests.

Tests against the click-based CLI in ``molbuilder/cli.py``.  Three
roles:

  * every subcommand has a ``--help``
  * every subcommand parses the documented happy-path invocation
  * subcommand routing (``main(["X", ...]) -> proper handler``) works
    for the build verbs without hitting heavy external deps

Heavy dispatches (smiles needs RDKit, name needs PubChem) are tested via
mocks where reasonable and via ``--help`` only otherwise.  A deck is
rendered by `jobset prep` from a description; the rendered deck is the prep road's
(``tests/data/prep_protocol.toml``), value fidelity by the template
round-trip (``tests/test_template_roundtrip.py``).
"""

from __future__ import annotations


import numpy as np
import pytest

from molbuilder import cli


# --------------------------------------------------------------------- #
#  --help reachability for every subcommand                             #
# --------------------------------------------------------------------- #


_SUBCOMMANDS = [
    "peptide", "dna", "rna", "smiles", "name",
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
    monkeypatch.setattr("molbuilder.peptide.build_peptide", _make_capture(captured))
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
    monkeypatch.setattr("molbuilder.nucleic.build_dna", _make_capture(captured))
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
    monkeypatch.setattr("molbuilder.nucleic.build_dna", _make_capture(captured))
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
    monkeypatch.setattr("molbuilder.nucleic.build_rna", _make_capture(captured))
    cli.main(["rna", "AUGC", "--out", str(tmp_path / "r.xyz")])
    _seq, kwargs = captured[0]
    # When --form isn't passed, cli should not inject a default form
    # kwarg; the builder picks one.
    assert "form" not in kwargs


# --------------------------------------------------------------------- #
#  --pyscf-atom-block emission to stdout                                 #
# --------------------------------------------------------------------- #


def test_pyscf_atom_block_emits_to_stdout(monkeypatch, capsys, tmp_path):
    """The --pyscf-atom-block flag is a pipe-friendly stdout emitter
    documented in cli.py.  Verify it actually writes to stdout."""
    from molbuilder.structure import Structure
    monkeypatch.setattr(
        "molbuilder.peptide.build_peptide",
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


def test_validate_pretty_json_indents(capsys, tmp_path):
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    rc = cli.main(["validate", _write_xyz(tmp_path / "h2o.xyz", xyz),
                   "--pretty"])
    assert rc == 0
    out = capsys.readouterr().out
    # Pretty-printed JSON uses newlines + 2-space indent.
    assert "\n  " in out


# ---- Modify electrode-spec parser is case-insensitive on key ----- #

def test_modify_electrode_spec_key_case_insensitive():
    """``@CONTACT=`` / ``@Contact=`` are accepted (R3 fix: the parser
    lowercases the key).  Catches a regression where the ``.lower()`` call in
    ``_parse_electrode_spec`` is dropped."""
    upper_contact = cli._parse_electrode_spec("Au:111:3x3x2@CONTACT=2.4:+z=3")
    assert upper_contact["mode"] == "single"
    assert upper_contact["contact_distance"] == 2.4

    mixed_contact = cli._parse_electrode_spec("Au:111:3x3x2@Contact=2.4:-z=7")
    assert mixed_contact["mode"] == "single"
    assert mixed_contact["contact_distance"] == 2.4


# --------------------------------------------------------------------- #
#  Phase 5d: watch parse / tail subcommands                             #
# --------------------------------------------------------------------- #


def test_watch_tail_rejects_stdin(capsys):
    """`watch tail` needs a real file to poll -- stdin can't be
    re-read.  Reject with an explicit error rather than hanging."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["watch", "tail", "-"])
    assert exc.value.code == 2


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
    """Default --center='midpoint'."""
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
        # `config` because a Flask app has one, and `cmd_serve` records
        # THIS PROCESS's port on it (`web.app.serve_port`).  A stub that
        # stands in for a Flask app has to carry the parts of a Flask app
        # the caller uses.
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
    """`gap` is a PAIR's parameter -- the electrode-to-electrode distance,
    meaningless for one slab -- and a pair is two electrodes.

    Refused by name rather than falling into the generic "unknown key", which
    would read as a misspelling, and saying what to do instead.
    """
    import pytest as _pytest
    from molbuilder import cli
    with _pytest.raises(Exception, match=r"'@gap=' .*Pass --electrode twice"):
        cli._parse_electrode_spec("Au:111:3x3x2@gap=8.0:5,10")


def _run_electrode(tmp_path, st, spec):
    """Build `st`, run one `--electrode` flag over it, return the result."""
    from molbuilder.workingcopy_structure import StructureCodec
    inp = tmp_path / "in.xyz"
    StructureCodec().write(st, inp)
    out = tmp_path / "out.xyz"
    assert cli.main(["modify", str(inp), str(out), "--electrode", spec]) == 0
    return StructureCodec().load(out)


def _closest_metal_z(struct, above):
    zs = [float(p[2]) for e, p in zip(struct.elements, struct.positions)
          if e == "Au"]
    return min(z for z in zs if z > above)


def test_the_centroid_rule_moved_here_with_the_placement(tmp_path):
    """`--electrode ...:+z=I,J` centres on the CENTROID of the trailing index
    list — 1 index is that atom, 2 their midpoint, N their centroid.
    It is the CLI's, because the convenience is the CLI's; `add_slab`
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
    """The flag checks it, and says which index and how many atoms there
    are."""
    from molbuilder.structure import Structure
    st = Structure(elements=["S"], positions=np.array([[0., 0., 0.]]))
    inp = tmp_path / "in.xyz"
    from molbuilder.workingcopy_structure import StructureCodec
    StructureCodec().write(st, inp)
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

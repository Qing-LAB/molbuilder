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

import sys

import numpy as np
import pytest

from molbuilder import cli


# --------------------------------------------------------------------- #
#  --help reachability for every subcommand                             #
# --------------------------------------------------------------------- #


_SUBCOMMANDS = [
    "peptide", "dna", "rna", "smiles", "name",
    # "fdf" left with the verb (C2, 2026-08-11); "pyscf" stays until
    # decision 34 reworks the emitted-script path the same way.
    "pyscf",
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


def test_pyscf_reads_xyz_from_stdin(monkeypatch, tmp_path):
    """``molbuilder pyscf - out.py`` reads stdin, sniffs XYZ vs PDB from
    the first non-blank line, writes to a temp file, and feeds that into
    the standard convert() pipeline.  Without this you can't pipe
    ``molbuilder dna ATGC | molbuilder pyscf -``."""
    import io
    xyz = "2\nh2 stdin\nH 0 0 0\nH 0.74 0 0\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(xyz))
    out_py = tmp_path / "h2.py"
    rc = cli.main(["pyscf", "-", str(out_py), "--no-optimize",
                   "--no-density-fit"])
    assert rc == 0
    assert out_py.exists() and out_py.stat().st_size > 0


def test_pyscf_cli_exposes_review_fix_l_options(monkeypatch, tmp_path):
    """The cmd_pyscf subcommand surfaces 6 PySCFConfig fields that
    were silent CLI gaps before review-fix L:
        --diis-space, --damp, --ecp,
        --no-write-molwatch-log, --no-save-initial-xyz, --no-save-optimized-xyz.
    (Two former preopt-dispersion / preopt-density-fit flags retired
    with the preopt block in #534 commit 4b.)
    Smoke-check by running with all of them on a real input + asserting
    the generated script reflects each non-default setting."""
    import io
    xyz = "2\nh2 stdin\nH 0 0 0\nH 0.74 0 0\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(xyz))
    out_py = tmp_path / "h2.py"
    rc = cli.main([
        "pyscf", "-", str(out_py),
        "--no-optimize", "--no-density-fit",
        "--diis-space", "16",
        "--damp",       "0.4",
        "--ecp",        "lanl2dz",
        "--ecp-atoms",  "H",
        "--no-write-molwatch-log",
        "--no-save-initial-xyz",
        "--no-save-optimized-xyz",
    ])
    assert rc == 0
    text = out_py.read_text()
    # The mf.diis_space + mf.damp + ecp lines reflect the CLI values.
    assert "mf.diis_space = 16" in text
    assert "mf.damp = 0.4" in text
    # ONE shape since 2026-08-13: {element: name}, never a bare string.
    # The three-way ``or`` this replaces would have passed on the mere
    # substring "ecp=" anywhere in the file, comments included.
    assert "ecp        = {'H': 'lanl2dz'}," in text
    # molwatch / save toggles drop the corresponding code paths.
    assert "MolwatchEmitter" not in text
    assert "_initial.xyz" not in text
    # ``--no-save-optimized-xyz`` turns off the WRITE.  The READ is a separate
    # question and `restart` answers it: continuing is the default since
    # 2026-08-18, so a deck still reads a geometry a previous run left even
    # when it writes none of its own (`run-identity.md` § 4 rule 2 -- the two
    # gates are what "write a checkpoint but do not resume from one" needs).
    assert '_save_xyz(mol_eq, _mb_outfile(JOB + "_optimized.xyz")' not in text
    assert "_opt_path = _mb_outfile(JOB + \"_optimized.xyz\")" in text


def test_pyscf_cli_help_lists_all_review_fix_l_options():
    """Lighter sanity: --help mentions every new option name."""
    from click.testing import CliRunner
    runner = CliRunner()
    res = runner.invoke(cli.cli, ["pyscf", "--help"])
    assert res.exit_code == 0
    for flag in ("--diis-space", "--damp", "--ecp",
                 "--no-write-molwatch-log",
                 "--no-save-initial-xyz", "--no-save-optimized-xyz"):
        assert flag in res.output, f"missing {flag} in pyscf --help"
    # The preopt-* / geom-* flags are gone post-#534 commit 4b.
    for retired in ("--preopt-functional", "--preopt-basis",
                    "--preopt-max-steps", "--preopt-grms",
                    "--preopt-dispersion", "--no-preopt-density-fit",
                    "--geom-max-steps", "--geom-conv-energy",
                    "--geom-conv-grms", "--geom-conv-gmax"):
        assert retired not in res.output, (
            f"retired flag {retired} still appears in pyscf --help"
        )


def test_stdin_pdb_sniffs_correctly(monkeypatch, tmp_path):
    """Stdin sniff: a first line that isn't an integer is treated as
    PDB (HEADER / TITLE / ATOM / HETATM all qualify).  Ran through
    ``fdf`` until 2026-08-11; the sniff belongs to the shared stdin
    helper, so it repointed at pyscf rather than retiring."""
    import io
    pdb = (
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000  1.00  0.00           H\n"
        "ATOM      2  H   MOL A   1       0.740   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(pdb))
    out_py = tmp_path / "h2.py"
    rc = cli.main(["pyscf", "-", str(out_py), "--no-optimize",
                   "--no-density-fit"])
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
#  Phase 5e: add_dataclass_options decorator                            #
# --------------------------------------------------------------------- #


def test_add_dataclass_options_generates_click_flags(tmp_path, capsys):
    """The dataclass->click bridge maps field metadata onto click.option
    decorators: snake_case -> kebab-case flag, type from annotation,
    default from field default, help from metadata.  bool fields get
    a --foo / --no-foo pair.  Verify on a synthetic dataclass."""
    import click as _click
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        n_iter:   int   = dataclasses.field(default=10,
                                            metadata={"help": "iter count"})
        eps:      float = dataclasses.field(default=1e-3,
                                            metadata={"help": "tol"})
        verbose:  bool  = dataclasses.field(default=True,
                                            metadata={"help": "loud or quiet"})
        label:    str   = "default-label"

    seen = {}

    @_click.command()
    @cli.add_dataclass_options(_Cfg)  # the helper
    def demo(**kw):
        seen.update(kw)

    from click.testing import CliRunner
    runner = CliRunner()
    res = runner.invoke(demo, ["--n-iter", "20", "--eps", "1e-5",
                               "--no-verbose", "--label", "custom"])
    assert res.exit_code == 0, res.output
    assert seen == {"n_iter": 20, "eps": 1e-5, "verbose": False,
                    "label": "custom"}


def test_add_dataclass_options_skip_filters_fields(tmp_path):
    """`skip=` excludes named fields -- the corresponding click options
    don't appear on the command, so the underlying function doesn't
    receive them.  Used when a command already has those options
    defined manually (cf. cmd_pyscf today)."""
    import click as _click
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        a: int = 1
        b: int = 2

    @_click.command()
    @cli.add_dataclass_options(_Cfg, skip=("b",))
    def demo(**kw):
        return kw

    from click.testing import CliRunner
    runner = CliRunner()
    res = runner.invoke(demo, ["--b", "5"])
    # "--b" was skipped, so click rejects it as an unknown option.
    assert res.exit_code != 0
    assert "no such option" in res.output.lower() or "unrecognized" in res.output.lower()


def test_add_dataclass_options_tier_filters_advanced_only():
    """`tier="advanced"` keeps only fields whose metadata["tier"] is
    "advanced" -- the path to splitting basic vs advanced flag groups."""
    import click as _click
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        basic_a: int = dataclasses.field(default=1, metadata={"tier": "basic"})
        adv_b:   int = dataclasses.field(default=2, metadata={"tier": "advanced"})
        adv_c:   int = dataclasses.field(default=3, metadata={"tier": "advanced"})

    @_click.command()
    @cli.add_dataclass_options(_Cfg, tier="advanced")
    def demo(**kw):
        return kw

    from click.testing import CliRunner
    runner = CliRunner()
    # basic_a should be absent
    res = runner.invoke(demo, ["--basic-a", "10"])
    assert res.exit_code != 0
    # adv_b should be present
    res = runner.invoke(demo, ["--adv-b", "20", "--adv-c", "30"])
    assert res.exit_code == 0


def test_add_dataclass_options_works_on_real_pyscf_config():
    """End-to-end: the helper applies cleanly to PySCFConfig (real
    dataclass with mixed field types and metadata).  The generated
    --diis-space option lands as int with default 8 and a help line."""
    import click as _click
    from molbuilder.config.pyscf import PySCFConfig

    @_click.command()
    @cli.add_dataclass_options(PySCFConfig)
    def demo(**kw):
        return kw

    from click.testing import CliRunner
    runner = CliRunner()
    res = runner.invoke(demo, ["--help"])
    assert res.exit_code == 0
    out = res.output
    # The PySCFConfig fields land as kebab-case flags:
    for flag in ("--diis-space", "--damp", "--scf-conv-tol",
                 "--functional", "--basis"):
        assert flag in out, f"missing {flag} in --help output"


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


def test_pyscf_cli_exposes_every_non_skip_pyscf_field():
    """Bridge invariant for PySCFConfig (same idea as above)."""
    import dataclasses
    from click.testing import CliRunner
    from molbuilder.config.pyscf import PySCFConfig

    runner = CliRunner()
    res = runner.invoke(cli.cli, ["pyscf", "--help"])
    assert res.exit_code == 0, res.output
    out = res.output
    for fld in dataclasses.fields(PySCFConfig):
        if fld.metadata.get("skip_cli"):
            continue
        flag = "--" + fld.name.replace("_", "-")
        if fld.type in ("bool", bool):
            no_flag = "--no-" + fld.name.replace("_", "-")
            assert flag in out or no_flag in out, \
                f"Bridge dropped bool field {fld.name}"
        else:
            assert flag in out, f"Bridge dropped scalar field {fld.name}"


def _h2_xyz_at(path):
    path.write_text("2\nh2\nH 0 0 0\nH 0.74 0 0\n")
    return str(path)


def _stub_pyscf_summary(out_path):
    return {"py": str(out_path), "n_atoms": 2, "charge": 0, "label": "h2"}


@pytest.mark.parametrize("flag,cli_val,attr,expected", [
    # ONE case per value branch of ``add_dataclass_options`` -- the single
    # machine that generates every PySCF option from the dataclass.  A case
    # per FIELD proved nothing the branch case does not: all twenty rows of
    # the old sweep ran the same four code paths (collapsed 2026-08-19).
    # The rule: declared data and same-branch nouns earn no case of their
    # own; a distinct CODE PATH does.  The representatives are chosen for
    # the real edge inside their branch:
    ("--net-charge",   "-2",    "net_charge",   -2),      # int, NEGATIVE value after the flag
    ("--scf-conv-tol", "1e-10", "scf_conv_tol", 1e-10),   # float, scientific notation
    ("--functional",   "PBE",   "functional",   "PBE"),   # plain str
    ("--solvent",      "water", "solvent",      "water"), # Optional[str] -- the Union unwrap
])
def test_pyscf_cli_override_propagates_to_pyscf_config(
        flag, cli_val, attr, expected, monkeypatch, tmp_path):
    """A generated option's value reaches its PySCFConfig field."""
    captured = {}

    def fake_convert(input_path, py_path, config):
        captured["cfg"] = config
        return _stub_pyscf_summary(py_path)
    monkeypatch.setattr("molbuilder.pyscf.convert", fake_convert)

    in_xyz = _h2_xyz_at(tmp_path / "h2.xyz")
    out_py = tmp_path / "h2.py"
    rc = cli.main(["pyscf", in_xyz, str(out_py), flag, cli_val])
    assert rc == 0
    assert getattr(captured["cfg"], attr) == expected, (
        f"{flag} {cli_val!r}: expected PySCFConfig.{attr}={expected!r}, "
        f"got {getattr(captured['cfg'], attr)!r}"
    )


@pytest.mark.parametrize("attr,default,off_flag", [
    # The bool branch emits a --x/--no-x PAIR; its two directions are the
    # two code paths.  One case each (was ten same-path cases, collapsed
    # 2026-08-19).
    ("symmetry",    False, "--symmetry"),        # default False -> positive flag sets True
    ("density_fit", True,  "--no-density-fit"),  # default True  -> negative flag sets False
])
def test_pyscf_cli_bool_flags_round_trip(
        attr, default, off_flag, monkeypatch, tmp_path):
    """Both directions of the generated dual bool flag reach the config."""
    captured = []

    def fake_convert(input_path, py_path, config):
        captured.append(config)
        return _stub_pyscf_summary(py_path)
    monkeypatch.setattr("molbuilder.pyscf.convert", fake_convert)

    in_xyz = _h2_xyz_at(tmp_path / "h2.xyz")
    out_py = tmp_path / "h2.py"

    rc = cli.main(["pyscf", in_xyz, str(out_py), off_flag])
    assert rc == 0
    expected_after = not default
    assert getattr(captured[-1], attr) is expected_after, (
        f"{off_flag} did not flip {attr} to {expected_after!r}"
    )


# The 20-case "each default renders in the FDF" sweep that sat here was
# retired 2026-08-19.  Every case re-ran the section walk the deck-runner
# tests already pin, over values that are DECLARED DATA in the catalogue;
# the failure it feared -- a field wired to the CLI that the generator
# ignores -- is caught, for every field including tomorrow's, by
# tests/test_every_form_field_reaches_the_deck.py, which fails NAMING the
# field whenever changing it cannot change the deck.

# ---- Modify electrode-spec parser is case-insensitive on key ----- #


# ---- Bridge: metadata['choices'] -> click.Choice ---------------- #


def test_add_dataclass_options_emits_click_choice_when_metadata_set():
    """A dataclass field with ``metadata['choices']=...`` lands as a
    ``click.Choice`` option, so a typo fails at CLI parse time instead
    of waiting for the downstream tool (SIESTA / PySCF) to error out
    much later."""
    import click as _click
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        method: str = dataclasses.field(
            default="RKS",
            metadata={"choices": ("RKS", "UKS", "RHF", "UHF")},
        )

    @_click.command()
    @cli.add_dataclass_options(_Cfg)
    def demo(**kw):
        return kw

    from click.testing import CliRunner
    runner = CliRunner()

    # Bad value -> click error (exit code 2).
    res = runner.invoke(demo, ["--method", "FOO"])
    assert res.exit_code != 0
    out = (res.output or "").lower()
    assert "invalid value" in out or "not one of" in out, res.output

    # Good value passes through.
    res = runner.invoke(demo, ["--method", "UHF"])
    assert res.exit_code == 0


def test_add_dataclass_options_choices_appear_in_help():
    """``--help`` for a Choice-typed field shows the valid options so
    the CLI is self-documenting."""
    import click as _click
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        method: str = dataclasses.field(
            default="RKS",
            metadata={"choices": ("RKS", "UKS", "RHF")},
        )

    @_click.command()
    @cli.add_dataclass_options(_Cfg)
    def demo(**kw):
        return kw

    from click.testing import CliRunner
    res = CliRunner().invoke(demo, ["--help"])
    assert res.exit_code == 0
    # click renders Choice options as "[a|b|c]" in --help.  With R2's
    # ``case_sensitive=False`` the rendered list is lowercased, so we
    # match case-insensitively.
    out = res.output.lower()
    assert "rks" in out and "uks" in out and "rhf" in out
    # And the default still shows in the original case.
    assert "RKS" in res.output


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


def test_add_dataclass_options_rejects_unknown_field_type():
    """A dataclass field with an annotation the bridge can't handle
    (Sequence[str], Tuple, dict, ...) must raise TypeError instead of
    silently coercing to ``type=str``.  The user is expected to mark
    such fields ``skip_cli=True`` and hand-roll a click.option at the
    call site."""
    import click as _click
    import dataclasses
    from typing import Sequence

    @dataclasses.dataclass
    class _Cfg:
        species: Sequence[str] = dataclasses.field(default_factory=list)

    with pytest.raises(TypeError) as exc:
        @_click.command()
        @cli.add_dataclass_options(_Cfg)
        def demo(**kw):
            return kw

    msg = str(exc.value)
    assert "species" in msg
    assert "skip_cli" in msg


def test_add_dataclass_options_skip_cli_field_is_not_type_checked():
    """A field marked ``skip_cli=True`` is exempt from the type check
    -- the bridge doesn't try to generate an option for it, so weird
    types are fine."""
    import click as _click
    import dataclasses
    from typing import Sequence

    @dataclasses.dataclass
    class _Cfg:
        species: Sequence[str] = dataclasses.field(
            default_factory=list,
            metadata={"skip_cli": True},
        )

    # Must NOT raise.
    @_click.command()
    @cli.add_dataclass_options(_Cfg)
    def demo(**kw):
        return kw


def test_add_dataclass_options_choices_on_bool_is_an_error():
    """``choices=`` on a bool field is meaningless (bool fields surface
    as a ``--foo / --no-foo`` pair).  Bridge must reject the
    combination loudly."""
    import click as _click
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        flag: bool = dataclasses.field(
            default=True, metadata={"choices": ("true", "false")},
        )

    with pytest.raises(TypeError) as exc:
        @_click.command()
        @cli.add_dataclass_options(_Cfg)
        def demo(**kw):
            return kw
    assert "choices" in str(exc.value)


def test_modify_electrode_spec_key_case_insensitive():
    """``@GAP=`` / ``@Gap=`` / ``@CONTACT=`` are accepted (R3 fix:
    the parser lowercases the key).  Catches a regression where the
    ``.lower()`` call in ``_parse_electrode_spec`` is dropped --
    earlier mutation testing showed 0 test failures when this was
    silently removed."""
    upper = cli._parse_electrode_spec("Au:111:3x3x2@GAP=8.0:5,10")
    assert upper["mode"] == "pair"
    assert upper["gap"] == 8.0

    mixed = cli._parse_electrode_spec("Au:111:3x3x2@Gap=8.0:5,10")
    assert mixed["mode"] == "pair"
    assert mixed["gap"] == 8.0

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


def test_modify_electrode_pair_mode_one_call(tmp_path):
    """Pair mode: one --electrode flag, two slabs (one +z, one -z),
    @gap= sets electrode-to-electrode distance."""
    inp  = tmp_path / "in.xyz"
    oriented = tmp_path / "oriented.xyz"
    outp = tmp_path / "junction.xyz"
    inp.write_text(_bdt_stub_xyz())
    cli.main(["modify", str(inp), str(oriented), "--orient-axis", "0,3"])
    rc = cli.main(["modify", str(oriented), str(outp),
                   "--electrode", "Au:111:3x3x2@gap=9.0:3,0"])
    assert rc == 0
    text = outp.read_text()
    # 4 molecule atoms + 9 atoms/layer × 2 layers × 2 sides = 4 + 36
    n = int(text.splitlines()[0])
    assert n == 4 + 36


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


def test_modify_electrode_stepped_two_pair_specs(tmp_path):
    """Stepped 3×3 close + 4×4 far on both sides -- two --electrode flags
    in one call (both pair-mode, geometrically independent)."""
    inp  = tmp_path / "in.xyz"
    oriented = tmp_path / "oriented.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    cli.main(["modify", str(inp), str(oriented), "--orient-axis", "0,3"])
    rc = cli.main(["modify", str(oriented), str(outp),
                   "--electrode", "Au:111:3x3x1@gap=9.0:3,0",
                   "--electrode", "Au:111:4x4x1@gap=14.0:3,0"])
    assert rc == 0
    text = outp.read_text()
    n = int(text.splitlines()[0])
    # 4 molecule + (9 + 16) × 2 sides = 4 + 50
    assert n == 4 + 50


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
        cli.cli, ["serve", "--no-auth", "--host", "0.0.0.0", "--port", "8099",
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
        def run(self, **kw):
            calls["run_kwargs"] = kw

    def _fake_create_app(*, config=None):
        calls["config"] = config
        return _FakeApp()

    monkeypatch.setattr(_appmod, "create_app", _fake_create_app)
    res = CliRunner().invoke(
        cli.cli, ["serve", "--no-auth", "--host", "127.0.0.1", "--port", "8099",
                  "--no-supervise"])
    assert res.exit_code == 0, res.output
    assert calls["config"] == {}                 # no-auth seam
    assert calls["run_kwargs"].get("ssl_context") is None   # plain http

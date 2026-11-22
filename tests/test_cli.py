"""CLI surface smoke tests.

Tests against the click-based CLI in ``molbuilder/cli.py``.  Three
roles:

  * every subcommand has a ``--help``
  * every subcommand parses the documented happy-path invocation
  * a build verb builds what was typed -- read off what the real builder
    built, a backend not installed skipped (`engines/builders.md`)

A name lookup needs PubChem over the network and is reached by ``--help``
only.  A deck is
rendered by `jobset prep` from a description; the rendered deck is the prep road's
(``tests/data/prep_protocol.toml``), value fidelity by the template
round-trip (``tests/test_template_roundtrip.py``).
"""

from __future__ import annotations


import re

import numpy as np
import pytest

from support import tooling

from molbuilder import cli


# --------------------------------------------------------------------- #
#  --help reachability for every subcommand                             #
# --------------------------------------------------------------------- #


def test_every_command_of_the_tree_answers_help():
    """Every command's ``--help`` answers 0, walked from the tree itself so a
    command added later is asked too.  THE SILENT FAILURE: a click
    misconfiguration a command's own tests never reach, so a person asking
    what a verb does meets a traceback.  `process/conventions.md` § 3."""
    import click
    from click.testing import CliRunner

    def walk(cmd, path):
        r = CliRunner().invoke(cli.cli, path + ["--help"], prog_name="molbuilder")
        assert r.exit_code == 0, (path, r.output)
        if isinstance(cmd, click.Group):
            for sub, child in cmd.commands.items():
                walk(child, path + [sub])
    walk(cli.cli, [])


def test_every_verb_a_groups_help_names_runs():
    """A group's help, as printed, sends a person only to commands that run
    (`process/conventions.md` § 3, the CLI-wide conventions): each command
    it names -- marked ````name````, in that group or from the root, or a
    `molbuilder …` line of an example -- resolves to a command, and that
    command's ``--help`` answers 0.

    THE SILENT FAILURE: a help that tells a person to run a verb that was
    renamed or retired -- they type what the help said and meet a usage
    error, sent there by the help itself.

    Every group of the command tree, walked from the tree itself.

    MUTATIONS THIS MUST FAIL AGAINST: a group's own verb renamed in its
    help; another group's verb renamed where a help names it; a verb
    renamed in an example line."""
    import re

    import click
    from click.testing import CliRunner

    root = cli.cli

    def resolved(words, group):
        """The command path ``words`` names from ``group``, or None -- the
        words a group's subcommands name, down to the first command."""
        cmd, path = group, []
        for word in words:
            if not isinstance(cmd, click.Group):
                break
            if word not in cmd.commands:
                return None
            cmd = cmd.commands[word]
            path.append(word)
        return path

    def helped(path):
        r = CliRunner().invoke(root, path + ["--help"], prog_name="molbuilder")
        assert r.exit_code == 0, (path, r.output)
        return r.output

    named, gone = [], []

    def check(where, words, group, group_path):
        rel = resolved(words, group)
        full = (group_path + rel) if rel is not None else resolved(words, root)
        named.append(" ".join(words[:len(full or words)]))
        if full is None or CliRunner().invoke(root, full + ["--help"]).exit_code:
            gone.append(f"{where}: {' '.join(words)}")

    def walk(group, path):
        text = helped(path)
        where = " ".join(["molbuilder"] + path)
        for name in set(re.findall(r"``([^`]+)``", text)):
            if "." not in name:
                check(where, name.split(), group, path)
        for line in text.splitlines():
            words = line.split()
            if words[:1] == ["molbuilder"] and len(words) > 1:
                # From the root, down while each word names a subcommand:
                # the first word past a command is its argument.
                check(where, words[1:], root, [])
        for sub, cmd in group.commands.items():
            if isinstance(cmd, click.Group):
                walk(cmd, path + [sub])
    walk(root, [])
    assert named, "no group's help names a command"
    assert not gone, f"a help names commands that do not run: {sorted(gone)}"


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
#  Build subcommands: what was typed reaches the builder                 #
#                                                                        #
#  Read off what the real builder built -- the pair the call wrote,      #
#  opened through the codec -- and what it said on stderr, against what  #
#  `engines/builders.md` states; never off a recorder standing in for    #
#  the builder, which passed while a flag changed nothing.  The real     #
#  builders answer in seconds here (measured 2026-11-14: 0.3-3.7 s); a   #
#  row whose backend is not installed is skipped, said why.              #
# --------------------------------------------------------------------- #


def _built(tmp_path, *typed):
    """``molbuilder <typed> --out <pair>``: what it built, read back."""
    from molbuilder.workingcopy_structure import StructureCodec
    pair = tmp_path / "built.xyz"
    assert cli.main([*typed, "--out", str(pair)]) == 0
    return StructureCodec().load(pair)


def _residues(built):
    """The residue names in order, one a residue (`builders.md`, *The uniform
    output contract*: a build carries ``residue_ids`` and ``residue_names``
    an atom)."""
    return [name for _, name in dict.fromkeys(zip(built.residue_ids,
                                                  built.residue_names))]


@pytest.mark.parametrize("tools, typed, residues, said", [
    # A peptide built residue by residue, its C-terminus with no OXT said on
    # stderr as every finding is (`builders.md` § 1, `validation.md` R5).
    ((), ("peptide", "ARNDC"),
     ["ALA", "ARG", "ASN", "ASP", "CYS"], {"build.peptide_aldehyde"}),
    # The backend typed is the one that built: RDKit embeds a conformer, and
    # the form a DNA is asked when none is typed (§ 4, ``DEFAULT_FORM``) is
    # said not built (§ 5).
    (("rdkit",), ("dna", "ATGC", "--backend", "rdkit"),
     ["DA", "DT", "DG", "DC"], {"build.form"}),
    (("rdkit",), ("rna", "AUGC", "--backend", "rdkit"), None, {"build.form"}),
    # An RNA is built A-form, its own, with nothing to say (§ 3).
    (("threedna",), ("rna", "AUGC"), None, set()),
], ids=["peptide", "dna-backend", "rna-backend", "rna-own-form"])
def test_a_build_subcommand_builds_what_was_typed(tmp_path, capsys, tools,
                                                  typed, residues, said):
    """Each build subcommand reaches its builder with the sequence, the
    backend and the form typed: what was built carries the residues the
    sequence names, and the build says exactly the findings `builders.md`
    states for what it could not build as asked."""
    for tool in tools:
        tooling.needs_backend(tool)
    built = _built(tmp_path, *typed)
    if residues:
        assert _residues(built) == residues
    err = capsys.readouterr().err
    assert set(re.findall(r"warn \[([a-z_.]+)\]", err)) == said, err


@pytest.mark.parametrize("typed, absent, words", [
    # The grammar refuses "with the position" (`builders.md` § 2).
    (("peptide", "A-C"), None, "position 1"),
    # "A SMILES that holds no atom is refused" (§ 6).
    (("smiles", ""), None, "no atom"),
    # A tool this install lacks, named (§ 1); the engines made to fail the
    # one way a test does it (`support.tooling.make_absent`).
    (("peptide", "GG"), "hydrogens", "OpenBabel"),
], ids=["a-builder-refuses", "an-empty-input", "a-tool-this-install-lacks"])
def test_a_builders_refusal_is_its_error_message(tmp_path, capsys,
                                                 monkeypatch, typed, absent,
                                                 words):
    """`builders.md` § 1: a refusal of a build (`chemistry.BuildRefused`) or
    a tool this install lacks (`chemistry.BackendUnavailable`) reaches the
    command line through `cli.main`'s one door, as its ``Error:`` message and
    exit 1 -- the builder's own words, never a traceback -- after the line
    every call opens with, the version (the user, 2026-10-18), and nothing
    is written.  The silent failures (F12's review, 2026-11-18): every
    builder refusal ended in a traceback, and an empty SMILES in a Python
    error from the engine's insides.

    MUTATION THIS MUST FAIL AGAINST: `cli.main` without either type in its
    door."""
    from molbuilder import __version__
    if absent:
        tooling.make_absent(monkeypatch, absent)
    pair = tmp_path / "built.xyz"
    with pytest.raises(SystemExit) as ended:
        cli.main([*typed, "--out", str(pair)])
    assert ended.value.code == 1
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == f"molbuilder {__version__}", lines
    assert lines[1].startswith("Error:") and words in lines[1], lines
    assert "Traceback" not in "\n".join(lines), lines
    assert not pair.exists()


def test_a_build_says_the_structures_own_findings(tmp_path, capsys):
    """`builders.md` § 8: a build command prints the structure's own
    findings beside every structure it builds, every severity -- atoms on
    top of each other among them, as a SMILES of several molecules is built
    (the user, 2026-11-18: "we have an atom collision detection, that's the
    warning we should have").  The silent failure: ``O.O`` written as two
    waters 0.00 A apart, the command saying nothing -- the check never ran
    here, and the printer dropped every error it was not raising."""
    pair = tmp_path / "two.xyz"
    cli.main(["smiles", "O.O", "--out", str(pair)])
    err = capsys.readouterr().err
    assert "warn [geometry.min_distance]" in err, err
    assert pair.exists()


def test_a_title_typed_is_the_title_built(tmp_path):
    """``--title`` reaches the builder: what was built is called by it."""
    assert _built(tmp_path, "peptide", "A", "--title", "my run").title == (
        "my run")


def test_a_form_typed_is_the_form_built(tmp_path):
    """``--form`` reaches the DNA builder: A, the compact form (`builders.md`'s
    *Vocabulary*), is shorter end to end than the B built by default -- the first
    phosphate to the last, a distance no axis convention enters."""
    tooling.needs_backend("threedna")

    def span(*typed):
        built = _built(tmp_path, "dna", "ATGC", *typed)
        p = built.positions[[i for i, e in enumerate(built.elements)
                             if e == "P"]]
        return float(np.linalg.norm(p[-1] - p[0]))
    assert span("--form", "A") < span()


@pytest.mark.parametrize("strand", [("dna", "ATGC"), ("rna", "AUGC")],
                         ids=["dna", "rna"])
def test_a_terminal_typed_is_the_terminal_built(tmp_path, strand):
    """``--terminal 5P`` reaches the builder: the fiber's 5'-phosphate is kept,
    where the both-ends-OH default strips it (`builders.md` § 5) -- one
    phosphorus more."""
    tooling.needs_backend("threedna")

    def phosphori(*typed):
        return list(_built(tmp_path, *strand, *typed).elements).count("P")
    assert phosphori("--terminal", "5P") == phosphori() + 1


@pytest.mark.parametrize("strand", [("dna", "ATGC"), ("rna", "AUGC")],
                         ids=["dna", "rna"])
def test_no_protonate_phosphates_leaves_each_phosphate_charged(tmp_path,
                                                               strand):
    """``--no-protonate-phosphates`` reaches the builder: by default an H is
    added to each deprotonated non-bridging phosphate oxygen (`builders.md`
    § 7); with the flag none is -- one H fewer a phosphate, three between
    four bases.  On the backend that leaves them deprotonated, tleap (§ 7)."""
    tooling.needs_backend("amber")

    def hydrogens(*typed):
        return list(_built(tmp_path, *strand, "--backend", "amber",
                           *typed).elements).count("H")
    assert hydrogens() - hydrogens("--no-protonate-phosphates") == 3


# --------------------------------------------------------------------- #
#  --pyscf-atom-block emission to stdout                                 #
# --------------------------------------------------------------------- #


def test_pyscf_atom_block_emits_to_stdout(capsys, tmp_path):
    """`--pyscf-atom-block` writes what was built to stdout as PySCF's atom
    block, ``<element> <x> <y> <z>`` an atom, for a person to pipe into a
    PySCF script.  THE SILENT FAILURE: the flag taken and the block never
    written -- the build still says what it built, on stderr.  The real
    peptide builder; the block read against the pair the same call wrote."""
    from molbuilder.workingcopy_structure import StructureCodec
    pair = tmp_path / "ala.xyz"
    rc = cli.main(["peptide", "A", "--out", str(pair), "--pyscf-atom-block"])
    assert rc == 0
    built = StructureCodec().load(pair)
    rows = [ln.split() for ln in capsys.readouterr().out.splitlines()
            if ln.strip()]
    assert [r[0] for r in rows] == list(built.elements), rows
    assert np.allclose([[float(v) for v in r[1:4]] for r in rows],
                       built.positions, atol=1e-4)


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
    assert body["engine"] is None



def test_validate_refuses_a_kind_its_engine_does_not_run(capsys, tmp_path):
    """`molbuilder validate --engine pyscf --calculation transport`: refused
    as its flags are, naming the engines the kind runs on (the kind table's
    one door), never a traceback -- until 2.64.0 the gate's refusal escaped
    the command line as one."""
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", _write_xyz(tmp_path / "h2o.xyz", xyz),
                  "--engine", "pyscf", "--calculation", "transport"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "transport" in err and "siesta" in err, err


def test_validate_judges_a_kind_on_what_its_description_starts_from(
        capsys, tmp_path):
    """`molbuilder validate --engine siesta --calculation vibration` judges
    the config a description of that kind starts from with nothing stated --
    the one default (`engines/template.md` § 5.4, § 6.3a) -- so the
    relaxation it says runs first runs to the tolerance the kind recommends.
    THE SILENT FAILURE: the config class's general default judged instead --
    through 2.69.42 the finding named 0.02 eV/Å for a vibration that runs
    at 0.01.  The structure is the road's: a peptide built by the command
    line."""
    import json
    from molbuilder.template import catalogue, default_of, one
    pair = tmp_path / "ala.xyz"
    assert cli.main(["peptide", "A", "--out", str(pair)]) == 0
    capsys.readouterr()
    assert cli.main(["validate", str(pair), "--engine", "siesta",
                     "--calculation", "vibration"]) == 0
    issues = json.loads(capsys.readouterr().out)["issues"]
    (said,) = [i["message"] for i in issues
               if i["where"] == "config.already_relaxed"]
    tol = default_of(one(catalogue(), "relax_force_tol", engine="siesta"),
                     "vibration")
    assert f"{tol:g} eV/Å" in said, said


def test_validate_exit_on_error_returns_2(monkeypatch, capsys, tmp_path):
    """--exit-on-error makes the command non-zero when any error-severity
    issue fires: a structure whose cell is mirrored (`cell.left_handed`),
    saved as a pair through the codec."""
    import json
    import numpy as np
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    bad = tmp_path / "bad.xyz"
    StructureCodec().write(Structure(
        elements=["H", "H"], positions=np.array([[5.0, 5, 5], [5, 5, 5.74]]),
        cell=np.array([[0, 10.0, 0], [10.0, 0, 0], [0, 0, 10.0]])), bad)
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", str(bad), "--exit-on-error"])
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

@pytest.mark.parametrize("key,side", [("CONTACT", "+z"), ("Contact", "-z")])
def test_modify_electrode_spec_key_case_insensitive(tmp_path, key, side):
    """``@CONTACT=`` / ``@Contact=`` are ``@contact=`` (R3: the key's case is
    not significant): `molbuilder modify --electrode` puts the slab on the
    side named, its closest layer the stated 2.4 Å from its anchor.  THE
    FAILURE: a key in capitals refused as an unknown key."""
    from molbuilder.structure import Structure
    st = Structure(elements=["S", "S"],
                   positions=np.array([[0., 0., 0.], [0., 0., 2.]]))
    anchor = 1 if side == "+z" else 0
    out = _run_electrode(tmp_path, st,
                         f"Au:111:2x2x1@{key}=2.4:{side}={anchor}")
    metal = [float(p[2]) for e, p in zip(out.elements, out.positions)
             if e == "Au"]
    at = float(st.positions[anchor][2])
    gap = min(metal) - at if side == "+z" else at - max(metal)
    assert gap == pytest.approx(2.4, abs=1e-6), (side, metal)


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
    with pytest.raises(SystemExit) as exc:
        cli.main(["modify", str(inp), str(outp),
                  "--delete", "1",
                  "--orient-axis", "0,3"])
    # A usage error's code: a run that ended any other way exits too, so
    # `SystemExit` alone proves nothing.
    assert exc.value.code == 2
    assert not outp.exists()


def test_modify_orient_only(tmp_path):
    """Default --center='midpoint'."""
    inp  = tmp_path / "in.xyz"
    outp = tmp_path / "out.xyz"
    inp.write_text(_bdt_stub_xyz())
    rc = cli.main(["modify", str(inp), str(outp),
                   "--orient-axis", "0,3", "--yes"])
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
                   "--delete", "1", "--yes",
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
    cli.main(["modify", str(inp), str(oriented), "--orient-axis", "0,3",
              "--yes"])
    rc = cli.main(["modify", str(oriented), str(outp), "--yes",
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
    rc = cli.main(["modify", str(inp), "-", "--delete", "1", "--yes"])
    assert rc == 0
    captured = capsys.readouterr()
    stdout_xyz = captured.out
    assert stdout_xyz.startswith("3\n"), (
        f"expected 3-atom XYZ on stdout; got:\n{stdout_xyz!r}"
    )
    # Step 2: feed that text in via stdin to a second invocation
    monkeypatch.setattr("sys.stdin", io.StringIO(stdout_xyz))
    rc = cli.main(["modify", "-", str(outp), "--orient-axis", "0,2", "--yes"])
    assert rc == 0
    final = outp.read_text()
    assert final.startswith("3\n")


#: The rows of the agreement: the file, the atom index typed, what is said,
#: and the atoms left.  An engine's record -- the road's CO2, written in the
#: engine's held-first order with its order record -- is opened in your
#: order, said naming the file and the order in its words (§ 2.2a, I8), so
#: index 0 is your first oxygen, not the carbon.
_AGREEMENT_ROWS = [
    dict(name="a file in the user order", remapped=False, delete="1",
         said=("count from 0", "atom index 1 (C)", "--yes"),
         left=["S", "C", "S"]),
    dict(name="an engine's record, opened in the user order: the indices count "
              "in the user order", remapped=True, delete="0",
         said=("in.xyz", "held atoms first", "count from 0",
               "count in the user order", "atom index 0 (O)", "--yes"),
         left=["O", "C"]),
]


@pytest.mark.parametrize("row", _AGREEMENT_ROWS,
                         ids=[r["name"] for r in _AGREEMENT_ROWS])
def test_modify_asks_to_agree_that_atom_indices_count_from_0(row, tmp_path):
    """`model/overview.md` § 2: the atom indices typed at the CLI are the
    code's own, counted from 0, while the UI's atom list numbers the same
    atoms from 1 -- so `modify` says so, names the atoms it was given, and
    acts only once that is agreed: with no one to ask (no terminal) it is
    refused, naming `--yes`, and writes nothing; agreed, the atom at that
    index is the one that goes.  `modify` is an editor (§ 2.2a, I8): a file
    in an engine's order is opened in the user order, said, and the indices count in
    yours."""
    from click.testing import CliRunner
    from molbuilder.workingcopy_structure import StructureCodec
    inp, outp = tmp_path / "in.xyz", tmp_path / "out.xyz"
    if row["remapped"]:
        from support.road import co2_in_engine_order
        StructureCodec().write(co2_in_engine_order(), inp, engine_record=True)
    else:
        inp.write_text(_bdt_stub_xyz())                 # S C C S
    refused = CliRunner().invoke(cli.cli, ["modify", str(inp), str(outp),
                                           "--delete", row["delete"]])
    assert refused.exit_code != 0, refused.output
    for words in row["said"]:
        assert words in refused.output, (words, refused.output)
    assert not outp.exists()
    agreed = CliRunner().invoke(cli.cli, ["modify", str(inp), str(outp),
                                          "--delete", row["delete"], "--yes"])
    assert agreed.exit_code == 0, agreed.output
    assert StructureCodec().load(outp).elements == row["left"]


def test_modify_refuses_a_file_of_several_frames(tmp_path):
    """`model/structure.md` § 2.3: an edit acts on one structure, and the
    command line has no one to ask which frame -- so a file of several
    frames is refused, naming how many and the ways to take one out, and
    nothing is written."""
    from click.testing import CliRunner
    inp, outp = tmp_path / "in.xyz", tmp_path / "out.xyz"
    inp.write_text("2\na\nH 0 0 0\nH 0 0 0.74\n2\nb\nH 0 0 0\nH 0 0 0.80\n")
    res = CliRunner().invoke(cli.cli, ["modify", str(inp), str(outp),
                                       "--rotate", "z:90"])
    assert res.exit_code != 0, res.output
    for words in ("holds 2 frames", "frame=i", "Modify tab"):
        assert words in res.output, (words, res.output)
    assert not outp.exists()


# --------------------------------------------------------------------- #
#  serve --no-auth: loopback-only guard (security-relevant)             #
# --------------------------------------------------------------------- #


def test_serve_no_auth_refuses_non_loopback_host(monkeypatch):
    """--no-auth must REFUSE a non-loopback bind so an unauthenticated
    server is never exposed off the machine.

    The app is a stand-in that records a start: were the refusal to regress,
    the real one would bind 0.0.0.0 here and never return."""
    from click.testing import CliRunner
    import molbuilder.web.app as _appmod

    started = []

    class _App:
        config: dict = {}

        def run(self, **kw):
            started.append(kw)

    monkeypatch.setattr(_appmod, "create_app", lambda **_: _App())
    res = CliRunner().invoke(
        cli.cli, ["serve", "foreground", "--no-auth", "--host", "0.0.0.0", "--port", "8099",
                  "--no-supervise"])
    assert res.exit_code != 0
    assert "loopback" in res.output.lower()
    assert not started


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
# `--electrode`: one slab a flag
# ---------------------------------------------------------------------------

def _run_electrode(tmp_path, st, spec):
    """Build `st`, run one `--electrode` flag over it, return the result."""
    from molbuilder.workingcopy_structure import StructureCodec
    inp = tmp_path / "in.xyz"
    StructureCodec().write(st, inp)
    out = tmp_path / "out.xyz"
    assert cli.main(["modify", str(inp), str(out), "--electrode", spec,
                     "--yes"]) == 0
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
        cli.main(["modify", str(inp), str(tmp_path / "o.xyz"), "--yes",
                  "--electrode", "Au:111:2x2x1@contact=2.4:+z=5"])


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
            "modify", str(src), str(out), "--yes",
            "--electrode", f"Au:111:3x3x3@contact=2.4{reg}:+z=1"])
        assert res.exit_code == 0, res.output
        p = np.asarray(StructureCodec().load(out).positions, dtype=float)
        up = p[p[:, 2] > 1.5]
        made[tag] = up[np.isclose(up[:, 2], up[:, 2].min())][:, :2].mean(axis=0)

    step = float(np.linalg.norm(made["a"] - made["b"]))
    assert step > 1.0, (
        f"the registry never reached the builder: the starting layer moved "
        f"{step:.4f} Å")


@pytest.mark.parametrize("engine", [None, "siesta"])
def test_validate_refuses_a_mirrored_cell_with_or_without_an_engine(
        capsys, tmp_path, engine):
    """`science/validation.md` § 4.1: a box that is no box -- a mirrored,
    left-handed lattice -- is an error whatever will be run on it
    (*representability is blocking*), so `molbuilder validate` says
    `cell.left_handed` and exits 2 under --exit-on-error with no engine named
    as with one.  The silent failure (plan A1.1): with no --engine the
    structure's box was never checked -- n_errors 0, exit 0 -- while
    --engine siesta refused the same file."""
    import json

    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    mirrored = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, -10.0]])
    path = StructureCodec().write(
        Structure(elements=["H", "H"],
                  positions=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.74]]),
                  cell=mirrored), tmp_path / "h2.xyz")
    with pytest.raises(SystemExit) as exc:
        cli.main(["validate", str(path), "--exit-on-error",
                  *(["--engine", engine, "--calculation", "optimization"]
                    if engine else [])])
    assert exc.value.code == 2
    body = json.loads(capsys.readouterr().out)
    assert [i["where"] for i in body["issues"]
            if i["severity"] == "error"] == ["cell.left_handed"], body


def _every_command(group=cli.cli, path=()):
    """Each command's words under ``group``, groups included."""
    for name, command in sorted(getattr(group, "commands", {}).items()):
        yield (*path, name)
        yield from _every_command(command, (*path, name))


def _command_at(path):
    command = cli.cli
    for name in path:
        command = command.commands[name]
    return command


@pytest.mark.parametrize("columns", [80, 52], ids=lambda c: f"{c}-columns")
@pytest.mark.parametrize("path", list(_every_command()), ids=" ".join)
def test_help_names_the_program_as_a_person_types_it(capsys, monkeypatch,
                                                     path, columns):
    """`job-system.md` § 5.3: each command's ``--help`` says molbuilder's
    command as a person types it (`support.road.TYPED`, the chapter's
    words) -- its ``Usage:`` line and every example -- never click's own
    guess, ``python -m molbuilder``, nor a bare ``molbuilder <verb>`` that
    no shell finds; and each such line on one line, whole, a line a person
    copies: never refilled into a run-on paragraph, never broken where the
    help's column ends, on a wide terminal or a narrow one, its prose after
    ``#`` (*one command a line*)."""
    from support.road import TYPED
    monkeypatch.setenv("COLUMNS", str(columns))
    with pytest.raises(SystemExit) as done:
        cli.main([*path, "--help"])
    assert done.value.code == 0
    said = capsys.readouterr().out
    usage = next((ln for ln in said.splitlines()
                  if ln.lower().startswith("usage: ")), "")
    assert usage[len("usage: "):].startswith(f"{TYPED} {' '.join(path)}"), said
    verbs = "|".join(map(re.escape, cli.cli.commands))
    bare = re.compile(rf"(?<![\w./-])(?<!-m )molbuilder (?:{verbs})\b")
    assert not bare.search(said), said
    lines = said.splitlines()
    assert " ".join(said.split()).count(TYPED) == sum(
        ln.count(TYPED) for ln in lines), f"a line broken mid-program:\n{said}"
    for ln in lines:
        for at in (m.start() for m in re.finditer(re.escape("`" + TYPED), ln)):
            assert "`" in ln[at + 1:], f"a line broken mid-command: {ln}\n{said}"
    for ln in lines:
        typed = ln.strip()
        outside = re.sub(r"`[^`]*`", "", typed)
        if TYPED in outside and not typed.lower().startswith("usage: "):
            # a shell's continuation of the line before (`&&`, `|`) aside
            assert re.sub(r"^(&&|\|\|?)\s*", "", outside).startswith(TYPED), (
                f"a typed line not first on its line: {ln}\n{said}")
        if typed.startswith(TYPED + " "):
            assert not re.search(r"\S {2,}[^#\s]", typed[len(TYPED):]), (
                f"prose on a typed line, not after `#`: {ln}\n{said}")
    printed = {ln.strip() for ln in lines}
    written = [ln.strip() for ln in (_command_at(path).help or "").splitlines()
               if ln.strip().startswith(TYPED + " ")]
    for line in written:
        assert line in printed, f"refilled, not printed whole: {line}\n{said}"

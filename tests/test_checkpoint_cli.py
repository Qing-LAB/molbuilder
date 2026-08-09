"""`molbuilder snapshot` — the verbs a person actually types (§ 5).

**Contract:** [`checkpointing.md`](?doc=execution/checkpointing.md) § 5 (the
commands and the two labels) · § 5.1 (what `list` must show) · § 7.1 · A4 · A5.

These test the CLI as its own surface, not as a thin wrapper: a `Repo` that
behaves and a CLI that hides it are the same failure to a user. So the
assertions are on **what is printed and what the exit code is**, since those are
the whole of what the CLI promises.

Retired verbs get their own test. A verb that is merely undocumented is still a
verb somebody's script calls, and the contract removed these outright.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from molbuilder.cli import cli

BIG = b"\x01" * 5000


@pytest.fixture()
def calc(tmp_path, checkpoint_config):
    """The classification is server-wide (S1c); the folder carries none."""
    checkpoint_config(size_limit_bytes=1024, engines={"generic": []})
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    return root


@pytest.fixture()
def mb(calc):
    runner = CliRunner()

    def invoke(*args):
        return runner.invoke(cli, ["snapshot", *args, "-p", str(calc)])
    return invoke


def _ids(output):
    """The 7-character state ids a `list` printed, newest first."""
    ids = []
    for line in output.splitlines():
        parts = line.split()
        if parts and parts[0] == "->":
            parts = parts[1:]
        if parts and len(parts[0]) == 7 and parts[0].isalnum():
            ids.append(parts[0])
    return ids


# ------------------------------------------------------------------ #
#  § 5 — the six verbs, and only those                                #
# ------------------------------------------------------------------ #


def test_the_group_offers_exactly_the_contracts_verbs():
    out = CliRunner().invoke(cli, ["snapshot", "--help"]).output
    for verb in ("init", "save", "list", "tag", "restore", "config"):
        assert f"\n  {verb}" in out, f"{verb} is missing from the group"


@pytest.mark.parametrize("verb", ["checkpoint", "branch", "migrate-manifest"])
def test_a_retired_verb_is_gone_not_merely_undocumented(mb, verb):
    """`checkpoint` became `save`; `branch` and `migrate-manifest` were
    removed by the contract.  A verb left working is a verb a script calls."""
    assert mb(verb).exit_code != 0


def test_no_surface_offers_a_text_only_restore(mb):
    """A4: a restore returns the whole folder or it does not happen.

    `--no-binaries` rewound the text and left every big file, which is a folder
    no save ever produced — reached on purpose rather than by accident.
    """
    mb("init")
    assert "--no-binaries" not in mb("restore", "--help").output
    assert mb("restore", "HEAD", "--no-binaries").exit_code != 0


# ------------------------------------------------------------------ #
#  init                                                               #
# ------------------------------------------------------------------ #


def test_init_reports_the_calculation_and_the_first_state(mb):
    result = mb("init")
    assert result.exit_code == 0
    assert "BDT_Au_relax" in result.output, "L3: a state names its calculation"
    assert len(_ids(result.output)) == 1


def test_init_twice_is_not_an_error_and_says_where_you_stand(mb):
    mb("init")
    result = mb("init")
    assert result.exit_code == 0
    assert "already" in result.output.lower()


def test_a_calculation_name_needing_repair_is_refused(tmp_path):
    root = tmp_path / "has spaces!"
    root.mkdir()
    result = CliRunner().invoke(cli, ["snapshot", "init", "-p", str(root)])
    assert result.exit_code != 0


# ------------------------------------------------------------------ #
#  save — the note is required (L3)                                   #
# ------------------------------------------------------------------ #


def test_save_without_a_note_is_refused(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("x\n")
    assert mb("save").exit_code != 0


def test_save_reports_the_state_it_made(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("x\n")
    result = mb("save", "-m", "stage 1 converged")
    assert result.exit_code == 0
    assert "stage 1 converged" in result.output


def test_save_says_plainly_when_nothing_changed(mb):
    mb("init")
    result = mb("save", "-m", "nothing happened")
    assert result.exit_code == 0
    assert "nothing changed" in result.output.lower()


# ------------------------------------------------------------------ #
#  § 5.1 — what `list` has to show                                    #
# ------------------------------------------------------------------ #


def test_list_shows_the_note_the_parent_and_where_you_stand(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("one\n")
    mb("save", "-m", "stage 1 converged, 41 steps")
    out = mb("list").output

    assert "stage 1 converged, 41 steps" in out, "the note is the point"
    assert "from " in out, "each state says where it came from"
    assert "->" in out, "where the folder stands must be visible"


def test_list_shows_a_fork_as_two_states_from_one_parent(mb, calc):
    """§ 7.1 read off the surface a person actually uses."""
    mb("init")
    (calc / "job.XV").write_text("one\n")
    mb("save", "-m", "stage 1")
    stage1 = _ids(mb("list").output)[0]
    (calc / "job.XV").write_text("200 Ry\n")
    mb("save", "-m", "attempt A")
    mb("restore", stage1)
    (calc / "job.XV").write_text("300 Ry\n")
    mb("save", "-m", "attempt B")

    out = mb("list").output
    assert "attempt A" in out and "attempt B" in out
    assert out.count(f"from {stage1}") == 2, (
        "both attempts must show the same parent, or the list cannot say they "
        "are alternatives")


def test_list_shows_tags_and_flags_unsaved_work(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("one\n")
    mb("save", "-m", "stage 1")
    mb("tag", "stage1-good", "-m", "geometry I trust")
    assert "stage1-good" in mb("list").output

    (calc / "job.XV").write_text("edited\n")
    assert "unsaved" in mb("list").output.lower()


def test_list_on_a_folder_that_is_not_a_checkpoint_folder_says_so(mb):
    result = mb("list")
    assert result.exit_code != 0
    assert "init" in result.output


# ------------------------------------------------------------------ #
#  restore — A5 through the surface                                   #
# ------------------------------------------------------------------ #


def test_restore_names_what_will_be_lost_and_refuses(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("saved\n")
    mb("save", "-m", "stage 1")
    first = _ids(mb("list").output)[0]
    (calc / "job.XV").write_text("unsaved\n")
    (calc / "new.bin").write_bytes(BIG)

    result = mb("restore", first)
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert "job.XV" in combined and "new.bin" in combined, (
        "a count tells nobody anything; the files must be named")
    assert "--force" in combined, "the way through must be in the message"


def test_at_a_terminal_the_question_is_asked_and_yes_is_an_answer(
        mb, calc, monkeypatch):
    """§ 7's flowchart has a QUESTION, and `yes` is one of its two ways through.

    The CLI only ever offered `--force`: it printed the files and exited, so
    the interactive path the contract describes -- and that this command's own
    help promised -- did not exist.  A person had to retype the whole command
    with a flag to answer a question they had just been asked.
    """
    import molbuilder.cli as cli_mod
    mb("init")
    (calc / "job.XV").write_text("saved\n")
    mb("save", "-m", "stage 1")
    first = _ids(mb("list").output)[0]
    (calc / "job.XV").write_text("unsaved\n")

    # The runner replaces sys.stdin wholesale, so the terminal check is the
    # seam -- not stdin itself.
    monkeypatch.setattr(cli_mod, "_stdin_is_a_terminal", lambda: True)
    result = CliRunner().invoke(
        cli, ["snapshot", "restore", first, "-p", str(calc)], input="y\n")
    assert result.exit_code == 0, result.output
    assert (calc / "job.XV").read_text() == "saved\n"


def test_answering_no_changes_nothing(mb, calc, monkeypatch):
    """The answer is honoured in both directions -- that is what makes it a
    question rather than a formality."""
    import molbuilder.cli as cli_mod
    mb("init")
    (calc / "job.XV").write_text("saved\n")
    mb("save", "-m", "stage 1")
    first = _ids(mb("list").output)[0]
    (calc / "job.XV").write_text("unsaved\n")

    monkeypatch.setattr(cli_mod, "_stdin_is_a_terminal", lambda: True)
    result = CliRunner().invoke(
        cli, ["snapshot", "restore", first, "-p", str(calc)], input="n\n")
    assert result.exit_code != 0
    assert (calc / "job.XV").read_text() == "unsaved\n", (
        "answering no must leave the folder exactly as it was")


def test_restore_with_force_completes(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("saved\n")
    mb("save", "-m", "stage 1")
    first = _ids(mb("list").output)[0]
    (calc / "job.XV").write_text("unsaved\n")

    assert mb("restore", first, "--force").exit_code == 0
    assert (calc / "job.XV").read_text() == "saved\n"


def test_restore_accepts_a_tag(mb, calc):
    mb("init")
    (calc / "job.XV").write_text("good\n")
    mb("save", "-m", "the geometry I trust")
    mb("tag", "stage1-good", "-m", "trusted")
    (calc / "job.XV").write_text("elsewhere\n")
    mb("save", "-m", "elsewhere")

    assert mb("restore", "stage1-good").exit_code == 0
    assert (calc / "job.XV").read_text() == "good\n"


def test_restore_of_an_unknown_state_fails_clearly(mb):
    mb("init")
    result = mb("restore", "no-such-state")
    assert result.exit_code != 0


# ------------------------------------------------------------------ #
#  config — explains, rather than dumps                               #
# ------------------------------------------------------------------ #


def test_config_shows_the_limit_and_which_side_goes_where(mb):
    mb("init")
    out = mb("config").output
    assert "1024" in out, "the size limit must be visible as a number"
    assert "archive" in out and "git" in out, (
        "which side of the limit goes where is the thing being explained")
    assert "molbuilder.json" in out, (
        "S1c: the classification has one home and the user must be told where")


def test_config_names_the_calculation(mb):
    mb("init")
    assert "BDT_Au_relax" in mb("config").output

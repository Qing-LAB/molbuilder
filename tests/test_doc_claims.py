"""What a contract states, run through the code: `block_size`'s two states,
the vocabulary a declared type must come from, and the Task setup tab
refusing a folder with two templates.
"""
from __future__ import annotations


from molbuilder import template
from molbuilder.jobset._cli import describe_cmd


# --------------------------------------------------------------------- #
#  Gate A -- a worked example must be a VALID example                    #
#                                                                        #
#  audit-2026-08-14 3.1/3.2: the contract's own TOML carried a DUPLICATE  #
#  KEY (so it did not parse at all), one example omitted `category` and   #
#  two omitted `type` and `help` -- all four of them keys 3 calls         #
#  REQUIRED on every item.  A contract whose illustrations its own reader #
#  would refuse teaches the wrong shape to everyone who copies one.       #
# --------------------------------------------------------------------- #


def test_block_size_is_a_plain_int():
    """The two states of ``block_size`` (tuning.md § 2.11, revised
    2026-08-15): absent = *auto*, the keyword is not emitted and SIESTA uses
    its own automatic; ``N`` = use N, verbatim.

    IT IS NOT ``pow2``, and that is the point of this test.  ``pow2`` does not
    check -- ``template._shape`` SNAPS the value down to the nearest power of
    two -- so a benchmarked 24 silently became 16 and nothing recorded why.
    The power-of-two rule is real but belongs to a different keyword under a
    narrower condition: the manual states it for ``Diag.BlockSize``, only with
    a GPU-enabled ELPA, where breaking it is not even an error (ELPA falls
    back to the CPU).  `prep` realigns it there, where the GPU flag and the
    rank count are both known; `pow2` survives in BENCH-MARKS, a constraint
    the benchmark puts on its own sweep.
    """

    item = template.one(template.catalogue(), "block_size", engine="siesta")
    assert item.type == "int", (
        "block_size must be a plain int -- `pow2` silently rewrites "
        "a measured value, which is the opposite of honouring a benchmark")


# --------------------------------------------------------------------- #
#  The template file has ONE door                                        #
# --------------------------------------------------------------------- #

def test_a_folder_holding_two_templates_is_refused_by_the_tab_not_picked(
        web_client, isolated_projects_root, monkeypatch):
    """**A calculation folder with a second template: the Task-setup tab
    refuses and names both** -- it never picks one.

    `engines/template.md` § 4.3 / `job-contracts.md` § 6.3 name the file
    ``<label>.template.toml``: ``template_file.template_path`` forms it and
    ``template_file.of`` finds it, refusing a folder that holds two.
    SEVEN call sites formed it themselves until 2026-08-17, in two
    incompatible ways -- from ``task.json``'s label, and by
    ``sorted(glob("*.template.toml"))[0]`` -- so a folder holding two
    templates had the web tab and ``prep`` reading DIFFERENT FILES, and the
    values the tab showed were not the values the job would run.

    Asked of the route a person's tab calls, on a calculation `molbuilder describe`
    wrote, with a leftover template dropped beside it.

    MUTATION THIS MUST FAIL AGAINST: the tab's lookup going back to
    ``sorted(folder.glob("*.template.toml"))[0]``.
    """
    from click.testing import CliRunner

    from molbuilder.template_file import template_path

    tree = isolated_projects_root
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.chdir(tree.parent)
    r = CliRunner().invoke(describe_cmd, [
        "--structure", "P/structure/h2.xyz",
        "--bundle", "P/optimization/H2", "--engine", "pyscf",
        "--calculation", "optimization", "--name", "H2"])
    assert r.exit_code == 0, r.output
    calc = tree / "P" / "optimization" / "H2"
    own = template_path(calc, "H2")
    assert own.is_file(), sorted(p.name for p in calc.iterdir())
    (calc / "x.template.toml").write_text(own.read_text())

    resp = web_client.get(f"/api/task-setup/folder?dir={calc}")
    body = resp.get_json()["template"]
    assert body["ok"] is False, (
        f"a folder holding two templates was answered: {body}")
    assert own.name in body["error"] and "x.template.toml" in body["error"], (
        f"the refusal must name BOTH templates so the person can remove the "
        f"leftover: {body['error']!r}")

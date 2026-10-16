"""What a contract states, run through the code: `block_size`'s two states,
the vocabulary a declared type must come from, the Task setup tab refusing a
folder with two templates, and the allocation example `engines/stages.md`
§ 6.8a prints, read back by the reader that stores it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from molbuilder import template


# --------------------------------------------------------------------- #
#  Gate A -- a worked example must be a VALID example                    #
#                                                                        #
#  audit-2026-08-14 3.1/3.2: the contract's own TOML carried a DUPLICATE  #
#  KEY (so it did not parse at all), one example omitted `category` and   #
#  two omitted `type` and `help` -- all four of them keys 3 calls         #
#  REQUIRED on every item.  A contract whose illustrations its own reader #
#  would refuse teaches the wrong shape to everyone who copies one.       #
# --------------------------------------------------------------------- #


def test_block_size_is_a_plain_int_and_survives_the_round_trip():
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
    from molbuilder.config.siesta import SiestaConfig

    item = template.one(template.read_template(
        template.template_with_values(SiestaConfig())), "block_size")
    assert item.type == "int", (
        "block_size must be a plain int -- `pow2` silently rewrites "
        "a measured value, which is the opposite of honouring a benchmark")

    def _round_trip(v):
        return template.one(template.read_template(template.template_with_values(
            SiestaConfig(block_size=v))), "block_size").value

    # Every value a user may set survives untouched, power of two or not.
    for v in (1, 16, 24, 64, 96, 100, 128):
        assert _round_trip(v) == v, f"{v} was rewritten"

    # Absent stays absent -- that is how *auto* is said, and it is what makes
    # the deck omit the keyword.
    assert _round_trip(None) is None


def test_a_declared_type_must_be_in_the_vocabulary():
    """``metadata['decl_type']`` is checked against ``TYPES`` -- a typo there
    would otherwise put an unknown type into every template."""
    import dataclasses
    fld = dataclasses.field(metadata={"decl_type": "not_a_type",
                                      "category": ("execution",),
                                      "help": "x"})
    fld.name, fld.type = "x", int
    with pytest.raises(ValueError, match="not in the type vocabulary"):
        template.declaration_for(fld, int)


# --------------------------------------------------------------------- #
#  The template file has ONE door                                        #
# --------------------------------------------------------------------- #

def test_a_folder_holding_two_templates_is_refused_by_the_tab_not_picked(
        web_client, isolated_projects_root, monkeypatch):
    """**A calculation folder with a second template: the Task-setup tab
    refuses and names both** -- it never picks one.

    `engines/template.md` § 4.3 / `job-contracts.md` § 6.3 name the file
    ``<label>.template.toml``: ``template.template_path`` forms it and
    ``template.find_template`` finds it, refusing a folder that holds two.
    SEVEN call sites formed it themselves until 2026-08-17, in two
    incompatible ways -- from ``task.json``'s label, and by
    ``sorted(glob("*.template.toml"))[0]`` -- so a folder holding two
    templates had the web tab and ``prep`` reading DIFFERENT FILES, and the
    values the tab showed were not the values the job would run.

    Asked of the route a person's tab calls, on a calculation `jobset init`
    wrote, with a leftover template dropped beside it.

    MUTATION THIS MUST FAIL AGAINST: the tab's lookup going back to
    ``sorted(folder.glob("*.template.toml"))[0]``.
    """
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.template import template_path

    tree = isolated_projects_root
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "h2.xyz").write_text(
        "2\nh2\nH 0 0 0\nH 0 0 0.74\n")
    monkeypatch.chdir(tree.parent)
    r = CliRunner().invoke(jobset_group, [
        "init", "--structure", "P/structure/h2.xyz",
        "--bundle", "P/optimization/H2", "--engine", "pyscf",
        "--calculation", "optimization",
        "--shape", "flat", "--name", "H2"])
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


def test_the_allocation_example_in_its_OWNING_contract_is_what_the_reader_stores():
    """§ 6.8a of `engines/stages.md` owns the `allocation` block, and the
    example it prints must be the spelling the reader actually stores.

    **This is the drift-direction check, not a spelling check.** On
    2026-08-24 the record's spelling changed by user directive, and the
    change was made in the code and in `task.py`'s docstring -- a
    RESTATEMENT -- while § 6.8a went on saying *"values are spelled as a
    person types them"* and printing `"time": "4h"`.  Fixing a
    code-vs-contract finding by editing the code and the nearest comment
    is how the contract ends up the last place telling the truth.

    Feeding the document's own example through the reader closes that: if
    either side moves, they stop agreeing here.
    """
    import json as _json
    import re
    from molbuilder.task import _allocation_from_obj
    root = Path(__file__).resolve().parents[1]
    doc = (root / "docs" / "engines" / "stages.md").read_text(encoding="utf-8")
    body = doc[doc.index("### 6.8a"):]
    m = re.search(r"```json\n(.*?)\n```", body, re.S)
    assert m, "§ 6.8a's allocation example could not be found -- repoint this"
    example = _json.loads("{" + m.group(1).rstrip().rstrip(",") + "}"
                          if not m.group(1).strip().startswith("{")
                          else m.group(1))
    assert "allocation" in example, (
        f"§ 6.8a's example is not an allocation block: {example!r}")

    stored = _allocation_from_obj(example)
    documented = example["allocation"]
    for field in ("time", "mem", "domain"):
        if documented.get(field):
            assert getattr(stored, field) == documented[field], (
                f"§ 6.8a prints allocation.{field} = "
                f"{documented[field]!r}, but the reader stores "
                f"{getattr(stored, field)!r}.  The contract that OWNS this "
                f"key and the code that reads it disagree about what the "
                f"file holds.")

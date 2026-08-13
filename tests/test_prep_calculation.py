"""`prep`, entire — the five steps, on the machine that will run it.

Contract: ``docs/execution/project-layout.md`` § 2.3.1 (the five steps, and why
their order is forced) · ``docs/execution/generator.md`` § 5 (the element, and
why the deck writer reads the allocation) · ``docs/execution/architecture.md``
§ 4.1.

**This file is the migration's proof.** Steps 2 and 3 did not exist inside
`prep` until 2026-08-11: the decks were finished at `molbuilder fdf` time, on a
machine that could not know the rank count, and `prep` refused unless they were
already in the bundle. Everything here would have been impossible to write.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from molbuilder import describe as D
from molbuilder.config.siesta import SiestaConfig
from molbuilder.jobset.model import Resources
from molbuilder.jobset.prep import PrepError, prep_calculation
from molbuilder.siesta.stages import default_siesta_stages
from molbuilder.structure import Structure


@pytest.fixture
def calc(tmp_path):
    """A described calculation, exactly as `jobset describe` leaves it.

    The activation config is the **dotted, bundle-scoped**
    ``.molbuilder.json`` — the project scope the wrapper writer resolves
    from the script's own directory.  An undotted ``molbuilder.json`` here
    is INERT (that name is the cwd-first server scope), and until 2026-08-12
    this fixture wrote exactly that: the tests then silently resolved
    ``script_generation.activation`` from the developer's repo-root config —
    14 of 24 failed under an isolated cwd+HOME while green in-repo.
    """
    struct = Structure(elements=["S", "C", "C", "H"],
                       positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.78],
                                           [1.21, 0.0, 2.48], [2.15, 0.0, 1.94]]),
                       vacuum=(10.0, 10.0, 10.0))
    src = tmp_path / "bdt.xyz"
    src.write_text(struct.to_xyz())
    dest = tmp_path / "calc"
    desc = D.build_description(
        struct, SiestaConfig(system_label="calc", mesh_cutoff=300.0),
        default_siesta_stages("publishable"),
        engine="siesta", shape="hierarchical", name="calc", source=str(src))
    D.write_description(desc, dest)
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "source /opt/conda/etc/profile.d/conda.sh"}}))
    return dest


# --------------------------------------------------------------------- #
#  The migration — prep RENDERS the deck                                 #
# --------------------------------------------------------------------- #

def test_prep_renders_the_deck_it_used_to_demand(calc):
    """**§ 9.3's one real migration.**  `prep` refused unless the deck was
    already in the bundle root — *"render the inputs before prep"* — because
    the producer ran at *produce*.  A described calculation carries no deck at
    all, so this passing is the migration."""
    assert not list(calc.glob("*.fdf"))
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=32))
    assert (calc / "calc_01_coarse.fdf").is_file()


def test_the_five_steps_all_leave_their_mark(calc):
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    assert (calc / "environment.json").is_file()        # 1 resolve the machine
    assert (calc / "calc_01_coarse.fdf").is_file()      # 3 render the deck
    assert (calc / "calc_01_coarse.run.sh").is_file()   # 4 render the wrapper
    assert (calc / "01_coarse").is_dir()                # 5 build the directory


def test_floor_three_is_written_by_prep_not_read_by_it(calc):
    """`describe` writes floor 2 only.  Until `prep` derived floor 3 from it,
    there was nothing to derive it from and nothing that did."""
    assert not (calc / "job-set.json").exists()
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    js = json.loads((calc / "job-set.json").read_text())
    assert [j["name"] for j in js["jobs"]] == ["coarse"]
    assert js["jobs"][0]["script"] == "calc_01_coarse.fdf"


# --------------------------------------------------------------------- #
#  The deck and the launch agree BY CONSTRUCTION                         #
# --------------------------------------------------------------------- #

def test_the_deck_records_the_rank_count_it_was_rendered_for(calc):
    """**The `-np 14` class of failure, made unconstructible.**

    A deck derives values from the rank count — ``BlockSize`` above all — so a
    deck rendered without one says ``mpi_np auto`` and then gets launched at 32.
    Both now come from one resolved element, so they cannot disagree.
    """
    from molbuilder.parse.scripts.bench_marks import _extract_bench_marks_dict
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=32))
    marks = _extract_bench_marks_dict(
        (calc / "calc_01_coarse.fdf").read_text())
    assert marks.get("mpi_np") == 32


def test_the_launch_agreement_holds_for_a_deck_prep_just_made(calc):
    """`check_launch_matches_deck` refuses a deck rendered for another launch.
    A deck `prep` itself rendered must never trip it — and it did, on the first
    run of this path, because the deck was rendered from the values alone."""
    from molbuilder.jobset.agreement import launch_agreement
    from molbuilder.jobset._cli import _load
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=32))
    js, _ = _load(str(calc))
    assert launch_agreement(calc, js.jobs[0]).verdict == "agrees"


def test_the_allocation_reaches_the_jobs_resources(calc):
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=16, cpus_per_task=4,
                                          max_memory_mb=4096))
    js = json.loads((calc / "job-set.json").read_text())
    r = js["jobs"][0]["resources"]
    assert (r["mpi_np"], r["cpus_per_task"], r["max_memory_mb"]) == (16, 4, 4096)


# --------------------------------------------------------------------- #
#  Step 2 really resolved — the stage's own science is in the deck       #
# --------------------------------------------------------------------- #

def test_the_named_stages_overrides_are_what_got_rendered(calc):
    """`coarse` is CG at a loose force tolerance and `medium` is Broyden at a
    tighter one.  Rendering the wrong rung would be silent, so this names the
    values rather than asserting a file exists."""
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    coarse = (calc / "calc_01_coarse.fdf").read_text()
    assert "MD.TypeOfRun CG" in coarse
    assert "MD.MaxForceTol 0.05" in coarse

    prep_calculation(calc, "medium", allocation=Resources(mpi_np=8))
    medium = (calc / "calc_02_medium.fdf").read_text()
    assert "MD.TypeOfRun Broyden" in medium
    assert "MD.MaxForceTol 0.04" in medium


def test_the_deck_carries_its_stages_token_and_header(calc):
    """The stage's artifact token reaches the rendered deck: every filename
    molbuilder chooses picks it up, and the ``# Stage <token> --`` line names
    the science of the config being rendered, so the comment cannot drift from
    the keywords below it (decision 27).  Gated here because `prep` is the
    token's producer now -- ``molbuilder fdf --stage N`` was, until it went."""
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    coarse = (calc / "calc_01_coarse.fdf").read_text()
    assert "calc_01_coarse.out" in coarse
    assert "# Stage 01_coarse --" in coarse


def test_the_template_supplies_what_no_stage_varies(calc):
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    assert "MeshCutoff 300.0" in (calc / "calc_01_coarse.fdf").read_text()


# --------------------------------------------------------------------- #
#  Refusals                                                              #
# --------------------------------------------------------------------- #

def test_a_folder_with_no_description_is_refused_by_name(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PrepError, match=r"jobset describe"):
        prep_calculation(tmp_path / "empty", "coarse",
                         allocation=Resources(mpi_np=8))


def test_a_structure_that_changed_since_describing_is_refused(calc, tmp_path):
    """§ 6.3's witness earning its place: the description records a formula and
    an atom count, so building a *different* calculation under the same id is
    caught rather than discovered in the results.

    The mutated file is the CALCULATION'S OWN copy — `describe` copies the
    structure in since 2026-08-12 (M9's walk found nothing made "beside the
    calculation first" true), and that copy is what `prep` reads."""
    (calc / "bdt.xyz").write_text(
        Structure(elements=["H", "H"],
                  positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
                  vacuum=(10.0, 10.0, 10.0)).to_xyz())
    with pytest.raises(PrepError, match=r"structure has changed"):
        prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))


def test_a_stage_that_is_not_in_the_ladder_is_refused(calc):
    with pytest.raises(PrepError, match=r"coarse"):
        prep_calculation(calc, "nonesuch", allocation=Resources(mpi_np=8))


# --------------------------------------------------------------------- #
#  The warm-retry budget — § 6.2's one no-flag row, on THIS route        #
# --------------------------------------------------------------------- #

def test_the_warm_retry_budget_travels_the_described_route(calc):
    """A-5 (final review, 2026-08-13): `job-contracts.md § 6.2`'s last row
    says ``continue_retries`` is *"translated at resolve.py — rides the
    element's Resources; prep bakes it into the wrapper"*.  Until the fix
    nothing performed that translation on this route: the web route hands
    the value straight to the wrapper writer, while the described route
    resolved an allocation that never carried it — so the wrapper rendered
    NO retry loop and `job-system.md § 4.1`'s "travels the whole way" was
    true of one road out of two.

    Pinned on the EMITTED TEXT because the defect class is a value that
    travels correctly and is dropped at the last hop.  Both precedences:
    the template's answer rides by default; an explicit allocation wins."""
    tpl = calc / "calc.template.toml"
    head, sep, tail = tpl.read_text().partition("[item.continue_retries]")
    assert sep, "the template lost its continue_retries item"
    body, nxt, rest = tail.partition("\n[item.")
    assert "value = 1" in body, body
    tpl.write_text(head + sep + body.replace("value = 1", "value = 4", 1)
                   + nxt + rest)
    prep_calculation(calc, "coarse", allocation=Resources(mpi_np=8))
    text = (calc / "calc_01_coarse.run.sh").read_text()
    assert "_siesta_retry_max=4" in text, (
        "the template's warm-retry budget never reached the wrapper -- "
        "the § 6.2 translation at resolve.py is broken again (A-5)")
    # an explicitly stated allocation wins over the template's answer
    prep_calculation(calc, "coarse",
                     allocation=Resources(mpi_np=8, continue_retries=2))
    text = (calc / "calc_01_coarse.run.sh").read_text()
    assert "_siesta_retry_max=2" in text

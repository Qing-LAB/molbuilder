"""The setting door -- one API for what is done to a setting's value, asked
by every place a value lives (`docs/engines/template.md` § 5.4).

PINS: § 5.4's questions -- may it stand here (``accepts``: the home, then
``null``, then the catalogue's type, then § 5.3's door), what it is,
spelled (``read_text``), the one default (``default_of``);
`execution/job-request.md` § 3.2 (``null`` on a card is *unset*, for an
item whose unset is an answer) and § 3.2a (a card's wall spelled as the
allocation's).

PREVENTS, each found by the 2.27.0 review and read in its code (plan § 1w):
a run card's value checked for no type and no choice; a card's ``null``
reaching ``int()`` at prep; a shared or rung-fixed override passing the
description's check and refused first at prep; a page reading ``0x10`` as a
number and any text but ``true`` as false; a card's wall stored as typed
while the allocation's is spelled; a benchmark row naming what every trial
fixes saved and then overruled at prep.

Category: API -- the door asked, its answer read; and the job request's
reader and prep's card read through it.  The homes' roads are the save
table in ``test_hard_limits.py`` and the rows in
``data/launch_values.toml``.  A RULE CHANGES, ITS ROWS CHANGE.
"""
from __future__ import annotations

import pytest

from molbuilder.template import (accepts, catalogue, default_of, one,
                                 read_text, recommended_for, why_not)


def _item(name, engine="siesta"):
    return one(catalogue(), name, engine=engine)


#: ``(item, home, kind, stage, value, refused with -- None: it stands)``.
ACCEPTS = [
    # A RUN CARD: the catalogue's type and choices, its limits, `null` only
    # for an item whose unset is an answer.
    ("mpi_np", "run_card", "optimization", None, 8, None),
    ("mpi_np", "run_card", "optimization", None, 8.0, None),
    ("mpi_np", "run_card", "optimization", None, "8",
     ", which is not a whole number"),
    ("mpi_np", "run_card", "optimization", None, 0,
     ": it must be greater than 0"),
    ("gpu_count", "run_card", "optimization", "relax", None, None),
    ("use_gpu", "run_card", "optimization", None, None,
     ", but this setting has no unset"),
    ("use_gpu", "run_card", "optimization", None, "yes",
     ", which is not true or false"),
    ("diag_algorithm", "run_card", "optimization", None, "elpa",
     ", which is not one of 'ScaLAPACK', 'ELPA-1STAGE', 'ELPA-2STAGE'"),
    ("gpu_count", "run_card", "transport", "transmission", 2,
     ", which only the seed / electrode_L / electrode_R / device rungs of a "
     "transport calculation read, not 'transmission'"),
    ("ts_solution_method", "run_card", "optimization", None, "BTD",
     ", which an optimization does not carry"),
    ("mesh_cutoff", "run_card", "optimization", None, 300.0,
     ", which is not an execution setting of siesta"),
    # A BENCHMARK ROW: a run setting a trial may vary, or the comparison
    # measures another calculation at each point; a point is a value.
    ("block_size", "bench", "optimization", None, 16, None),
    ("block_size", "bench", "optimization", None, None,
     ", but a benchmark row's points are values to try"),
    ("mesh_cutoff", "bench", "optimization", None, 300.0,
     ", which siesta does not declare as an `execution` parameter"),
    ("continue_retries", "bench", "optimization", None, 1,
     ", which every benchmark trial fixes at 0"),
    ("max_memory_mb", "bench", "optimization", None, 4000,
     ", which a benchmark's grid does not sweep"),
    # A STAGE'S OVERRIDE: what is computed, not shared, not fixed, read by
    # its rung; then its type and § 5.3's door.
    ("mpi_np", "override", "optimization", "tight", 8, ", a run setting"),
    ("net_charge", "override", "optimization", "tight", 1,
     ", which is shared by every stage of this calculation"),
    ("solution_method", "override", "transport", "device", "diagon",
     ", which the rung fixes"),
    ("relax_steps", "override", "optimization", "tight", 2.5,
     ", which is not a whole number -- this setting counts things"),
    ("kgrid", "override", "optimization", "tight", "4,4,1",
     ", which is not three whole numbers, written as a list"),
    ("kgrid", "override", "optimization", "tight", [4, 4, 0],
     ": each component must be greater than 0"),
    ("kgrid", "override", "optimization", "tight", [4.0, 4, 1], None),
    # ...§ 5.3's door: a choice the kind does not offer, a component it
    # fixes.
    ("relax_type", "override", "vibration", "relax", "none",
     ", which a vibration does not offer"),
    # A PREP FLAG: the launch shape alone; no cap is the row left blank.
    ("max_memory_mb", "flag", "optimization", None, 0,
     ": it must be greater than 0"),
    ("diag_algorithm", "flag", "optimization", None, "ScaLAPACK",
     ", which no prep flag states"),
]


@pytest.mark.parametrize("name, home, kind, stage, value, refused", ACCEPTS)
def test_a_value_stands_or_is_refused_once_in_its_home(name, home, kind,
                                                       stage, value, refused):
    """One value, one answer, wherever it lives (§ 5.4): the clause read
    after *"<who> sets <name> = <value>"*, or none.

    MUTATIONS THIS MUST FAIL AGAINST: the home check dropped (a run setting
    stands as an override; a GPU count stands on the transmission's card);
    `null` taken by every item (``use_gpu: null`` stands); the type judged
    on the raw value (``8.0`` refused for a count) or not at all (``"8"``
    stands)."""
    said = accepts(_item(name), value, home, engine="siesta", kind=kind,
                   stage=stage)
    if refused is None:
        assert said is None, said
    else:
        assert said is not None and said.startswith(refused), said


def test_a_template_value_meets_the_rules_of_its_kind():
    """The template names no kind, so the kind's rules are asked of its
    values where it meets its kind (§ 5.3, `template.why_not`, asked by
    `resolve`): a transport calculation's third k component is fixed at 1,
    and a template stating 2 is refused rather than run."""
    said = why_not(_item("kgrid"), [4, 4, 2], engine="siesta",
                   kind="transport")
    assert said is not None and said.startswith(
        ", whose z component a transport calculation fixes at 1"), said


#: ``(item, engine, what was sent, read as -- or None, refused with)``.
READ = [
    ("mpi_np", "siesta", " 8 ", 8, None),
    ("mpi_np", "siesta", "1e3", 1000, None),
    ("mpi_np", "siesta", "0x10", None, ", which is not a whole number"),
    ("max_scf_iter", "siesta", "2.5", None, ", which is not a whole number"),
    ("mesh_cutoff", "siesta", "300", 300.0, None),
    ("mesh_cutoff", "siesta", "1_000", None, ", which is not a number"),
    ("mesh_cutoff", "siesta", 300, 300.0, None),
    ("use_gpu", "siesta", "false", False, None),
    ("use_gpu", "siesta", "True", None, ", which is not true or false"),
    ("unpaired_electrons", "siesta", "2", 2, None),
    ("unpaired_electrons", "siesta", "free", "free", None),
    ("basis_size", "siesta", "XYZ", None, ", which is not one of 'SZ'"),
    ("kgrid", "siesta", "4,4,1", (4, 4, 1), None),
    ("kgrid", "siesta", "4x4x1", (4, 4, 1), None),
    ("kgrid", "siesta", "4 4 1", (4, 4, 1), None),
    ("kgrid", "siesta", [4, 4, 1], (4, 4, 1), None),
    ("kgrid", "siesta", "4,4", None, ", which is not three whole numbers"),
    ("kgrid", "siesta", "[4, 4, 1]", (4, 4, 1), None),
    ("kgrid_displacement", "siesta", "0.5,0.5,0.0", (0.5, 0.5, 0.0), None),
    ("species_order", "siesta", "Au,C, H S", ["Au", "C", "H", "S"], None),
    ("mpi_np", "siesta", "  ", None, None),
]


@pytest.mark.parametrize("name, engine, sent, value, refused", READ)
def test_what_a_page_sends_is_read_as_its_type(name, engine, sent, value,
                                               refused):
    """Every page reads with the door (§ 5.4) -- a cell, a card row, a
    form: a number in decimal, never ``0x10`` or ``1_000``; ``true`` /
    ``false`` and nothing else; an enum's member by its spelling, ``"2"``
    the count 2; a grid in its three spellings or as the list.  A blank is
    nothing typed.

    MUTATIONS THIS MUST FAIL AGAINST: numbers read by Python's ``float``
    (``1_000`` reads); a bool read as ``text == "true"`` (``True`` is
    false); the type not judged after reading (``2.5`` stands for a
    count)."""
    got, why = read_text(_item(name, engine), sent)
    if refused is None:
        assert why is None, why
        assert got == value and type(got) is type(value), got
    else:
        assert why is not None and why.startswith(refused), why


def test_the_one_default_is_the_kinds_then_the_items():
    """The one default (§ 5.4): a kind's `recommended`, else the item's
    `default`; none for an item whose not-set line says what applies.

    MUTATION THIS MUST FAIL AGAINST: the kind's recommendation skipped (a
    vibration relaxes to the optimization's tolerance)."""
    tol = _item("relax_force_tol")
    assert recommended_for(tol, "vibration") not in (None, tol.default)
    assert default_of(tol, "vibration") == recommended_for(tol, "vibration")
    assert default_of(tol, "optimization") == tol.default
    assert default_of(_item("mpi_np"), "optimization") is None


def _request(stages_card=None, card=None):
    """A job request for the one-stage optimization of
    ``test_job_request_allocation``, with the calculation's card ``card``
    and the stage's ``stages_card``."""
    from molbuilder.job_request import SCHEMA, from_dict
    from test_job_request_allocation import _DESCRIPTION
    entry = {"group": 1, **({"execution": stages_card}
                            if stages_card is not None else {})}
    return from_dict({"schema": SCHEMA, "stages": {"coarse": entry},
                      **({"execution": card} if card is not None else {})},
                     _DESCRIPTION)


def test_a_stages_unset_gpu_count_runs_that_stage_on_no_gpu():
    """`job-request.md` § 3.2: a rung's ``null`` leaves an optional item
    unset on that rung, over the calculation's card -- a calculation-wide
    GPU card and a rung's ``"use_gpu": false, "gpu_count": null`` run that
    rung on no GPU.

    MUTATION THIS MUST FAIL AGAINST: the launch shape read without the
    door (``int(None)``, the traceback the 2.27.0 review found)."""
    from molbuilder.jobset.prep_inputs import card_settings, declared_run_shape
    request = _request({"use_gpu": False, "gpu_count": None},
                       {"use_gpu": True, "gpu_count": 2, "mpi_np": 4})
    assert declared_run_shape(request, "coarse") == {"mpi_np": 4}
    assert card_settings(request, "coarse")["use_gpu"] == (False,
                                                           "stage_card")
    assert declared_run_shape(request) == {"mpi_np": 4, "gres": "gpu:2"}


def test_a_cards_wall_is_spelled_as_the_allocations():
    """`job-request.md` § 3.2a: a card's ``time`` is read through the
    allocation's door and held in SLURM's spelling; a ``null`` wall or
    queue is refused, naming the key -- left out, the allocation's is used.

    MUTATIONS THIS MUST FAIL AGAINST: the card's wall stored as typed; a
    ``null`` wall passed through (it erased the allocation's at the fold)."""
    request = _request({"time": "4h"}, {"domain": "public"})
    assert request.stages["coarse"].execution["time"] == "0-04:00:00"
    assert request.to_dict()["stages"]["coarse"]["execution"]["time"] \
        == "0-04:00:00"
    for card, said in (({"time": None}, "'time' is this run's wall"),
                       ({"domain": None}, "'domain' is a queue's name"),
                       ({"time": "soon"}, "time: 'soon' is not a duration")):
        with pytest.raises(ValueError) as e:
            _request(card)
        assert "stages 'coarse' execution" in str(e.value), e.value
        assert said in str(e.value), e.value


def test_a_page_asks_the_door_and_keeps_no_reader(web_client):
    """`POST /api/task-setup/value` (`web/web-api.md`): what was typed,
    read and judged in its home -- the value in the item's spelling, or
    the one refusal; a blank is nothing typed; a card's wall read as the
    allocation's.

    MUTATION THIS MUST FAIL AGAINST: the endpoint judging the type alone
    (a GPU count on the transmission's card stands)."""
    def ask(**body):
        r = web_client.post("/api/task-setup/value", json={
            "engine": "siesta", "calculation": "optimization",
            "home": "run_card", **body})
        assert r.status_code == 200, r.get_json()
        return r.get_json()

    assert ask(item="mpi_np", text="8")["value"] == 8
    assert ask(item="kgrid", text="4x4x1", home="override",
               stage="tight")["value"] == [4, 4, 1]
    assert ask(item="time", text="4h")["value"] == "0-04:00:00"
    assert ask(item="mpi_np", text="")["blank"] is True
    assert ask(item="mpi_np", text="0x10")["refusal"] == \
        "mpi_np = '0x10', which is not a whole number."
    said = ask(item="gpu_count", text="2", calculation="transport",
               stage="transmission")["refusal"]
    assert "not 'transmission'" in said, said
    assert ask(item="use_gpu", text=None)["refusal"].startswith(
        "use_gpu = null, but this setting has no unset")


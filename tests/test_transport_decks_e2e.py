"""What prep writes for a transport calculation, on a junction a real SIESTA
relaxed (`engines/transport.md` § 3.1; `process/testing.md` § 0, tier 3).

A transport calculation starts from a finished relaxation, and takes its level
of theory from it, so its decks exist only after one: the relaxation is made
here with the engine, once a pass (`support.real_runs`), saved as a pair the
way the Results tab saves one, and every test cites that pair.  The decks are
prep's, read where prep wrote them (`process/testing.md` § 3a); what a row can
say is a row of ``tests/data/the_transport_deck.toml``.  No rung of the
transport ladder runs here: `test_transport_on_a_real_junction_e2e.py` runs them.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from _road import conda_hook, env_available
from support.road import (describe_calculation, jobset, one_rank, prep_groups,
                          road_structure, run_road_case, set_by_the_person,
                          stage_file)

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

_LABEL = "au_bdt_au_test"

TABLE = tomllib.loads((Path(__file__).parent / "data"
                       / "the_transport_deck.toml").read_text())


def _prepared(tmp_path, monkeypatch, pair, *, groups=("1", "2"), where="here",
              template=None, card=None, bias="0", frames=1):
    """A transport calculation citing the real ``pair``, at ``bias`` in
    ``frames`` frames, ``template``'s values and the run ``card``'s set, and
    ``groups`` prepared: the calculation's folder."""
    bundle = describe_calculation(tmp_path / where, monkeypatch, name=_LABEL,
                                  calculation="transport", bias=bias,
                                  frames=frames, structure=pair)
    one_rank(bundle, **(card or {}))
    if template:
        set_by_the_person(bundle, template)
    prep_groups(bundle, *groups)
    return bundle


def _deck(bundle, rung="device", point=None) -> str:
    """A rung's deck where prep wrote it, in its run or the point's folder --
    the device's by default, the transmission's when a check is about what
    tbtrans reads (the two are separate texts, `transport.md` § 6.1b)."""
    from molbuilder.jobset.materialize import latest_attempt
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    home = stage_home(bundle, task_of(bundle), rung)
    run = latest_attempt(home.dir)
    return ((run / point.rel if point else run)
            / home.names.name(".fdf")).read_text()


def _electrode_rows(deck: str, lead: str) -> dict:
    """A lead's ``%block TS.Elec.<lead>`` as ``{line: value}``, through the
    fdf reader."""
    from molbuilder.parse.fdf import read_fdf
    _scalars, blocks = read_fdf(deck)
    return {r[0].lower(): r[1] for r in blocks[f"tselec{lead.lower()}"]}


@pytest.mark.parametrize("case", TABLE["case"],
                         ids=[c["name"] for c in TABLE["case"]])
def test_the_deck_says_what_the_junction_and_its_description_state(
        case, real_au_bdt_au, tmp_path, monkeypatch):
    run_road_case(TABLE, {**case, "structure": real_au_bdt_au.pair}, tmp_path,
                  monkeypatch)


def test_the_canonical_junction_prepares_clean(real_au_bdt_au, tmp_path,
                                                monkeypatch):
    """A correctly-ordered, properly-labeled Au-BDT-Au junction is prepared,
    every rung of it: the gates a real prep runs -- the SIESTA validator, the
    shared checks, `validate_transport_kind` -- refuse none.

    And NO SPIN FINDING of any severity: gold is a noble s1 metal, not an
    open-d one, and the junction repeats, so the electronic state resolves
    it restricted (`science/chemistry-correctness.md` § 2a.1b) -- the gold
    lead once told to go open-shell, which is what the README beside the
    fixture says this system shows.
    """
    from molbuilder.parse.fdf import fdf_key, read_fdf
    bundle = describe_calculation(tmp_path, monkeypatch, name=_LABEL,
                                  calculation="transport",
                                  structure=real_au_bdt_au.pair)
    r = jobset("prep", "task", "--group", "1", "--group", "2", "--group", "3",
               "--bundle", bundle, "--target", "this")
    assert r.exit_code == 0, r.output
    for where in ("config.spin_treatment", "config.unpaired_electrons"):
        assert f"[{where}]" not in r.output, r.output
    scalars, _blocks = read_fdf(_deck(bundle))
    assert scalars[fdf_key("Spin")] == ["non-polarized"], scalars


#: Each electrode block says how many atoms the lead holds.  UNEVEN ON PURPOSE:
#: a hardcoded count satisfies any check on a symmetric junction -- measured on
#: the Au-BDT-Au junction, whose leads are 3 and 3.
@pytest.mark.parametrize("which, left, right, first", [
    ("au_bdt_au", 3, 3, 1), ("uneven", 8, 9, 2),
], ids=["no buffer, even leads", "a buffer atom, uneven leads"])
def test_every_electrode_block_carries_the_manuals_required_lines(
        which, left, right, first, real_au_bdt_au, real_uneven_chain, tmp_path,
        monkeypatch):
    """SCIENCE. An electrode block missing a required line is not a deck: the
    SIESTA 5.4.0 manual, "There are a few lines that must be present, HS,
    semi-inf-dir, electrode-pos, chem-pot.  The remaining options are
    optional."

    `elec-pos` sat inside `if buffer_idx:` until 2026-09-15, so an ORDINARY
    junction -- no buffer atoms -- got two electrode blocks without it
    (`engines/transport.md` 3.3). It was probably harmless, and that is
    exactly why it needs a test rather than a reading: molbuilder sorts the
    junction so the electrodes ARE the first and last atoms, which is where
    an omitted position would land anyway, so the deck relied on an
    undocumented default agreeing with the truth. Both cases are
    parametrised because only one of them was broken.
    """
    run = real_au_bdt_au if which == "au_bdt_au" else real_uneven_chain
    bundle = _prepared(tmp_path, monkeypatch, run.pair)
    blocks = dict(re.findall(r"%block TS\.Elec\.(\w+)(.*?)%endblock",
                             _deck(bundle), re.S))
    assert sorted(blocks) == ["L", "R"], sorted(blocks)
    deck = _deck(bundle)
    assert _electrode_rows(deck, "L")["used-atoms"] == str(left), blocks
    assert _electrode_rows(deck, "R")["used-atoms"] == str(right), blocks
    for name, body in blocks.items():
        missing = [k for k in ("HS", "semi-inf", "elec-pos", "chem-pot")
                   if k not in body]
        assert not missing, (
            f"%block TS.Elec.{name} omits {missing}, which the SIESTA 5.4.0 "
            f"manual lists as lines that MUST be present "
            f"(`engines/transport.md` § 3.3):\n{body}")

    # AND THE RIGHT ONE, NOT MERELY ONE: a presence-check cannot see an
    # electrode anchored at the WRONG end, or one atom off, on the buffer.
    #
    # The lower electrode is anchored by SIESTA's number of its FIRST atom
    # (`begin`) and the upper by its LAST (`end`, counted back from the end of
    # the sorted structure, § 3.3): a buffer atom past the lower lead moves its
    # first atom along.
    from molbuilder.parse.fdf import read_fdf
    _scalars, rows = read_fdf(deck)

    def anchor(lead):
        return next(r[1:] for r in rows[f"tselec{lead}"]
                    if r[0].lower() == "elec-pos")
    assert anchor("l") == ["begin", str(first)], blocks
    assert anchor("r") == ["end", "-1"], blocks


def test_the_leads_greens_functions_are_the_runs_once_per_voltage(
        real_au_bdt_au, tmp_path, monkeypatch):
    """`engines/transport.md` § 2a.11, *What a run's points share*: a device
    point's deck names each lead's surface Green's function in its RUN's
    scratch folder (`tsgf/`, the vocabulary's), the file named on the
    voltage, and asks the engine to reuse it -- so two frames at one
    voltage name the same file, two voltages never do, a point with no
    level names the run's own folder, and the transmission names none
    (tbtrans writes no such file).

    MUTATION THIS MUST FAIL AGAINST: the path left at the point (every
    frame writes its own 49 GiB pair); the voltage dropped from the name
    (the second voltage reads the first's file and the engine dies on the
    chemical potential); the folder dropped (the pair lies loose in the
    run); `GF.ReUse` not written; a `GF` line in the transmission's
    blocks."""
    from pathlib import PurePosixPath

    from molbuilder.transport.stages import Point

    def named(bundle, point):
        deck = _deck(bundle, point=point)
        rows = {lead: _electrode_rows(deck, lead) for lead in ("L", "R")}
        assert all(r["gf.reuse"] == "true" for r in rows.values()), rows
        here = PurePosixPath("run-0") / (point.rel if point else "")
        # The path as the engine resolves it from the point's folder.
        return {lead: resolved(PurePosixPath(
            *(p for p in (here / r["gf"]).parts if p != ".")))
            for lead, r in rows.items()}

    def resolved(p: PurePosixPath) -> str:
        out = []
        for part in p.parts:
            if part == "..":
                out.pop()
            else:
                out.append(part)
        return "/".join(out)

    points = _prepared(tmp_path, monkeypatch, real_au_bdt_au.pair, where="points", frames=2,
                       bias="0,0.2")
    f0 = named(points, Point(frame=0, volts=0.2, of=2))
    f1 = named(points, Point(frame=1, volts=0.2, of=2))
    assert f0 == f1 == {"L": f"run-0/tsgf/{_LABEL}.v0.2.TSGFL",
                        "R": f"run-0/tsgf/{_LABEL}.v0.2.TSGFR"}, (f0, f1)
    other = named(points, Point(frame=1, volts=0.0, of=2))
    assert other == {"L": f"run-0/tsgf/{_LABEL}.v0.TSGFL",
                     "R": f"run-0/tsgf/{_LABEL}.v0.TSGFR"}, other
    one = _prepared(tmp_path, monkeypatch, real_au_bdt_au.pair, where="one",
                    groups=("1", "2", "3"))
    alone = named(one, None)
    assert alone == {"L": f"run-0/tsgf/{_LABEL}.v0.TSGFL",
                     "R": f"run-0/tsgf/{_LABEL}.v0.TSGFR"}, alone
    for lead in ("L", "R"):
        rows = _electrode_rows(_deck(one, "transmission"), lead)
        assert "gf" not in rows and "gf.reuse" not in rows, rows


def test_the_device_deck_writes_the_bias_windows_line_and_the_file_switch(
        real_au_bdt_au, tmp_path, monkeypatch):
    """`engines/transport.md` § 2a.11, *The levers*: the device deck writes
    the bias window's line at every point -- `TS.Contours.nEq` naming
    `line`, and `TS.Contour.nEq.line` with the window the cutoff item gives
    and the spacing the card sets -- and the file switch,
    `TS.Elecs.Out-of-core`, TranSIESTA's own defaults written rather than
    left to it.  A spacing or a switch set on the card reaches the deck.

    MUTATION THIS MUST FAIL AGAINST: the block not written (the engine's
    silent 0.01 eV); the window typed with a number of its own rather than
    the cutoff item's; the switch left to the engine."""
    from molbuilder.parse.fdf import fdf_key, fdf_true, read_fdf
    from molbuilder.transport.stages import Point

    def read(where, **set_on_it):
        return read_fdf(_deck(
            _prepared(tmp_path, monkeypatch, real_au_bdt_au.pair, where=where, bias="0,0.2",
                      **set_on_it),
            point=Point(frame=None, volts=0.2)))

    scalars, blocks = read("as-described")
    assert blocks[fdf_key("TS.Contours.nEq")] == [["line"]], blocks
    rows = {r[0].lower(): r[1:] for r in blocks[fdf_key("TS.Contour.nEq.line")]}
    assert rows["part"] == ["line"] and rows["method"] == ["mid-rule"], rows
    assert rows["delta"] == ["0.01", "eV"], rows
    assert " ".join(rows["from"]) == "- 5 kT - |V|/2 to |V|/2 + 5 kT", rows
    assert fdf_true(scalars[fdf_key("TS.Elecs.Out-of-core")]), scalars

    scalars, blocks = read("set", card={"ts_elecs_out_of_core": False},
                           template={"negf_neq_spacing_ev": 0.02,
                                     "negf_neq_fermi_cutoff_kt": 7.5})
    rows = {r[0].lower(): r[1:] for r in blocks[fdf_key("TS.Contour.nEq.line")]}
    assert rows["delta"] == ["0.02", "eV"], rows
    assert " ".join(rows["from"]) == "- 7.5 kT - |V|/2 to |V|/2 + 7.5 kT", rows
    assert not fdf_true(scalars[fdf_key("TS.Elecs.Out-of-core")]), scalars


def test_the_file_estimate_is_the_engines_own(real_au_bdt_au, tmp_path, monkeypatch):
    """`engines/transport.md` § 2a.11, *What it costs*: the size a lead's
    surface Green's function file will have, from the device deck as prep
    writes it and the lead's orbital count, is what TranSIESTA itself
    prints at the run's start -- two runs of ours, read off their outputs:
    the Au–BDT–Au junction at 0 V (1425 orbitals, 5×5×1, 123 poles at 300 K:
    *49.170 GB*) and the test tree's chain at 0.2 V (208 orbitals, its lead
    run's own count, one k-point, 123 + 123 poles and a 46-point line:
    *194.092 MB*; SIESTA's units are 1024-based, and its print adds a few
    KiB of header to the matrices).

    MUTATION THIS MUST FAIL AGAINST: the k-points counted as the grid's
    (25, not 13); the +2 dropped; the second reservoir's poles or the line
    left out under bias; the window's kT taken at another temperature."""
    from molbuilder.transport.gf_files import estimate_from
    from molbuilder.transport.stages import Point
    gib, mib = 1024 ** 3, 1024 ** 2
    at_0v = _deck(_prepared(
        tmp_path, monkeypatch, real_au_bdt_au.pair, where="0V",
        template={"kgrid": (5, 5, 1), "negf_eq_pole_ev": 10.0,
                  "electronic_temperature": 300.0}))
    assert abs(estimate_from(at_0v, 1425) / gib - 49.170) < 0.001
    at_bias = _deck(_prepared(
        tmp_path, monkeypatch, real_au_bdt_au.pair, where="bias", bias="0,0.2",
        template={"kgrid": (1, 1, 1), "negf_eq_pole_ev": 10.0,
                  "electronic_temperature": 300.0}),
        point=Point(frame=None, volts=0.2))
    # SIESTA's print adds its header -- the cell, the coordinates, the
    # k-points, the energies -- a few KiB beside the matrices: 6 KiB here.
    assert abs(estimate_from(at_bias, 208) / mib - 194.092) < 0.01
    # THE GRID AS TRANSIESTA PLACES IT (`find_kgrid.F`): an even grid holds
    # Gamma and the zone boundary unshifted -- 4×4×1 is 10 points, 8 with
    # the half-step shift.
    from molbuilder.transport.gf_files import k_points_time_reversed
    assert k_points_time_reversed([(4, 0.0), (4, 0.0), (1, 0.0)]) == 10
    assert k_points_time_reversed([(4, 0.5), (4, 0.5), (1, 0.0)]) == 8


def test_the_transmission_window_carries_the_configured_point_count(
        real_au_bdt_au, tmp_path, monkeypatch):
    """TBtrans' contour window is a BLOCK, and its ``points`` row must be the
    value the person states -- a value the catalogue's default is not -- as
    molbuilder reads it back."""
    from molbuilder.parse.fdf import read_fdf
    deck = _deck(_prepared(tmp_path, monkeypatch, real_au_bdt_au.pair, groups=("1", "2", "3"),
                           template={"transmission_n_points": 301,
                                     "transmission_emin_ev": -3.0,
                                     "transmission_emax_ev": 1.5}),
                 "transmission")
    _scalars, blocks = read_fdf(deck)
    window = blocks.get("tbtcontourwindow")
    assert window, "no %block TBT.Contour.window in the transmission deck"
    rows = dict((r[0].lower(), r[1:]) for r in window)
    assert rows.get("points") == ["301"]
    assert rows.get("part") == ["line"], (
        "tbtrans refuses anything but a line part for this contour")
    # The bounds are the x-axis of T(E).  Pinned against what was stated, not
    # against E_F: whether a `%block TBT.Contour` line's energies are
    # absolute or E_F-relative is an open question (`transport/record.py`),
    # so this says what the deck
    # carries, nothing about what it means.
    from _deck import fdf_energy_window
    lo, hi = fdf_energy_window(deck, "TBT.Contour.window")
    assert lo == pytest.approx(-3.0)
    assert hi == pytest.approx(1.5)
    # AS MOLBUILDER READS IT BACK, for a sharing point's grid (§ 2a.16).
    from molbuilder.transport.energies import deck_window
    assert deck_window(deck) == (-3.0, 1.5, 301)


def _declared(pair, tmp_path, monkeypatch, calculation, stage, groups,
              **template):
    """The species a deck `jobset prep` wrote declares, in the order of its
    ``%block ChemicalSpeciesLabel`` -- read as SIESTA reads them."""
    from molbuilder.parse.fdf import fdf_key, read_fdf
    bundle = describe_calculation(tmp_path, monkeypatch,
                                  calculation=calculation, stage_strategy="",
                                  structure=pair)
    if template:
        set_by_the_person(bundle, template)
    prep_groups(bundle, *groups)
    _scalars, blocks = read_fdf(stage_file(bundle, stage, ".fdf").read_text())
    return [row[2] for row in blocks[fdf_key("ChemicalSpeciesLabel")]]


def test_the_override_reaches_a_transport_deck_too(real_au_bdt_au, tmp_path,
                                                   monkeypatch):
    """`species_order` is a catalogue row a person can fill in.  Every
    SIESTA deck honoured it and NO transport deck could see it, because the
    lifted geometry block took no config -- the config was already in scope
    at the call site and simply not passed (`deck.py`).  Global means
    global (user, 2026-09-23).
    """
    assert _declared(real_au_bdt_au.pair, tmp_path, monkeypatch, "transport",
                     "device", ("1", "2"), species_order=["S", "H", "C", "Au"]) == \
        ["S", "H", "C", "Au"], "the person's own order, honoured verbatim"


@pytest.mark.parametrize("calculation, stage, groups", [
    ("optimization", "coarse", ("1",)),
    ("transport", "device", ("1", "2")),
], ids=["a relaxation's deck", "a device's deck"])
def test_both_emitters_ask_the_same_function(
        calculation, stage, groups, real_au_bdt_au, tmp_path, monkeypatch):
    """The defect this rule closed: two emitters, two orders, one structure.

    Asked of the decks prep writes for each rather than by reading the source,
    so a third rule appearing anywhere would have to pass here too.
    """
    from molbuilder.chemistry import species_order
    assert _declared(real_au_bdt_au.pair, tmp_path, monkeypatch, calculation,
                     stage, groups) == species_order(
        road_structure(real_au_bdt_au.pair).elements), \
        "the default rule, on the one structure"


def test_a_tbtrans_step_s_record_lists_no_scf_setting(real_au_bdt_au, tmp_path,
                                                      monkeypatch):
    """What `test_effective_parameters_record.py` asks of every step, of the
    step whose program reads none of SIESTA's SCF settings."""
    from test_effective_parameters_record import the_fence_holds
    bundle = describe_calculation(tmp_path, monkeypatch, stage_strategy="",
                                  calculation="transport",
                                  structure=real_au_bdt_au.pair)
    prep_groups(bundle, "1", "2", "3")
    the_fence_holds(bundle, "05_transmission", "tbt_k_grid", "write_forces")


def test_a_tbtrans_step_s_run_script_starts_tbtrans_its_banner_lined_up(
        real_au_bdt_au, tmp_path, monkeypatch):
    """A transport calculation's transmission step: prep's deck states TBtrans
    as the program that reads it (`transport.deck.transport_spec`), and the
    run script starts TBtrans and names it in its banner -- whose labels
    line up on the longest one it prints, ``TBtrans version``
    (`running-a-job.md` § 1)."""
    bundle = describe_calculation(tmp_path, monkeypatch, name="T",
                                  calculation="transport",
                                  structure=real_au_bdt_au.pair)
    prep_groups(bundle, "1", "2", "3")
    text = stage_file(bundle, "transmission", ".run.sh").read_text()
    assert '_program_bin="$(command -v tbtrans ' in text, \
        "the run script does not start tbtrans"
    assert "command -v siesta " not in text
    head = text[text.index("===== molbuilder run script"):]
    head = head[:head.index('echo "========')]
    rows = re.findall(r'echo "  (\S[^"]*?) : ', head)
    assert "TBtrans version" in [r.rstrip() for r in rows], rows
    colons = {len(r) for r in rows}
    assert len(colons) == 1, f"the banner's labels do not line up: {rows}"


@pytest.mark.parametrize("frames, bias, walk", [
    (2, "0,0.2", "one job walking its 4 points (each point's wall, summed)"),
    (2, "0", "one job walking its 2 points (each point's wall, summed)"),
    (1, "0,0.2", "one job walking its 2 points (each point's wall, summed)"),
    (1, "0", None),
], ids=["two frames, two voltages", "two frames, one voltage",
        "one frame, two voltages", "one frame, one voltage"])
def test_a_run_of_points_is_told_from_its_citation_before_any_prep(
        real_au_bdt_au, tmp_path, monkeypatch, frames, bias, walk):
    """`execution/runs-of-points.md` § 1.2: its wall is typed before its first
    prep, so the run card says what the job walks then, and the count is what
    the cited frame set holds.

    THE SILENT FAILURE: the count read from the record prep writes -- *its 2
    points* said of a run of 4, and of a frame set with one voltage nothing, so
    its wall is typed for one run rather than each point's."""
    from support.road import _road_points_before_prep
    _road_points_before_prep({"device": walk}, describe_calculation(
        tmp_path, monkeypatch, name="T", calculation="transport",
        structure=real_au_bdt_au.pair, frames=frames, bias=bias))


def test_a_citation_answers_what_its_relaxation_ran_at(real_uneven_chain,
                                                       tmp_path, monkeypatch):
    """`engines/transport.md` § 3.1: a transport calculation is at the level
    of theory of the geometry it cites, unless the person says otherwise: the
    seed's and the device's decks state what the relaxation's deck stated, the
    k-point mesh by its rule -- 3 x 3 x 2 relaxed, the pair across carries over
    and the transport axis is one point (`citation_defaults._apply_kgrid`).

    THE SILENT FAILURE: a device computed at the catalogue's defaults on a
    geometry relaxed at something else, or sampling the axis it conducts
    along, with nothing said."""
    from molbuilder.parse.fdf import fdf_key, read_fdf
    from support.real_runs import QUICK
    bundle = _prepared(tmp_path, monkeypatch, real_uneven_chain.pair)
    for rung in ("seed", "device"):
        said, blocks = read_fdf(stage_file(bundle, rung, ".fdf").read_text())
        assert said[fdf_key("MeshCutoff")] == [str(QUICK["mesh_cutoff"]),
                                               "Ry"], (rung, said)
        assert said[fdf_key("PAO.BasisSize")] == [QUICK["basis_size"]], (
            rung, said)
        assert [row[:3] for row in blocks[fdf_key("kgrid_Monkhorst_Pack")]] == [
            ["3", "0", "0"], ["0", "3", "0"], ["0", "0", "1"]], (rung, blocks)


def test_a_voltage_outside_its_recommended_range_is_warned_where_it_is_saved(
        real_au_bdt_au, tmp_path, monkeypatch, web_client):
    """`engines/template.md` § 5.3: a point of the bias list is a value of its
    item, so one outside the item's range is a recommendation's business: said,
    and saved as asked.  The list is changed on Task setup, where a person
    changes it, the transmission's window opened to reach it, as it must.

    THE SILENT FAILURE: a voltage nobody meant, typed one digit off, saved
    without a word."""
    import json

    from support.road import read_request, save_both
    bundle = describe_calculation(tmp_path, monkeypatch, name="T",
                                  calculation="transport",
                                  structure=real_au_bdt_au.pair, bias="0,0.2")
    set_by_the_person(bundle, {"transmission_emin_ev": -5.0,
                               "transmission_emax_ev": 5.0})
    task = json.loads((bundle / "task.json").read_text())
    task["bias"]["voltages_v"] = [0.0, 6.0]
    r = save_both(bundle, task, read_request(bundle), web_client)
    assert r.status_code == 200, r.get_json()
    (found,) = [f for f in r.get_json()["findings"]
                if f["where"] == "task.bias"]
    assert found["severity"] == "warn", found
    assert "the bias list holds 6 V, outside the recommended range" in \
        found["message"], found


def test_a_relaxation_at_another_charge_cannot_be_cited(real_charged_formate,
                                                        tmp_path, monkeypatch):
    """§ 3.1 and ES7: the leads set a junction's electron number, so a geometry
    relaxed charged is not this junction's -- said at describe, naming the
    charge and the way out."""
    said = describe_calculation(tmp_path, monkeypatch, name="T",
                                calculation="transport",
                                structure=real_charged_formate.pair,
                                refused=True)
    assert "states a net charge of -1" in said, said
    assert "Relax the junction neutral" in " ".join(said.split()), said


def test_the_device_s_scratch_folder_is_said_on_every_page(
        real_au_bdt_au, tmp_path, monkeypatch, web_client):
    """`engines/transport.md` § 2a.11: the device stage's row on `jobset status`,
    Task setup and the Results ladder carries one line about the files its run
    keeps, read by one door.

    THE SILENT FAILURE: the person choosing a queue on Task setup is not told a
    run will want tens of GiB of scratch, which only the terminal said."""
    bundle = describe_calculation(
        tmp_path, monkeypatch, name="T", calculation="transport",
        structure=real_au_bdt_au.pair, bias="0,0.2")
    r = jobset("prep", "task", "--group", "1", "--group", "2", "--group", "3",
               "--bundle", bundle, "--target", "this", "--np", "2",
               "--cpus-per-task", "1")
    assert r.exit_code == 0, r.output
    status = jobset("status", "--bundle", bundle)
    rows = web_client.get(f"/api/task-setup/folder?dir={bundle}").get_json()[
        "ladder"]["stages"]
    said = {row["stage"]: row["scratch"] for row in rows}
    assert said["device"] and said["device"].startswith("tsgf/: ")
    assert said["device"] in status.output, status.output
    assert said["transmission"] is None
    # ...AND THE FOLDER ITSELF SAYS WHAT IT IS on the Results tab, where nothing
    # in it opens (`web/results.md` § 2): the place note is drawn from this.
    place = web_client.get(
        "/api/results/dir?path=" + str(bundle / "04_device" / "run-0" / "tsgf")
    ).get_json()["place"]
    assert place["role"] == "scratch" and "the engine's scratch" in place["says"]
    run = web_client.get(
        "/api/results/dir?path=" + str(bundle / "04_device" / "run-0")
    ).get_json()["place"]
    assert "the engine's scratch" not in run["says"]


def test_a_structure_carries_the_state_of_the_run_it_came_from(
        real_charged_formate, prepared):
    """ES7: a structure saved from a finished run carries that run's record,
    and a blank item takes the recorded value -- a calculation of a charged
    relaxation starts charged."""
    saved = road_structure(real_charged_formate.pair)
    _d, _s, fdf = prepared(real_charged_formate.pair, "siesta")
    assert re.search(r"^NetCharge\s+-1$", fdf, re.M), fdf
    assert ("recorded: recorded by the run this structure came out of "
            f"({saved.info['calculation']['source']})") in fdf
    # EDITED SINCE (a geometry or cell op): no longer the structure that run
    # came out of, so its record is not assumed (`web/molview.md` § 8.4).
    _d, _s, fdf = prepared(saved.translated((0.0, 0.0, 0.1)), "siesta")
    assert re.search(r"^NetCharge\s+\+0$", fdf, re.M), fdf
    assert "recorded by the run" not in fdf


def test_a_lead_s_seam_is_said_by_prep_as_on_the_card(real_au_bdt_au, tmp_path,
                                                      monkeypatch):
    """The seam of a lead's periodic boundary, measured when the card reads the
    lead (`tests/test_transport_wizard.py`), is said in the lead deck's own
    findings too, which prep prints and writes into its `.validation.txt`
    (`engines/transport.md` § 7.1: measured and reported): a person who never
    opens the card is told the same, at the same weight."""
    from molbuilder.transport.wizard import extract_electrode_model
    (seam,) = [n for n in extract_electrode_model(
        road_structure(real_au_bdt_au.pair), "L-electrode").notes
        if n.where == "lead.seam"]
    bundle = describe_calculation(tmp_path, monkeypatch, name="j",
                                  calculation="transport",
                                  structure=real_au_bdt_au.pair)
    r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
               "--target", "this")
    assert r.exit_code == 0, r.output
    assert f"{seam.severity} [lead.seam]: {seam.message}" in r.output, r.output
    assert seam.message in stage_file(bundle, "electrode_L",
                                      ".validation.txt").read_text()

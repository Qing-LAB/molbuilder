"""The transport calculation on a junction relaxed with the real SIESTA, its
whole ladder run with SIESTA, TranSIESTA and TBtrans (`engines/transport.md`
§ 1.1, § 2a.11, § 3.1; plan § 5y, run E).

THE JUNCTION is the smallest one with the parts a transport calculation is
made of, and the one whose answer is known: a perfect chain of hydrogen at
one spacing -- a half-filled band, a metal -- its two ends the electrode
blocks, nine atoms each, and the four atoms between them the device's own,
every atom held but the middle two.  A perfect single channel conducts one
quantum, G = G0 (Landauer), and its device is bulk-like up to each electrode
boundary, so its screening is complete there (`engines/transport.md`
§ 2a.13, *the lead layer count inside the device*).  One pseudopotential,
the suite's own (`tests/fixtures/psml`), and a minimal basis keep every rung
to seconds.

THE ROAD: the junction relaxed as an ordinary optimization, its template set
as a person sets it -- each value distinguishable from the catalogue's
default, and the spin unrestricted with a fixed count; then the transport
calculation described citing that run, on the CLI and through the Transport
tab's describe; its spin set restricted in its template, as a person sets
it; the seed and both leads prepared and launched as one group; the device's
two-point self-consistent sweep launched, launched again warm (refused:
every point is done), and cold; the transmission over the device's newest
run; the record.

THE TRANSPORT RUNS RESTRICTED because a polarized moment that floats, as
every transport rung's does, starts each atom at its full moment, aligned
(SIESTA's `m_new_dm.F90`).  On a junction of hydrogen alone that leaves no
minority density anywhere: the GGA potential of the empty channel puts its
states near 3000 eV and the run stays fully polarized -- measured 2026-10-08,
the leads at 8 of 8 and their Fermi level at 2176 eV, the device's SCF
stalled.  A polarized junction needs its starting moments stated (plan K22).

Every expectation is the contract's, read off what the commands answered as
the runs were made and off the folders and records they left.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-siesta env + a detectable conda hook"),
]

#: Nine hydrogens an electrode block, one ångström apart -- a lead cell
#: longer than the minimal basis's interaction range (§ 0.3, I11) -- and four
#: more of the same chain between the blocks.  NINE, NOT EIGHT: the
#: half-filled chain's Fermi points are k = ±π/2a, and an even block folds
#: both onto one k-point of its own cell, where the lead's surface Green's
#: function is degenerate -- measured 2026-10-09 with eight, T = 0.97 at
#: E_F exactly and 0.9998 at every other energy, at 0 V only.  An odd block
#: keeps the two apart.
_SPACING = 1.0
_LEAD = 9
_WIRE = 4
_BOX = 8.0

#: What the relaxation's template is set to before it runs -- each
#: distinguishable from the catalogue's default (300 Ry / DZP / 300 K /
#: unpolarized / a mixing weight of 0.02 and a history of 8), so a value that
#: reaches the transport template was read, not defaulted
#: (`engines/transport.md` § 3.1).  The mixer is one a hydrogen chain
#: converges with quickly; the cited run's SCF settings start each transport
#: SCF stage (TD6, 2026-10-09).
RELAXED_WITH = dict(mesh_cutoff=150.0, basis_size="SZ",
                    electronic_temperature=350.0,
                    spin_treatment="unrestricted", unpaired_electrons=2,
                    mixing_weight=0.1, pulay_history=6)

_BIAS = (0.0, 0.2)


def _junction() -> dict:
    """The junction as `jobset init` takes it: one chain, its ends the
    electrode blocks and its middle the bridge; every atom held but the
    middle two, whose neighbours on each side are the same; the room at the
    transport boundary one lead spacing (§ 6.1c, I12)."""
    from molbuilder.transport.sort import (REGION_BRIDGE,
                                           REGION_LEFT_ELECTRODE,
                                           REGION_RIGHT_ELECTRODE)
    n = _LEAD + _WIRE + _LEAD
    zs = [i * _SPACING for i in range(n)]
    labels = ([REGION_LEFT_ELECTRODE] * _LEAD + [REGION_BRIDGE] * _WIRE
              + [REGION_RIGHT_ELECTRODE] * _LEAD)
    regions: dict = {}
    for i, lab in enumerate(labels):
        regions.setdefault(lab, []).append(i)
    middle = (_LEAD + _WIRE // 2 - 1, _LEAD + _WIRE // 2)
    regions["frozen_atoms"] = [i for i in range(n) if i not in middle]
    return {"elements": ["H"] * n,
            "positions": [[_BOX / 2, _BOX / 2, z] for z in zs],
            "regions": regions,
            "cell": [_BOX, _BOX, n * _SPACING],
            "axis_kind": ["isolated", "isolated", "transport"]}


def _write_junction(tree: Path) -> str:
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    j = _junction()
    (tree / "P" / "structure").mkdir(parents=True, exist_ok=True)
    StructureCodec().write(
        Structure(elements=j["elements"],
                  positions=np.asarray(j["positions"], dtype=float),
                  regions=j["regions"], cell=np.diag(j["cell"]),
                  axis_kind=tuple(j["axis_kind"])),
        tree / "P" / "structure" / "junction.xyz")
    return "P/structure/junction.xyz"


def _run_card(bundle: Path) -> None:
    """One rank, one thread: the run states its shape (`architecture.md`
    § 5.2)."""
    task = json.loads((bundle / "task.json").read_text())
    task["execution"] = {**task.get("execution", {}), "mpi_np": 1,
                         "omp_threads": 1}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))


def _files(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}


def _set_by_the_person(template: Path, name: str, value) -> None:
    """One item of a calculation's template changed as a person changes it
    in the file: its value, and its source `person` (`engines/template.md`
    § 6.6)."""
    from molbuilder.template import one, read_template
    lines = template.read_text().splitlines(keepends=True)
    at = lines.index(f"[item.{name}]\n")
    end = next((i for i in range(at + 1, len(lines))
                if lines[i].startswith("[")), len(lines))
    for i in range(at + 1, end):
        if lines[i].startswith("value = "):
            lines[i] = f"value = {json.dumps(value)}\n"
        elif lines[i].startswith("source = "):
            lines[i] = 'source = "person"\n'
    template.write_text("".join(lines))
    assert one(read_template(template.read_text()), name).value == value


@pytest.fixture(scope="module")
def junction(tmp_path_factory):
    """The relaxation cited and the transport calculation run on it: a
    `RealRun` of the transport calculation, with ``cite``, the relaxation's
    citation, ``cited_before`` -- the cited run's files as the relaxation
    left them -- and every answer the road gave, by step."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.template import (config_from_template, template_path,
                                     template_with_values)
    from molbuilder.web.app import create_app
    from support.real_runs import RealRun, Said, _on_the_road, _taken, _typed
    from support.road import jobset
    with _on_the_road(tmp_path_factory, "junction") as tree:
        # THE RELAXATION -- an ordinary optimization, one stage.
        r = jobset("init", "--structure", _write_junction(tree),
                   "--bundle", "P/opt/J", "--engine", "siesta",
                   "--calculation", "optimization", "--shape", "hierarchical",
                   "--name", "J", "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        relax = RealRun(tree=tree, bundle=tree / "P" / "opt" / "J")
        _run_card(relax.bundle)
        path = template_path(relax.bundle, "J")
        cfg = dataclasses.replace(
            config_from_template(path.read_text(), SiestaConfig),
            **RELAXED_WITH)
        path.write_text(template_with_values(cfg, engine="siesta"))
        _taken(relax, "prep", "prep", "task", "--stage", "coarse",
               "--target", "this")
        _taken(relax, "launch", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--yes")
        cite = "P/opt/J/01_coarse/run-0"
        cited_before = _files(tree / cite)

        # THE TRANSPORT CALCULATION, described on the CLI.
        r = jobset("init", "--calculation", "transport", "--engine", "siesta",
                   "--shape", "hierarchical", "--bundle", "P/transport/T",
                   "--slot", f"junction={cite}",
                   "--bias", ",".join(f"{v:g}" for v in _BIAS),
                   "--no-low-bias-approximation")
        assert r.exit_code == 0, r.output
        run = RealRun(tree=tree, bundle=tree / "P" / "transport" / "T")
        run.said["init"] = Said(r.exit_code, r.output)
        _run_card(run.bundle)
        run.template_at_init = (run.bundle / "T.template.toml").read_text()
        # THE PERSON'S CHOICE in the calculation's own template: the spin
        # restricted (the module's head says why).  The mixing weight the
        # chain converges at in tens of cycles came with the citation.
        _set_by_the_person(run.bundle / "T.template.toml", "spin_treatment",
                           "restricted")

        # ...and through the Transport tab's describe, and its card's read
        # of the citation.
        client = create_app(config={}).test_client()
        b = client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T2", junction=cite, bias=[0.0]))
        run.said["describe"] = Said(b.status_code, json.dumps(b.get_json()))
        a = client.get(f"/api/transport/describe_attempt?path={cite}")
        run.said["describe_attempt"] = Said(a.status_code,
                                            json.dumps(a.get_json()))

        # THE REFUSALS THAT COME AFTER A CITATION IS READ (`task.py`).
        for step, extra in (("bias not from 0", ("--shape", "hierarchical",
                                                 "--bias", "0.2,0.4")),
                            ("flat", ("--shape", "flat"))):
            r = jobset("init", "--engine", "siesta",
                       "--calculation", "transport",
                       "--bundle", "P/transport/X",
                       "--slot", f"junction={cite}", *extra)
            run.said[step] = Said(r.exit_code, r.output)

        # THE LADDER.
        leads = ("--stage", "seed", "--stage", "electrode_L",
                 "--stage", "electrode_R")
        _taken(run, "prep seed and leads", "prep", "task", *leads,
               "--target", "this")
        _taken(run, "launch seed and leads", "launch", "task", *leads,
               "--mode", "direct", "--yes")
        _taken(run, "prep device", "prep", "task", "--stage", "device",
               "--target", "this")
        _taken(run, "launch device", "launch", "task", "--stage", "device",
               "--mode", "direct", "--yes")
        _taken(run, "status device", "status", "device")
        _typed(run, "launch device again", "launch", "task", "--stage",
               "device", "--mode", "direct", "--yes")
        _taken(run, "launch device cold", "launch", "task", "--stage",
               "device", "--mode", "direct", "--yes", "--cold")
        _taken(run, "status device after cold", "status", "device")
        _taken(run, "prep transmission", "prep", "task", "--stage",
               "transmission", "--target", "this")
        _taken(run, "launch transmission", "launch", "task", "--stage",
               "transmission", "--mode", "direct", "--yes")
        _taken(run, "summarize", "summarize", "task")
        run.cite = cite
        run.cited_before = cited_before
        run.relax = relax
    yield run


def _said(run, step, *words):
    said = run.said[step]
    for w in words:
        assert w in said.output, f"{step}: {w!r} not in:\n{said.output}"


def _record(run):
    from molbuilder.transport.record import record_path
    from molbuilder.task import read_task
    task = read_task(run.bundle / "task.json")
    return json.loads(record_path(run.bundle, task.label).read_text())


# --------------------------------------------------------------------- #
#  The citation (`engines/transport.md` § 3.1)                           #
# --------------------------------------------------------------------- #

def test_a_finished_relaxation_composes_and_says_how_it_ended(junction):
    """The citation is a relaxation run of molbuilder's own that finished;
    the composed copy carries how it ended, and its leads are the labelled
    blocks."""
    from molbuilder.transport.compose import classify_citation, compose_junction
    cited = classify_citation(junction.tree / junction.cite)
    assert cited.concluded and cited.exit_code == 0
    out = compose_junction(junction.cite, tree_root=junction.tree)
    assert out.provenance["evidence"] == cited.concluded
    assert out.provenance["relaxation"]["exit_code"] == 0
    assert len(out.electrode_left.elements) == _LEAD
    assert out.provenance["swap_electrodes"] is False


def test_the_rename_is_the_calculations_own_copy_and_the_cited_run_untouched(
        junction):
    """`transport.md` § 4: the electrode rename is applied to the
    calculation's own copy of the junction; the cited run's files are read,
    never written -- by every rung run on it, too."""
    from molbuilder.transport.compose import compose_junction
    from molbuilder.transport.sort import (REGION_LEFT_ELECTRODE,
                                           REGION_RIGHT_ELECTRODE)
    out = compose_junction(junction.cite, tree_root=junction.tree,
                           swap_electrodes=True)
    regions = out.sorted.structure.regions
    zs = np.asarray(out.sorted.structure.positions)[:, 2]
    assert min(zs[regions[REGION_LEFT_ELECTRODE]]) > max(
        zs[regions[REGION_RIGHT_ELECTRODE]])
    assert out.provenance["swap_electrodes"] is True
    assert _files(junction.tree / junction.cite) == junction.cited_before


def test_the_record_answers_for_the_citation_it_was_composed_from(junction,
                                                                   tmp_path):
    """`load_compose_record`'s "no record" is said in words: nothing
    composed here, or a record composed from another citation."""
    from molbuilder.transport.compose import (compose_junction,
                                              load_compose_record,
                                              write_compose_record)
    rec = tmp_path / "calc"
    rec.mkdir()
    why: list = []
    assert load_compose_record(rec, citation=junction.cite,
                               tree_root=junction.tree, why=why) is None
    assert "no slot-provenance.json" in why[0]
    write_compose_record(rec, compose_junction(junction.cite,
                                               tree_root=junction.tree))
    other = junction.cite.replace("run-0", "run-1")
    why = []
    assert load_compose_record(rec, citation=other, tree_root=junction.tree,
                               why=why) is None
    assert junction.cite in why[0] and other in why[0]
    assert load_compose_record(rec, citation=junction.cite,
                               tree_root=junction.tree) is not None


def test_a_cited_run_brings_its_settings_and_its_spin_on_both_roads(junction):
    """The cited run's deck DEFAULTS the shared settings into this
    calculation's own template, on the CLI and on the tab alike; its spin
    treatment arrives written, and a fixed count does not -- TranSIESTA
    cannot hold one -- so it is left blank and floats.  And its SCF mixer
    starts every transport SCF stage, said to be the cited run's (TD6,
    `engines/transport.md` § 3.1)."""
    from molbuilder.template import one, read_template
    describe = json.loads(junction.said["describe"].output)
    assert junction.said["describe"].exit_code == 200, describe
    for tmpl in (read_template(junction.template_at_init),
                 read_template(describe["files"][1]["text"])):
        assert one(tmpl, "mesh_cutoff").value == 150.0
        assert one(tmpl, "basis_size").value == "SZ"
        assert one(tmpl, "electronic_temperature").value == 350.0
        assert one(tmpl, "spin_treatment").value == "unrestricted"
        assert one(tmpl, "unpaired_electrons").value is None
        for name, value in (("mixing_weight", 0.1), ("pulay_history", 6)):
            assert one(tmpl, name).value == value, name
            assert one(tmpl, name).source == "cited", name


def test_each_leads_measurements_reach_the_card_under_its_name(junction):
    """The lead gate's measurements are `info` findings of the card's
    answer, each lead's prefixed with the lead it is about."""
    body = json.loads(junction.said["describe_attempt"].output)
    assert junction.said["describe_attempt"].exit_code == 200, body
    assert body["citation"] == junction.cite, body
    measured = [f["message"] for f in body["findings"]
                if f["severity"] == "info"]
    for lead in ("L-electrode", "R-electrode"):
        assert any(m.startswith(f"{lead}: the periodic seam")
                   for m in measured), measured


def test_a_bias_list_starts_from_equilibrium(junction):
    r = junction.said["bias not from 0"]
    assert r.exit_code != 0 and "must start at 0.0" in r.output, r.output


def test_a_flat_transport_description_is_refused(junction):
    """The five stages exchange files through their own runs, which flat has
    no folders to hold (§ 2a.11)."""
    r = junction.said["flat"]
    assert r.exit_code != 0 and "hierarchical" in r.output, r.output


# --------------------------------------------------------------------- #
#  The sweep -- one run, its points inside (§ 2a.11)                     #
# --------------------------------------------------------------------- #

def _point(run, rung, n, v):
    return run.bundle / rung / f"run-{n}" / f"v{v:g}"


def test_a_sweep_is_one_run_its_points_inside(junction):
    """The device's run holds a folder per voltage, each with what the rungs
    downstream take from it; `status` says how many points are done."""
    from molbuilder.task import read_task
    label = read_task(junction.bundle / "task.json").label
    for v in _BIAS:
        p = _point(junction, "04_device", 0, v)
        for suffix in (".TS.HSX", ".TSDE"):
            assert (p / f"{label}{suffix}").is_file(), (p, suffix)
    _said(junction, "status device", "2 of 2 points done")


def test_launched_again_warm_with_every_point_done_it_is_refused(junction):
    """No run is opened with nothing to run in it: the refusal names the
    `--cold` line."""
    r = junction.said["launch device again"]
    assert r.exit_code != 0, r.output
    assert "--cold" in r.output, r.output
    assert not (junction.bundle / "04_device" / "run-2").exists()


def test_launched_cold_every_point_runs_again(junction):
    """Cold, nothing is taken over: every point of the new run is walked,
    each start written beside it -- the 0 V point from the seed's density,
    what its run gathered (its `.gathered-from`, and no `.continued-from`,
    which a point taken over would carry), and the next from the point
    before it in this walk (its `.continued-from`)."""
    from molbuilder.runrecord import read_gathered_from
    zero = _point(junction, "04_device", 1, _BIAS[0])
    assert not (zero / ".continued-from").exists()
    assert any(g["from"] == "01_seed/run-0"
               for g in read_gathered_from(zero)), zero
    start = (_point(junction, "04_device", 1, _BIAS[1])
             / ".continued-from").read_text()
    assert "run-1" in start and f"v{_BIAS[0]:g}" in start, start


def test_the_transmission_takes_the_devices_newest_run_whole(junction):
    _said(junction, "prep transmission", "04_device/run-1")
    for v in _BIAS:
        assert (_point(junction, "05_transmission", 0, v)).is_dir(), v


# --------------------------------------------------------------------- #
#  The record (§ 2a.12)                                                  #
# --------------------------------------------------------------------- #

def test_the_record_is_the_sweeps_and_says_its_treatment(junction):
    rec = _record(junction)
    assert rec["treatment"] == "self-consistent"
    assert [p["bias_v"] for p in rec["points"]] == list(_BIAS)


def test_a_perfect_wire_conducts_one_quantum(junction):
    """The science the junction was chosen for: one orbital a site, one band
    through E_F, so T(E_F) = 1 and G = G0 · T(E_F) = G0 at equilibrium
    (`engines/transport.md` § 2a.12; Landauer).  A ladder that ran end to
    end with a lead attached wrongly -- its self-energy on the wrong side,
    its levels off the device's E_F -- still draws a curve; this number is
    what tells the two apart."""
    zero = next(p for p in _record(junction)["points"] if p["bias_v"] == 0.0)
    assert zero["conductance_g0"] == pytest.approx(1.0, abs=0.02), zero


def test_a_points_current_is_the_junctions_total(junction):
    """TBtrans prints one spin channel's current; the record's is the
    junction's total -- twice the printed figure for a non-polarized point --
    with the printed figure beside it and the spin its own deck states
    (`engines/transport.md` § 2a.12)."""
    points = _record(junction)["points"]
    assert [p["spin"] for p in points] == ["non-polarized"] * len(_BIAS)
    biased = [p for p in points if p["bias_v"] != 0.0]
    assert biased and all(p["current_a_printed"] for p in biased), points
    for p in points:
        assert np.isclose(p["current_a"], 2.0 * p["current_a_printed"]), p

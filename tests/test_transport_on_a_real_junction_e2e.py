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
tab's describe; its spin set restricted in its template, and three SCF
stages' own mixing weights on Task setup's stage table, as a person sets
them; the seed and both leads prepared and launched as one group; the
device's two-point self-consistent sweep launched, launched again warm
(refused: every point is done), and cold; the transmission over the device's
newest run; the record.  And the relaxation saved as a structure pair
(`molbuilder xv2xyz --from-run`); a SIESTA vibration of that pair, its
two free atoms; a frame set of three written from the vibration's highest
mode by a script through the structure's API, as the frame generator writes
one (frame 0 the pair, frames at +/-sqrt(3) sigma, the mode's definition
stated in full, `model/structure.md` § 2.2f); and a second transport
calculation citing the set, its two-point self-consistent sweep at every
frame: a point per frame and voltage, the record's average over the frames,
and the data file.  And the definition's rules
(`tests/data/frame_set_definition.toml`): the set with one rule broken each,
cited at `jobset init`.

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
import tomllib
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

#: Each SCF stage's own mixing weight, set on its tab -- every one distinct
#: from the cited run's 0.1 and from each other, so a deck carrying another
#: stage's value is told apart; electrode_R sets none and runs the
#: template's, the cited run's (`engines/transport.md` § 2a.13, § 3.8.2a).
_OWN_MIXING = {"seed": 0.12, "electrode_L": 0.15, "device": 0.08}

#: The frame set the second calculation cites, written by a script from a
#: real SIESTA vibration of the relaxed pair, as the frame generator writes
#: one (`model/structure.md` § 2.2f; `engines/vibration.md` § 5.10 ③): frame 0
#: the pair, then the Gauss-Hermite nodes of the mode's thermal distribution
#: at +/-sqrt(3) sigma, each with its weight
#: (`science/vibrational-averaging.md` § 5.1).
_NODES_SIGMA = (0.0, 3 ** 0.5, -(3 ** 0.5))
_WEIGHTS = (2 / 3, 1 / 6, 1 / 6)
_TEMPERATURE_K = 300.0
_N_FRAMES = len(_NODES_SIGMA)

#: The definition's rules (§ 2.2f), each a case of its own.
_DEFINITION_CASES = tomllib.loads(
    (Path(__file__).parent / "data" / "frame_set_definition.toml")
    .read_text())["case"]


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


def _mode_frame_set(tree: Path, pair: str, result: str):
    """The frame set a script writes from the vibration run ``result`` (its
    folder, tree-relative), through the structure's own doors
    (`model/structure.md` § 2.2d-2.2f): its highest mode; frame 0 the pair,
    then each node's frame, every atom displaced by ``q * L``; the
    definition stated in full -- the mode, frequency and zero-point amplitude
    as the result states them, the temperature, the spread from them; each
    frame's plain displacement, largest atom's, ``q``, ``q / sigma`` and
    weight; every atom's mass, the result's; where the mode came from."""
    import hashlib
    from molbuilder.frameset import MASS_CHANNEL
    from molbuilder.sidecars.spectra import parse_spectra_json
    from molbuilder.spectra.derived import thermal_spread_amu12_ang
    from molbuilder.structure import AtomChannel
    from molbuilder.workingcopy_structure import StructureCodec
    path = tree / result / "V.spectra.json"
    res = parse_spectra_json(str(path))
    mode = res.modes[-1]
    q_zp = res.to_dict()["modes"][-1]["zero_point_amplitude_amu12_ang"]
    sigma = thermal_spread_amu12_ang(q_zp, mode.frequency_cm1, _TEMPERATURE_K)
    L = np.zeros((res.n_atoms_total, 3))
    L[res.free_atom_idxs] = mode.eigenvector_canonical
    base = StructureCodec().load(tree / pair)
    qs = [node * sigma for node in _NODES_SIGMA]
    built = base.with_frames(np.stack([base.positions + q * L for q in qs]))
    built.set_channel(MASS_CHANNEL, AtomChannel(
        "value", {i: float(m) for i, m in
                  enumerate(res.equilibrium_masses_amu)}))
    for name, value, unit in (
            ("mode_index_1based", mode.index_1based, None),
            ("frequency_cm1", mode.frequency_cm1, "cm-1"),
            ("temperature_k", _TEMPERATURE_K, "K"),
            ("zero_point_amplitude_amu12_ang", q_zp, "amu^1/2 angstrom"),
            ("sigma_amu12_ang", sigma, "amu^1/2 angstrom")):
        built.set_customized(name, value, unit=unit)
    for f, (q, w) in enumerate(zip(qs, _WEIGHTS)):
        moved = q * L
        for name, value, unit in (
                ("displacement_ang", float(np.linalg.norm(moved)),
                 "angstrom"),
                ("max_atom_displacement_ang",
                 float(np.linalg.norm(moved, axis=1).max()), "angstrom"),
                ("q_amu12_ang", q, "amu^1/2 angstrom"),
                ("node_sigma", q / sigma, None),
                ("weight", w, None)):
            built.set_customized(name, value, unit=unit, frame=f)
    built.set_info("vibration", {
        "run": result,
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return built


def _with_case(built, case):
    """The good set with ``case``'s one change (`tests/data/
    frame_set_definition.toml`): rows restated, scaled or dropped, the mass
    channel dropped, a unit restated, every frame at frame 0's coordinates
    -- or none of the definition at all."""
    from molbuilder.frameset import MASS_CHANNEL
    if case.get("none"):
        bare = built.with_frames(np.asarray(built.frames, dtype=float))
        for row in list(bare.customized_rows()):
            bare.remove_customized(row["name"])
        for f in range(bare.n_frames):
            for row in list(bare.customized_rows(f)):
                bare.remove_customized(row["name"], frame=f)
        bare.annotations.pop(MASS_CHANNEL, None)
        (bare.info or {}).pop("vibration", None)
        return bare

    def row(name, frame):
        return next(r for r in built.customized_rows(frame)
                    if r["name"] == name)
    for e in case.get("set", ()):
        r = row(e["row"], e.get("frame"))
        built.set_customized(e["row"], e["value"], unit=r.get("unit"),
                             frame=e.get("frame"))
    for e in case.get("scale", ()):
        r = row(e["row"], e.get("frame"))
        built.set_customized(e["row"], r["value"] * e["by"],
                             unit=r.get("unit"), frame=e.get("frame"))
    for e in case.get("unit", ()):
        r = row(e["row"], e.get("frame"))
        built.set_customized(e["row"], r["value"], unit=e["unit"],
                             frame=e.get("frame"))
    for e in case.get("drop", ()):
        built.remove_customized(e["row"], frame=e.get("frame"))
    if case.get("drop_channel"):
        built.annotations.pop(MASS_CHANNEL, None)
    if case.get("still"):
        built = built.with_frames(
            np.stack([np.asarray(built.frames[0], dtype=float)]
                     * built.n_frames),
            frame_rows=[built.customized_rows(f)
                        for f in range(built.n_frames)])
    return built


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
    in the file: its value -- written in, for an item left blank -- and its
    source `person` (`engines/template.md` § 6.6)."""
    from molbuilder.template import one, read_template
    lines = template.read_text().splitlines(keepends=True)
    at = lines.index(f"[item.{name}]\n")
    end = next((i for i in range(at + 1, len(lines))
                if lines[i].startswith("[")), len(lines))
    said = {"value = ": f"value = {json.dumps(value)}\n",
            "source = ": 'source = "person"\n'}
    for i in range(at + 1, end):
        for key, line in list(said.items()):
            if lines[i].startswith(key):
                lines[i] = line
                del said[key]
    lines[at + 1:at + 1] = list(said.values())
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

        # THE PAIR -- the finished relaxation saved as a structure pair on
        # the command line, as the Results tab's export saves it: what the
        # run declared about its atoms and what it says about itself.
        from click.testing import CliRunner
        from molbuilder import cli
        from molbuilder.runs import run_of
        pair = "P/structure/relaxed.xyz"
        r = CliRunner().invoke(cli.cli, [
            "xv2xyz", str(run_of(tree / cite).carried(".XV")),
            str(tree / pair), "--from-run"])
        assert r.exit_code == 0, r.output

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
        # ...AND THREE SCF STAGES' OWN MIXING WEIGHTS, on Task setup's stage
        # table -- the page's door, its Save.
        client = create_app(config={}).test_client()
        task = json.loads((run.bundle / "task.json").read_text())
        for st in task["stages"]:
            if st["name"] in _OWN_MIXING:
                st["overrides"] = {"mixing_weight": _OWN_MIXING[st["name"]]}
        task["varies"] = sorted(set(task["varies"]) | {"mixing_weight"})
        s = client.post("/api/task-setup/save",
                        json={"dest": str(run.bundle),
                              "text": json.dumps(task)})
        assert s.status_code == 200, s.get_json()

        # ...and through the Transport tab's describe, and its card's read
        # of the citation.
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

        # THE MODE -- a SIESTA vibration of the relaxed pair, on the road:
        # its free atoms the bridge's middle two, held as the relaxation held
        # them, the box ticked (the pair IS the relaxed geometry), at the
        # relaxation's own settings.
        from molbuilder.workingcopy_structure import StructureCodec
        r = jobset("init", "--structure", pair, "--bundle", "P/frequency/V",
                   "--engine", "siesta", "--shape", "hierarchical",
                   "--calculation", "vibration", "--name", "V",
                   "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        vib = RealRun(tree=tree, bundle=tree / "P" / "frequency" / "V")
        _run_card(vib.bundle)
        vt = json.loads((vib.bundle / "task.json").read_text())
        vt["stages"] = [s for s in vt["stages"] if s["name"] == "freq"]
        (vib.bundle / "task.json").write_text(json.dumps(vt, indent=2))
        for name, value in {**RELAXED_WITH, "already_relaxed": True}.items():
            _set_by_the_person(vib.bundle / "V.template.toml", name, value)
        _taken(vib, "prep freq", "prep", "task", "--stage", "freq",
               "--target", "this")
        _taken(vib, "launch freq", "launch", "task", "--stage", "freq",
               "--mode", "direct", "--yes")
        vibration = "P/frequency/V/01_freq/run-0"

        # THE FRAME SET, written from that mode by a script through the
        # structure's own doors, the definition stated in full.
        frame_set = "P/structure/frames.xyz"
        StructureCodec().write(_mode_frame_set(tree, pair, vibration),
                               tree / frame_set)

        # THE SET CITED -- a second transport calculation, its
        # pseudopotentials from the library, its spin restricted as the
        # first's; the whole ladder, a point per frame and voltage.
        r = jobset("init", "--calculation", "transport", "--engine", "siesta",
                   "--shape", "hierarchical", "--bundle", "P/transport/F",
                   "--slot", f"junction={frame_set}",
                   "--psml-lib", "pseudopotential",
                   "--bias", ",".join(f"{v:g}" for v in _BIAS),
                   "--no-low-bias-approximation")
        assert r.exit_code == 0, r.output
        from_pair = RealRun(tree=tree, bundle=tree / "P" / "transport" / "F")
        _run_card(from_pair.bundle)
        run.pair_template = (from_pair.bundle / "F.template.toml").read_text()
        _set_by_the_person(from_pair.bundle / "F.template.toml",
                           "spin_treatment", "restricted")
        _taken(from_pair, "prep seed and leads", "prep", "task", *leads,
               "--target", "this")
        _taken(from_pair, "launch seed and leads", "launch", "task", *leads,
               "--mode", "direct", "--yes")
        _taken(from_pair, "prep device", "prep", "task", "--stage", "device",
               "--target", "this")
        _taken(from_pair, "launch device", "launch", "task", "--stage",
               "device", "--mode", "direct", "--yes")
        _taken(from_pair, "status device", "status", "device")
        _taken(from_pair, "prep transmission", "prep", "task", "--stage",
               "transmission", "--target", "this")
        _taken(from_pair, "launch transmission", "launch", "task", "--stage",
               "transmission", "--mode", "direct", "--yes")
        _taken(from_pair, "summarize", "summarize", "task")

        # THE DEFINITION'S RULES (`tests/data/frame_set_definition.toml`):
        # the good set with one change each, cited at `jobset init`, the door
        # that refuses it; the set stating none of the definition described,
        # its seed and leads prepared -- which composes the junction -- and
        # its record read where the Results tab reads it.
        run.definition_said = {}
        for k, case in enumerate(_DEFINITION_CASES):
            path = f"P/structure/case{k}.xyz"
            StructureCodec().write(
                _with_case(_mode_frame_set(tree, pair, vibration), case),
                tree / path)
            bundle = f"P/transport/C{k}"
            r = jobset("init", "--calculation", "transport",
                       "--engine", "siesta", "--shape", "hierarchical",
                       "--bundle", bundle, "--slot", f"junction={path}",
                       "--psml-lib", "pseudopotential",
                       "--bias", ",".join(f"{v:g}" for v in _BIAS),
                       "--no-low-bias-approximation")
            if "refused" in case:
                run.definition_said[case["name"]] = Said(r.exit_code,
                                                         r.output)
                continue
            assert r.exit_code == 0, r.output
            c = RealRun(tree=tree, bundle=tree / bundle)
            _run_card(c.bundle)
            _set_by_the_person(c.bundle / f"C{k}.template.toml",
                               "spin_treatment", "restricted")
            _taken(c, "prep seed and leads", "prep", "task", *leads,
                   "--target", "this")
            got = client.get(f"/api/transport/record?path={c.bundle}")
            run.definition_said[case["name"]] = (got.status_code,
                                                 got.get_json())
        run.vibration = vibration
        run.frame_set = frame_set
        run.cite = cite
        run.cited_before = cited_before
        run.relax = relax
        run.pair = pair
        run.from_pair = from_pair
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


def test_a_pair_saved_from_the_run_defaults_the_template_as_the_run_does(
        junction):
    """§ 3.1: a structure pair saved from a finished relaxation -- here by
    `molbuilder xv2xyz --from-run`, as the Results tab exports one, and a
    frame set written from it -- is a citation, and its record defaults the
    transport template exactly as the run it was saved from does: one reader
    of the recorded contract (`parse.contract.contract_of`), so the two kinds
    cannot default a template differently.  Its pseudopotentials come from
    the library named at init, the person's; its record names its kind and
    frame count."""
    from molbuilder.template import one, read_template, select
    from molbuilder.transport.compose import load_compose_record
    by_run = read_template(junction.template_at_init)
    by_pair = read_template(junction.pair_template)
    def cited(tmpl):
        return {it.name: it.value for it in select(tmpl, engine="siesta")
                if it.source == "cited"}
    assert "mixing_weight" in cited(by_run) and "mesh_cutoff" in cited(by_run)
    assert cited(by_pair) == cited(by_run)
    psml = one(by_pair, "psml_lib")
    assert (psml.value, psml.source) == ("pseudopotential", "person")
    record = load_compose_record(junction.from_pair.bundle,
                                 citation=junction.frame_set,
                                 tree_root=junction.tree)
    assert record is not None
    assert (record.provenance["kind"], record.provenance["frames"]) == (
        "pair", _N_FRAMES)


def _prepared_decks(run, stage):
    """The decks ``stage`` prepared: its one deck in the stage's folder, or
    each point's (`transport.stages.point_folders`) -- named by the run's
    one naming door (`runfiles.RunNames`)."""
    from molbuilder.jobset.materialize import stage_home
    from molbuilder.runfiles import RunNames
    from molbuilder.task import read_task
    from molbuilder.transport.stages import point_folders
    task = read_task(run.bundle / "task.json")
    home = stage_home(run.bundle, task, stage)
    names = RunNames.of(task.label, home.token, task.shape)
    folders = [f for f, _v in point_folders(run.bundle, task, stage)]
    return [f / names.name(".fdf") for f in (folders or [home.dir])]


def test_each_scf_stage_writes_its_own_mixer_and_the_transmission_none(
        junction):
    """SCF-RUNG (`engines/transport.md` § 2a.13 Class C, § 3.8.2a; TD12): a
    mixing weight set on one stage's tab reaches that stage's decks alone; a
    stage that sets none writes the template's -- the cited run's 0.1, TD6;
    and the transmission, which runs no SCF, writes none.  Each deck is read
    through the one fdf reader (`parse.fdf.parse_fdf_params`).  The
    mechanism once sent the T(E) window to the device's deck."""
    from molbuilder.parse.fdf import parse_fdf_params
    want = {**_OWN_MIXING, "electrode_R": RELAXED_WITH["mixing_weight"],
            "transmission": None}
    for stage, weight in want.items():
        decks = _prepared_decks(junction, stage)
        assert decks, stage
        for deck in decks:
            got = parse_fdf_params(deck.read_text()).mixing_weight
            assert (got is None if weight is None
                    else got == pytest.approx(weight)), (deck, got, weight)


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
#  A frame set -- a point per frame and voltage (§ 2a.9, § 2a.11)        #
# --------------------------------------------------------------------- #

def _frame_point(run, rung, f, v):
    from molbuilder.transport.stages import Point
    return run.bundle / rung / "run-0" / Point(frame=f, volts=v).rel


def _the_geometry_it_ran(point):
    """The first geometry step of the run in ``point``, read as the Results
    tab reads a run: the file that holds its result (`runs.openable`),
    parsed by the registry."""
    from molbuilder.parse import detect
    from molbuilder.runs import openable
    output, trail = openable(point)
    assert output is not None, trail
    steps = [fr for fr in detect(output).parse(str(output)).frames
             if fr.structure is not None]
    return np.asarray(steps[0].structure.positions, dtype=float)


def _the_set(junction):
    """The cited frame set, every frame, read through the codec."""
    from molbuilder.workingcopy_structure import StructureCodec
    return StructureCodec().load(junction.tree / junction.frame_set,
                                 frames=True)


def test_a_frame_sets_device_runs_a_point_per_frame_and_voltage(junction):
    """§ 2a.11's frame axis: the device's one run holds a level for each axis
    that varies -- a frame folder, a voltage folder in it -- each point
    holding what the rungs downstream take from it, computed on ITS frame's
    geometry; `status` counts the points with their axes, *k of N points
    done (F frames × V voltages)*."""
    from molbuilder.task import read_task
    from molbuilder.transport.record import composed_junction
    run = junction.from_pair
    label = read_task(run.bundle / "task.json").label
    for f in range(_N_FRAMES):
        for v in _BIAS:
            p = _frame_point(run, "04_device", f, v)
            for suffix in (".TS.HSX", ".TSDE"):
                assert (p / f"{label}{suffix}").is_file(), (p, suffix)
    # Against frame 0's point at the same voltage, each frame's point moved
    # its atoms as the composed set's own frame moves them -- the deck's
    # atom order -- and by nothing else.
    composed = np.asarray(composed_junction(run.bundle).frames, dtype=float)
    for v in _BIAS:
        at_0 = _the_geometry_it_ran(_frame_point(run, "04_device", 0, v))
        for f in range(1, _N_FRAMES):
            moved = (_the_geometry_it_ran(_frame_point(run, "04_device", f, v))
                     - at_0)
            assert np.allclose(moved, composed[f] - composed[0],
                               atol=1e-4), (f, v)
            assert np.abs(moved).max() > 1e-3, (f, v)
    n = _N_FRAMES * len(_BIAS)
    _said(run, "status device",
          f"{n} of {n} points done ({_N_FRAMES} frames × "
          f"{len(_BIAS)} voltages)")


def test_each_frame_starts_from_the_seed_and_its_voltages_chain_in_it(
        junction):
    """§ 2a.9 *Both axes*: frames are independent -- each frame's first
    voltage starts from what the run gathered, the seed's density (its own
    `.gathered-from`, no `.continued-from`) -- and within a frame each next
    voltage starts from the converged point before it in THAT frame (its
    `.continued-from`)."""
    from molbuilder.jobset.materialize import stage_home
    from molbuilder.runfiles import RunNames
    from molbuilder.runrecord import read_continued_from, read_gathered_from
    from molbuilder.task import read_task
    run = junction.from_pair
    task = read_task(run.bundle / "task.json")
    names = RunNames.of(task.label, stage_home(run.bundle, task,
                                               "device").token, task.shape)
    for f in range(_N_FRAMES):
        first = _frame_point(run, "04_device", f, _BIAS[0])
        assert read_continued_from(first, names, 0) is None, first
        assert any(g["from"] == "01_seed/run-0"
                   for g in read_gathered_from(first)), first
        nxt = _frame_point(run, "04_device", f, _BIAS[1])
        assert read_continued_from(nxt, names, 0) == str(
            first.relative_to(run.bundle)), nxt


def test_a_transmission_point_reads_the_device_at_its_frame_and_voltage(
        junction):
    """§ 2a.11: each transmission point gathers the device's Hamiltonian of
    the same frame and the same voltage, never another's."""
    from molbuilder.runrecord import read_gathered_from
    run = junction.from_pair
    for f in range(_N_FRAMES):
        for v in _BIAS:
            got = read_gathered_from(_frame_point(run, "05_transmission", f, v))
            hsx = [g["from"] for g in got if g["file"].endswith(".TS.HSX")]
            assert hsx == [str(_frame_point(run, "04_device", f, v)
                               .relative_to(run.bundle))], (f, v, got)


def test_a_frame_sets_record_has_a_point_per_frame_and_voltage(junction):
    """§ 2a.12's frame dimension: a frame set's record has a point per frame
    and voltage, each carrying its frame, its folder's tokens and its
    frame's `customized` rows whole -- what the cited set states for that
    frame -- and the I-V is per frame: each row tagged by the frame its
    point ran."""
    from molbuilder.transport.stages import Point
    rec = _record(junction.from_pair)
    cited = _the_set(junction)
    got = {(p["frame"], p["bias_v"]): p for p in rec["points"]}
    assert sorted(got) == [(f, v) for f in range(_N_FRAMES)
                           for v in _BIAS], sorted(got)
    for (f, v), p in got.items():
        assert p["tokens"] == Point(frame=f, volts=v).rel, p["tokens"]
        assert p["customized"] == cited.customized_rows(f), (f, p)
    iv = rec["iv"]
    assert list(zip(iv["frame"], iv["voltages_v"], iv["current_a"])) == [
        (p["frame"], p["bias_v"], p["current_a"]) for p in rec["points"]]


def _frame_curves(run, label, volts):
    """Each frame's T(E) at ``volts``, read from the transmission TBtrans
    wrote at that frame's point, through its parser."""
    from molbuilder.parse.engines.tbtrans import transmission_files
    from molbuilder.transport.record import parse_avtrans
    curves = []
    for f in range(_N_FRAMES):
        files = transmission_files(
            _frame_point(run, "05_transmission", f, volts), label)
        energies, t = parse_avtrans(files["unpolarized"][0].read_text())
        curves.append(np.asarray(t, dtype=float))
    return np.asarray(energies, dtype=float), curves


def test_the_mode_average_is_every_frames_curve_at_its_stated_weight(
        junction):
    """§ 2a.12, `science/vibrational-averaging.md` § 5.2: at each voltage
    the record's average is the sum over every frame of its T(E) -- read
    from the transmission TBtrans wrote at that frame's point -- at the
    weight the frame states; its change is from the base frame's; at E_F
    the averaged conductance beside the base's and the change in per cent.
    The mode's definition -- the cited set's structure rows -- the weights
    with their sum and tolerance, and the assumptions are stated beside the
    numbers."""
    from molbuilder.frameset import read as read_frame_set
    from molbuilder.task import read_task
    run = junction.from_pair
    label = read_task(run.bundle / "task.json").label
    avg = _record(run)["average"]
    fs = read_frame_set(_the_set(junction))
    assert avg["frames"] == _N_FRAMES
    for name, value in fs.structure_rows().items():
        assert avg[name] == value, (name, avg[name], value)
    assert avg["weights"] == list(_WEIGHTS) and avg["tolerance"] == 1e-6
    assert abs(avg["weight_sum"] - 1.0) <= avg["tolerance"]
    assert avg["why"] is None and len(avg["assumptions"]) == 6
    assert [e["bias_v"] for e in avg["at"]] == list(_BIAS)
    for e in avg["at"]:
        assert e["waits_for"] == [] and e["why"] is None, e
        energies, curves = _frame_curves(run, label, e["bias_v"])
        expect = sum(w * c for w, c in zip(_WEIGHTS, curves))
        assert np.allclose(e["transmission"], expect, rtol=1e-12, atol=0)
        assert np.allclose(e["delta_transmission"], expect - curves[0],
                           rtol=1e-9, atol=1e-15)
        g = [float(np.interp(0.0, energies, c)) for c in curves]
        g_avg = sum(w * x for w, x in zip(_WEIGHTS, g))
        assert e["conductance_g0"] == pytest.approx(g_avg, rel=1e-12)
        assert e["base_conductance_g0"] == pytest.approx(g[0], rel=1e-12)
        assert e["conductance_change_percent"] == pytest.approx(
            100.0 * (g_avg - g[0]) / g[0], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("case", _DEFINITION_CASES,
                         ids=[c["name"] for c in _DEFINITION_CASES])
def test_the_definition_is_checked_at_the_citation(junction, case):
    """`model/structure.md` § 2.2f: a mode's frame set states what every
    frame is, completely and consistently, and the transport citation door
    checks it frame by frame before anything runs -- each rule broken once,
    on the real mode's set, refused at `jobset init` naming the frame and the
    row; a set stating none of it a family of frames with no average, said
    so (`tests/data/frame_set_definition.toml`)."""
    said = junction.definition_said[case["name"]]
    if "refused" in case:
        assert said.exit_code != 0, said.output
        for words in case["refused"]:
            assert words in said.output, (words, said.output)
        return
    status, body = said
    assert status == 200, body
    avg = body["record"]["average"]
    assert avg["at"] == [], avg
    for words in case["no_average"]:
        assert words in avg["why"], (words, avg["why"])


def test_the_data_file_holds_every_point_beside_its_frames_definition(
        junction):
    """§ 2a.12, *the data file*: `<label>.transport.nc`, read through its
    door, holds every point on one grid -- frame x voltage x energy, the
    I-V's voltages the description's -- its T(E) TBtrans's own, each frame's
    definition rows and every atom's mass and coordinates beside it, the
    device's facts, the I-V and the mode's average each at its own point as
    the record states them; every dimension a coordinate of its own name and
    the labels of its points coordinates beside it; every variable says what
    it is, a raw one where it was read from; and a calculation of one
    structure has the same layout, one frame."""
    from molbuilder.frameset import FRAME_ROWS, MASS_CHANNEL
    from molbuilder.task import read_task
    from molbuilder.transport.datafile import (VARIABLES, data_file_path,
                                               read_data_file)
    from molbuilder.transport.record import composed_junction
    run = junction.from_pair
    label = read_task(run.bundle / "task.json").label
    got = read_data_file(data_file_path(run.bundle, label))
    V = {k: v["data"] for k, v in got["variables"].items()}
    assert list(V["frame"]) == list(range(_N_FRAMES))
    assert list(V["frame_token"]) == [f"f{f + 1:03d}" for f in range(_N_FRAMES)]
    assert list(V["bias_v"]) == list(_BIAS)
    assert list(V["iv_bias_v"]) == list(_BIAS)
    for b, v in enumerate(_BIAS):
        energies, curves = _frame_curves(run, label, v)
        assert np.array_equal(V["energy_ev"], energies)
        for f in range(_N_FRAMES):
            assert V["done"][f, b] == 1
            assert np.array_equal(V["transmission"][f, b], curves[f])
        assert np.allclose(V["average_transmission"][b],
                           sum(w * c for w, c in zip(_WEIGHTS, curves)),
                           rtol=1e-12, atol=0)
    composed = composed_junction(run.bundle)
    for r in FRAME_ROWS:
        assert list(V[r.name]) == [composed.customized_value(r.name, frame=f)
                                   for f in range(_N_FRAMES)], r.name
    masses = composed.annotations[MASS_CHANNEL].data
    assert list(V["mass_amu"]) == [masses[i] for i in range(composed.n_atoms)]
    assert np.array_equal(V["positions_ang"],
                          np.asarray(composed.frames, dtype=float))
    # EACH VALUE PLACED BY ITS OWN POINT, as the record states it: the
    # device's facts, the I-V, the averaged conductance.
    rec = _record(run)
    device = next(s for s in rec["stages"] if s["stage"] == "device")
    for p in device["by_point"]:
        f, b = p["frame"], list(_BIAS).index(p["bias_v"])
        assert V["device_ef_ev"][f, b] == p["negf"]["ef"], (f, b)
        assert V["device_vha_ev"][f, b] == p["negf"]["vha_ev"], (f, b)
    for f, v, i in zip(rec["iv"]["frame"], rec["iv"]["voltages_v"],
                       rec["iv"]["current_a"]):
        assert V["iv_current_a"][f, list(_BIAS).index(v)] == i, (f, v)
    for e in rec["average"]["at"]:
        assert V["average_conductance_g0"][list(_BIAS).index(e["bias_v"])] \
            == e["conductance_g0"], e["bias_v"]
    # EVERY DIMENSION A COORDINATE OF ITS OWN NAME, and the labels of a
    # dimension's points named as its coordinates (CF `coordinates`).
    dims = {d for v in got["variables"].values() for d in v["dims"]}
    for d in dims:
        assert got["variables"][d]["dims"] == (d,), d
    frame_labels = {"frame_token", *(r.name for r in FRAME_ROWS)}
    attrs_of = {k: v["attrs"] for k, v in got["variables"].items()}
    assert set(attrs_of["transmission"]["coordinates"].split()) \
        == frame_labels
    assert set(attrs_of["positions_ang"]["coordinates"].split()) \
        == frame_labels | {"element", MASS_CHANNEL}
    for var in VARIABLES:
        attrs = attrs_of[var.name]
        assert attrs["long_name"] and attrs["definition"], var.name
        assert "units" in attrs and attrs["kind"] == var.kind, var.name
        if var.kind == "raw":
            assert attrs["source"], var.name
    one = read_data_file(data_file_path(junction.bundle, "T"))["variables"]
    assert one["transmission"]["data"].shape[:2] == (1, len(_BIAS))
    assert one["transmission"]["dims"] == ("frame", "bias_v", "energy_ev")


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

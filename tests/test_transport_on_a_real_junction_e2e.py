"""The transport calculation on a junction relaxed with the real SIESTA, its
whole ladder run with SIESTA, TranSIESTA and TBtrans (`engines/transport.md`
§ 1.1, § 2a.11, § 3.1; plan § 5y, run E).

THE JUNCTION is the smallest one with the parts a transport calculation is
made of, and the one whose answer is known: a perfect chain of hydrogen at
one spacing -- a half-filled band, a metal -- its two ends the electrode
blocks, nine atoms each, and the four atoms between them the device's own,
every atom held but the middle two -- listed bridge first, then the right
block, then the left, so the order TranSIESTA needs (the blocks at the ends,
`transport/sort.py`) moves every atom and every result has to come back
through the junction's order record.  A perfect single channel conducts one
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
and the data file; and the same frames stating no rows, a family.

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
import os
from pathlib import Path

import numpy as np
import pytest

from _road import conda_hook, env_available
from molbuilder.parse.engines import siesta_grammar as _G
from molbuilder.transport.stages import TRANSPORT_STAGES
from support.machines import SLURM
from support.road import molbuilder

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

#: THE ORDER THE JUNCTION IS LISTED IN, by each atom's place along the chain
#: (0 the lowest): the bridge, then the right block, then the left.
#: TranSIESTA's order is the chain's, so the sort moves every atom, and its
#: record is not its own inverse -- a result put back the wrong way round is
#: told apart (`model/overview.md` § 2.2a).
_LISTED = (tuple(range(_LEAD, _LEAD + _WIRE))
           + tuple(range(_LEAD + _WIRE, 2 * _LEAD + _WIRE))
           + tuple(range(_LEAD)))


def _at(place: int) -> int:
    """The index, in the junction as listed, of the atom at ``place`` along
    the chain."""
    return _LISTED.index(place)


#: THE ORDER RECORD OF TRANSIESTA'S ORDER, the chain's, as every rung's deck
#: and the composed junction state it.
_CHAIN = {"key": "transport",
          "sorted_to_original": [_at(p) for p in range(len(_LISTED))],
          "original_to_sorted": list(_LISTED)}


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

#: A mode's frame set's rows as `model/structure.md` § 2.2f's table states
#: them -- ``(name, kind, unit, words, page unit)``, the structure's and each
#: frame's -- what the Transport tab's card shows for the set the fixture
#: wrote (§ 3.8.1).
_SET_ROWS = (("mode_index_1based", "given", None, "Mode", None),
             ("frequency_cm1", "given", "cm-1", "Frequency", "cm⁻¹"),
             ("temperature_k", "given", "K", "Temperature", "K"),
             ("zero_point_amplitude_amu12_ang", "given", "amu^1/2 angstrom",
              "Zero-point amplitude", "amu½·Å"),
             ("sigma_amu12_ang", "derived", "amu^1/2 angstrom",
              "Thermal spread σ", "amu½·Å"))
_FRAME_ROWS = (("displacement_ang", "measured", "angstrom",
                "Displacement", "Å"),
               ("max_atom_displacement_ang", "measured", "angstrom",
                "Largest atom move", "Å"),
               ("q_amu12_ang", "derived", "amu^1/2 angstrom",
                "Position along the mode, q", "amu½·Å"),
               ("node_sigma", "derived", None, "q in units of σ", None),
               ("weight", "given", None, "Weight", None))

#: The PDOS of a selection asked where the Results tab asks it
#: (`/api/transport/pdos`, `web/results.md` § 2.5), a row each: every atom of
#: the junction, or the ``atoms`` it names in the user order; the point ``at`` --
#: (frame index, voltage) -- whose total DOS it is, or whose ``region``'s;
#: or the words it is ``refused`` with.
_PDOS_ASKS = (
    {"name": "a frame set's point", "calc": "P/transport/F",
     "query": "frame=1&point=0.2&orbitals=all", "at": (1, 0.2)},
    # THE BRIDGE, NAMED IN THE USER ORDER: its atoms are listed first, and
    # TranSIESTA's order puts the left block there -- the answer is the
    # bridge's only through the junction's order record.
    {"name": "the bridge's atoms, named in the user order",
     "calc": "P/transport/F", "query": "frame=1&point=0.2&orbitals=all",
     "atoms": tuple(_at(p) for p in range(_LEAD, _LEAD + _WIRE)),
     "region": "bridge", "at": (1, 0.2)},
    {"name": "by orbital type, through the device run's orbital index",
     "calc": "P/transport/F", "query": "frame=2&point=0&orbitals=s",
     "at": (2, 0.0)},
    # THE FILTER ITSELF: the minimal basis holds s orbitals alone, so a p
    # selection is refused by name -- the row above passes with or without
    # the filter.
    {"name": "an orbital type the selection's basis does not hold",
     "calc": "P/transport/F", "query": "frame=1&point=0&orbitals=p",
     "refused": "the selected atoms have no p orbital"},
    {"name": "a frame set's point that names no frame",
     "calc": "P/transport/F", "query": "point=0.2",
     "refused": "name the frame (its index, 0 … 2)"},
    {"name": "one structure's point that names a frame",
     "calc": "P/transport/T", "query": "frame=0&point=0.2",
     "refused": "cites one structure"},
)

def _junction() -> dict:
    """The junction as `molbuilder describe` takes it, listed as :data:`_LISTED`:
    one chain, its ends the electrode blocks and its middle the bridge;
    every atom held but the middle two, whose neighbours on each side are
    the same; the room at the transport boundary one lead spacing (§ 6.1c,
    I12)."""
    from molbuilder.transport.sort import (REGION_BRIDGE,
                                           REGION_LEFT_ELECTRODE,
                                           REGION_RIGHT_ELECTRODE)
    n = len(_LISTED)
    labels = ([REGION_LEFT_ELECTRODE] * _LEAD + [REGION_BRIDGE] * _WIRE
              + [REGION_RIGHT_ELECTRODE] * _LEAD)
    regions: dict = {}
    for i, place in enumerate(_LISTED):
        regions.setdefault(labels[place], []).append(i)
    middle = (_LEAD + _WIRE // 2 - 1, _LEAD + _WIRE // 2)
    regions["frozen_atoms"] = [i for i, place in enumerate(_LISTED)
                               if place not in middle]
    return {"elements": ["H"] * n,
            "positions": [[_BOX / 2, _BOX / 2, place * _SPACING]
                          for place in _LISTED],
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
    from molbuilder.frame_average import MASS_CHANNEL
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


def _family_of(built):
    """The same frames stating no rows, no channel and no ``info.vibration``
    -- each frame its own curve (`model/structure.md` § 2.2f)."""
    from molbuilder.frame_average import MASS_CHANNEL
    bare = built.with_frames(np.asarray(built.frames, dtype=float))
    for row in list(bare.customized_rows()):
        bare.remove_customized(row["name"])
    for f in range(bare.n_frames):
        for row in list(bare.customized_rows(f)):
            bare.remove_customized(row["name"], frame=f)
    bare.annotations.pop(MASS_CHANNEL, None)
    (bare.info or {}).pop("vibration", None)
    return bare


def _run_card(bundle: Path) -> None:
    """One rank, one thread: the run states its shape (`architecture.md`
    § 5.2), on the calculation's run card."""
    from support.road import read_request, write_request
    request = read_request(bundle)
    request["execution"] = {**request.get("execution", {}), "mpi_np": 1,
                            "omp_threads": 1}
    write_request(bundle, request)


def _files(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in d.iterdir() if p.is_file()}


#: WHAT THE DEVICE RAN, CHANGED SINCE -- each row a copy of the single
#: voltage's calculation, its device launched, a template value then changed
#: as a person changes it; its transmission prepared now and its launch asked
#: dry is refused naming the keyword (`engines/transport.md` § 6.1, *the
#: same engine input*): a number, a block -- the k-points' offset, which
#: changes the grid's rows and nothing derived from its size.
_ANOTHER_INPUT = [
    dict(name="a number", copy="mesh-changed", item="mesh_cutoff",
         value=160.0, names="MeshCutoff"),
    dict(name="a block", copy="k-offset-changed", item="kgrid_displacement",
         value=(0.5, 0.5, 0.0), names="kgrid_Monkhorst_Pack"),
]


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
    from molbuilder.template import config_from_template, template_with_values
    from molbuilder.template_file import template_path
    from molbuilder.web.app import create_app
    from support.real_runs import RealRun, Said, _on_the_road, _taken, _typed
    from support.road import group_words, ladder_edited

    def seed_and_leads(on):
        # ONE GROUP, ONE JOB (`engines/stages.md` § 2): the defaults put
        # the seed and the two leads in group 1.
        return group_words(on.bundle, "seed", "electrode_L", "electrode_R")
    with _on_the_road(tmp_path_factory, "junction") as tree:
        # THE RELAXATION -- an ordinary optimization, one stage.
        r = molbuilder("describe", "--structure", _write_junction(tree),
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
        _taken(relax, "prep", "prep", "task",
               *group_words(relax.bundle, "coarse"), "--target", "this")
        _taken(relax, "launch", "launch", "task", "--stage", "coarse",
               "--mode", "direct", "--foreground", "--yes")
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
            "xv2xyz", str(run_of(tree / cite).on_label(".XV")),
            str(tree / pair), "--from-run"])
        assert r.exit_code == 0, r.output

        # THE TRANSPORT CALCULATION, described on the CLI.
        r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
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
        from support.road import read_request, save_both
        s = save_both(run.bundle, task, read_request(run.bundle), client)
        assert s.status_code == 200, s.get_json()

        # ...and through the Transport tab's describe, which writes the
        # description into the folder it names (`engines/transport.md`
        # § 3.8.2a), and its card's read of the citation.
        run.described_at = tree / "P" / "transport" / "T2"
        run.described_at.mkdir()
        b = client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T2", junction=cite, bias=[0.0],
            dest=str(run.described_at)))
        run.said["describe"] = Said(b.status_code, json.dumps(b.get_json()))
        # ...AND AGAIN, INTO THE FOLDER IT HAS JUST DESCRIBED: Describe is
        # one act, and that folder is now a calculation (§ 3.8.2a).
        run.described_before = {f.name: f.read_bytes()
                                for f in run.described_at.iterdir()}
        b = client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T2", junction=cite, bias=[0.0, 0.5],
            low_bias_approximation=True, dest=str(run.described_at)))
        run.said["describe again"] = Said(b.status_code,
                                          json.dumps(b.get_json()))
        run.described_after = {f.name: f.read_bytes()
                               for f in run.described_at.iterdir()}
        a = client.get(f"/api/transport/describe_attempt?path={cite}")
        run.said["describe_attempt"] = Said(a.status_code,
                                            json.dumps(a.get_json()))

        # THE REFUSALS THAT COME AFTER A CITATION IS READ (`task.py`).
        for step, extra in (("bias not from 0", ("--shape", "hierarchical",
                                                 "--bias", "0.2,0.4")),
                            ("flat", ("--shape", "flat"))):
            r = molbuilder("describe", "--engine", "siesta",
                       "--calculation", "transport",
                       "--bundle", "P/transport/X",
                       "--slot", f"junction={cite}", *extra)
            run.said[step] = Said(r.exit_code, r.output)

        # THE LADDER, a group at a time, each prepared once the one before
        # it has run.
        _taken(run, "prep seed and leads", "prep", "task",
               *seed_and_leads(run), "--target", "this")
        _taken(run, "launch seed and leads", "launch", "task",
               *seed_and_leads(run), "--mode", "direct", "--foreground", "--yes")
        _taken(run, "prep device", "prep", "task",
               *group_words(run.bundle, "device"), "--target", "this")
        _taken(run, "launch device", "launch", "task", "--stage", "device",
               "--mode", "direct", "--foreground", "--yes")
        _taken(run, "status device", "status", "device")
        _typed(run, "launch device again", "launch", "task", "--stage",
               "device", "--mode", "direct", "--foreground", "--yes")
        _taken(run, "launch device cold", "launch", "task", "--stage",
               "device", "--mode", "direct", "--foreground", "--yes", "--cold")
        _taken(run, "status device after cold", "status", "device")
        _taken(run, "prep transmission", "prep", "task",
               *group_words(run.bundle, "transmission"), "--target", "this")
        _taken(run, "launch transmission", "launch", "task", "--stage",
               "transmission", "--mode", "direct", "--foreground", "--yes")
        _taken(run, "summarize", "summarize", "task")

        run.probe = _the_decks_a_person_sets(tree, cite, client)

        # THE MODE -- a SIESTA vibration of the relaxed pair, on the road:
        # its free atoms the bridge's middle two, held as the relaxation held
        # them, the box ticked (the pair IS the relaxed geometry), at the
        # relaxation's own settings.
        from molbuilder.workingcopy_structure import StructureCodec
        r = molbuilder("describe", "--structure", pair, "--bundle", "P/frequency/V",
                   "--engine", "siesta", "--shape", "hierarchical",
                   "--calculation", "vibration", "--name", "V",
                   "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        vib = RealRun(tree=tree, bundle=tree / "P" / "frequency" / "V")
        _run_card(vib.bundle)
        ladder_edited({"removed": ["relax"]}, vib.bundle)
        for name, value in {**RELAXED_WITH, "already_relaxed": True}.items():
            _set_by_the_person(vib.bundle / "V.template.toml", name, value)
        _taken(vib, "prep freq", "prep", "task",
               *group_words(vib.bundle, "freq"), "--target", "this")
        _taken(vib, "launch freq", "launch", "task", "--stage", "freq",
               "--mode", "direct", "--foreground", "--yes")
        vibration = "P/frequency/V/01_freq/run-0"

        # THE FRAME SET, written from that mode by a script through the
        # structure's own doors, the definition stated in full.
        frame_set = "P/structure/frames.xyz"
        StructureCodec().write(_mode_frame_set(tree, pair, vibration),
                               tree / frame_set)

        # THE SET CITED -- a second transport calculation, its
        # pseudopotentials from the library, its spin restricted as the
        # first's; the whole ladder, a point per frame and voltage.
        r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
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
        _taken(from_pair, "prep seed and leads", "prep", "task",
               *seed_and_leads(from_pair), "--target", "this")
        _taken(from_pair, "launch seed and leads", "launch", "task",
               *seed_and_leads(from_pair), "--mode", "direct", "--foreground", "--yes")
        _taken(from_pair, "prep device", "prep", "task",
               *group_words(from_pair.bundle, "device"), "--target", "this")
        _taken(from_pair, "launch device", "launch", "task", "--stage",
               "device", "--mode", "direct", "--foreground", "--yes")
        _taken(from_pair, "status device", "status", "device")
        _taken(from_pair, "prep transmission", "prep", "task",
               *group_words(from_pair.bundle, "transmission"),
               "--target", "this")
        _taken(from_pair, "launch transmission", "launch", "task", "--stage",
               "transmission", "--mode", "direct", "--foreground", "--yes")
        _taken(from_pair, "summarize", "summarize", "task")

        # ONE VOLTAGE -- a device that runs once, no points: launched, then
        # again warm (its own density stands), then again --cold, which still
        # takes the inputs its first launch gathered (`job-system.md` § 5.4).
        r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
                   "--shape", "hierarchical", "--bundle", "P/transport/S",
                   "--slot", f"junction={cite}", "--bias", "0.0")
        assert r.exit_code == 0, r.output
        single = RealRun(tree=tree, bundle=tree / "P" / "transport" / "S")
        _run_card(single.bundle)
        _set_by_the_person(single.bundle / "S.template.toml",
                           "spin_treatment", "restricted")
        _taken(single, "prep seed and leads", "prep", "task",
               *seed_and_leads(single), "--target", "this")
        _taken(single, "launch seed and leads", "launch", "task",
               *seed_and_leads(single), "--mode", "direct", "--foreground", "--yes")
        _taken(single, "prep device", "prep", "task",
               *group_words(single.bundle, "device"), "--target", "this")
        for step, extra in (("launch device", ()),
                            ("launch device again", ()),
                            ("launch device cold", ("--cold",))):
            _taken(single, step, "launch", "task", "--stage", "device",
                   "--mode", "direct", "--foreground", "--yes", *extra)
        run.single = single

        # WHAT THE DEVICE RAN, CHANGED SINCE (`_ANOTHER_INPUT`): each a
        # copy, changed as a person changes it, its transmission prepared now
        # and its launch asked dry.
        from support.real_runs import copy_of
        run.another = {}
        for row in _ANOTHER_INPUT:
            def changed_since(c, row=row):
                _set_by_the_person(c.bundle / "S.template.toml", row["item"],
                                   row["value"])
                _taken(c, "prep transmission", "prep", "task",
                       *group_words(c.bundle, "transmission"), "--target",
                       "this")
            changed = copy_of(single, row["copy"], changed_since)
            run.another[row["name"]] = _typed(
                changed, "launch transmission dry", "launch", "task",
                "--stage", "transmission", "--mode", "direct", "--dry-run",
                "--yes")

        # A LAUNCHED RUNG OF POINTS AND ITS JOB REQUEST (`execution/job-
        # request.md` § 5): a copy, the device's memory cap set on its run
        # card and its group prepared again -- its runs kept, no run opened
        # -- and a copy with a voltage added, prepared again: refused, what
        # the stage computes changed.
        from support.road import read_request, write_request

        def device_runs(c):
            return " ".join(sorted(p.name for p in
                                   (c.bundle / "04_device").glob("run-*")))

        def request_changed(c):
            req = read_request(c.bundle)
            device = req["stages"]["device"]
            device["execution"] = {**device.get("execution", {}),
                                   "max_memory_mb": 4000}
            write_request(c.bundle, req)
            c.said["runs before"] = Said(0, device_runs(c))
            _typed(c, "prep device again", "prep", "task",
                   *group_words(c.bundle, "device"), "--target", "this")
            c.said["runs after"] = Said(0, device_runs(c))
            _typed(c, "status device", "status", "device")
        run.request_changed = copy_of(run, "request-changed",
                                      request_changed)

        def point_added(c):
            described = c.bundle / "task.json"
            t = json.loads(described.read_text())
            t["bias"]["voltages_v"] = [*t["bias"]["voltages_v"], 0.4]
            described.write_text(json.dumps(t, indent=2))
            _typed(c, "prep device with a point added", "prep", "task",
                   *group_words(c.bundle, "device"), "--target", "this")
        run.point_added = copy_of(run, "point-added", point_added)

        # A TRANSMISSION STOPPED IN ITS LOOP (`model/parse.md` § 5d.5;
        # `run-reports.md` § 2.2): a copy of the single voltage, its loop
        # made long -- the most energy points the item takes, fifty times
        # the default's; its device ran at the Gamma point, so a finer
        # k-point grid is refused by TBtrans -- launched in the background;
        # once TBtrans states a share done, the jobs launched from here
        # asked (`job-system.md` § 6.0a), the run stopped with the line its
        # launch printed and let end, and the jobs asked twice more.
        from support.real_runs import _stopped_by_its_line, _until
        from support.replay import output_lines

        def stopped_in_its_loop(c):
            template = c.bundle / "S.template.toml"
            _set_by_the_person(template, "transmission_n_points", 20001)
            _taken(c, "prep transmission", "prep", "task",
                   *group_words(c.bundle, "transmission"), "--target",
                   "this")
            launched = _taken(c, "launch transmission", "launch", "task",
                              "--stage", "transmission", "--mode", "direct",
                              "--yes")

            def underway():
                try:
                    lines = output_lines(c.attempt("transmission"))
                except (OSError, AttributeError):
                    return False
                return any((_progress_stated(ln) or (0.0,))[0] > 0
                           for ln in lines)
            _until(underway, within=600, what="its loop underway")
            c.jobs["while it runs"] = molbuilder("jobs", "status")
            _stopped_by_its_line(c, "stop as its launch said", launched)
            c.jobs["after its stop"] = molbuilder("jobs", "status")
            c.jobs["again"] = molbuilder("jobs", "status")
        run.stopped = copy_of(single, "stopped-in-its-loop",
                              stopped_in_its_loop)

        # THE SAME FRAMES STATING NO ROWS -- a family, each frame its own
        # curve (`model/structure.md` § 2.2f).
        run.family = "P/structure/family.xyz"
        StructureCodec().write(
            _family_of(_mode_frame_set(tree, pair, vibration)),
            tree / run.family)

        # THE TRANSPORT TAB'S CARD, for each kind it cites (§ 3.8.1): what
        # the describe door says each citation's frames are, and whether the
        # shared panel asks a pseudopotential directory of the person.
        run.tab_said = {}
        for kind, cited in (("run", cite), ("mode's set", frame_set),
                            ("family", run.family)):
            d = client.get(f"/api/transport/describe_attempt?path={cited}")
            s = client.get(f"/api/transport/schema?engine=siesta"
                           f"&surface=shared&junction={cited}")
            run.tab_said[kind] = (d.get_json(), s.get_json())
        # ...and Task setup's echo of the frame-set calculation (its § 3).
        run.folder_said = client.get(
            f"/api/task-setup/folder?dir={from_pair.bundle}").get_json()
        # THE PDOS OF A SELECTION, every atom of the junction, asked where
        # the Results tab asks it, a row of `_PDOS_ASKS` each.
        everyone = range(StructureCodec().load(tree / frame_set,
                                               frames=True).n_atoms)
        run.pdos_said = {}
        for ask in _PDOS_ASKS:
            atoms = ",".join(str(i) for i in ask.get("atoms", everyone))
            got = client.get(f"/api/transport/pdos?path={tree / ask['calc']}"
                             f"&atoms={atoms}&{ask['query']}")
            run.pdos_said[ask["name"]] = (got.status_code, got.get_json())
        # A RUN OF POINTS SENT TO A QUEUE IS ONE JOB (`job-system.md` § 6.0):
        # a calculation citing the same run, prepared for a named machine
        # with queues -- its device's walk, its two points' walls summed,
        # placed at prep with its fit: said over a queue shorter than the
        # sum, and its one header written for the queue it names.  Nothing
        # runs.
        run.walk = _a_walk_prepared_for_a_queue(tree, cite)
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


#: The device's wall, per point, on the walk's run card -- two points sum to
#: 80 minutes, past the short queue's hour.
_POINT_WALL = "00:40:00"


def _a_walk_prepared_for_a_queue(tree: Path, cite: str):
    """A calculation citing ``cite`` at :data:`_BIAS`, its device stating
    :data:`_POINT_WALL` a point, prepared for ``cluster`` -- this machine's
    record with a scheduler and two queues, a short one and a long one,
    copied in as a remote machine's record is (`configuration.md` § 5):
    first naming the short queue, then the long one.  What each prep said,
    and the walk's header as the second wrote it."""
    import dataclasses as _dc
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runfiles import LAUNCH_DIR, GroupNames, RunNames
    from molbuilder.scheduler import Domain
    from molbuilder.scheduler.record import (machine_scope_path,
                                             named_environment_path,
                                             read_environment,
                                             write_environment)
    from molbuilder.task import of as task_of
    from support.real_runs import RealRun, _typed
    from support.road import group_words, read_request, write_request
    from molbuilder.config_dir import ensure_private_dir
    here = read_environment(machine_scope_path())
    # THE NAMED TARGET'S RECORD, copied here as one probed there is.
    ensure_private_dir(named_environment_path("cluster").parent)
    write_environment(_dc.replace(here, scheduler=SLURM, domains=[
        Domain(name="short", partition="cpu", qos="short",
               max_time="01:00:00", max_cores=16,
               node_types=[{"cores": 16, "nodes": 2}]),
        Domain(name="long", partition="cpu", qos="long",
               max_time="1-00:00:00", max_cores=16,
               node_types=[{"cores": 16, "nodes": 2}])]),
        named_environment_path("cluster"))
    r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
               "--shape", "hierarchical", "--bundle", "P/transport/W",
               "--slot", f"junction={cite}",
               "--bias", ",".join(f"{v:g}" for v in _BIAS),
               "--no-low-bias-approximation")
    assert r.exit_code == 0, r.output
    walk = RealRun(tree=tree, bundle=tree / "P" / "transport" / "W")
    _run_card(walk.bundle)
    # THE SPIN RESTRICTED, as the calculation the module runs has it.
    _set_by_the_person(walk.bundle / "W.template.toml", "spin_treatment",
                       "restricted")
    request = read_request(walk.bundle)
    request["allocation"] = {"time": "01:00:00", "mem": "4G"}
    device = request["stages"]["device"]
    device["execution"] = {**device.get("execution", {}),
                           "time": _POINT_WALL}
    for queue in ("short", "long"):
        request["allocation"]["domain"] = queue
        write_request(walk.bundle, request)
        _typed(walk, f"prep on {queue}", "prep", "task",
               *group_words(walk.bundle, "device"), "--target", "cluster")
    described = task_of(walk.bundle)
    home = stage_home(walk.bundle, described, "device")
    names = RunNames.of(described.label, home.token, described.shape)
    header = (home.dir / LAUNCH_DIR
              / GroupNames.of_points(names.stem).name(".sbatch"))
    walk.header = header.read_text() if header.is_file() else None
    # ...and the headers anywhere else below the stage: a point's own,
    # beside its deck, or a run's copy of one.
    walk.point_headers = sorted(
        str(h.relative_to(home.dir))
        for h in home.dir.rglob(names.name(".sbatch")))
    return walk


def _record(run):
    from molbuilder.transport.record import record_path
    from molbuilder.task import of as task_of
    task = task_of(run.bundle)
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


def test_a_run_that_relaxed_no_geometry_is_not_cited(junction):
    """`engines/transport.md` § 3.1: a citation is a relaxation.  A run whose
    output reports a self-consistent field alone -- this calculation's own
    seed rung, or a vibration's force-constant stage, whose last geometry is
    a displacement -- is refused by name.  The silent failure: a geometry
    nothing relaxed composed as the junction."""
    from molbuilder.transport.compose import ComposeError, classify_citation
    seed = junction.attempt("seed")
    with pytest.raises(ComposeError, match="relaxed no geometry"):
        classify_citation(seed)


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
    regions = out.junction.regions
    zs = np.asarray(out.junction.positions)[:, 2]
    assert min(zs[regions[REGION_LEFT_ELECTRODE]]) > max(
        zs[regions[REGION_RIGHT_ELECTRODE]])
    assert out.provenance["swap_electrodes"] is True
    assert _files(junction.tree / junction.cite) == junction.cited_before


def test_the_record_answers_for_the_citation_it_was_composed_from(
        junction, tmp_path, monkeypatch):
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
    written = next(n for n in describe["written"]
                   if n.endswith(".template.toml"))
    for tmpl in (read_template(junction.template_at_init),
                 read_template((junction.described_at / written)
                               .read_text())):
        assert one(tmpl, "mesh_cutoff").value == 150.0
        assert one(tmpl, "basis_size").value == "SZ"
        assert one(tmpl, "electronic_temperature").value == 350.0
        assert one(tmpl, "spin_treatment").value == "unrestricted"
        assert one(tmpl, "unpaired_electrons").value is None
        for name, value in (("mixing_weight", 0.1), ("pulay_history", 6)):
            assert one(tmpl, name).value == value, name
            assert one(tmpl, name).source == "cited", name


def _cited_on_the_tab(page, base, junction) -> dict:
    """The Transport tab citing the relaxation as a returning person's tab
    cites it: its session note, restored through the describe-and-adopt
    path a pick takes (`lib/transport/core.js` `_restoreSession`) -- and
    what the describe door answered the page."""
    page.goto(f"{base}/transport-calculation",
              wait_until="domcontentloaded")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.workspace "
        "      && typeof window.molbuilder.workspace.persist "
        "         === 'function'", timeout=10000)
    assert page.evaluate(
        """async (j) => {
             const ws = window.molbuilder.workspace;
             const id = {workspace_id: ws.workspaceId('transport:panel'),
                         state_index: 0};
             ws.persist('transport:panel', {v: 3, junction: j}, id);
             for (let i = 0; i < 50; i++) {
               const n = await ws.readState(id);
               if (n && n.junction === j) return true;
               await new Promise(r => setTimeout(r, 100));
             }
             return false;
           }""", junction.cite), "the session note never landed"
    with page.expect_response(
            lambda r: "/api/transport/describe_attempt" in r.url,
            timeout=30000) as described:
        page.reload(wait_until="domcontentloaded")
    page.wait_for_function(
        "(j) => (document.getElementById('transport-junction-readout') || {})"
        "        .textContent === j", arg=junction.cite, timeout=15000)
    return described.value.json()


def test_the_tabs_chemistry_card_answers_for_the_cited_run(junction, page,
                                                           monkeypatch):
    """The Transport tab's chemistry card, on the relaxation it cites
    (`science/chemistry-correctness.md` § 2.5, § 2a.1a): the charge 0 by
    rule; the spin the cited run's -- unrestricted (ES7) -- its count
    floating, since a transport rung cannot hold a fixed one (ES6); and
    stating restricted on the shared panel moves the card to a closed
    shell.  Cited as a returning person's tab cites it.

    MUTATION THIS MUST FAIL AGAINST: the card asked without the panel's
    spin (the analyze door dropping it) -- stating restricted leaves the
    card where the cited run's record put it."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.live_server import serve
    from test_chemistry_card_e2e import _answered, _blocks
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    with serve() as base:
        _cited_on_the_tab(page, base, junction)
        _answered(page, "SIESTA")
        blocks = _blocks(page)
        assert list(blocks) == ["SIESTA"], "transport runs on SIESTA alone"
        card = blocks["SIESTA"]
        assert "0 — rule: a transport junction is neutral" in card, card
        assert "unrestricted, the moment floats (free)" in card, card

        page.select_option("#t-spin-treatment", "restricted")
        _answered(page, "closed shell")
        card = _blocks(page)["SIESTA"]
        assert "restricted (closed shell, 2S = 0)" in card, card


def test_the_tab_shows_what_the_citation_measured(junction, page,
                                                  monkeypatch):
    """Every finding the describe door answers the Transport tab for the
    junction it cites -- each lead's measurements among them -- is drawn on
    the tab, a row each, by the one renderer (`science/validation.md`
    § 4.1 R2, R2a, R3; `web/ui-contract.md` § 5.1).

    THE SILENT FAILURE: the tab's script finds no renderer on the page -- it
    reads it off a global that only a separate entry point publishes -- and
    draws nothing, so what the composition measured never reaches the
    person who cited the junction.

    MUTATIONS THIS MUST FAIL AGAINST: the page without the renderer's
    global entry point; a finding drawn twice."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.findings import assert_every_finding_drawn
    from support.live_server import serve
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    with serve() as base:
        said = _cited_on_the_tab(page, base, junction)["findings"]
        assert said, "the describe door measured nothing on the junction"
        assert_every_finding_drawn(page, ["#transport-junction-findings"],
                                   said)


def test_describe_is_one_act_and_refuses_a_described_folder(junction):
    """`engines/transport.md` § 3.8.2a: the describe door writes a
    description into a folder that holds none and refuses one that holds one
    -- the one-calculation-per-folder refusal `molbuilder describe` asks, in the door
    that writes -- leaving the folder exactly as it was.  Without it a second
    Describe silently replaces a calculation Task setup has since changed:
    its rungs' values, its run settings."""
    said = junction.said["describe again"]
    assert said.exit_code == 409, said.output
    assert "one calculation per folder" in json.loads(said.output)["error"]
    assert junction.described_after == junction.described_before


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
        # THE RECORDED CONTRACT'S ANSWERS -- the pseudopotentials apart: a
        # cited run's travel with it, a pair's are the person's directory
        # (§ 3.1; asserted below).
        return {it.name: it.value for it in select(tmpl, engine="siesta")
                if it.source == "cited" and it.name != "psml_lib"}
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


#: THE ATOM FREED in the rows below: the right block's atom next to the
#: bridge, which the relaxation held.
_FREED = _at(_LEAD + _WIRE)


@pytest.mark.parametrize("change, severity, words, as_the_deck", [
    ("hold", "info", "holds 21: every atom it held, and 1 more", False),
    ("free", "warn", f"leaves atom index {_FREED} (H) of them free", False),
    ("free", "warn", f"leaves atom index {_FREED} (H) of them free", True),
], ids=["holds-more-for-the-spectrum", "frees-an-atom-the-relaxation-held",
        "names-it-in-your-order-on-the-decks-sorted-copy"])
def test_a_vibration_of_the_relaxed_pair_is_judged_by_what_it_held(
        junction, change, severity, words, as_the_deck):
    """`engines/vibration.md` § 2.3a, on the road it was written for: the
    junction relaxed with every atom held but the middle two, saved as a
    pair whose `info.relaxation` record names the atoms the relaxation held
    (`held_atom_keys`) -- history the labels do not change.  Holding one
    middle atom more for the spectrum is the partial Hessian over fewer
    atoms, every one balanced: information.  Freeing the right electrode's
    atom next to the bridge, which the relaxation held, is a warning naming
    it: its force was never balanced -- named in the user order also when the
    check is asked of the deck's held-first copy, as prep asks it, where that
    atom stands at another index (`model/overview.md` § 2.2a, I3: the sorted
    order never reaches a person)."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate
    from molbuilder.workingcopy_structure import StructureCodec
    s = StructureCodec().load(junction.tree / junction.pair)
    held = set(s.frozen_atoms)
    assert len(held) == 20 and _FREED in held, sorted(held)
    s.frozen_atoms = sorted(held | {_at(_LEAD + _WIRE // 2 - 1)}
                            if change == "hold" else held - {_FREED})
    if as_the_deck:
        from molbuilder.transport.sort import deck_sort_key, sort_by
        s = sort_by(s, deck_sort_key("siesta", "vibration")).structure
        # the row's premise: in the copy the freed atom has another index
        assert s.map_from_user_index([_FREED]) != [_FREED]
    cfg = SiestaConfig(system_label="J", psml_lib="pseudopotential",
                       already_relaxed=True, **RELAXED_WITH)
    found = [i for i in validate(s, cfg, calculation="vibration")
             if i.message.startswith("The relaxation held")]
    assert [(i.severity, words in i.message) for i in found] == [
        (severity, True)], found


def test_a_pair_whose_relaxation_left_an_electrode_atom_free_is_refused(
        junction):
    """`engines/transport.md` § 1.1, Step 0: a lead is the bulk its
    relaxation held still, and the record says which atoms that was -- so a
    pair saved from the relaxation whose electrode label is then widened by
    the wire's two atoms beside it, the one the relaxation moved held now
    too, is refused at the citation, naming that atom.  The silent failure
    (plan V1.43): the label alone was read, and the pair composed a lead out
    of an atom nothing held while the bridge relaxed."""
    import shutil

    from molbuilder.transport.compose import ComposeError, compose_junction
    from molbuilder.workingcopy_structure import StructureCodec
    s = StructureCodec().load(junction.tree / junction.pair)
    moved, beside = _at(_LEAD + 1), _at(_LEAD)
    assert moved not in s.frozen_atoms and beside in s.frozen_atoms, (
        sorted(s.frozen_atoms))
    s.regions["L-electrode"] = sorted(
        set(s.regions["L-electrode"]) | {beside, moved})
    s.regions["bridge"] = [i for i in s.regions["bridge"]
                           if i not in (beside, moved)]
    s.frozen_atoms = sorted(set(s.frozen_atoms) | {moved})
    where = junction.tree / "P" / "relabelled"
    try:
        StructureCodec().write(s, where / "relabelled.xyz")
        with pytest.raises(ComposeError) as refused:
            compose_junction("P/relabelled/relabelled.xyz",
                             tree_root=junction.tree)
    finally:
        shutil.rmtree(where, ignore_errors=True)
    said = str(refused.value)
    assert "NOT HELD by the relaxation" in said, said
    assert f"atom index {moved} (H)" in said, said


def _prepared_decks(run, stage):
    """The decks ``stage`` prepared: its one deck in the stage's folder, or
    each point's (`transport.stages.point_folders`) -- named by the run's
    one naming door (`runfiles.RunNames`)."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    from molbuilder.transport.stages import point_folders
    task = task_of(run.bundle)
    home = stage_home(run.bundle, task, stage)
    names = home.names(task.shape)
    folders = [f for f, _v in point_folders(run.bundle, task, stage)]
    return [f / names.name(".fdf") for f in (folders or [home.dir])]


#: The ladder's rungs, a tab each (`engines/transport.md` § 3.8.2a).
_RUNGS = TRANSPORT_STAGES


def _tab_rows():
    """``(rung, item)`` of every value a rung's tab offers as a control,
    asked of the tab's own schema (`catalogue_to_form_schema`, the rung
    surface) -- so an item added tomorrow is a row the day it appears -- but
    those whose effect is real and elsewhere
    (`support.form_fields.NOT_IN_THE_DECK`, each with its reason)."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from support.form_fields import NOT_IN_THE_DECK
    rows = []
    for rung in _RUNGS:
        schema = catalogue_to_form_schema(
            "siesta", f"t-{rung}", calculation="transport", surface="rung",
            rung=rung)
        rows += [(rung, field["name"])
                 for section in schema["sections"]
                 for field in section["fields"]
                 if not field.get("locked")
                 and ("siesta", field["name"]) not in NOT_IN_THE_DECK]
    return rows


def _the_decks_a_person_sets(tree, cite: str, client):
    """THE DECKS A PERSON'S VALUES ARE ASKED OF -- a calculation described
    citing the relaxation ``cite``, its spin restricted as the others',
    every group prepared and none launched; each rung's decks as they render
    with nothing set, prepared again with nothing set, with each value a
    rung's tab offers set alone on it, and with each value the shared panel
    offers set alone in the template (the rows below)."""
    from support.real_runs import RealRun, _taken
    from support.road import group_words
    # AT A VOLTAGE AS WELL AS AT ZERO, each its own device run: TranSIESTA
    # reads its bias window only under bias (`engines/transport.md` § 6.1),
    # so an item it offers is read in the device's deck under bias.
    r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
               "--shape", "hierarchical", "--bundle", "P/transport/D",
               "--slot", f"junction={cite}",
               "--bias", ",".join(f"{v:g}" for v in _BIAS),
               "--no-low-bias-approximation")
    assert r.exit_code == 0, r.output
    probe = RealRun(tree=tree, bundle=tree / "P" / "transport" / "D")
    _run_card(probe.bundle)
    _set_by_the_person(probe.bundle / "D.template.toml",
                       "spin_treatment", "restricted")
    _taken(probe, "prep", "prep", "task",
           *group_words(probe.bundle, *_RUNGS), "--target", "this")
    probe.unset = {rung: [d.read_text()
                          for d in _prepared_decks(probe, rung)]
                   for rung in _RUNGS}
    # ...PREPARED AGAIN WITH NOTHING SET, every group...
    _taken(probe, "again", "prep", "task",
           *group_words(probe.bundle, *_RUNGS), "--target", "this")
    probe.again = {rung: [d.read_text()
                          for d in _prepared_decks(probe, rung)]
                   for rung in _RUNGS}
    # ...EACH VALUE A RUNG'S TAB OFFERS, set alone on that rung, its
    # group prepared again (`_tab_rows`)...
    unset = (probe.bundle / "task.json").read_text()
    probe.set = {(rung, name): _set_on_its_tab(probe, client, unset,
                                               rung, name)
                 for rung, name in _tab_rows()}
    (probe.bundle / "task.json").write_text(unset)
    # ...AND EACH VALUE THE SHARED PANEL OFFERS, set alone in the
    # template, every group prepared again (`_shared_rows`).
    template = (probe.bundle / "D.template.toml").read_text()
    probe.shared = {name: _set_in_the_template(probe, template, name)
                    for name in _shared_rows()}
    (probe.bundle / "D.template.toml").write_text(template)
    return probe


def _another_than_runs(probe, rung: str, name: str, task: dict):
    """``(item, was, value)``: what ``rung`` runs for ``name`` with nothing
    set on it, and a value one step from it among what a transport
    calculation offers (`support.form_fields`, `template.offered`).  WHAT
    RUNS for a blank item is what answers it: a rule's answer the rung's
    deck states (`resolve.BLANK_ANSWERED_BY` -- a lead's many k-points turn
    `parallel_over_k` on), read by the one fdf reader; else the engine's
    own default the catalogue states (`engine_default`); else the item's
    documented default."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.parse.fdf import fdf_key, fdf_true, read_fdf
    from molbuilder.resolve import BLANK_ANSWERED_BY, resolved_ladder
    from molbuilder.task import Task
    from molbuilder.template import catalogue, offered, one
    from support.form_fields import another_value
    template = (probe.bundle / "D.template.toml").read_text()
    item = one(catalogue(), name, engine="siesta")
    was = getattr(dict(resolved_ladder(template, Task.from_dict(task),
                                       SiestaConfig))[rung], name)
    if was is None and item.type == "bool" and name in BLANK_ANSWERED_BY:
        stated = read_fdf(probe.unset[rung][0])[0].get(
            fdf_key(item.engine_key))
        was = fdf_true(stated) if stated else None
    if was is None:
        was = item.engine_default[0] if item.engine_default else item.default
    return item, was, another_value(
        item, was, choices=offered(item, "siesta", "transport") or None)


def _set_on_its_tab(probe, client, unset: str, rung: str, name: str):
    """``name`` set alone on ``rung`` of the calculation described as
    ``unset`` (its ``task.json``), as the tab writes it -- through Task
    setup's save, a value one step from what the rung runs now
    (:func:`_another_than_runs`), among the rung's own -- a tab offers no
    run setting, the run card's (`execution/job-request.md` § 2) -- and the
    rung's group prepared again: ``{was, value, saved, said, decks}``, the
    decks' texts as prep wrote them."""
    from support.real_runs import _typed
    from support.road import group_words, read_request, save_both
    task = json.loads(unset)
    _item, was, value = _another_than_runs(probe, rung, name, task)
    stage = next(s for s in task["stages"] if s["name"] == rung)
    stage["overrides"] = {name: value}
    task["varies"] = [name]
    saved = save_both(probe.bundle, task, read_request(probe.bundle), client)
    said = _typed(probe, f"{rung} {name}", "prep", "task",
                  *group_words(probe.bundle, rung), "--target", "this")
    return {"was": was, "value": value,
            "saved": (saved.status_code, saved.get_json()), "said": said,
            "decks": [d.read_text() for d in _prepared_decks(probe, rung)]}


@pytest.mark.parametrize("rung, name", _tab_rows(),
                         ids=lambda v: str(v))
def test_every_value_a_rungs_tab_offers_reaches_its_deck(junction, rung,
                                                         name):
    """A value set on a rung's tab changes what that rung's engine reads
    (`engines/transport.md` § 3.8.2a: a tab offers the items its rung owns
    and the items any rung may set; § 3.8.9: the deck writes each rung its
    own).  Each row is one value set alone on one rung, as the tab writes it,
    one step from what the rung runs with nothing set, and that rung's group
    prepared again (`_set_on_its_tab`); its decks, where
    prep wrote them, are asked whether the engine now reads another input
    than with nothing set in at least one of them, by the gather's own
    question (`siesta.engine_input.engine_input_differences`, § 6.1) -- the
    probe is described at a voltage as well as at zero, so an item the engine
    reads under bias alone is read in one.
    THE SILENT FAILURE: a rung's layout is a hand list of items in
    `transport/deck.py` beside the catalogue's `stages`, so an item the tab
    offers and the layout leaves out is set, stored, and never written --
    the run takes the engine's default and nothing says so (§ 3.4.2: an item
    absent from a stage's layout is not written).  Found 2026-10-16:
    `TBT.Progress` taken out of the transmission's section turned no test
    red."""
    from molbuilder.siesta.engine_input import engine_input_differences
    row = junction.probe.set[(rung, name)]
    assert row["value"] is not None and row["value"] != row["was"], row
    assert row["saved"][0] == 200, row["saved"]
    assert row["said"].exit_code == 0, row["said"].output
    unset = junction.probe.unset[rung]
    assert len(row["decks"]) == len(unset), row["decks"]
    assert any(engine_input_differences(before, deck)
               for before, deck in zip(unset, row["decks"])), (
        f"{rung}: {name} set to {row['value']!r} (it ran {row['was']!r}) -- "
        f"no deck of the rung gives the engine another input")


#: Shared values this junction admits no other value for, each with why.
_NO_OTHER_VALUE_HERE = {
    "species_order":
        "the junction holds one element, H: there is no other order",
    "unpaired_electrons":
        "on a transport calculation the spin treatment decides it -- 0 "
        "restricted, floating unrestricted, a stated count refused "
        "(`engines/transport.md` § 3.1's spin note) -- so no value of its "
        "own changes a deck",
}


def _shared_rows():
    """Every value the shared panel offers as a control
    (`catalogue_to_form_schema`, the shared surface), but those whose
    effect is real and elsewhere (`support.form_fields.NOT_IN_THE_DECK`)
    and those this junction admits no other value for
    (`_NO_OTHER_VALUE_HERE`)."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from support.form_fields import NOT_IN_THE_DECK
    schema = catalogue_to_form_schema("siesta", "t-shared",
                                      calculation="transport",
                                      surface="shared")
    return [field["name"] for section in schema["sections"]
            for field in section["fields"]
            if not field.get("locked")
            and ("siesta", field["name"]) not in NOT_IN_THE_DECK
            and field["name"] not in _NO_OTHER_VALUE_HERE]


def _set_in_the_template(probe, unset: str, name: str):
    """``name`` set alone in the calculation's template, its text as
    described being ``unset``, as a person sets it there (the shared panel
    edits the template, `engines/transport.md` § 3.8.2), every group
    prepared again: ``{was, value, said, decks}``, the decks of every rung
    that reads the item (`template.reads`)."""
    from molbuilder.template import catalogue, one, reads
    from support.real_runs import _typed
    from support.road import group_words
    template = probe.bundle / "D.template.toml"
    template.write_text(unset)
    task = json.loads((probe.bundle / "task.json").read_text())
    item = one(catalogue(), name, engine="siesta")
    readers = [r for r in _RUNGS if reads(item, "siesta", "transport", r)]
    _item, was, value = _another_than_runs(probe, readers[0], name, task)
    _set_by_the_person(template, name, value)
    said = _typed(probe, f"shared {name}", "prep", "task",
                  *group_words(probe.bundle, *_RUNGS), "--target", "this")
    return {"was": was, "value": value, "said": said,
            "decks": {r: [d.read_text() for d in _prepared_decks(probe, r)]
                      for r in readers}}


@pytest.mark.parametrize("name", _shared_rows())
def test_every_shared_value_reaches_every_deck_that_reads_it(junction,
                                                             name):
    """A value set on the shared panel -- in the calculation's template --
    changes what every rung that reads it gives its engine
    (`engines/transport.md` § 3.8.9: one value, every rung; § 2a.13 Class
    A).  Each row is one value set alone in the template and every group
    prepared again (`_set_in_the_template`); each reading rung's deck is
    asked, as the rung rows ask theirs.  THE SILENT FAILURE: a rung's
    layout leaves out a shared item it reads -- `electronic_temperature`
    out of the transmission's section, `electrodes_bulk` out of the TBtrans
    or TranSIESTA section -- and that rung runs on the engine's own value
    while every other rung runs on yours."""
    from molbuilder.siesta.engine_input import engine_input_differences
    row = junction.probe.shared[name]
    assert row["value"] is not None and row["value"] != row["was"], row
    assert row["said"].exit_code == 0, row["said"].output
    for rung, decks in row["decks"].items():
        unset = junction.probe.unset[rung]
        assert len(decks) == len(unset), (rung, decks)
        for before, deck in zip(unset, decks):
            assert engine_input_differences(before, deck), (
                f"{rung}: {name} set to {row['value']!r} (it ran "
                f"{row['was']!r}) -- the deck gives the engine the same "
                f"input")


def test_preparing_again_with_nothing_set_changes_no_deck(junction):
    """Every group prepared again with nothing changed gives every rung's
    engine the same input -- the premise each row above stands on: a
    difference a row finds is its value's, not one a second prep makes on
    its own (`engines/transport.md` § 6.1, *the same engine input*)."""
    from molbuilder.siesta.engine_input import engine_input_differences
    probe = junction.probe
    for rung in _RUNGS:
        for before, again in zip(probe.unset[rung], probe.again[rung]):
            assert engine_input_differences(before, again) == [], rung


def test_each_scf_stage_writes_its_own_mixer_and_the_transmission_none(
        junction):
    """SCF-RUNG (`engines/transport.md` § 2a.13 Class C, § 3.8.2a; TD12):
    with several stages' own mixing weights set at once, each stage's decks
    state its own value exactly and no other's; a stage that sets none
    writes the template's -- the cited run's 0.1, TD6; and the transmission,
    which runs no SCF, writes none -- what the rows above, one value at a
    time and asking only that the input changed, do not ask.  Each deck is read
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


def test_every_rung_of_the_junction_states_its_order(junction):
    """`model/overview.md` § 2.2a, I5, I7: every rung written from the
    composed junction -- the seed, and the device's and the transmission's
    points, which a later prep renders from the composition record -- states
    in its deck the order record of TranSIESTA's order, the chain's
    (:data:`_LISTED`); a lead's deck, a cell of its own, states none.  The
    silent failure: a rung whose deck states no order, so its run reads as
    yours while its atoms are in TranSIESTA's.  Each run asked through the
    run door (`runs.run_of`) and its reader (`runs.declared`)."""
    from molbuilder.runs import declared, run_of
    from molbuilder.transport.stages import Point
    runs = ([(junction.attempt(s), w) for s, w in
             (("seed", _CHAIN), ("electrode_L", None), ("electrode_R", None))]
            + [(junction.attempt(rung) / Point(volts=v).rel, _CHAIN)
               for rung in ("device", "transmission") for v in _BIAS])
    for where, want in runs:
        got = declared(run_of(where)).atom_order
        assert (None if got is None else got.to_dict()) == want, where


def test_a_launched_rung_of_points_is_prepared_again_for_its_job_request(
        junction):
    """`execution/job-request.md` § 5: a launched group, its job request
    changed, is prepared again -- every member rendered anew, no run opened,
    the next launch opening the next -- on a rung that runs per point as on
    any other.  The silent failure: a run opened under the finished ones, so
    the stage reads as prepared and never launched, and the transmission's
    gather refuses the device's newest run as never launched.

    MUTATION THIS MUST FAIL AGAINST: a rung of points prepared again opening
    a run whatever its runs were."""
    c = junction.request_changed
    said = c.said["prep device again"]
    assert said.exit_code == 0, said.output
    assert "prepared device again: launched" in said.output, said.output
    assert c.said["runs after"].output == c.said["runs before"].output, (
        c.said["runs before"].output, c.said["runs after"].output)
    assert "prepared, not launched" not in c.said["status device"].output, (
        c.said["status device"].output)


def test_a_point_added_to_a_launched_rung_is_refused(junction):
    """`execution/job-request.md` § 5: a launched stage's decks are the
    calculation its runs ran, and a voltage added to a launched device is
    another -- refused, naming what was added and the way back, nothing
    written.  The silent failure: the new point's deck, with no copy to be
    compared with, prepared as though nothing had changed.

    MUTATION THIS MUST FAIL AGAINST: the points of a launched rung not
    asked against the ones its runs ran."""
    said = junction.point_added.said["prep device with a point added"]
    assert said.exit_code != 0, said.output
    assert "its points now are not the ones its runs ran" in said.output, \
        said.output
    assert "added: v0.4" in said.output, said.output


@pytest.mark.parametrize("row", _ANOTHER_INPUT, ids=lambda r: r["name"])
def test_a_run_of_another_engine_input_is_not_taken(junction, row):
    """`engines/transport.md` § 6.1, *the same engine input*: a transport
    rung's launch takes an upstream run only when it ran what the upstream
    renders now.  Each row of `_ANOTHER_INPUT` -- a number, a block --
    changed after the device ran: the transmission's dry launch is refused,
    naming the keyword and the device's run.  The silent failure: the
    transmission integrating a device Hamiltonian computed on another mesh
    or k-grid."""
    said = junction.another[row["name"]]
    assert said.exit_code != 0, said.output
    assert row["names"] in said.output, said.output
    assert "04_device/run-2" in said.output, said.output


def _same_engine_input_cases():
    import tomllib
    data = Path(__file__).parent / "data" / "same_engine_input.toml"
    return tomllib.loads(data.read_text())["case"]


@pytest.mark.parametrize("case", _same_engine_input_cases(),
                         ids=lambda c: c["name"])
def test_the_geometry_is_compared_where_siesta_places_it(junction, case):
    """`engines/transport.md` § 6.1: the atoms' places are compared in Å,
    each within 10⁻⁶ Å (`structure.SAME_PLACE_ANG`).  The device's deck
    rendered from the recorded junction, and again from it with one bridge
    atom moved along x by the row's distance, through the one builder
    (`gather.rung_deck_now`) -- nothing written.  The silent failure: a
    device run on another geometry taken as this one's, or a run refused
    for the rounding of a box placed again."""
    import dataclasses

    from molbuilder.jobset.gather import composed_junction, rung_deck_now
    from molbuilder.siesta.engine_input import engine_input_differences
    from molbuilder.transport.sort import REGION_BRIDGE
    base = junction.single.bundle
    from molbuilder.calculation import open as open_calculation
    calc = open_calculation(base)
    composed = composed_junction(calc)
    atom = composed.junction.regions[REGION_BRIDGE][0]
    positions = np.array(composed.junction.positions, dtype=float)
    positions[atom, 0] += case["moved_ang"]
    moved = dataclasses.replace(
        composed, junction=composed.junction.replace(positions=positions))
    differs = engine_input_differences(
        rung_deck_now(calc, "device", composed),
        rung_deck_now(calc, "device", moved))
    if case.get("names"):
        assert any(case["names"] in d for d in differs), differs
    else:
        assert differs == [], differs


def _left_to_the_engine_cases():
    import tomllib
    data = Path(__file__).parent / "data" / "same_engine_input.toml"
    return tomllib.loads(data.read_text())["left_to_the_engine"]


def _left_to_the_engine(text: str, left_out) -> str:
    """``text`` with ``left_out`` taken out of it, as a deck that leaves
    them to the engine reads -- a keyword's line by its name, a block whole
    by ``%block <name>``, a block's row by ``row <first word>``."""
    out, skip = [], None
    for line in text.splitlines():
        words = line.split()
        if skip is not None:
            if words[:2] == ["%endblock", skip]:
                skip = None
            continue
        if words[:1] == ["%block"] and f"%block {words[1]}" in left_out:
            skip = words[1]
            continue
        if words and (words[0] in left_out or f"row {words[0]}" in left_out):
            continue
        out.append(line)
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("case", _left_to_the_engine_cases(),
                         ids=lambda c: c["name"])
def test_what_a_deck_leaves_to_the_engine_is_read_as_siesta_reads_it(
        junction, case):
    """`engines/transport.md` § 6.1, *the same engine input*: the device's
    deck rendered now at the row's voltage, through the one builder
    (`gather.rung_deck_now`), against the same deck with the row's
    statements left to the engine -- the engine's scratch rows, the file
    switch, the bias window -- and the row's changes made to the deck now.
    The silent failures: a device run prepared by an older molbuilder, its
    deck leaving to the engine what this one states, refused as another
    calculation and its hours of compute thrown away; or a real difference
    -- a spacing other than the engine's -- taken as none."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.gather import composed_junction, rung_deck_now
    from molbuilder.siesta.engine_input import engine_input_differences
    from molbuilder.transport.stages import frames_of, rung_points
    base = junction.bundle
    calc = open_calculation(base)
    point = next(p for p in rung_points(calc.task, "device",
                                        frames=frames_of(base))
                 if p.volts == case["volts"])
    now = rung_deck_now(calc, "device", composed_junction(calc), point=point)
    ran = _left_to_the_engine(now, case.get("left_out", []))
    for old, new in case.get("changed", []):
        assert old in now, (old, now)
        now = now.replace(old, new)
    differs = engine_input_differences(ran, now)
    if case.get("names"):
        for name in case["names"]:
            assert any(name in d for d in differs), (name, differs)
        assert len(differs) == len(case["names"]), differs
    else:
        assert differs == [], differs


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
    downstream take from it; `status` says how many points are done.  Each
    point's progress log is in the run's point, which writes it, and none
    is left in the point's folder below the stage, where nothing writes to
    it again (`project-layout.md` § 1.0)."""
    from molbuilder.runfiles import WRITTEN, role_of
    from molbuilder.task import of as task_of
    label = task_of(junction.bundle).label
    progress = {a.role for a in WRITTEN if a.output == "progress"}
    for v in _BIAS:
        p = _point(junction, "04_device", 0, v)
        for suffix in (".TS.HSX", ".TSDE"):
            assert (p / f"{label}{suffix}").is_file(), (p, suffix)
        assert any(role_of(f.name) in progress for f in p.iterdir()), p
        prepared = junction.bundle / "04_device" / f"v{v:g}"
        left = [f.name for f in prepared.iterdir()
                if role_of(f.name) in progress]
        assert left == [], (prepared, left)
    _said(junction, "status device", "2 of 2 points done")


@pytest.mark.parametrize("step, under", [
    ("launch seed and leads", "launch/"),          # a group's walk
    ("launch device", "04_device/launch/"),        # a sweep's points
])
def test_a_walk_run_here_names_its_log(junction, step, under):
    """A group's walk and a sweep's say where their members' output goes --
    the walk's log under `launch/`, which the walk writes, its own errors
    among it (`execution/job-system.md` § 6.0).  The silent failure: a walk
    left in the background, its output's place unsaid."""
    said = junction.said[step].output
    log = next((ln.split("the walk's output: ", 1)[1].strip()
                for ln in said.splitlines() if "the walk's output: " in ln),
               None)
    assert log is not None and log.startswith(under), said
    assert (junction.bundle / log).is_file(), log


def test_a_walk_of_points_is_placed_at_prep_its_walls_summed(junction):
    """A run of points goes to a queue as one job walking every point, its
    wall the points' summed (`job-system.md` § 6.0): prep places that one
    request on the queue it names and says its fit -- on a queue shorter
    than the sum, the wall it is over, with the numbers -- before anything
    is sent; the scheduler decides (`scheduler.md` R6)."""
    r = junction.walk.said["prep on short"]
    assert r.exit_code == 0, r.output
    for words in (f"one job walking its {len(_BIAS)} points (their walls "
                  f"summed)", "-t 0-01:20:00",
                  "1h20m is over short's wall (01:00:00)"):
        assert words in r.output, r.output


def test_a_walk_of_points_is_written_at_prep_as_the_one_header_launch_sends(
        junction):
    """On a queue that takes it, prep writes the walk's one header -- the
    summed wall, the queue it names -- and prints every value of it, so
    launch sends it as it is (`submission.md` S4); and it is the run's one
    header: no point carries one of its own, which nothing sends."""
    from molbuilder.scheduler.emit import Directives
    from molbuilder.scheduler.kinds import kind
    r = junction.walk.said["prep on long"]
    assert r.exit_code == 0, r.output
    lines = Directives.lines_of(junction.walk.header or "", kind("slurm"))
    assert "#SBATCH -t 0-01:20:00" in lines, lines
    assert "#SBATCH -p cpu" in lines and "#SBATCH -q long" in lines, lines
    assert "header: " in r.output and "-t 0-01:20:00" in r.output, r.output
    assert junction.walk.point_headers == [], junction.walk.point_headers


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
    assert any(str(src) == "01_seed/run-0"
               for src, _fn in read_gathered_from(zero)), zero
    start = (_point(junction, "04_device", 1, _BIAS[1])
             / ".continued-from").read_text()
    assert "run-1" in start and f"v{_BIAS[0]:g}" in start, start


def test_a_device_launched_warm_then_cold_keeps_the_seeds_density(junction):
    """One voltage, so one run per launch: run 1, launched again warm,
    continues from run 0's own density and records only what the gather
    gave it beside that; run 2, launched again --cold, still takes every
    input its first launch gathered -- the seed's density and each lead's
    Hamiltonian, each from the run that made it (`job-system.md` § 5.4,
    *either way a transport rung keeps the inputs its first launch
    gathered*)."""
    from molbuilder.runrecord import read_gathered_from
    device = junction.single.bundle / "04_device"
    first = {fn: str(src) for src, fn in read_gathered_from(device / "run-0")}
    cold = {fn: str(src) for src, fn in read_gathered_from(device / "run-2")}
    assert cold == first, (first, cold)
    assert any(fn.endswith(".DM") and src == "01_seed/run-0"
               for fn, src in cold.items()), cold


def test_the_transmission_takes_the_devices_newest_run_whole(junction):
    """Its first launch gathers from the device's newest run, and says
    which (`job-system.md` § 6.0, step 1)."""
    _said(junction, "launch transmission", "04_device/run-1")
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
    from molbuilder.task import of as task_of
    from molbuilder.transport.compose import recorded_junction
    run = junction.from_pair
    label = task_of(run.bundle).label
    for f in range(_N_FRAMES):
        for v in _BIAS:
            p = _frame_point(run, "04_device", f, v)
            for suffix in (".TS.HSX", ".TSDE"):
                assert (p / f"{label}{suffix}").is_file(), (p, suffix)
    # Against frame 0's point at the same voltage, each frame's point moved
    # its atoms as the composed set's own frame moves them -- the deck's
    # atom order -- and by nothing else.
    composed = np.asarray(recorded_junction(run.bundle).structure.frames,
                          dtype=float)
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
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runfiles import RunNames
    from molbuilder.runrecord import read_continued_from, read_gathered_from
    from molbuilder.task import of as task_of
    run = junction.from_pair
    task = task_of(run.bundle)
    names = RunNames.of(task.label, stage_home(run.bundle, task,
                                               "device").token, task.shape)
    for f in range(_N_FRAMES):
        first = _frame_point(run, "04_device", f, _BIAS[0])
        assert read_continued_from(first, names, 0) is None, first
        assert any(str(src) == "01_seed/run-0"
                   for src, _fn in read_gathered_from(first)), first
        nxt = _frame_point(run, "04_device", f, _BIAS[1])
        assert str(read_continued_from(nxt, names, 0)) == str(
            first.relative_to(run.bundle)), nxt


def test_the_leads_greens_functions_are_the_rounds_one_pair(junction):
    """§ 2a.11 *What a run's points share*: the device run keeps the leads'
    surface Green's function files in its own `tsgf/`, stamped the run's
    scratch; the walk runs round by round -- every frame's point at a
    voltage before the next voltage -- and empties the folder as each
    round begins and when it ends, saying what it removed in its log; the
    round's first point computed the pair and every later frame's point
    found and read it, as the engine says in its own output (*Using
    GF-file*); `status device` says the folder is empty, and the stage's
    line before a launch says the size from the leads' own orbital count.

    MUTATION THIS MUST FAIL AGAINST: the `GF` path left at the point or
    loose in the run (no file ever in `tsgf/`, nothing emptied); the frame
    loop outside (a v0.2 point before the last v0 point); the emptying
    skipped (the pair still in the folder at the end); `GF.ReUse` off (no
    *Using GF-file*); the estimate not reading the lead's run."""
    from molbuilder import calcdirs
    from molbuilder.jobset.naming import stage_home
    from molbuilder.parse import detect
    from molbuilder.runs import openable, run_of
    from molbuilder.task import of as task_of
    from molbuilder.transport.gf_files import status_line
    run = junction.from_pair
    task = task_of(run.bundle)
    device = run.bundle / "04_device" / "run-0"
    folder = device / "tsgf"
    assert folder.is_dir() and calcdirs.where(folder).what == calcdirs.SCRATCH
    # ...and the walk's machinery beside the runs is the stage's support,
    # never a point (`project-layout.md` § 1.4a, *Where am I*).
    assert calcdirs.where(run.bundle / "04_device" / "launch").what \
        == calcdirs.SUPPORT
    assert [p.name for p in folder.iterdir() if ".TSGF" in p.name] == []
    log = next((run.bundle / "04_device" / "launch").glob("*.log")).read_text()
    lines = [ln.split("] ", 1)[1] for ln in log.splitlines() if "] [" not in ln
             and "] " in ln]
    starts = [ln.split("-> ", 1)[1].split(" starts")[0]
              for ln in lines if " starts" in ln and "-> " in ln]
    # THE ROUNDS: every frame at the first voltage, then every frame at the
    # next -- the frame loop inside.
    volts_of = lambda name: name.rsplit("/", 1)[1]
    assert starts and [volts_of(n) for n in starts] == sorted(
        [volts_of(n) for n in starts], key=lambda t: float(t[1:])), starts
    emptied = [ln for ln in lines if "tsgf/" in ln]
    assert len(emptied) == len(_BIAS) + 1, emptied
    assert emptied[0].endswith("tsgf/ empty"), emptied
    assert all(" emptied -- " in ln and ".TSGF" in ln for ln in emptied[1:]), emptied
    for f in range(_N_FRAMES):
        for v in _BIAS:
            p = _frame_point(run, "04_device", f, v)
            assert not [x.name for x in p.iterdir() if ".TSGF" in x.name], p
            output, trail = openable(p)
            assert output is not None, trail
            said = Path(output).read_text(errors="replace")
            assert ("Using GF-file" in said) == (f > 0), (f, v)
    _said(run, "status device", "tsgf/: empty")
    # BEFORE A LAUNCH, THE SIZE: from the leads' run outputs, read through
    # the registry as the line reads them, and the prepared decks.
    lead_out = run_of(stage_home(run.bundle, task, "electrode_L").dir
                      / "run-0").stdout
    n_orb = detect(str(lead_out)).parse(str(lead_out)).runtime_info["n_orbitals"]
    from molbuilder.calculation import open as open_calculation
    device_job = open_calculation(run.bundle).jobset.job("device")
    line = status_line(run.bundle, task, stage_home(run.bundle, task,
                                                    "device").dir, None,
                       device_job.scratch_dir)
    # A SIZE IN THE UNIT IT REACHES (`units.size_words`, the one spelling).
    import re
    assert line.startswith("tsgf/: one pair at a time, about ") and \
        f"({n_orb} orbitals × " in line and re.search(
            r"about [0-9.]+ (B|KiB|MiB|GiB) a lead at 0 V", line), line


def test_a_transmission_point_reads_the_device_at_its_frame_and_voltage(
        junction):
    """§ 2a.11: each transmission point gathers the device's Hamiltonian of
    the same frame and the same voltage, never another's."""
    from molbuilder.runrecord import read_gathered_from
    run = junction.from_pair
    for f in range(_N_FRAMES):
        for v in _BIAS:
            got = read_gathered_from(_frame_point(run, "05_transmission", f, v))
            hsx = [str(src) for src, fn in got if fn.endswith(".TS.HSX")]
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
    wrote at that frame's point, through its parser -- its energies at the
    full precision of the point's `.TBT.nc`, which the `.TBT.AVTRANS` prints
    to five decimals (§ 2a.12, *the data file*)."""
    from molbuilder.parse.engines.tbtrans_nc import point_averages
    from molbuilder.transport.transiesta import LEAD_NAMES, LEAD_PAIR
    curves = []
    for f in range(_N_FRAMES):
        point = _frame_point(run, "05_transmission", f, volts)
        got = point_averages(point, label, pair=LEAD_PAIR,
                             leads=tuple(LEAD_NAMES))["unpolarized"]
        energies = got["energy_ev"]
        curves.append(np.asarray(got["transmission"], dtype=float))
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
    from molbuilder.task import of as task_of
    run = junction.from_pair
    label = task_of(run.bundle).label
    avg = _record(run)["average"]
    fs = read_frame_set(_the_set(junction))
    assert avg["frames"] == _N_FRAMES
    for name, value in fs.structure_rows().items():
        assert avg[name] == value, (name, avg[name], value)
    assert avg["rows"] == list(fs.structure) and avg["warnings"] == []
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


def test_the_data_file_holds_every_point_beside_its_frames_definition(
        junction):
    """§ 2a.12, *the data file*: `<label>.transport.nc`, read through its
    door, holds every point on one grid -- frame x voltage x energy, the
    I-V's voltages the description's -- its T(E) TBtrans's own, each frame's
    definition rows and every atom's element, mass and coordinates beside it
    in the user order, the cited set's (`model/overview.md` § 2.2a, I1), the
    device's facts, the I-V and the mode's average each at its own point as
    the record states them; every dimension a coordinate of its own name and
    the labels of its points coordinates beside it; every variable says what
    it is, a raw one where it was read from; and a calculation of one
    structure has the same layout, one frame."""
    from molbuilder.frame_average import FRAME_ROWS, MASS_CHANNEL
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import (VARIABLES, data_file_path,
                                               read_data_file)
    run = junction.from_pair
    label = task_of(run.bundle).label
    got = read_data_file(data_file_path(run.bundle, label))
    V = {k: v["data"] for k, v in got["variables"].items()}
    assert list(V["frame"]) == list(range(_N_FRAMES))
    assert list(V["frame_token"]) == [f"f{f:03d}" for f in range(_N_FRAMES)]
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
    cited = _the_set(junction)
    for r in FRAME_ROWS:
        assert list(V[r.name]) == [cited.customized_value(r.name, frame=f)
                                   for f in range(_N_FRAMES)], r.name
    assert [json.loads(x) for x in V["frame_rows"]] == [
        cited.customized_rows(f) for f in range(_N_FRAMES)]
    masses = cited.annotations[MASS_CHANNEL].data
    assert list(V["atom"]) == list(range(cited.n_atoms))
    assert list(V["element"]) == list(cited.elements)
    assert list(V["mass_amu"]) == [masses[i] for i in range(cited.n_atoms)]
    assert np.array_equal(V["positions_ang"],
                          np.asarray(cited.frames, dtype=float))
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
    frame_labels = {"frame_token", "frame_rows",
                    *(r.name for r in FRAME_ROWS)}
    attrs_of = {k: v["attrs"] for k, v in got["variables"].items()}
    assert set(attrs_of["transmission"]["coordinates"].split()) \
        == frame_labels
    assert set(attrs_of["positions_ang"]["coordinates"].split()) \
        == frame_labels | {"element", MASS_CHANNEL, "user_atom_index",
                           "engine_atom_index"}
    for var in VARIABLES:
        attrs = attrs_of[var.name]
        assert attrs["long_name"] and attrs["definition"], var.name
        assert "units" in attrs and attrs["kind"] == var.kind, var.name
        if var.kind == "raw":
            assert attrs["source"], var.name
    one = read_data_file(data_file_path(junction.bundle, "T"))["variables"]
    assert one["transmission"]["data"].shape[:2] == (1, len(_BIAS))
    assert one["transmission"]["dims"] == ("frame", "bias_v", "energy_ev")


def test_the_data_file_holds_every_runs_convergence_on_its_grid(junction):
    """§ 2a.12, *the data file*: every run's convergence on the grid it
    varies over -- each device point's SCF history in its own (frame,
    voltage) cell on the `scf_cycle` axis, the cycles past its last the fill
    value; the once-run rungs on the `rung` axis; each point's state; the
    NEGF charges by TranSIESTA's own columns; the leads compared -- each as
    the record it is composed from states it."""
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import data_file_path, read_data_file
    run = junction.from_pair
    label = task_of(run.bundle).label
    got = read_data_file(data_file_path(run.bundle, label))
    V = {k: v["data"] for k, v in got["variables"].items()}
    rec = _record(run)
    by = {s["stage"]: s for s in rec["stages"]}
    for p in by["device"]["by_point"]:
        cell = (p["frame"], list(_BIAS).index(p["bias_v"]))
        rows = p["scf"]
        assert rows and V["device_scf_cycles"][cell] == len(rows), cell
        assert list(V["device_scf_ddmax"][cell][:len(rows)]) == [
            r["dDmax"] for r in rows], cell
        assert list(V["device_scf_phase"][cell][:len(rows)]) == [
            r["phase"] for r in rows], cell
        assert np.isnan(V["device_scf_ddmax"][cell][len(rows):]).all(), cell
        assert V["device_state"][cell] == p["state"], cell
        assert V["device_scf_converged"][cell] == int(p["scf_converged"])
        charges = p["negf"]["charges"]
        parts = list(V["charge_part"])
        assert {k: V["device_charge"][cell][parts.index(k)]
                for k in charges} == charges, cell
        # EVERY NEGF ROW'S CHARGES, and the run's phases (@4).
        for c, r in enumerate(rows):
            for k, q in (r.get("charges") or {}).items():
                assert V["device_scf_charge"][cell][c][parts.index(k)] == q
        assert json.loads(V["device_scf_phases"][cell]) == p["scf_phases"]
    assert list(V["rung"]) == ["seed", "electrode_L", "electrode_R"]
    for r, name in enumerate(V["rung"]):
        rows = by[name]["scf"]
        assert V["rung_scf_cycles"][r] == len(rows), name
        assert list(V["rung_scf_energy_ev"][r][:len(rows)]) == [
            c["energy"] for c in rows], name
        assert V["rung_fermi_ev"][r] == by[name]["fermi_ev"], name
        assert V["rung_state"][r] == by[name]["state"], name
    for p in rec["points"]:
        cell = (p["frame"], list(_BIAS).index(p["bias_v"]))
        assert V["point_state"][cell] == p["state"] == "finished", cell
    assert got["attrs"]["leads_ef_differ_ev"] == rec["leads"]["differ_ev"]
    assert got["attrs"]["Conventions"] == "CF-1.8"


def _export(junction, monkeypatch, run, fmt, to):
    """`molbuilder jobset export` of ``run`` into ``to``, the one door
    pointed at the junction's tree."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.road import jobset
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    r = jobset("export", "--bundle", run.bundle, "--format", fmt,
               "--to", to)
    assert r.exit_code == 0, r.output
    return r.output


def _listing(folder):
    return sorted(str(f.relative_to(folder)) for f in folder.rglob("*")
                  if ".git" not in f.parts and ".binsnapshots" not in f.parts)


def test_an_export_is_the_data_file_and_its_tables(junction, monkeypatch,
                                                   tmp_path):
    """§ 2a.12, *Export*: the command's NetCDF is the data file `summarize
    task` wrote, variable for variable; its CSV is the same values as long
    tables -- T(E) at each frame, voltage and energy -- with their README;
    the report's route hands the tables over as one zip, named; and nothing
    is written into the calculation's folder.  The silent failure: an export
    whose numbers are not the data file's, or which leaves files in the
    calculation."""
    import csv
    import io
    import zipfile
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import data_file_path, read_data_file
    from molbuilder.web.app import create_app
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.road import jobset
    run = junction.from_pair
    label = task_of(run.bundle).label
    before = _listing(run.bundle)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    inside = jobset("export", "--bundle", run.bundle, "--format", "csv",
                    "--to", run.bundle / "out")
    assert inside.exit_code != 0 and "inside the calculation" in (
        inside.output), inside.output
    _export(junction, monkeypatch, run, "netcdf", tmp_path)
    _export(junction, monkeypatch, run, "csv", tmp_path)
    assert _listing(run.bundle) == before, "nothing written in it"
    want = read_data_file(data_file_path(run.bundle, label))["variables"]
    got = read_data_file(tmp_path / f"{label}.transport.nc")["variables"]
    assert sorted(got) == sorted(want)
    for name, v in want.items():
        a, b = np.asarray(v["data"]), np.asarray(got[name]["data"])
        same = (np.allclose(a, b, equal_nan=True, rtol=0, atol=0)
                if a.dtype.kind == "f" else np.array_equal(a, b))
        assert same, name
    rows = list(csv.DictReader(io.StringIO(
        (tmp_path / f"{label}.transport-csv"
         / "frame_bias_v_energy_ev.csv").read_text())))
    T = np.asarray(want["transmission"]["data"])
    E = np.asarray(want["energy_ev"]["data"])
    for r in rows:
        f = int(r["frame"])
        b = list(_BIAS).index(float(r["bias_v"]))
        e = int(np.argmin(np.abs(E - float(r["energy_ev"]))))
        assert float(r["transmission"]) == T[f, b, e], r
    assert len(rows) == T.size
    readme = (tmp_path / f"{label}.transport-csv" / "README.txt").read_text()
    assert "frame_bias_v_energy_ev.csv" in readme and "transmission:" in readme
    answer = create_app(config={}).test_client().get(
        f"/api/transport/export?path={run.bundle / 'task.json'}&format=csv")
    assert answer.status_code == 200, answer.get_json()
    assert (f"{label}.transport-csv.zip"
            in answer.headers["Content-Disposition"])
    names = zipfile.ZipFile(io.BytesIO(answer.data)).namelist()
    assert f"{label}.transport-csv/README.txt" in names
    assert f"{label}.transport-csv/frame_bias_v_energy_ev.csv" in names


def test_an_export_reads_no_data_file_and_says_what_is_missing(
        junction, monkeypatch, tmp_path):
    """§ 2a.12, *Export* (the user, 2026-10-23: "the export should only
    contain information that is already processed"): a point whose DOS per
    orbital is not kept exports without its DOS by region -- its reading
    said `not read yet` in the file, the export's own line saying how many
    and how to read them -- and no data file is read: its digest is still
    not there after.  The silent failure: an export reading a cluster's
    gigabytes on a press of a button, or one whose empty regions say
    nothing of why."""
    import shutil
    from molbuilder.parse.engines.tbtrans_nc import data_files, digest_of
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import read_data_file
    run = junction
    label = task_of(run.bundle).label
    data = data_files(_point(run, "05_transmission", 0, _BIAS[0]),
                      label)["unpolarized"]
    kept = tmp_path / "kept.digest.nc"
    shutil.move(str(digest_of(data)), str(kept))
    try:
        out = _export(junction, monkeypatch, run, "netcdf", tmp_path)
        assert "1 finished point exports without its DOS by region" in out, out
        assert not digest_of(data).exists(), "the export read a data file"
        V = {k: v["data"] for k, v in read_data_file(
            tmp_path / f"{label}.transport.nc")["variables"].items()}
        b = list(_BIAS).index(_BIAS[0])
        assert "not read yet" in str(V["dos_per_orbital_reading"][0, b])
        assert np.isnan(np.asarray(V["dos_region"][0, b], float)).all()
    finally:
        shutil.move(str(kept), str(digest_of(data)))


def test_the_files_say_which_index_their_atoms_are_listed_by(junction):
    """`model/overview.md` § 2.2b: the data file is listed by user index,
    each atom's engine index the composed junction's order record's; a
    point's digest by engine index, each atom's user index the same
    record's; `molbuilder atom-index` says so of the point's folder and maps
    an index each way.  The silent failure: a script asking sisl about the
    wrong atoms, nothing in its file it could have checked."""
    from molbuilder.atom_index import atom_index_map
    from molbuilder.parse.engines.tbtrans_nc import data_files, digest_of
    from molbuilder.task import of as task_of
    from molbuilder.transport.compose import recorded_junction
    from molbuilder.transport.datafile import data_file_path
    from support.road import molbuilder
    run = junction
    label = task_of(run.bundle).label
    composed = recorded_junction(run.bundle).structure
    n, record = composed.n_atoms, composed.atom_order
    # THE RECORD'S OWN LISTS, as its sidecar states them -- the expectation,
    # never the map's code.
    engine_of_user = (list(record.original_to_sorted) if record is not None
                      else list(range(n)))
    user_of_engine = (list(record.sorted_to_original) if record is not None
                      else list(range(n)))
    data = atom_index_map(data_file_path(run.bundle, label))
    assert (data.listed_by, list(data.user), list(data.engine)) == (
        "user_index", list(range(n)), engine_of_user)
    point = _point(run, "05_transmission", 0, _BIAS[0])
    digest = digest_of(data_files(point, label)["unpolarized"])
    kept = atom_index_map(digest)
    assert (kept.listed_by, list(kept.user), list(kept.engine)) == (
        "engine_index", user_of_engine, list(range(n)))
    assert kept.stated_by == str(digest)
    out = molbuilder("atom-index", str(point), "--to-engine", "0").output
    assert f"user index 0 -> engine index {engine_of_user[0]}" in out, out
    out = molbuilder("atom-index", str(point), "--to-user",
                     str(engine_of_user[0])).output
    assert "-> user index 0" in out, out
    # A LEAD'S FILE GATHERED INTO THE DEVICE RUN is its electrode cell's own
    # (§ 2.2a, I6) -- refused, never answered with the junction's map.
    from molbuilder.identity import rung_label
    from molbuilder.atom_index import AtomIndexUnknown
    lead = (_point(run, "04_device", 0, _BIAS[0])
            / f"{rung_label(label, 'electrode_L')}.TSHS")
    with pytest.raises(AtomIndexUnknown, match="electrode_L rung's file"):
        atom_index_map(lead)
    # ...AND IN ITS OWN RUN it is that run's, a structure of its own: the
    # engine column alone.
    own = atom_index_map(run.bundle / "02_electrode_L" / "run-0"
                         / f"{rung_label(label, 'electrode_L')}.XV")
    assert (own.listed_by, own.user) == ("engine_index", None), own


def test_an_older_junction_is_refused_to_an_editor_and_marked_for_a_viewer(
        junction, tmp_path):
    """`model/overview.md` § 2.2a, I8, I9; § 2.2b, F4: the composed junction
    whose sidecar states no order record -- written before the field --
    is refused to an editor's read and a builder's, and opened for a viewer
    marked `order_unstated` and said; a save from that view states
    `not_stated`.  On a copy of the calculation's records and its junction,
    the sidecar as a molbuilder before v14 wrote it.  The silent failure:
    TranSIESTA's order opened and saved as the user's."""
    import json
    import shutil
    from molbuilder.runfiles import JUNCTION_FILE, JUNCTION_SIDECAR_FILE
    from molbuilder.workingcopy_structure import (ORDER_UNSTATED,
                                                  StructureCodec)
    run = junction.single
    calc = tmp_path / "P" / "transport" / run.bundle.name
    calc.mkdir(parents=True)
    for f in run.bundle.iterdir():
        if f.is_file():
            shutil.copy(f, calc / f.name)
    side = calc / JUNCTION_SIDECAR_FILE
    payload = json.loads(side.read_text())
    del payload["atom_order"], payload["atom_index_map"]
    side.write_text(json.dumps({**payload, "schema_version": 13}))
    with pytest.raises(ValueError, match="states no order record"):
        StructureCodec().load(calc / JUNCTION_FILE, frames=True,
                              in_user_order=True, said_out=[])
    said: list = []
    seen = StructureCodec().load(calc / JUNCTION_FILE, frames=True,
                                 said_out=said, opened=(ORDER_UNSTATED,))
    assert seen.order_unstated and [n.where for n in said] == [
        ORDER_UNSTATED], said
    out = tmp_path / "saved.xyz"
    StructureCodec().write(seen.frame_at(0), out)
    written = json.loads(out.with_suffix(".molstruct.json").read_text())
    assert written["atom_index_map"]["atoms_listed_by"] == "not_stated"


def test_a_scripts_parse_keeps_a_reading_that_states_both_indices(
        junction, tmp_path):
    """`model/parse.md` § 3, § 5e: `molbuilder.postprocess.parse` of a
    transmission point's `.TBT.nc` hands the parser the point's run's map,
    so the reading it keeps states each atom's user and engine index -- as
    the report's own reading does.  The silent failure: a person's script
    keeping a reading the report then calls out of date, reading the
    cluster's gigabytes again."""
    import shutil
    from molbuilder.atom_index import atom_index_map
    from molbuilder.parse.engines.tbtrans_nc import data_files, digest_of
    from molbuilder.postprocess import parse
    from molbuilder.task import of as task_of
    run = junction
    label = task_of(run.bundle).label
    data = data_files(_point(run, "05_transmission", 0, _BIAS[0]),
                      label)["unpolarized"]
    kept = tmp_path / "kept.digest.nc"
    shutil.move(str(digest_of(data)), str(kept))
    try:
        got = parse(data)
        want = atom_index_map(data.parent)
        assert (got.atom_index_map.user, got.atom_index_map.engine) == (
            want.user, want.engine), got.atom_index_map
        written = atom_index_map(digest_of(data))
        assert (written.user, written.engine) == (want.user, want.engine)
    finally:
        digest_of(data).unlink(missing_ok=True)
        shutil.move(str(kept), str(digest_of(data)))


def test_a_scripts_read_structure_gives_the_map_of_what_it_returns(junction):
    """`model/parse.md` § 3; `model/overview.md` § 2.2b:
    `postprocess.read_structure` of the calculation's junction -- the
    engine's record, its atoms in TranSIESTA's order -- returns it in the
    user order with the atom index map OF THAT STRUCTURE: listed by user
    index, row u's engine index the row of the engine's record that atom is.
    The silent failure: a script asks sisl about row u of the returned
    structure through the file's own map, listed by engine index -- another
    atom."""
    from molbuilder.postprocess import read_structure
    from molbuilder.workingcopy_structure import StructureCodec
    path = junction.bundle / _record(junction)["junction_file"]
    struct, atoms, _said = read_structure(path)
    record = StructureCodec().load(path)
    assert atoms.listed_by == "user_index"
    assert list(atoms.engine) != list(range(struct.n_atoms)), (
        "the junction's sort moved no atom: this row bounds nothing")
    for u in range(struct.n_atoms):
        assert np.allclose(struct.positions[u],
                           record.positions[atoms.engine[u]]), u


def test_the_data_file_holds_the_calculations_own_records(junction):
    """§ 2a.12, *the data file* (@4): the calculation's own records -- its
    description, template and job request, each the file's text -- the
    provenance chain, and each point's transmission k-points, as the
    record states them.  The silent failure: a result no one can rerun or
    compare, its settings left behind."""
    from molbuilder.job_request import path_of as request_path
    from molbuilder.task import of as task_of, path_of as task_path
    from molbuilder.template_file import template_path
    from molbuilder.transport.datafile import data_file_path, read_data_file
    run = junction
    label = task_of(run.bundle).label
    got = read_data_file(data_file_path(run.bundle, label))
    attrs = got["attrs"]
    assert attrs["description"] == task_path(run.bundle).read_text()
    assert attrs["template"] == template_path(run.bundle, label).read_text()
    assert attrs["job_request"] == request_path(run.bundle).read_text()
    rec = _record(run)
    assert json.loads(attrs["provenance_chain"]) == rec["provenance"]["chain"]
    V = {k: v["data"] for k, v in got["variables"].items()}
    for p in rec["points"]:
        b = list(_BIAS).index(p["bias_v"])
        assert V["tbt_k_points"][0, b] == p["k_points"], p
        assert V["tbt_k_method"][0, b] == p["k_method"], p


def test_a_ladder_without_a_transmission_exports_what_exists(
        junction, monkeypatch, tmp_path):
    """§ 2a.12, *Export* (the user, 2026-10-22: "export what exists"): a
    calculation whose transmission never ran exports its rungs' facts and
    every SCF history -- the seed's, the leads', the device's -- its energy
    axis empty, where `summarize task` refuses.  The silent failure: the
    convergence of a ladder still running out of reach until it ends --
    and why each value is what it is: each run's state in the run door's
    sentence, and at every voltage why there is no frames' average, as the
    record says them."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import read_data_file
    from molbuilder.transport.record import collect_record
    run = junction.single
    _export(junction, monkeypatch, run, "netcdf", tmp_path)
    V = {k: v["data"] for k, v in read_data_file(
        tmp_path / f"{task_of(run.bundle).label}.transport.nc")[
            "variables"].items()}
    assert len(V["energy_ev"]) == 0
    assert list(V["rung"]) == ["seed", "electrode_L", "electrode_R"]
    assert all(n > 0 for n in V["rung_scf_cycles"]), V["rung_scf_cycles"]
    assert (np.asarray(V["device_scf_cycles"]) > 0).all()
    assert not np.asarray(V["done"]).any()
    rec = collect_record(open_calculation(run.bundle), partial=True)
    stage = {s["stage"]: s for s in rec["stages"]}
    assert [V["rung_detail"][r] for r in range(3)] == [
        stage[n]["detail"] for n in V["rung"]]
    assert all(V["rung_detail"]), V["rung_detail"]
    device = stage["device"].get("by_point") or [stage["device"]]
    assert set(np.asarray(V["device_detail"]).ravel()) == {
        d["detail"] for d in device}
    assert list(V["average_why"]) == [rec["average"]["why"]] * len(
        V["bias_v"]), V["average_why"]


def test_the_tab_cites_a_frame_set_and_says_what_each_frame_is(junction):
    """§ 3.8.1: the Transport tab cites either kind of § 3.1 -- and for a
    pair the describe door says what its frames are: a mode's set its rows
    in the vocabulary's words, its masses, where the mode came from and its
    weights (the wire `frame_average.answer` builds from
    `frame_average.ROWS`); a family its count and no rows; a run one structure.  The shared panel
    asks the pseudopotential directory of the person exactly for a pair,
    which brings none; and Task setup's folder answer echoes the frame-set
    calculation's frames as the describe door states them."""
    from molbuilder.frame_average import MASS_CHANNEL
    from molbuilder.workingcopy_structure import StructureCodec
    # WHAT THE CARD SAYS, from § 2.2f's table and the rows the script wrote
    # -- the pair's own, read back -- never from the definition's reader.
    pair = StructureCodec().load(junction.tree / junction.frame_set,
                                 frames=True)

    def stated(name, frame=None):
        return next(r["value"] for r in pair.customized_rows(frame)
                    if r["name"] == name)
    expect = {
        "structure": [{"name": n, "value": stated(n), "unit": u, "kind": k,
                       "words": w, "page_unit": pu}
                      for n, k, u, w, pu in _SET_ROWS],
        "columns": [{"name": n, "unit": u, "kind": k, "words": w,
                     "page_unit": pu}
                    for n, k, u, w, pu in _FRAME_ROWS],
        "frames": [[stated(row[0], f) for row in _FRAME_ROWS]
                   for f in range(_N_FRAMES)],
        "masses_amu": [float(pair.annotations[MASS_CHANNEL].data[i])
                       for i in range(pair.n_atoms)],
        "vibration": pair.info["vibration"],
        "weights": list(_WEIGHTS),
        "weight_sum": pytest.approx(1.0, abs=1e-12),
        "warnings": [],
        # THE FRAMES STATE A MODE, said by the door so the card keeps no
        # list of the rows that decide it (`transport.md` § 3.8.1).
        "mode": True,
    }
    # ...and those rows the script's own: its weights, its nodes, its
    # temperature.
    assert [row[-1] for row in expect["frames"]] == list(_WEIGHTS)
    assert [row[3] for row in expect["frames"]] == pytest.approx(
        list(_NODES_SIGMA), rel=1e-12)
    assert stated("temperature_k") == _TEMPERATURE_K
    fields = {}
    for kind, (described, schema) in junction.tab_said.items():
        assert described["ok"] and described["citation"], (kind, described)
        assert schema["ok"], (kind, schema)
        fields[kind] = {f["name"] for sec in schema["schema"]["sections"]
                        for f in sec["fields"]}
    assert junction.tab_said["mode's set"][0]["frame_set"] == {
        "n_frames": _N_FRAMES, "definition": expect}
    assert junction.tab_said["family"][0]["frame_set"] == {
        "n_frames": _N_FRAMES, "definition": None}
    assert junction.tab_said["run"][0]["frame_set"] == {
        "n_frames": 1, "definition": None}
    assert "psml_lib" in fields["mode's set"] and "psml_lib" in fields["family"]
    assert "psml_lib" not in fields["run"]
    # TASK SETUP ECHOES THE SAME FRAMES, through the same helper
    # (`web/task-setup.md` § 3).
    assert junction.folder_said["frame_set"] \
        == junction.tab_said["mode's set"][0]["frame_set"]


#: Two roundings stand between a point's total DOS as TBtrans printed it
#: (`.TBT.AVDOS`, eight digits: at most 5e-8 relative) and its digest's
#: orbitals summed (TBtrans's single precision, which the digest keeps:
#: 6e-8) -- `model/parse.md` § 5e.
_PRINTED_AND_SINGLE_RTOL = 2e-7


@pytest.mark.parametrize("ask", _PDOS_ASKS, ids=lambda a: a["name"])
def test_a_selections_pdos_is_asked_at_its_frame_and_voltage(junction, ask):
    """`web/results.md` § 2.5: the PDOS of a selection is asked at the
    selected frame and voltage -- the per-orbital DOS summed over the atoms'
    orbitals, so every atom of the junction (the device region's counted,
    the rest named outside it) is that point's total DOS, and the bridge's
    atoms its bridge's, as the record states them, and no other point's; the
    atoms named in the user order, asked and answered (`model/overview.md`
    § 2.2a, I1); a frame set's point names its frame, one structure's names
    none, and either asked otherwise is refused by name (`web/web-api.md`,
    `/api/transport/pdos`)."""
    status, body = junction.pdos_said[ask["name"]]
    if "refused" in ask:
        assert status == 400 and ask["refused"] in body["error"], body
        return
    assert status == 200 and body["ok"], body
    asked = sorted(ask.get("atoms", range(len(_LISTED))))
    assert sorted(body["atoms"] + body["outside_device"]) == asked, body

    def curve(p):
        dos = p["dos"]
        return dos["by_label"][ask["region"]] if "region" in ask \
            else dos["total"]
    points = _record(junction.from_pair)["points"]
    point = next(p for p in points if (p["frame"], p["bias_v"]) == ask["at"])
    assert body["energy_ev"] == point["dos"]["energy_ev"]
    # A REGION is the same digest's orbitals summed; the TOTAL is TBtrans's
    # printed one (`model/parse.md` § 5e).
    rtol = 1e-9 if "region" in ask else _PRINTED_AND_SINGLE_RTOL
    assert np.allclose(body["pdos"], curve(point), rtol=rtol,
                       atol=0), ask["name"]
    others = [p for p in points if (p["frame"], p["bias_v"]) != ask["at"]]
    assert others and not any(
        np.allclose(body["pdos"], curve(p), rtol=1e-6, atol=0)
        for p in others), "the points' DOS do not tell them apart"


def test_summarize_keeps_each_points_reading_and_its_digest_reads_alone(
        junction):
    """`model/parse.md` § 5e, `engines/transport.md` § 2a.12: `summarize
    task` reads each transmission point's data file, saying so, keeps its
    reading beside it and the record states it kept; the digest describes
    itself, so a person's script reads it with netCDF4 alone -- its schema
    and the data file it read; the device's DOS per orbital, whose columns
    summed are the total DOS TBtrans printed, each column's atom one of the
    device's; each atom's element and position; TBtrans's own T(E), on the
    record's energies.  The silent failure: a digest a script cannot use
    alone -- its elements gone when the Hamiltonian's geometry is not found,
    its orbitals summed not the total TBtrans printed."""
    import netCDF4
    from molbuilder.parse.engines.tbtrans_nc import (DIGEST_SCHEMA,
                                                     data_files, digest_of,
                                                     kept_state)
    from molbuilder.task import of as task_of
    _said(junction, "summarize", "its DOS per orbital, kept beside it")
    label = task_of(junction.bundle).label
    for p in _record(junction)["points"]:
        folder = _point(junction, "05_transmission", 0, p["bias_v"])
        data = data_files(folder, label)["unpolarized"]
        assert kept_state(data).state == "kept"
        assert [d["state"] for d in p["digest"]] == ["kept"], p["digest"]
        with netCDF4.Dataset(str(digest_of(data))) as ds:
            assert ds.schema == DIGEST_SCHEMA
            assert ds.data_file == data.name
            assert int(ds.data_file_size) == data.stat().st_size
            assert list(ds["energy"][:]) == p["dos"]["energy_ev"]
            dos = np.asarray(ds["DOS"][:], dtype=float)
            assert np.allclose(dos.sum(axis=1), p["dos"]["total"],
                               rtol=_PRINTED_AND_SINGLE_RTOL, atol=0)
            assert set(ds["column_atom"][:]) <= set(ds["device_atom"][:])
            assert list(ds["element"][:]) == ["H"] * len(_LISTED)
            assert ds["position"].shape == (len(_LISTED), 3)
            assert list(ds["transmission"][:]) == p["transmission"]


def _held(data, parse):
    """A reading asked again while another reading of this process -- two
    requests of one server -- holds the file."""
    from molbuilder.parse.engines.tbtrans_nc import ReadingClaimed, kept_state
    seen: dict = {}

    def at(_k, _n):
        if not seen:
            seen["state"] = kept_state(data)
            try:
                parse(data, fresh=True)
            except ReadingClaimed as exc:
                seen["refused"] = str(exc)
    parse(data, fresh=True, progress=at)
    assert seen["state"].state == "being read"
    assert seen["state"].claim["alive"] is True
    assert f"process {os.getpid()}" in seen["refused"], seen


def _gone(data, parse):
    """A reading whose process ended in its first k-point: its claim stays,
    said gone, and the next reading takes it."""
    import subprocess
    import sys
    from molbuilder.parse.engines.tbtrans_nc import kept_state
    code = ("import os, sys\n"
            "from molbuilder.parse.engines.tbtrans_nc import "
            "TBTransNcFileParser as P\n"
            "P.parse(sys.argv[1], fresh=True,"
            " progress=lambda k, n: os._exit(9))\n")
    r = subprocess.run([sys.executable, "-c", code, str(data)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 9, r.stderr
    left = kept_state(data)
    assert left.state == "being read" and left.claim["alive"] is False
    assert "that process is gone" in left.to_wire()["said"]
    parse(data, fresh=True)


def _gone_beside_a_kept_reading(data, parse):
    """A reading again whose process ended in its first k-point, beside the
    reading kept before it: the kept one is still the file's, and the next
    asker answers it, removing the claim."""
    from molbuilder.parse.engines.tbtrans_nc import kept_state
    parse(data)
    assert kept_state(data).state == "kept"
    _gone(data, lambda _data, fresh: None)
    assert parse(data).kept


def _taken_over(data, parse):
    """A reading taken over by another: the first stops at its next k-point,
    the digest is the taker's."""
    from molbuilder.parse.engines.tbtrans_nc import ReadingClaimed
    seen: dict = {}

    def at(_k, _n):
        if not seen:
            seen["taker"] = parse(data, fresh=True, take_over=True)
    with pytest.raises(ReadingClaimed):
        parse(data, fresh=True, progress=at)
    assert not seen["taker"].kept


#: `model/parse.md` § 5e, *one reading of a data file at a time*: each case
#: ends with the file's reading kept and no claim or part left beside it.
_CLAIMS = {"held by a reading under way": _held,
           "left by a reading whose process is gone": _gone,
           "left beside a kept reading": _gone_beside_a_kept_reading,
           "taken over": _taken_over}


@pytest.mark.parametrize("case", list(_CLAIMS))
def test_one_reading_of_a_data_file_at_a_time(junction, tmp_path, case):
    """`model/parse.md` § 5e: a reading claims its data file; another asker
    is refused by name, saying whose reading it is; a claim whose process is
    gone on this host is taken by the next reading; a claim taken over stops
    its reading at its next k-point.  Each on a copy of a real point.  The
    silent failure: two readings writing one digest, or a point left
    unreadable by a reading that died."""
    import shutil
    from molbuilder.parse.engines.tbtrans_nc import (TBTransNcFileParser,
                                                     data_files, digest_of,
                                                     kept_state)
    from molbuilder.task import of as task_of
    label = task_of(junction.bundle).label
    point = tmp_path / "point"
    shutil.copytree(_point(junction, "05_transmission", 0, _BIAS[0]), point)
    data = data_files(point, label)["unpolarized"]
    _CLAIMS[case](data, TBTransNcFileParser.parse)
    assert kept_state(data).state == "kept"
    digest = digest_of(data)
    beside = sorted(f.name for f in point.iterdir())
    assert [n for n in beside if n.startswith(digest.name)] \
        == [digest.name], beside


def test_the_report_draws_the_junction_in_the_user_order(junction, page,
                                                     monkeypatch):
    """`web/results.md` § 2.5, § 3; `model/overview.md` § 2.2a, I1: the
    transport report draws the composed junction -- the engine's record, in
    TranSIESTA's order -- put back in the user order, the order its PDOS
    selections are named in, and says so: the viewer's notice names the
    order the file was in, and no line says its atoms are in the engine's
    order.  Mounted through the registry with the page's default context, on
    the server's answer for the calculation's `task.json`
    (`/api/results/dir`), as the Results page's dispatcher mounts it.  The
    silent
    failure: a drawing in TranSIESTA's order whose selections the PDOS door
    reads as yours."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.live_server import serve
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    task = junction.from_pair.bundle / "task.json"
    with serve() as base:
        page.goto(f"{base}/results", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => window.molbuilder && window.molbuilder.inspectors"
            "      && window.molbuilder.inspectors.pick")
        name = page.evaluate(
            """async (file) => {
                const dir = file.slice(0, file.lastIndexOf('/'));
                const answer = await (await fetch(
                    '/api/results/dir?path=' + encodeURIComponent(dir))).json();
                const hit = answer.files.find((f) => f.name === 'task.json');
                const meta = {role: hit.role, label: hit.label,
                              stage: hit.stage, parser: hit.parser,
                              engine: answer.engine};
                const reg = window.molbuilder.inspectors;
                const host = document.getElementById('inspector-host');
                host.innerHTML = '';
                reg.mount(host, file, reg.createDefaultContext(host), meta);
                return reg.pick(file, meta).name;
            }""", str(task))
        assert name == "transport", name
        page.wait_for_function(
            "() => [...document.querySelectorAll('.molviewer-notices')]"
            "      .some((n) => n.textContent.includes('put back in the user order'))",
            timeout=30000)
        notices = " ".join(page.locator(".molviewer-notices").all_inner_texts())
        assert "TranSIESTA's order" in notices, notices
        assert page.locator(".molviewer-order").is_hidden()


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


# --------------------------------------------------------------------- #
#  A transmission in the Results tab, and how far it is (`web/results.md` #
#  § 2, § 2.5, § 3a; `execution/run-reports.md` § 2.2)                    #
# --------------------------------------------------------------------- #

#: Every transmission point: ``(calculation, frame, voltage)`` -- the sweep's
#: voltages, and the frame set's frames at each.
_TRANSMISSION_POINTS = ([("T", None, v) for v in _BIAS]
                        + [("F", f, v) for f in range(_N_FRAMES)
                           for v in _BIAS])

#: TBtrans's own lines for its loop are found through the family's one
#: table (`siesta_grammar` as `_G`, `model/parse.md` § 5d.5): where each falls
#: in the run's output and what it states.  The expected words are the
#: contract's.
#: A pass's words, as the contract names a pass (`model/parse.md` § 5d.5).
_PASS_WORDS = {"up": "spin up", "down": "spin down",
               "unpolarized": "unpolarized"}


def _progress_stated(line: str):
    """``(per cent done, seconds left)`` a TBtrans progress line states, or
    ``None`` for any other line."""
    m = _G.TBT_PROGRESS.match(line)
    if m:
        return _G.fortran_float(m.group(1)), _G.fortran_float(m.group(2))
    m = _G.TBT_INITIAL_ETA.match(line)
    return (0.0, _G.fortran_float(m.group(1))) if m else None


def _completed_in(line: str):
    """The seconds a pass's ``tbt: Completed in`` states, or ``None``."""
    m = _G.TBT_COMPLETED.match(line)
    return _G.fortran_float(m.group(1)) if m else None


def _passes_planned(lines):
    """The passes a TBtrans output says it will make, in the contract's
    words."""
    return [_PASS_WORDS[ch] for ch in next(filter(None, map(
        _G.tbt_channels, lines)))]


def _span(seconds: float) -> str:
    """A length of time under 90 s, as every page says one -- whole seconds
    (`web/ui-contract.md` § 4); this junction's TBtrans takes a fraction of
    one."""
    assert seconds < 90, seconds
    return f"{int(seconds + 0.5)}s"


def _transmission_point(junction, row):
    """``(calculation, folder, point)`` of a `_TRANSMISSION_POINTS` row --
    the point as its wire form names it, ``{frame, of, bias_v}``."""
    name, f, v = row
    if name == "T":
        return (junction.bundle, _point(junction, "05_transmission", 0, v),
                {"frame": None, "of": None, "bias_v": v})
    return (junction.from_pair.bundle,
            _frame_point(junction.from_pair, "05_transmission", f, v),
            {"frame": f, "of": _N_FRAMES, "bias_v": v})


def _results_dir(junction, monkeypatch, folder):
    """The Results tab's answer for ``folder`` (`/api/results/dir`), the one
    door pointed at the junction's tree."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.web.app import create_app
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    got = create_app(config={}).test_client().get(
        f"/api/results/dir?path={folder}")
    assert got.status_code == 200, got.get_json()
    return got.get_json()


@pytest.mark.parametrize("row", _TRANSMISSION_POINTS,
                         ids=lambda r: f"{r[0]}-f{r[1]}-v{r[2]:g}")
def test_a_transmission_runs_folder_opens_its_report_at_its_point(
        junction, monkeypatch, row):
    """TBtrans writes no file a viewer reads on its own, so a transmission
    run's folder opens its calculation's report -- the description at the
    root, `task.json`, the registry's `transport-task` -- at the folder's
    point, its row sent whole for the menu to list first (`web/results.md`
    § 2).  The silent failure, measured on a real run 2026-10-16: the folder
    opened the progress log the wrapper writes at launch -- one preview
    frame, never updated by TBtrans -- and a transmission running for days
    showed a structure and no plot.  The folder and the one prep rendered
    the point's deck into say where they are (`project-layout.md` § 1.4a,
    invariant 6b): a folder prep made with no `calcdir.json` is read as *not
    marked*."""
    calc, folder, point = _transmission_point(junction, row)
    said = _results_dir(junction, monkeypatch, folder)
    assert Path(said["openable"]).resolve() == (calc / "task.json").resolve(), (
        said["openable"], said["attempts"])
    assert said["report"]["path"] == said["openable"], said["report"]
    assert said["report"]["parser"] == "transport-task", said["report"]
    assert said["point"] == point, said["point"]
    # WHERE THE FOLDER IS (`project-layout.md` § 1.4a, *Where am I*): a point
    # of the stage's run -- and its twin below the stage, the folder prep
    # rendered the point's deck into, a container that says so (invariant 6b).
    rel = folder.relative_to(calc / "05_transmission" / "run-0").as_posix()
    where = ("what", "role", "stage", "run", "trial", "point")
    assert {k: said["place"][k] for k in where} == {
        "what": "point", "role": "run", "stage": "05_transmission", "run": 0,
        "trial": None, "point": rel}, said["place"]
    prepared = _results_dir(junction, monkeypatch,
                            calc / "05_transmission" / rel)["place"]
    assert {k: prepared[k] for k in where} == {
        "what": "point", "role": "container", "stage": "05_transmission",
        "run": None, "trial": None, "point": rel}, prepared


@pytest.mark.parametrize("row", _TRANSMISSION_POINTS,
                         ids=lambda r: f"{r[0]}-f{r[1]}-v{r[2]:g}")
def test_a_transmissions_records_say_its_loop_as_tbtrans_measured_it(
        junction, monkeypatch, row):
    """A finished transmission's run record says its loop as TBtrans measured
    it -- each pass named, with the time its ``Completed in`` states -- and
    no estimate: the estimate ends with its pass (`model/parse.md` § 5d.1b,
    § 5d.5; `web/results.md` § 3a).  The calculation's record keeps its
    passes' loop time on the point, summed -- the time a point still to come
    is judged by (`engines/transport.md` § 2a.12).  The silent failures, measured on the
    junction's own runs 2026-10-16: a finished run's record still said it
    was 98 % done with 0 s left, and the points' loop times were read and
    dropped."""
    from support.replay import output_lines
    _calc, folder, point = _transmission_point(junction, row)
    lines = output_lines(folder)
    names = _passes_planned(lines)
    took = [s for s in map(_completed_in, lines) if s is not None]
    assert names and len(took) == len(names), (names, took)
    time = _results_dir(junction, monkeypatch,
                        folder)["record"]["computation"]["time"]
    assert time["progress"] == "completed: " + ", ".join(
        f"{w} ({i} of {len(names)}) in {_span(s)}"
        for i, (w, s) in enumerate(zip(names, took), start=1)), time
    assert "estimated_at" not in time, time
    calc = junction if row[0] == "T" else junction.from_pair
    mine = next(p for p in _record(calc)["points"]
                if (p["frame"], p["bias_v"]) == (point["frame"],
                                                 point["bias_v"]))
    assert mine["loop_elapsed_s"] == pytest.approx(sum(took)), mine


def test_a_transmission_stopped_in_its_loop_says_where_it_stopped(
        junction, monkeypatch):
    """A transmission stopped part way through its loop -- left in the
    background and stopped with the line its launch printed -- is failed,
    stopped before its end (`running-a-job.md` § 4.2), and says where its
    loop stopped, in the one sentence: its run record, its monitor's last
    [STATUS] line and the calculation's record of the point
    (`model/parse.md` § 5d.5; `run-reports.md` § 2.2;
    `engines/transport.md` § 2a.12).  The silent failures: a stopped run
    still quoting its last estimate as if it went on, or said as
    finished."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.report_fields import pct_words
    from molbuilder.runs import run_of
    from molbuilder.web.app import create_app
    from support.replay import output_lines, status_lines
    stopped = junction.stopped
    folder = stopped.attempt("transmission")
    lines = output_lines(folder)
    assert not any(_completed_in(ln) is not None for ln in lines), (
        "TBtrans finished its loop before it was stopped")
    names = _passes_planned(lines)
    pct = [p for p, _s in filter(None, map(_progress_stated, lines))][-1]
    said = (f"stopped during {names[0]} (1 of {len(names)}) at "
            f"{pct_words(pct)} %")
    time = _results_dir(junction, monkeypatch,
                        folder)["record"]["computation"]["time"]
    assert time["progress"] == said, time
    assert "estimated_at" not in time, time
    last = status_lines(run_of(folder).file(".monitor.log"))[-1]
    assert "failed" in last and said in last, last
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    record = create_app(config={}).test_client().get(
        f"/api/transport/record?path={stopped.bundle}").get_json()["record"]
    point = next(p for p in record["failed"]
                 if p["attempt"] == str(folder.relative_to(stopped.bundle)))
    assert point["state"] == "failed", point
    assert "stopped before its end" in point["why"], point
    assert point["progress"] == {"said": said}, point


def test_a_job_left_running_is_listed_stopped_by_its_line_and_said_once(
        junction):
    """`job-system.md` § 6.0a: a run left in the background is listed by
    `molbuilder jobs status` while it goes -- its id, its process going, its
    member's run in the status door's words; the line its launch printed,
    `molbuilder jobs stop <id>`, typed at a terminal and answered yes, stops
    it, and the calculation's ledger says so; the next status says how it
    ended, once, and the one after lists it no more.  The silent failures:
    a run going for days that nothing on the machine lists, a stop that
    leaves the engine running, a job said running after its process went,
    and a list that grows with every job that ever ran."""
    from molbuilder.jobset.ledger import LEDGER_FILE
    from support.road import job_lines
    stopped = junction.stopped
    job = next(ln.split("molbuilder jobs stop", 1)[1].strip()
               for ln in stopped.said["launch transmission"].output
               .splitlines() if "molbuilder jobs stop" in ln)
    host, pid = job.rsplit("-", 1)
    left = f"{job}  run here on {host}, in the background, pid {pid} -- "
    run = "05_transmission/run-0"
    going = stopped.jobs["while it runs"]
    assert going.exit_code == 0, going.output
    mine = job_lines(going.output, job)
    assert mine and mine[0] == left + "its process is going", going.output
    assert any(ln.startswith(f"    transmission  {run}  running")
               for ln in mine), mine
    said = stopped.said["stop as its launch said"]
    assert "stop it? [y/N]" in said.output, said.output
    assert "stopped: SIGTERM to its process group" in said.output, said.output
    after = stopped.jobs["after its stop"]
    mine = job_lines(after.output, job)
    assert mine and mine[0] == left + "its process has ended", after.output
    assert any(ln.startswith(f"    transmission  {run}  failed -- stopped "
                             f"before its end") for ln in mine), mine
    assert mine[-1] == "  ended -- said once; its address is removed", mine
    assert not job_lines(stopped.jobs["again"].output, job), \
        stopped.jobs["again"].output
    lines = [json.loads(x) for x in (stopped.bundle / LEDGER_FILE)
             .read_text().splitlines() if x.strip()]
    stop = [e for e in lines if e["verb"] == "jobs stop" and e["job"] == job]
    assert [(e["decision"], e.get("answer")) for e in stop] == [
        ("question", "yes"), ("stopped", None)], stop


def test_the_monitor_says_a_transmissions_loop_as_it_grows(junction,
                                                         tmp_path):
    """A transmission states no SCF row and no step: where it is, is
    TBtrans's account of its loop.  The monitor, shown the run's output as
    it grew -- from before its pass began, then a wake at each row's line --
    says it on its start line and on a [STATUS] line at each advance, the
    line every channel is sent, with the pass underway's share done and time
    left in the report's fields; and at the end, its process gone, how it
    ended beside it (`execution/run-reports.md` § 2.2, § 4.1a;
    `model/parse.md` § 5d.5).  The silent failures: a run going on for days
    whose monitor log says nothing after it began, a pass begun whose first
    estimate is hours away said as nothing, and one that ended still
    quoting the last estimate of its loop."""
    from molbuilder import monitor
    from support.replay import output_lines, replay, statuses
    folder = _point(junction, "05_transmission", 0, _BIAS[0])
    lines = output_lines(folder)
    s = lines
    names = _passes_planned(s)
    data = next(i for i, ln in enumerate(s) if _G.TBT_DATA_FILE.match(ln))
    at = [i for i, ln in enumerate(s) if _progress_stated(ln)]
    done = next(i for i, ln in enumerate(s) if _completed_in(ln) is not None)
    k = int(next(filter(None, map(_G.TBT_KPOINTS.match, s))).group(1))
    # A TBtrans output holds its own contours alone.
    e = sum(int(m.group(1)) for m in map(_G.TBT_CONTOUR_POINTS.match, s) if m)
    # THE JUNCTION IS RESTRICTED (its setup): one pass.
    assert names == ["unpolarized"] and len(at) >= 2, (names, at)
    one = f"{names[0]} (1 of 1)"

    def estimate(i):
        pct, left = _progress_stated(s[i])
        return (f"{one}: {round(pct, 2):g} % done, about {_span(left)} left "
                f"(TBtrans's estimate)")

    completed = f"completed: {one} in {_span(_completed_in(s[done]))}"
    # THE ROWS, a wake each: the line the output reaches, what the summary
    # line says there, and the report's (done_pct, eta_s) -- the pass
    # underway's, none once it completed.
    rows = [(data + 1, f"{one}: begun, no estimate yet", (None, None)),
            (at[0] + 1, estimate(at[0]), _progress_stated(s[at[0]])),
            (at[1] + 1, estimate(at[1]), _progress_stated(s[at[1]])),
            (done + 1, completed, (None, None))]
    _there, watched, grow = replay(folder, tmp_path, data)
    for n, (to, words, fields) in enumerate(rows):
        st = monitor.run_monitor(watched, interval=1, watch_pid=0,
                                 max_ticks=1, sleep=lambda _, to=to: grow(to))
        if n == 0:
            start = next(ln for ln in watched.path(".monitor.log")
                         .read_text().splitlines() if "[MONITOR] start" in ln)
            assert (f"setting up its loop: {k} k-point"
                    f"{'' if k == 1 else 's'} × {e} energies, 1 pass "
                    f"({names[0]})") in start, start
        said = statuses(watched)
        assert len(said) == n + 1 and words in said[-1], said
        assert (st.done_pct, st.eta_s) == fields, (st.done_pct, st.eta_s)
    # THE END: the whole output, its process gone -- how it ended, as the
    # Results tab reads its files, beside its loop.
    grow(len(lines))
    monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                        max_ticks=1, sleep=lambda _: None)
    end = statuses(watched)[-1]
    assert end.split("] [STATUS] ", 1)[1].startswith("finished"), end
    assert completed in end, end


def test_a_device_says_where_it_is_before_its_first_negf_row(junction,
                                                             tmp_path):
    """`model/parse.md` § 2b (`doing`), `running-a-job.md` § 4.2,
    `run-reports.md` § 2.2: a device spends its first stretch computing the
    leads' surface Green's functions and prints no SCF row there.  Shown
    the run's output as it grew, the monitor's status line and the run
    door's *running* detail say where it is in the same words -- which
    lead's function it is computing and the file's estimated size, the
    functions computed -- then the rows speak: a first point's periodic
    SCF, then the NEGF row's iteration.  The silent failure: Sol's device
    job 66108241 (2026-10-19), hours in that stretch with its row saying
    only *running*."""
    import re
    from molbuilder import monitor
    from molbuilder.parse.dirs.rundir import run_state_of
    from molbuilder.parse.engines.transiesta import scf_words
    from molbuilder.runfiles import RunNames
    from molbuilder.task import of as task_of
    from support.replay import output_lines, replay, statuses
    folder = _point(junction, "04_device", 0, _BIAS[0])
    s = output_lines(folder)
    begun = [i for i, ln in enumerate(s) if _G.TS_GF_FOR.match(ln)]
    sized = [i for i, ln in enumerate(s) if _G.TS_GF_SIZE.match(ln)]
    done = [i for i, ln in enumerate(s) if _G.TS_GF_DONE.match(ln)]
    charge = next(i for i, ln in enumerate(s) if _G.TS_CHARGE_START.match(ln))
    row = next(i for i, ln in enumerate(s)
               if re.match(_G.SCF_ROW_ERE, ln) and "ts-" in ln.lower())
    assert len(begun) == 2 and len(sized) == 2 and len(done) == 2, (
        begun, sized, done)
    assert begun[0] < sized[0] < done[0] < begun[1] < done[1] < charge < row
    lead = _G.TS_GF_FOR.match(s[begun[1]]).group(1)
    m = _G.TS_GF_SIZE.match(s[sized[1]])
    size = f"{m.group(1)} {m.group(2)}"
    # A FIRST POINT'S PERIODIC ROWS come between the leads' files and the
    # charge line (`transport.md` § 2a.12): at the charge line the last
    # periodic row speaks.
    periodic = [i for i, ln in enumerate(s)
                if re.match(_G.SCF_ROW_ERE, ln) and "ts-" not in ln.lower()
                and i < charge]
    assert periodic and done[1] < periodic[0] < charge, (done, periodic, charge)
    k = int(s[periodic[-1]].split()[1])
    rows = [
        (sized[1] + 1, f"computing the {lead} lead's surface Green's "
                       f"function (estimated {size})"),
        (done[1] + 1, "the leads' surface Green's functions computed"),
        (charge + 1, scf_words(_G.PHASE_PERIODIC, k)),
        (row + 1, scf_words(_G.PHASE_NEGF, 1)),
    ]
    assert rows[-1][1] == "negf SCF iteration 1"
    label = task_of(junction.bundle).label
    names = RunNames.of(label, "04_device", "hierarchical")
    there, watched, grow = replay(folder, tmp_path, begun[0])
    for n, (to, words) in enumerate(rows):
        monitor.run_monitor(watched, interval=1, watch_pid=0, max_ticks=1,
                            sleep=lambda _, to=to: grow(to))
        said = statuses(watched)
        assert said and words in said[-1], (words, said)
        # THE RUN DOOR'S ONE SCAN, on the same folder, carries the same
        # words (`RunEnding.doing`, what its *running* detail says) -- the
        # replayed folder keeps the finished run's conclusion, so the state
        # read here is the finished run's; the words are the output's.
        st = run_state_of(there, names)
        assert st.endings[st.active_source].doing == words, (words, st)


def test_a_devices_own_contours_are_no_transmission_loop(junction, tmp_path):
    """A biased device prints its own contours in the words TBtrans prints
    its energies in -- ``ts:   line contour points = N`` -- and they are
    TranSIESTA's, never read as a transmission's loop (`model/parse.md`
    § 5d.5: only the ``>> TBtrans contour <<`` section's are TBtrans's).  The
    monitor, shown the junction's biased device point's output whole, says
    no loop on its start line.  The silent failure: every biased device's
    status line and messages saying it is setting up a transmission's
    loop."""
    from molbuilder import monitor
    from support.replay import output_lines, replay
    folder = _point(junction, "04_device", 0, _BIAS[-1])
    lines = output_lines(folder)
    assert any(_G.TBT_CONTOUR_POINTS.match(ln) for ln in lines), (
        f"{folder}: the device states no contour points -- the row asks "
        f"nothing")
    _there, watched, _grow = replay(folder, tmp_path, len(lines))
    monitor.run_monitor(watched, interval=1, watch_pid=0, max_ticks=1,
                        sleep=lambda _: None)
    start = next(ln for ln in watched.path(".monitor.log").read_text()
                 .splitlines() if "[MONITOR] start" in ln)
    assert "its loop" not in start and "(1 of" not in start, start


#: What every surface says in a TranSIESTA device's force's place
#: (`model/parse.md` § 5d.6).
_NO_FORCES = ("TranSIESTA does not compute forces in this run (it says they "
              "are wrong), and so not the stress either")


def test_a_device_says_it_computes_no_forces_where_the_force_would_be(
        junction, monkeypatch, tmp_path):
    """TranSIESTA computes no forces on molbuilder's device -- its
    continued-fraction equilibrium contour switches them off whatever the
    deck asks -- and says so, *ALL FORCES AFTER TRANSIESTA HAS RUN ARE
    WRONG*; the stress is built from what it then zeroes.  So the device's
    run record and its monitor's closing line say it where the force would
    be, with no force and no stress (`model/parse.md` § 5d.6;
    `execution/run-reports.md` § 2.7).  The silent failure, the junction's
    own device 2026-10-16: *max force 0.0896 eV/Ang*, a number TranSIESTA
    says is wrong."""
    from molbuilder import monitor
    from support.replay import output_lines, replay, statuses
    folder = _point(junction, "04_device", 0, _BIAS[0])
    verdict = _results_dir(junction, monkeypatch, folder)["record"]["verdict"]
    assert verdict.get("forces_not_computed") == _NO_FORCES, verdict
    assert not {"max_force_free_ev_ang", "max_force_held_ev_ang",
                "stress_voigt_kbar"} & set(verdict), verdict
    _there, watched, _grow = replay(folder, tmp_path,
                                    len(output_lines(folder)))
    st = monitor.run_monitor(watched, interval=1, watch_pid=999_999_999,
                             max_ticks=1, sleep=lambda _: None)
    closing = statuses(watched)[-1]
    assert _NO_FORCES in closing and "max force" not in closing, closing
    assert (st.max_force, st.max_force_held) == (None, None), st


def _on_the_results_tab(page, base, folder):
    """The Results tab, its folder ``folder`` -- the folder the sidebar
    shares -- once its file menu has chosen."""
    page.goto(f"{base}/molbuilder")
    page.wait_for_function(
        "() => window.molbuilder && window.molbuilder.projects"
        "      && typeof window.molbuilder.projects.setShared"
        "         === 'function'")
    page.evaluate("(d) => window.molbuilder.projects.setShared(d, '')",
                  str(folder))
    page.goto(f"{base}/results")
    page.wait_for_function(
        "() => document.getElementById('results-file-picker-select')"
        "      .value !== ''", timeout=30000)


def _run_panel_rows(page) -> dict:
    """The Run panel's rows, label -> value."""
    return page.evaluate("""() => Object.fromEntries(
        [...document.querySelectorAll('#results-run-panel .rp-row')].map(
            (r) => [r.querySelector('.rp-key').textContent,
                    r.querySelector('.rp-val').textContent]))""")


def _as_js(n: float) -> str:
    """A number as the page prints it: JavaScript's, so ``-0`` is ``0``."""
    return "0" if n == 0 else f"{n:g}"


def test_the_results_tab_says_a_devices_forces_and_a_leads_stress(
        junction, page, monkeypatch):
    """`web/trajectory.md` § 2, `web/results.md` § 3a: opened on the device's
    point, the max-force plot's place holds the sentence that TranSIESTA
    computes no forces, and the Run panel's verdict says it where the force
    would be, with no force and no stress; opened on a lead, the Run panel
    states the lead's largest force and its stress in the (x, y) plane --
    xx, yy and xy of the tensor its record keeps in SIESTA's Voigt order
    (xx, yy, zz, yz, xz, xy; `model/parse.md` § 5d.6).  The silent
    failures: the device's plot drawing forces TranSIESTA says are wrong;
    the stress read and shown nowhere."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.live_server import serve
    device = _point(junction, "04_device", 0, _BIAS[0])
    lead = junction.bundle / "02_electrode_L" / "run-0"
    verdict = _results_dir(junction, monkeypatch, lead)["record"]["verdict"]
    stress = verdict["stress_voigt_kbar"]
    assert len(stress) == 6 and "max_force_held_ev_ang" not in verdict, (
        verdict)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    with serve() as base:
        _on_the_results_tab(page, base, device)
        # THE PLOT'S BOX comes with the trajectory viewer, mounted once the
        # menu has chosen the device's output.
        page.wait_for_function(
            """(s) => { const box = document.getElementById('force-plot');
                        return !!box && box.textContent === s; }""",
            arg=_NO_FORCES, timeout=60000)
        page.click("#results-run-panel .rp-toggle")
        rows = _run_panel_rows(page)
        assert rows.get("Forces") == _NO_FORCES, rows
        assert not {"Max force", "Stress in the (x, y) plane"} & set(rows), (
            rows)
        # THE LEAD, the tab re-pointed as a person does it -- the sidebar at
        # its folder, then Refresh (`results.md` § 2.1: the tab keeps its own
        # folder); the panel stays open over its new record.
        page.evaluate("(d) => window.molbuilder.projects.setShared(d, '')",
                      str(lead))
        page.click("#results-file-picker-refresh")
        page.wait_for_function(
            """() => [...document.querySelectorAll('#results-run-panel .rp-key')]
                .some((k) => k.textContent === 'Stress in the (x, y) plane')""",
            timeout=30000)
        rows = _run_panel_rows(page)
        # THE PAGE'S FOUR FIGURES (`run-panel.js`, `toPrecision(4)`).
        free = verdict["max_force_free_ev_ang"]
        force = _as_js(float(f"{free:.4g}"))
        assert rows.get("Max force") == f"{force} eV/Å on the free atoms", (
            rows)
        assert rows.get("Stress in the (x, y) plane") == (
            f"xx {_as_js(stress[0])} · yy {_as_js(stress[1])} · "
            f"xy {_as_js(stress[5])} kbar"), rows


def test_the_results_tab_opens_a_transmission_folder_at_its_point(
        junction, page, monkeypatch):
    """`web/results.md` § 2, § 3a: pointed at a transmission run's folder --
    the frame set's frame 1 at the second voltage -- the Results tab's menu
    lists the calculation's report first, chooses it, and opens it with the
    folder's point selected, its frame and its voltage (the I–V table's
    row); the header says it is the folder's result, its calculation's
    report, and does not flag it as a file outside the folder; the Run
    panel says the run's loop as its record states it (§ 3a, `model/parse.md`
    § 5d.5); and a point you pick in the report stays picked when a return
    to the tab reads the folder again.  The silent failures: the report
    opened at its first point -- frame 0, 0 V -- under a folder that ran
    another; the header saying the report is not in the folder; your pick
    undone each time you came back."""
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from support.live_server import serve
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    folder = _frame_point(junction.from_pair, "05_transmission", 1, _BIAS[1])
    task = (junction.from_pair.bundle / "task.json").resolve()
    with serve() as base:
        _on_the_results_tab(page, base, folder)
        menu = page.evaluate("""() => {
            const s = document.getElementById('results-file-picker-select');
            return {chosen: s.value, first: s.options[0].value}; }""")
        assert Path(menu["first"]).resolve() == task, menu
        assert menu["chosen"] == menu["first"], menu
        page.wait_for_function(
            """([f, v]) => [...document.querySelectorAll(
                   '.transport-iv-row.is-selected')]
                 .some((r) => r.dataset.frame === f && r.dataset.bias === v)""",
            arg=["1", f"{_BIAS[1]:g}"], timeout=60000)
        # THE REPORT IS THE FOLDER'S OWN ANSWER, never flagged as a file
        # that is not in it (`results.md` § 2).
        flagged = page.evaluate("""() => document.getElementById(
            'results-current-file').classList.contains('is-diverged')""")
        header = page.inner_text("#results-current-file")
        assert not flagged and "its result is its calculation's report" \
            in header, header
        # THE RUN PANEL, opened: the run's loop in its record's words.
        page.click("#results-run-panel .rp-toggle")
        panel = page.inner_text("#results-run-panel")
        assert "TBtrans's loop" in panel \
            and "completed: unpolarized (1 of 1) in " in panel, panel
        # YOUR PICK, then a return to the tab: the folder read again, its
        # answer announced -- and the point you picked still the one shown.
        page.click('.transport-curves-card .panel-tab[data-tab="iv"]')
        page.click('.transport-iv-row[data-frame="0"][data-bias="0"]'
                   ' .transport-iv-pick')
        mine = """() => [...document.querySelectorAll(
            '.transport-iv-row.is-selected')]
            .some((r) => r.dataset.frame === '0' && r.dataset.bias === '0')"""
        page.wait_for_function(mine, timeout=30000)
        page.evaluate("""() => {
            window.__announced = 0;
            document.addEventListener(
                window.molbuilder.constants.EVENT_FILE_SELECTED,
                () => { window.__announced += 1; });
            document.dispatchEvent(new Event('visibilitychange')); }""")
        page.wait_for_function("() => window.__announced >= 1",
                               timeout=30000)
        assert page.evaluate(mine), "the rescan undid the point you picked"


# --------------------------------------------------------------------- #
#  A calculation that reuses this one (`engines/transport.md` § 2a.16)   #
# --------------------------------------------------------------------- #

#: The fitted window's range asked -- wider than the catalogue's, so the
#: transmission computes only what lies outside the finished grid.
_WIDER = (-3.0, 3.0)


@pytest.fixture(scope="module")
def reused(junction, tmp_path_factory):
    """Calculations that reuse the finished one, `P/transport/T`, each made
    on the road as a person makes it (`engines/transport.md` § 2a.16): the
    same question, every group copied in; the voltage list widened, the
    device's and the transmission's done points taken over and the new one
    walked -- the transmission prepared while the device, prepared before
    it, has not launched; then that list narrowed, refused; the window
    fitted wider, each transmission point computing the energies the
    finished one did not; the same window three times finer; the low-bias
    approximation, and its window fitted
    wider, the transmission that runs once sharing as its one point; and
    the Transport tab's describe of a calculation reused.  And what a folder
    that cannot be reused is refused with."""
    from molbuilder.web.app import create_app
    from support.real_runs import RealRun, Said, _road_at, _taken, _typed
    from support.road import molbuilder

    finished = "P/transport/T"
    out = {}

    def described(name, *flags, reuses=finished):
        r = molbuilder("describe", "--calculation", "transport",
                       "--reuses", reuses, "--bundle",
                       f"P/transport/{name}", *flags)
        made = RealRun(tree=junction.tree,
                       bundle=junction.tree / "P" / "transport" / name)
        made.said["describe"] = Said(r.exit_code, r.output)
        assert r.exit_code == 0, r.output
        return made

    # EVERY GROUP OF THE DEFAULTS (1 1 1 2 3), named at one prep.
    every_group = ("--group", "1", "--group", "2", "--group", "3")

    with _road_at(junction.tree.parent, tmp_path_factory):
        # REFUSED BEFORE ANYTHING IS WRITTEN.
        out["refused"] = {}
        for step, flags, folder in (
                ("not transport", ("--reuses", "P/opt/J"), "P/transport/R1"),
                ("inside", ("--reuses", finished), f"{finished}/inner"),
                ("its label", ("--reuses", finished, "--name", "R"),
                 "P/transport/R3")):
            r = molbuilder("describe", "--calculation", "transport",
                           *flags, "--bundle", folder)
            out["refused"][step] = (Said(r.exit_code, r.output),
                                    (junction.tree / folder).exists())

        # THE SAME QUESTION: every group copied in.
        same = described("same")
        _typed(same, "prep bare", "prep", "task")
        _taken(same, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(same, "status", "status")
        _taken(same, "summarize", "summarize", "task")
        out["same"] = same

        # MORE VOLTAGES, SELF-CONSISTENT: the done points taken over.
        wider = described("wider", "--bias", "0.0,0.2,0.4",
                          "--no-low-bias-approximation")
        _typed(wider, "prep bare", "prep", "task")
        _taken(wider, "prep", "prep", "task", "--group", "1", "--group", "2",
               "--target", "this")
        _taken(wider, "prep transmission", "prep", "task", "--group", "3",
               "--target", "this")
        for g, stage in ((2, "device"), (3, "transmission")):
            _taken(wider, f"launch {stage}", "launch", "task", "--group",
                   str(g), "--mode", "direct", "--foreground", "--yes")
        _taken(wider, "status transmission", "status", "transmission")
        _taken(wider, "summarize", "summarize", "task")
        out["wider"] = wider

        # ...THAT LIST NARROWED AGAIN, reusing the wider one: nothing left
        # to walk.
        r = molbuilder("describe", "--calculation", "transport", "--reuses",
                       "P/transport/wider", "--bundle", "P/transport/narrow",
                       "--bias", "0.0,0.2", "--no-low-bias-approximation")
        assert r.exit_code == 0, r.output
        narrow = RealRun(tree=junction.tree,
                         bundle=junction.tree / "P" / "transport" / "narrow")
        _taken(narrow, "prep", "prep", "task", *every_group,
               "--target", "this")
        _typed(narrow, "launch device", "launch", "task", "--group", "2",
               "--mode", "direct", "--foreground", "--yes")
        out["narrow"] = narrow

        # THE WINDOW FITTED WIDER: each point computes what the finished
        # one did not.
        fitted = described("fitted", "--window",
                           f"{_WIDER[0]:g},{_WIDER[1]:g}")
        _typed(fitted, "prep bare", "prep", "task")
        _taken(fitted, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(fitted, "launch transmission", "launch", "task", "--group",
               "3", "--mode", "direct", "--foreground", "--yes")
        _taken(fitted, "summarize", "summarize", "task")
        out["fitted"] = fitted

        # THE SAME WINDOW, THREE TIMES FINER: each point's new energies
        # between the finished ones, its deck's contours interleaved.
        finer = described("finer", "--window", "-2,2", "--finer", "3")
        _taken(finer, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(finer, "launch transmission", "launch", "task", "--group",
               "3", "--mode", "direct", "--foreground", "--yes")
        _taken(finer, "summarize", "summarize", "task")
        out["finer"] = finer

        # THE LOW-BIAS APPROXIMATION -- the device and the transmission run
        # once -- then its window fitted wider: the transmission that runs
        # once shares as its one point.
        lowbias = described("lowbias", "--low-bias-approximation")
        _typed(lowbias, "prep bare", "prep", "task")
        _taken(lowbias, "prep", "prep", "task", *every_group,
               "--target", "this")
        for g in ("2", "3"):
            _taken(lowbias, f"launch group {g}", "launch", "task", "--group",
                   g, "--mode", "direct", "--foreground", "--yes")
        lowfit = described("lowfit", "--window",
                           f"{_WIDER[0]:g},{_WIDER[1]:g}",
                           reuses="P/transport/lowbias")
        _typed(lowfit, "prep bare", "prep", "task")
        _taken(lowfit, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(lowfit, "launch transmission", "launch", "task", "--group",
               "3", "--mode", "direct", "--foreground", "--yes")
        _taken(lowfit, "summarize", "summarize", "task")
        out["lowbias"], out["lowfit"] = lowbias, lowfit

        # THE TRANSPORT TAB'S DESCRIBE of the finished calculation's folder.
        client = create_app(config={}).test_client()
        page = junction.tree / "P" / "transport" / "page"
        page.mkdir()
        a = client.get(f"/api/transport/describe_attempt?path={finished}")
        # THE SHARED PANEL AS THE PAGE SENDS IT, nothing changed: each field
        # as it is drawn, a blank one null (`formSchema.collectForm`).
        sch = client.get(f"/api/transport/schema?engine=siesta&surface=shared"
                         f"&reuses={finished}").get_json()["schema"]
        shared = {f["name"]: f.get("value") for sec in sch["sections"]
                  for f in sec["fields"]}
        d = client.post("/api/transport/describe", json=dict(
            engine="siesta", name="page", junction=a.get_json()["citation"],
            reuses=finished, bias=[0.0, 0.2], low_bias_approximation=False,
            dest=str(page), stages={}, shared=shared))
        out["page"] = (a.get_json(), d.status_code, d.get_json(), page)
    out["finished"] = junction
    return out


def _runs_of(made, stage):
    from molbuilder.jobset.naming import stage_home
    from molbuilder.paths import attempt_dir, attempts_in
    from molbuilder.task import of as task_of
    home = stage_home(made.bundle, task_of(made.bundle), stage)
    return [attempt_dir(home.dir, n) for n in attempts_in(home.dir)]


def test_a_folder_that_cannot_be_reused_is_refused_before_anything_is_written(
        reused):
    """`engines/transport.md` § 2a.16, *Describing it*: refused before
    anything is written -- a folder that is not a transport calculation's, a
    new folder inside the finished calculation -- and a flag naming what the
    finished calculation fixes, its label."""
    for step, words in (("not transport", "is not a transport calculation"),
                        ("inside", "is inside the finished calculation"),
                        ("its label", "the label is the finished "
                                      "calculation's")):
        said, written = reused["refused"][step]
        assert said.exit_code != 0 and words in said.output, (step, said)
        assert not written, step


def test_the_same_question_copies_every_group_in(reused):
    """§ 2a.16, *At prep*: every member's newest run finished and the same
    calculation as the deck rendered now -- every group copied in, each run
    under its own name with its `.copied-from` naming the finished run and
    its launch record; the manifest said with the groups bare `prep task`
    offers; the runs read as finished runs of this calculation, *copied
    from*; and the record is the finished one's."""
    import json
    from molbuilder.runrecord import read_copied_from
    from molbuilder.transport.stages import TRANSPORT_STAGES
    same, finished = reused["same"], reused["finished"]
    bare = same.said["prep bare"].output
    for stage in TRANSPORT_STAGES:
        assert f"{stage}" in bare
    assert bare.count("copied in") == 3, bare
    for stage in TRANSPORT_STAGES:
        mine, theirs = _runs_of(same, stage), _runs_of(finished, stage)
        assert [r.name for r in mine] == [r.name for r in theirs], stage
        for r in mine:
            assert read_copied_from(r) == (
                f"P/transport/T/{r.parent.name}/{r.name}"), r
            assert (r / "run.json").is_file(), r
    status = same.said["status"].output
    assert status.count("copied from P/transport/T/") == 5, status
    a = json.loads((finished.bundle / "T.transport.json").read_text())
    b = json.loads((same.bundle / "T.transport.json").read_text())
    for pa, pb in zip(a["points"], b["points"]):
        assert pb["transmission"] == pa["transmission"]
        assert pb["current_a"] == pa["current_a"]
        assert pb["current_from"] == "tbtrans"


def test_more_voltages_take_the_finished_points_over(reused):
    """§ 2a.16, *At prep* and *What it gives*: more voltages, self-
    consistent -- the seed and both leads copied in; at its first launch
    each rung of points takes over the finished calculation's done points,
    the same calculation given the same inputs, and walks the new voltage
    from the closest converged one before it; each point taken over names
    the finished point; the ledger keeps a line for each."""
    import json
    from molbuilder.jobset.ledger import LEDGER_FILE
    from molbuilder.runrecord import read_copied_from
    wider, finished = reused["wider"], reused["finished"]
    # THE FINISHED CALCULATION'S NEWEST RUN of each rung -- its device was
    # launched again cold, so its newest is not its first.
    newest = {stage: _runs_of(finished, stage)[-1]
              for stage in ("device", "transmission")}
    shown = {stage: f"T/{r.parent.name}/{r.name}"
             for stage, r in newest.items()}
    bare = wider.said["prep bare"].output
    for stage in ("device", "transmission"):
        assert (f"{stage}: takes over 0 V, 0.2 V from {shown[stage]}; walks "
                f"0.4 V") in bare, bare
    # ...AND SO WHEN THE TRANSMISSION IS PREPARED while the device, prepared
    # by the prep before, has not launched: its points as they will be.
    later = wider.said["prep transmission"].output
    assert (f"transmission: takes over 0 V, 0.2 V from "
            f"{shown['transmission']}; walks 0.4 V") in later, later
    device = wider.said["launch device"].output
    assert f"takes over 0 V, 0.2 V from {shown['device']}" in device
    assert "runs 0.4 V (from 0.2 V's T.TSDE)" in device
    for stage in ("device", "transmission"):
        run = _runs_of(wider, stage)[0]
        assert read_copied_from(run / "v0") == f"P/transport/{shown[stage]}/v0"
        assert read_copied_from(run / "v0.4") is None
    taken = [json.loads(x) for x in (wider.bundle / LEDGER_FILE)
             .read_text().splitlines() if '"taken over"' in x]
    assert sorted((t["stage"], t["point"]) for t in taken) == [
        ("device", "0 V"), ("device", "0.2 V"),
        ("transmission", "0 V"), ("transmission", "0.2 V")]
    rec = json.loads((wider.bundle / "T.transport.json").read_text())
    assert [p["bias_v"] for p in rec["points"]] == [0.0, 0.2, 0.4]
    assert [bool(p.get("copied_from")) for p in rec["points"]] == [
        True, True, False]


def test_a_narrower_list_is_refused_naming_the_cold_launch(reused):
    """§ 2a.16: every point the finished run holds and fewer, so nothing is
    left to walk -- refused at the launch, naming the cold launch."""
    said = reused["narrow"].said["launch device"]
    assert said.exit_code != 0
    assert "nothing to run" in said.output and "--cold" in said.output


def test_a_fitted_window_computes_only_the_energies_the_finished_point_did_not(
        reused):
    """§ 2a.16, *A transmission point's energies* and *A wider window,
    fitted*: the window asked fitted to the finished grid -- every finished
    energy a point of it, recorded as fitted; each transmission point
    shares the energies the finished point computed and computes the rest,
    as line contours in its deck; its `shared/` holds the finished point's
    files and their record; and the point's result is its grid, joined --
    every energy of it, T(E) at the shared energies the finished point's,
    its current the record's, said so."""
    import json
    import numpy as np
    from molbuilder.runrecord import read_copied_from
    from molbuilder.transport.energies import grid
    fitted, finished = reused["fitted"], reused["finished"]
    said = fitted.said["describe"].output
    # THE CATALOGUE'S DEFAULT WINDOW, -2.005…2.005 eV at 401 points,
    # fitted to -3…3 eV (the contract's own example) -- the transmission's
    # own values, where the page's *use these* writes them.
    assert ("-3…3 eV asked, fitted to the finished grid -- "
            "-3.00500…3.00500 eV at 601 points") in said, said
    own = next(st["overrides"] for st in json.loads(
        (fitted.bundle / "task.json").read_text())["stages"]
               if st["name"] == "transmission")
    bare = fitted.said["prep bare"].output
    assert bare.count("shares 401 of 601 energies") == 2, bare
    assert bare.count("computes 200 here") == 2, bare
    run = _runs_of(fitted, "transmission")[0]
    for pt in ("v0", "v0.2"):
        assert read_copied_from(run / pt / "shared") == (
            f"P/transport/T/05_transmission/run-0/{pt}")
    window = grid(own["transmission_emin_ev"], own["transmission_emax_ev"],
                  own["transmission_n_points"])
    a = json.loads((finished.bundle / "T.transport.json").read_text())
    b = json.loads((fitted.bundle / "T.transport.json").read_text())
    for pa, pb in zip(a["points"], b["points"]):
        e = np.asarray(pb["energy_ev"])
        assert e.size == 601 and np.max(np.abs(e - window)) < 1e-9
        at = [int(np.argmin(np.abs(e - x))) for x in pa["energy_ev"]]
        assert np.array_equal(np.asarray(pb["transmission"])[at],
                              pa["transmission"])
        assert pb["current_from"] == "joined"
        assert pb["current_a_printed"] is None
        assert pb["shared"]["held"] == 401 and pb["shared"]["computed"] == 200
        if pa["current_a"]:
            assert abs(pb["current_a"] / pa["current_a"] - 1) < 1e-4


def test_a_finer_grid_keeps_every_finished_energy_and_reads_its_data_file(
        reused):
    """§ 2a.16, *A wider window, fitted* -- three times finer: each point's
    new energies lie between the finished ones, its deck's contours
    interleaved, which TBtrans keeps in its data file contour by contour
    and prints in its text files sorted (`model/parse.md` § 5e, *Both
    ascending*); the point's result is its grid, ascending, T at the
    finished energies the finished point's; its data file read by
    `summarize task`, its digest kept, its regions drawn.  The silent
    failure: every point sharing a finer grid refused as not one grid."""
    import json
    import numpy as np
    from molbuilder.transport.energies import grid
    finer, finished = reused["finer"], reused["finished"]
    own = next(st["overrides"] for st in json.loads(
        (finer.bundle / "task.json").read_text())["stages"]
               if st["name"] == "transmission")
    window = grid(own["transmission_emin_ev"], own["transmission_emax_ev"],
                  own["transmission_n_points"])
    a = json.loads((finished.bundle / "T.transport.json").read_text())
    b = json.loads((finer.bundle / "T.transport.json").read_text())
    assert not b["failed"], b["failed"]
    assert len(b["points"]) == len(a["points"])
    for pa, pb in zip(a["points"], b["points"]):
        e = np.asarray(pb["energy_ev"])
        assert e.size == 3 * len(pa["energy_ev"])
        assert np.max(np.abs(e - window)) < 1e-9 and np.all(np.diff(e) > 0)
        at = [int(np.argmin(np.abs(e - x))) for x in pa["energy_ev"]]
        assert np.array_equal(np.asarray(pb["transmission"])[at],
                              pa["transmission"])
        assert pb["shared"]["held"] == len(pa["energy_ev"])
        assert [d["state"] for d in pb["digest"]] == ["kept", "kept"], (
            pb["digest"])
        # THE REGIONS ON THE SAME ENERGIES AS THE TOTAL: their sum is it.
        regions = np.sum([np.asarray(v) for v in
                          pb["dos"]["by_label"].values()], axis=0)
        assert np.allclose(regions, pb["dos"]["total"],
                           rtol=_PRINTED_AND_SINGLE_RTOL, atol=0)


def test_a_transmission_that_runs_once_shares_as_its_one_point(reused):
    """§ 2a.16, *A transmission point's energies*: under the low-bias
    approximation the transmission runs once -- its one point, the run
    itself -- and shares the energies the finished run computed, its
    `shared/` in its run's folder, its result its grid joined; and a rung
    the finished calculation ran at points, run once here, is prepared here,
    said so."""
    import json
    from molbuilder.runrecord import read_copied_from
    lowbias, lowfit = reused["lowbias"], reused["lowfit"]
    bare = lowbias.said["prep bare"].output
    assert ("holds the points 0 V, 0.2 V; this calculation runs it once, "
            "at 0 V") in bare, bare
    bare = lowfit.said["prep bare"].output
    assert ("0 V    shares 401 of 601 energies with "
            "lowbias/05_transmission/run-0 -- computes 200 here") in bare, bare
    run = _runs_of(lowfit, "transmission")[0]
    assert read_copied_from(run / "shared") == (
        "P/transport/lowbias/05_transmission/run-0")
    (point,) = json.loads((lowfit.bundle / "T.transport.json").read_text()
                          )["points"]
    assert point["n_energies"] == 601 and point["current_from"] == "joined"
    assert point["shared"] == {
        "from": "P/transport/lowbias/05_transmission/run-0", "held": 401,
        "computed": 200}


def test_the_transport_tab_describes_a_calculation_that_reuses_one(reused):
    """§ 2a.16, *On the page*: the junction card's picker takes a transport
    calculation's folder -- the describe door answers its meta line, its
    junction's citation and `reuses` -- and Describe writes what `molbuilder
    describe --reuses` writes: its description with `reuses`, its template
    as the panel left it, its composed junction and its
    pseudopotentials."""
    import json
    attempt, code, answer, page = reused["page"]
    assert attempt["reuses"] == "P/transport/T"
    assert attempt["summary"].startswith("TRANSPORT CALCULATION · T · ")
    assert code == 200 and answer["ok"], answer
    task = json.loads((page / "task.json").read_text())
    assert task["reuses"] == "P/transport/T"
    # NOTHING CHANGED ON THE PANEL, nothing changed in the template: each
    # value and whose it is, as the finished calculation's.
    assert ((page / "T.template.toml").read_text()
            == (reused["finished"].bundle / "T.template.toml").read_text())
    for f in ("junction.xyz", "slot-provenance.json", "H.psml"):
        assert (page / f).is_file(), f

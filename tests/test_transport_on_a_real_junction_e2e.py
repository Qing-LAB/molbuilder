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
from support.road import _all_written, molbuilder, one_rank

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

#: THE VIBRATION'S ONE SETTING THE RELAXATION DID NOT USE -- its SCF history,
#: which moves no converged result -- so a frame set stamped with the
#: vibration's settings, not the relaxation's it was saved from, is told
#: apart (the 2026-10-24 measurement on AuBDTAu).
_VIBRATED_HISTORY = 7

_BIAS = (0.0, 0.2)

#: Each SCF stage's own mixing weight, set on its tab -- every one distinct
#: from the cited run's 0.1 and from each other, so a deck carrying another
#: stage's value is told apart; electrode_R sets none and runs the
#: template's, the cited run's (`engines/transport.md` § 2a.13, § 3.8.2a).
_OWN_MIXING = {"seed": 0.12, "electrode_L": 0.15, "device": 0.08}

#: The frame set the second calculation cites, written by the frame generator
#: from a real SIESTA vibration of the relaxed pair (`molbuilder frameset
#: from-mode`; `engines/vibration.md` § 5.10 ③): frame 0 the pair, then the
#: three-point Gauss-Hermite nodes at +/-sqrt(3) sigma, each with its weight
#: (`science/vibrational-averaging.md` § 5.1), sigma the mode's spread at the
#: temperature stated.
_NODES_SIGMA = (0.0, 3 ** 0.5, -(3 ** 0.5))
_WEIGHTS = (2 / 3, 1 / 6, 1 / 6)
_TEMPERATURE_K = 300.0
_N_FRAMES = len(_NODES_SIGMA)

#: THE SEVEN-POINT RULE, from Abramowitz & Stegun Table 25.10 (n = 7): its
#: nodes x and weights w, so each frame sits at q / sigma = sqrt(2) x with
#: weight w / sqrt(pi) (`science/vibrational-averaging.md` § 5.1) -- the
#: order the generator writes them in, frame 0 and then each pair, nearest
#: first and its positive side first (`engines/vibration.md` § 5.10 ③).
_X7 = (0.0, 0.816287882859, -0.816287882859, 1.673551628767,
       -1.673551628767, 2.651961356835, -2.651961356835)
_W7 = (0.810264617557, 0.425607252610, 0.425607252610, 0.054515582819,
       0.054515582819, 0.000971781245, 0.000971781245)

#: hc / k_B, cm K (CODATA 2018's second radiation constant): a mode of
#: wavenumber nu holds n = 1 / (exp(nu c2 / T) - 1) quanta at T
#: (`science/vibrational-averaging.md` § 2a).
_C2_CM_K = 1.438776877

#: A mode's frame set's rows as `model/structure.md` § 2.2f's table states
#: them -- ``(name, kind, unit, words, page unit)``, the structure's and each
#: frame's -- what the Transport tab's card shows for the set the fixture
#: wrote (§ 3.8.1); the generator's rule and masses as `engines/vibration.md`
#: § 5.10 ③ states them.
_SET_ROWS = (("mode_index_1based", "given", None, "Mode", None),
             ("frequency_cm1", "given", "cm-1", "Frequency", "cm⁻¹"),
             ("temperature_k", "given", "K", "Temperature", "K"),
             ("mean_quanta", "derived", None, "Mean quanta n", None),
             ("zero_point_amplitude_amu12_ang", "given", "amu^1/2 angstrom",
              "Zero-point amplitude", "amu½·Å"),
             ("max_displacement_ang", "measured", "angstrom",
              "Outermost move", "Å"),
             ("sigma_amu12_ang", "derived", "amu^1/2 angstrom",
              "Spread σ", "amu½·Å"),
             ("sampling_rule", "given", None, "Sampling rule", None),
             ("sampling_nodes", "given", None, "Nodes", None),
             ("mass_source", "given", None, "Masses", None))
_SAMPLING = {"sampling_rule": "gauss-hermite", "sampling_nodes": 3,
             "mass_source": "standard atomic weights"}
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


def _highest_mode(tree: Path, run: str) -> int:
    """The highest mode of vibration run ``run`` (tree-relative), as its
    result numbers it -- the result found through the one door for a
    result on the label (`runs.on_label_at`: at the root of the run in
    phases, `engines/vibration.md` § 5.11 § 2)."""
    from molbuilder.runfiles import ROLE_SPECTRA
    from molbuilder.runs import on_label_at
    from molbuilder.sidecars.spectra import parse_spectra_json
    result = on_label_at(tree / run, ROLE_SPECTRA)
    return int(parse_spectra_json(result).modes[-1].index_1based)


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
    from support.road import group_words

    def seed_and_leads(on):
        # ONE GROUP, ONE JOB (`engines/stages.md` § 2): the defaults put
        # the seed and the two leads in group 1.
        return group_words(on.bundle, "seed", "electrode_L", "electrode_R")
    with _on_the_road(tmp_path_factory, "junction") as tree:
        # THE RELAXATION -- an ordinary optimization, one stage.
        r = molbuilder("describe", "--structure", _write_junction(tree),
                   "--bundle", "P/opt/J", "--engine", "siesta",
                   "--calculation", "optimization", "--name", "J", "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        relax = RealRun(tree=tree, bundle=tree / "P" / "opt" / "J")
        one_rank(relax.bundle)
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
        cited_before = _all_written(tree / cite)

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
                   "--bundle", "P/transport/T",
                   "--slot", f"junction={cite}",
                   "--bias", ",".join(f"{v:g}" for v in _BIAS),
                   "--no-low-bias-approximation")
        assert r.exit_code == 0, r.output
        run = RealRun(tree=tree, bundle=tree / "P" / "transport" / "T")
        run.said["init"] = Said(r.exit_code, r.output)
        one_rank(run.bundle)
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
        run.described_before = _all_written(run.described_at)
        b = client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T2", junction=cite, bias=[0.0, 0.5],
            low_bias_approximation=True, dest=str(run.described_at)))
        run.said["describe again"] = Said(b.status_code,
                                          json.dumps(b.get_json()))
        run.described_after = _all_written(run.described_at)
        # ...AND INTO THE KIND'S TOPIC ITSELF, a folder above a calculation's
        # (`job-contracts.md` § 2.5).
        run.topic = tree / "P" / "transport"
        run.topic_before = _all_written(run.topic)
        b = client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T3", junction=cite, bias=[0.0],
            dest=str(run.topic)))
        run.said["describe into a topic"] = Said(b.status_code,
                                                 json.dumps(b.get_json()))
        run.topic_after = _all_written(run.topic)
        a = client.get(f"/api/transport/describe_attempt?path={cite}")
        run.said["describe_attempt"] = Said(a.status_code,
                                            json.dumps(a.get_json()))

        # THE REFUSAL THAT COMES AFTER A CITATION IS READ (`task.py`).
        r = molbuilder("describe", "--engine", "siesta",
                   "--calculation", "transport",
                   "--bundle", "P/transport/X",
                   "--slot", f"junction={cite}", "--bias", "0.2,0.4")
        run.said["bias not from 0"] = Said(r.exit_code, r.output)

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
        from support.real_runs import stage_status
        run.device_runs_after_refusal = stage_status(run.bundle,
                                                     "device").attempts
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
        # one of the bridge's middle two atoms free, the other held with
        # every atom the relaxation held -- a partial Hessian over fewer
        # atoms, every one balanced, which prep says as information
        # (`engines/vibration.md` § 2.3a) -- so its force constants displace
        # one atom, not two; the box ticked (the pair IS the relaxed
        # geometry), at the relaxation's settings but its SCF history
        # (`_VIBRATED_HISTORY`).  Its own copy of the pair, written through
        # the structure's doors; the pair stays as the relaxation saved it.
        from molbuilder.workingcopy_structure import StructureCodec
        one_free = "P/structure/relaxed-one-free.xyz"
        held_more = StructureCodec().load(tree / pair)
        held_more.frozen_atoms = sorted(set(held_more.frozen_atoms)
                                        | {_at(_LEAD + _WIRE // 2 - 1)})
        StructureCodec().write(held_more, tree / one_free)
        r = molbuilder("describe", "--structure", one_free, "--bundle", "P/frequency/V",
                   "--engine", "siesta", "--calculation", "vibration", "--name", "V",
                   "--psml-lib", "pseudopotential")
        assert r.exit_code == 0, r.output
        vib = RealRun(tree=tree, bundle=tree / "P" / "frequency" / "V")
        one_rank(vib.bundle)
        for name, value in {**RELAXED_WITH,
                            "pulay_history": _VIBRATED_HISTORY,
                            "already_relaxed": True}.items():
            _set_by_the_person(vib.bundle / "V.template.toml", name, value)
        _taken(vib, "prep freq", "prep", "task",
               *group_words(vib.bundle, "freq"), "--target", "this")
        _taken(vib, "launch freq", "launch", "task", "--stage", "freq",
               "--mode", "direct", "--foreground", "--yes")
        vibration = "P/frequency/V/01_freq/run-0"

        # THE FRAME SET, written from its highest mode by the generator on
        # the command line (`engines/vibration.md` § 5.10 ③).
        frame_set = "P/structure/frames.xyz"
        mode = _highest_mode(tree, vibration)
        r = molbuilder("frameset", "from-mode", vibration, "--mode",
                       str(mode), "--temperature", f"{_TEMPERATURE_K:g}",
                       "--out", frame_set)
        assert r.exit_code == 0, r.output
        run.frameset_cli = r.output

        # THE SET CITED -- a second transport calculation, its
        # pseudopotentials from the library, its spin restricted as the
        # first's; the whole ladder, a point per frame and voltage.
        r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
                   "--bundle", "P/transport/F",
                   "--slot", f"junction={frame_set}",
                   "--psml-lib", "pseudopotential",
                   "--bias", ",".join(f"{v:g}" for v in _BIAS),
                   "--no-low-bias-approximation")
        assert r.exit_code == 0, r.output
        from_pair = RealRun(tree=tree, bundle=tree / "P" / "transport" / "F")
        one_rank(from_pair.bundle)
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
        # ITS RECORD WHILE NO TRANSMISSION HAS RUN, where the Results tab
        # asks it (`engines/transport.md` § 2a.12).
        run.average_waiting = client.get(
            f"/api/transport/record?path={from_pair.bundle}").get_json()
        _taken(from_pair, "prep transmission", "prep", "task",
               *group_words(from_pair.bundle, "transmission"),
               "--target", "this")
        _taken(from_pair, "launch transmission", "launch", "task", "--stage",
               "transmission", "--mode", "direct", "--foreground", "--yes")
        # A SELECTION'S PDOS BEFORE ANY READING IS KEPT (`web/web-api.md`,
        # `/api/transport/pdos`).
        got = client.get(f"/api/transport/pdos?path={from_pair.bundle}"
                         f"&atoms=0&frame=1&point=0.2&orbitals=all")
        run.pdos_before = (got.status_code, got.get_json())
        _taken(from_pair, "summarize", "summarize", "task")

        # ONE VOLTAGE -- a device that runs once, no points: launched, then
        # again warm (its own density stands), then again --cold, which still
        # takes the inputs its first launch gathered (`job-system.md` § 5.4).
        r = molbuilder("describe", "--calculation", "transport", "--engine", "siesta",
                   "--bundle", "P/transport/S",
                   "--slot", f"junction={cite}", "--bias", "0.0")
        assert r.exit_code == 0, r.output
        single = RealRun(tree=tree, bundle=tree / "P" / "transport" / "S")
        one_rank(single.bundle)
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
            from support.real_runs import stage_status
            return " ".join(str(n) for n in
                            stage_status(c.bundle, "device").attempts)

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
            # A VOLTAGE ADDED ON TASK SETUP after the device ran -- its Save,
            # the page's door -- and the calculation's files around it.
            from support.road import _all_written as every_file
            from support.road import save_both
            t = json.loads((c.bundle / "task.json").read_text())
            t["bias"]["voltages_v"] = [*t["bias"]["voltages_v"], 0.4]
            c.files_before = every_file(c.bundle)
            c.saved = save_both(c.bundle, t, read_request(c.bundle), client)
            c.files_after = every_file(c.bundle)
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

        # A DEVICE WALK STOPPED AFTER ITS FIRST POINT (§ 2a.11, *Launched
        # again*): a copy of the voltage list, its device launched again
        # cold in the background, stopped with the line its launch printed
        # once its 0 V point is done -- a point takes some 15 s, so the next
        # is not -- then launched again warm.
        def walk_stopped(c):
            from support.real_runs import point_row, stage_status
            ran = stage_status(c.bundle, "device").attempts
            launched = _taken(c, "launch device cold", "launch", "task",
                              "--stage", "device", "--mode", "direct",
                              "--yes", "--cold")

            def first_point_done():
                # ITS 0 V POINT, as `jobset status` reads the new run.
                s = stage_status(c.bundle, "device")
                return s.attempts != ran and point_row(s, _BIAS[0])["done"]
            _until(first_point_done, within=600, what="its first point done")
            _stopped_by_its_line(c, "stop as its launch said", launched)
            # THE STOPPED RUN AS IT ENDED, as `jobset status` reads it.
            run.walk_ended = stage_status(c.bundle, "device")
            _typed(c, "launch device warm", "launch", "task", "--stage",
                   "device", "--mode", "direct", "--foreground", "--yes")
        run.walk_stopped = copy_of(run, "walk-stopped", walk_stopped)

        # THE SAME FRAMES STATING NO ROWS -- a family, each frame its own
        # curve (`model/structure.md` § 2.2f).
        run.family = "P/structure/family.xyz"
        StructureCodec().write(
            _family_of(StructureCodec().load(tree / frame_set, frames=True)),
            tree / run.family)
        # THE SPECTRUM VIEWER'S DOOR to the same generator
        # (`web/spectra.md` § 4.3), asked with the paths the page sends --
        # the result's folder and the tree picker's choice, each absolute:
        # seven frames of the same mode, then the same path again, which is
        # refused, never written over -- and, first, what the dialog shows
        # at that temperature, the load door's answer for the same result.
        from molbuilder.runfiles import ROLE_SPECTRA
        from molbuilder.runs import on_label_at
        run.frameset_shown = client.post("/api/spectra/load", json={
            "path": str(on_label_at(tree / vibration, ROLE_SPECTRA)),
            "frameset_temperature_k": _TEMPERATURE_K}).get_json()
        ask = {"path": str(on_label_at(tree / vibration, ROLE_SPECTRA)),
               "mode": mode,
               "temperature_k": _TEMPERATURE_K, "nodes": 7,
               "out": str(tree / "P" / "structure" / "frames7.xyz")}
        run.frameset_said = [
            (a.status_code, a.get_json()) for a in (
                client.post("/api/spectra/frameset", json=ask),
                client.post("/api/spectra/frameset", json=ask))]
        run.frameset_mode = mode

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

        def back_to_before_its_launch(refusal: str):
            # THE WAY BACK A REFUSAL PRINTS, typed as printed
            # (`execution/job-request.md` § 5): the folder saved as it
            # stands, the state before the group's first launch restored --
            # which puts the calculation's own files back too.
            import shlex
            back = [shlex.split(ln.strip())[1:]
                    for ln in refusal.splitlines()
                    if ln.strip().startswith("molbuilder checkpoint ")]
            assert [w[:2] for w in back] == [["checkpoint", "save"],
                                            ["checkpoint", "restore"]], \
                refusal
            for words in back:
                r = molbuilder(*words, input="y\n")
                assert r.exit_code == 0, r.output

        def changed_from_the_beginning(c, stage, template, changes):
            # WHAT A LAUNCHED STAGE COMPUTES, CHANGED in the template, as a
            # person changes it: its group's prep refused naming the way
            # back, the way back taken, the template set again, the group
            # prepared.
            for name, value in changes:
                _set_by_the_person(template, name, value)
            refused = _typed(c, f"prep {stage} refused", "prep", "task",
                             *group_words(c.bundle, stage), "--target",
                             "this")
            assert refused.exit_code != 0, refused.output
            back_to_before_its_launch(refused.output)
            for name, value in changes:
                _set_by_the_person(template, name, value)
            _taken(c, f"prep {stage}", "prep", "task",
                   *group_words(c.bundle, stage), "--target", "this")

        # A TRANSMISSION WHOSE EVERY POINT FAILS (`run-reports.md` § 2.9): a
        # copy of the frame set, its transmission k-point grid set to 3 x 3
        # -- its device ran at the Gamma point, and TBtrans stops as it
        # starts on a grid of 3 or more along x and y (`tbt_k_grid`'s help)
        # -- prepared again from the beginning and launched; the walk goes
        # on past each point.
        def transmission_refused(c):
            changed_from_the_beginning(c, "transmission",
                                       c.bundle / "F.template.toml",
                                       [("tbt_k_grid", (3, 3, 1))])
            _typed(c, "launch transmission", "launch", "task", "--stage",
                   "transmission", "--mode", "direct", "--foreground",
                   "--yes")
            _typed(c, "status transmission", "status", "transmission")
        run.refused = copy_of(from_pair, "transmission-refused",
                              transmission_refused)

        # A DEVICE WHOSE POINTS FINISH UNCONVERGED (`engines/transport.md`
        # § 2a.9): a copy of the voltage list, the device's own SCF let
        # finish unconverged (`SCF.MustConverge` false) within two
        # iterations -- its own values on Task setup's stage table, the
        # page's Save, refused for a launched stage and saved once the way
        # back is taken -- prepared and launched: each point's run finishes,
        # so the walk goes on along its frame.
        def the_devices_own_scf(own):
            # THE DEVICE'S OWN SCF VALUES on Task setup's stage table, the
            # page's Save -- refused for a launched stage, saved once the way
            # back is taken -- then the device prepared and launched.
            def alter(c):
                def save():
                    task = json.loads((c.bundle / "task.json").read_text())
                    for st in task["stages"]:
                        if st["name"] == "device":
                            st["overrides"] = {**(st.get("overrides") or {}),
                                               **own}
                    task["varies"] = sorted(set(task["varies"]) | set(own))
                    return save_both(c.bundle, task, read_request(c.bundle),
                                     client)
                refused = save()
                assert refused.status_code == 409, refused.get_json()
                back_to_before_its_launch(refused.get_json()["error"])
                saved = save()
                assert saved.status_code == 200, saved.get_json()
                _taken(c, "prep device", "prep", "task",
                       *group_words(c.bundle, "device"), "--target", "this")
                _typed(c, "launch device", "launch", "task", "--stage",
                       "device", "--mode", "direct", "--foreground", "--yes")
                _typed(c, "status device", "status", "device")
            return alter
        run.unconverged = copy_of(run, "device-unconverged",
                                  the_devices_own_scf(
                                      {"scf_must_converge": False,
                                       "max_scf_iter": 2}))
        # A DEVICE WHOSE FIRST VOLTAGE DOES NOT FINISH (§ 2a.9): its SCF
        # given one iteration and SIESTA's own rule, stop unconverged -- the
        # retry the run card allows too -- so its frame's chain ends there
        # and the walk never runs the next voltage.
        run.aborted = copy_of(run, "device-aborted",
                              the_devices_own_scf({"max_scf_iter": 1}))
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
    from molbuilder.runfiles import LAUNCH_DIR, GroupNames
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
               "--bundle", "P/transport/W",
               "--slot", f"junction={cite}",
               "--bias", ",".join(f"{v:g}" for v in _BIAS),
               "--no-low-bias-approximation")
    assert r.exit_code == 0, r.output
    walk = RealRun(tree=tree, bundle=tree / "P" / "transport" / "W")
    one_rank(walk.bundle)
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
    names = home.names
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
    """The calculation's transport record, through its parser."""
    from molbuilder.parse.sidecars.transport import TransportRecordFileParser
    from molbuilder.transport.record import record_path
    from molbuilder.task import of as task_of
    task = task_of(run.bundle)
    return TransportRecordFileParser.parse(
        record_path(run.bundle, task.label)).payload


# --------------------------------------------------------------------- #
#  The citation (`engines/transport.md` § 3.1)                           #
# --------------------------------------------------------------------- #

def test_a_finished_relaxation_composes_and_says_how_it_ended(junction):
    """§ 3.1, *What is said about how it ended*: the citation is a relaxation
    run of molbuilder's own that finished; the composed copy the
    calculation recorded at its prep carries the run's conclusion line and
    what its output says it converged -- *geometry yes*, the status verb's
    words -- and its leads are the labelled blocks."""
    from molbuilder.transport.compose import load_compose_record
    record = load_compose_record(junction.bundle, citation=junction.cite,
                                 tree_root=junction.tree)
    assert record is not None
    said = record.provenance
    assert said["evidence"].startswith("rc=0 at "), said
    assert said["relaxation"]["converged"] == "geometry yes", said
    assert said["relaxation"]["exit_code"] == 0, said
    assert len(record.electrode_left.elements) == _LEAD
    assert said["swap_electrodes"] is False


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
    assert _all_written(junction.tree / junction.cite) == \
        junction.cited_before


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


def _cited_on_the_tab(page, base, junction, cite=None) -> dict:
    """The Transport tab citing the relaxation -- or ``cite``, a path from
    the tree's root -- as a returning person's tab cites it: its session
    note, restored through the describe-and-adopt path a pick takes
    (`lib/transport/core.js` `_restoreSession`) -- and what the describe
    door answered the page."""
    cite = cite or junction.cite
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
           }""", cite), "the session note never landed"
    with page.expect_response(
            lambda r: "/api/transport/describe_attempt" in r.url,
            timeout=30000) as described:
        page.reload(wait_until="domcontentloaded")
    answer = described.value.json()
    page.wait_for_function(
        "(j) => (document.getElementById('transport-junction-readout') || {})"
        "        .textContent === j", arg=answer["citation"], timeout=15000)
    return answer


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


def test_describe_refuses_a_folder_above_a_calculations(junction):
    """`job-contracts.md` § 2.5, `web/task-setup.md` § 2: a calculation sits
    at the bottom of project / topic / its folder -- a topic is not a
    destination.  Describe into the kind's own topic is refused in the
    words every door that writes a calculation says it in (Build's
    hand-over too: `files.calculation_folder_refusal`), the topic left as
    it was.  THE SILENT FAILURE: the description written into the topic,
    every later transport calculation of the project sitting inside it.
    MUTATION THIS MUST FAIL AGAINST: the door asking no depth."""
    from molbuilder.web.blueprints.files import calculation_folder_refusal
    said = junction.said["describe into a topic"]
    assert said.exit_code == 400, said.output
    assert json.loads(said.output)["error"] == calculation_folder_refusal(
        junction.topic), said.output
    assert junction.topic_after == junction.topic_before


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
    # ...THE RELAXATION'S, never the vibration's that wrote the set.
    assert cited(by_pair)["pulay_history"] == RELAXED_WITH["pulay_history"] \
        != _VIBRATED_HISTORY, cited(by_pair)
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
    names = home.names
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
               "--bundle", "P/transport/D",
               "--slot", f"junction={cite}",
               "--bias", ",".join(f"{v:g}" for v in _BIAS),
               "--no-low-bias-approximation")
    assert r.exit_code == 0, r.output
    probe = RealRun(tree=tree, bundle=tree / "P" / "transport" / "D")
    one_rank(probe.bundle)
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
        assert len(probe.unset[rung]) == len(probe.again[rung]) > 0, rung
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
    assert c.said["runs after"].output == c.said["runs before"].output, (
        c.said["runs before"].output, c.said["runs after"].output)
    assert "prepared, not launched" not in c.said["status device"].output, (
        c.said["status device"].output)


def test_a_voltage_added_after_the_device_ran_is_not_saved(junction):
    """`web/task-setup.md` § 8 (the user, 2026-10-27: "if something is run,
    then save will simply refuse to save ... you have to return to a
    checkpoint to reprep"): a voltage added on Task setup after the device
    ran is refused by Save -- `409`, naming the stage and its points, and
    the way back, the state before its group's first launch -- and nothing
    is saved or written.  More voltages are a new calculation that reuses
    this one (`engines/transport.md` § 2a.16).  The silent failure: a
    finished run's description changed under it, read by every verb as a
    calculation the run never computed."""
    c = junction.point_added
    said = c.saved.get_json()
    assert c.saved.status_code == 409, said
    error = said["error"]
    assert "device" in error and "0.4 V" in error, error
    for line in ("molbuilder checkpoint save -p",
                 "molbuilder checkpoint restore -p",
                 "molbuilder jobset prep task --group"):
        assert line in error, error
    # NOTHING SAVED, NOTHING WRITTEN: every file of the calculation, its
    # history among them, as before the Save.
    assert c.files_after == c.files_before, (
        sorted(set(c.files_after) ^ set(c.files_before)))


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
    """`engines/transport.md` § 3, *Gotchas*: ``--bias`` must start at
    ``0.0`` -- a list starts from equilibrium, which each voltage then
    continues from.  The silent failure: a sweep described from a voltage
    with no equilibrium point, its first point starting from nothing it
    can continue."""
    r = junction.said["bias not from 0"]
    assert r.exit_code != 0 and "must start at 0.0" in r.output, r.output


# --------------------------------------------------------------------- #
#  The sweep -- one run, its points inside (§ 2a.11)                     #
# --------------------------------------------------------------------- #

def _point(run, stage, v, frame=None):
    """The point at voltage ``v`` -- of frame ``frame``, a frame set's -- in
    ``stage``'s first run (`RealRun.attempt`, `transport.stages.Point.rel`)."""
    from molbuilder.transport.stages import Point
    return run.attempt(stage) / Point(frame=frame, volts=v).rel


def test_a_sweep_is_one_run_its_points_inside(junction):
    """The device's run holds a folder per voltage, each with what the rungs
    downstream take from it; `status` says how many points are done.  Each
    point's progress log is in the run's point, which writes it, and none
    is left in the point's folder below the stage, where nothing writes to
    it again (`project-layout.md` § 1.0)."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.runfiles import WRITTEN, role_of
    from molbuilder.task import of as task_of
    from molbuilder.transport.stages import Point
    task = task_of(junction.bundle)
    label = task.label
    home = stage_home(junction.bundle, task, "device").dir
    progress = {a.role for a in WRITTEN if a.output == "progress"}
    for v in _BIAS:
        p = _point(junction, "device", v)
        for suffix in (".TS.HSX", ".TSDE"):
            assert (p / f"{label}{suffix}").is_file(), (p, suffix)
        assert any(role_of(f.name) in progress for f in p.iterdir()), p
        prepared = home / Point(volts=v).rel
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
    # THE DEVICE'S RUNS RIGHT AFTER IT: still the one that ran.
    assert junction.device_runs_after_refusal == [0]


def test_launched_cold_every_point_runs_again(junction):
    """Cold, nothing is taken over: every point of the new run is walked,
    each start written beside it -- the 0 V point from the seed's density,
    what its run gathered (its `.gathered-from`, and no `.continued-from`,
    which a point taken over would carry), and the next from the point
    before it in this walk (its `.continued-from`)."""
    from molbuilder.task import of as task_of
    from support.real_runs import point_row, stage_status
    label = task_of(junction.bundle).label
    device = stage_status(junction.bundle, "device")
    assert device.attempt == "run-1", device.attempt
    zero, then = (point_row(device, v) for v in _BIAS)
    assert zero["started_from"] is None and zero["done"], zero
    assert f"{label}.DM <- 01_seed/run-0" in device.gathered, device.gathered
    assert then["started_from"] == f"04_device/run-1/v{_BIAS[0]:g}", then


def test_a_device_launched_warm_then_cold_keeps_the_seeds_density(junction):
    """One voltage, so one run per launch: run 1, launched again warm,
    continues from run 0's own density and records only what the gather
    gave it beside that; run 2, launched again --cold, still takes every
    input its first launch gathered -- the seed's density and each lead's
    Hamiltonian, each from the run that made it (`job-system.md` § 5.4,
    *either way a transport rung keeps the inputs its first launch
    gathered*)."""
    from molbuilder.runs import folder_answer

    def record(n):
        """Run ``n``'s record, as the Results tab opens its folder."""
        return folder_answer(junction.single.attempt("device", n))["record"]

    def given(n):
        return {g["file"]: g["from"]
                for g in record(n)["deck"].get("gathered_from", [])}
    first = given(0)
    # WARM: it continues from run 0, its density run 0's own, and records
    # only what the gather gave it beside that -- the leads' Hamiltonians.
    warm = given(1)
    assert record(1)["computation"]["launch"]["continued_from"] == (
        "04_device/run-0")
    assert warm and not any(fn.endswith(".DM") for fn in warm), warm
    assert all(warm[fn] == first[fn] for fn in warm), (first, warm)
    # ...AND SAYS SO: the warm launch copied its own run's files alone, and
    # took the leads' Hamiltonians each from the run that made it -- the
    # line it printed and the ledger's `gathers` (`job-system.md` § 5.4;
    # the 2.40 review's BE3: they were said as copied from run 0, and the
    # ledger kept no source for them).
    import json
    import re
    from molbuilder.jobset.ledger import LEDGER_FILE
    said = junction.single.said["launch device again"].output
    copied = re.search(r"copied ([^;\n]*)", said)
    assert copied and not any(fn in copied.group(1) for fn in warm), said
    assert "took the inputs its kind gathered" in said, said
    entries = [json.loads(x) for x in (junction.single.bundle / LEDGER_FILE)
               .read_text().splitlines() if x.strip()]
    again = max(i for i, e in enumerate(entries)
                if e.get("decision") == "continues"
                and e.get("stage") == "device")
    gathers = next(e for e in entries[again:]
                   if e.get("decision") == "gathers"
                   and e.get("stage") == "device")
    assert {c["file"]: c["from"] for c in gathers["carried"]} == warm, (
        gathers, warm)
    cold = given(2)
    assert cold == first, (first, cold)
    assert any(fn.endswith(".DM") and src == "01_seed/run-0"
               for fn, src in cold.items()), cold


def test_launched_again_warm_it_takes_the_done_points_over(junction):
    """§ 2a.11, *Launched again*: a device walk stopped after its first
    point, launched again warm -- the point done in the stopped run is taken
    over into the new run, its `.continued-from` naming the run that
    computed it; the point not done is walked in the new run, starting from
    the point before it in this walk; the stopped run is kept as it ended.
    The silent failure: a done point run again, or a point not done copied
    as done.

    MUTATION THIS MUST FAIL AGAINST: the warm launch taking nothing over
    (the 0 V point recomputed, its start the seed's density)."""
    from molbuilder.runs import folder_answer
    from support.real_runs import point_row, stage_status
    c = junction.walk_stopped
    # THE STOP, as the walk's premise needs it: after the first point,
    # before the second.
    ended = junction.walk_ended
    first, second = (point_row(ended, v) for v in _BIAS)
    assert first["done"] and not second["done"], ended.points
    warm = c.said["launch device warm"]
    assert warm.exit_code == 0, warm.output
    device = stage_status(c.bundle, "device")
    assert device.attempts == [*ended.attempts, device.attempts[-1]], (
        ended.attempts, device.attempts)
    # THE NEW RUN: both points done -- the first taken over, naming the run
    # that computed it; the second walked from it.
    zero, then = (point_row(device, v) for v in _BIAS)
    assert zero["done"] and then["done"], device.points
    assert zero["started_from"] == f"{ended.dir}/{first['folder']}", zero
    assert then["started_from"] == f"{device.dir}/{zero['folder']}", then
    # THE STOPPED RUN, kept as it ended, as the Results tab opens its
    # points: its first point finished, its second not.
    state = {p["bias_v"]: (folder_answer(c.bundle / ended.dir / p["folder"])
                           ["status"] or {}).get("state")
             for p in (first, second)}
    assert state[_BIAS[0]] == "finished" and state[_BIAS[1]] != "finished", \
        state


def test_a_frame_sets_average_waits_for_every_frame(junction):
    """§ 2a.12: a point not done is a gap in the family, and the average at
    that voltage waits for it -- never computed from the frames that happen
    to be there.  The record asked where the Results tab asks it, the
    frame set's device run and no transmission yet: at each voltage the
    average names every frame it waits for and holds no value.  The silent
    failure: an average holding a value no frame's transmission gave, or
    not naming the frames it waits for."""
    answer = junction.average_waiting
    assert answer["ok"], answer
    at = answer["record"]["average"]["at"]
    assert [e["bias_v"] for e in at] == list(_BIAS), at
    for e in at:
        assert e["waits_for"] == list(range(_N_FRAMES)), e
        assert e["transmission"] is None and e["conductance_g0"] is None, e


def test_a_selections_pdos_is_refused_before_its_reading_is_kept(junction):
    """`web/web-api.md`, `/api/transport/pdos` (`web/results.md` § 2.5):
    the transmission launched and nothing summarized, so no point's reading
    of its data file is kept -- the PDOS of a selection is refused by name,
    saying so.  The silent failure: a curve computed from no reading."""
    status, body = junction.pdos_before
    assert status == 400 and "not read yet" in body["error"], body


def test_the_transmission_takes_the_devices_newest_run_whole(junction):
    """Its first launch gathers from the device's newest run, and says
    which (`job-system.md` § 6.0, step 1)."""
    from molbuilder.task import of as task_of
    from support.real_runs import point_row, stage_status
    label = task_of(junction.bundle).label
    _said(junction, "launch transmission", "04_device/run-1")
    # EACH POINT'S DEVICE INPUT, from that run at its own voltage -- never
    # 0 V from one run and 0.2 V from another.
    transmission = stage_status(junction.bundle, "transmission")
    for v in _BIAS:
        took = point_row(transmission, v)["took"]
        assert f"{label}.TS.HSX <- 04_device/run-1/v{v:g}" in took, (v, took)


# --------------------------------------------------------------------- #
#  A frame set -- a point per frame and voltage (§ 2a.9, § 2a.11)        #
# --------------------------------------------------------------------- #

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
            p = _point(run, "device", v, frame=f)
            for suffix in (".TS.HSX", ".TSDE"):
                assert (p / f"{label}{suffix}").is_file(), (p, suffix)
    # Against frame 0's point at the same voltage, each frame's point moved
    # its atoms as the composed set's own frame moves them -- the deck's
    # atom order -- and by nothing else.
    composed = np.asarray(recorded_junction(run.bundle).structure.frames,
                          dtype=float)
    for v in _BIAS:
        at_0 = _the_geometry_it_ran(_point(run, "device", v, frame=0))
        for f in range(1, _N_FRAMES):
            moved = (_the_geometry_it_ran(_point(run, "device", v, frame=f))
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
    from molbuilder.runrecord import read_continued_from, read_gathered_from
    run = junction.from_pair
    for f in range(_N_FRAMES):
        first = _point(run, "device", _BIAS[0], frame=f)
        assert read_continued_from(first) is None, first
        assert any(str(src) == "01_seed/run-0"
                   for src, _fn in read_gathered_from(first)), first
        nxt = _point(run, "device", _BIAS[1], frame=f)
        assert str(read_continued_from(nxt)) == str(
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
    from molbuilder.runfiles import LAUNCH_DIR
    from molbuilder.runs import openable, run_of
    from molbuilder.task import of as task_of
    from molbuilder.transport.gf_files import status_line
    run = junction.from_pair
    task = task_of(run.bundle)
    launch = stage_home(run.bundle, task, "device").dir / LAUNCH_DIR
    folder = run.attempt("device") / "tsgf"
    assert folder.is_dir() and calcdirs.where(folder).what == calcdirs.SCRATCH
    # ...and the walk's machinery beside the runs is the stage's support,
    # never a point (`project-layout.md` § 1.4a, *Where am I*).
    assert calcdirs.where(launch).what == calcdirs.SUPPORT
    assert [p.name for p in folder.iterdir() if ".TSGF" in p.name] == []
    log = next(launch.glob("*.log")).read_text()
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
            p = _point(run, "device", v, frame=f)
            assert not [x.name for x in p.iterdir() if ".TSGF" in x.name], p
            output, trail = openable(p)
            assert output is not None, trail
            said = Path(output).read_text(errors="replace")
            assert ("Using GF-file" in said) == (f > 0), (f, v)
    _said(run, "status device", "tsgf/: empty")
    # BEFORE A LAUNCH, THE SIZE: from the leads' run outputs, read through
    # the registry as the line reads them, and the prepared decks.
    lead_out = run_of(run.attempt("electrode_L")).stdout
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
            got = read_gathered_from(_point(run, "transmission", v, frame=f))
            hsx = [str(src) for src, fn in got if fn.endswith(".TS.HSX")]
            assert hsx == [str(_point(run, "device", v, frame=f)
                               .relative_to(run.bundle))], (f, v, got)


def test_a_points_run_panel_says_how_its_run_was_launched(junction):
    """`web/results.md` (the Run panel's *Computation*), `model/parse.md`
    § 5d.1b: a point of a run of points reads its run's launch record -- the
    ``run.json`` of the ``run-<n>`` folder above it -- as the run's state
    does; it is given no ``continued_from``, a walked point starting from
    the point before it -- and its deck's section lists what its run
    gathered, at ``run-<n>/`` where the points share it (§ 5d.1), as
    `status` states the run's gather.  The run's own folder holding the
    points is their container: no state of its own (`project-layout.md`
    § 1.4a).  THE SILENT FAILURES (the 2.40 review's BE2): every point's
    Computation section empty -- no mode, no launch time; a point saying it
    was given nothing but its own start; the run's folder given a state
    the status door never states.  MUTATIONS THIS MUST FAIL AGAINST: the
    deck's gather read in the point's folder alone (`record._deck`); a
    status asked of the run's folder (`runs.folder_answer`)."""
    from molbuilder import runs
    from molbuilder.calculation import register_readers
    from support.real_runs import stage_status
    register_readers()
    run = junction.from_pair
    gathered = set(stage_status(run.bundle, "device").gathered)
    assert gathered, "the device's run gathered nothing"
    for v in _BIAS:
        record = runs.record_of(runs.run_of(_point(run, "device", v,
                                                   frame=1)))
        launch = record["computation"]["launch"]
        assert launch["mode"] == "direct" and launch["launched_at"], launch
        assert "continued_from" not in launch, launch
        given = {f"{g['file']} <- {g['from']}"
                 for g in record["deck"].get("gathered_from", ())}
        assert gathered <= given, (v, gathered - given)
    assert runs.folder_answer(run.attempt("device"))["status"] is None


def test_every_prep_log_at_the_root_is_said_as_molbuilders(junction):
    """`web/results.md` § 3b, `job-contracts.md` § 2.2: a file molbuilder
    writes is read back with its run's label -- at the calculation's root a
    lead's prep log on the lead's own stem -- and the folder's listing, the
    Results tab's directory door, says each as the file card does.  THE
    SILENT FAILURE (the 2.40 review's BE5): the file card said each lead's
    log was not molbuilder's.  MUTATION THIS MUST FAIL AGAINST: the
    listing's row reading a file with the run's label alone
    (`runs._file_row`)."""
    from molbuilder.runs import about, folder_answer
    logs = sorted(junction.single.bundle.glob("*.pipeline.log"))
    assert any("electrode" in p.name for p in logs), logs
    assert all(about(p)["ours"] for p in logs), {
        p.name: about(p) for p in logs}
    listed = {f["name"]: f["about"] for f in
              folder_answer(junction.single.bundle)["files"]}
    assert all(listed[p.name] == about(p) for p in logs), {
        p.name: (listed[p.name], about(p)) for p in logs}


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


# --------------------------------------------------------------------- #
#  A run of points reports as one run (`run-reports.md` § 2.4, § 2.9)   #
# --------------------------------------------------------------------- #

#: A NUMBER IN A MESSAGE: a count, an index, a voltage or a value read off a
#: run, compared as a number; the words around them compared as they are.
_NUMBER = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def _same_words(said: str, want: str, rel: float = 1e-3) -> None:
    """``said`` is ``want`` -- its words exactly, each number within ``rel``
    of the one wanted: a conductance read off TBtrans's text files, on
    their printed energies, beside the record's, on the data file's full
    precision (`run-reports.md` § 2.9)."""
    import re
    shape = re.compile(_NUMBER)
    assert shape.sub("#", said) == shape.sub("#", want), (said, want)
    got = [float(x) for x in shape.findall(said)]
    assert got == pytest.approx([float(x) for x in shape.findall(want)],
                                rel=rel), (said, want)


def _run_index(run, stage: str, n: int):
    """The run's index and its rows by the one done rule, as the job read
    them (`transport.points_index`)."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    from molbuilder.transport import points_index as pi
    task = task_of(run.bundle)
    names = stage_home(run.bundle, task, stage).names
    folder = run.attempt(stage, n)
    index = pi.read_index(pi.index_path(folder, names))
    return index, pi.rows_of(index, folder)


def _round_said(rec, stage: str, volts: float, frames: int) -> str:
    """WHAT A ROUND'S MILESTONE SAYS, from the record: each frame's device
    point converged in its NEGF cycles, or its transmission's G/G0 =
    T(E_F) and the frames' average at the voltage (`run-reports.md`
    § 2.9)."""
    from molbuilder.structure import frame_words
    head = f"{volts:g} V"
    if stage == "device":
        device = next(x for x in rec["stages"] if x["stage"] == "device")
        cycles = {p["frame"]: p["negf"]["cycles"]
                  for p in device["by_point"] if p["bias_v"] == volts}
        said = [f"converged in {cycles[f]} NEGF iterations"
                for f in sorted(cycles, key=lambda f: f or 0)]
        if frames == 1:
            return f"{head}: {said[0]}"
        return f"{head} -- " + "; ".join(
            f"{frame_words(f, frames)}: {w}" for f, w in enumerate(said))
    g = {p["frame"]: p["conductance_g0"] for p in rec["points"]
         if p["bias_v"] == volts}
    if frames == 1:
        return f"{head} -- G/G0 {g[None]:.4g}"
    avg = next(e for e in rec["average"]["at"] if e["bias_v"] == volts)
    return (f"{head} -- G/G0: " + ", ".join(
        f"{frame_words(f, frames)} {g[f]:.4g}" for f in range(frames))
        + f"; the frames' average {avg['conductance_g0']:.4g}")


def _count(done: int, n: int, frames: int, failed: int = 0) -> str:
    return (f"{done} of {n} points done"
            + (f", {failed} failed" if failed else "")
            + f" ({frames} frame{'' if frames == 1 else 's'} × "
              f"{len(_BIAS)} voltages)")


@pytest.mark.parametrize("calc, stage, n, frames", [
    (None, "device", 1, 1),
    (None, "transmission", 0, 1),
    ("from_pair", "device", 0, _N_FRAMES),
    ("from_pair", "transmission", 0, _N_FRAMES),
], ids=["one structure's device, cold", "one structure's transmission",
        "a frame set's device", "a frame set's transmission"])
def test_a_run_of_points_reports_as_one_run(junction, calc, stage, n,
                                            frames):
    """`run-reports.md` § 2.4, § 2.9; `engines/transport.md` § 2a.11: a
    transport sweep's run sends one `start`, a `milestone` when a voltage is
    done at every frame -- the round -- and one `finish`: each point's
    monitor a member, sending neither of its own.  The round says each
    frame's point -- a device's NEGF iterations, a transmission's G/G0 and
    the frames' average -- as the record states them; the finish says the
    last round, and every message the run's count.  The silent failures: a
    `start` and a `finish` per point (six of each on a frame set), a
    round's numbers not the record's."""
    from support.real_runs import notified, told
    run = getattr(junction, calc) if calc else junction
    assert notified(run, stage, n) == ["start", "milestone", "finish"]
    rec = _record(run)
    total = frames * len(_BIAS)
    milestone, = told(run, "milestone", stage, n)
    _same_words(milestone.split(" | ")[0] + " | " + milestone.split(" | ")[1],
                f"running ({_round_said(rec, stage, _BIAS[0], frames)}) | "
                f"the run: {_count(frames, total, frames)}")
    finish, = told(run, "finish", stage, n)
    _same_words(finish.split(" | ")[0] + " | " + finish.split(" | ")[1],
                f"finished ({_round_said(rec, stage, _BIAS[1], frames)}) | "
                f"the run: {_count(total, total, frames)}")


def test_a_frame_sets_point_says_its_run_and_itself(junction):
    """`run-reports.md` § 2.8: a member monitor of a frame set's device
    point says its run's count and its own point from the run's index --
    in a transport sweep's word, *point* -- the pace over the points this
    run computed, none left once the walk ended, and the run's clock from
    its first point's start.  The silent failure: a point's card saying
    only its own SCF, nothing of the six."""
    import time
    from molbuilder.jobset.naming import stage_home
    from molbuilder.monitor import WatchedRun
    from molbuilder.runrecord import stamp_time
    from molbuilder.task import of as task_of
    run = junction.from_pair
    task = task_of(run.bundle)
    names = stage_home(run.bundle, task, "device").names
    folder = run.attempt("device")
    _index, rows = _run_index(run, "device", 0)
    me = rows[-1]
    watched = WatchedRun(names=names, run=0, directory=folder / me["rel"],
                         launched_in=folder)
    now = time.time()
    st = watched.read(now - 1.0, now)
    text = st.as_text()
    total = _N_FRAMES * len(_BIAS)
    assert st.run_words == _count(total, total, _N_FRAMES), text
    assert f"this point: {me['words']}" in text, text
    assert (st.units_done, st.units_left) == (total, 0), text
    assert "each point takes " in text and "this point " in text, text
    first = min(stamp_time(r["started"]).timestamp() for r in rows)
    assert abs(st.elapsed_s - (now - first)) < 1.0, (st.elapsed_s,
                                                      now - first)


def test_a_failed_point_is_said_once_per_reason_and_its_round_by_the_milestone(
        junction):
    """`run-reports.md` § 2.9: a transmission whose every point fails --
    TBtrans refusing its k-point grid -- says the first failure at once and
    no second of the same reason; a round whose points all ended is its
    milestone, each frame's point not done and why; the finish says the run
    failed, the first point not done and how many, and the last round.
    `status` counts the points that failed beside those done.  The silent
    failures: a message per failed point, a round never said because no
    point of it was done, a run said finished."""
    from support.real_runs import notified, told
    from support.real_runs import stage_status
    c = junction.refused
    n = stage_status(c.bundle, "transmission").attempts[-1]
    index, rows = _run_index(c, "transmission", n)
    total = _N_FRAMES * len(_BIAS)
    assert not any(r["done"] for r in rows), rows
    assert len({r["why"] for r in rows}) == 1, rows
    # THE PREMISE: TBtrans stops at its start, every point's engine exiting
    # not 0 as its report step stamped it -- so the run's state, the status
    # door's rule (finished only when every point's run ended with exit 0),
    # is *failed*.  The COUNT'S *failed*: every point that ended without
    # being done.
    assert all(x["exit"] not in (0, "0") for x in index["points"]), (
        index["points"])
    failed = total
    assert notified(c, "transmission", n) == [
        "start", "part_failed", "milestone", "finish"]

    def round_said(volts):
        from molbuilder.structure import frame_words
        return (f"{volts:g} V -- G/G0: " + ", ".join(
            f"{frame_words(r['frame'], _N_FRAMES)} not done -- {r['why']}"
            for r in rows if r["bias_v"] == volts))
    said, = told(c, "part_failed", "transmission", n)
    assert said.startswith(
        f"running ({rows[0]['words']}: not done -- {rows[0]['why']}) | "
        f"the run: {_count(0, total, _N_FRAMES, failed=1)} |"), \
        said
    milestone, = told(c, "milestone", "transmission", n)
    assert milestone.startswith(
        f"running ({round_said(_BIAS[0])}) | the run: "
        f"{_count(0, total, _N_FRAMES, failed=_N_FRAMES)} |"), \
        milestone
    finish, = told(c, "finish", "transmission", n)
    assert finish.startswith(
        f"failed ({rows[0]['words']}: not done -- {rows[0]['why']}; "
        f"{total} of {total} points not done; {round_said(_BIAS[1])}) | "
        f"the run: {_count(0, total, _N_FRAMES, failed=failed)} |"), finish
    status = c.said["status transmission"].output
    assert _count(0, total, _N_FRAMES, failed=failed) in status, status


def test_a_device_point_that_finishes_unconverged_is_walked_on_and_said_once(
        junction):
    """`engines/transport.md` § 2a.9: a voltage that does not FINISH ends
    its frame's chain -- one that finishes unconverged (`SCF.MustConverge`
    false) hands its state on, though it is not done, and the walk goes on.
    `run-reports.md` § 2.9: the report asks the walk's own question, so the
    run sends one `finish`, after its last point, stating the run as the
    status door does -- finished, every point's run finished, converged or
    not -- with the first point not done and why, and the last round.  The
    silent failures: a failed `finish` while the walk still computes, then a
    second; a run said failed that `status` says finished."""
    from support.real_runs import notified, stage_status, told
    c = junction.unconverged
    device = stage_status(c.bundle, "device")
    n = device.attempts[-1]
    _index, rows = _run_index(c, "device", n)
    assert rows[0]["why"] == "finished, not converged (SCF NO)", rows
    assert all(r["ended"] and r["state"] == "finished" for r in rows), rows
    assert notified(c, "device", n) == ["start", "milestone", "finish"]
    left = [r for r in rows if not r["done"]]
    finish, = told(c, "finish", "device", n)
    assert finish.startswith(
        f"{device.state} ({rows[0]['words']}: not done -- {rows[0]['why']}"
        + (f"; {len(left)} of {len(rows)} points not done" if len(left) > 1
           else "") + f"; {_BIAS[1]:g} V: "), (finish, device.state)
    assert device.state == "finished", device.detail


def test_a_voltage_that_does_not_finish_ends_its_frame_and_the_run_once(
        junction):
    """`engines/transport.md` § 2a.9, § 2a.11: a voltage that does not
    finish -- its SCF stopped unconverged, as SIESTA stops it -- ends its
    frame's chain, and the walk never runs the next voltage.
    `run-reports.md` § 2.9: the run's `finish` comes after the point that
    ended the chain, the run failed; the voltage never run is said so,
    where its chain ended, and counted failed, as `status` counts a point
    the ended walk never reached.  The silent failures: a finish that never
    comes, waiting on a point the walk will not run; a count that disagrees
    with `status`."""
    from support.real_runs import notified, stage_status, told
    c = junction.aborted
    device = stage_status(c.bundle, "device")
    n = device.attempts[-1]
    _index, rows = _run_index(c, "device", n)
    first, then = rows
    assert first["ended"] and first["state"] == "failed", first
    assert then["not_run"] == (f"its frame's chain ended at "
                               f"{first['words']}, which did not finish"), \
        then
    assert notified(c, "device", n) == ["start", "finish"]
    finish, = told(c, "finish", "device", n)
    count = _count(0, 2, 1, failed=2)
    assert finish.startswith(
        f"failed ({first['words']}: not done -- {first['why']}; 2 of 2 "
        f"points not done; {_BIAS[0]:g} V: not done -- {first['why']}) | "
        f"the run: {count} |"), finish
    assert device.state == "failed" and count in device.detail, (
        device.state, device.detail)


def test_a_point_the_ended_walk_never_reached_reads_failed_everywhere(
        junction, tmp_path):
    """`engines/transport.md` § 2a.11: a point the ended walk never reached
    is failed (`continuation.never_reached`), never one still to come -- as
    `status` states it, so the record's point (the Results tab's), the data
    file's device state, the Results tab's folder answer and the run's
    state each say it, every one asking the status rows.  The device whose
    first voltage did not finish: its second, never run.  THE SILENT
    FAILURE: the page, the data file and the folder calling the point
    pending while `status` counts it failed.  MUTATIONS THIS MUST FAIL
    AGAINST: the record's point state the run door's own (`concluded`,
    `record._rung_points`); a point's folder read without its run's rows
    (`runs._as_its_point` answering ``st``)."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.continuation import NEVER_REACHED
    from molbuilder.runs import folder_answer, run_answer
    from molbuilder.transport.datafile import read_data_file
    from molbuilder.transport.export import export
    from molbuilder.transport.record import collect_record
    from support.real_runs import point_row, stage_status
    c = junction.aborted
    device = stage_status(c.bundle, "device")
    never = point_row(device, _BIAS[1])
    assert never["state"] == "failed" and not never["done"], device.points
    folder = c.bundle / device.dir / never["folder"]
    calc = open_calculation(c.bundle)
    by = next(s for s in collect_record(calc, partial=True)["stages"]
              if s["stage"] == "device")["by_point"]
    at = next(p for p in by if p["bias_v"] == _BIAS[1])
    assert (at["state"], at["detail"]) == ("failed", NEVER_REACHED), at
    got = export(calc, "netcdf")
    (tmp_path / got.name).write_bytes(got.files[got.name])
    V = {k: v["data"]
         for k, v in read_data_file(tmp_path / got.name)["variables"].items()}
    k = list(V["bias_v"]).index(_BIAS[1])
    assert [str(s) for s in V["device_state"][..., k].ravel()] == [
        "failed"], V["device_state"]
    assert [str(s) for s in V["device_detail"][..., k].ravel()] == [
        NEVER_REACHED], V["device_detail"]
    status = folder_answer(folder)["status"]
    assert (status["state"], status["detail"]) == ("failed", NEVER_REACHED), \
        status
    assert run_answer(folder)["state"] == "failed", run_answer(folder)


def test_a_stopped_walk_says_no_finish_and_its_warm_launch_one(junction):
    """`run-reports.md` § 2.4: a device walk stopped after its first point
    sends its `start`, and no `finish` -- the run door reads it failed; the
    walk launched again warm is a run of its own: its `start`, the point it
    takes over listed as taken over in its index, and its `finish` saying
    the round it walked.  The silent failures: a stopped walk said
    finished, a taken-over point said not done."""
    from support.real_runs import notified, stage_status, told
    c = junction.walk_stopped
    ended = junction.walk_ended
    stopped = ended.attempts[-1]
    said = notified(c, "device", stopped)
    assert said[0] == "start" and "finish" not in said, said
    assert set(said) <= {"start", "milestone"}, said
    warm = stage_status(c.bundle, "device").attempts[-1]
    assert notified(c, "device", warm) == ["start", "finish"]
    _index, rows = _run_index(c, "device", warm)
    assert [(r["bias_v"], r["done"], bool(r["started_from"]))
            for r in rows] == [(_BIAS[0], True, True),
                               (_BIAS[1], True, False)], rows
    finish, = told(c, "finish", "device", warm)
    assert finish.startswith(f"finished ({_BIAS[1]:g} V: converged in "), \
        finish
    assert f"| the run: {_count(2, 2, 1)} |" in finish, finish


def _frame_curves(run, label, volts):
    """Each frame's T(E) at ``volts``, read from the transmission TBtrans
    wrote at that frame's point, through its parser -- its energies at the
    full precision of the point's `.TBT.nc`, which the `.TBT.AVTRANS` prints
    to five decimals (§ 2a.12, *the data file*)."""
    from molbuilder.parse.engines.tbtrans_nc import point_averages
    from molbuilder.transport.transiesta import LEAD_NAMES, LEAD_PAIR
    curves = []
    for f in range(_N_FRAMES):
        point = _point(run, "transmission", volts, frame=f)
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
    for row in fs.structure:
        assert avg[row["name"]] == row["value"], (row, avg[row["name"]])
    assert avg["rows"] == list(fs.structure) and avg["warnings"] == []
    assert avg["weights"] == pytest.approx(list(_WEIGHTS), rel=1e-12)
    assert avg["tolerance"] == 1e-6
    assert abs(avg["weight_sum"] - 1.0) <= avg["tolerance"]
    assert avg["why"] is None and len(avg["assumptions"]) == 6
    # THE FRAME SET THE AVERAGE IS TAKEN OVER, cited whole: the pair and its
    # sha256, its rows with their meanings, the vibration and every atom's
    # mass in the user order -- the cited pair's own, read back.
    from molbuilder.frame_average import MASS_CHANNEL
    from molbuilder.persist import sha256_of
    from molbuilder.sidecars.molstruct import sidecar_path_for
    cited, path = _the_set(junction), junction.tree / junction.frame_set
    masses = cited.annotations[MASS_CHANNEL].data
    assert _record(run)["frame_set"] == {
        "citation": junction.frame_set,
        "files": {p.name: sha256_of(p)
                  for p in (path, sidecar_path_for(path))},
        "rows": cited.customized_rows(),
        "vibration": cited.info["vibration"],
        MASS_CHANNEL: [masses[i] for i in range(cited.n_atoms)]}
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
    the labels of its points coordinates beside it -- every variable says
    what it is, which the one writer holds (`ncdata.check_said`); and a
    calculation of one
    structure has the same layout, one frame."""
    from molbuilder.frame_average import FRAME_ROWS, MASS_CHANNEL
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import data_file_path, read_data_file
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
    # EACH FRAME ROW'S DEFINITION its meaning as the set writes it
    # (`model/datafiles.md` § 2).
    for r in FRAME_ROWS:
        note = next(x["note"] for x in cited.customized_rows(0)
                    if x["name"] == r.name)
        assert attrs_of[r.name]["definition"].startswith(note), r.name
    one = read_data_file(data_file_path(junction.bundle, "T"))["variables"]
    assert one["transmission"]["data"].shape[:2] == (1, len(_BIAS))
    assert one["transmission"]["dims"] == ("frame", "bias_v", "energy_ev")


def test_the_frames_are_reaveraged_over_another_motion(junction):
    """`science/vibrational-averaging.md` § 5.4: the frame-set calculation's
    data file, re-averaged by `molbuilder.postprocess.reaveraged` -- at the
    set's own temperature with no swing, its weights the set's own and its
    average the record's, the data file's own; under a swing turning at the
    junction's spread, weights reproducing the motion's spread
    sqrt(sigma_T^2 + Q0^2 / 2), and three frames said blind to its shape;
    a swing turning past the outermost frame refused; a calculation of one
    structure refused, having no motion to weight.  The silent failures: a
    re-averaging that reads another variable or axis than the record
    averaged (its own motion then disagreeing with the record), or one that
    answers past the frames."""
    from molbuilder.frame_average import ReaverageError
    from molbuilder.postprocess import reaveraged
    from molbuilder.task import of as task_of
    from molbuilder.transport.datafile import data_file_path, read_data_file
    run = junction.from_pair
    path = data_file_path(run.bundle, task_of(run.bundle).label)
    V = {k: v["data"] for k, v in read_data_file(path)["variables"].items()}
    own = reaveraged(path, temperature_k=_TEMPERATURE_K)
    assert own["weights"] == pytest.approx(list(_WEIGHTS), rel=1e-9)
    assert own["bias_v"] == list(_BIAS)
    assert np.allclose(own["transmission"], V["average_transmission"],
                       rtol=1e-9, atol=0, equal_nan=True)
    assert np.allclose(own["conductance_g0"], V["average_conductance_g0"],
                       rtol=1e-9, atol=0, equal_nan=True)
    assert own["said"] == []
    sigma_t = float(V["sigma_amu12_ang"])
    q_zp = float(V["zero_point_amplitude_amu12_ang"])
    swing = reaveraged(path, temperature_k=_TEMPERATURE_K,
                       swing_quanta=(sigma_t / (2.0 * q_zp)) ** 2)
    q, w = np.asarray(V["q_amu12_ang"]), np.asarray(swing["weights"])
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert float((w * q ** 2).sum()) == pytest.approx(1.5 * sigma_t ** 2,
                                                      rel=1e-9)
    assert any("blind to the swing's shape" in s for s in swing["said"])
    with pytest.raises(ReaverageError, match="beyond the outermost frame"):
        reaveraged(path, temperature_k=_TEMPERATURE_K,
                   swing_quanta=(sigma_t / q_zp) ** 2)
    with pytest.raises(ReaverageError, match="no position along a mode"):
        reaveraged(data_file_path(junction.bundle, "T"),
                   temperature_k=_TEMPERATURE_K)


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
    # THE BRIDGE'S CHARGE FROM 0 V -- each frame against its own 0 V point,
    # the D column (engines/transport.md § 2a.12, *The device's charges*).
    d = list(V["charge_part"]).index("D")
    for f in range(V["device_charge"].shape[0]):
        for b in range(len(_BIAS)):
            assert V["device_d_change_from_0v"][f, b] == pytest.approx(
                V["device_charge"][f, b, d] - V["device_charge"][f, 0, d],
                abs=1e-9), (f, b)
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
    before = _all_written(run.bundle)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(junction.tree))
    inside = jobset("export", "--bundle", run.bundle, "--format", "csv",
                    "--to", run.bundle / "out")
    assert inside.exit_code != 0 and "inside the calculation" in (
        inside.output), inside.output
    _export(junction, monkeypatch, run, "netcdf", tmp_path)
    _export(junction, monkeypatch, run, "csv", tmp_path)
    assert _all_written(run.bundle) == before, "nothing written in it"
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
    data = data_files(_point(run, "transmission", _BIAS[0]),
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
    n = composed.n_atoms
    # THE CHAIN'S ORDER, as the junction is built (`_CHAIN`) -- the
    # expectation, never the map's code nor a record that may be missing.
    engine_of_user = list(_CHAIN["original_to_sorted"])
    user_of_engine = list(_CHAIN["sorted_to_original"])
    assert composed.atom_order is not None
    assert list(composed.atom_order.original_to_sorted) == engine_of_user
    data = atom_index_map(data_file_path(run.bundle, label))
    assert (data.listed_by, list(data.user), list(data.engine)) == (
        "user_index", list(range(n)), engine_of_user)
    point = _point(run, "transmission", _BIAS[0])
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
    from molbuilder.kind_table import rung_label
    from molbuilder.atom_index import AtomIndexUnknown
    lead = (_point(run, "device", _BIAS[0])
            / f"{rung_label(label, 'electrode_L')}.TSHS")
    with pytest.raises(AtomIndexUnknown, match="electrode_L rung's file"):
        atom_index_map(lead)
    # ...AND IN ITS OWN RUN it is that run's, a structure of its own: the
    # engine column alone.
    own = atom_index_map(run.attempt("electrode_L")
                         / f"{rung_label(label, 'electrode_L')}.XV")
    assert (own.listed_by, own.user) == ("engine_index", None), own


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
    data = data_files(_point(run, "transmission", _BIAS[0]),
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


def test_the_command_line_says_what_the_temperature_means(junction):
    """`engines/vibration.md` § 5.10 ③: `molbuilder frameset from-mode`
    prints the mode's mean number of quanta and its spread against its
    zero-point amplitude at the temperature stated -- the numbers the pair
    it wrote states -- then each frame and where the pair is.  The silent
    failure: a set written at a temperature the person never sees the
    meaning of, or a printout of other numbers than the file's."""
    from molbuilder.workingcopy_structure import StructureCodec
    pair = StructureCodec().load(junction.tree / junction.frame_set,
                                 frames=True)
    rows = {r["name"]: r["value"] for r in pair.customized_rows(None)}
    said = junction.frameset_cli
    assert (f"at {_TEMPERATURE_K:g} K the mode holds "
            f"{rows['mean_quanta']:.4g} quanta on average; its spread is "
            f"{rows['sigma_amu12_ang'] / rows['zero_point_amplitude_amu12_ang']:.4f}"
            f" x its zero-point amplitude") in said, said
    assert f"wrote {junction.tree / junction.frame_set}" in said, said


def test_the_spectrum_viewers_door_writes_the_same_generators_set(junction):
    """`web/spectra.md` § 4.3: the viewer's door asks the one generator --
    seven frames of the same mode at the same temperature, each frame's
    position along the mode and weight as the seven-point rule gives them,
    and its outermost frames moving the atom that moves most as far as the
    dialog said they would, the load door's answer at that temperature; the
    same path asked again is refused, never written over.  The silent
    failures: a door that drops the rule it was asked (three frames written
    when seven were), a dialog promising another move than the frames make,
    a pair written over that a calculation cites."""
    (status, said), (again, refused) = junction.frameset_said
    assert status == 200 and said["ok"], said
    assert said["path"] == "P/structure/frames7.xyz", said
    frames = said["frames"]
    assert [f["q_sigma"] for f in frames] == pytest.approx(
        [2 ** 0.5 * x for x in _X7], rel=1e-9, abs=1e-12)
    assert [f["weight"] for f in frames] == pytest.approx(
        [w / np.pi ** 0.5 for w in _W7], rel=1e-9)
    shown = junction.frameset_shown
    assert shown["ok"], shown
    assert shown["results"]["frameset_temperature_k"] == _TEMPERATURE_K
    (m,) = [m for m in shown["results"]["modes"]
            if m["index_1based"] == junction.frameset_mode]
    assert max(f["max_atom_displacement_ang"] for f in frames) == \
        pytest.approx(m["frameset"]["outermost_move_ang"]["7"], rel=1e-9)
    assert again == 400 and "already exist" in refused["error"], refused


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

    def stated(name, frame=None, key="value"):
        return next(r[key] for r in pair.customized_rows(frame)
                    if r["name"] == name)
    expect = {
        "structure": [{"name": n, "value": stated(n), "unit": u, "kind": k,
                       "words": w, "page_unit": pu,
                       "means": stated(n, key="note")}
                      for n, k, u, w, pu in _SET_ROWS],
        "columns": [{"name": n, "unit": u, "kind": k, "words": w,
                     "page_unit": pu, "means": stated(n, 0, key="note")}
                    for n, k, u, w, pu in _FRAME_ROWS],
        "frames": [[stated(row[0], f) for row in _FRAME_ROWS]
                   for f in range(_N_FRAMES)],
        "masses_amu": [float(pair.annotations[MASS_CHANNEL].data[i])
                       for i in range(pair.n_atoms)],
        "vibration": pair.info["vibration"],
        "weights": [stated("weight", f) for f in range(_N_FRAMES)],
        "weight_sum": pytest.approx(1.0, abs=1e-12),
        "warnings": [],
        # THE FRAMES STATE A MODE, said by the door so the card keeps no
        # list of the rows that decide it (`transport.md` § 3.8.1).
        "mode": True,
    }
    # ...and those rows the generator's: its weights, its nodes, the
    # temperature stated, the mode's quanta and spread at it, and the
    # outermost move measured on the frames written.
    assert [row[-1] for row in expect["frames"]] == pytest.approx(
        list(_WEIGHTS), rel=1e-12)
    assert [row[3] for row in expect["frames"]] == pytest.approx(
        list(_NODES_SIGMA), rel=1e-12)
    assert stated("temperature_k") == _TEMPERATURE_K
    n = 1.0 / np.expm1(stated("frequency_cm1") * _C2_CM_K / _TEMPERATURE_K)
    assert stated("mean_quanta") == pytest.approx(n, rel=1e-6)
    assert stated("sigma_amu12_ang") == pytest.approx(
        stated("zero_point_amplitude_amu12_ang") * (2 * n + 1) ** 0.5,
        rel=1e-9)
    assert {n: stated(n) for n in _SAMPLING} == _SAMPLING
    # ...every row written with its meaning, the vocabulary's own words
    # (`model/structure.md` § 2.2f).
    from molbuilder.frame_average import ROWS
    means = {r.name: r.means for r in ROWS}
    for frame in (None, *range(_N_FRAMES)):
        for r in pair.customized_rows(frame):
            assert r["note"] == means[r["name"]], (frame, r)
    # ...and every number a frame states is ITS COORDINATES' -- each atom's
    # move from frame 0, as the .xyz holds them to its six decimals, and the
    # position along the mode from them and the masses -- so a frame written
    # at another step than its rows say is told apart.
    xyz = np.asarray(pair.frames, dtype=float)
    masses = np.asarray(expect["masses_amu"])
    for f in range(_N_FRAMES):
        moved = xyz[f] - xyz[0]
        each = np.linalg.norm(moved, axis=1)
        assert stated("max_atom_displacement_ang", f) == pytest.approx(
            each.max(), abs=2e-6), f
        assert stated("displacement_ang", f) == pytest.approx(
            np.linalg.norm(moved), abs=1e-5), f
        assert abs(stated("q_amu12_ang", f)) == pytest.approx(
            np.sqrt((masses * each ** 2).sum()), abs=1e-4), f
    assert stated("max_displacement_ang") == pytest.approx(
        np.linalg.norm(xyz - xyz[0], axis=2).max(), abs=2e-6)
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
        folder = _point(junction, "transmission", p["bias_v"])
        data = data_files(folder, label)["unpolarized"]
        assert kept_state(data).state == "kept"
        assert [d["state"] for d in p["digest"]] == ["kept"], p["digest"]
        with netCDF4.Dataset(str(digest_of(data))) as ds:
            assert ds.schema == DIGEST_SCHEMA
            assert ds.data_file == data.name
            assert int(ds.data_file_size) == data.stat().st_size
            assert list(ds["energy_ev"][:]) == p["dos"]["energy_ev"]
            dos = np.asarray(ds["DOS"][:], dtype=float)
            assert np.allclose(dos.sum(axis=1), p["dos"]["total"],
                               rtol=_PRINTED_AND_SINGLE_RTOL, atol=0)
            assert (set(ds["device_orbital_atom"][:])
                    <= set(ds["device_atom"][:]))
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
    shutil.copytree(_point(junction, "transmission", _BIAS[0]), point)
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
    """`engines/transport.md` § 2a.10, *The labelling rule*: the record names
    the treatment and the bias points -- a self-consistent sweep's, every
    voltage its own point.  The silent failure: a curve read as the stronger
    claim, its record not saying which it is."""
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
        return (junction.bundle, _point(junction, "transmission", v),
                {"frame": None, "of": None, "bias_v": v})
    return (junction.from_pair.bundle,
            _point(junction.from_pair, "transmission", v, frame=f),
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
    from molbuilder.jobset.naming import stage_home
    from molbuilder.paths import FIRST_ATTEMPT, attempt_dir
    from molbuilder.task import of as task_of
    home = stage_home(calc, task_of(calc), "transmission").dir
    rel = folder.relative_to(attempt_dir(home, FIRST_ATTEMPT)).as_posix()
    where = ("what", "role", "stage", "run", "trial", "point")
    assert {k: said["place"][k] for k in where} == {
        "what": "point", "role": "run", "stage": "05_transmission", "run": 0,
        "trial": None, "point": rel}, said["place"]
    prepared = _results_dir(junction, monkeypatch,
                            home / rel)["place"]
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
    folder = _point(junction, "transmission", _BIAS[0])
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
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    from support.replay import output_lines, replay, statuses
    folder = _point(junction, "device", _BIAS[0])
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
    names = stage_home(junction.bundle, task_of(junction.bundle),
                       "device").names
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
    folder = _point(junction, "device", _BIAS[-1])
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
    folder = _point(junction, "device", _BIAS[0])
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
    device = _point(junction, "device", _BIAS[0])
    lead = junction.attempt("electrode_L")
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
    folder = _point(junction.from_pair, "transmission", _BIAS[1], frame=1)
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
    it, has not launched; then that list narrowed, every group copied in;
    the window fitted wider, each transmission point computing the energies
    the finished one did not; the same window three times finer; the
    low-bias approximation, its 0 V results copied in, and its window
    fitted wider, the transmission that runs once sharing as its one point;
    the mode's frame set cited in place of the junction a calculation of
    one voltage ran, its device launched twice, the second warm; one
    geometry where a frame set ran, copied in; a calculation whose
    transmission was stopped, reused; and the Transport tab's describe of a
    calculation reused.  And what a folder that cannot be reused is refused
    with, a set whose frame 0 is another junction among them."""
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
        # A SET WHOSE FRAME 0 IS ANOTHER JUNCTION: the mode's outer frame
        # taken out as a pair of its own, through the structure's doors.
        from molbuilder.workingcopy_structure import StructureCodec
        moved = "P/structure/moved.xyz"
        StructureCodec().write(_the_set(junction).frame_at(1),
                               junction.tree / moved)
        # REFUSED BEFORE ANYTHING IS WRITTEN.
        out["refused"] = {}
        for step, flags, folder in (
                ("not transport", ("--reuses", "P/opt/J"), "P/transport/R1"),
                ("inside", ("--reuses", finished), f"{finished}/inner"),
                ("its label", ("--reuses", finished, "--name", "R"),
                 "P/transport/R3"),
                ("another junction", ("--reuses", finished, "--slot",
                                      f"junction={moved}"),
                 "P/transport/R4")):
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

        # A CALCULATION OF ONE VOLTAGE, ITS DEVICE LAUNCHED TWICE, THE
        # SECOND WARM -- its newest device run's own record names the leads
        # alone, what it carries the seed's density too (§ 2a.16, *At
        # prep*) -- then THE MODE'S FRAME SET IN PLACE OF ITS JUNCTION
        # (*A frame set of the same junction*): its frame 0 that junction,
        # one atom more held -- the seed and the leads copied in, frame 0's
        # points taken over at the first launch, the other frames walked.
        from support.road import group_words
        r = molbuilder("describe", "--calculation", "transport",
                       "--engine", "siesta", "--bundle", "P/transport/O",
                       "--slot", f"junction={junction.cite}", "--bias", "0.0")
        assert r.exit_code == 0, r.output
        once = RealRun(tree=junction.tree,
                       bundle=junction.tree / "P" / "transport" / "O")
        one_rank(once.bundle)
        _set_by_the_person(once.bundle / "O.template.toml",
                           "spin_treatment", "restricted")
        leads = group_words(once.bundle, "seed", "electrode_L", "electrode_R")
        _taken(once, "prep seed and leads", "prep", "task", *leads,
               "--target", "this")
        _taken(once, "launch seed and leads", "launch", "task", *leads,
               "--mode", "direct", "--foreground", "--yes")
        for rung in ("device", "transmission"):
            _taken(once, f"prep {rung}", "prep", "task",
                   *group_words(once.bundle, rung), "--target", "this")
            for step in ((f"launch {rung}", f"launch {rung} again")
                         if rung == "device" else (f"launch {rung}",)):
                _taken(once, step, "launch", "task", "--stage", rung,
                       "--mode", "direct", "--foreground", "--yes")
        out["once"] = once
        framed = described("framed", "--slot",
                           f"junction={junction.frame_set}",
                           reuses="P/transport/O")
        _typed(framed, "prep bare", "prep", "task")
        _taken(framed, "prep", "prep", "task", *every_group,
               "--target", "this")
        for g in ("2", "3"):
            _taken(framed, f"launch group {g}", "launch", "task", "--group",
                   g, "--mode", "direct", "--foreground", "--yes")
        _taken(framed, "summarize", "summarize", "task")
        out["framed"] = framed

        # ONE GEOMETRY WHERE THE FINISHED CALCULATION RAN A FRAME SET: the
        # set's first frame -- the junction itself -- its results copied
        # in, re-laid, and nothing runs (§ 2a.16, *A narrower question*).
        first = described("first", "--slot", f"junction={junction.cite}",
                          "--bias", "0.0", reuses="P/transport/framed")
        _typed(first, "prep bare", "prep", "task")
        _taken(first, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(first, "summarize", "summarize", "task")
        out["first"] = first
        # ...AND WHERE IT RAN A FRAME SET AT EACH OF TWO VOLTAGES: the first
        # frame's points copied in, re-laid as this calculation's points.
        fone = described("fone", "--slot", f"junction={junction.cite}",
                         reuses="P/transport/F")
        _typed(fone, "prep bare", "prep", "task")
        _taken(fone, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(fone, "summarize", "summarize", "task")
        out["fone"] = fone
        out["F"] = RealRun(tree=junction.tree,
                           bundle=junction.tree / "P" / "transport" / "F")

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

        # VOLTAGES OF BOTH SIGNS, the leads copied in and the device
        # prepared -- what its first launch walks is asked of its plan.
        signed = described("signed", "--bias", "0,0.2,0.4,-0.2,-0.4",
                           "--no-low-bias-approximation")
        _taken(signed, "prep", "prep", "task", "--group", "1", "--group",
               "2", "--target", "this")
        out["signed"] = signed

        # ...THAT LIST NARROWED AGAIN, reusing the wider one: a narrower
        # question, answered from the finished results -- every group
        # copied in at prep, nothing runs (§ 2a.16, *A narrower question*).
        r = molbuilder("describe", "--calculation", "transport", "--reuses",
                       "P/transport/wider", "--bundle", "P/transport/narrow",
                       "--bias", "0.0,0.2", "--no-low-bias-approximation")
        assert r.exit_code == 0, r.output
        narrow = RealRun(tree=junction.tree,
                         bundle=junction.tree / "P" / "transport" / "narrow")
        _typed(narrow, "prep bare", "prep", "task")
        _taken(narrow, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(narrow, "summarize", "summarize", "task")
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
        # once, at 0 V: the finished 0 V results, copied in, and nothing
        # runs -- then its window fitted wider: the transmission that runs
        # once shares as its one point.
        lowbias = described("lowbias", "--low-bias-approximation")
        _typed(lowbias, "prep bare", "prep", "task")
        _taken(lowbias, "prep", "prep", "task", *every_group,
               "--target", "this")
        _taken(lowbias, "summarize", "summarize", "task")
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

        # A CALCULATION WHOSE TRANSMISSION WAS STOPPED IN ITS LOOP, reused:
        # what finished copied in, the transmission computed here, its
        # newest run's state named (§ 2a.16, *Which of O's results*).
        again = described(
            "again",
            reuses=junction.stopped.bundle.relative_to(junction.tree)
            .as_posix())
        _typed(again, "prep bare", "prep", "task")
        out["again"] = again

        # A RUNG OF POINTS PREPARED, ITS ENERGY POINTS CHANGED BEFORE ITS
        # FIRST LAUNCH, on Task setup's Save: `status` says it
        # (`job-system.md`, *The task*, *Changed since prepared*).
        from support.road import read_request, save_both
        edited = described("edited", "--window", "-2,2", "--finer", "3")
        _taken(edited, "prep", "prep", "task", *every_group,
               "--target", "this")
        task_now = json.loads((edited.bundle / "task.json").read_text())
        for st in task_now["stages"]:
            if st["name"] == "transmission":
                st["overrides"] = {**st.get("overrides", {}),
                                   "transmission_n_points": 7}
        task_now["varies"] = sorted(set(task_now["varies"])
                                    | {"transmission_n_points"})
        r = save_both(edited.bundle, task_now, read_request(edited.bundle))
        assert r.status_code == 200, r.get_json()
        out["edited"] = edited

        # THE TRANSPORT TAB'S DESCRIBE of the finished calculation's folder.
        client = create_app(config={}).test_client()
        page = junction.tree / "P" / "transport" / "page"
        page.mkdir()
        a = client.get(f"/api/transport/describe_attempt?path={finished}")
        # THE SHARED PANEL AS THE PAGE SENDS IT: an extension's shared
        # values are the finished calculation's, every field locked and
        # never collected (`formSchema.collectForm`) -- nothing.
        sch = client.get(f"/api/transport/schema?engine=siesta&surface=shared"
                         f"&reuses={finished}").get_json()["schema"]
        shared = {f["name"]: f.get("value") for sec in sch["sections"]
                  for f in sec["fields"] if not f.get("locked")}
        body = dict(
            engine="siesta", name="page", junction=a.get_json()["citation"],
            reuses=finished, bias=[0.0, 0.2], low_bias_approximation=False,
            dest=str(page), stages={}, shared=shared)
        d = client.post("/api/transport/describe", json=body)
        out["page"] = (a.get_json(), d.status_code, d.get_json(), page)
        # ...AND WHAT AN EXTENSION KEEPS: the surfaces it draws, and what
        # its describe refuses, into a folder that stays empty.
        locks = {}
        for surface, rung in (("shared", None), ("rung", "seed"),
                              ("rung", "electrode_L"), ("rung", "device"),
                              ("rung", "transmission")):
            got = client.get(
                f"/api/transport/schema?engine=siesta&surface={surface}"
                + (f"&rung={rung}" if rung else "")
                + f"&reuses={finished}").get_json()
            fields = [f for sec in got["schema"]["sections"]
                      for f in sec["fields"]]
            ours = [f for f in fields
                    if (f.get("locked") or {}).get("of") == "T"]
            locks[rung or surface] = {
                "locked": sum(1 for f in fields if f.get("locked")),
                "ours": len(ours), "fields": len(fields),
                # A FIXED VALUE DRAWN BLANK where the run had one.
                "blank": [f["name"] for f in ours
                          if f["locked"]["value"] is None
                          and f.get("default") is not None],
                # WHAT THE WAY OUT CARRIES: each value, whose it is, and
                # whether anybody gave it one.
                "chosen": {f["name"]: (f["locked"]["chosen"],
                                       f["locked"]["blank"], f.get("source"),
                                       "value" in f) for f in ours}}
        kept = junction.tree / "P" / "transport" / "kept"
        kept.mkdir()
        refusals = {}
        # EACH AS THE PAGE SENDS IT: one voltage carries no treatment
        # (`low_bias_approximation` null, `engines/transport.md` § 2a.10).
        for step, change in (("a seed value", {"stages": {"seed": {
                                 "mixing_weight": 0.31}}}),
                             ("a shared value", {"shared": {
                                 "mesh_cutoff": 310}})):
            r = client.post("/api/transport/describe",
                            json={**body, "dest": str(kept), **change})
            refusals[step] = (r.status_code, r.get_json().get("error"),
                              [f.get("where") for f in
                               r.get_json().get("findings") or []])
        # ...AND TASK SETUP'S SAVE OF THE EXTENSION holds the same limits.
        made = json.loads((page / "task.json").read_text())
        for st in made["stages"]:
            if st["name"] == "seed":
                st["overrides"] = {**st.get("overrides", {}),
                                   "mixing_weight": 0.31}
        made["varies"] = sorted(set(made["varies"]) | {"mixing_weight"})
        r = save_both(page, made, read_request(page))
        refusals["Task setup's Save of a seed value"] = (
            r.status_code, r.get_json().get("error"), [])
        # ...AND OF AN EXTENSION THE COMMAND LINE WROTE: a change of its
        # rename is refused.
        made = json.loads((narrow.bundle / "task.json").read_text())
        made["swap_electrodes"] = not made.get("swap_electrodes", False)
        saves = {"its rename": save_both(narrow.bundle, made,
                                         read_request(narrow.bundle))}
        out["kept"] = (locks, refusals, kept,
                       {k: (r.status_code, r.get_json().get("error"))
                        for k, r in saves.items()})
    out["finished"] = junction
    return out


def _copied_in(plan: str) -> list:
    """The groups a plan says are copied in, each by its own line --
    `group N  <members>  copied in` (`engines/transport.md` § 2a.16, *The
    manifest*) -- never a count of the words, which a group prepared here
    may say of a stage it takes inputs from."""
    import re
    return [int(m.group(1)) for m in re.finditer(
        r"^\s*group (\d+)  .*  copied in$", plan, re.M)]


def _runs_of(made, stage):
    """Every run of ``stage``, oldest first, as the status door lists them,
    each its folder (`RealRun.attempt`)."""
    from support.real_runs import stage_status
    return [made.attempt(stage, n)
            for n in stage_status(made.bundle, stage).attempts]


def _resolves(bundle, said) -> bool:
    """A ``<file> <- <run>`` line the status door states: the file is in
    that run's folder of this calculation."""
    fn, _, ref = said.partition(" <- ")
    return (Path(bundle) / ref / fn).is_file()


def test_a_folder_that_cannot_be_reused_is_refused_before_anything_is_written(
        reused):
    """`engines/transport.md` § 2a.16, *Describing it*: refused before
    anything is written -- a folder that is not a transport calculation's, a
    new folder inside the finished calculation -- a flag naming what the
    finished calculation fixes, its label -- and, *A frame set of the same
    junction*, a set whose frame 0 is not the junction it ran."""
    for step, words in (("not transport", "is not a transport calculation"),
                        ("inside", "is inside the finished calculation"),
                        ("its label", "the label is the finished "
                                      "calculation's"),
                        ("another junction", "P/structure/moved.xyz: its "
                                             "frame 0 is not the junction T "
                                             "ran")):
        said, written = reused["refused"][step]
        assert said.exit_code != 0 and words in said.output, (step, said)
        assert not written, step


#: A QUESTION THE FINISHED RESULTS ANSWER (§ 2a.16, *A group copied in*,
#: *A narrower question*): the calculation, the one it reuses, the voltages
#: it asks, where the finished calculation's newest run holds the first
#: result it takes ("" -- the same folders, every run copied whole), and
#: what that run holds that it does not ask.
_ANSWERED = [
    ("same", "finished", [0.0, 0.2], "", ()),
    ("narrow", "wider", [0.0, 0.2], "", ("0.4 V",)),
    ("first", "framed", [0.0], "f000",
     ("frame index 1 (of 3) · 0 V", "frame index 2 (of 3) · 0 V")),
    ("fone", "F", [0.0, 0.2], "f000",
     ("frame index 1 (of 3) · 0 V", "frame index 2 (of 3) · 0.2 V")),
    ("lowbias", "finished", [0.0], "v0", ("0.2 V",)),
]


@pytest.mark.parametrize("name, finished, bias, at, unasked", _ANSWERED,
                         ids=[row[0] for row in _ANSWERED])
def test_a_question_the_finished_results_answer_is_copied_in(
        reused, name, finished, bias, at, unasked, monkeypatch):
    """§ 2a.16, *A group copied in* and *A narrower question* (the user,
    2026-10-25: "One rule: reuse completes"; "when citing a multi frame
    calculation, first frame result is used ... same checking for validate
    if this reuse is correct should still be in place"): the same question,
    fewer voltages, one geometry where a frame set ran -- at one voltage and
    at two -- and the low-bias approximation where the finished calculation
    ran voltages self-consistently.  The plan says every group is copied in
    and names what the copy holds that is not asked; nothing runs.  In the
    same folders every run is copied whole under its own name; otherwise
    the asked results are re-laid where this calculation keeps them, a run
    of points stamped a container.  Each result names the one it is a copy
    of, one hop; what each was given is a file of this calculation -- the
    transmission's device the device's result at its own point -- and the
    record holds the voltages asked, each the finished result, copied.  The
    silent failure: a copy whose record of its inputs is lost -- a launch
    again sends the engine without the leads' files, a later calculation
    reusing it takes nothing over."""
    from molbuilder import calcdirs
    from molbuilder.runs import folder_answer
    from molbuilder.task import of as task_of
    from molbuilder.transport.stages import TRANSPORT_STAGES
    from support.real_runs import stage_status
    made, theirs = reused[name], reused[finished]
    bare = made.said["prep bare"].output
    assert _copied_in(bare) == [1, 2, 3], bare
    # EVERY RESULT OF THE RUNS IT TAKES FINISHED: nothing to warn of.
    assert "warning:" not in bare, bare
    for words in unasked:
        assert words in bare, bare
    if unasked:
        assert (": not asked here -- left behind)" if at
                else ": not asked here -- copied with it, never read)") \
            in bare, bare
    label = task_of(made.bundle).label
    where = theirs.bundle.relative_to(theirs.tree).as_posix()

    def one_hop(status, folder):
        """What a copy of the finished result in ``folder`` names: its own
        copy's origin where the finished calculation took it over, else
        the result itself."""
        row = next((p for p in status.points if p["folder"] == folder), {})
        return row.get("copied_from") or f"{where}/{status.dir}/{folder}"
    # EACH STAGE AS `jobset status` READS IT, here and in the finished one.
    mine = {st: stage_status(made.bundle, st) for st in TRANSPORT_STAGES}
    for stage, here in mine.items():
        ran = stage_status(theirs.bundle, stage)
        if at and stage in ("device", "transmission"):
            # RE-LAID as this calculation's run 0, opened by the Results tab
            # as its layout says: a run of points a container.
            assert here.attempts == [0], here.attempts
            role = folder_answer(made.bundle / here.dir / here.attempt)[
                "place"]["role"]
            assert role == ("container" if here.points else "run"), role
            if not here.points:
                assert here.copied_from == one_hop(
                    ran, f"{ran.attempt}/{at}"), here.copied_from
            for p in here.points:
                theirs_at = f"{ran.attempt}/{at}/{p['folder'].split('/', 1)[1]}"
                assert p["copied_from"] == one_hop(ran, theirs_at), p
        else:
            # COPIED WHOLE, every run under its own name, one hop.
            assert here.attempts == ran.attempts, (here.attempts,
                                                   ran.attempts)
            assert here.copied_from == (ran.copied_from or (
                f"{where}/{ran.dir}/{ran.attempt}")), here.copied_from
        # WHAT IT WAS GIVEN, each a file of this calculation.
        lines = here.gathered + [t for p in here.points for t in p["took"]]
        assert all(_resolves(made.bundle, ln) for ln in lines), (stage, lines)
        # ITS STAGE FOLDER IS ONE, as every folder prep makes below the
        # calculation (`project-layout.md` invariant 6b) -- a member copied
        # whole that runs once opening no run of its own here.
        assert calcdirs.where(made.bundle / here.dir).what == \
            calcdirs.STAGE, stage
    assert mine["device"].gathered, mine["device"].gathered
    # ...AND TASK SETUP NAMES EACH COPY, each by its folder (`web/task-setup.md`
    # § 3, *A group copied in*): every run the Results ladder lists of each
    # stage -- every one of them a copy here -- not the newest alone.
    from molbuilder.jobset.runstatus import every_run
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.web.app import create_app
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(made.tree))
    ladder = create_app(config={}).test_client().get(
        f"/api/task-setup/folder?dir={made.bundle}").get_json()["ladder"]
    tab = {r["stage"]: r["copied"] for r in ladder["stages"]}
    for stage, here in mine.items():
        runs = every_run(made.bundle, task_of(made.bundle), here)
        assert runs and all(r["copied_from"] for r in runs), (stage, runs)
        assert sorted((c["dir"], c["copied_from"]) for c in tab[stage]) == \
            sorted((r["dir"], r["copied_from"]) for r in runs), (stage, tab)
    # THE TRANSMISSION GIVEN THE DEVICE'S RESULT AT ITS OWN POINT.
    device, transmission = mine["device"], mine["transmission"]
    if transmission.points:
        for p, d in zip(transmission.points, device.points):
            assert (f"{label}.TS.HSX <- {device.dir}/{d['folder']}"
                    in p["took"]), (p, d)
    else:
        assert (f"{label}.TS.HSX <- {device.dir}/{device.attempt}"
                in transmission.gathered), transmission.gathered
    # THE RECORD: the voltages asked, each the finished result, copied.
    def by_point(run):
        return {((p.get("frame") or 0), p["bias_v"]): p
                for p in _record(run)["points"]}
    here, there = by_point(made), by_point(theirs)
    assert sorted(v for _f, v in here) == bias, here.keys()
    for key, point in here.items():
        assert point.get("copied_from"), point
        assert point["transmission"] == there[key]["transmission"], key
        assert point["current_a"] == there[key]["current_a"], key


def test_a_result_that_did_not_finish_is_computed_here_and_named(reused):
    """§ 2a.16, *Which of O's results can be used* and *What is never
    copied* (the user, 2026-10-25: "newest, must be finished ... only
    finished ones will guarantee the correct result"): a calculation
    reusing one whose transmission was stopped in its loop -- the seed, the
    leads and the device copied in; the transmission prepared here, the
    plan naming its newest run and that it failed; no older run stands in;
    and the plan warns of the result that is not done, how it ended, and
    that it can mean a problem in the physics or in the parameter choices
    (*The manifest*; the user, 2026-10-25).  The silent failure: a result
    that never finished copied in as if it had."""
    bare = reused["again"].said["prep bare"].output
    assert _copied_in(bare) == [1, 2], bare
    # THE PLAN'S LINE for group 3 -- not the list of the groups it offers.
    (line,) = [ln for ln in bare.splitlines()
               if ln.lstrip().startswith("group 3  ")
               and "  prepared here -- " in ln]
    assert "failed" in line \
        and "S-stopped-in-its-loop/05_transmission/run-0" in line, line
    (warned,) = [ln.strip() for ln in bare.splitlines()
                 if ln.strip().startswith("warning:")]
    assert warned.startswith("warning: S-stopped-in-its-loop/"
                             "05_transmission/run-0 holds a result that is "
                             "not done") \
        and "stopped before its end" in warned \
        and "a problem in the physics or in the parameter choices" \
        in warned, warned


def test_a_frame_set_of_the_same_junction_takes_over_frame_0(reused):
    """§ 2a.16, *A frame set of the same junction*: the mode's set cited in
    place of the junction O ran -- its frame 0 that junction, its held set
    larger -- is the calculation's junction, every frame, its label O's; the
    seed and both leads are copied in; at their first launch the device and
    the transmission take over O's point on frame 0 and walk every other
    frame's -- O's device launched twice, the second warm, so what its
    newest run carries, not its own record alone, is what the point was
    given (*At prep*); the record holds a point per frame, frame 0's copied
    from O."""
    from molbuilder.parse.sidecars.transport import TransportRecordFileParser
    from molbuilder.runrecord import read_copied_from, read_gathered_from
    from molbuilder.task import of as task_of
    from molbuilder.transport.compose import read_provenance, recorded_junction
    from molbuilder.transport.stages import frames_of, rung_points
    framed, once, junction = (reused["framed"], reused["once"],
                              reused["finished"])
    label = task_of(once.bundle).label
    # WHAT THE CASE IS: the set holds more atoms than the junction O ran,
    # and O's newest device run gathered the leads alone, its first the
    # seed's density with them.
    ran = recorded_junction(once.bundle).structure.restored()
    assert set(_the_set(junction).frozen_atoms) > set(ran.frozen_atoms)
    first, newest = _runs_of(once, "device")
    assert f"{label}.DM" in [fn for _r, fn in read_gathered_from(first)]
    assert f"{label}.DM" not in [fn for _r, fn in read_gathered_from(newest)]
    assert ("the junction is P/structure/frames.xyz, its "
            f"{_N_FRAMES} frame(s); its first frame is the first frame O ran"
            ) in framed.said["describe"].output
    task = task_of(framed.bundle)
    assert task.slots["junction"] == junction.frame_set
    assert task.label == label
    assert read_provenance(framed.bundle)["citation"] == junction.frame_set
    assert recorded_junction(framed.bundle).structure.n_frames == _N_FRAMES
    for stage in ("seed", "electrode_L", "electrode_R"):
        for r in _runs_of(framed, stage):
            assert read_copied_from(r) == (
                f"P/transport/O/{r.parent.name}/{r.name}"), r
    for stage in ("device", "transmission"):
        theirs = _runs_of(once, stage)[-1]
        run = _runs_of(framed, stage)[0]
        assert read_copied_from(run) is None, run
        points = rung_points(task, stage, frames=frames_of(framed.bundle))
        assert len(points) == _N_FRAMES, points
        for pt in points:
            got = read_copied_from(run / pt.rel)
            assert got == (f"P/transport/O/{theirs.parent.name}/"
                           f"{theirs.name}" if pt.frame == 0 else None), (
                stage, pt, got)
    rec = TransportRecordFileParser.parse(
        framed.bundle / f"{label}.transport.json").payload
    assert sorted(p["frame"] for p in rec["points"]) == list(range(_N_FRAMES))
    for p in rec["points"]:
        assert bool(p.get("copied_from")) == (p["frame"] == 0), p


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


def test_a_voltage_starts_from_the_closest_one_before_it_in_its_frame(
        reused, tmp_path):
    """§ 2a.9 *Both axes*: each walked voltage starts from the closest
    voltage computed before it in its frame (the SIESTA manual: copy the
    TSDE "from the closest, previously, calculated bias") -- and the walk
    runs it only once that point's step succeeded, which the run's index
    records as the point it needs (§ 2a.11).  Voltages of both signs,
    `0, 0.2, 0.4, -0.2, -0.4`, the device's first launch planned cold, so
    every point is walked: -0.2 V needs 0 V, not 0.4 V, the voltage walked
    before it; -0.4 V needs -0.2 V.  API-LEVEL: the plan is the launch
    verb's own (`submit.plan_launch`); the index is read through its
    reader.  THE SILENT FAILURE: the -0.2 V point started from the 0.4 V
    density, and not run when the 0.4 V point failed.  MUTATION THIS MUST
    FAIL AGAINST: a point's start the voltage before it in its frame's
    list (`_before` answering ``earlier[-1]``)."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.naming import stage_home
    from molbuilder.jobset.submit import plan_launch
    from molbuilder.task import of as task_of
    from molbuilder.transport import points_index as pi
    from molbuilder.transport.stages import Point
    signed = reused["signed"]
    plan = plan_launch(open_calculation(signed.bundle), mode="direct",
                       only="device", cold=True, record=False,
                       told=dict(kind="task", stage="device", trial=None,
                                 mode="direct", flags=["--cold"]))
    names = stage_home(signed.bundle, task_of(signed.bundle),
                       "device").names
    planned = pi.index_path(signed.attempt("device"), names)
    (tmp_path / planned.name).write_text(plan.writes.read_text(planned))
    index = pi.read_index(tmp_path / planned.name)
    needs = {e["bias_v"]: e["needs"] for e in index["points"]}
    def at(v):
        return Point(frame=None, volts=v).rel
    assert needs == {0.0: None, 0.2: at(0.0), 0.4: at(0.2), -0.2: at(0.0),
                     -0.4: at(-0.2)}, needs


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


@pytest.mark.parametrize("which", ["fitted", "finer"])
def test_a_sharing_points_round_says_the_records_conductance(reused, which):
    """§ 2a.16, *the point's result is its grid, joined*; `run-reports.md`
    § 2.9: a transmission point that shares energies with a finished point
    computes only part of its grid -- a wider window's wings, a finer
    one's energies between the finished ones -- and the round that point
    ends in says its G/G0 as the record does, T(E_F) on the grid joined --
    whole, as the job reads each point (`points_index.point_fact`): the
    message prints four digits, which this junction's T(E), near 1 across
    the window, cannot tell apart.  The silent failure: a conductance
    interpolated across the energies the point did not compute itself."""
    from molbuilder.jobset.naming import stage_home
    from molbuilder.task import of as task_of
    from molbuilder.transport.points_index import point_fact
    from support.real_runs import notified, told
    c = reused[which]
    assert notified(c, "transmission", 0) == ["start", "milestone", "finish"]
    rec = _record(c)
    index, rows = _run_index(c, "transmission", 0)
    task = task_of(c.bundle)
    names = stage_home(c.bundle, task, "transmission").names
    record_g = {p["bias_v"]: p["conductance_g0"] for p in rec["points"]}
    for r in rows:
        g, why = point_fact(index, r, c.attempt("transmission", 0), names)
        assert g == pytest.approx(record_g[r["bias_v"]], rel=1e-9), (
            r["rel"], g, why, record_g)
    milestone, = told(c, "milestone", "transmission", 0)
    _same_words(milestone.split(" | ")[0],
                f"running ({_round_said(rec, 'transmission', _BIAS[0], 1)})")
    finish, = told(c, "finish", "transmission", 0)
    _same_words(finish.split(" | ")[0],
                f"finished ({_round_said(rec, 'transmission', _BIAS[1], 1)})")


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
    itself -- and, its window fitted wider, shares the energies the
    finished run holds -- here a copy of the finished 0 V point -- its
    `shared/` in its run's folder, its result its grid joined."""
    from molbuilder.runrecord import read_copied_from
    lowfit = reused["lowfit"]
    bare = lowfit.said["prep bare"].output
    assert ("0 V    shares 401 of 601 energies with "
            "lowbias/05_transmission/run-0 -- computes 200 here") in bare, bare
    # ONE HOP (*A group copied in*): the shared files' `.copied-from` names
    # the result that computed them -- T's 0 V point, of which lowbias holds
    # a copy; the record, the run they are shared from.
    run = _runs_of(lowfit, "transmission")[0]
    assert read_copied_from(run / "shared") == (
        "P/transport/T/05_transmission/run-0/v0")
    (point,) = _record(lowfit)["points"]
    assert point["n_energies"] == 601 and point["current_from"] == "joined"
    assert point["shared"] == {
        "from": "P/transport/lowbias/05_transmission/run-0", "held": 401,
        "computed": 200}


def test_an_extension_keeps_what_its_finished_calculation_computed_once(
        reused):
    """§ 2a.16, *Extending it* (the user, 2026-10-22, 2026-10-23): an
    extension changes what is computed per voltage; the shared values and
    the rungs that run once -- the seed, the leads -- are the finished
    calculation's, drawn at its values and never a control; its voltages
    are the person's to add or remove.  The describe door answers the
    limits; Describe keeps the fixed rungs' own values, refuses a change of
    a fixed one, writing nothing, and answers what will be reused.  The
    silent failure: an extension whose seed was edited on the page, its
    every group run again while the person thought they only added a
    voltage."""
    import json
    attempt, code, answer, page = reused["page"]
    locks, refusals, kept, saves = reused["kept"]
    assert attempt["extend"]["fixed"] == ["seed", "electrode_L",
                                          "electrode_R"], attempt["extend"]
    # EVERY CONTROL'S SENTENCE, whole, from the one door.
    assert set(attempt["extend"]["says"]) >= {
        "bias", "treatment", "window", "shared", "fixed", "seed",
        "electrode_L", "electrode_R", "device", "transmission"}, \
        attempt["extend"]["says"]
    # Every shared and seed field drawn, never a control -- the finished
    # calculation's, or its rung's role's -- at the value the run had; the
    # device's and the transmission's offered.
    for name in ("shared", "seed", "electrode_L"):
        got = locks[name]
        assert got["locked"] == got["fields"] and got["ours"], (name, got)
        assert not got["blank"], (name, got["blank"])
    for name in ("device", "transmission"):
        assert locks[name]["ours"] == 0, (name, locks[name])
    # THE WAY OUT CARRIES WHAT A PERSON CHOSE, and nothing else: a rung's
    # own values, and a shared value its template says is the person's --
    # never a default, never a value cited from the junction.
    finished = json.loads((reused["finished"].bundle / "task.json")
                          .read_text())
    own = {s["name"]: s.get("overrides", {}) for s in finished["stages"]}
    for name in ("shared", "seed", "electrode_L"):
        for field, (chosen, blank, source, stated) in (
                locks[name]["chosen"].items()):
            mine = field in own.get(name, {})
            assert chosen == (mine or source == "person"), (name, field,
                                                            source)
            # ...AND A VALUE NOBODY GAVE STAYS BLANK, never cited anew.
            assert blank == (not mine and not stated), (name, field)
    # EACH REFUSED BY NAME, saying whose the value is.
    for step, said in (("a seed value", "seed's values are fixed by T"),
                       ("a shared value", "the shared values are fixed by T"),
                       ("Task setup's Save of a seed value",
                        "seed's values are fixed by T")):
        status, error, _where = refusals[step]
        assert status == 400 and said in error, (step, error)
    assert not any(kept.iterdir()), list(kept.iterdir())
    # An extension the command line wrote: its rename stays.
    assert saves["its rename"][0] == 400 and \
        "swap_electrodes" in saves["its rename"][1], saves
    # The fixed rungs hold the finished calculation's own values.
    made = json.loads((page / "task.json").read_text())
    for st in made["stages"]:
        if st["name"] in ("seed", "electrode_L", "electrode_R"):
            assert st.get("overrides", {}) == own[st["name"]], st
    assert answer["manifest"][0].startswith("reuses P/transport/T"), answer


def test_a_launched_groups_run_card_change_is_prepared_again_for_its_request(
        junction):
    """`job-system.md`, *The task*, *Changed since prepared*, the launched
    row; `job-request.md` § 5: a launched group whose job request alone
    changed -- the ranks of the finished junction's transmission raised --
    is said *changed since prepared* by the status door, each difference the
    run card's, and its next step is its prep again with its runs kept,
    never the three lines that start the group from the beginning.  The
    silent failure: a person following the status to the restore, throwing
    the finished runs away for a bigger request."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.prepared_with import RUN_CARD
    from molbuilder.jobset.runstatus import jobset_status
    from support.real_runs import RealRun, copy_of, stage_status
    from support.road import read_request, write_request

    def more_ranks(copy):
        req = read_request(copy.bundle)
        req["execution"]["mpi_np"] = int(req["execution"]["mpi_np"]) + 1
        write_request(copy.bundle, req)

    base = RealRun(tree=junction.tree,
                   bundle=junction.tree / "P" / "transport" / "T")
    made = copy_of(base, "ranks", more_ranks)
    said = stage_status(made.bundle, "transmission").changed
    assert said and all(d.kind == RUN_CARD for d in said), said
    step = next(s for s in jobset_status(open_calculation(made.bundle)).steps
                if "transmission" in s.stages)
    assert step.state == "changed since prepared", step
    assert "its runs are kept" in step.words, step.words
    assert any("prep task --group" in ln for ln in step.lines), step.lines
    assert not any("checkpoint restore" in ln for ln in step.lines), step.lines


def test_a_rung_of_points_changed_after_its_prep_is_said_by_status(reused):
    """`job-system.md`, *The task*, *Changed since prepared*: a rung of
    points' energy points changed on Task setup after its prep -- the
    status door says the value, was and now, and its group's next step is
    its prep; nothing is refused, the launch sending what prep wrote.  The
    silent failure: a transmission's prepared decks holding the old energy
    points with nothing saying so."""
    from molbuilder.calculation import open as open_calculation
    from molbuilder.jobset.runstatus import jobset_status
    from support.real_runs import stage_status
    bundle = reused["edited"].bundle
    said = stage_status(bundle, "transmission").changed
    assert any(d.said.startswith("transmission_n_points: ")
               and d.said.endswith("→ 7") for d in said), said
    step = next(s for s in jobset_status(open_calculation(bundle)).steps
                if "transmission" in s.stages)
    assert step.state == "changed since prepared", step
    assert any(ln.startswith("molbuilder jobset prep task --group")
               for ln in step.lines), step


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
    # ...THE COLUMNS THE FINISHED CALCULATION VARIES kept, as the command
    # line keeps them (`engines/stages.md` § 6.2: `varies` is the person's
    # intent) -- T's own mixing weights, promoted on Task setup.  MUTATION
    # THIS MUST FAIL AGAINST: the reuse branch writing the page's own
    # columns alone (`fresh_varies`).
    finished = json.loads((reused["finished"].bundle / "task.json")
                          .read_text())
    assert "mixing_weight" in finished["varies"], finished["varies"]
    assert set(finished["varies"]) <= set(task["varies"]), task["varies"]
    # NOTHING CHANGED ON THE PANEL, nothing changed in the template: each
    # value and whose it is, as the finished calculation's.
    assert ((page / "T.template.toml").read_text()
            == (reused["finished"].bundle / "T.template.toml").read_text())
    for f in ("junction.xyz", "slot-provenance.json", "H.psml"):
        assert (page / f).is_file(), f


def test_the_tab_cites_a_frame_set_of_the_junction_it_extends(
        reused, page, monkeypatch):
    """§ 2a.16, *A frame set of the same junction* and *On the page*, on the
    Transport tab (`web/tabs.md` § 4), driven by its clicks: extending O,
    the page says EXTENDING O, draws neither the shared panel, the
    chemistry card nor the seed's and the leads' tabs, folds what O fixes
    into a view-only list, and shows the plan before anything is written;
    the junction choice names O's junction, one frame; *Choose a frame
    set…* opens the picker, whose line for the mode's set says its frame 0
    is the junction O ran, and citing it moves the readout, the junction
    line and the plan to the set; a set whose frame 0 is another junction
    is refused where it is picked, the set held staying held; a reload
    keeps the set; choosing O's junction goes back; and the set cited
    again, Describe writes it as the junction of a calculation reusing O,
    its label O's.

    MUTATIONS THIS MUST FAIL AGAINST: the page sending O's junction while
    showing the set; the reload dropping the set; the picker asking without
    `reuses`, so the moved set cites; a plan not asked again when the
    junction changes."""
    import json
    from playwright.sync_api import expect
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.task import of as task_of
    from support.live_server import serve
    junction, once = reused["finished"], reused["once"]
    tree = junction.tree.resolve()
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    dest = tree / "P" / "transport" / "tabframed"
    dest.mkdir()
    # THE SIDEBAR ON THE NEW FOLDER, the most recent place a tab opens on
    # (`web/projects.md` § 2) -- where Describe writes.
    page.add_init_script(
        "try { sessionStorage.setItem('molbuilder.current_dir', "
        f"{json.dumps(str(dest))}); }} catch (_) {{}}")
    readout = page.locator("#transport-junction-readout")
    status = page.locator("#transport-status")

    def line():
        return page.locator("#transport-extend-junction").inner_text()

    def pick(rel):
        """The frame-set picker, by its rows: P, then structure, opened;
        ``rel`` selected; its line read once the door answered; cited."""
        page.locator("#transport-extend-frameset-btn").click()
        dialog = page.locator("dialog.molbuilder-tree-picker-dialog")
        for folder in ("P", "P/structure"):
            dialog.locator(f'li.tp-node[data-path="{tree / folder}"]'
                           " > .tp-row > .tp-twisty").click()
        dialog.locator(f'li.tp-node[data-path="{tree / rel}"]'
                       " > .tp-row").click()
        meta = dialog.locator(".tp-meta")
        expect(meta).not_to_have_text("")
        expect(meta).not_to_have_text("Reading…", timeout=60000)
        said = meta.inner_text()
        dialog.get_by_role("button", name="Cite this").click()
        return said

    plan = page.locator("#transport-send-manifest")
    with serve() as base:
        _cited_on_the_tab(page, base, junction, cite="P/transport/O")
        assert "the one O ran, one frame" in line(), line()
        # AN EXTENSION'S OWN LAYOUT: what you are doing; O's fixed values
        # not drawn as controls, but listed, folded; the plan before
        # Describe -- O's own junction, every group copied in.
        expect(page.locator("#transport-mode")).to_contain_text(
            "EXTENDING O")
        # ...ITS FIVE PARTS, in order, the way out in the first.
        for card, part in (("#junction-card", "1. What you are doing"),
                           ("#parameters-card", "3. What you can change"),
                           ("#transport-fixed-card", "4. Fixed by O"),
                           ("#composite-card", "5. The plan, then Describe")):
            expect(page.locator(f"{card} > h2")).to_have_text(part)
        expect(page.locator("#transport-extend-out "
                            "#transport-extend-leave-btn")).to_be_visible()
        for card in ("#shared-card", "#chemistry-card"):
            expect(page.locator(card)).to_be_hidden()
        for rung in ("seed", "electrode_L", "electrode_R"):
            expect(page.locator(f'.tab-btn[data-tab="{rung}"]')).to_be_hidden()
        expect(page.locator('.tab-btn[data-tab="device"]')).to_be_visible()
        page.locator("#transport-extend-fixed > summary").click()
        expect(page.locator("#transport-extend-fixed-list")).to_contain_text(
            "Shared by every stage")
        expect(plan).to_contain_text("group 2  device", timeout=120000)
        assert _copied_in(plan.inner_text()) == [1, 2, 3], plan.inner_text()
        # A CHANGE ASKS THE PLAN AGAIN: the device's mixing weight typed --
        # the device then computed here, its difference named -- and a
        # reload brings the value back, said, with the way back to O's own.
        page.locator('.tab-btn[data-tab="device"]').click()
        weight = page.locator("#t-device-mixing-weight")
        weight.fill("0.07")
        weight.dispatch_event("change")
        expect(plan).to_contain_text("ran another calculation",
                                     timeout=120000)
        page.reload(wait_until="domcontentloaded")
        notice = page.locator("#transport-extend-restored")
        expect(notice).to_contain_text(
            "Restored values you typed for O before", timeout=60000)
        # THE VALUE TYPED, and no field left blank (a blank is no value).
        assert "0.07" in notice.inner_text(), notice.inner_text()
        assert "null" not in notice.inner_text(), notice.inner_text()
        page.locator("#transport-extend-own-values-btn").click()
        expect(notice).to_be_hidden()
        # THE PLAN ASKED AGAIN for O's own values -- its answer waited for,
        # never read off a box a reload left empty.
        expect(plan).to_contain_text("group 3  transmission", timeout=120000)
        expect(plan).not_to_contain_text("ran another calculation",
                                         timeout=120000)
        assert _copied_in(plan.inner_text()) == [1, 2, 3], plan.inner_text()
        said = pick(junction.frame_set)
        assert "FRAME 0 IS THE JUNCTION O RAN" in said, said
        expect(readout).to_have_text(junction.frame_set, timeout=30000)
        assert "a frame set whose first frame is the junction O ran" in line(), \
            line()
        # ...THE PLAN FOLLOWS THE SET: frame 0's results taken over.
        expect(plan).to_contain_text("takes over frame index 0",
                                     timeout=120000)
        # ANOTHER JUNCTION, refused where it is picked.
        said = pick("P/structure/moved.xyz")
        assert "its frame 0 is not the junction O ran" in said, said
        expect(status).to_contain_text("its frame 0 is not the junction O "
                                       "ran", timeout=30000)
        expect(readout).to_have_text(junction.frame_set)
        # A RELOAD KEEPS THE SET -- and O's own values put back stay put:
        # nothing restored to say.
        page.reload(wait_until="domcontentloaded")
        expect(readout).to_have_text(junction.frame_set, timeout=60000)
        expect(plan).to_contain_text("group 3  transmission", timeout=120000)
        expect(notice).to_be_hidden()
        assert "a frame set whose first frame is the junction O ran" in line(), \
            line()
        # BACK TO O'S JUNCTION, and the set cited again.
        page.locator("#transport-extend-own-btn").click()
        expect(readout).to_have_text(junction.cite)
        assert "the one O ran, one frame" in line(), line()
        pick(junction.frame_set)
        expect(readout).to_have_text(junction.frame_set, timeout=30000)
        send = page.locator("#transport-send-btn")
        expect(send).to_be_enabled(timeout=60000)
        with page.expect_response(
                lambda r: r.url.endswith("/api/transport/describe"),
                timeout=120000) as described:
            send.click()
        answer = described.value.json()
        assert answer.get("ok"), answer
    task = task_of(dest)
    assert task.slots["junction"] == junction.frame_set
    assert task.reuses == "P/transport/O"
    assert task.label == task_of(once.bundle).label

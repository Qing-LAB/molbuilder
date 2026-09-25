"""What each of the three citable things brings — `engines/transport.md` § 3.1.

The middle case is why this file exists.  It was ruled on 2026-08-29
(`archive/2026-09-01-transport-design.md` § 4.1b, *"the condition has three
shades"*), the live contract carried only two, and when the parameter path
moved onto the template on 2026-09-16 the code that acted on it lost its
caller and nothing noticed -- because every path that READS the record
survived.  The tab went on printing "contract RECORDED" about settings
nothing applied.

Asserted through `siesta_config_from_citation`, which `template.md` § 6.4
names as the whole of the filling, so this pins what the TEMPLATE is written
from -- which is what every one of the five decks then renders from.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.structure import Structure
from molbuilder.transport.citation_defaults import siesta_config_from_citation
from molbuilder.workingcopy_structure import StructureCodec

#: Deliberately NOT the catalogue's defaults, so a value that arrives proves
#: it was read rather than defaulted: the defaults are 300 Ry / DZP / 1x1x1.
RECORDED = {
    "siesta_mesh_cutoff_ry":    400,
    "basis_size":               "TZP",
    "xc_functional":            "GGA",
    "xc_authors":               "PBE",
    "k_mesh_transverse":        [4, 4, 1],
    "electronic_temperature_k": 350.0,
}


def _junction(**info):
    s = Structure(
        elements=["Au", "Au", "S", "Au", "Au"],
        positions=np.array([[0., 0, z] for z in (0, 2.4, 5.0, 7.6, 10.0)]),
        cell=np.diag([12., 12., 14.4]),
        axis_kind=("periodic", "periodic", "transport"),
        regions={"L-electrode": [0, 1], "bridge": [2], "R-electrode": [3, 4]})
    if info:
        s.set_info("calculation", info)
    return s


def _cite(tmp_path, struct):
    StructureCodec().write(struct, tmp_path / "junction.xyz")
    return siesta_config_from_citation(tmp_path, label="T")


def test_a_saved_structure_that_remembers_its_run_brings_its_settings(tmp_path):
    """§ 3.1's middle case, and the one that was lost.

    Measured before the restore: this pair produced 300 Ry, DZP and
    Gamma-only -- and Gamma-only is not merely a different number.  § 7:
    "Gamma-only (1x1x1) is wrong for periodic metallic leads: even with the
    lead atoms frozen it gives a poorly defined E_F", and T(E) is measured
    against that E_F, so every feature lands at the wrong energy while the
    run converges and looks normal.
    """
    cfg = _cite(tmp_path, _junction(
        engine="siesta", source="JunctionRelax.fdf", contract=dict(RECORDED)))
    assert cfg.mesh_cutoff == pytest.approx(400.0)
    assert cfg.basis_size == "TZP"
    assert cfg.xc_functional == "GGA"
    assert cfg.xc_authors == "PBE"
    assert cfg.electronic_temperature == pytest.approx(350.0)
    # The transverse pair carries; the transport axis is NOT sampled (I8).
    assert cfg.kgrid == (4, 4, 1)
    assert cfg.tbt_k_grid == (4, 4, 1)


def test_a_saved_structure_with_no_run_behind_it_leaves_the_rows_unanswered_on_both_roads(
        isolated_projects_root, web_client):
    """§ 3.1's third case, on the two roads that write a description.

    Nothing was measured, so nothing is claimed: every `citation` row the
    pair does not answer is written VALUELESS into the template
    (`engines/transport.md` § 3.8.3), by `jobset init` and by the browser's
    describe door alike -- one door, `transport_template_text`.  Until
    2026-09-24 the CLI wrote 300 Ry and DZP into those rows while the same
    pair described in the browser left them empty: two files for one
    description, and the CLI's claimed a basis no run had said.  The
    remembered pair beside it is the contrast that proves the filter is
    *unanswered*, not *every citation row*: a value the record answers
    arrives on both roads.  What `prep` writes into the deck for an empty
    row -- the documented default, marked -- is § 6.6 obligation 4's, not
    this file's.
    """
    from pathlib import Path

    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.template import one, read_template

    root = isolated_projects_root
    codec = StructureCodec()
    bare = root / "J" / "structure" / "bare"
    bare.mkdir(parents=True)
    codec.write(_junction(), bare / "junction.xyz")
    remembered = root / "J" / "structure" / "remembered"
    remembered.mkdir(parents=True)
    codec.write(_junction(engine="siesta", source="JunctionRelax.fdf",
                          contract=dict(RECORDED)),
                remembered / "junction.xyz")

    def cli(slot, bundle):
        r = CliRunner().invoke(jobset_group, [
            "init", "--calculation", "transport", "--shape", "hierarchical",
            "--bundle", bundle, "--slot", f"junction={slot}"])
        assert r.exit_code == 0, r.output
        return read_template(
            (root / bundle / f"{Path(bundle).name}.template.toml")
            .read_text())

    def browser(slot):
        r = web_client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T", junction=slot, bias=[0.0]))
        assert r.status_code == 200, r.get_json()
        return read_template(r.get_json()["files"][1]["text"])

    for tmpl in (cli("J/structure/bare", "J/transport/Tbare"),
                 browser("J/structure/bare")):
        for name in ("mesh_cutoff", "basis_size", "xc_functional",
                     "electronic_temperature", "kgrid"):
            assert one(tmpl, name).value is None, (
                f"{name} carries a value nobody said")
    for tmpl in (cli("J/structure/remembered", "J/transport/Tmem"),
                 browser("J/structure/remembered")):
        assert one(tmpl, "mesh_cutoff").value == pytest.approx(400.0)
        assert one(tmpl, "basis_size").value == "TZP"


def test_a_record_with_no_contract_block_brings_none(tmp_path):
    """`recorded_contract_of` answers None unless the block holds a NON-EMPTY
    `contract`, so a pair carrying only provenance is the third case."""
    cfg = _cite(tmp_path, _junction(engine="siesta", source="x.fdf"))
    assert cfg.mesh_cutoff == pytest.approx(300.0)


def test_the_transport_axis_is_forced_to_one_whichever_source_answered(
        tmp_path):
    """The record states a relaxation's THREE-axis grid; a transport
    calculation does not sample the transport axis at all.  One rule, both
    sources -- `_apply_kgrid` -- so a deck and a record cannot force it
    differently."""
    rec = dict(RECORDED, k_mesh_transverse=[3, 5, 7])
    cfg = _cite(tmp_path, _junction(engine="siesta", source="x.fdf",
                                    contract=rec))
    assert cfg.kgrid == (3, 5, 1), "the third component is not the deck's"


# ------------------------------------------------------------------ #
#  "No record" has three causes, and they are not the same news      #
# ------------------------------------------------------------------ #

def _why(tmp_path, provenance=None):
    import json
    from molbuilder.transport.compose import (PROVENANCE_FILE,
                                              load_compose_record)
    if provenance is not None:
        (tmp_path / PROVENANCE_FILE).write_text(json.dumps(provenance))
    why: list = []
    assert load_compose_record(tmp_path, citation="new/attempt",
                               why=why) is None
    return why[0]


def test_nothing_composed_here_says_so(tmp_path):
    assert "no slot-provenance.json" in _why(tmp_path)


def test_a_record_for_a_DIFFERENT_citation_says_which(tmp_path):
    """The likeliest cause on a travelled folder, and the one that read as
    "the record is not beside task.json" -- sending a person looking for a
    file they are standing on.  Re-point a slot after re-relaxing, prep on
    a cluster, and the record IS there; it is just the previous attempt's.
    """
    msg = _why(tmp_path, {"citation": "old/attempt", "form": "relaxation"})
    assert "old/attempt" in msg and "new/attempt" in msg


def test_an_incomplete_record_names_the_missing_files(tmp_path):
    msg = _why(tmp_path, {"citation": "new/attempt", "form": "relaxation"})
    assert "incomplete" in msg and "junction.xyz" in msg

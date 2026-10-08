"""A citation brings its settings (`engines/transport.md` § 3.1, § 3.8.0):
the cited relaxation run's deck answers the electronic description, into
this calculation's own template, on both roads that write a description --
`jobset init --slot junction=…` and the Transport tab's describe -- and the
travelled record answers for the citation it was composed from.

Every cited run here is made on the road: the junction described, its
template set as a person sets it, prepared and launched on the stand-in
engine.  Until 2026-10-08 these cases cited hand-written
`.xyz + .molstruct.json` pairs carrying a recorded contract, the citation
form that went with decision 7 (plan § 5x.5).
"""
from __future__ import annotations

import dataclasses

from molbuilder.config.siesta import SiestaConfig

#: What the relaxation's template is set to before it runs -- each
#: distinguishable from the catalogue's default (300 Ry / DZP / 1x1x1 /
#: 300 K), so a value that reaches the transport template was read, not
#: defaulted.
RELAXED_WITH = dict(mesh_cutoff=400.0, basis_size="TZP", kgrid=(4, 4, 1),
                    kgrid_displacement=(0.5, 0.25, 0.75),
                    electronic_temperature=350.0)


def _relaxed_run(tmp_path, monkeypatch, name="J", **values):
    """A finished relaxation of the test junction on the road, its template
    holding ``RELAXED_WITH`` and ``values`` -- the person's edit of the
    template file, through the template module's own doors -- and its
    citation: ``(tree root, citation)``."""
    from conftest import write_machine_record
    from molbuilder.template import (config_from_template, template_path,
                                     template_with_values)
    from support.road import describe_calculation, jobset
    from test_transport_compose import _junction
    write_machine_record()
    bundle = describe_calculation(tmp_path, monkeypatch, name=name,
                                  structure=_junction(), stage_strategy="")
    path = template_path(bundle, name)
    cfg = dataclasses.replace(
        config_from_template(path.read_text(), SiestaConfig),
        **RELAXED_WITH, **values)
    path.write_text(template_with_values(cfg, engine="siesta"))
    monkeypatch.setenv("MB_STAND_IN_LEAVES_XV", "1")
    for args in (("prep", "task", "--stage", "coarse", "--bundle",
                  str(bundle), "--target", "this"),
                 ("launch", "task", "--stage", "coarse", "--mode", "direct",
                  "--yes", "--bundle", str(bundle))):
        got = jobset(*args)
        assert got.exit_code == 0, got.output
    return tmp_path / "projects", f"P/optimization/{name}/01_coarse/run-0"


def test_a_cited_run_brings_its_settings_and_its_spin_on_both_roads(
        tmp_path, monkeypatch, web_client):
    """§ 3.1: the cited run's deck DEFAULTS the shared settings into this
    calculation's own template, on the CLI and on the tab alike.  Its spin
    treatment arrives written; a fixed count does not, since TranSIESTA
    cannot hold one (§ 3.1's spin note): it is left blank and floats.  The
    transport axis's k-point is forced to one whichever source answered
    (`_apply_kgrid`)."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    from molbuilder.template import one, read_template
    root, cite = _relaxed_run(tmp_path, monkeypatch,
                              spin_treatment="unrestricted",
                              unpaired_electrons=2)
    r = CliRunner().invoke(jobset_group, [
        "init", "--calculation", "transport", "--engine", "siesta",
        "--shape", "hierarchical", "--bundle", "P/transport/T",
        "--slot", f"junction={cite}"])
    assert r.exit_code == 0, r.output
    b = web_client.post("/api/transport/describe", json=dict(
        engine="siesta", name="T2", junction=cite, bias=[0.0]))
    assert b.status_code == 200, b.get_json()
    for tmpl in (read_template((root / "P/transport/T/T.template.toml")
                               .read_text()),
                 read_template(b.get_json()["files"][1]["text"])):
        assert one(tmpl, "mesh_cutoff").value == 400.0
        assert one(tmpl, "basis_size").value == "TZP"
        assert list(one(tmpl, "kgrid").value) == [4, 4, 1]
        assert one(tmpl, "electronic_temperature").value == 350.0
        assert one(tmpl, "spin_treatment").value == "unrestricted"
        assert one(tmpl, "unpaired_electrons").value is None


def test_a_charged_relaxation_is_refused_by_name_and_leaves_no_folder(
        tmp_path, monkeypatch, web_client):
    """ES7 (`science/chemistry-correctness.md` § 2a): a transport
    calculation's boundaries are open and the leads set the electron
    number, so a junction relaxed charged is not this one -- refused by
    name on both roads, and no half-described folder is left."""
    from click.testing import CliRunner
    from molbuilder.jobset._cli import jobset_group
    root, cite = _relaxed_run(tmp_path, monkeypatch, name="Q", net_charge=-1)
    r = CliRunner().invoke(jobset_group, [
        "init", "--calculation", "transport", "--engine", "siesta",
        "--shape", "hierarchical", "--bundle", "P/transport/Tq",
        "--slot", f"junction={cite}"])
    assert r.exit_code != 0 and "net charge of -1" in r.output, r.output
    assert not (root / "P/transport/Tq").exists(), (
        "a refused description left a folder behind")
    b = web_client.post("/api/transport/describe", json=dict(
        engine="siesta", name="Tq", junction=cite, bias=[0.0]))
    assert b.status_code == 400
    assert "net charge of -1" in b.get_json()["error"]


def test_the_record_answers_for_the_citation_it_was_composed_from(
        tmp_path, monkeypatch):
    """`load_compose_record`'s "no record" has causes that are not the same
    news, said in words: nothing composed here; a record composed from
    ANOTHER citation -- the likeliest on a travelled folder whose slot was
    re-pointed after a re-relaxation, and the one that read as "the record
    is not beside task.json" while a person stood on it."""
    from molbuilder.transport.compose import (compose_junction,
                                              load_compose_record,
                                              write_compose_record)
    root, cite = _relaxed_run(tmp_path, monkeypatch)
    rec = tmp_path / "calc"
    rec.mkdir()
    why: list = []
    assert load_compose_record(rec, citation=cite, tree_root=root,
                               why=why) is None
    assert "no slot-provenance.json" in why[0]
    write_compose_record(rec, compose_junction(cite, tree_root=root))
    other = cite.replace("run-0", "run-1")
    why = []
    assert load_compose_record(rec, citation=other, tree_root=root,
                               why=why) is None
    assert cite in why[0] and other in why[0]
    assert load_compose_record(rec, citation=cite, tree_root=root) is not None

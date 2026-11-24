"""A structure's row says its unit in one place, never two
(`model/structure.md` § 2.2d; the user, 2026-11-24: "we shouldn't allow
inconsistency in such important topic"; "when we have a quantity that has a
unit, then its unit is then blank").

A name that ends in a unit, as the units table spells it in a name
(`model/datafiles.md` § 2c), carries its unit and leaves ``unit`` blank; a
number whose name carries none states its unit, any spelling; a text or
true/false states none.  The rule holds where a row is MADE -- the script's
doors (`Structure.set_customized`, `Structure.with_frames`' frame rows) and
MolView's Info page at Apply, which asks the server's one rule
(`/api/structure/row`; the user's answer of 2026-11-24, "A"); a row in a
record already written is read as it is (the user's choice of 2026-11-24:
nothing molbuilder wrote stops opening) and carried as read by every copy --
the save, the download and the hand-over from any tab among them.
API-level, through the structure's and the web's own doors.

The silent failure it guards: a row whose name says one unit and whose
``unit`` says another, written -- a number nothing can read right; or a
record from last week no longer opening, or no longer leaving a tab.
"""
from __future__ import annotations

import json

import pytest

_H2 = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"

#: ROWS AN EARLIER MOLBUILDER WROTE (2.81.0's door kept any unit): a count
#: of no unit, a unit its name contradicts, a unit its name carries -- and a
#: frame's count of no unit, as 2.80.0's frame generator wrote it.
_WRITTEN = [{"name": "weight", "value": 0.5},
            {"name": "atom_a", "value": 2, "unit": "1"},
            {"name": "frequency_cm1", "value": 2548.37, "unit": "cm-1"}]
_WRITTEN_FRAME = [{"name": "node_sigma", "value": 0.0}]
#: ...as every reader reads them: the unit its name carries not kept.
_READ = [{"name": "weight", "value": 0.5},
         {"name": "atom_a", "value": 2, "unit": "1"},
         {"name": "frequency_cm1", "value": 2548.37}]


@pytest.mark.parametrize("name, value, unit, kept, refused", [
    ("displacement_ang", 0.1, None, None, None),
    ("displacement_ang", 0.1, "angstrom", None, None),
    ("displacement_ang", 0.1, "Å", None, None),
    ("force_ev_per_ang", 0.1, "eV/Å", None, None),
    ("temperature_k", 300.0, "K ", None, None),
    ("displacement_ang", 0.1, "nm", None,
     r"ends in _ang, which says its unit is 'angstrom'.*"
     r"\(displacement_nm\)"),
    ("lattice_a", 4.08, "Å", None,
     r"its unit is 'A' \(ampere\).*\(lattice_a_ang\)"),
    ("force_ev_per_ang", 0.1, None, None, None),
    ("weight", 0.5, "1", "1", None),
    ("time_fs", 2.0, None, None,
     r"states its unit in unit -- any spelling \('fs', 'deg'\), '1' for a "
     r"count"),
    ("weight", 0.5, " ", None, "a number whose name carries no unit"),
    ("torsion", 30.0, "deg", "deg", None),
    ("a", 3.905, "angstrom", "angstrom", None),
    ("k", 4, None, None, "a number whose name carries no unit"),
    ("x_per_ang", 2.0, None, None, "a number whose name carries no unit"),
    ("sampling_rule", "gauss-hermite", None, None, None),
    ("sampling_rule", "gauss-hermite", "1", None, None),
    ("sampling_rule", "gauss-hermite", "K", None, "a text has no unit"),
    ("relaxed", True, "K", None, "a true/false has no unit"),
], ids=["its unit in its name", "the same unit beside it, not kept",
        "the same unit as a page shows it, not kept",
        "a unit per another as a page shows it, not kept",
        "the same unit with a space, not kept",
        "another unit beside it, its rename named", "a one-letter ending "
        "read, said in words, "
        "its rename named from a page's look", "a unit per another, in its "
        "name", "a count's 1",
        "a number of no unit stating none, any spelling said",
        "a unit of spaces is none", "any spelling of a unit",
        "a bare name carries no unit", "a bare name of no unit stating none",
        "the inverse of a unit is none of the table's", "a text",
        "a text's 1, not kept", "a text with a unit",
        "a true/false with a unit"])
def test_a_row_made_says_its_unit_in_one_place(name, value, unit, kept,
                                               refused):
    """§ 2.2d: a row written through the script's door is kept with its
    unit where it belongs -- blank for a name that carries one, in any of
    the table's looks of it; the stated unit for a number of none, a bare
    name among them -- or refused by name with the fix.  The silent
    failures: a contradicting unit stored beside its name; a number stored
    with no unit at all; a lattice constant named ``a`` read as amperes."""
    from molbuilder.structure import Structure
    s = Structure.from_xyz(_H2)
    if refused is not None:
        with pytest.raises(ValueError, match=refused):
            s.set_customized(name, value, unit=unit)
        assert s.customized_rows() == []
        return
    s.set_customized(name, value, unit=unit)
    row = s.customized_rows()[0]
    assert row.get("unit") == kept and row["value"] == value, row


@pytest.mark.parametrize("row, kept", [
    ({"name": "weight", "value": 0.5}, None),
    ({"name": "displacement_ang", "value": 0.1, "unit": "angstrom"},
     {"name": "displacement_ang", "value": 0.1}),
], ids=["a frame's count of no unit, refused", "a frame's unit its name "
        "carries, not kept"])
def test_a_frame_set_built_says_its_frames_units_in_one_place(row, kept):
    """§ 2.2d, § 2.2e: `with_frames` makes each frame's rows, so they hold
    the rule as `set_customized`'s do.  The silent failure: a notebook's
    frame set whose weights state no unit, met only at a later save, or
    never."""
    from molbuilder.structure import Structure
    s = Structure.from_xyz(_H2)
    if kept is None:
        with pytest.raises(ValueError,
                           match="'weight'.*a number whose name carries no "
                                 "unit"):
            s.with_frames([s.positions], frame_rows=[[row]])
        return
    built = s.with_frames([s.positions], frame_rows=[[row]])
    assert built.customized_rows(frame=0) == [kept]


def test_rows_already_written_are_read_and_carried_as_written(tmp_path):
    """§ 2.2d (the user's choice of 2026-11-24): a row in a record already
    written -- a count an earlier molbuilder wrote with no unit, a unit its
    name contradicts -- is read as it is; the same unit beside its name is
    not kept; and molbuilder's own copy of the structure (the codec's write:
    the frame generator's, a command's) carries the rows as read.  The
    silent failures: a run, a result or a set made last week no longer
    opening; a frame set refused over a row its vibration only carried."""
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec
    whole = Structure.from_xyz(_H2).to_dict()
    whole["metadata"]["customized"] = {"rows": _WRITTEN}
    read = Structure.from_dict(whole)
    assert read.customized_rows() == _READ
    written = StructureCodec().write(read, tmp_path / "copy.xyz")
    assert StructureCodec().load(written).customized_rows() == _READ


@pytest.mark.parametrize("name, value, unit, answered", [
    ("weight", 0.5, None,
     "the named value ('weight'): a number whose name carries no unit"),
    ("bond_a", 0.96, "Å", "the name ends in _a, which says its unit is 'A' "
                          "(ampere)"),
    ("temperature_k", 300, "K", {"name": "temperature_k", "value": 300}),
    ("electrode", "left", "1", {"name": "electrode", "value": "left"}),
    ("n_layers", 3, "1", {"name": "n_layers", "value": 3, "unit": "1"}),
], ids=["a count stating no unit", "a unit its name contradicts",
        "the unit its name carries, not kept", "a text's 1, not kept",
        "a count's 1"])
def test_the_info_page_makes_a_row_by_the_one_rule(web_client, name, value,
                                                   unit, answered):
    """§ 2.2d: the Info page's Apply -- the one place a person makes a row
    in the browser -- asks the server's one rule (`/api/structure/row`): the
    row in its one form, or the rule's refusal in its words (400).  The
    silent failure: a row made in the browser whose unit contradicts its
    name, or a count stating none, carried into every file after it."""
    r = web_client.post("/api/structure/row", json={
        "name": name, "value": value, "unit": unit, "note": None})
    if isinstance(answered, str):
        assert r.status_code == 400, r.get_json()
        assert answered in r.get_json()["error"], r.get_json()
        return
    assert r.status_code == 200 and r.get_json()["row"] == answered, \
        r.get_json()


def _body(rows, frame_rows):
    """What a viewer sends a door: the envelope, its rows as the record
    holds them (`web-api.md` § 1)."""
    from support.envelope import envelope
    body = {"structure": envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])}
    body["structure"]["metadata"]["customized"] = {"rows": rows,
                                                   "frames": [frame_rows]}
    return body


@pytest.mark.parametrize("door", ["save", "download", "hand-over"])
def test_every_tab_carries_a_records_rows_as_written(isolated_projects_root,
                                                     web_client, door):
    """§ 2.2d (the user's answer of 2026-11-24): a record an earlier
    molbuilder wrote -- a frame set whose counts state no unit, a run's
    structure -- leaves any tab by the save, the download and the hand-over
    as it came, its rows as every reader reads them.  The silent failure:
    an older set no longer downloadable or savable from the Results tab,
    whose Info page cannot state a unit."""
    from molbuilder.projects import populate_project_skeleton
    from molbuilder.workingcopy_structure import StructureCodec
    from support.road import fresh_calc_dir
    if door == "save":
        populate_project_skeleton(isolated_projects_root / "proj",
                                  project_name="proj")
        pair = isolated_projects_root / "proj" / "structure" / "pair.xyz"
        route, extra = "/api/structure/save", {"path": str(pair)}
    elif door == "download":
        route, extra = "/api/structure/export", {"name": "pair"}
    else:
        folder = fresh_calc_dir(isolated_projects_root, species=["H"])
        route, extra = "/api/task-setup/handover", {
            "dest": str(folder), "engine": "siesta",
            "calculation": "optimization", "name": "probe calc",
            "params": {"system_label": "probe"}}
    r = web_client.post(route, json=dict(_body(_WRITTEN, _WRITTEN_FRAME),
                                          **extra))
    assert r.status_code == 200 and r.get_json()["ok"], r.get_json()
    if door == "download":
        sidecar = next(f["text"] for f in r.get_json()["files"]
                       if f["name"].endswith(".molstruct.json"))
        held = json.loads(sidecar)["customized"]
        rows, frame_rows = held["rows"], held["frames"][0]
    else:
        if door == "hand-over":
            from molbuilder.runfiles import TASK_HANDOVER_FILE
            source = json.loads((folder / TASK_HANDOVER_FILE).read_text())
            pair = folder / source["structure"]["source"]
        read = StructureCodec().load(pair)
        rows, frame_rows = read.customized_rows(), read.customized_rows(
            frame=0)
    assert rows == _READ and frame_rows == _WRITTEN_FRAME, (rows, frame_rows)

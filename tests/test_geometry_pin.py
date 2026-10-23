"""The pin, through the read door -- the runner of
``tests/data/geometry_pin.toml`` (``docs/model/structure-molstruct.md`` § 3).

Every row writes the table's ``[pair]`` through ``StructureCodec.write``,
as a project save does -- or its ``[remapped]`` pair, sorted for an engine
and written through ``write(..., engine_record=True)``, the composer's one
door -- makes the one edit the row names to the files on disk, and reads the
pair back through one door.

Keys of a ``[[case]]``:

``pair``
    ``remapped`` -- the ``[remapped]`` pair, sorted by its ``key``
    (``transport.sort.sort_by``) and written as an engine's record.
``plain``
    the pair is the molecule alone -- no cell, no axis kinds, no labels.
``format``
    ``xyz`` (the default) or ``pdb``: the geometry file the pair is written as.
``atom_line``
    ``{index, line}``: the ``index``-th atom line of the geometry file (0-based,
    in file order) replaced by ``line`` -- an atom moved by hand.
``comment``
    the XYZ comment line replaced by this text -- a title edited by hand.
``sidecar``
    keys whose values replace the sidecar's -- the cell, the axis kinds, a
    label set edited by hand.
``as_v12``
    the sidecar rewritten as a molbuilder before v13 wrote it: schema 12, its
    pin the retired ``structure_hash``, no ``geometry_hash``, no
    ``atom_index_map``.
``as_v13``
    the sidecar rewritten as a molbuilder before v14 wrote it: schema 13, no
    ``atom_order``, no ``atom_index_map``.
``marked``
    ``true``: the pair as a viewer of a run whose deck states no order holds
    it -- ``order_unstated`` (`model/overview.md` § 2.2b, F4).
``stated``
    the ``atoms_listed_by`` the written pair's sidecar states; and the
    pair's read then says ``read_said``.
``door``
    ``load`` -- ``StructureCodec().load(path)``, the door every builder reads
    through; ``route_load`` -- the pair in a project, then ``POST
    /api/build/load`` with ``{path}``, the browser's one load door;
    ``write`` -- the pair written for a person, ``StructureCodec().write``;
    ``route_save`` / ``route_export`` -- the structure's envelope sent to
    ``POST /api/structure/save`` / ``/api/structure/export``, as a viewer
    sends it, and the pair the route wrote or answered read back.
``sidecar_written``
    the pair's sidecar is on disk beside the geometry.
``refused``
    the door raises ``GeometryChangedError`` with this text in its message;
    otherwise it opens, with the pair's labels applied, in the user order -- or,
    for the ``remapped`` pair, carrying ``order``.
``unreadable``
    the door raises ``ValueError`` -- an unreadable sidecar's
    ``MolstructJsonError`` among them -- its message holding each of these
    words, what the contract says of the refusal.
``order``
    ``true``: the opened ``remapped`` pair -- the road's CO2 in its engine's
    order (`support.road.co2_in_engine_order`) -- carries the road's order
    record (`support.road.CO2_HELD_FIRST`); restored, it is the pair as
    built.  A ``remapped`` row without it opens as built, in the user order.
``body``
    what a ``route_load`` row adds to ``{path}`` -- ``restore``, an editor's
    open.
``said``
    ``{where, severity, words}``: the route's answer carries this notice,
    its message holding each of ``words`` -- what the contract says it
    names; a ``route_load`` row without it carries none of its kinds (a
    changed geometry, a restore, a pair that states no order).
``read_said``
    ``{where, severity, words}``: the codec's read of the written pair says
    this notice, as ``said``.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from molbuilder.sidecars import molstruct
from molbuilder.structure import Structure
from molbuilder.workingcopy_structure import (GEOMETRY_CHANGED,
                                              ORDER_NOT_STATED, RESTORED,
                                              StructureCodec)
from support.road import (CO2, CO2_HELD_FIRST, co2_in_engine_order,
                          road_structure)
from molbuilder.structure_pair import GeometryChangedError

_TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "geometry_pin.toml").read_text())
_PAIR = _TABLE["pair"]
_CASES = _TABLE["case"]
#: The notices a ``route_load`` row reads: a changed geometry, a restore, a
#: pair that states no order (F4).
_SAID = (GEOMETRY_CHANGED, RESTORED, ORDER_NOT_STATED)


def _the_pair(plain: bool) -> Structure:
    """The table's pair -- or, ``plain``, its atoms alone."""
    atoms = dict(elements=list(_PAIR["elements"]),
                 positions=np.asarray(_PAIR["positions"], dtype=float),
                 title=_PAIR["title"])
    if plain:
        return Structure(**atoms)
    return Structure(**atoms, cell=np.asarray(_PAIR["cell"], dtype=float),
                     axis_kind=tuple(_PAIR["axis_kind"]),
                     regions=dict(_PAIR["regions"]))


def _opened_as_built(case, got: Structure) -> None:
    """The remapped pair, opened: carrying the road's order record and,
    restored, the road's CO2 as built -- or, a row without ``order``, as
    built already."""
    if case.get("order"):
        assert got.atom_order.to_dict() == CO2_HELD_FIRST, got.atom_order
        got = got.restored()
    want = road_structure(CO2)
    assert (got.elements, got.regions, got.atom_order) == (
        want.elements, want.regions, None), got
    assert np.allclose(got.positions, want.positions)


def _says_unreadable(case, refusal) -> None:
    """The refusal holds each of the row's ``unreadable`` words."""
    assert all(w in str(refusal) for w in case["unreadable"]), str(refusal)


def _said(case, answer) -> None:
    """The route's answer carries the row's ``said`` notice, and no other of
    the kinds a row reads."""
    notices = [n for n in answer.get("notices", []) if n["where"] in _SAID]
    want = case.get("said")
    assert [(n["where"], n["severity"]) for n in notices] == (
        [(want["where"], want["severity"])] if want else []), notices
    if want:
        assert all(w in notices[0]["message"] for w in want["words"]), notices


def _written_by_the_route(case, built: Structure, path: Path,
                          client) -> None:
    """``built`` sent as the browser sends it -- its envelope -- to the save
    route (written at ``path``) or the export route (its files put at
    ``path``'s stem), the answer's notices read, and the pair read back."""
    if case["door"] == "route_save":
        answer = client.post("/api/structure/save", json={
            "structure": built.to_dict(), "path": str(path)})
    else:
        answer = client.post("/api/structure/export", json={
            "structure": built.to_dict(), "name": path.stem})
    assert answer.status_code == 200, answer.get_json()
    for f in answer.get_json().get("files", []):
        (path.parent / f["name"]).write_text(f["text"])
    _said(case, answer.get_json())
    if "stated" in case:
        side = json.loads(molstruct.sidecar_path_for(path).read_text())
        assert side["atom_index_map"]["atoms_listed_by"] == case["stated"], (
            side["atom_index_map"])
        said: list = []
        got = StructureCodec().load(path, said_out=said)
        _said({"said": case.get("read_said")},
              {"notices": [{"where": n.where, "severity": n.severity,
                            "message": n.message} for n in said]})
        assert got.order_unstated is False, "a read of a file leaves it false"
        assert (got.elements, got.regions) == (built.elements, built.regions)
        return
    _opened_as_built(case, StructureCodec().load(path))


def _edited(case, path: Path) -> None:
    """The row's edits, made to the files as a person makes them."""
    lines = path.read_text().split("\n")
    if "atom_line" in case:
        if path.suffix == ".pdb":
            where = [i for i, ln in enumerate(lines)
                     if ln.startswith(("ATOM  ", "HETATM"))]
        else:
            where = list(range(2, 2 + int(lines[0])))
        lines[where[case["atom_line"]["index"]]] = case["atom_line"]["line"]
    if "comment" in case:
        lines[1] = case["comment"]
    path.write_text("\n".join(lines))
    side = molstruct.sidecar_path_for(path)
    payload = json.loads(side.read_text())
    payload.update(case.get("sidecar", {}))
    if case.get("as_v12"):
        del payload["geometry_hash"], payload["atom_index_map"]
        payload.update(schema_version=12, structure_hash="0" * 64)
    if case.get("as_v13"):
        del payload["atom_order"], payload["atom_index_map"]
        payload.update(schema_version=13)
    side.write_text(json.dumps(payload))


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_the_pin_through_the_read_door(case, tmp_path, monkeypatch,
                                       web_client):
    remapped = case.get("pair") == "remapped"
    if remapped:
        built = co2_in_engine_order()
    else:
        built = _the_pair(case.get("plain", False))
    if case.get("marked"):
        built.order_unstated = True
    name = "co." + case.get("format", "xyz")
    if case["door"].startswith("route_"):
        from conftest import _point_the_one_door_at
        tree = _point_the_one_door_at(tmp_path / "tree", monkeypatch.setenv)
        (tree / "P").mkdir()
        path = tree / "P" / name
    else:
        path = tmp_path / name
    if case["door"] == "write":
        with pytest.raises(ValueError) as refused:
            StructureCodec().write(built, path)
        _says_unreadable(case, refused.value)
        return
    if case["door"] in ("route_save", "route_export"):
        _written_by_the_route(case, built, path, web_client)
        return
    StructureCodec().write(built, path, engine_record=remapped)
    if case.get("sidecar_written"):
        assert molstruct.sidecar_path_for(path).is_file(), (
            sorted(p.name for p in path.parent.iterdir()))
    _edited(case, path)

    if case["door"] == "route_load":
        answer = web_client.post(
            "/api/build/load", json={"path": str(path), **case.get("body", {})})
        assert answer.status_code == 200, answer.get_json()
        _said(case, answer.get_json())
        if remapped:
            _opened_as_built(case, Structure.from_dict(
                answer.get_json()["structure"]))
        return

    if "refused" in case:
        with pytest.raises(GeometryChangedError,
                           match=case["refused"].replace(".", r"\.")):
            StructureCodec().load(path)
        return
    if "unreadable" in case:
        with pytest.raises(ValueError) as refused:
            StructureCodec().load(path)
        _says_unreadable(case, refused.value)
        return
    got = StructureCodec().load(path)
    if remapped:
        _opened_as_built(case, got)
        return
    assert got.atom_order is None, got.atom_order
    assert got.regions == ({} if case.get("plain") else
                           dict(case.get("sidecar", {}).get(
                               "regions", _PAIR["regions"]))), got.regions

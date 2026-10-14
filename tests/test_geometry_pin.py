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
    pin the retired ``structure_hash``, no ``geometry_hash``.
``as_v13``
    the sidecar rewritten as a molbuilder before v14 wrote it: schema 13, no
    ``atom_order``.
``door``
    ``load`` -- ``StructureCodec().load(path)``, the door every builder reads
    through; ``route_load`` -- the pair in a project, then ``POST
    /api/build/load`` with ``{path}``, the browser's one load door;
    ``write`` -- the pair written for a person, ``StructureCodec().write``.
``sidecar_written``
    the pair's sidecar is on disk beside the geometry.
``refused``
    the door raises ``GeometryChangedError`` with this text in its message;
    otherwise it opens, with the pair's labels applied, in your order -- or,
    for the ``remapped`` pair, carrying ``order``.
``unreadable``
    the door raises ``ValueError`` -- an unreadable sidecar's
    ``MolstructJsonError`` among them -- with this text in its message.
``order``
    the order record the opened structure carries, ``{key,
    sorted_to_original, original_to_sorted}``; restored, it is the
    ``[remapped]`` pair as built.
``said``
    ``{where, severity, words}``: the route's answer carries this notice; a
    ``route_load`` row without it carries none of its kind.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from molbuilder.sidecars import molstruct
from molbuilder.structure import Structure
from molbuilder.workingcopy_structure import StructureCodec

_TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "geometry_pin.toml").read_text())
_PAIR = _TABLE["pair"]
_REMAPPED = _TABLE["remapped"]
_CASES = _TABLE["case"]
_NOTICE = "structure.geometry_changed"


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


def _the_remapped() -> Structure:
    """The table's ``[remapped]`` pair as built, in your order."""
    return Structure(elements=list(_REMAPPED["elements"]),
                     positions=np.asarray(_REMAPPED["positions"], dtype=float),
                     regions=dict(_REMAPPED["regions"]))


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
        del payload["geometry_hash"]
        payload.update(schema_version=12, structure_hash="0" * 64)
    if case.get("as_v13"):
        del payload["atom_order"]
        payload.update(schema_version=13)
    side.write_text(json.dumps(payload))


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_the_pin_through_the_read_door(case, tmp_path, monkeypatch):
    remapped = case.get("pair") == "remapped"
    if remapped:
        from molbuilder.transport.sort import sort_by
        built = sort_by(_the_remapped(), _REMAPPED["key"]).structure
    else:
        built = _the_pair(case.get("plain", False))
    name = "co." + case.get("format", "xyz")
    if case["door"] == "route_load":
        from conftest import _point_the_one_door_at
        tree = _point_the_one_door_at(tmp_path / "tree", monkeypatch.setenv)
        (tree / "P").mkdir()
        path = tree / "P" / name
    else:
        path = tmp_path / name
    if case["door"] == "write":
        with pytest.raises(ValueError, match=re.escape(case["unreadable"])):
            StructureCodec().write(built, path)
        return
    StructureCodec().write(built, path, engine_record=remapped)
    if case.get("sidecar_written"):
        assert molstruct.sidecar_path_for(path).is_file(), (
            sorted(p.name for p in path.parent.iterdir()))
    _edited(case, path)

    if case["door"] == "route_load":
        from molbuilder.web.app import create_app
        answer = create_app(config={}).test_client().post(
            "/api/build/load", json={"path": str(path)})
        assert answer.status_code == 200, answer.get_json()
        notices = [n for n in answer.get_json().get("notices", [])
                   if n["where"] == _NOTICE]
        want = case.get("said")
        if want is None:
            assert notices == [], notices
            return
        assert [(n["severity"], want["words"] in n["message"])
                for n in notices] == [(want["severity"], True)], notices
        return

    if "refused" in case:
        with pytest.raises(molstruct.GeometryChangedError,
                           match=case["refused"].replace(".", r"\.")):
            StructureCodec().load(path)
        return
    if "unreadable" in case:
        with pytest.raises(ValueError, match=re.escape(case["unreadable"])):
            StructureCodec().load(path)
        return
    got = StructureCodec().load(path)
    if remapped:
        assert got.atom_order.to_dict() == case["order"], got.atom_order
        back, want = got.restored(), _the_remapped()
        assert (back.elements, back.regions, back.atom_order) == (
            want.elements, want.regions, None), back
        assert np.allclose(back.positions, want.positions)
        return
    assert got.atom_order is None, got.atom_order
    assert got.regions == ({} if case.get("plain") else
                           dict(case.get("sidecar", {}).get(
                               "regions", _PAIR["regions"]))), got.regions

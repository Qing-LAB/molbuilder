"""A frame set, down its road -- the runner of ``tests/data/frame_sets.toml``.

A frame set is made by scripts through the structure's own doors
(``docs/model/structure.md`` § 2.2e), so those doors are its road: every row
builds the table's one ``[set]`` with ``with_frames`` and ``set_customized``,
as a script does, and asks one door of it.

Keys of a ``[[case]]``:

``door``
    ``save_load`` -- ``StructureCodec.write`` then ``load(**load)``;
    ``route_load`` -- the pair saved into a project, then ``POST
    /api/build/load`` with ``{path, **body}``;
    ``viewer`` -- the route's answer to ``{path, frames: true}`` installed in
    a MolView model of ``mode`` under node, then -- unless refused -- its
    ``exportFile`` of every frame posted to ``/api/structure/export`` and the
    files read back with ``load(frames=True)``;
    ``take`` -- ``Structure.take(order)``;
    ``sorted_restored`` -- ``transport.sort.sort_by(set, SORT_HELD_FIRST)``,
    the set in an engine's order, then its ``restored()``;
    ``edit`` -- the edit named by ``edit`` (``_EDITS``);
    ``read_document`` -- ``Structure.from_xyz(document)``;
    ``lone_file`` -- ``document`` written alone, no sidecar, and read through
    every door that opens a file for a person (``structure.md`` § 2.3):
    ``StructureCodec.load(..., said_out=[...])``, which ``jobset init`` and
    the CLI print from, and ``POST /api/build/load`` with ``{path}`` and with
    ``{text, filename}`` -- each answering the same, in the same sentence.
``held``
    the set's held atoms, in place of the table's.
``refused``
    the door raises (or the route answers 400) with this text in its message.
``expect``
    what comes back -- each key read by :func:`_check`.
"""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from molbuilder import modify
from molbuilder.structure import AtomChannel, Structure
from molbuilder.workingcopy_structure import StructureCodec

_TABLE = tomllib.loads(
    (Path(__file__).parent / "data" / "frame_sets.toml").read_text())
_SET = _TABLE["set"]
_CASES = _TABLE["case"]


def _the_set(held=None) -> Structure:
    """The table's set, built the way a script builds one -- ``held`` its
    held atoms, when a row names its own."""
    frames = np.asarray(_SET["frames"], dtype=float)
    regions = (dict(_SET["regions"]) if held is None
               else {"frozen_atoms": list(held)})
    base = Structure(elements=list(_SET["elements"]), positions=frames[0],
                     cell=_SET["cell"], regions=regions)
    built = base.with_frames(frames, frame_rows=_SET["frame_rows"])
    for row in _SET["rows"]:
        built.set_customized(row["name"], row["value"], unit=row.get("unit"),
                             note=row.get("note"))
    for name, values in (_SET.get("channels") or {}).items():
        built.set_channel(name, AtomChannel(
            "value", {i: float(v) for i, v in enumerate(values)}))
    for key, value in (_SET.get("info") or {}).items():
        built.set_info(key, value)
    return built


_EDITS = {
    "translated":        lambda s: s.translated([0.0, 0.0, 0.1]),
    "translate_some":    lambda s: modify.translate(s, [0.0, 0.0, 0.1],
                                                    indices=[1]),
    "delete_atoms":      lambda s: modify.delete_atoms(s, [1]),
    "add_atom":          lambda s: modify.add_atom(s, "H", 0, [0.0, 1.0, 0.0]),
    "append_structure":  lambda s: modify.append_structure(s.frame_at(0), s),
    "replace_positions": lambda s: s.replace(positions=s.positions + 0.1),
}


def _route_load(tmp_path, monkeypatch, built, body, client=None):
    from conftest import _point_the_one_door_at
    from molbuilder.web.app import create_app
    tree = _point_the_one_door_at(tmp_path / "tree", monkeypatch.setenv)
    (tree / "P").mkdir()
    path = tree / "P" / "set.xyz"
    StructureCodec().write(built, path)
    client = client or create_app(config={}).test_client()
    answer = client.post("/api/build/load", json={"path": str(path), **body})
    return answer.status_code, answer.get_json()


_MODEL = (Path(__file__).resolve().parents[1]
          / "molbuilder/web/static/lib/molview/model.js")


def _viewer(answer, mode):
    """The route's answer installed in a MolView model under node; the
    stand-in server answers only the body the model sends for a file it
    names no frame of -- the whole file (`molview.md` § 9.3)."""
    from tests._node_esm import run_node
    stand_in = (
        "globalThis.fetch = async (route, init) => {\n"
        "  const body = JSON.parse(init.body);\n"
        "  const ok = route === '/api/build/load' && body.frames === true;\n"
        "  return { ok: ok, status: ok ? 200 : 400, json: async () => (ok ? "
        + json.dumps(answer)
        + " : { ok: false, error: 'not the whole file: ' + init.body }) };\n"
        "};\n")
    return run_node([], (
        "const { createModel } = await import("
        + json.dumps(_MODEL.as_uri()) + ");\n"
        "const v = createModel({ mode: " + json.dumps(mode) + " });\n"
        "let out;\n"
        "try {\n"
        "  await v.installMolecule({ path: 'P/set.xyz' });\n"
        "  out = { export: v.exportFile({ from: 0, to: v.frameCount() - 1 }) };\n"
        "} catch (e) { out = { refused: e.message }; }\n"
        "console.log(JSON.stringify(out));\n"), globals_js=stand_in)


def _check(expect, got: Structure, built: Structure, *, written=None,
           answer=None, said=None) -> None:
    for key, want in expect.items():
        if key == "same_set":
            assert got.n_frames == built.n_frames
            assert np.allclose(got.frames, built.frames)
            assert got.customized == built.customized
            assert got.regions == built.regions
            assert np.allclose(got.cell, built.cell)
        elif key == "sidecar":
            from molbuilder.sidecars import molstruct
            payload = molstruct.load(molstruct.sidecar_path_for(written))
            for k, v in want.items():
                assert payload[k] == v, (k, payload[k])
        elif key == "n_frames":
            assert got.n_frames == want
        elif key == "positions":
            assert np.allclose(got.positions, want)
        elif key == "rows":
            assert got.customized_rows() == want
        elif key == "frame_rows":
            assert got.customized_rows(frame=0) == want
        elif key == "engine_offset":
            if want == "none":
                assert got.engine_offset is None
            else:
                assert np.allclose(got.engine_offset, want, atol=1e-9)
        elif key == "n_frames_held":
            assert answer["n_frames"] == want
        elif key == "envelope_frames":
            assert len(answer["structure"]["frames"]) == want
            assert "positions" not in answer["structure"]
        elif key == "atom_z_per_frame":
            assert np.allclose(got.frames[:, :, 2].T, want)
        elif key == "regions":
            assert got.regions == want
        elif key == "rows_unchanged":
            assert got.customized == built.customized
        elif key == "cell":
            assert (got.cell is None) if want == "none" else \
                np.allclose(got.cell, want)
        elif key == "axis_kind":
            assert list(got.axis_kind) == want
        elif key == "said":
            assert said and want in said[0], said
        else:
            raise KeyError(f"frame_sets.toml: no check for {key!r}")


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_frame_set(case, tmp_path, monkeypatch):
    built = _the_set(case.get("held"))
    door = case["door"]
    refused = case.get("refused")
    expect = case.get("expect", {})

    if door == "viewer":
        from molbuilder.web.app import create_app
        client = create_app(config={}).test_client()
        status, answer = _route_load(tmp_path, monkeypatch, built,
                                     {"frames": True}, client=client)
        assert status == 200, answer
        out = _viewer(answer, case["mode"])
        if refused:
            assert refused in out.get("refused", ""), out
            return
        assert "export" in out, out
        files = client.post("/api/structure/export", json={
            "structure": out["export"]["structure"], "name": "back"}
        ).get_json()
        assert files["ok"], files
        for f in files["files"]:
            (tmp_path / f["name"]).write_text(f["text"])
        got = StructureCodec().load(tmp_path / "back.xyz", frames=True)
        _check(expect, got, built)
        return

    if door == "route_load":
        status, answer = _route_load(tmp_path, monkeypatch, built,
                                     case.get("body", {}))
        if refused:
            assert status == 400 and refused in answer["error"], answer
            return
        assert status == 200, answer
        got = Structure.from_dict(answer["structure"])
        _check(expect, got, built, answer=answer)
        return

    if door == "save_load":
        written = StructureCodec().write(built, tmp_path / "set.xyz")
        act = lambda: StructureCodec().load(written, **case.get("load", {}))  # noqa: E731
    elif door == "take":
        written = None
        act = lambda: built.take(case["order"])  # noqa: E731
    elif door == "sorted_restored":
        from molbuilder.transport.sort import SORT_HELD_FIRST, sort_by
        written = None
        in_engine_order = sort_by(built, SORT_HELD_FIRST).structure
        assert in_engine_order.atom_order is not None, "the sort moved nothing"
        act = in_engine_order.restored
    elif door == "edit":
        written = None
        act = lambda: _EDITS[case["edit"]](built)  # noqa: E731
    elif door == "read_document":
        written = None
        act = lambda: Structure.from_xyz(case["document"])  # noqa: E731
    elif door == "lone_file":
        from conftest import _point_the_one_door_at
        from molbuilder.web.app import create_app
        tree = _point_the_one_door_at(tmp_path / "tree", monkeypatch.setenv)
        (tree / "P").mkdir()
        lone = tree / "P" / "lone.xyz"
        lone.write_text(case["document"])
        from molbuilder.workingcopy_structure import LONE_FILE
        read_said: list = []
        got = StructureCodec().load(lone, said_out=read_said)
        assert [(s.where, s.severity) for s in read_said] == [
            (LONE_FILE, "info")], read_said
        said = [s.message for s in read_said]
        _check(expect, got, built, said=said)
        client = create_app(config={}).test_client()
        for body in ({"path": str(lone)},
                     {"text": case["document"], "filename": lone.name}):
            answer = client.post("/api/build/load", json=body).get_json()
            assert answer["ok"], answer
            notices = [n["message"] for n in answer.get("notices", [])
                       if n["where"] == LONE_FILE]
            assert notices == said, (body, notices)
            _check(expect, Structure.from_dict(answer["structure"]), built,
                   said=notices)
        return
    else:
        raise KeyError(f"frame_sets.toml: no door {door!r}")

    if refused:
        with pytest.raises(ValueError, match=re.escape(refused)):
            act()
        return
    _check(expect, act(), built, written=written)

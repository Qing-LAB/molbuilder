"""The checkpoint HTTP surface (§ 5's verbs, over the wire).

**Contract:** [`checkpointing.md`](?doc=execution/checkpointing.md) § 5 · § 7.1 ·
A4 · A5 · L3 · L4 · S1c. Envelope: [`web-api.md`](?doc=web/web-api.md) § 1.

**Why this file exists separately from the CLI's.** The same `Repo` sits under
both, so a rule proved once is proved — but a *surface* can drop a refusal, add
a parameter the contract removed, or default something that must be asked. Those
are the failures this looks for, and none of them are visible from the module.

Real filesystem, real git, no mocks.
"""
from __future__ import annotations

import pytest

from molbuilder.web.app import create_app

BIG = b"\x01" * 5000


@pytest.fixture()
def calc(tmp_path, checkpoint_config):
    """The classification is server-wide (S1c); the folder carries none."""
    checkpoint_config(size_limit_bytes=1024, engines={"generic": []})
    root = tmp_path / "BDT_Au_relax"
    root.mkdir()
    (root / "job.fdf").write_text("SystemLabel job\n")
    return root


@pytest.fixture()
def client():
    # ``config={}`` is the no-auth local-server shape the other route tests
    # use; the checkpoint routes are not where auth is exercised.
    return create_app(config={}).test_client()


def _get(client, route, calc, **params):
    query = "&".join([f"path={calc}"] + [f"{k}={v}" for k, v in params.items()])
    return client.get(f"/api/checkpoint/{route}?{query}")


def _post(client, route, calc, **body):
    return client.post(f"/api/checkpoint/{route}",
                       json={"path": str(calc), **body})


@pytest.fixture()
def started(client, calc):
    _post(client, "init", calc, note="set up")
    return calc


# ------------------------------------------------------------------ #
#  The route set is the contract's verbs, and nothing retired         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("route", ["commit", "branch", "migrate-manifest"])
def test_a_retired_verb_has_no_route(client, started, route):
    """`commit` became `save`; `branch` and `migrate-manifest` were removed.

    A route left mounted is a route somebody's script calls, whatever the
    documentation says.
    """
    assert _post(client, route, started, note="x").status_code == 404


def test_there_is_no_route_that_edits_the_classification(client, started):
    """S1c: the classification has ONE home, and it is molbuilder.json.

    A per-folder editor is what let two folders behave differently for no
    recorded reason — and let somebody change the rules between a save and a
    restore, which is the hazard I2c is about.
    """
    assert _post(client, "config", started,
                 archive_globs=["*.DM"]).status_code in (404, 405)


# ------------------------------------------------------------------ #
#  A4 — no partial restore on this surface either                     #
# ------------------------------------------------------------------ #


def test_include_binaries_is_refused_rather_than_ignored(client, started):
    """It rewound the text and left every big file — a folder no save ever
    held — and skipped the archive verification on the way.

    Refused, not silently dropped: a caller that still sends it believes it is
    getting a text-only restore, and quietly giving them a full one is a
    different operation than the one they asked for.
    """
    state = _get(client, "list", started).get_json()["states"][0]["id"]
    r = _post(client, "restore", started, state=state, include_binaries=False)
    assert r.status_code == 400
    assert "A4" in r.get_json()["error"] or "whole folder" in r.get_json()["error"]


# ------------------------------------------------------------------ #
#  state — where you stand, and what is unsaved                       #
# ------------------------------------------------------------------ #


def test_state_reports_where_the_folder_stands(client, started):
    body = _get(client, "state", started).get_json()
    assert body["ok"] and body["initialized"]
    assert body["standing_at"]["note"] == "set up"
    assert body["clean"] and body["unsaved"] == []


def test_state_names_the_three_shapes(client, started, calc):
    (calc / "keep.txt").write_text("a\n")
    (calc / "gone.txt").write_text("b\n")
    _post(client, "save", started, note="two files")
    (calc / "keep.txt").write_text("edited\n")
    (calc / "fresh.bin").write_bytes(BIG)
    (calc / "gone.txt").unlink()

    body = _get(client, "state", started).get_json()
    assert not body["clean"]
    assert "keep.txt" in body["changed"]
    assert "fresh.bin" in body["added"]
    assert "gone.txt" in body["deleted"]


def test_state_on_a_plain_folder_says_it_is_not_initialised(client, calc):
    body = _get(client, "state", calc).get_json()
    assert body["ok"] and body["initialized"] is False


def test_state_refuses_a_path_that_is_not_a_directory(client, tmp_path):
    assert _get(client, "state", tmp_path / "nope").status_code == 400


# ------------------------------------------------------------------ #
#  save — L3's note is required, not defaulted                        #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("body", [{}, {"note": ""}, {"note": "   "}])
def test_save_without_a_note_is_refused(client, started, calc, body):
    (calc / "job.XV").write_text("x\n")
    r = _post(client, "save", started, **body)
    assert r.status_code == 400
    assert "note" in r.get_json()["error"]


def test_save_returns_the_state_it_made(client, started, calc):
    (calc / "job.XV").write_text("x\n")
    body = _post(client, "save", started,
                 note="stage 1 converged, 41 steps").get_json()
    assert body["ok"] and body["changed"]
    assert body["state"]["note"] == "stage 1 converged, 41 steps"
    assert body["state"]["calculation"] == "BDT_Au_relax"
    assert body["state"]["archive"], "every state names its archive (I2b)"


def test_save_says_when_nothing_changed_rather_than_inventing_a_state(
        client, started):
    body = _post(client, "save", started, note="nothing happened").get_json()
    assert body["ok"] and body["changed"] is False and body["state"] is None


# ------------------------------------------------------------------ #
#  list — § 7.1's fork must be visible over the wire                  #
# ------------------------------------------------------------------ #


def test_list_shows_a_fork_as_two_states_from_one_parent(client, started, calc):
    (calc / "job.XV").write_text("one\n")
    stage1 = _post(client, "save", started, note="stage 1").get_json()["state"]
    (calc / "job.XV").write_text("200\n")
    _post(client, "save", started, note="attempt A")
    _post(client, "restore", started, state=stage1["id"])
    (calc / "job.XV").write_text("300\n")
    _post(client, "save", started, note="attempt B")

    body = _get(client, "list", started).get_json()
    parents = [s["parent"] for s in body["states"] if s["note"].startswith("attempt")]
    assert parents == [stage1["id"], stage1["id"]], (
        "the panel cannot draw alternatives unless both name the same parent")


def test_list_carries_tags_and_where_you_stand(client, started, calc):
    (calc / "job.XV").write_text("x\n")
    state = _post(client, "save", started, note="stage 1").get_json()["state"]
    _post(client, "tag", started, name="stage1-good", note="trusted")
    body = _get(client, "list", started).get_json()
    assert body["standing_at"]["id"] == state["id"], (
        "/list and /state must name the same field the same way")
    assert [t["name"] for t in body["tags"]] == ["stage1-good"]
    assert "stage1-good" in [t for s in body["states"] for t in s["tags"]]


def test_list_on_a_plain_folder_is_a_protocol_error(client, calc):
    assert _get(client, "list", calc).status_code == 409


# ------------------------------------------------------------------ #
#  restore — A5 as an advisory, force as the way through              #
# ------------------------------------------------------------------ #


def test_restore_refuses_unsaved_work_and_names_every_file(client, started,
                                                           calc):
    (calc / "job.XV").write_text("saved\n")
    state = _post(client, "save", started, note="stage 1").get_json()["state"]
    (calc / "job.XV").write_text("unsaved\n")
    (calc / "new.bin").write_bytes(BIG)

    r = _post(client, "restore", started, state=state["id"])
    assert r.status_code == 200, "a refusal the user can act on is an advisory"
    body = r.get_json()
    assert body["ok"] is False
    assert "job.XV" in body["error"] and "new.bin" in body["error"]
    assert body["errors_only"], "the canonical envelope shape (web-api.md § 1)"


def test_restore_with_force_completes(client, started, calc):
    (calc / "job.XV").write_text("saved\n")
    state = _post(client, "save", started, note="stage 1").get_json()["state"]
    (calc / "job.XV").write_text("unsaved\n")
    body = _post(client, "restore", started,
                 state=state["id"], force=True).get_json()
    assert body["ok"] and body["state"]["id"] == state["id"]
    assert (calc / "job.XV").read_text() == "saved\n"


def test_restore_accepts_a_tag(client, started, calc):
    (calc / "job.XV").write_text("good\n")
    _post(client, "save", started, note="the geometry I trust")
    _post(client, "tag", started, name="stage1-good", note="trusted")
    (calc / "job.XV").write_text("elsewhere\n")
    _post(client, "save", started, note="elsewhere")

    assert _post(client, "restore", started,
                 state="stage1-good").get_json()["ok"]
    assert (calc / "job.XV").read_text() == "good\n"


def test_restore_of_an_unknown_state_is_a_404(client, started):
    assert _post(client, "restore", started,
                 state="no-such-state").status_code == 404


def test_restore_without_a_state_is_a_protocol_error(client, started):
    assert _post(client, "restore", started).status_code == 400


# ------------------------------------------------------------------ #
#  tag — L4, and the note that makes it worth having                  #
# ------------------------------------------------------------------ #


def test_a_tag_needs_a_name_and_a_note(client, started):
    assert _post(client, "tag", started, note="why").status_code == 400
    assert _post(client, "tag", started, name="x").status_code == 400


def test_nothing_tags_a_state_on_your_behalf(client, started, calc):
    for i in range(3):
        (calc / "job.XV").write_text(f"{i}\n")
        _post(client, "save", started, note=f"stage {i} converged")
    assert _get(client, "list", started).get_json()["tags"] == []


# ------------------------------------------------------------------ #
#  config — read only, and it says where to edit                      #
# ------------------------------------------------------------------ #


def test_config_reports_the_limit_and_where_it_lives(client, started):
    body = _get(client, "config", started).get_json()
    assert body["size_limit_bytes"] == 1024
    assert body["edit_in"] == "molbuilder.json"
    assert body["calculation"] == "BDT_Au_relax"


# ------------------------------------------------------------------ #
#  init                                                               #
# ------------------------------------------------------------------ #


def test_init_is_idempotent_and_says_so(client, calc):
    first = _post(client, "init", calc, note="set up").get_json()
    assert first["ok"] and first["already"] is False
    second = _post(client, "init", calc, note="set up").get_json()
    assert second["ok"] and second["already"] is True


def test_init_refuses_a_calculation_name_needing_repair(client, tmp_path):
    root = tmp_path / "has spaces!"
    root.mkdir()
    (root / "job.fdf").write_text("x\n")
    body = _post(client, "init", root, note="set up").get_json()
    assert body["ok"] is False and body["errors_only"]

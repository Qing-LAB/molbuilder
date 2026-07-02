"""Working-copy persistence core (molbuilder/workingcopy.py) — drives the
S1-S6 acceptance scenarios + the § 9 risks through a trivial two-file codec."""
import hashlib
import json
from pathlib import Path

import pytest

from molbuilder import workingcopy as wc


# ---- a trivial two-file artifact: <stem>.txt (source/identity) + sidecar ---- #
class TxtCodec:
    def load(self, source_path):
        p = Path(source_path)
        meta_p = p.with_name(f"{p.stem}.meta.json")
        tags = json.loads(meta_p.read_text())["tags"] if meta_p.exists() else []
        return {"body": p.read_text(), "tags": tags}

    def hash_source(self, source_path):
        return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()

    def files(self, data, target):
        target = Path(target)
        meta = (target.with_name(f"{target.stem}.meta.json"),
                (json.dumps({"tags": data["tags"]}) + "\n").encode())
        src = (target, data["body"].encode())
        return [meta, src]                     # identity/source (target) LAST

    def scratch_blob(self, data):
        return dict(data)

    def from_scratch(self, blob):
        return dict(blob)


CODEC = TxtCodec()


@pytest.fixture
def project(tmp_path):
    (tmp_path / "mol.txt").write_text("ATOM 1\nATOM 2\n")
    return tmp_path


def _open(project, session="s1"):
    return wc.WorkingCopy.open(project / "mol.txt", CODEC,
                               session=session, project_dir=project)


# --------------------------- happy path (S3) ------------------------------- #
def test_open_edit_commit_writes_both_files_and_clears_scratch(project):
    w = _open(project)
    assert w.data == {"body": "ATOM 1\nATOM 2\n", "tags": []}
    w.update({"body": "ATOM 1\nATOM 2\n", "tags": ["L-electrode"]})
    assert w.is_dirty()
    scratch = w._scratch_file()
    assert scratch.exists()                    # mirrored to scratch...
    assert not (project / "mol.meta.json").exists()   # ...NOT to durable yet

    r = w.commit(project / "mol.txt")
    assert r.ok and r.reason == "committed"
    assert json.loads((project / "mol.meta.json").read_text())["tags"] == ["L-electrode"]
    assert not scratch.exists()                # scratch cleared on success
    assert not w.is_dirty()


# --------------------------- reload (S1) ----------------------------------- #
def test_reload_restores_from_scratch(project):
    w = _open(project)
    w.update({"body": "X\n", "tags": ["t"]})
    # Simulate reload: read the scratch record back + recover().
    orphans = wc.list_orphans(project, live_sessions=[])   # no live session
    assert len(orphans) == 1
    w2 = wc.WorkingCopy.recover(orphans[0], CODEC, project_dir=project)
    assert w2.data == {"body": "X\n", "tags": ["t"]}
    assert w2.source == project / "mol.txt"


# --------------------------- changed underneath (S4) ----------------------- #
def test_commit_refuses_when_source_changed(project):
    w = _open(project)
    w.update({"body": "edited\n", "tags": ["t"]})
    # Someone edits mol.txt on disk (in another tool) -> hash changes.
    (project / "mol.txt").write_text("REORDERED\n")
    r = w.commit(project / "mol.txt")
    assert not r.ok and r.reason == "mismatch"
    assert r.expected_hash != r.actual_hash
    # Durable meta NOT written (no laundering).
    assert not (project / "mol.meta.json").exists()
    # "force" lets the app override the gate.
    r2 = w.commit(project / "mol.txt", on_mismatch="force")
    assert r2.ok


# --------------------------- save-as of a new artifact (S6) ---------------- #
def test_open_new_first_commit_is_save_as(project):
    w = wc.WorkingCopy.new(CODEC, session="s1", project_dir=project,
                           data={"body": "fresh\n", "tags": ["a"]})
    w.update({"body": "fresh\n", "tags": ["a"]})
    r = w.commit(project / "brand_new.txt")
    assert r.ok
    assert (project / "brand_new.txt").read_text() == "fresh\n"
    assert w.source == project / "brand_new.txt"        # re-anchored


# --------------------------- re-anchor: second save (§9.4) ----------------- #
def test_second_save_on_same_copy_does_not_falsely_refuse(project):
    w = _open(project)
    w.update({"body": "v1\n", "tags": ["t"]})
    assert w.commit(project / "mol.txt").ok
    # Continue editing the SAME working copy + save again -> must not trip the
    # gate on the file the first commit just wrote.
    w.update({"body": "v2\n", "tags": ["t", "u"]})
    r = w.commit(project / "mol.txt")
    assert r.ok, "re-anchor failed: second save wrongly refused"
    assert (project / "mol.txt").read_text() == "v2\n"


# --------------------------- save-as bypasses the source gate -------------- #
def test_save_as_to_different_target_ignores_source_change(project):
    w = _open(project)
    w.update({"body": "mine\n", "tags": ["t"]})
    (project / "mol.txt").write_text("CHANGED\n")        # source moved underneath
    r = w.commit(project / "copy.txt")                   # save-as elsewhere
    assert r.ok, "save-as should not gate on the untouched source"
    assert (project / "copy.txt").read_text() == "mine\n"


# --------------------------- discard -------------------------------------- #
def test_discard_drops_scratch_without_writing_durable(project):
    w = _open(project)
    w.update({"body": "x\n", "tags": ["t"]})
    scratch = w._scratch_file()
    assert scratch.exists()
    w.discard()
    assert not scratch.exists()
    assert not (project / "mol.meta.json").exists()


# --------------------------- crash recovery (§10) -------------------------- #
def test_orphans_listed_recovered_discarded(project):
    w = _open(project, session="dead-session")
    w.update({"body": "unsaved\n", "tags": ["t"]})
    # A different (live) session is running now; the dead one's scratch orphans.
    orphans = wc.list_orphans(project, live_sessions=["live-session"])
    assert len(orphans) == 1 and orphans[0].session == "dead-session"
    # recover, then discard.
    rec = wc.WorkingCopy.recover(orphans[0], CODEC, project_dir=project)
    assert rec.data["body"] == "unsaved\n"
    wc.discard_orphan(orphans[0])
    assert wc.list_orphans(project, live_sessions=["live-session"]) == []


def test_clean_all_wipes_scratch(project):
    _open(project, "a").update({"body": "1", "tags": []})
    _open(project, "b").update({"body": "2", "tags": []})
    assert wc.clean_all(project) == 2
    assert wc.clean_all(project) == 0

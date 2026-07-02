"""Working-copy core (molbuilder/workingcopy.py) — load, edit (draft), save."""
import json
from pathlib import Path

import pytest

from molbuilder import workingcopy as wc


class TxtCodec:
    """A trivial two-file artifact: <stem>.txt + <stem>.meta.json."""
    def load(self, source_path):
        p = Path(source_path)
        meta_p = p.with_name(f"{p.stem}.meta.json")
        tags = json.loads(meta_p.read_text())["tags"] if meta_p.exists() else []
        return {"body": p.read_text(), "tags": tags}

    def files(self, data, target):
        target = Path(target)
        return [(target, data["body"].encode()),
                (target.with_name(f"{target.stem}.meta.json"),
                 (json.dumps({"tags": data["tags"]}) + "\n").encode())]

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


def test_open_edit_save_writes_files_and_clears_draft(project):
    w = _open(project)
    assert w.data == {"body": "ATOM 1\nATOM 2\n", "tags": []}
    w.update({"body": "ATOM 1\nATOM 2\n", "tags": ["L-electrode"]})
    assert w.is_dirty()
    draft = w._scratch_file()
    assert draft.exists()                              # draft kept...
    assert not (project / "mol.meta.json").exists()    # ...but NOT saved yet
    saved = w.save(project / "mol.txt")
    assert saved == (project / "mol.txt").resolve()
    assert json.loads((project / "mol.meta.json").read_text())["tags"] == ["L-electrode"]
    assert not draft.exists()                          # draft dropped after save
    assert not w.is_dirty()


def test_save_as_new_path(project):
    w = _open(project)
    w.update({"body": "edited\n", "tags": ["t"]})
    w.save(project / "copy.txt")
    assert (project / "copy.txt").read_text() == "edited\n"
    assert json.loads((project / "copy.meta.json").read_text())["tags"] == ["t"]


def test_save_overwrites_disk_freely(project):
    # No gate: even if the file changed on disk, save writes the browser's copy.
    w = _open(project)
    w.update({"body": "mine\n", "tags": []})
    (project / "mol.txt").write_text("SOMETHING ELSE\n")
    w.save(project / "mol.txt")
    assert (project / "mol.txt").read_text() == "mine\n"


def test_reload_restores_draft(project):
    w = _open(project)
    w.update({"body": "X\n", "tags": ["t"]})
    orphans = wc.list_orphans(project, live_sessions=[])
    assert len(orphans) == 1
    w2 = wc.WorkingCopy.recover(orphans[0], CODEC, project_dir=project)
    assert w2.data == {"body": "X\n", "tags": ["t"]}


def test_new_first_save_is_save_as(project):
    w = wc.WorkingCopy.new(CODEC, session="s1", project_dir=project,
                           data={"body": "fresh\n", "tags": ["a"]})
    w.update(w.data)
    w.save(project / "brand_new.txt")
    assert (project / "brand_new.txt").read_text() == "fresh\n"


def test_discard_drops_draft_without_writing(project):
    w = _open(project)
    w.update({"body": "x\n", "tags": ["t"]})
    draft = w._scratch_file()
    assert draft.exists()
    w.discard()
    assert not draft.exists()
    assert not (project / "mol.meta.json").exists()


def test_orphans_recover_discard(project):
    w = _open(project, session="dead")
    w.update({"body": "unsaved\n", "tags": ["t"]})
    orphans = wc.list_orphans(project, live_sessions=["live"])
    assert len(orphans) == 1 and orphans[0].session == "dead"
    assert wc.WorkingCopy.recover(orphans[0], CODEC,
                                  project_dir=project).data["body"] == "unsaved\n"
    wc.discard_orphan(orphans[0])
    assert wc.list_orphans(project, live_sessions=["live"]) == []


def test_clean_all(project):
    _open(project, "a").update({"body": "1", "tags": []})
    _open(project, "b").update({"body": "2", "tags": []})
    assert wc.clean_all(project) == 2
    assert wc.clean_all(project) == 0

"""The framed notebook server's config -- copied, not imported.

Contract: `docs/web/jupyter.md` § 4.2.  `molbuilder.jupyter.prepare_lab_home`
copies this file verbatim into the Lab home and hands the copy to
jupyter-server as an ABSOLUTE ``ServerApp.config_file``.

**IT EXISTS FOR ONE THING: no `.ipynb_checkpoints/` in the projects tree.**
Jupyter writes that directory beside every notebook it saves -- a directory in
every folder somebody has opened a notebook in, swept up by result scans,
carried along by every copy to a cluster, and holding a stale duplicate of
work nobody asked it to keep.  Jupyter has no switch for it:
``FileCheckpoints.checkpoint_dir`` only RENAMES the directory, and pointing it
at one shared absolute path is worse than the problem -- the checkpoint file
is named after the notebook alone, so two ``Untitled.ipynb`` in different
folders collide and a restore hands back the wrong file.  What the contents
manager does take is a ``checkpoints_class``, and jupyter-server ships no
no-op one, so molbuilder writes it.  A Jupyter config file is executed Python,
which is why the class can live here rather than on ``PYTHONPATH``.

**WHY A FILE UNDER `data/` AND NOT A MODULE.**  It was 40 lines of Python
inside a string literal in `jupyter.py` until 2026-09-15, where no linter,
import or test could reach it -- and on 2026-09-14 an edit deleted it by
accident and shipped a `NameError` on every notebook start, with nothing to
catch it.  `inspect.getsource` is this project's pattern for generated code
(`trajectory_log/emitter.py`, `pyscf/input.py`), but it cannot serve here:
``NoCheckpoints`` must subclass jupyter_server's ``AsyncCheckpoints`` at
class-definition time, and the shepherd runs in the HOST env, which does not
have jupyter_server and must not need it.  A file under `data/` is never
imported by molbuilder, so it may name jupyter_server freely -- while
pyflakes, an editor and `ast.parse` all still read it as Python.

``c`` is the config object jupyter-core injects when it execs this file; it is
undefined to a linter reading the file on its own, which is what the two
``noqa`` markers are for.
"""
from datetime import datetime, timezone

from jupyter_server.services.contents.checkpoints import AsyncCheckpoints


class NoCheckpoints(AsyncCheckpoints):
    """Answer the checkpoint API without writing anything to disk."""

    async def create_checkpoint(self, contents_mgr, path):
        return {"id": "no-checkpoint",
                "last_modified": datetime.now(timezone.utc)}

    async def list_checkpoints(self, path):
        return []

    async def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        return None

    async def delete_checkpoint(self, checkpoint_id, path):
        return None

    async def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        # REFUSES rather than quietly doing nothing: with `list_checkpoints`
        # empty Lab offers nothing to restore, and a path that could still be
        # reached must never silently discard an edit.
        from tornado.web import HTTPError
        raise HTTPError(
            400,
            "molbuilder runs this notebook server with checkpoints disabled, "
            "so there is nothing to restore.",
        )


# Both sections: `AsyncContentsManager` redeclares the trait that
# `ContentsManager` defines, and the running manager inherits from both.
c.ContentsManager.checkpoints_class = NoCheckpoints        # noqa: F821
c.AsyncContentsManager.checkpoints_class = NoCheckpoints   # noqa: F821

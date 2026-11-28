"""Every example notebook molbuilder ships, run as a person runs it --
`engines/vibration-frames.md` § 6.1 -- on `molbuilder-jupyternb`'s own
Python.

The silent failures: an example that no longer runs on the code it shows --
a call renamed, a closed form of the science the toolkit stopped meeting, a
figure no longer drawn, a cell that errors -- shipped to every
installation's `notebooks/examples/`; and a notebook writing into the
person's projects tree.  Each notebook's own asserts are the checks (the
motion toolkit's, each against the science, § 6.1); this runs each whole,
as a person meets it -- written into a projects tree of the test's own by
the door installation asks (`projects.write_examples`), run from its folder
there, its saved outputs cleared -- twice in one kernel as *Run All* twice
does, its kernel's environment the Jupyter tab's door's
(`jupyter.notebook_environment`) on a base cleared of molbuilder's
settings, that tree its projects root; transport's with no run of the
person's named, its last cells skipped (`engines/transport.md`, *The
script, and the notebook*).  Saved outputs are not compared (§ 6.1: they
hold the numpy of the run that saved them).

Run where the opt-in `molbuilder-jupyternb` env is installed; skipped
elsewhere.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from molbuilder import data_dir
from support.tooling import needs

pytestmark = needs("molbuilder-jupyternb")

#: THE RUNNER, on the notebook env's own Python (`nbclient`, `nbformat`, its
#: recipe's): the notebook read, its saved outputs cleared, its cells twice in
#: one kernel -- *Run All*, then *Run All* again, each cell a copy of its own
#: -- and the notebook so run written for the test to read.
_RUNNER = """
import copy, sys
import nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs, cell.execution_count = [], None
again = copy.deepcopy(nb.cells)
for cell in again:
    cell["id"] = cell["id"][:58] + "-again"
nb.cells = nb.cells + again
NotebookClient(nb, kernel_name="python3", timeout=900, record_timing=False,
               resources={"metadata": {"path": sys.argv[3]}}).execute()
nbformat.write(nb, sys.argv[2])
"""

#: EVERY EXAMPLE NOTEBOOK MOLBUILDER SHIPS, by name.
_SHIPPED = sorted(p.name for p in (data_dir() / "examples").glob("*.ipynb"))


@pytest.mark.parametrize("name", _SHIPPED)
def test_every_shipped_notebook_runs_whole_twice_in_one_kernel(name,
                                                              tmp_path):
    """§ 6.1: each notebook molbuilder ships, written into a projects tree as
    installation writes it and run from its folder there, runs to its end
    twice in one kernel -- every code cell run, none with an error, every
    assert of its own holding (the first failing one named by its cell),
    every cell that draws (tagged ``plot``) drawing its figure both times --
    and the tree it was given holds afterwards exactly the examples written,
    as written: nothing saved into it."""
    from _road import env_bin
    from molbuilder.jupyter import notebook_environment
    from molbuilder.projects import write_examples
    assert _SHIPPED, "molbuilder ships no example notebook"
    sentinel = tmp_path / "sentinel"
    written = {d.path: d.path.read_bytes()
               for d in write_examples(base=sentinel)}
    notebook = next(p for p in written if p.name == name)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("MOLBUILDER_")}
    base.update(MOLBUILDER_PROJECTS=str(sentinel / "projects"),
                MOLBUILDER_CONFIG_DIR=str(tmp_path / "cfg"),
                TMPDIR=str(tmp), MPLCONFIGDIR=str(tmp_path / "mpl"),
                IPYTHONDIR=str(tmp_path / "ipython"),
                JUPYTER_RUNTIME_DIR=str(tmp_path / "jupyter-runtime"),
                JUPYTER_DATA_DIR=str(tmp_path / "jupyter-data"))
    ran = tmp_path / "ran.ipynb"
    done = subprocess.run(
        [str(env_bin("molbuilder-jupyternb") / "python"), "-c", _RUNNER,
         str(notebook), str(ran), str(notebook.parent)],
        env=notebook_environment("", base=base), capture_output=True,
        text=True, timeout=1800)
    assert done.returncode == 0, (done.stdout + done.stderr)[-6000:]
    cells = json.loads(ran.read_text())["cells"]
    half = len(cells) // 2
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    assert [i for i in code if not cells[i].get("execution_count")] == []
    assert [i for i in code for o in cells[i].get("outputs", [])
            if o["output_type"] == "error"] == []
    plots = [i for i, c in enumerate(cells[:half])
             if "plot" in c.get("metadata", {}).get("tags", [])]
    assert plots, f"no cell of {name} is tagged to draw"
    for i in plots + [half + i for i in plots]:
        drew = [o for o in cells[i].get("outputs", [])
                if "image/png" in o.get("data", {})]
        assert drew, (f"cell {i % half} drew no figure "
                      f"({'second' if i >= half else 'first'} run)")
    files = {p: p.read_bytes() for p in sentinel.rglob("*") if p.is_file()}
    assert files == written, sorted(str(p) for p in set(files) ^ set(written))

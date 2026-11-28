"""THE EXAMPLES' RUNS, MADE DOWN MOLBUILDER'S ROAD -- and the notebooks run on
them, saved with their outputs (`engines/vibration-frames.md` § 6.1).

Run from the repository root, in the host env, naming a scratch folder of the
tool's own -- its projects tree, its configuration, its state.  Of yours it
READS one thing, and writes nothing: which env is this machine's notebook env
and its manager (`envs` in the `molbuilder.json` the configuration it runs
with holds -- yours unless `MOLBUILDER_CONFIG_DIR` names another), to run the
notebooks there:

    "${CONDA_PREFIX:?}/bin/python" tools/make_example_runs.py SCRATCH
    "${CONDA_PREFIX:?}/bin/python" tools/make_example_runs.py SCRATCH --only notebooks

1. **The machine**, as an installation sets one up: `molbuilder envs
   init-config --yes` in SCRATCH/config -- the recommended activation, this
   machine probed into that configuration's own record.
2. **Each example's run**, as a person makes it: its structure saved as a
   pair in SCRATCH/projects, `molbuilder describe`, its template's settings
   set as a person edits the file (`template.with_values_set`), `molbuilder
   jobset prep task` and `molbuilder jobset launch task`; then the three
   files the motion toolkit reads -- the description, the run's own record,
   the result -- copied into the example's folder in molbuilder's package
   (`data/examples/`), each replacing the one there.
3. **Every example notebook molbuilder ships** -- those that read a run
   above, and the others (transport's by hand, no run of yours named) --
   its saved outputs cleared, run start to end on `molbuilder-jupyternb`'s
   own Python by `nbclient`, through the one door into an env
   (`envs.builds.dispatch_into_env`), its kernel's environment the Jupyter
   tab's (`jupyter.notebook_environment`) on a base cleared of molbuilder's
   settings, its projects root SCRATCH/sentinel/projects, which must stay
   empty -- and saved with its outputs, in place, only where every code cell
   ran with no error.

`--only runs` makes the runs alone; `--only notebooks` runs the notebooks on
the runs molbuilder carries.  Each step's time is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

REPO = Path(__file__).resolve().parent.parent
# THIS CHECKOUT'S MOLBUILDER, in this process too: the commands it starts run
# from the repository root, and what it imports must be the same code.
sys.path.insert(0, str(REPO))


@dataclass(frozen=True)
class Example:
    """One example's run, the one its notebook reads: ``carried`` the run's
    calculation folder in `data/examples/`;
    ``project``, ``bundle`` where the run is made in the scratch tree, from
    its root; ``name`` the calculation's (its label); ``structure`` its
    elements and positions (Å) in a box of ``box`` Å; ``settings`` its
    template's items as a person sets them; ``stage`` the stage launched;
    ``threads`` the run's cores (`jobset prep --cpus-per-task`)."""
    carried: str
    project: str
    bundle: str
    name: str
    structure: Tuple[Tuple[str, Tuple[float, float, float]], ...]
    box: float
    settings: Dict[str, object]
    stage: str
    threads: int


#: METHANOL, STAGGERED (Cs): the hydroxyl H anti to the in-plane methyl H, so
#: the torsion starts at its minimum, not at the eclipsed saddle -- C, O, the
#: methyl's three H, the hydroxyl H, centred in a 10 Å box.  Its record's
#: temperature the notebook's 300 K; infrared the default -- the notebook
#: reads the C-O stretch's intensity -- and Raman off, which it does not read.
METHANOL = Example(
    carried="vibration-frames-run/methanol",
    project="P", bundle="P/spectrum/methanol", name="methanol",
    structure=(("C", (-0.0467, 0.6629, 0.0)), ("O", (-0.0467, -0.7570, 0.0)),
               ("H", (-1.0861, 0.9753, 0.0)), ("H", (0.4359, 1.0812, 0.8913)),
               ("H", (0.4359, 1.0812, -0.8913)), ("H", (0.8550, -1.0811, 0.0))),
    box=10.0, settings={"temperature_K": 300.0}, stage="freq", threads=4)

EXAMPLES = (METHANOL,)

#: THE NOTEBOOK'S RUNNER, on the notebook env's own Python: read, its saved
#: outputs cleared, run start to end in a kernel of its own, and written back
#: with its outputs only where every code cell ran with no error -- no cell
#: skipped by a tag, and no timing recorded in the file, which would change
#: at every run.
_RUNNER = """
import sys
import nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs, cell.execution_count = [], None
NotebookClient(nb, kernel_name="python3", timeout=900, record_timing=False,
               skip_cells_with_tag="", resources={"metadata": {"path": sys.argv[2]}}).execute()
bad = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"
       and (not c.execution_count
            or any(o.output_type == "error" for o in c.outputs))]
if bad:
    sys.exit(f"cells {bad} did not run, or ran with an error: not saved")
nbformat.write(nb, sys.argv[1])
"""


def _places(scratch: Path) -> Dict[str, Path]:
    """The scratch folder's parts, made where they are not: the projects
    tree, the configuration, the state and runtime folders, the notebooks'
    working folder and their projects root, which must stay empty."""
    places = {k: scratch / v for k, v in (
        ("projects", "projects"), ("config", "config"),
        ("xdg_config", "xdg-config"), ("state", "state"),
        ("runtime", "runtime"), ("cache", "cache"), ("work", "work"),
        ("sentinel", "sentinel/projects"), ("tmp", "tmp"),
        ("mpl", "mpl"), ("ipython", "ipython"),
        ("jupyter_runtime", "jupyter-runtime"),
        ("jupyter_data", "jupyter-data"))}
    for path in places.values():
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)
    return places


def _cleared() -> Dict[str, str]:
    """This process's environment with none of molbuilder's settings."""
    return {k: v for k, v in os.environ.items()
            if not k.startswith("MOLBUILDER_")}


def _molbuilder(places: Dict[str, Path], *words: str) -> str:
    """One `molbuilder` command, as a person types it, in the scratch tree
    and configuration -- molbuilder's own child, ``sys.executable``; what it
    said, or the tool stopped with it."""
    env = _cleared()
    env.update(MOLBUILDER_PROJECTS=str(places["projects"]),
               MOLBUILDER_CONFIG_DIR=str(places["config"]),
               XDG_CONFIG_HOME=str(places["xdg_config"]),
               XDG_STATE_HOME=str(places["state"]),
               XDG_RUNTIME_DIR=str(places["runtime"]),
               XDG_CACHE_HOME=str(places["cache"]))
    print(f"$ molbuilder {' '.join(words)}", flush=True)
    done = subprocess.run([sys.executable, "-m", "molbuilder", *words],
                          cwd=REPO, env=env, capture_output=True, text=True)
    said = done.stdout + done.stderr
    if done.returncode != 0:
        sys.exit(f"molbuilder {' '.join(words)} exited {done.returncode}:\n"
                 f"{said}")
    return said


def make_run(example: Example, places: Dict[str, Path]) -> None:
    """``example``'s run down the road in the scratch tree, then the three
    files the toolkit reads copied into molbuilder's package."""
    import numpy as np
    from molbuilder import data_dir
    from molbuilder.calculation import open as open_calculation
    from molbuilder.engine_table import config_class
    from molbuilder.projects import populate_project_skeleton
    from molbuilder.runfiles import CALCDIR_FILE, ROLE_SPECTRA, TASK_FILE
    from molbuilder.runs import on_label_at
    from molbuilder.structure import Structure
    from molbuilder.template import with_values_set
    from molbuilder.workingcopy_structure import StructureCodec
    tree = places["projects"]
    bundle = tree / example.bundle
    if bundle.exists():
        sys.exit(f"{bundle} is there already: name a new scratch folder")
    # THE STRUCTURE, SAVED AS A PERSON SAVES ONE: its project made whole by
    # the tree's one door, the pair written by the codec.
    project = tree / example.project
    if not project.exists():
        populate_project_skeleton(project, project_name=example.project)
    half = example.box / 2.0
    s = Structure(elements=[e for e, _ in example.structure],
                  positions=np.array([p for _, p in example.structure]) + half,
                  cell=np.diag([example.box] * 3),
                  axis_kind=("isolated",) * 3)
    at = f"{example.project}/structure/{example.name}.xyz"
    StructureCodec().write(s, tree / at)
    _molbuilder(places, "describe", "--structure", at, "--bundle",
                example.bundle, "--engine", "pyscf", "--calculation",
                "vibration", "--name", example.name)
    # ITS SETTINGS, AS A PERSON EDITS THE TEMPLATE: each said to be theirs
    # (`engines/template.md` § 6.6).
    calc = open_calculation(bundle)
    calc.template.path.write_text(with_values_set(
        calc.template_text, config_class(str(calc.task.engine)),
        calculation=str(calc.task.calculation), values=example.settings,
        source="person"))
    _molbuilder(places, "jobset", "prep", "task", "--group", "1", "--target",
                "this", "--cpus-per-task", str(example.threads), "--bundle",
                example.bundle)
    began = time.monotonic()
    print(_molbuilder(places, "jobset", "launch", "task", "--stage",
                      example.stage, "--mode", "direct", "--foreground",
                      "--yes", "--bundle", example.bundle)[-2000:])
    print(f"the run took {time.monotonic() - began:.0f} s", flush=True)
    # THE THREE FILES THE TOOLKIT READS, by the doors that name them: the
    # description, and of the run that holds the result its own record and
    # the result.
    run = next(r for r in sorted(bundle.glob(f"*_{example.stage}/run-*"))
               if on_label_at(r, ROLE_SPECTRA) is not None)
    result = on_label_at(run, ROLE_SPECTRA)
    into = data_dir() / "examples" / example.carried
    for source in (bundle / TASK_FILE, run / CALCDIR_FILE, result):
        target = into / source.relative_to(bundle)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(f"  copied {target.relative_to(REPO)}", flush=True)


def run_notebook(notebook: Path, places: Dict[str, Path]) -> None:
    """``notebook``, one molbuilder ships, run start to end on
    `molbuilder-jupyternb`'s own Python and saved with its outputs; the
    projects root it is given left empty, or the tool stops."""
    from molbuilder.diagnostics import detect
    from molbuilder.envs.builds import conda_run_argv, dispatch_into_env
    from molbuilder.envs.recipes import effective_name, recipe_of_category
    from molbuilder.jupyter import notebook_environment
    caps = detect()
    env_name = effective_name(recipe_of_category("jupyter"), caps)
    base = _cleared()
    base.update(MOLBUILDER_PROJECTS=str(places["sentinel"]),
                MOLBUILDER_CONFIG_DIR=str(places["config"]),
                TMPDIR=str(places["tmp"]), MPLCONFIGDIR=str(places["mpl"]),
                IPYTHONDIR=str(places["ipython"]),
                JUPYTER_RUNTIME_DIR=str(places["jupyter_runtime"]),
                JUPYTER_DATA_DIR=str(places["jupyter_data"]))
    began = time.monotonic()
    rc, said = dispatch_into_env(
        conda_run_argv(caps.conda_binary, env_name, "python", "-c", _RUNNER,
                       str(notebook), str(places["work"])),
        caps.env_prefix(env_name), cwd=REPO,
        env=notebook_environment("", base=base), timeout=3600)
    if rc != 0:
        sys.exit(f"{notebook.name} did not run to its end:\n{said[-6000:]}")
    left = sorted(places["sentinel"].rglob("*"))
    if left:
        sys.exit(f"{notebook.name} wrote into the projects root it was "
                 f"given: {left}")
    cells = json.loads(notebook.read_text())["cells"]
    print(f"{notebook.name}: ran in {time.monotonic() - began:.0f} s, "
          f"{len(cells)} cells, {notebook.stat().st_size // 1024} KB",
          flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scratch", type=Path,
                        help="a folder of the tool's own")
    parser.add_argument("--only", choices=("runs", "notebooks"))
    asked = parser.parse_args(argv)
    places = _places(asked.scratch.resolve())
    if asked.only != "notebooks":
        _molbuilder(places, "envs", "init-config", "--yes")
        for example in EXAMPLES:
            make_run(example, places)
    if asked.only != "runs":
        from molbuilder import data_dir
        for notebook in sorted((data_dir() / "examples").glob("*.ipynb")):
            run_notebook(notebook, places)
    return 0


if __name__ == "__main__":
    sys.exit(main())

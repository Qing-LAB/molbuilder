"""The monitor's shipped files run beside a job with molbuilder absent.

The premise the monitor rests on (`execution/run-reports.md` § 2.3): the files
`runwrap.MONITOR_COMPANIONS` names travel to the machine that runs the job and
execute under the job's own python, where molbuilder is not installed.  This
runs them there -- a copy of a real finished run, the bundle beside it -- and
asks the shipped monitor how the run ended.

It lived in `test_layering.py` until 2026-09-27, beside the source scans of
the layer rule, which were retired then (user: the layer rule is a static
review matter, `process/code-audit.md` § 1c; "run folder does not have
molbuilder env so the error would show up anyway" -- and this is where it
shows up).
"""
from __future__ import annotations

import os
from pathlib import Path


def test_every_file_that_ships_beside_a_job_imports_without_molbuilder(tmp_path):
    """The premise the whole monitor rests on, reproduced rather than asserted.

    `runwrap.MONITOR_COMPANIONS` travels to the machine that runs the job --
    inside ONE file, `mb_monitor.pyz` (`runwrap.MONITOR_BUNDLE`) -- and is
    executed by **the job's own python**, inside a backend env where molbuilder
    is not installed and numpy is not either.  So every module in that table has
    to import with the package absent, each reaching the next through its
    two-way import -- the package first, the copy in the bundle second -- as
    `config_dir` always has.

    **And it has to READ with the framework's readers, not merely import.**
    Since 2026-09-26 the monitor reads a run through the Results tab's own
    status door and the output's one parser -- each family's reading pass --
    shipped beside it (`execution/run-reports.md` § 2.3).  So the probe stages the whole table
    into a copy of a REAL finished run -- the measured H2 relaxation under
    `tests/fixtures/siesta_relax` (an API-level test on a measured fixture: the
    road cannot run with molbuilder absent) -- and asks the shipped monitor how
    it ended.  It must be the SHIPPED readers answering, from the bundle, and
    their answer must be the package's own `run_status`'s.

    No layer rule can see this.  A travelling module importing `persist` is
    L1 importing L1 -- perfectly legal, and fatal here, because only these
    files travel.  And the failure is SILENT: `runwrap` records the last
    time it happened (`config_dir.py` added to one stager and not the other),
    and what it cost was *every production run's monitor dying at import with
    stderr to /dev/null* -- no [MACHINE], no status, no util.csv, no reports.
    """
    import json
    import shutil
    import subprocess
    import sys

    from molbuilder.parse.dirs import run_status
    from molbuilder.runwrap import MONITOR_BUNDLE, monitor_bundle

    src = (Path(__file__).resolve().parent / "fixtures" / "siesta_relax"
           / "01_relax" / "run-0")
    run = tmp_path / "run-0"
    shutil.copytree(src, run)
    (run / MONITOR_BUNDLE).write_bytes(monitor_bundle())

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    # The condition has to be real, or this test proves nothing: from that
    # directory, with no PYTHONPATH, `molbuilder` must genuinely be unimportable
    # (it is deliberately not pip-installed -- it runs as `python -m molbuilder`
    # from the repo root).
    control = subprocess.run([sys.executable, "-c", "import molbuilder"],
                             cwd=run, env=env, capture_output=True,
                             text=True, timeout=120)
    assert control.returncode != 0, (
        "molbuilder is importable from the staging directory, so this test "
        "cannot reproduce a compute node.  Is it pip-installed?")

    probe = (
        # the bundle runs with itself first on the path, as `python
        # mb_monitor.pyz` puts it
        f"import sys; sys.path.insert(0, {MONITOR_BUNDLE!r})\n"
        "import json, os, mb_monitor as M\n"
        # the flat fallback must actually resolve, not merely not raise
        "assert M.default_notify_path().name == 'notify'\n"
        # the readers are the COPIES beside the job, not molbuilder's -- and
        # so is everything the parser itself reads through
        "P = sys.modules[M.SiestaReader.__module__]\n"
        "where = {n: getattr(o, '__module__', getattr(o, '__name__', ''))\n"
        "         for n, o in (('status', M.run_status),\n"
        "                      ('siesta', M.SiestaReader),\n"
        "                      ('molwatch', M.MolwatchReader),\n"
        "                      ('grammar', P._G), ('rules', P.compile_rules),\n"
        "                      ('names', M._rf))}\n"
        "where['from'] = os.path.basename(os.path.dirname(M.__file__))\n"
        "w = M.WatchedRun(label='H2', stage='01_relax', run=0)\n"
        "st = w.conclude(w.read(0.0, 1.0))\n"
        "print(json.dumps({'where': where, 'state': st.state,\n"
        "                  'detail': st.detail, 'converged': st.converged,\n"
        "                  'energy': st.energy, 'text': st.as_text()}))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=run, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (
        "a file that ships beside a job cannot be imported, or cannot read the "
        f"run, without molbuilder:\n{done.stdout}{done.stderr}")
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["where"] == {"status": "job", "siesta": "siesta_reader",
                            "molwatch": "molwatch_reader",
                            "grammar": "siesta_grammar",
                            "rules": "_section_rules",
                            "names": "runfiles",
                            "from": MONITOR_BUNDLE}, got["where"]

    # THE SAME ANSWER THE PACKAGE GIVES: one status door, shipped or not.
    here = run_status(src, "H2_01_relax*")
    assert (got["state"], got["detail"]) == (here.state, here.detail), got
    speaker = here.endings[here.active_source]
    assert got["converged"] == {k: v for k, v in speaker.phases.items()}, got
    assert got["energy"] is not None, (
        f"the shipped parser read no energy off the real .out: {got['text']}")

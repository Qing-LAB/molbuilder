"""One warm-file inventory per engine, and every surface derives from it.

Contract: `execution/staged-runs-implementation-plan.md`, item 12c, whose *Done when* is stated
as a mutation and is written that way here — **add a suffix to the list and
watch both behaviours change.**  Since U17 (2026-08-12) the ``--cold`` mover
is a NAME SWEEP (`job-contracts.md` § 4.1) and no longer reads the list at
all — so its half of the mutation inverted: the banner must show the new
suffix, and the mover must NOT (an enumeration reappearing there is the
snapshot-of-one-build regression the sweep replaced).

**The failure this prevents is silent and asymmetric.** Two lists that agree
today, with nothing keeping them agreeing:

* add a warm hook to the **carry/banner** list alone and a ``--cold`` run
  warm-starts from it anyway — a contaminated calculation that reports
  success;
* add it to the **``--cold`` mover** alone and a run announces *"initial run
  (clean state)"* and then has that very file moved aside as warm state.

`run-identity.md § 5` says the banner is the half that must never be weakened,
because it is the one always present. SIESTA's pair was derived from a single
tuple by P3's Review 2 for exactly this reason. **PySCF's was missed until
2026-08-10**: its mover covered five suffixes while its banner tested ``.chk``
alone, so a run holding only ``<JOB>_optimized.xyz`` hit the second failure.

The third list — the one that fed attempt directories — is gone: P7 unit 1
retired ``_attempt_dir_block``, and the two helpers that built filenames for
it went too. **The plan predicted they had no other caller and was wrong**:
``validation/identity.py`` used them, then stripped the id back off to recover
the suffixes it actually wanted. It reads the tuples directly now, so the
subtraction landed and removed a re-derivation on the way out.
"""
from __future__ import annotations

import json
import re

import pytest

from molbuilder import runwrap
from molbuilder.jobset.model import Resources


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    """The renders resolve script_generation config from cwd + HOME/XDG
    (B-9, 2026-08-13): unsandboxed, every wrapper here folded in the
    developer's repo-root molbuilder.json, so the banner/mover surfaces
    under test varied by machine.  Sandboxed, with the activation the
    writer requires DECLARED by the test."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    cwd = tmp_path_factory.mktemp("cwd")
    (cwd / "molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": "true"}}))
    monkeypatch.chdir(cwd)
    # THE SANDBOX IS THE CONFIG ROOT.  This config was read through the
    # working-directory step, which is gone (configuration.md § 2.1a) --
    # without naming the directory the write lands in a file nothing
    # opens, and the test passes having configured nothing.
    monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(cwd))


ENGINES = (
    pytest.param("siesta", "_SIESTA_WARM_SUFFIXES", ".fdf",
                 "SystemLabel job\n", id="siesta"),
    pytest.param("pyscf", "_PYSCF_WARM_SUFFIXES", ".py",
                 'JOB = "job"\nimport pyscf\n', id="pyscf"),
)


def _wrapper(tmp_path, ext, body, **kw):
    p = tmp_path / f"job{ext}"
    p.write_text(body)
    return runwrap.render_run_wrapper(p, env="molbuilder-siesta",
                                      resources=Resources(**kw))


@pytest.mark.parametrize("engine,const,ext,body", ENGINES)
def test_adding_a_suffix_changes_the_banner_and_only_the_banner(
        tmp_path, monkeypatch, engine, const, ext, body):
    """12c's *Done when*, re-stated for the § 4.1 name sweep (U17,
    2026-08-12).

    A suffix nobody would invent is added to the engine's one tuple.  The
    startup banner must show it -- the banner ENUMERATES, and a suffix it
    does not test is a run announcing a clean start over live warm state.
    The ``--cold`` mover must NOT show it: since U17 the mover is a NAME
    SWEEP (*"everything the id names, minus what molbuilder wrote"*,
    job-contracts § 4.1), so any suffix is covered by construction and an
    enumeration REAPPEARING in the mover is the snapshot-of-one-build
    regression the sweep replaced.  (The sweep's behaviour itself --
    files actually moving -- is executed in test_runwrap_cold_restart.)
    """
    marker = "_mb_probe.state"
    before = _wrapper(tmp_path, ext, body)
    assert marker not in before, "the probe suffix must not already occur"

    monkeypatch.setattr(runwrap, const,
                        tuple(getattr(runwrap, const)) + (marker,))
    after = _wrapper(tmp_path, ext, body)

    banner, mover = _two_surfaces(after)

    assert re.search(rf'\[ -e "[^"]*{re.escape(marker)}" \]', banner), (
        "the startup banner does not read the engine's warm list: a run whose "
        "only warm file has this suffix would announce a clean start")
    assert '"$_warm_label".*' in mover, (
        "the --cold mover lost its name-sweep glob: files named by the "
        "run's id would survive --cold")
    assert marker not in mover, (
        "the --cold mover enumerates a warm suffix again: a list is a "
        "snapshot of one build, and job-contracts § 4.1 replaced it with "
        "the name sweep")


#: The generated comments that open the two blocks, in the order the wrapper
#: emits them: the ``--cold`` mover first, the startup banner after it.
#:
#: Anchoring on both is what makes the two assertions independent. An earlier
#: version of this file split on the first occurrence of the word *"cold"* --
#: which lands in an unrelated ``--help`` line -- so one block's match
#: satisfied the other's assertion, and two mutations retyping the mover's
#: list survived. **This module's own subject, reproduced inside its test.**
_MOVER_HEADING = "--- Cold restart: SAY WHAT WOULD BE LOST, THEN STOP"
_BANNER_HEADING = "--- Runtime status banner"


def _two_surfaces(text: str):
    """Split a wrapper into its ``(banner, --cold mover)`` halves.

    The two halves of the contract have to be checked apart, or a list read
    correctly by one of them covers for the other.
    """
    for anchor in (_MOVER_HEADING, _BANNER_HEADING):
        assert anchor in text, (
            f"{anchor!r} is not where this test expects it; update the anchor "
            "rather than widening the search, which is how the split stopped "
            "separating anything the first time")
    i, j = text.index(_MOVER_HEADING), text.index(_BANNER_HEADING)
    assert i < j, "the wrapper's block order moved; the split is now wrong"
    return text[j:], text[i:j]


def test_there_is_exactly_one_warm_inventory_per_engine():
    """`job-contracts.md` § 4.2: **one warm-file inventory per engine.**

    Stated as the set of inventories `runwrap` declares. Two lists per engine
    agree by luck: add a warm suffix to one and a `--cold` run silently
    warm-starts from the file the other does not know about -- a contaminated
    calculation that reports success.

    **An equality, so it fails both ways**: a third inventory fails whatever it
    is called, and one of these two disappearing fails too. It replaced three
    `not hasattr` assertions naming symbols this program deleted, which pinned
    the absence of the past rather than the shape of the present.
    """
    import inspect
    src = inspect.getsource(runwrap)
    declared = set(re.findall(r"^(_[A-Z_]*WARM[A-Z_]*)\s*=", src, re.M))
    assert declared == {"_SIESTA_WARM_SUFFIXES", "_PYSCF_WARM_SUFFIXES"}, (
        f"runwrap declares warm inventories {sorted(declared)}; the contract "
        "gives each engine exactly one")
    # U5 (job-contracts § 4.2a): one FILE per engine, and these names are
    # its READ, never a second listing -- a literal tuple reappearing
    # here is the fork § 4.2a retired coming back.
    for name in declared:
        m = re.search(rf"^{name}\s*=\s*(.+)$", src, re.M)
        assert "_warm_inventory(" in m.group(1), (
            f"{name} is not derived from the warm-files loader: "
            f"{m.group(1)!r}")


@pytest.mark.parametrize("engine,const,ext,body", ENGINES)
def test_no_generated_wrapper_changes_directory(tmp_path, engine, const,
                                                ext, body):
    """`job-contracts.md § 2.1`: **the caller's working directory is the
    contract** — both launchers establish it, and neither the wrapper nor the
    engine ever navigates.

    ``_attempt_dir_block`` was the one thing in the system that broke it: it
    resolved ``run-<n>/``, created it, and ``cd``'d in, so everything after
    that line ran somewhere the caller had not chosen. Retiring it restored an
    invariant rather than tidying one, and this is what holds it.
    """
    text = _wrapper(tmp_path, ext, body)
    cds = re.findall(r"(?m)^\s*cd\s+\S+", text)
    assert cds == [], f"{engine} wrapper navigates: {cds}"


def test_the_wrapper_entry_points_take_exactly_these_parameters():
    """**Both** entry points, each stated as what it accepts.

    `running-a-job.md` § 2.2a: the wrapper activates and execs. Each parameter
    is something a caller bakes in at generate time, so the set is a contract
    surface -- a new one means new work on a compute node and should be argued
    for rather than appear.

    **All three take the allocation whole** (`architecture.md` § 3.1, rule A8),
    and differ only in what else they need. `render_run_wrapper` returns the
    inner script's text, so it also takes the sizing inputs read off the DECK:
    `n_atoms`. `render_wrappers` returns everything step 4
    produces -- the wrapper, the `.sbatch` when the machine has a queue, the
    monitor a SIESTA job carries -- and reads those two off the deck itself.
    `write_run_wrapper` writes what it rendered.

    `render_run_wrapper` named four of `Resources`' fields in its own signature
    until 2026-08-18, which is the destructure A8 forbids one call short of the
    door: the caller re-assembled the object rather than passing it.

    **This set shrank from twelve to four on 2026-08-17, and the shrinking is
    the fix.** Eleven loose keyword arguments meant two callers passing two
    different subsets: `jobset/prep.py` passed ten and wrote a `.sbatch` asking
    for `-c 8` beside a `.run.sh` baking an OMP default of `1`, while
    `web/blueprints/build.py` passed five and wrote a correct `.run.sh` beside
    a `.sbatch` with no `-c` at all. The door had already lost `max_memory_mb`
    the same way on 2026-08-11. A door with N loose arguments has 2^N ways to
    be called and one that is right; with the object there is no subset to
    choose, which is why this assertion is now the narrow one.

    ``env`` and ``emit_sbatch`` stay loose because neither is a per-job fact --
    ``env`` is a per-invocation override (`prep --env`) and ``emit_sbatch`` is
    a surface's choice about what to write.  ``project_dir`` (2026-08-24,
    layout repair M1) is the BUNDLE's scope, stated by the caller: the script
    is born in its job directory now, so a scope derived from the script's
    parent would read config one level below the bundle's own files.  The
    test is ownership: a field with a home in § 3's table arrives in that
    home or not at all.

    ``machine_record`` (2026-08-24) is loose for the same reason
    ``project_dir`` is: it is not a per-JOB fact, so it has no home on
    ``Resources``.  It is the record of the machine this script is generated
    FOR, and it carries the one thing that differs between the machine that
    prepped and the machine that runs -- how a shell enters an environment
    there.  Reading that from local config instead is the 2026-08-24 Sol
    failure: `prep --target sol` baked `source /home/.../conda.sh` and every
    trial died on a path that exists only on the workstation
    (`preparing-for-another-machine.md` § 3).
    """
    import inspect
    assert set(inspect.signature(runwrap.render_run_wrapper).parameters) == {
        "script_path", "resources", "env", "n_atoms", "project_dir",
        "machine_record",
    }
    assert set(inspect.signature(runwrap.render_wrappers).parameters) == {
        "script_path", "resources", "env", "emit_sbatch", "project_dir",
        "machine_record",
    }
    assert set(inspect.signature(runwrap.write_run_wrapper).parameters) == {
        "script_path", "resources", "env", "emit_sbatch", "project_dir",
        "machine_record",
    }

"""The sweep's reader — trials' artifacts → ``bench-result.json``.

Moved ``bench/summarize.py`` → ``jobset/summarize.py`` 2026-08-12
(follow-up to the U-program): everything it reads is the JOB SET's —
`job-set.json` for discovery, `materialize` for the trial directories,
the winner's own deck for the mechanism — and its verdict feeds
`jobset prep run`.  A jobset reader living in `bench/` made the package
boundary lie about the dependency direction.

Serves ``molbuilder jobset summarize bench`` (step 6 u4): discovery keyed
by ``job-set.json``'s own data, each trial parsed with the pure parsers in
:mod:`molbuilder.bench.result`, the verdict written as a proposal of
WHAT WAS MEASURED -- the winning configuration and its knobs.  A wall
and a memory are NOT proposed: they were derived from a safety factor
and an assumed iteration count until 2026-08-24 and reached `sbatch`
through this file (`execution/project-layout.md` § 2.3.2).

The OLD half — ``discover_points``' directory-name regex,
``summarize_bundle``/``run_summarize`` over the shipped bundle format —
was DELETED 2026-08-12 (u5) with that lifecycle: the token is an
identifier, never a parser target (`job-contracts.md` § 6.3).
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..bench.result import (
    BenchPoint, BenchResult, build_bench_result, compare_asked_to_ran,
    machine_brief, machine_census, mismatch_phrase, parse_effective_run,
    parse_mpi_ranks,
)

_RUN_IDX = re.compile(r"-run(\d+)\.")


def _read(path: Path, *, tail: Optional[int] = None,
          head: Optional[int] = None) -> str:
    """Whole file, or its first ``head`` / last ``tail`` bytes.  The split
    matters: a SIESTA ``.out`` announces its launch (the rank count) in
    the first KB and its fate (the end-of-run markers) in the last —
    reading one window for both answers one of them wrong."""
    try:
        if head is not None:
            with path.open("rb") as fh:
                return fh.read(head).decode("utf-8", "replace")
        if tail is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > tail:
                fh.seek(-tail, 2)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _latest_run_file(d: Path, basename: str, suffix: str) -> Optional[Path]:
    """The ``<basename>-runN.<suffix>`` with the highest run index N."""
    cands = list(d.glob(f"{basename}-run*.{suffix}"))
    if not cands:
        return None

    def _idx(p: Path) -> int:
        m = _RUN_IDX.search(p.name)
        return int(m.group(1)) if m else -1
    return max(cands, key=_idx)


#: How far into a SIESTA ``.out`` the setup lines can sit.  The launch
#: header (``* Running on N nodes``) is in the first KB, but the parallel
#: grid and the eigensolver are printed AFTER the basis and pseudopotential
#: report -- ~49 KB into a real 42-atom run, and further for a bigger
#: system.  A 16 KB head window (the one that reads the rank count) sees
#: neither, so this is a deliberately generous but still bounded read.
_SETUP_WINDOW = 512 * 1024


def _wrapper_log(d: Path, basename: str) -> Path:
    """The most recent ``<basename>.runwrap-<stamp>.log`` in ``d``.

    The stamp is ``%Y%m%d-%H%M%S`` (``runwrap.py``), so lexical order IS
    chronological order.  Returns a non-existent path when there is none
    -- ``_read`` turns that into ``""`` and the caller simply learns less.
    """
    logs = sorted(d.glob(f"{basename}.runwrap-*.log"))
    return logs[-1] if logs else d / f"{basename}.runwrap-none.log"


def deck_value(deck: Path, keyword: str) -> Optional[str]:
    """The value an fdf deck gives ``keyword``, or ``None`` if absent.

    **First match wins, because that is what SIESTA does.** libfdf's
    ``fdf_locate`` walks from the first line and stops at the first
    label that matches (``do while ((.not. fdf_locate) .and. ...)``), so
    a deck that names a keyword twice is read with its FIRST value.

    One reader, because there were two and they disagreed: this function
    returned the first match while ``_winner_mechanism`` looped to the
    end and kept the LAST, so on a deck with a duplicated keyword the
    verdict named an algorithm SIESTA had not used -- and that verdict is
    what `prep run` offers to apply to the production calculation.
    Comparison is through ``_norm``, so ``Diag.Algorithm`` /
    ``diag_algorithm`` / ``DIAGALGORITHM`` are one keyword.
    """
    if not deck.is_file():
        return None
    want = _norm(keyword)
    for line in _read(deck).splitlines():
        toks = line.split("#", 1)[0].split()
        if len(toks) >= 2 and _norm(toks[0]) == want:
            return toks[1]
    return None


def _trial_deck(d: Path, basename: str) -> Path:
    """The deck a trial ran, beside its results."""
    return d / f"{basename}.fdf"


def parse_point(label: str, d: Path, basename: str, engine: str,
                knobs: Dict, point: Optional[Dict] = None) -> BenchPoint:
    """Parse one point's artifacts (in directory ``d``, output basename
    ``basename``) into a :class:`BenchPoint`."""
    metrics: Dict = {}
    knobs = dict(knobs)
    # What the JOB SET asked for, snapshotted BEFORE the recovery below
    # can fold an observation into it.  A CPU point whose job-set carries
    # no rank count has it recovered FROM the .out -- comparing that
    # against the same .out is comparing a number with itself, and the
    # empty result reads as "checked and matched" when nothing was
    # checked.  The asked side must never contain a measurement.
    asked = dict(knobs)

    # THROUGH THE REGISTRY, like every other file this project reads.
    # The wrapper's three instruments -- the SCF-timing tee, the monitor
    # log and the utilisation samples -- became registered parsers on
    # 2026-09-04 (`parse.md` § 5c).  Until then this module opened them
    # and regex'd their bytes itself: a second read stack for a class of
    # file the first one had simply never been extended to.
    #
    # THE LATEST ATTEMPT'S, like the .out beside them.  These carried no
    # run index until 2026-08-27, so a re-run appended to one and
    # truncated the other; now every per-run artifact is indexed and
    # every one is read the same way.
    from molbuilder.parse import parse as _parse
    from molbuilder.parse.instruments import utilisation as _utilisation

    def _metrics(path):
        """A registered parser's metrics, or ``{}``.

        Fail-soft like every other reader in this module: this is a
        reporter, and a trial whose instrument file is truncated or
        missing simply tells us less.
        """
        if path is None:
            return {}
        try:
            return dict(_parse(Path(path)).metrics)
        except Exception:                            # noqa: BLE001
            return {}

    metrics.update(_metrics(_latest_run_file(d, basename, "scf-timing.log")))

    _mon = _metrics(_latest_run_file(d, basename, "monitor.log"))
    bound = _mon.get("bound")
    machine: Dict = _mon.get("machine") or {}

    # Utilisation: the monitor's OWN means when it wrote them, the samples
    # otherwise -- ONE resolver decides, so the two sources cannot
    # disagree here (`parse/instruments/utilisation.py` owns the why;
    # user ruling 2026-09-03).
    util = _latest_run_file(d, basename, "util.csv")
    if util is not None:
        metrics.update(_utilisation(_mon, _metrics(util)))

    # The .out is read TWICE, one window each, because its two answers
    # live at opposite ends: the end-of-run markers in the tail, and the
    # "Running on N nodes" launch HEADER in the first KB.  Until U11
    # (2026-08-12) the ranks were searched in the tail window, so any run
    # whose .out outgrew 16 KB -- i.e. any real run -- silently lost its
    # rank count, and the verdict's CPU half had no np.
    out = _latest_run_file(d, basename, "out")
    out_tail = _read(out, tail=16384) if out is not None else ""
    out_head = _read(out, head=_SETUP_WINDOW) if out is not None else ""
    # ONE READER PER QUESTION (`model/parse.md` § 2b, P-S4).  This scanned
    # for its own `_DONE_MARKERS` tuple until 2026-08-25 -- a SECOND answer
    # to "did this run end", beside the engine parser that owns it.  They
    # disagreed exactly where it mattered: the private tuple's own comment
    # knew a capped bench with `SCF.MustConverge .false.` exits cleanly,
    # and the parser did not, so the bench summary rendered six healthy
    # trials as failures while the record beside them said completed.
    if out is None:
        state = "unknown"
    else:
        # The CHEAP door onto the one table (`parse/engines/_run_ending`,
        # `model/parse.md` § 2b).  Not the full parser: that builds Frames
        # -- positions and forces as numpy arrays -- and this needs one
        # string field.  Measured on a six-trial sweep of 152 KB files:
        # 272 ms through the full parse, 21 ms through the scan, on a
        # summary that polls every 15 s.  Same markers either way, because
        # both read `FATAL_MARKERS`.
        from ..parse.engines._run_ending import scan_ending
        state = ("completed"
                 if scan_ending(_read(out)).run_state == "ended"
                 else "incomplete")
    if engine == "cpu" and "mpi_np" not in knobs:
        n = parse_mpi_ranks(out_head)
        if n is not None:
            knobs["mpi_np"] = n

    # What the trial REALLY ran with, and whether that is what it was
    # asked to run.  Until 2026-08-13 nothing compared the two: the
    # requested knobs were written into the record as though they were
    # the measurement, so a silent fallback (ELPA -> the CPU solver, a
    # launcher handing back fewer ranks, OMP_NUM_THREADS set in the
    # environment) produced a row whose label described a run that never
    # happened -- and `choose_winner` ranked it against the others.
    effective = parse_effective_run(out_head, _read(_wrapper_log(d, basename)))
    deck = _trial_deck(d, basename)
    alg = deck_value(deck, "Diag.Algorithm")
    if alg is not None:
        asked["diag_algorithm"] = alg
    # The block size the deck REQUESTED, recorded beside the one SIESTA
    # settled on.  Both travel in the record because the pair is useful
    # benchmark data -- but they are not compared, and a divergence
    # never bars a trial: see the note on ``compare_asked_to_ran``'s
    # pairs.  SIESTA adapting the block size to the rank count is
    # documented behaviour, not a trial running something else.
    bs = deck_value(deck, "BlockSize")
    if bs is not None:
        try:
            effective["blocksize_asked"] = int(bs)
        except ValueError:
            pass                      # a non-numeric BlockSize: not ours to judge
    mismatch = compare_asked_to_ran(asked, effective)

    return BenchPoint(label=label, engine=engine, knobs=knobs,
                      metrics=metrics, bound=bound, state=state,
                      effective=effective, mismatch=mismatch,
                      point=dict(point or {}), machine=machine)



def _read_environment(bundle: Path) -> Dict:
    """The machine a result was measured on, as data for ``bench-result.json``.

    Through the one door (N1).  This was the SECOND reader of
    ``environment.json`` and it returned a raw ``dict`` where `prep`'s returned
    an ``Environment`` -- one file, two readers, two types.  It reads the
    object now and asks it for its own dict, so the shape a result records is
    the shape the record has.
    """
    from ..scheduler import FILENAME, read_environment
    env = read_environment(Path(bundle) / FILENAME)
    return env.to_dict() if env is not None else {}


def _norm(key: str) -> str:
    return key.lower().replace(".", "").replace("-", "").replace("_", "")


def _read_system(bundle: Path) -> Dict:
    """Minimal system descriptor (engine + atom count).

    From the DESCRIPTION first: ``task.json``'s witness records exactly
    these two facts, and a described sweep always has one.  Until
    2026-08-12 this looped over ``job-gpu.fdf`` / ``job-cpu.fdf`` — the
    OLD bench-bundle deck names, whose writer died with the fold (step 6
    u5) — so every described sweep's ``bench-result.json.system``
    silently degraded to ``{"engine": "siesta"}`` (2026-08-12 plan A9).
    Description-less
    bundles fall back to any root deck; like every reader in this
    module, absence degrades rather than raises — this is a reporter.
    """
    from ..task import FILENAME, read_task
    desc = Path(bundle) / FILENAME
    if desc.is_file():
        try:
            t = read_task(desc)
            return {"engine": t.engine, "n_atoms": t.structure.atoms}
        except Exception:
            pass    # malformed description: fall through to the decks
    sysd: Dict = {"engine": "siesta"}
    # Decks are born in their stage directories since the layout repair
    # (roadmap 7.10 M1); the bare-root pattern stays first for a bundle
    # prepped before it.  This is already the degraded path (a malformed
    # description), so breadth beats precision here.
    _decks = (list(Path(bundle).glob("*.fdf"))
              + list(Path(bundle).glob("*/*.fdf")))
    for fdf in sorted(_decks):
        for line in _read(fdf).splitlines():
            toks = line.split("#", 1)[0].split()
            if len(toks) >= 2 and _norm(toks[0]) == "numberofatoms":
                try:
                    sysd["n_atoms"] = int(float(toks[1]))
                except ValueError:
                    pass
                break
        if "n_atoms" in sysd:
            break
    return sysd


def discover_points_from_jobset(bundle, jobset) -> List[BenchPoint]:
    """Discovery keyed by the DATA floor 3 wrote — the fold's reader
    (plan step 6 u4).

    Each trial's directory comes from the naming authority
    (``job_dir_names``) and its knobs from the job's own ``resources`` —
    never by parsing a directory name back (`job-contracts.md` § 6.3: the
    token is an identifier, not a parser target).

    **There is no stage parameter** (U12, 2026-08-12).  A described
    sweep's job-set lives in its stage's ``bench/`` container (U1), so
    the set IS the scope — every job in it belongs to the stage that was
    named at the surface.  The ``stage`` filter that stood here was a raw
    suffix match (``tok.endswith(f"_{{stage}}")`` — `coarse` matched
    `extra_coarse`) that bypassed the one stage resolver, and after U1
    its only reachable use was the description-less fallback, whose jobs
    carry no tokens at all: a stage arg filtered out EVERY point and
    wrote an empty verdict.  A filter whose only reachable answer is
    wrong is retired, not repaired.

    Its regex-keyed predecessor ``discover_points`` died with the OLD
    bundle format (u5).
    """
    from .materialize import job_dir_names, run_dir, shape_of
    bundle = Path(bundle)
    dirs = job_dir_names(jobset, shape_of(jobset, bundle))
    pts: List[BenchPoint] = []
    for j in jobset.jobs:
        # Knobs speak the EXCHANGE vocabulary -- the job-set's own field
        # names (jobset/model.Resources) -- because the choice they feed
        # goes straight back into an allocation (U13, 2026-08-12).  They
        # were renamed to grid words here (ranks/cores_per_rank) and
        # renamed BACK by the offer: two renames for nothing, and a third
        # vocabulary for one fact (job-contracts § 6 note: one language).
        knobs: Dict = {}
        if j.resources.mpi_np:
            knobs["mpi_np"] = j.resources.mpi_np
        if j.resources.cpus_per_task:
            knobs["cpus_per_task"] = j.resources.cpus_per_task
        if j.resources.gres:
            knobs["gres"] = j.resources.gres
        # THE LATEST ATTEMPT WHERE THERE IS ONE, the container otherwise --
        # `runstatus`'s own rule, and shape-agnostic, so it answered for a
        # hierarchical stage long before a trial had attempts to find
        # (`project-layout.md` § 1.5a).  Reading `dirs[j.name]` regardless
        # is blind to the attempt layer: a re-measured trial's artifacts
        # sit one level down.
        _d = bundle / dirs[j.name]
        pts.append(parse_point(
            j.name, run_dir(_d), Path(j.script).stem,
            "gpu" if j.resources.gres else "cpu", knobs,
            point=dict(j.point)))
    return pts


def _winner_mechanism(bundle, jobset, label: str) -> Dict:
    """HOW the winning trial computed -- read from ITS OWN deck, never
    re-derived (U13b, 2026-08-12).  The offer used to pin
    ``diag_algorithm='ELPA-1STAGE'`` from ``engine == 'gpu'`` alone --
    inventing the mechanism the measurement never named, and wrong the
    day the grid grows a second eigensolver.  The deck the trial RAN is
    on disk beside its results; its BENCH-MARKS block says gpu_mode and
    its body says the algorithm."""
    job = next((j for j in jobset.jobs if j.name == label), None)
    if job is None:
        return {}
    from .materialize import job_dir_names, run_dir, shape_of
    import os as _os
    _c = Path(bundle) / job_dir_names(jobset, shape_of(jobset, bundle))[label]
    d = run_dir(_c)                     # the one rule, in the layout layer
    deck = d / _os.path.basename(job.script)
    text = _read(deck)
    if not text:
        return {}
    mech: Dict = {}
    from ..parse.scripts.bench_marks import _extract_bench_marks_dict
    marks = _extract_bench_marks_dict(text) or {}
    if "gpu_mode" in marks:
        mech["use_gpu"] = str(marks["gpu_mode"]).lower() == "true"
    # Through the ONE deck reader.  This loop kept the LAST match while
    # `deck_value` takes the first -- two readers of one file, disagreeing
    # on a deck that names the keyword twice, and this is the copy whose
    # answer `prep run` offers to apply to the production calculation.
    alg = deck_value(deck, "Diag.Algorithm")
    if alg is not None:
        mech["diag_algorithm"] = alg
    return mech


def bench_record(jobset, bundle, *, now_iso: Optional[str] = None
                 ) -> BenchResult:
    """The sweep's record, built ONE way — points, then the verdict, then
    HOW the winner computed.

    **Both readers of a sweep come through here**, and that is the whole
    point.  ``run_summarize_jobset`` (which writes ``bench-result.json``)
    and ``sweep_view`` (which the Results tab reads) each used to compose
    this themselves, with the same four arguments — and then differed:
    only the writing path enriched ``choice`` with ``_winner_mechanism``.
    So the same sweep produced two different verdicts depending on which
    door you came through, which is exactly the second path
    `bench-summary.md` B2 forbids, in the pair of functions that cite it.
    """
    res = build_bench_result(
        discover_points_from_jobset(bundle, jobset),
        environment=_read_environment(Path(bundle)),
        system=_read_system(Path(bundle)),
        now_iso=now_iso)
    # HOW the winner computed, read from its own deck (U13b) -- part of the
    # verdict, not a decoration on the file that happens to be written.
    if res.choice.get("label"):
        mech = _winner_mechanism(bundle, jobset, res.choice["label"])
        if mech:
            res.choice["mechanism"] = mech
    return res


def run_summarize_jobset(jobset, bundle, *,
                         out=None, now_iso: Optional[str] = None,
                         stage: Optional[str] = None):
    """Summarize a described sweep: write ``bench-result.json`` (the
    archival record) and RETURN the report for the caller to print.

    Returns ``(BenchResult, out_path, report_text)``, where ``report_text``
    is ``None`` when there is no verdict to report.

    **The report is printed, not written** *(2026-09-04, user ruling)*.  It
    was ``bench-recommendation.txt`` beside the record; measured across the
    whole project tree, ZERO had ever been written.  The use it was built
    for -- submit a sweep to a cluster, come back to the directory and read
    the answer -- is served by asking, which is what a terminal is for, and
    a report nothing consumes is a print.  `job-system.md` § 7.1.

    The record stays because it is the sweep's ARCHIVAL trace: the trials,
    their numbers, the machine each ran on, and the verdict, surviving the
    trials' artifacts being archived.  Note that nothing reads it back --
    the panel recomputes through ``sweep_view`` -- so if archiving a sweep
    is not a use anybody has, this write is the next thing to retire.
    """
    res = bench_record(jobset, bundle, now_iso=now_iso)
    out_path = Path(out) if out else Path(bundle) / "bench-result.json"
    out_path.write_text(res.to_json() + "\n", encoding="utf-8")
    return res, out_path, recommendation_text(res, stage=stage)



# `RECOMMENDATION_NAME = "bench-recommendation.txt"` stood here until
# 2026-09-04.  The report is printed by `jobset summarize` now, not
# written: zero of those files had ever been produced across the whole
# project tree (`job-system.md` § 7.1).  It had been `run-config.toml`,
# an editable TOML `prep run` folded into the launch, until 2026-09-02 --
# so this is the second half of the same retirement.

#: Catalogue item type -> the python type its TOML value must carry.
#: Non-scalar types (lists, text) are absent on purpose: nothing sweeps
#: them, so no proposal writes them.
_ITEM_TYPES = {"bool": bool, "int": int, "pow2": int, "float": float,
               "str": str, "enum": str}


def _pins_vocabulary(engine: str) -> Dict[str, type]:
    """The [pins] section's legal fields: every non-machine ``execution``
    item of this engine, typed from its catalogue declaration -- the SAME
    one-door membership rule the declaration lane uses
    (`_declared_execution_pins`, `generator.md` § 4.3a).  A hand table
    (``use_gpu``/``diag_algorithm``) stood here until 2026-08-21, so
    the proposal writer could emit a swept knob -- ``block_size`` -- that
    this reader then refused: summarize's own output failing `prep run`.
    """
    from ..template import catalogue, select
    vocab: Dict[str, type] = {}
    for it in select(catalogue(), engine=engine):
        if "execution" not in (it.category or ()) or it.allocation:
            continue
        py = _ITEM_TYPES.get(it.type)
        if py is not None:
            vocab[it.name] = py
    return vocab


def recommendation_text(res: BenchResult, *, stage: Optional[str] = None
                        ) -> Optional[str]:
    """The benchmark's REPORT, printed by ``jobset summarize``, or ``None``
    when the result concludes nothing.

    **Nothing reads this file.**  It is what the sweep found, said in
    sentences, for a person to read and act on -- and the action is writing
    an ``execution`` block in ``task.json``, which is the only thing
    ``prep run`` consults (`architecture.md` § 5.2).

    It was ``run-config.toml`` until 2026-09-02: an editable TOML the next
    ``prep run`` folded into the launch, labelled *"recommendation, not
    decision"* while functioning as a decision.  That made a benchmark a
    second way for a run parameter to arrive, and a second arrival route is
    what every silent-value defect in this lane has been.  *(User ruling:
    "the run parameter needs to be explicitly decided/written … benchmark
    recommendation should be named such that it is understood not as a user
    input but for result presentation.")*

    So this writes the exact JSON to paste, rather than a file to edit: the
    measurement stops one step short of the launch, and that step is a person
    deciding, visibly, in the file that records decisions.
    """
    choice = res.choice or {}
    if not choice:
        return None
    knobs = choice.get("knobs") or {}
    mech = choice.get("mechanism") or {}
    stage_word = stage or "<stage>"

    # "NOTHING READS THIS FILE" until 2026-09-04, when it stopped being a
    # file.  The sentence still has a job: nothing APPLIES this, and the
    # line exists so a reader does not assume the next run will use it.
    out = [f"molbuilder bench recommendation -- {stage_word}",
           "NOTHING APPLIES THIS.  It is what the benchmark found; the "
           "decision is yours to write.",
           ""]
    # The rationale already reads as a sentence ("G1K4C6 fastest (2.3
    # s/iter); vs ..."), so it is not labelled again.
    rationale = choice.get("rationale") or choice.get("label")
    if rationale:
        out += [f"  {rationale}", ""]

    # WHAT TO WRITE -- the `execution` block that would use this winner.
    # `omp_threads` and `gpu_count` are `execution`'s names for what the
    # record calls `cpus_per_task` and a `gres` string (`_cli._AS_RESOURCE`);
    # naming them here in the RECORD's vocabulary would hand over a block
    # `task.json` refuses.
    block: Dict = {}
    if knobs.get("mpi_np") is not None:
        block["mpi_np"] = int(knobs["mpi_np"])
    if knobs.get("cpus_per_task") is not None:
        block["omp_threads"] = int(knobs["cpus_per_task"])
    if knobs.get("gres"):
        try:
            from ..scheduler.quantities import parse_gres
            n = sum(parse_gres(str(knobs["gres"])).values())
            if n:
                block["gpu_count"] = int(n)
        except Exception:                                     # noqa: BLE001
            pass
    # THE VALUE AXES TOO, from the winner's own POINT -- not only from
    # `mechanism`.  `mechanism` is HOW the winner computed (the eigensolver,
    # the device); a value axis like `block_size` is a coordinate of the
    # grid and lives in `point`.  Reading only `mechanism` dropped the
    # measured `block_size` from a winner whose own label was
    # `G0K4C1block_size128` -- a report that names a shape while silently
    # omitting one of its measured coordinates sends a person to run at
    # something nobody benchmarked.  (Caught by `test_value_axes.py`, which
    # is the file that exists for exactly this axis, 2026-09-02.)
    # `res.system["engine"]` -- the description's own answer, put there by
    # `_system_block` from `read_task`.  This read `getattr(res, "engine")`
    # until 2026-09-04, and `BenchResult` has no such attribute: measured
    # `hasattr(...) is False`, so the `or "siesta"` fired every time and a
    # `getattr` was doing the work of a hardcoded literal while looking
    # dynamic.  Harmless so far only because the bench lane refuses any
    # non-SIESTA description by name (`_cli.py`) -- but this report is
    # written for a person to read, so the day that lane admits PySCF it
    # would offer SIESTA's pin vocabulary for a PySCF run.
    vocab = _pins_vocabulary((res.system or {}).get("engine") or "siesta")
    for src in ((choice.get("point") or {}), (mech or {})):
        for name, val in sorted(src.items()):
            if name in vocab:
                block[name] = val

    if mech:
        out += ["  it computed with"]
        for name, val in sorted(mech.items()):
            out += [f"      {name} = {val!r}"]
        out += [""]

    if block:
        out += ["To run at it, put this in task.json and save:", ""]
        import json as _json
        body = _json.dumps(block, indent=2, sort_keys=True).split("\n")
        out += ['    "execution": ' + body[0]]
        out += ["    " + ln for ln in body[1:]]
        out += [""]
    out += ["Until you do, `prep run` sizes the launch from the target's own",
            "width (architecture.md 5.2) -- a benchmark does not steer a run.",
            ""]
    return "\n".join(out)


def _fmt_wall(seconds: float) -> str:
    """``41`` -> ``41s``, ``245`` -> ``4m05s``, ``7523`` -> ``2h05m``."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _point_table(points: List[BenchPoint]):
    """The measurement table: header line + one row per point, columns
    sized to their content.  What was ASKED (np/thr/gpu) sits beside what
    was MEASURED (s/iter, wall, peak memory, mean utilisation) so the
    scaling is readable across rows; ``algorithm`` is what the trial
    ACTUALLY ran (``effective``), so a silent eigensolver fallback shows
    in the table itself.  A value nothing measured prints ``--``; the
    GPU columns appear only when the sweep has any GPU signal.
    """
    # GPU columns key off what the sweep ASKED to be (engine/gres), not
    # off metric presence: the monitor samples gpu0_* as zeros on a
    # GPU-less run, and three columns of 0 on a CPU sweep are noise --
    # while a genuine GPU trial showing sm% 0 is exactly worth seeing.
    gpu = any(p.engine == "gpu" or p.knobs.get("gres") for p in points)
    # The machine column keys off signal the same way: present when any
    # trial recorded one (`generator.md` § 4.4b -- shown per trial), absent
    # for records predating the [MACHINE] line rather than a column of --.
    machined = any(p.machine for p in points)

    def _num(v, fmt="{:g}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "--"

    cols = [
        ("point", "l", lambda p: p.label or "--"),
        *([("machine", "l", lambda p: machine_brief(p.machine) or "--")]
          if machined else []),
        ("np", "r", lambda p: _num(p.knobs.get("mpi_np"))),
        ("thr", "r", lambda p: _num(p.knobs.get("cpus_per_task"))),
        *([("gpu", "l", lambda p: str(p.knobs.get("gres") or "--"))]
          if gpu else []),
        ("algorithm", "l",
         lambda p: str(p.effective.get("diag_algorithm") or "--")),
        ("s/iter", "r", lambda p: _num(p.s_per_iter())),
        ("iters", "r", lambda p: _num(p.metrics.get("iters_measured"))),
        ("wall", "r",
         lambda p: (_fmt_wall(p.metrics["wall_s"])
                    if isinstance(p.metrics.get("wall_s"), (int, float))
                    else "--")),
        ("peak-mem", "r",
         lambda p: _num(p.metrics.get("peak_rss_gb"), "{:.1f}G")),
        ("cpu%", "r",
         lambda p: _num(p.metrics.get("cpu_mean_pct"), "{:.0f}")),
        *([("gpu-sm%", "r",
            lambda p: _num(p.metrics.get("gpu_sm_mean_pct"), "{:.0f}")),
           ("vram", "r",
            lambda p: _num(p.metrics.get("gpu_vram_peak_gb"), "{:.1f}G"))]
          if gpu else []),
        ("bound", "l", lambda p: p.bound or "--"),
        ("state", "l", lambda p: p.state),
    ]
    header = [name for name, _, _ in cols]
    body = [[fn(p) for _, _, fn in cols] for p in points]
    widths = [max(len(header[i]), *(len(r[i]) for r in body))
              if body else len(header[i]) for i in range(len(cols))]

    def _row(cells):
        return ("  " + "  ".join(
            cells[i].rjust(widths[i]) if cols[i][1] == "r"
            else cells[i].ljust(widths[i])
            for i in range(len(cols)))).rstrip()

    return _row(header), [_row(r) for r in body]


def summary_text(res: BenchResult, out_path: Path, *,
                 report: Optional[str] = None,
                 stage: Optional[str] = None) -> str:
    """The verb's stdout: the measurement table, the verdict, THE REPORT,
    and what to do next.  ``report`` is ``run_summarize_jobset``'s third
    return; ``stage`` names the stage in the next-commands.
    """
    lines = ["bench-summarize: measured points (fastest first)"]
    ranked = sorted(
        res.points,
        key=lambda p: (p.s_per_iter() if p.s_per_iter() is not None
                       else float("inf")))
    head, rows = _point_table(ranked)
    lines.append(head)
    for p, row in zip(ranked, rows):
        lines.append(row)
        # A trial that ran something else is not a slower trial, it is a
        # different experiment.  Say so on its own line rather than
        # leaving the reader to find it in the JSON.
        if p.mismatch:
            lines.append(f"    !! ran something other than asked: "
                         f"{mismatch_phrase(p.mismatch)} "
                         f"-- excluded from the choice")
    # WHICH MACHINES, said plainly when there is more than one kind
    # (`generator.md` § 4.4b).  A statement, not a warning: a mixed
    # CPU/GPU sweep spans machines by construction and may be exactly the
    # intended experiment -- the reader judges the comparison, this line
    # makes sure they know it is one.  ONE census (`machine_census`),
    # shared with `sweep_view`'s payload.
    census = machine_census(res.points)
    if len(census) > 1:
        parts = ", ".join(
            f"{brief} ({n} trial{'s' if n != 1 else ''})"
            for brief, n in census)
        lines.append(
            f"  trials ran on {len(census)} kinds of node: {parts}")

    # "Every timed trial" must be a claim about the TIMED set (R2-2):
    # `any(p.mismatch)` over ALL points fired this sentence when nothing
    # had been timed at all -- one unfinished point with mismatch data
    # and the summary asserted a census it never took.
    # "timed" is choose_winner's own definition (completed AND timed):
    # a mid-flight point can carry a parsed s_per_iter while still
    # incomplete, and counting it here once flipped the verdict line to
    # "no completed, timed trial" beside a mismatch row it had just
    # printed.
    _timed = [p for p in res.points
              if p.state == "completed" and p.s_per_iter() is not None]
    if res.choice:
        lines.append(f"  winner: {res.choice.get('rationale')}")
    elif _timed and all(p.mismatch for p in _timed):
        lines.append("  NO WINNER: every timed trial ran something other "
                     "than it was asked to.  The times are real but they do "
                     "not measure the settings on their labels -- fix the "
                     "cause and re-run before trusting a choice.")
    else:
        # B4's sibling on THIS surface: an empty verdict is said, with the
        # state census that explains it -- "no winner" and "no trial has
        # run yet" are different situations wearing one empty dict.
        by_state = {}
        for p_ in res.points:
            by_state[p_.state] = by_state.get(p_.state, 0) + 1
        census = ", ".join(f"{n} {s}" for s, n in sorted(by_state.items()))
        lines.append(f"  NO VERDICT: no completed, timed trial to rank "
                     f"({census}).  Submit trials and summarize again.")
    # The coverage clause (honesty on a partial sweep): a verdict drawn
    # from three of eleven prepped points says so on its face.
    if res.choice:
        timed = sum(1 for p_ in res.points
                    if p_.state == "completed" and p_.s_per_iter() is not None)
        if timed < len(res.points):
            lines.append(f"  coverage: {timed} of {len(res.points)} prepped "
                         f"points measured -- the verdict ranks what ran.")
    lines.append(f"  wrote: {out_path}  (the record)")
    # THE CONNECTION SURFACE (roadmap § 0.1 B5): the summary ends with what
    # to do, not only what was found.  What to do is WRITE THE DECISION --
    # the report says what, and nothing applies it for you (§ 2.3.2).
    if res.choice:
        stage_word = stage or "<stage>"
        # THE REPORT ITSELF, here on stdout.  It used to be written beside
        # the record and this block told you to go and read it; nobody ever
        # did, because zero were ever written.  You asked the question, so
        # the answer belongs in the answer.
        if report:
            lines.append("")
            lines.append(report.rstrip())
            lines.append("")
        lines.append("  next:")
        lines.append(f"    1. put the `execution` block above in task.json  "
                     f"# stages[{stage_word}].execution, or calculation-wide")
        lines.append(f"    2. molbuilder jobset prep run {stage_word}"
                     f"     # uses what you wrote, and nothing else")
        lines.append(f"    3. molbuilder jobset launch run {stage_word}"
                     f" --mode submit|direct")
    return "\n".join(lines)


#: How far above a sweep's ``job-set.json`` its calculation root can sit.
#: The deepest documented layout is ``<calc>/<NN>_<stage>/bench/`` -- two --
#: and the bound keeps a failed search from walking to ``/`` and matching
#: something it has no business matching.
_BUNDLE_SEARCH_DEPTH = 4


def bundle_for_sweep_file(jobset, path) -> Path:
    """Which CALCULATION a sweep's ``job-set.json`` belongs to.

    **The file's own directory is NOT the bundle**, and reading it as one
    fails silently rather than loudly: ``job_dir_names`` hands out
    directories relative to the calculation root (``01_coarse/bench/
    bench-G1K1C1``), so resolving them against the file's directory points
    at paths that do not exist -- every trial then reports ``unknown``,
    every measurement is ``None``, and the sweep shows no verdict while
    looking like a sweep that simply has not run yet.

    Rather than climb a fixed number of levels -- which would encode one
    of the three layouts `materialize.bench_container` supports -- ask the
    naming authority where the trials should be and walk up until they are
    actually THERE.  The answer is checked against the disk, so a wrong
    guess cannot be returned as a right one.
    """
    from .materialize import job_dir_names, shape_of
    start = Path(path).parent
    candidates = [start, *list(start.parents)[:_BUNDLE_SEARCH_DEPTH]]
    for cand in candidates:
        try:
            dirs = job_dir_names(jobset, shape_of(jobset, cand))
        except Exception:
            continue
        if dirs and all((cand / d).is_dir() for d in dirs.values()):
            return cand
    raise ValueError(
        f"could not find the calculation {Path(path).name} belongs to: no "
        f"directory at or above {start} holds this sweep's trial directories")


def sweep_view(jobset, bundle) -> Dict:
    """The whole sweep, composed for a READER: the record a summarize
    would write, beside where every trial is right now.

    Contract: ``docs/web/bench-summary.md``.  Its **B1** is why this
    function is a composer and not an analysis: every figure here is
    produced by a door that already owns it --

      * :func:`discover_points_from_jobset` -- which trials, their knobs
        and their sweep coordinates;
      * :func:`~molbuilder.bench.result.build_bench_result` -- the record,
        including ``choice``: the VERDICT, which is
        :func:`~molbuilder.bench.result.choose_winner`'s answer and
        already refuses to crown a trial that ran something other than
        what it was asked for;
      * :func:`~molbuilder.jobset.runstatus.jobset_status` -- queued /
        running / failed, which ``BenchPoint.state`` does NOT answer (that
        field describes the artifacts on disk, not the run's position).

    Nothing is recomputed here.  **B2** is the reason that matters:
    ``submission.md`` § 3 records a summary that showed "170 minutes" for
    five 38-minute jobs because it worked out its own total a second way,
    and a view comparing six trials has six chances to repeat it.

    Read-only: unlike :func:`run_summarize_jobset` this writes nothing, so
    it is safe to call while the sweep is still running -- which is
    exactly when a person watches it.

    Returns a JSON-ready dict.  ``trials`` is one entry per job, in the
    job-set's own order.
    """
    from .runstatus import jobset_status

    bundle = Path(bundle)
    # ONE record-builder, shared with `run_summarize_jobset` -- so the
    # verdict this view shows is the verdict that gets written, down to
    # `choice["mechanism"]`.  Composing it here a second time is what B2
    # forbids, and is what this function did until 2026-08-25.
    res = bench_record(jobset, bundle, now_iso=utc_now_iso())
    points = res.points
    status = jobset_status(jobset, bundle)

    # POSITION is the join, and it has to be: a BenchPoint's label is the
    # JOB's name while a StageStatus's name comes from its StageRef -- the
    # STAGE -- and for a sweep trial those are different strings.  Both
    # readers walk ``jobset.jobs`` in order and emit exactly one entry per
    # job, so the zip is sound; the guard is here so that if either ever
    # learns to skip one, this raises instead of quietly pairing trial N's
    # measurement with trial N+1's state.
    if len(points) != len(status.stages):
        raise ValueError(
            f"the sweep's two readers disagree on how many trials it has "
            f"({len(points)} measured, {len(status.stages)} statuses); "
            f"refusing to pair them by position")

    trials = []
    for p, st in zip(points, status.stages):
        trials.append({
            "label":      p.label,
            "engine":     p.engine,
            "point":      dict(p.point),
            "knobs":      dict(p.knobs),
            "effective":  dict(p.effective),
            "mismatch":   dict(p.mismatch),
            "metrics":    dict(p.metrics),
            "s_per_iter": p.s_per_iter(),
            "bound":      p.bound,
            # What kind of node was under the run, and its one short
            # spelling -- SPELLED HERE, because the page composes nothing
            # (B1): a second brief-rule in JS would be two spellings of
            # one node.  "" when the record predates the [MACHINE] line.
            "machine":       dict(p.machine),
            "machine_brief": machine_brief(p.machine),
            "artifacts":  p.state,      # what the files say
            "state":      st.state,     # where the RUN is (§ 2's door)
            "detail":     st.detail,
            "dir":        st.dir,
            "attempts":   list(st.attempts),
        })

    # WHICH MACHINES are in play -- THE census (`machine_census`, the
    # same one the terminal statement reads), as JSON.  The page states
    # it when there is more than one and never judges it (`generator.md`
    # § 4.4b, bench-summary.md B5).
    machines = [{"brief": brief, "trials": n}
                for brief, n in machine_census(points)]

    return {
        "name":         status.name,
        "engine":       status.engine,
        "kind":         getattr(jobset, "kind", ""),
        "complete":     status.complete,
        "generated_at": res.generated_at,
        "environment":  res.environment,
        "system":       res.system,
        # The verdict, whole -- including the empty {} that means "nothing
        # timed, or every timed trial ran something other than its label".
        "choice":       res.choice,
        "varied":       swept_coordinates(points),
        "machines":     machines,
        "trials":       trials,
        "n_trials":     len(trials),
        "n_done":       sum(1 for s in status.stages if s.state == "finished"),
    }


def swept_coordinates(points: List[BenchPoint]) -> List[str]:
    """Which coordinate keys this sweep actually VARIED, ascending.

    Selection, not measurement: it reads each trial's ``point`` (the
    coordinate the sweep declared -- `generator.md` § 4.3a) and keeps the
    keys that do not hold the same value everywhere.  A view plots the
    verdict axis against one of these; a sweep that varied nothing
    comparable gets ``[]`` and is shown as a table rather than a chart of
    one column (bench-summary.md § 3).
    """
    keys = sorted({k for p in points for k in (p.point or {})})
    varied = []
    for k in keys:
        seen = {repr(p.point.get(k)) for p in points if p.point}
        if len(seen) > 1:
            varied.append(k)
    return varied


def utc_now_iso() -> str:
    """UTC timestamp ``YYYY-MM-DDThh:mm:ssZ`` (moved from the deleted
    ``bench/prep.py`` at u5 -- its one surviving caller is the summarize
    verb's stamp)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "parse_point", "discover_points_from_jobset", "run_summarize_jobset",
    "sweep_view", "swept_coordinates", "bundle_for_sweep_file",
    "bench_record",
    "summary_text", "utc_now_iso",
    "recommendation_text",
]

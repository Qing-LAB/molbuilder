"""The sweep's reader — trials' artifacts → ``bench-result.json``.

Everything it reads is the JOB SET's — `job-set.json` for discovery,
`materialize` for the trial directories, the winner's own deck for the
mechanism — and its verdict is a report a person reads, and writes into
`task.json` (nothing applies it: there is no second rung, `job-system.md`
§ 7.1).

Serves ``molbuilder jobset summarize bench`` (step 6 u4): discovery keyed
by ``job-set.json``'s own data, each trial parsed with the pure parsers in
:mod:`molbuilder.bench.result`, the verdict written as a proposal of
WHAT WAS MEASURED -- the winning configuration and its knobs.  A wall
and a memory are NOT proposed (`execution/project-layout.md` § 2.3.2).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..bench.result import (
    BenchPoint, BenchResult, build_bench_result, compare_asked_to_ran,
    machine_brief, machine_census, mismatch_phrase, parse_effective_run,
)



def _read(path: Path, *, head: Optional[int] = None) -> str:
    """Whole file, or its first ``head`` bytes -- where a SIESTA ``.out``
    announces its launch (the rank count)."""
    try:
        if head is not None:
            with path.open("rb") as fh:
                return fh.read(head).decode("utf-8", "replace")
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


#: How far into a SIESTA ``.out`` the setup lines can sit.  The launch
#: header (``* Running on N nodes``) is in the first KB, but the parallel
#: grid and the eigensolver are printed AFTER the basis and pseudopotential
#: report -- ~49 KB into a real 42-atom run, and further for a bigger
#: system.  A 16 KB head window (the one that reads the rank count) sees
#: neither, so this is a deliberately generous but still bounded read.
_SETUP_WINDOW = 512 * 1024


def deck_value(deck: Path, keyword: str) -> Optional[str]:
    """The value an fdf deck gives ``keyword``, or ``None`` if absent.

    **First match wins, because that is what SIESTA does.** libfdf's
    ``fdf_locate`` walks from the first line and stops at the first
    label that matches (``do while ((.not. fdf_locate) .and. ...)``), so
    a deck that names a keyword twice is read with its FIRST value.

    Comparison is through ``_norm``, so ``Diag.Algorithm`` /
    ``diag_algorithm`` / ``DIAGALGORITHM`` are one keyword.
    """
    if not deck.is_file():
        return None
    # One deck reader: `parse/fdf.py`.  It also tracks `%block`, so a
    # block's first token is never read as a keyword.
    from ..parse.fdf import _parse_fdf
    scalars, _blocks = _parse_fdf(_read(deck))
    got = scalars.get(_norm(keyword))
    return got[0] if got else None


def parse_point(label: str, run, engine: str,
                knobs: Dict, point: Optional[Dict] = None) -> BenchPoint:
    """Parse one point's artifacts -- the files of its ``run``, the run
    door's `Run` (``None``: no run of ours there yet) -- into a
    :class:`BenchPoint`.  Every file is taken at the run's ONE index, so
    a point's figures never mix two runs."""
    def _at_run(role):
        return run.file(role, run.run) if run is not None else None
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
    # log and the utilisation samples -- are registered parsers
    # (`parse.md` § 5c).
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

    # SECONDS PER ITERATION from the trial's stamped SCF rows, by the one
    # rule every surface uses (`scf_timing_rows.timing_of`, `model/parse.md`
    # § 5c): the SIESTA family's tee, else a PySCF trial's progress log,
    # whose rows its deck stamps.  The tee's registered parser wraps the same
    # function.
    from molbuilder.parse.instruments.scf_timing_rows import timing_of
    _timing = (_at_run(".scf-timing.log")
               or (run.file(".molwatch.log") if run is not None else None))
    metrics.update(timing_of(_timing))

    _mon = _metrics(_at_run(".monitor.log"))
    bound = _mon.get("bound")
    machine: Dict = _mon.get("machine") or {}

    # Utilisation: the monitor's OWN means when it wrote them, the samples
    # otherwise -- ONE resolver decides, so the two sources cannot
    # disagree here (`parse/instruments/utilisation.py` owns the why;
    # user ruling 2026-09-03).
    util = _at_run(".util.csv")
    if util is not None:
        metrics.update(_utilisation(_mon, _metrics(util)))

    # The .out answers two questions.
    #
    #   * HOW IT ENDED -- `ending_of`, the one door (`model/parse.md`
    #     § 2b, P-S4), which reads the whole file: the SCF-convergence
    #     markers appear once per cycle rather than at the end.  Frames are
    #     not built for it -- this needs one string field, on a summary
    #     that polls every 15 s.
    #   * THE RANK COUNT -- the "Running on N nodes" launch banner, in
    #     the first KB.
    out = run.stdout if run is not None else None
    out_head = _read(out, head=_SETUP_WINDOW) if out is not None else ""
    state = "unknown"
    if out is not None:
        from ..parse.engines._run_ending import ending_of
        try:
            state = ("completed" if ending_of(out).run_state == "ended"
                     else "incomplete")
        except OSError:
            pass
    # THE KNOBS ARE WHAT WAS ASKED, and only that: the ranks the run got
    # are `effective`'s (below, SIESTA's own count).  Every trial states its
    # counts (`generator.md` § 4.3a).

    # What the trial REALLY ran with, and whether that is what it was
    # asked to run: a silent fallback (ELPA -> the CPU solver, a launcher
    # handing back fewer ranks, OMP_NUM_THREADS set in the environment)
    # must not yield a row whose label describes a run that never happened.
    _wlog = run.session_log if run is not None else None
    effective = parse_effective_run(out_head,
                                    _read(_wlog) if _wlog is not None else "")
    deck = run.deck if run is not None else None
    alg = deck_value(deck, "Diag.Algorithm") if deck is not None else None
    if alg is not None:
        asked["diag_algorithm"] = alg
    # The block size the deck REQUESTED, recorded beside the one SIESTA
    # settled on.  Both travel in the record because the pair is useful
    # benchmark data -- but they are not compared, and a divergence
    # never bars a trial: see the note on ``compare_asked_to_ran``'s
    # pairs.  SIESTA adapting the block size to the rank count is
    # documented behaviour, not a trial running something else.
    bs = deck_value(deck, "BlockSize") if deck is not None else None
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

    Through the one door (N1): it reads the object and asks it for its own
    dict, so the shape a result records is the shape the record has.
    """
    from ..scheduler import read_environment
    from ..scheduler.record import calculation_record
    env = read_environment(calculation_record(bundle))
    return env.to_dict() if env is not None else {}


#: FDF KEYWORD MATCHING, FROM THE MODULE THAT OWNS IT.  `parse/fdf.py::_norm`
#: is fdf's real rule -- case-insensitive and blind to ``.``, ``-`` and ``_``,
#: so SIESTA reads ``MeshCutoff``, ``Mesh.Cutoff`` and ``mesh_cutoff`` as one
#: keyword.
from ..parse.fdf import _norm   # noqa: E402  -- fdf's own matching rule


def _read_system(bundle: Path) -> Dict:
    """Minimal system descriptor (engine + atom count).

    From the DESCRIPTION first: ``task.json``'s witness records exactly
    these two facts, and a described sweep always has one.
    Description-less bundles fall back to any root deck; like every reader in this
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
    # Decks are born in their stage directories.  This is already the
    # degraded path (a malformed description), so breadth beats precision
    # here.
    from ..runfiles import find_by_role
    _root = Path(bundle)
    try:
        _subs = [s for s in _root.iterdir() if s.is_dir()]
    except OSError:
        # A MISSING BUNDLE DEGRADES.  `find_by_role` tolerates one; a bare
        # `iterdir()` does not.
        _subs = []
    _decks = (find_by_role(_root, ".fdf")
              + [d for sub in _subs for d in find_by_role(sub, ".fdf")])
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

    **There is no stage parameter**: a described sweep's job-set lives in
    its stage's bench container, so the set IS the scope — every job in it
    belongs to the stage that was named at the surface.
    """
    from .materialize import job_dir_names, run_dir, shape_of
    from .model import gpu_request
    bundle = Path(bundle)
    dirs = job_dir_names(jobset, shape_of(jobset, bundle))
    pts: List[BenchPoint] = []
    for j in jobset.jobs:
        # Knobs speak the EXCHANGE vocabulary -- the job-set's own field
        # names (jobset/model.Resources) -- because the choice they feed
        # goes straight back into an allocation (job-contracts § 6 note:
        # one language).
        # ALL THREE, every trial -- a CPU trial's `gres` is null, not
        # absent.  The GPU request goes
        # through the one door (`model.gpu_request`): its side and its
        # count are one answer.
        gpus = gpu_request(j.resources)
        knobs: Dict = {"mpi_np": j.resources.mpi_np,
                       "cpus_per_task": j.resources.cpus_per_task,
                       "gres": gpus.gres if gpus.uses else None}
        # THE LATEST ATTEMPT WHERE THERE IS ONE, the container otherwise
        # (`run_dir`, `project-layout.md` § 1.5a): a re-measured trial's
        # artifacts sit one level down.
        _d = bundle / dirs[j.name]
        from ..runs import run_of
        pts.append(parse_point(
            j.name, run_of(run_dir(_d)),
            "gpu" if gpus.uses else "cpu", knobs,
            point=dict(j.point)))
    return pts


def _winner_mechanism(bundle, jobset, label: str) -> Dict:
    """HOW the winning trial computed -- read from ITS OWN deck, never
    re-derived (U13b).  The deck the trial RAN is
    on disk beside its results; its BENCH-MARKS block says gpu_mode and
    its body says the algorithm."""
    job = next((j for j in jobset.jobs if j.name == label), None)
    if job is None:
        return {}
    from .materialize import job_dir_names, run_dir, shape_of
    from ..runs import run_of
    _c = Path(bundle) / job_dir_names(jobset, shape_of(jobset, bundle))[label]
    run = run_of(run_dir(_c))          # the trial's run, the run door's
    deck = run.deck if run is not None else None
    text = _read(deck) if deck is not None else ""
    if not text:
        return {}
    mech: Dict = {}
    from ..script_emit import _extract_bench_marks_dict
    marks = _extract_bench_marks_dict(text) or {}
    if "gpu_mode" in marks:
        mech["use_gpu"] = str(marks["gpu_mode"]).lower() == "true"
    # Through the ONE deck reader.
    alg = deck_value(deck, "Diag.Algorithm")
    if alg is not None:
        mech["diag_algorithm"] = alg
    return mech


def bench_record(jobset, bundle, *, now_iso: Optional[str] = None
                 ) -> BenchResult:
    """The sweep's record, built ONE way — points, then the verdict, then
    HOW the winner computed.

    **Both readers of a sweep come through here** --
    ``run_summarize_jobset`` (which writes ``bench-result.json``) and
    ``sweep_view`` (which the Results tab reads) -- so one sweep has one
    verdict whichever door you came through (`bench-summary.md` B2).
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

    **The report is printed, not written** *(2026-09-04, user ruling)*.  The
    use it serves -- submit a sweep to a cluster, come back to the directory and read
    the answer -- is served by asking, which is what a terminal is for, and
    a report nothing consumes is a print.  `job-system.md` § 7.1.

    The record stays because it is the sweep's ARCHIVAL trace: the trials,
    their numbers, the machine each ran on, and the verdict, surviving the
    trials' artifacts being archived.  Nothing reads it back -- the panel
    recomputes through ``sweep_view`` -- and it is kept for that archival
    reason alone (settled 2026-09-06, user: *"keep it -- archiving a sweep
    IS a use"*, `job-system.md` § 7.1).
    """
    res = bench_record(jobset, bundle, now_iso=now_iso)
    from ..runfiles import BENCH_RESULT_FILE
    out_path = Path(out) if out else Path(bundle) / BENCH_RESULT_FILE
    out_path.write_text(res.to_json() + "\n", encoding="utf-8")
    return res, out_path, recommendation_text(res, stage=stage)



#: Catalogue item type -> the python type its TOML value must carry.
#: Non-scalar types (lists, text) are absent on purpose: nothing sweeps
#: them, so no proposal writes them.
_ITEM_TYPES = {"bool": bool, "int": int, "pow2": int, "float": float,
               "str": str, "enum": str}


def _pins_vocabulary(engine: str) -> Dict[str, type]:
    """The [pins] section's legal fields: every non-machine ``execution``
    item of this engine, typed from its catalogue declaration -- the SAME
    one-door membership rule the declaration lane uses
    (`_declared_execution_pins`, `generator.md` § 4.3a).
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

    **Nothing applies it.**  It is what the sweep found, said in
    sentences, for a person to read and act on -- and the action is writing
    an ``execution`` block in ``task.json``, which is the only thing
    ``prep task`` consults (`architecture.md` § 5.2).  *(User ruling:
    "the run parameter needs to be explicitly decided/written … benchmark
    recommendation should be named such that it is understood not as a user
    input but for result presentation.")*

    So this writes the exact JSON to paste, rather than a file to edit: the
    measurement stops one step short of the launch, and that step is a person
    deciding, visibly, in the file that records decisions.
    """
    if not res.choice.get("label"):
        return None
    choice = res.choice
    knobs = choice["knobs"]
    mech = choice.get("mechanism") or {}
    stage_word = stage or "<stage>"

    # Nothing APPLIES this, and the line exists so a reader does not assume
    # the next run will use it.
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
    # record calls `cpus_per_task` and a `gres` string (`model.AS_RESOURCE`);
    # naming them here in the RECORD's vocabulary would hand over a block
    # `task.json` refuses.
    block: Dict = {}
    if knobs.get("mpi_np") is not None:
        block["mpi_np"] = int(knobs["mpi_np"])
    if knobs.get("cpus_per_task") is not None:
        block["omp_threads"] = int(knobs["cpus_per_task"])
    if knobs["gres"]:
        # The trial's own request, written by prep (`gpu:<n>`): it parses,
        # or the record is not ours -- a parse error is said, never passed
        # over.
        from ..scheduler.quantities import parse_gres
        n = sum(parse_gres(str(knobs["gres"])).values())
        if n:
            block["gpu_count"] = int(n)
    # THE VALUE AXES TOO, from the winner's own POINT -- not only from
    # `mechanism`.  `mechanism` is HOW the winner computed (the eigensolver,
    # the device); a value axis like `block_size` is a coordinate of the
    # grid and lives in `point`.
    # `res.system["engine"]` -- the description's own answer, put there by
    # `_read_system` from `read_task`.
    vocab = _pins_vocabulary(res.system["engine"])
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
    out += ["Until you do, `prep task` launches at what task.json states, and",
            "refuses a launch value it states nowhere (architecture.md 5.2) --",
            "a benchmark does not steer a run.",
            ""]
    return "\n".join(out)


def _fmt_duration(seconds: float) -> str:
    """``41`` -> ``41s``, ``245`` -> ``4m05s``, ``7523`` -> ``2h05m``.

    It formats a DURATION, not a wall clock (`model/parse.md` § 2a).
    """
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
    in the table itself.  A value nothing measured prints ``--``.  EVERY
    COLUMN, EVERY SWEEP: a CPU trial's GPU columns say ``no gpu`` and
    ``--``, and a trial with no recorded machine ``--``.
    """

    def _num(v, fmt="{:g}"):
        return fmt.format(v) if isinstance(v, (int, float)) else "--"

    cols = [
        ("point", "l", lambda p: p.label or "--"),
        ("machine", "l", lambda p: machine_brief(p.machine) or "--"),
        ("np", "r", lambda p: _num(p.knobs.get("mpi_np"))),
        ("thr", "r", lambda p: _num(p.knobs.get("cpus_per_task"))),
        # A TRIAL THAT ASKED FOR NO GPU says so, as the page does
        # (`bench-summary.js`): `--` is a value nothing measured, and this
        # is an asked column.
        ("gpu", "l", lambda p: str(p.knobs["gres"] or "no gpu")),
        ("algorithm", "l",
         lambda p: str(p.effective.get("diag_algorithm") or "--")),
        ("s/iter", "r", lambda p: _num(p.s_per_iter())),
        ("iters", "r", lambda p: _num(p.metrics.get("iters_measured"))),
        # NAMED AFTER THE FIELD IT PRINTS: it is the MONITORED WINDOW, not
        # the job's wall time, and on a trial the scheduler killed it is a
        # lower bound.
        ("monitored", "r",
         lambda p: (_fmt_duration(p.metrics["monitored_elapsed_s"])
                    if isinstance(p.metrics.get("monitored_elapsed_s"), (int, float))
                    else "--")),
        ("peak-mem", "r",
         lambda p: _num(p.metrics.get("mem_peak_gb"), "{:.1f}G")),
        ("cpu%", "r",
         lambda p: _num(p.metrics.get("cpu_mean_pct"), "{:.0f}")),
        ("gpu-sm%", "r",
         lambda p: _num(p.metrics.get("gpu_sm_mean_pct"), "{:.0f}")),
        ("vram", "r",
         lambda p: _num(p.metrics.get("gpu_vram_peak_gb"), "{:.1f}G")),
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
                 stage: Optional[str] = None, base=None) -> str:
    """The verb's stdout: the measurement table, the verdict, THE REPORT,
    and what to do next.  ``report`` is ``run_summarize_jobset``'s third
    return; ``stage`` names the stage in the next-commands and ``base`` the
    calculation they act on (`commands`).
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

    # THE VERDICT AS THE RECORD SAYS IT -- the winner, or the one sentence
    # saying why there is none (`bench.result.choose_winner`).
    if res.choice.get("label"):
        lines.append(f"  winner: {res.choice['rationale']}")
    else:
        lines.append(f"  no winner: {res.choice['none']}.")
    # The coverage clause (honesty on a partial sweep): a verdict drawn
    # from three of eleven prepared points says so on its face.
    if res.choice.get("label"):
        timed = sum(1 for p_ in res.points
                    if p_.state == "completed" and p_.s_per_iter() is not None)
        if timed < len(res.points):
            lines.append(f"  coverage: {timed} of {len(res.points)} prepared "
                         f"points measured -- the verdict ranks what ran.")
    lines.append(f"  wrote: {out_path}  (the record)")
    # THE CONNECTION SURFACE (roadmap § 0.1 B5): the summary ends with what
    # to do, not only what was found.  What to do is WRITE THE DECISION --
    # the report says what, and nothing applies it for you (§ 2.3.2).
    if res.choice.get("label"):
        stage_word = stage or "<stage>"
        # THE REPORT ITSELF, here on stdout: you asked the question, so the
        # answer belongs in the answer.
        if report:
            lines.append("")
            lines.append(report.rstrip())
            lines.append("")
        lines.append("  next:")
        lines.append(f"    1. put the `execution` block above in task.json, "
                     f"as stages[{stage_word}].execution or calculation-wide")
        if stage:
            # ONE COMMAND A LINE, the step's sentence above it (`commands`):
            # the calculation named when it is known, the mode stated where
            # its config sets none.
            from .commands import command, launch_lines, words_for
            lines.append("    2. prep the stage's run -- it uses what you "
                         "wrote, and nothing else:")
            lines.append("         " + command("prep", *words_for("task", stage),
                                                base=base))
            lines.append("    3. then launch it:")
            lines += ["         " + ln
                      for ln in launch_lines("task", stage, base=base)]
        else:
            lines.append("    2. prep the stage's run, then launch it -- "
                         "`prep task` reads what you wrote, and nothing else")
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
    of the layouts `paths.bench_container` supports -- ask the
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
    # forbids.
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
            # one node.  "" when the record has no [MACHINE] line.
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
        # The verdict, whole -- the winner, or `none`: why there is none.
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
    """UTC timestamp ``YYYY-MM-DDThh:mm:ssZ`` -- the summarize verb's stamp,
    and :func:`sweep_view`'s."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "parse_point", "discover_points_from_jobset", "run_summarize_jobset",
    "sweep_view", "swept_coordinates", "bundle_for_sweep_file",
    "bench_record",
    "summary_text", "utc_now_iso",
    "recommendation_text",
]

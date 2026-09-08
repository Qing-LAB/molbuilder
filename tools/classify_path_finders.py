#!/usr/bin/env python3
"""Every place molbuilder SEARCHES for a path, and the door that NAMES it.

WHY THIS IS A TOOL AND NOT A TABLE IN A DOCUMENT.  `plans/plan.md` § 5k is the
plan; its § 5k.1 table was measured by hand on 2026-09-08 and will be wrong by
the time anyone acts on it.  § 5a is the rule -- *a row is evidence of when it
was written; re-derive before acting* -- so the survey ships as something you
run.

WHAT IT LOOKS FOR.  A search is `glob`, `rglob`, `iterdir`, `os.listdir`,
`os.scandir` or `fnmatch`.  A search is INTERESTING when its pattern spells a
name molbuilder itself composes: the run-file grammar (`job-contracts.md`
§ 2.2a) and the directory layout (`project-layout.md` § 1).  Searching for
`*.psml` or `*.md` is not interesting -- those names belong to somebody else
and no door of ours composes them.

WHY IT MATTERS.  Every one of those names has a door that BUILDS it and, with
one exception, none that FINDS it, so a caller with a question spells a
pattern and the layout rule gains a site.  This counts the sites and names the
door each one should have asked.

    python tools/classify_path_finders.py            # the summary
    python tools/classify_path_finders.py --list owned
    python tools/classify_path_finders.py --json

OVERRIDES ARE KEYED BY (file, enclosing function, pattern) -- NEVER BY LINE.
`classify_source_reads.py` keyed its reasons by line number and two of them
came unanchored the first time tests were deleted around them, one landing on
an unrelated assertion (2026-09-08).  A reason someone wrote after reading a
site must survive an edit to the lines above it.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "molbuilder"

SEARCH_ATTRS = {"glob", "rglob", "iterdir", "listdir", "scandir"}

#: What OUR names look like, and the door that composes each.  A pattern that
#: matches none of these is somebody else's name.
OWNED = [
    ("-run",            "runfiles.compose(run=…) / runfiles.tail",
     "the -run<N> counter (job-contracts.md § 6.3)"),
    ("run-",            "materialize.resolve_attempt / materialize.attempts",
     "the attempt directory (project-layout.md § 1.5)"),
    ("bench-",          "materialize.trial_dir",
     "a trial's directory"),
    ("bench_",          "materialize.bench_container",
     "the flat bench container"),
    ("/bench",          "materialize.bench_container",
     "the hierarchical bench container"),
    ("job-set.json",    "jobset.model.FILENAME (+ materialize.sweep_set_paths)",
     "the set a prep derives"),
    ("bench-result",    "summarize.run_summarize_jobset",
     "the verdict"),
    ("environment.json", "scheduler.record.FILENAME",
     "the machine record"),
    ("task.json",       "task.FILENAME", "the description"),
    (".fdf",            "runfiles.WRITTEN / runfiles.compose",
     "a SIESTA deck"),
    (".run.sh",         "runfiles.WRITTEN", "the wrapper"),
    (".sbatch",         "runfiles.WRITTEN", "the header"),
    (".molwatch.log",   "runfiles.WRITTEN", "the trajectory log"),
    (".concluded",      "runfiles.WRITTEN", "the conclusion marker"),
    (".molstruct.json", "runfiles.WRITTEN", "the structure sidecar"),
    ("runwrap-",        "runfiles.WRITTEN", "the wrapper's own log"),
    ("{label}",         "runfiles.stem", "the calculation's own file stem"),
    ("<molbuilder.jobset.model.FILENAME>", "jobset.model.FILENAME",
     "the set a prep derives"),
    (".template.toml",  "template.FILENAME / template_path",
     "the answers file"),
    (".source.xyz",     "runfiles.WRITTEN", "the structure pair"),
    (".XV",             "runfiles.WRITTEN", "a SIESTA restart coordinate"),
]

#: Names that are not ours: a third party's, or the language's.
FOREIGN_HINTS = (".psml", ".md", "*.py", "python*", ".dist-info", "x3dna",
                 ".MD.nc", ".AVTRANS", ".json\"" )

#: (file, function, pattern) -> (verdict, reason).  Stable across edits: an
#: edit above a site moves its line, never its function or its pattern.
#: Every entry was READ before it was written -- the syntactic pass above is a
#: first guess and these are the corrections, each with why.
_OVERRIDES: dict[tuple[str, str, str], tuple[str, str]] = {
    # -- the finders that already exist.  Not sites to migrate: they are what
    #    the framework is meant to have more of.
    ("molbuilder/jobset/materialize.py", "sweep_set_paths", "*/{_JS}"):
        ("door - a finder, not a caller",
         "the counterpart of `bench_container`, added 2026-09-08"),
    ("molbuilder/jobset/materialize.py", "sweep_set_paths", "*/*/{_JS}"):
        ("door - a finder, not a caller", "the hierarchical half of the same"),
    ("molbuilder/scheduler/record.py", "named_environments", "*.json"):
        ("door - a finder, not a caller",
         "`environments/<name>.json` -- this IS the door every caller asks, "
         "and the name it searches for is the file's own"),

    # -- ours, and the search is hand-spelled
    ("molbuilder/transport/compose.py", "classify_citation", "*.xyz"):
        ("owned - a door composes this name",
         "the structure PAIR: the two lines below pair each hit with its "
         "`.molstruct.json`, which is `runfiles.WRITTEN`'s sidecar"),
    ("molbuilder/transport/record.py", "collect_record", "*.out"):
        ("owned - a door composes this name",
         "engine stdout, which the wrapper names `<basename>-run<N>.out`"),
    ("molbuilder/validation/identity.py", "_foreign_state", "*{suffix}"):
        ("owned - a door composes this name",
         "THE CLEANEST CASE IN THE SURVEY: `suffix` already comes from the "
         "one rules file (`_engine_inventory` -> `_warm_inventory`, U3), so "
         "the VOCABULARY asks and only the SEARCH is spelled by hand.  The "
         "gap this framework closes, in one function"),
    ("molbuilder/web/blueprints/watch.py", "_resolve_run_directory",
     "os.path.join(directory, '*.out')"):
        ("owned - a door composes this name",
         "engine stdout again, through `os.path.join`, which the pattern "
         "reader does not unwrap"),

    # -- ours, but a different grammar: the workspace store, not run files
    ("molbuilder/web/blueprints/workspace_storage.py", "_state_indices",
     "{ws_id}.*.wc.json"):
        ("owned - workspace store (a separate grammar)",
         "the browser workspace's own files; same fault, different vocabulary "
         "-- out of scope for the run-file framework, in scope for the rule"),
    ("molbuilder/web/blueprints/workspace_storage.py", "_warn_if_residue_piling",
     "*.wc.json"):
        ("owned - workspace store (a separate grammar)", "the same store"),

    # -- somebody else's names
    ("molbuilder/envs/_cli.py", "_du", "*"):
        ("foreign - not a name we compose",
         "a byte count over a conda env: matches everything, names nothing"),
    ("molbuilder/envs/abi.py", "installed_package_version", "{name}-*.json"):
        ("foreign - not a name we compose", "conda-meta's own naming"),
    ("molbuilder/envs/doctor.py", "_read_conda_meta", "*.json"):
        ("foreign - not a name we compose", "conda-meta again"),
}


def _module_strings(tree: ast.AST) -> dict[str, str]:
    """Module-level ``NAME = "literal"``, so an f-string can be resolved.

    OUR NAMES ARE USUALLY NOT SPELLED IN THE SEARCH.  The first pass matched
    literal patterns and called `glob(f"*/{_JS}")` unclassified -- while `_JS`
    is `job-set.json`, one of the most owned names there is.  A survey that
    misses a name because the caller imported the constant would report the
    codebase as tidier than it is, and exactly where it is least tidy: a
    caller that knows the filename has a home, and still assembles the search
    by hand.
    """
    out: dict[str, str] = {}
    for n in getattr(tree, "body", []):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) \
           and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = n.value.value
        if isinstance(n, ast.ImportFrom):
            for al in n.names:
                # `from .model import FILENAME as _JS` -- the value is not here,
                # but the NAME tells us which door owns it.
                out.setdefault(al.asname or al.name, f"<{n.module}.{al.name}>")
    return out


def _pattern_of(node: ast.Call, consts: dict[str, str]) -> str:
    if not node.args:
        return ""
    a = node.args[0]
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return a.value
    if isinstance(a, ast.JoinedStr):
        parts = []
        for v in a.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name):
                parts.append(consts.get(v.value.id, "{" + v.value.id + "}"))
            else:
                parts.append("{?}")
        return "".join(parts)
    try:
        return ast.unparse(a)
    except Exception:                                    # pragma: no cover
        return "<?>"


def _enclosing(tree: ast.AST, node: ast.AST) -> str:
    best = "<module>"
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.lineno <= node.lineno <= (n.end_lineno or n.lineno):
                best = n.name
    return best


def survey() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(PKG.rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                              # pragma: no cover
            continue
        consts = _module_strings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if name not in SEARCH_ATTRS:
                continue
            pat = _pattern_of(node, consts)
            where = _enclosing(tree, node)
            owner = why = None
            for needle, door, what in OWNED:
                if needle in pat:
                    owner, why = door, what
                    break
            if owner:
                verdict = "owned - a door composes this name"
            elif not pat:
                verdict = "listing - no pattern (iterdir/scandir)"
            elif any(h in pat for h in FOREIGN_HINTS):
                verdict = "foreign - not a name we compose"
            elif not any(c in pat for c in "*?["):
                verdict = "exact - a single named file"
            else:
                verdict = "unclassified - READ IT"
            over = _OVERRIDES.get((rel, where, pat))
            if over:
                verdict, why = over
            rows.append(dict(file=rel, line=node.lineno, func=where,
                             call=name, pattern=pat, verdict=verdict,
                             composer=owner, names=why))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", metavar="PREFIX",
                    help="every site whose verdict starts with PREFIX")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = survey()
    if args.json:
        json.dump(rows, sys.stdout, indent=1)
        return 0
    if args.list:
        for r in rows:
            if r["verdict"].startswith(args.list):
                print(f'{r["file"]}:{r["line"]}  {r["func"]}()')
                print(f'    {r["call"]}({r["pattern"]!r})')
                if r["composer"]:
                    print(f'    -> ask {r["composer"]}   ({r["names"]})')
        return 0
    buckets: dict[str, int] = {}
    for r in rows:
        buckets[r["verdict"]] = buckets.get(r["verdict"], 0) + 1
    print(f"path searches in molbuilder/ : {len(rows)}")
    for k in sorted(buckets):
        print(f"  {k:44} {buckets[k]:4}")
    owned = [r for r in rows if r["verdict"].startswith("owned")]
    print(f"\nOWNED -- a door composes the name, none finds it: {len(owned)}")
    per: dict[str, list[str]] = {}
    for r in owned:
        # An override states the verdict and the REASON; it does not restate
        # the composer, so group those under the reason instead of sorting
        # `None` against a string (which is how this first ran).
        key = r["composer"] or f'(read: {r["names"]})'
        per.setdefault(key, []).append(f'{r["file"]}:{r["line"]}')
    for door in sorted(per):
        print(f"  {door}")
        for site in per[door]:
            print(f"      {site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

#: What OUR names look like, and the door that composes each.
#:
#: DERIVED FROM THE CATALOGUE, not copied beside it.  This was a hand-written
#: list until 2026-09-08 and it had already drifted: it claimed ``.XV`` was
#: ours, and ``.XV`` is SIESTA's own restart file.  `runfiles.WRITTEN` says
#: what MOLBUILDER writes and deliberately does not enumerate what an engine
#: writes -- *"an engine's output set depends on its version and on which
#: options are on, so enumerating THAT is a snapshot pretending to be a
#: rule"*.  A survey with its own copy of the vocabulary is the very habit it
#: exists to find.
def _owned_rules() -> "list[tuple[str, str, str]]":
    import sys
    sys.path.insert(0, str(ROOT))
    from molbuilder.jobset.materialize import TRIAL_PREFIX
    from molbuilder.paths import ATTEMPT_PREFIX
    from molbuilder.runfiles import WRITTEN
    rules = [
        (ATTEMPT_PREFIX, "paths.attempt_dir / paths.attempts_in",
         "the attempt directory (project-layout.md § 1.5)"),
        (TRIAL_PREFIX, "materialize.trial_dir / materialize.trials_in",
         "a trial's directory"),
        ("-run", "runfiles.compose(run=…) / runfiles.find",
         "the -run<N> counter (job-contracts.md § 6.3)"),
    ]
    # Every role the catalogue declares, longest first so `.source.xyz` is
    # recognised before `.xyz` -- which is NOT a role of ours and must not be
    # matched by one.
    for a in sorted(WRITTEN, key=lambda x: len(x.role), reverse=True):
        door = ("runfiles.find_by_role" if a.role.startswith(".")
                else "runfiles.find(dir, label, role=…)")
        rules.append((a.role, door, a.what))
    return rules


OWNED = _owned_rules()

#: Names that are not ours: a third party's, or the language's.
FOREIGN_HINTS = (".psml", ".md", "*.py", "python*", ".dist-info", "x3dna",
                 ".MD.nc", ".AVTRANS", ".json\"" )

#: (file, function, pattern) -> (verdict, reason).  Stable across edits: an
#: edit above a site moves its line, never its function or its pattern.
#: Every entry was READ before it was written -- the syntactic pass above is a
#: first guess and these are the corrections, each with why.
_OVERRIDES: dict[tuple[str, str, str], tuple[str, str]] = {
    # -- THE FINDERS.  Not sites to migrate: they are what § 4.5 asks for.
    # (`sweep_set_paths`' two globs were exempted here as a DOOR from
    #  2026-09-08 until N4 on 2026-09-09.  They are gone, not moved: it now
    #  asks `paths.bench_containers_in`, which is `bench_container`'s search
    #  half and did not exist when the exemption was written.  A door that
    #  spells the layout it searches is a door only until the layout has a
    #  finder -- which is what this tool's own `--check` said by failing when
    #  the entries came unanchored.)
    ("molbuilder/scheduler/record.py", "named_environments", "*.json"):
        ("door - a finder, not a caller",
         "`environments/<name>.json` -- this IS the door every caller asks, "
         "and the name it searches for is the file's own"),
    ("molbuilder/validation/identity.py", "warm_files_present", "{label}*"):
        ("door - a finder, not a caller",
         "THE SUBTRACTION ITSELF (`job-contracts.md` § 4.2): everything named "
         "after the label that `runfiles.is_ours` does not claim came from the "
         "engine.  It searches for OUR name in order to answer about what is "
         "NOT ours, and it is the door `check_id_change` and `runstatus` ask"),
    ("molbuilder/web/blueprints/workspace_storage.py", "_state_indices",
     "{ws_id}.*.wc.json"):
        ("door - a finder, not a caller",
         "A SELF-CONTAINED GRAMMAR THAT ALREADY OBEYS § 4.5.  The browser "
         "workspace store composes with `_state_path` and finds with this, "
         "both from the one `_STATE_SUFFIX` in the same module, and nothing "
         "outside the module spells `.wc.json` (checked 2026-09-08: the four "
         "hits in `workspace/dispatcher.js` are prose)"),
    ("molbuilder/web/blueprints/workspace_storage.py",
     "_warn_if_residue_piling", "*.wc.json"):
        ("door - a finder, not a caller",
         "the same store counting its own residue, through the same constant"),

    # -- DOOR-FED: the pattern is a PARAMETER, and its one producer is a door.
    #    A survey that reads syntax cannot see that; each of these was read.
    ("molbuilder/parse/dirs/job.py", "_enumerate_files", "match"):
        ("door-fed - the pattern comes from a door",
         "`match` is `paths.Shape.stage_glob` -- WHICH FILES ARE THIS RUNG'S, "
         "answered by the layout layer.  The per-suffix bucketing below it is "
         "parser dispatch over a vocabulary that is half the ENGINE's "
         "(`.XV`, `.STRUCT_OUT`, `.ANI`), which `runfiles.WRITTEN` "
         "deliberately does not carry"),
    ("molbuilder/projects.py", "find_geom_candidates", "pattern"):
        ("door-fed - the pattern comes from a door",
         "`_geom_output_patterns()` takes both PySCF spellings from "
         "`pyscf.input.ROLE_OPTIMIZED` / `ROLE_GEOM_TRAJ` (their one home, "
         "from `pyscf/warm-files.toml`).  WHICH engine outputs count as a "
         "startable geometry is this picker's own curation -- the rules file "
         "has no field for it -- so the selection stays and the spellings ask"),
    ("molbuilder/web/blueprints/watch.py", "_resolve_run_directory",
     "os.path.join(directory, optim_glob)"):
        ("door-fed - the pattern comes from a door",
         "`optim_glob` is `'*' + ROLE_GEOM_TRAJ`, the declared constant.  It "
         "cannot go through `runfiles.find_by_role`, and that refusal is the "
         "grammar's own rule rather than a gap: `_geom_optim.xyz` is an "
         "UNDERSCORE role, and without a label a trailing `_geom_optim.xyz` "
         "cannot be told from a stage token named `..._geom_optim` with `.xyz` "
         "as the role (`runfiles.parse`).  Refused rather than answered wrongly"),

    # -- SOMEBODY ELSE'S NAMES.  molbuilder composes none of these, so § 4.5
    #    gives them no door: *"for every name IT COMPOSES, the framework owns
    #    the search."*  What an engine writes is not knowable by enumeration
    #    (`job-contracts.md` § 4.2), which is why `runfiles.WRITTEN` is the
    #    list that CAN be complete and stops where our own writing stops.
    ("molbuilder/transport/compose.py", "classify_citation", "*.XV"):
        ("foreign - not a name we compose",
         "SIESTA's own restart file -- the geometry it saves and reloads.  "
         "The survey's hand-written vocabulary claimed this one as ours until "
         "2026-09-08, which is why the table is derived from `WRITTEN` now"),
    ("molbuilder/transport/compose.py", "classify_citation", "*.xyz"):
        ("foreign - not a name we compose",
         "A PERSON'S STRUCTURE FILE.  `WRITTEN` declares `.source.xyz` and "
         "`_initial.xyz`; a bare `.xyz` in a cited directory is whatever the "
         "user put there.  Its SIDECAR is ours, and the line below pairs each "
         "hit through `sidecars.molstruct.sidecar_path_for`"),
    ("molbuilder/validation/identity.py", "_foreign_state", "*{suffix}"):
        ("foreign - not a name we compose",
         "THE ENGINE'S WARM-RESTART SUFFIXES, and the vocabulary already comes "
         "from its one home (`warmfiles.inventory`, U3) -- what is spelled "
         "here is only the loop over it.  No door, because molbuilder does not "
         "compose `.XV` or `.DM`; `find_by_role` refuses a role outside "
         "`WRITTEN` on purpose, so that a typo is a refusal and not an empty "
         "list.  The question is also the mirror of `warm_files_present`: "
         "files keyed by SOME OTHER id, which needs the suffix list to tell an "
         "orphaned restart file from an unrelated one"),
    ("molbuilder/envs/_cli.py", "_du", "*"):
        ("foreign - not a name we compose",
         "a byte count over a conda env: matches everything, names nothing"),
    ("molbuilder/envs/abi.py", "installed_package_version", "{name}-*.json"):
        ("foreign - not a name we compose",
         "conda-meta's own naming, read to learn a package's version"),
    ("molbuilder/envs/doctor.py", "_read_conda_meta", "*.json"):
        ("foreign - not a name we compose",
         "conda-meta again -- the same directory, asked what is installed"),
}

#: The verdicts a site may NOT have once the migration is done.  `--check`
#: fails on either, and `tests/test_path_framework.py` runs `--check`.
#:
#: `owned` means a door composes the name and this caller spelled it anyway.
#: `unclassified` means nobody has read the site -- which is the same defect
#: one step earlier, because an unread site cannot be known to be either.
FAILING_VERDICTS = ("owned", "unclassified")

#: The undeclared hierarchy segments as of § 5l.6 N1 (2026-09-08).  `--check`
#: fails when this SET changes -- a third invented segment is a new level of the
#: tree created at a call site, which `project-layout.md` § 2.6 alone may do.
#: Shrinking it is the point (N5); growing it is the failure.
GUARDED_UNDECLARED = ("launch", "pseudos")



# --------------------------------------------------------------------------- #
#  THE OTHER HALF: a name BUILT by hand.                                      #
# --------------------------------------------------------------------------- #
#
# Everything above looks at SEARCHES, and a duplicate COMPOSER is invisible to
# it.  That is how `materialize.attempt_concluded` came to spell
# `f"{basename}-run{newest}.concluded"` on the line AFTER asking `latest_run`
# for that very counter, and how `submit.py` built the path
# `f"{names[j]}/run-{n}"` and handed it to `prepare_attempt` as the attempt to
# continue FROM.  Both measured 2026-09-08 -- by reviewing the search
# migration's own diff, not by any check.
#
# WHY THIS RULE IS DELIBERATELY NARROW.  Matching "a string that ends in a
# catalogued role" finds 35 sites of which two are real: `.clone.log`,
# `serve-<port>.log` and `jobset-decisions.log` all end in `.log` and belong to
# other grammars entirely.  A check with 33 exemptions is a nag list, not a
# guard -- and a nag list is how a real finding gets scrolled past.
#
# So it matches the two fragments of the grammar that carry RULES rather than
# just a spelling: the wrapper's `-run<N>` counter (`job-contracts.md` § 6.3 --
# a hyphen announces a COUNTER, an underscore a NAME) and the attempt
# directory's `run-<N>` prefix (`project-layout.md` § 1.5).  Both have exactly
# one composer, both are keyed on a NUMBER, and a hand-built one is how an
# off-by-one or a renamed prefix reaches disk.  Everything else about a name is
# the role, and the role is what `WRITTEN` and `find_by_role` already guard.
#
# WHAT IT CANNOT SEE, and this is a limit rather than a caveat: a name spelled
# inside a script molbuilder EMITS.  `siesta/makov_payne.py` carries
# `glob("*-run*.out")` as template TEXT for a script that ships beside a job --
# data here, code there, and no AST pass over `molbuilder/` can reach it.  That
# one is recorded in `plans/plan.md` § 5k: it cannot use `runfiles` until
# `runfiles` joins `runwrap.MONITOR_COMPANIONS`.

#: Grammar fragments with one composer, keyed on a number.
COUNTER_FRAGMENTS = ("-run", "run-")

#: The modules that ARE the grammar; they compose these for a living.
GRAMMAR_MODULES = ("molbuilder/runfiles.py", "molbuilder/paths.py")

#: (file, function, unparsed f-string) -> reason.  Same anchoring rule as
#: `_OVERRIDES`: (file, function, text), never a line number.
_COMPOSE_OVERRIDES: "dict[tuple[str, str, str], str]" = {}


def compositions(pkg: "pathlib.Path | None" = None,
                 root: "pathlib.Path | None" = None) -> "list[dict]":
    """Every f-string that builds a counter-keyed name by hand.

    An f-string qualifies when a literal piece ENDS with one of
    :data:`COUNTER_FRAGMENTS` and the next piece formats a value -- the number
    interpolated straight after the fragment, which is exactly what
    `runfiles.compose(run=N)` and `paths.attempt_name(n)` exist to do.
    """
    pkg = PKG if pkg is None else pkg
    root = ROOT if root is None else root
    rows: "list[dict]" = []
    for path in sorted(pkg.rglob("*.py")):
        rel = str(path.relative_to(root))
        if rel in GRAMMAR_MODULES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                              # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            vals = node.values
            fragment = None
            for i, v in enumerate(vals[:-1]):
                if (isinstance(v, ast.Constant)
                        and isinstance(v.value, str)
                        and v.value.endswith(COUNTER_FRAGMENTS)
                        and isinstance(vals[i + 1], ast.FormattedValue)):
                    fragment = v.value
                    break
            if fragment is None:
                continue
            try:
                text = ast.unparse(node)
            except Exception:                            # pragma: no cover
                text = "<?>"
            where = _enclosing(tree, node)
            rows.append(dict(
                file=rel, line=node.lineno, func=where, fragment=fragment,
                text=text,
                door=("runfiles.compose(run=N)" if fragment.endswith("-run")
                      else "paths.attempt_name(n)"),
                reason=_COMPOSE_OVERRIDES.get((rel, where, text))))
    return rows


def stale_compose_overrides(rows: "list[dict]") -> "list[tuple]":
    """Compose-side exemptions that match no site -- same failure as above."""
    live = {(r["file"], r["func"], r["text"]) for r in rows}
    return sorted(k for k in _COMPOSE_OVERRIDES if k not in live)



# --------------------------------------------------------------------------- #
#  THE THIRD AXIS: a DIRECTORY of the hierarchy, spelled at a caller.         #
# --------------------------------------------------------------------------- #
#
# The two passes above look at FILENAMES.  A directory built as
# `container / "launch"` or `base / "pseudos"` is invisible to both -- and
# `project-layout.md` § 2.6 is the authority on the tree, so a segment of it
# spelled at a call site is the same fault one level up.
#
# WHY THE VOCABULARY IS SCOPED TO A CALCULATION, measured 2026-09-08.  A broad
# net -- every `X / "plain-name"` -- finds 67 distinct segments across 128
# sites at 11% precision: `Ha/`, `eV/`, `n/a` (units), `application/json` (a
# MIME type), `bin`, `conda-meta`, `opt`, `envs` (conda's own trees), `docs/`,
# `refs/` (git's).  That is the nag list § 5l warns about, so the pass is
# scoped to the levels this standard owns.
#
# LEVELS ① AND ② ARE `projects.py`'s AND ARE ALREADY CLOSED: `CANONICAL_TOPICS`
# is declared and `validate_topic` REFUSES anything outside it, so a topic
# cannot be invented.  Including topics here found only noise --
# `f"transport/v{sv}"` is a schema string and `client.get("user/orgs")` is a
# GitHub endpoint, both flagged because `transport` and `user` are also topic
# names.
#
# WHAT IS GUARDED IS THE SET OF UNDECLARED SEGMENTS, NOT THE SITE COUNT.  The
# ten sites below go to zero in § 5l.6 step N5; what must not happen before
# then is a THIRD invented segment joining them silently.

def _calc_containers() -> "dict[str, str | None]":
    """The containers a CALCULATION holds, and the door that names each.

    ``None`` means **undeclared** -- the code uses the segment and nothing
    declares it, which is the finding this pass exists to hold still.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from molbuilder.checkpoint import ARCHIVE_DIR
    return {
        # declared, and reached through its door everywhere (checked 2026-09-08:
        # `_cli.py`'s hardcoded `base/"bench"` was closed 2026-08-13)
        "bench": "jobset.materialize.bench_container",
        # declared as a constant, so the name has one home
        ARCHIVE_DIR: "checkpoint.ARCHIVE_DIR",
        # UNDECLARED -- § 5l.6 N5
        "launch": None,
        "pseudos": None,
    }


CALC_CONTAINERS = _calc_containers()

#: (file, function, segment) -> reason.  Same anchoring rule as the two tables
#: above: never a line number.
_SEGMENT_OVERRIDES: "dict[tuple[str, str, str], str]" = {
    ("molbuilder/jobset/prep.py", "_pseudo_dir", "pseudos"):
        "THE DOOR ITSELF, and the finding is that it is PRIVATE -- three sites "
        "in its own module spell the segment again rather than call it.  N5 "
        "makes it a coordinate; until then this row keeps the door from being "
        "reported as one of its own violations",
}


def segments(pkg: "pathlib.Path | None" = None,
             root: "pathlib.Path | None" = None) -> "list[dict]":
    """Every place a calculation-level directory segment is CONSTRUCTED.

    Construction, not mention: a `Path` division, a string whose FIRST path
    component is the segment, or an `os.path.join` argument.  The
    first-component rule is what keeps Flask routes out -- `/api/bench/summary`
    starts with `/`, so its first component is empty (the throwaway version of
    this pass flagged three of those).
    """
    pkg = PKG if pkg is None else pkg
    root = ROOT if root is None else root
    names = set(CALC_CONTAINERS)
    rows: "list[dict]" = []
    seen: "set[tuple]" = set()
    for path in sorted(pkg.rglob("*.py")):
        rel = str(path.relative_to(root))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                              # pragma: no cover
            continue
        for node in ast.walk(tree):
            seg = how = None
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) \
               and isinstance(node.right, ast.Constant) \
               and node.right.value in names:
                seg, how = node.right.value, "Path /"
            else:
                txt = None
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    txt = node.value
                elif isinstance(node, ast.JoinedStr) and node.values and \
                        isinstance(node.values[0], ast.Constant):
                    txt = node.values[0].value
                if txt and "/" in txt and txt.split("/")[0] in names:
                    seg, how = txt.split("/")[0], "path string"
                elif isinstance(node, ast.Call) \
                        and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == "join":
                    for a in node.args:
                        if isinstance(a, ast.Constant) and a.value in names:
                            seg, how = a.value, "os.path.join"
                            break
            if seg is None:
                continue
            where = _enclosing(tree, node)
            key = (rel, node.lineno, seg)
            if key in seen:            # both rules can see one line
                continue
            seen.add(key)
            rows.append(dict(
                file=rel, line=node.lineno, func=where, segment=seg, how=how,
                door=CALC_CONTAINERS[seg],
                declared=CALC_CONTAINERS[seg] is not None,
                reason=_SEGMENT_OVERRIDES.get((rel, where, seg))))
    return rows


def undeclared_segments(rows: "list[dict]") -> "list[str]":
    """The segments the code builds and nothing declares -- the guarded number."""
    return sorted({r["segment"] for r in rows if not r["declared"]})


def stale_segment_overrides(rows: "list[dict]") -> "list[tuple]":
    live = {(r["file"], r["func"], r["segment"]) for r in rows}
    return sorted(k for k in _SEGMENT_OVERRIDES if k not in live)


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


def survey(pkg: "pathlib.Path | None" = None,
           root: "pathlib.Path | None" = None) -> list[dict]:
    """Every path search under *pkg*, classified.

    ``pkg``/``root`` are parameters so the GUARD CAN BE SHOWN TO FAIL.  A test
    that only asserts the real package is clean proves nothing about the check:
    it passes identically when the classifier has stopped classifying.  Pointed
    at a throwaway package holding one hand-spelled `glob("*.out")`, this must
    report `owned` -- and `tests/test_path_framework.py` asserts exactly that.
    """
    pkg = PKG if pkg is None else pkg
    root = ROOT if root is None else root
    rows: list[dict] = []
    for path in sorted(pkg.rglob("*.py")):
        rel = str(path.relative_to(root))
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


def stale_overrides(rows: list[dict]) -> list[tuple]:
    """Override keys that match no site any more.

    THE LESSON `classify_source_reads.py` PAID FOR (2026-09-08): its reasons
    were keyed by line number, an edit displaced two and orphaned a third, and
    the tool then reported a count for assertions nobody had reclassified.
    Keying by (file, function, pattern) survives an edit above the site -- but
    not a RENAME or a deletion, and a silently-dead exemption is how a rule
    stops applying without anyone deciding that.  So the dead keys are reported
    rather than ignored.
    """
    live = {(r["file"], r["func"], r["pattern"]) for r in rows}
    return sorted(k for k in _OVERRIDES if k not in live)


def _check(rows: list[dict]) -> int:
    bad = [r for r in rows
           if r["verdict"].startswith(FAILING_VERDICTS)]
    stale = stale_overrides(rows)
    built = [r for r in compositions() if r["reason"] is None]
    stale += stale_compose_overrides(compositions())
    seg_rows = segments()
    stale += stale_segment_overrides(seg_rows)
    undeclared = undeclared_segments(seg_rows)
    for r in built:
        print(f'HAND-BUILT   {r["file"]}:{r["line"]}  {r["func"]}()')
        print(f'    {r["text"][:110]}')
        print(f'    -> ask {r["door"]}')
    for r in bad:
        print(f'HANDCRAFTED  {r["file"]}:{r["line"]}  {r["func"]}()')
        print(f'    {r["call"]}({r["pattern"]!r})   [{r["verdict"]}]')
        if r["composer"]:
            print(f'    -> ask {r["composer"]}')
    for k in stale:
        print(f"STALE OVERRIDE  {k}  -- matches no site; the site was renamed, "
              f"moved or fixed.  Delete the entry or re-anchor it.")
    # THE SET, NOT THE SITE COUNT.  The ten sites go to zero in § 5l.6 N5;
    # what must not happen before then is a THIRD invented segment.
    if sorted(undeclared) != sorted(GUARDED_UNDECLARED):
        print(f"UNDECLARED HIERARCHY SEGMENTS changed: {undeclared!r}\n"
              f"    guarded set is {sorted(GUARDED_UNDECLARED)!r}\n"
              f"    `project-layout.md` § 2.6 is the authority on the tree; a "
              f"segment of it that nothing declares is a level invented at a "
              f"call site.  Declare it (§ 5l.6 N5) or do not build it.")
        return 1
    if bad or stale or built:
        print(f"\n{len(bad)} handcrafted search(es), {len(built)} hand-built "
              f"name(s), {len(stale)} stale override(s).  "
              f"`project-layout.md` § 4.5: for every name it composes, the "
              f"framework owns the search -- and composes it in one place.")
        return 1
    print(f"{len(rows)} path searches, none handcrafted; "
          f"{len(compositions())} counter-keyed names, all composed by a door; "
          f"{len(seg_rows)} hierarchy-segment sites over "
          f"{len(undeclared)} undeclared segment(s) {undeclared} "
          f"(§ 5l.6 N5 closes them); "
          f"{len(_OVERRIDES) + len(_COMPOSE_OVERRIDES) + len(_SEGMENT_OVERRIDES)} "
          f"recorded exemptions, all live.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", metavar="PREFIX",
                    help="every site whose verdict starts with PREFIX")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any site is owned/unclassified, or if an "
                         "override no longer matches a site")
    args = ap.parse_args()
    rows = survey()
    if args.check:
        return _check(rows)
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
    built = compositions()
    print(f"counter-keyed names BUILT by hand (the compose half): "
          f"{len([r for r in built if r['reason'] is None])} "
          f"of {len(built)} sites")
    for r in built:
        if r["reason"] is None:
            print(f'  {r["file"]}:{r["line"]}  {r["func"]}()  '
                  f'-> ask {r["door"]}')
    seg_rows = segments()
    print(f"hierarchy-segment sites (§ 2.6's tree, spelled at a call): "
          f"{len(seg_rows)}")
    for u in undeclared_segments(seg_rows):
        here = [r for r in seg_rows if r["segment"] == u]
        print(f"  {u!r} is UNDECLARED -- {len(here)} site(s): "
              + ", ".join(f'{r["file"].split("/")[-1]}:{r["line"]}'
                          for r in here))
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

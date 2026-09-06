#!/usr/bin/env python
"""Classify every test assertion that reads a file and checks its text.

WHY THIS IS A TOOL AND NOT A NUMBER IN A DOCUMENT.  This population has been
counted three times with three answers -- 233, 256, 173 -- because each count
used a different definition and none of them wrote it down.  The same thing
happened to the px/rem literals (777 -> 384 -> 160: three scopes, not three
measurements).  A count is a hypothesis; a count you cannot re-derive is not
even that.  Run this instead of quoting a figure.

    python tools/classify_source_reads.py            # the summary
    python tools/classify_source_reads.py --list 1   # every site in a bucket
    python tools/classify_source_reads.py --json     # machine-readable

THE DEFINITION.  An assertion is IN the population when any value it tests
traces -- inline, or through a local variable -- to a ``.read_text()`` /
``.read()`` call.  It is then split by WHAT WAS READ:

  REPO      the path resolves inside the package source tree: a file a person
            wrote and maintains.  These are the interesting ones.
  ARTIFACT  anything else -- a deck, a wrapper, an ``.sbatch``, a log, a JSON
            round-trip, a tmp_path.  Asserting that GENERATED output contains
            a keyword is a real property of a real product.  Never a defect.

and REPO reads are split again by what the assertion is FOR, which is
``docs/process/testing.md`` section 6's rule, not a new one:

  LINT   quantifies over a class -- "no file under lib/ contains setInterval",
         "no token is defined outside tokens.css", "these two sources agree".
         **Text is the correct instrument.** No runtime test can prove absence
         without exercising every path; reading the file settles it in a
         millisecond.  KEEP THESE.
  PIN    names one line and measures its spelling.  Behaviour-blind in both
         directions: it fires on a clean rename, and it passes while the thing
         it claims to check is broken.  These are the work.

A PIN is then routed by what would have to be true for the check to be honest:

  1 BROWSER  the answer depends on the CSS cascade, layout or real visibility.
             jsdom implements neither, so only a browser can answer.  Smallest
             bucket on purpose -- a browser test costs seconds and is where
             flakiness lives.  ``tests/`` already drives Playwright in 29 files.
  2 NODE     the answer needs the code to RUN but not to be painted: which
             requests fire, what DOM is written, what state survives a second
             click.  Milliseconds against a stub DOM.  35 files already do this
             through ``tests/_node_esm.py``.
  5 PYTHON   a pin on ``.py`` source.  Almost always cheaper to call the
             function; a few are one-home lints wearing a pin's clothes and are
             listed as KEEP overrides below.

OVERRIDES.  The syntactic rules above are a first pass.  Every site was then
READ, and the ones the rules got wrong are named in ``_OVERRIDES`` with the
reason -- because "the regex said so" is exactly the reasoning this tool
exists to replace.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# A read whose resolved path mentions one of these is a file a person wrote.
_REPO = re.compile(r"""['"](?:molbuilder|tests)/|'molbuilder'|"molbuilder"|'static'|'templates'""")
# ... unless it is plainly a temporary or generated tree.
_TMP = re.compile(r"tmp_path|tmpdir|\btmp_|mkdtemp")

# An assertion shaped like a LINT: it quantifies, or it asserts absence.
_LINT = re.compile(
    r"\bnot in\b|not re\.search|^not |\bnot \w+$|offenders|stray|climbers|spellers"
    r"|==\s*_THE_|<=\s*defined|_OWNED|inventory|required_text"
)
# Assertions on the RESULT of running the module (run_node / Playwright), which
# the tracer catches only because the harness read the file to load it.  These
# are bucket 2 already done right.
_RAN = re.compile(r"\bout\b|\bres\b|\bresult\b|\brun_node\b|\bpage\.|\bevaluate\b")
_SOURCEY = re.compile(r"\bsrc\b|\bcss\b|\bjs\b|\bhtml\b|\bhead\b|\bpage\b|\bshared\b")

KEEP_LINT = "3 KEEP - lint (text is the right instrument)"
KEEP_VEND = "4 KEEP - vendored or data file, not our source"
B_BROWSER = "1 convert - browser (cascade / layout / visibility)"
B_NODE    = "2 convert - node (behaviour, stub DOM)"
B_PYTHON  = "5 convert - python source (call the function)"

# file -> {line: (bucket, why)} -- corrections made by READING the site.
# A file listed in _OVERRIDE_FILES is corrected wholesale: use that form only
# when EVERY assertion in the file is the same kind, and say why.
_OVERRIDE_FILES: dict[str, tuple[str, str]] = {
    "tests/test_constants_module_consistency.py": (
        KEEP_LINT,
        "the file exists to check a JS constant and its Python twin still hold "
        "the same VALUE -- every assertion extracts the value and compares the "
        "two sources.  One-home enforcement, not spelling"),
}

_OVERRIDES: dict[str, dict[int, tuple[str, str]]] = {
    "tests/test_task_setup_tab.py": {
        ln: (B_BROWSER,
             "BLOCKED, not merely pending: `_targetArg()` returns \"\" unless a "
             "NAMED machine is chosen, and the page can only be driven to "
             "\"(this machine)\" without a named record in the live server's "
             "config root -- so an e2e check for --target passes whatever the "
             "code does.  MEASURED 2026-09-06: adding `_targetArg()` to the "
             "launch line left the e2e green.  Needs a fixture that supplies a "
             "named target; until then the pin beats a vacuous assertion")
        for ln in (1286, 1287)
    },
    "tests/test_structure_info_bridge.py": {
        ln: (B_BROWSER,
             "the aliasing runs inside `mountInspector` via "
             "`inspectorLifecycle.alias`, and the resets and the APPLY branch "
             "sit in `transition()` -- a reducer that exists only once a "
             "viewer is MOUNTED.  Nothing mounts one headless, so this is "
             "Playwright work, not node work.  Routed by file extension "
             "(.js -> node) until it was read, 2026-09-06")
        for ln in (369, 371, 374, 376)
    },
    "tests/test_contact_distance_reference.py": {
        ln: (KEEP_VEND,
             "reads the contact-distance REFERENCE TABLE (a data file) and "
             "checks every entry carries a citation -- a property of the data")
        for ln in (81, 84)
    },
    "tests/test_codemirror_vendor_bundle.py": {
        ln: (KEEP_VEND,
             "checks the VENDORED CodeMirror bundle declares its own "
             "dependency -- a property of a third-party artifact we ship")
        for ln in (120, 165, 169, 179)
    },
    "tests/test_one_home_for_a_constant.py": {
        ln: (KEEP_LINT,
             "the Bohr radius has ONE home; asserting the digits appear there "
             "and nowhere else is a one-home lint")
        for ln in (104, 111)
    },
    "tests/test_vendor_licenses.py": {
        ln: (KEEP_VEND, "a vendored licence must literally contain its text")
        for ln in (20, 35)
    },
}


def _resolve(expr: str, consts: dict[str, str], depth: int = 0) -> str:
    """Textually expand module-level constants inside a path expression."""
    if depth > 4:
        return expr
    out = expr
    for name, val in consts.items():
        if re.search(rf"\b{re.escape(name)}\b", out):
            out = re.sub(rf"\b{re.escape(name)}\b", f"({val})", out)
    return _resolve(out, consts, depth + 1) if out != expr else out


def _is_read(call: ast.AST) -> bool:
    f = getattr(call, "func", None)
    return isinstance(f, ast.Attribute) and f.attr in ("read_text", "read")


def collect() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(TESTS.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue

        consts: dict[str, str] = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                try:
                    consts[node.targets[0].id] = ast.unparse(node.value)
                except Exception:                      # pragma: no cover
                    pass

        reads: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = next((c for c in ast.walk(node.value)
                         if isinstance(c, ast.Call) and _is_read(c)), None)
            if call is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    reads.setdefault(target.id, ast.unparse(call.func.value))
        if not reads:
            continue

        rel = str(path.relative_to(ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            sources = []
            for n in ast.walk(node.test):
                if isinstance(n, ast.Call) and _is_read(n):
                    sources.append(ast.unparse(n.func.value))
                elif isinstance(n, ast.Name) and n.id in reads:
                    sources.append(reads[n.id])
            if not sources:
                continue

            full = _resolve(sources[0], consts)
            expr = ast.unparse(node.test)
            target = ("ARTIFACT" if _TMP.search(full) or not _REPO.search(full)
                      else "REPO")
            ext = next((e for e in (".css", ".js", ".html", ".py") if e in full), "?")

            if target == "ARTIFACT":
                bucket, why = "0 generated artifact - correct as text", ""
            elif _RAN.search(expr) and not _SOURCEY.search(expr):
                bucket, why = "0 already runs the code", "asserts on run output"
            elif _LINT.search(expr):
                bucket, why = KEEP_LINT, ""
            elif ext == ".css":
                bucket, why = B_BROWSER, ""
            elif ext == ".py":
                bucket, why = B_PYTHON, ""
            else:
                bucket, why = B_NODE, ""

            over = (_OVERRIDES.get(rel, {}).get(node.lineno)
                    or (_OVERRIDE_FILES.get(rel) if target == "REPO" else None))
            if over:
                bucket, why = over

            rows.append(dict(file=rel, line=node.lineno, target=target, ext=ext,
                             bucket=bucket, why=why, expr=expr[:160], read=full[:120]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", metavar="N", help="print every site whose bucket starts with N")
    ap.add_argument("--json", action="store_true", help="dump every row as JSON")
    args = ap.parse_args()

    rows = collect()
    if args.json:
        json.dump(rows, sys.stdout, indent=1)
        return 0

    if args.list:
        for r in rows:
            if r["bucket"].startswith(args.list):
                print(f"{r['file']}:{r['line']}\n    {r['expr']}")
                if r["why"]:
                    print(f"    -> {r['why']}")
        return 0

    repo = [r for r in rows if not r["bucket"].startswith("0")]
    print(f"assertions over a file's text : {len(rows):5d}  in "
          f"{len({r['file'] for r in rows})} files")
    print(f"  reading generated output    : {sum(1 for r in rows if r['bucket'].startswith('0')):5d}"
          "  (correct as text -- a property of a real product)")
    print(f"  reading hand-written source : {len(repo):5d}  in "
          f"{len({r['file'] for r in repo})} files\n")

    counts = collections.Counter(r["bucket"] for r in repo)
    for bucket in sorted(counts):
        print(f"  {bucket:52s} {counts[bucket]:4d}")
    convert = sum(v for k, v in counts.items() if "convert" in k)
    print(f"\n  {'TO CONVERT':52s} {convert:4d}")
    print(f"  {'TO KEEP':52s} {len(repo) - convert:4d}")

    for bucket in sorted(b for b in counts if "convert" in b):
        files = collections.Counter(r["file"] for r in repo if r["bucket"] == bucket)
        print(f"\n{bucket}  ({sum(files.values())} in {len(files)} files)")
        for f, n in files.most_common():
            print(f"   {n:4d}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

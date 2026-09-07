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
    # ---- ONE-HOME OWNERSHIP, which the ".css -> browser" rule gets wrong.
    # "WHICH SHEET DEFINES THIS" IS NOT A QUESTION A BROWSER CAN ANSWER: the
    # cascade yields a computed value, never the file it came from.  Each of
    # these names the one module that owns a vocabulary and asserts the tab
    # sheets do not redefine it -- absence over a set of files, which is
    # exactly what testing.md section 6 calls a LINT.
    "tests/test_results_blueprint.py": {
        ln: (KEEP_LINT,
             "form-components.css is the ONE home for the finding-row "
             "vocabulary (ui-contract.md 5.1), and the sibling line "
             "quantifies over EVERY severity the contract defines, so none "
             "renders bare")
        for ln in (951, 957)
    },
    "tests/test_molview_info_js.py": {
        62: (KEEP_LINT,
             "the info pane's four classes must be styled in the MODULE "
             "sheet, molview.css -- a page sheet that grew its own copy is "
             "the drift this catches, and only reading the files can see it"),
    },
    "tests/test_form_schema_diff_js.py": {
        279: (KEEP_LINT,
              "paired with `.rec-diff-list` NOT in the tab sheet: the widget "
              "is generic, so the forms module owns it and no tab may "
              "redefine it (ui-contract.md 1)"),
    },
    "tests/test_modify_css_residue.py": {
        33: (KEEP_LINT,
             "paired with the raw family list being absent: --font-mono has "
             "one home and the sheet must reach for it, not re-declare it"),
    },
    "tests/test_one_naming_authority.py": {
        73: (KEEP_LINT,
             "the rule is the ABSENCE line beside it -- prep must not compose "
             "the trial path by ANY spelling -- and this presence line stops "
             "that going vacuous on a prep that stopped asking the naming "
             "authority at all.  It was an exact three-deep argument list "
             "until 2026-09-06; what it stood for is measured on disk by "
             "test_prep_writes_where_job_dir_names_will_look"),
    },
    "tests/test_validation_delivery_contract.py": {
        208: (KEEP_LINT,
              "the rule is the two ABSENCE lines beside it -- no `xyz: null,` "
              "state mirror, no `factsForRequest` second assembler (comments "
              "stripped first, so prose about the retired door does not fire "
              "the guard).  This presence line stops both going vacuous on a "
              "tab that stopped reading the structure at all.  AND THE "
              "BEHAVIOUR IS NOT REACHABLE, verified by reading rather than "
              "assumed: Build mounts MolView `mode: \"readonly\"` -- \"this "
              "tab READS the structure ... it does not edit geometry\" -- so "
              "no control on the page can move an atom without a reload, and "
              "a reload refreshes a mirror too.  A browser test for staleness "
              "would pass whatever the code did"),
    },
    "tests/test_inspector_lifecycle_teardown.py": {
        ln: (KEEP_LINT,
             "counts `.addEventListener(` across a whole core and pins the "
             "shared scope to EXACTLY ONE registration site -- absence over "
             "a file, which no runtime test can prove without walking every "
             "path.  The behaviour half is driven: "
             "test_inspector_registry_e2e.py's two teardown tests")
        for ln in (172, 175)
    },
    "tests/test_projects_api_envelope_js.py": {
        701: (KEEP_LINT,
              "the ABSENCE half beside it -- no `fetch(\"/api/files` anywhere "
              "in preview.js -- is the rule (web/projects.md Principle 6: "
              "api.js is the sole caller).  This presence line is what stops "
              "that going vacuous on a preview.js that does no file IO at "
              "all.  The behaviour is driven by "
              "test_preview_modal_edit_save_e2e.py"),
    },
    "tests/test_validation_findings_js.py": {
        291: (KEEP_LINT,
              "quantifies over a DECLARED consumer list and is dominated by "
              "absence -- no consumer builds `className = \"issue-item\"` or "
              "maps `data-severity` itself.  validation.md 4.1 R2 says one "
              "channel into the UI, and a fourth renderer appearing beside "
              "the third is exactly what no behaviour test can catch.  This "
              "presence line keeps the two absence checks from passing "
              "trivially on a file that stopped rendering findings"),
    },
    "tests/test_task_setup_tab.py": {
        1214: (KEEP_LINT,
               "every class the machine card uses must ALREADY exist in the "
               "module sheet, and the paired line forbids any `.ts-target-*` "
               "-- i.e. no bespoke class invented for one card.  A browser "
               "sees the paint, not which sheet paid for it"),
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

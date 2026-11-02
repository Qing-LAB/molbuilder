# Test-suite audit brief — rogue tests, handcrafted tests, and whether each test earns its place

You are one of ten reviewers auditing the molbuilder test suite at /home/qqing/molbuilder (395 test files).
The user asked for this audit after, on 2026-10-30, a test helper (`tests/support/road.py::run_road_case`), run
by hand outside pytest, wrote a fabricated machine record into their REAL `~/.config/molbuilder/environment.json`
(setup line `true`, no machine, no probe date), and real runs on their machine failed for three days. They want
every test that can reach outside its sandbox found, every test built on handcrafted records found, and for each
test a justification of why it exists — or a verdict that it should not.

## Rules for you
- READ ONLY. Never edit a file in the repository. Never run pytest or any test, never run molbuilder commands,
  never import tests/support helpers in a script (that is exactly how the damage was done). Read files and grep.
  Use /usr/bin/grep.
- Read FIRST, whole: docs/process/testing.md (the suite's own contract: tiers, the pyramid, § 2a the projects
  tree, § 3b a test earns its place, § 3a end product never source, § 5 browser tests, § 6 patterns), and
  tests/conftest.py (the isolation: `config_root_is_never_the_developers` is FUNCTION-scoped autouse — it deletes
  MOLBUILDER_CONFIG_DIR and points XDG_CONFIG_HOME at a temp dir; `outside_the_developers_folders` is what a
  module-scoped fixture must call itself; `the_suite_leaves_your_config_alone` only fingerprints the real config
  before/after a whole session and reports, it prevents nothing; `isolated_projects_root` is opt-in).
- Then read EVERY file of your group WHOLE (and every tests/data/*.toml or fixture it reads). Do not sample.
- Verify every claim against the file text; give line numbers.

## What to find, per file

1. **ROGUE — can reach outside its sandbox** (the worst class). Any of:
   - reads or writes the real config dir (`~/.config/molbuilder`), the real state dir (`~/.local/state/molbuilder`),
     the repository's `projects/` tree, the user's home, `/tmp` paths outside `tmp_path`, `~/Work`, Sol paths,
     the dev server (port 8888), the real scheduler (`sbatch`, `squeue`), the real conda env manager (installs),
     the network (real URLs, notify channels);
   - a MODULE- or SESSION-scoped fixture (or a class-scoped one) that calls product code — prep, describe, launch,
     probe, the server, config readers/writers — without `outside_the_developers_folders` / an isolated root: the
     function-scoped autouse isolation does NOT apply to it;
   - `subprocess` / `Popen` / a server started with an `env=` built from scratch or filtered so that
     `XDG_CONFIG_HOME` / `XDG_STATE_HOME` / `HOME` could resolve to the real ones in the child;
   - a test or fixture that sets HOME / XDG_* itself to something not under a temp dir;
   - a helper (tests/support) that writes a config record, a machine record, a projects tree or a state file
     through `config_dir()`/`projects_root()`/`calculation_record` WITHOUT itself refusing a real root — so a call
     outside pytest (or from a module fixture) writes into the developer's real folders;
   - absolute paths to the developer's machine (`/home/qqing`, `~`), or `Path.home()` / `expanduser`.
2. **HANDCRAFTED — its input is not made by the road.** A test that writes by hand a file molbuilder itself
   writes or an engine writes — `environment.json`, `run.json`, `job-set.json`, `task.json`, `job-request.json`,
   `prepared-with.json`, `<stem>.concluded`, `<stem>.runwrap.log`, `.monitor.log`, a run's index, a deck, a
   sidecar, a SIESTA `.out` / PySCF `.pyscf.log` typed as a string — or fakes a server reply, an engine output, a
   scheduler answer, or plants state, instead of making it through the road (describe → prep → launch, the real
   engine) or reading a RECORDED real output (a fixture copied from a real run is a fair input to a reader; text
   typed to imitate one is not). Say what was handcrafted, where, and what the road would make instead.
3. **TEXT-PINNING** — asserts on source code text, doc text, or one file's text against another's (testing.md § 3a).
4. **OTHER DESIGN DEFECTS** — asserts only that something exists (API presence); cannot fail (no mutation of the
   code it names would turn it red); duplicates another test; tests a private name rather than a door; pins the
   code's current output with no contract source; states no goal/contract/silent failure in its first lines
   (testing.md § 3b); an e2e that does not need the real engine or browser.

## The verdict per file
`keep` (it earns its place as written) · `fix isolation` · `rewrite through the road` · `retire` (no silent
failure it guards, or a duplicate, or pins text) — with **the justification in one sentence: the silent failure
it guards (a real failure a person would meet), or "none found"**, and how its inputs are made today.

## Report
Write a markdown file to /home/qqing/.claude/jobs/142ef3ba/tmp/testaudit/<GROUP>.md (your group id is in your
task) with:
1. A summary: counts by verdict; every ROGUE finding listed first, each with file:line and the real folder it
   can reach.
2. One table row per file: file | kind (API / contract table / e2e real engine / e2e browser / field / helper)
   | contract it pins (doc §, or "none stated") | how inputs are made (road / recorded real / handcrafted / mixed)
   | rogue (file:line, what it reaches) | handcrafted (file:line, what) | other defects | verdict | justification.
3. Then a short paragraph: patterns you saw in how these tests were designed (how the handcrafting crept in).
Reply with the summary only (counts, the rogue findings, and the worst handcrafted ones in one line each).

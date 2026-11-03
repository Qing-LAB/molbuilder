# Review brief — the e2e revisions and 2.65.4's code, before the one e2e run

The e2e tests were reviewed against the contract (reports: /home/qqing/.claude/jobs/142ef3ba/tmp/e2erev/<G>.md),
revised by fixers (their reports: fix-<G>.md, brief fix_brief.md), and 2.65.4 (option B) was built: a describe of an
extension writes only its three statement files; its first prep takes the finished calculation's junction (or the
frame set, checked) and its pseudopotentials; one door answers "the junction this calculation has or will have"
(`transport/rung.junction_record`, `reuse.finished_junction`, `reuse.carry_pseudos`, `gather.composed_for_prep`,
`gather.ReadOnce.frames`). Also fixed in product code: the chemistry card's method word (`electronic_state._method`,
`chosen`; `web/blueprints/build.py`), the GPU notice (`lib/system-load-monitor.js`). Docs were corrected too
(`git diff -- docs/`). NOTHING has run on an engine yet: the e2e batch runs once, after this review.

## Rules (hard)
- READ ONLY. Never edit, create or delete a repository file. Write only your report.
- Never run pytest, `tools/testrun.py`, `molbuilder`, an engine, a browser or a server; never import `tests/` in a
  script. You may read files, `git diff`, `git show HEAD:<file>`, grep (/usr/bin/grep).
- Every finding: file:line, the evidence (code path read, contract sentence quoted with §), and why it matters.
  "Unverified" when you could not check it. No speculation.

## What to judge, for every changed or added test line in your scope (`git diff HEAD -- <files>`, read each file WHOLE)
1. **Will it pass on correct code?** Read the code path that produces what it asserts (molbuilder/…) and the fixture
   that makes the state. A new assertion that cannot pass on correct code (wrong words, wrong field, wrong order,
   a state the fixture never makes, a helper that does not exist, a wrong import) is a finding.
2. **Does its expectation come from the contract?** Quote the § sentence. An expectation copied from code text with
   no contract source, or contradicting the contract, is a finding.
3. **Can it fail on the defect it names?** A vacuous assertion (always true), or one that a plausible mutation of the
   named behaviour survives, is a finding.
4. **Did the revision handcraft input or plant state** (beyond what was already there and listed for F9)? Any new
   typed record, planted file, faked reply or monkeypatch of molbuilder internals is a finding.
5. **Retirements**: was anything retired that still guarded a real failure the contract states and no other test
   pins? Name it.
6. **Product code in scope** (if named in your task): read the touched modules WHOLE; judge correctness against the
   contract, every caller of a changed function (grep them), API unification (one door per fact — duplicated,
   redundant or handcrafted logic), and whether docs/docstrings now say what the code does.

## Report
Write /home/qqing/.claude/jobs/142ef3ba/tmp/e2erev/rev-<GROUP>.md: findings ranked (will-fail first, then wrong
expectation, vacuous, handcrafted, product defects, docs), each with file:line, evidence, and the fix in one line.
Then a one-paragraph verdict. Reply with the ranked finding list only (one line each).

#  molbuilder developer Makefile.
#
#  Targets:
#    make test         every test except ``slow``, e2e included, in one
#                      process (the runner is tools/testrun.py)
#    make test-all     full pytest sweep INCLUDING slow-marked tests
#    make precommit    install + run pre-commit hooks once on all files
#    make clean        drop pyc / __pycache__ / pytest cache

# THE HOST ENV'S OWN PYTHON, by its path (docs/ops/installation.md M6):
# never a bare `python`, which activation does not put first on PATH.  No
# env active: said, nothing run.
PY        ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,$(error no env is active -- run `conda activate molbuilder` first; molbuilder runs on the host env's own Python))
PYTEST    ?= $(PY) -m pytest

.PHONY: test test-all precommit clean help

help:
	@grep -E '^\.PHONY|^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | head -20

# Every test except `slow` and the atom-list (browser-only) file, e2e
# included.  What a commit must pass is the tests it touched, through
# tools/testrun.py (docs/process/testing.md § 7).
test:
	$(PYTEST) -q --no-header \
		--ignore=tests/test_atom_list_render_paths.py \
		-m "not slow"

# Full sweep INCLUDING slow-marked tests.  Use before pushing a
# long-pending branch or to debug a slow-only failure.
test-all:
	$(PYTEST) -q

# One-shot pre-commit run over the whole tree (useful before pushing
# a long-pending branch -- normal commits trigger only modified files).
precommit:
	@$(PY) -c "import pre_commit" 2>/dev/null || \
		(echo "pre-commit is not in the host env: molbuilder declares none, and the hook is optional (docs/process/conventions.md)"; exit 1)
	$(PY) -m pre_commit install
	$(PY) -m pre_commit run --all-files

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache

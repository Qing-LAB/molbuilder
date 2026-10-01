#  molbuilder developer Makefile.
#
#  Targets:
#    make test         every test except ``slow``, e2e included, in one
#                      process (the runner is tools/testrun.py)
#    make test-all     full pytest sweep INCLUDING slow-marked tests
#    make web          launch the dev server on :8080 (foreground)
#    make web-bg       launch the dev server in the background
#    make stop-web     kill any background dev server
#    make precommit    install + run pre-commit hooks once on all files
#    make clean        drop pyc / __pycache__ / pytest cache

PY        ?= python
PYTEST    ?= $(PY) -m pytest
HOST      ?= 127.0.0.1
PORT      ?= 8080

.PHONY: test test-all web web-bg stop-web precommit clean help

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

# Launch the Flask dev server in the foreground.  Ctrl-C to stop.
web:
	$(PY) -m molbuilder web --host $(HOST) --port $(PORT)

# Background variant -- writes PID file, redirects logs.
web-bg:
	@if [ -f /tmp/molbuilder-web.pid ] && kill -0 $$(cat /tmp/molbuilder-web.pid) 2>/dev/null; then \
		echo "dev server already running (PID $$(cat /tmp/molbuilder-web.pid))"; \
		exit 1; \
	fi
	@nohup $(PY) -m molbuilder web --host $(HOST) --port $(PORT) >/tmp/molbuilder-web.log 2>&1 & \
		echo $$! > /tmp/molbuilder-web.pid && \
		sleep 1 && \
		echo "dev server up at http://$(HOST):$(PORT)  (PID $$(cat /tmp/molbuilder-web.pid); log /tmp/molbuilder-web.log)"

stop-web:
	@if [ -f /tmp/molbuilder-web.pid ]; then \
		kill $$(cat /tmp/molbuilder-web.pid) 2>/dev/null || true; \
		rm -f /tmp/molbuilder-web.pid; \
		echo "dev server stopped"; \
	else \
		echo "no PID file at /tmp/molbuilder-web.pid"; \
	fi

# One-shot pre-commit run over the whole tree (useful before pushing
# a long-pending branch -- normal commits trigger only modified files).
precommit:
	@command -v pre-commit >/dev/null 2>&1 || \
		(echo "pre-commit not installed; run 'pip install pre-commit'"; exit 1)
	pre-commit install
	pre-commit run --all-files

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache

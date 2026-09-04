"""The This-machine tab's page script — `this-machine.md`.

The tab holds the only inputs in molbuilder a secret is ever typed into, and
the only screen that hands over a recipe for a machine this server cannot
reach.  Both are places where a page that *looks* right and emits the wrong
bytes fails in silence: an absent or malformed `notify` file simply means no
notifier, which is indistinguishable from never having set one up.

These drive the REAL functions out of `this-machine/page.js` under Node, for
the reason `test_task_setup_cell_readers_js.py` gives at length: the
controller cannot be imported without a DOM, so a test that does not run the
source can only check that names exist — and a stub returning the wrong thing
passes that.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "molbuilder/web/static/this-machine/page.js"


def _slice(src: str, start: str, end: str) -> str:
    i = src.index(start)
    return src[i:src.index(end, i)].rstrip()


# --------------------------------------------------------------------- #
#  the recipe is built from the form, not from its own placeholders      #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  what the page must not do                                            #
# --------------------------------------------------------------------- #


def test_the_channel_list_never_reads_a_secret_off_a_row():
    """The rule is enforced at the route (`notify_setup.py::_row`, which does
    not put one there), and this is the other half: the code that paints a
    row has nothing that would use a key or an unmasked address if a future
    route started sending one.

    Scoped to the painter rather than the whole file, because
    `clusterRecipe` legitimately handles a key -- one the person is typing,
    on its way OUT.  A whole-file substring search cannot tell those apart
    and flagged `spec.key` on the first run.
    """
    src = PAGE.read_text(encoding="utf-8")
    painter = src[src.index("function rowFor("):src.index("function paintChannels")]
    for leak in (".key", ".url", "unmasked"):
        assert leak not in painter, f"the row painter reads {leak}"
    assert "c.where" in painter, "the row must show the MASKED address"


# --------------------------------------------------------------------- #
#  It always writes here, and only TELLS you about anywhere else         #
# --------------------------------------------------------------------- #

def test_nothing_gates_the_save_on_execution_mode():
    """**The rule this page got wrong for five days.**

    `execution.mode` is `direct` (run in place) or through the scheduler, and
    it gates `.sbatch` submission **on this machine** (`running-a-job.md`
    § 5.4). The page read `submit` as *"the jobs run somewhere I cannot
    reach"* and refused to save — which refused a login node with SLURM, the
    machine the file most belongs on, and did not detect the real
    cross-machine case at all (a laptop preparing for a cluster is `direct`).

    The rule *(user, 2026-09-01)*: **every config file molbuilder manages is
    saved on the machine molbuilder runs on.**
    """
    src = PAGE.read_text(encoding="utf-8")
    for gone in ("canWrite", "can_write_here", "execution_mode"):
        assert gone not in src, f"the page still branches on {gone}"
    api = (ROOT / "molbuilder/web/blueprints/notify_setup.py").read_text()
    for gone in ("can_write_here", "execution_mode", "read_config"):
        assert gone not in api, f"the route still reports {gone}"


def test_it_generates_no_shell_and_holds_no_secret_for_another_machine():
    """Carrying a secret to the machine that builds or runs the task is the
    user's job **by design**, and the generated run script embeds no
    cleartext secret because that would violate the security protocol.

    So the page does not build a file for a machine it cannot see. It states
    what the script will look for and where — the one thing it can honestly
    say — and that statement contains no address, no key and no shell.
    """
    src = PAGE.read_text(encoding="utf-8")
    for gone in ("clusterRecipe", "renderRecipe", "heredoc", "EOF", "chmod"):
        assert gone not in src, f"the page still emits {gone}"


# RETIRED 2026-09-03 — test_the_stated_config_dir_rule_is_the_WHOLE_rule.
# It read `this_machine.html` off disk and checked three substrings were in
# it, while its docstring said "the behaviour itself is pinned by
# `test_config_dir_has_one_home.py`".  No such file has ever existed, and no
# browser test visited this page at all -- so the citation closed a question
# that was never open (`process/testing.md` § 3a.1).
#
# Replaced by tests/test_this_machine_e2e.py, which OPENS the <details> the
# rule lives in (a rule sealed inside a disclosure that never opens is stated
# to nobody -- the half a file-read could not see) and, more to the point,
# adds the check this page most needed and never had: a webhook typed into
# the form must not come back onto the screen in full.  Mutation-verified
# against `_mask()`.

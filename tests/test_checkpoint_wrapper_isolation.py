"""P4 invariant: ``runwrap`` never imports or invokes the checkpoint
module.  The wrapper is git-agnostic.  This is load-bearing for SLURM
compatibility — compute nodes execute the generated ``.run.sh`` with
no git access required.

See docs/protocols/run-checkpoints.md § 2 P4.
"""
from __future__ import annotations

import importlib
import inspect


def test_runwrap_does_not_import_checkpoint():
    """Static import-graph check: runwrap.py source must not reference
    the checkpoint module by name.  Catches any future refactor that
    accidentally couples them."""
    import molbuilder.runwrap as runwrap
    src = inspect.getsource(runwrap)
    forbidden = (
        "from molbuilder.checkpoint",
        "import molbuilder.checkpoint",
        "from .checkpoint",
        "import .checkpoint",
    )
    for needle in forbidden:
        assert needle not in src, (
            f"P4 violation: runwrap.py contains {needle!r}.  The "
            f"wrapper must be git-agnostic per docs/protocols/"
            f"run-checkpoints.md § 2 P4.  SLURM compute nodes must "
            f"be able to run the generated .run.sh without git."
        )


def test_checkpoint_module_not_in_runwrap_namespace():
    """Runtime check: after import, the runwrap module's namespace
    has no ``Repo`` / ``checkpoint`` / ``CheckpointError`` exports
    leaking from cross-import."""
    import molbuilder.runwrap as runwrap
    forbidden_attrs = ("Repo", "checkpoint", "CheckpointError",
                       "Checkpoint")
    for name in forbidden_attrs:
        # Allowed: a method or attribute that happens to share a name
        # but is locally defined.  The strict check is that the
        # checkpoint module itself isn't reachable via runwrap.
        v = getattr(runwrap, name, None)
        if v is None:
            continue
        # If reachable, make sure it's NOT the one from checkpoint.
        if getattr(v, "__module__", None) == "molbuilder.checkpoint":
            raise AssertionError(
                f"P4 violation: runwrap.{name} resolves to "
                f"molbuilder.checkpoint.{name}.")


def test_checkpoint_module_importable_standalone():
    """The checkpoint module loads without importing runwrap (loose
    coupling in both directions; runwrap is heavy)."""
    # Importing checkpoint should not also pull in runwrap as a
    # side-effect.  We can't test "did runwrap import?" reliably
    # after a prior test imported it; instead, check checkpoint's
    # module-level source for any runwrap reference.
    import molbuilder.checkpoint as cp
    src = inspect.getsource(cp)
    assert "from molbuilder.runwrap" not in src
    assert "import molbuilder.runwrap" not in src

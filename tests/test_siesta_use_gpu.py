"""The SIESTA GPU toggle's fields: ``SiestaConfig.use_gpu`` and
``diag_algorithm``, with their form metadata.

What a GPU deck says, and what its wrapper and header carry -- the env it
activates, the binding, the placement -- is the GPU contract's table, run
down the road (`tests/data/gpu_contract.toml`).
"""
from __future__ import annotations


from molbuilder.config.siesta import SiestaConfig


# --------------------------------------------------------------------- #
#  L1: SiestaConfig field shape                                          #
# --------------------------------------------------------------------- #


def test_use_gpu_default_is_off():
    """Safety default: a brand-new config does NOT request GPU.  This
    is load-bearing because the same .fdf is portable across CPU/GPU
    envs; a True default would silently route every SIESTA job into
    the GPU env regardless of whether the user has one installed."""
    cfg = SiestaConfig()
    assert cfg.use_gpu is False


def test_use_gpu_metadata_is_present():
    """The metadata a live reader takes off this field.

    The SIESTA form is built from the CATALOGUE; what reads this class is
    finding-placement (`workflow_group`) and the catalogue-agreement mirror
    (`engine_key`, `choices`, ...).
    """
    field = SiestaConfig.__dataclass_fields__["use_gpu"]
    md = field.metadata
    # ``staging`` -- the group whose items a parameter form deliberately does
    # NOT ask, because the Job Prep UI answers them (web/task-setup.md § 6.1,
    # § 6.2).  The class's ``workflow_group`` must MIRROR the catalogue item's
    # ``group``, which is ``staging`` (web/form-schema.md § 2): the form reads
    # the catalogue's, finding-placement reads this one, so a disagreement puts
    # the control on one card and its warnings on another.
    assert md["workflow_group"] == "staging"
    # The engine_key must reference the SIESTA fdf keyword so the
    # methods-text generator can cite it; an empty value would let
    # the field silently drift from the keyword the generator emits.
    assert "Diag.ELPA.GPU" in md["engine_key"]


def test_diag_algorithm_field_metadata():
    """A dropdown with three choices (ScaLAPACK + the two ELPA variants),
    default ScaLAPACK (safe on the precompiled CPU env), always shown (NOT
    gated behind GPU)."""
    field = SiestaConfig.__dataclass_fields__["diag_algorithm"]
    md = field.metadata
    assert md["workflow_group"] == "budget"
    assert md["choices"] == ("ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE")
    assert md["engine_key"] == "Diag.Algorithm"
    assert SiestaConfig().diag_algorithm == "ScaLAPACK"

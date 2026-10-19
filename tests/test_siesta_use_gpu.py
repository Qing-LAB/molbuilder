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
    """The facts a live reader takes about this item -- the catalogue's,
    through the one door (`template.items_of`; `engines/template.md`
    § 2.1a).  ``staging`` is the group whose items a parameter form
    deliberately does NOT ask, because the Job Prep UI answers them
    (web/task-setup.md § 6.1, § 6.2), and finding-placement files a finding
    about it on that card."""
    from molbuilder.template import items_of
    it = items_of(SiestaConfig)["use_gpu"]
    assert it.group == "staging"
    # The engine_key must reference the SIESTA fdf keyword so the
    # settings gate's message names what a person can search for.
    assert "Diag.ELPA.GPU" in it.engine_key


def test_diag_algorithm_field_metadata():
    """A dropdown with three choices (ScaLAPACK + the two ELPA variants),
    default ELPA-1STAGE (`engines/siesta.md` § 7: molbuilder's default, on
    CPU in the packaged env), always shown (NOT gated behind GPU)."""
    from molbuilder.template import items_of
    it = items_of(SiestaConfig)["diag_algorithm"]
    assert it.group == "budget"
    assert tuple(it.choices) == ("ScaLAPACK", "ELPA-1STAGE", "ELPA-2STAGE")
    assert it.engine_key == "Diag.Algorithm"
    assert SiestaConfig().diag_algorithm == "ELPA-1STAGE"

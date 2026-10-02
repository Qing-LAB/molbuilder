"""The SIESTA GPU toggle's two lower layers.

  * ``SiestaConfig.use_gpu``: dataclass field + form metadata.
  * ``render_fdf``: emits ``Diag.ELPA.GPU .true.`` iff the toggle is on.
  * ``runwrap._fdf_requests_gpu``: reads a deck's GPU keyword the way SIESTA
    does, for a run whose resources do not carry the answer.

What a GPU run's wrapper and header carry -- the env it activates, the
binding, the placement -- is the GPU contract's table, run down the road
(`tests/data/gpu_contract.toml`).  The wrapper-text tests that stood here
until 2026-10-02 retired into its rows, or with the rank and thread policy
they pinned (`architecture.md` § 5.2: every launch value is stated).
"""
from __future__ import annotations

from _deck import assert_fdf


import numpy as np
import pytest

from molbuilder.config.siesta import SiestaConfig
from molbuilder.siesta.input import render_fdf
from molbuilder.structure import Structure
from molbuilder import runwrap as _runwrap


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


def _mk_struct() -> Structure:
    """Minimal 1-atom water-stub: enough for render_fdf to succeed.  A per-side
    vacuum gives the derived cell a non-zero volume (a single atom's bbox is a
    point; vacuum=0 would be a degenerate box -- structure-periodicity.md)."""
    return Structure(
        elements      = ["H"],
        positions     = np.zeros((1, 3)),
        atom_names    = ["H1"],
        residue_ids   = [1],
        residue_names = ["UNL"],
        chain_ids     = ["A"],
        vacuum        = (12.0, 12.0, 12.0),
    )


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
    """The metadata a live reader still takes off this field.

    The docstring said *"the form schema is auto-built from dataclass
    metadata"* until 2026-08-17.  That direction was retired on 2026-08-15:
    the SIESTA form is built from the CATALOGUE, and what still reads this
    class is finding-placement (`workflow_group`) and the catalogue-agreement
    mirror (`engine_key`, `choices`, ...).
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


# --------------------------------------------------------------------- #
#  L1: render_fdf emission                                               #
# --------------------------------------------------------------------- #


def test_render_fdf_omits_gpu_keyword_by_default():
    """Off by default: the rendered .fdf must NOT contain the keyword
    when use_gpu=False, so a job rendered for a CPU-only host
    never accidentally routes to GPU on a different machine."""
    fdf = render_fdf(_mk_struct(), SiestaConfig())
    assert "Diag.ELPA.GPU" not in fdf


def test_render_fdf_emits_gpu_keyword_when_enabled():
    """``Diag.ELPA.GPU .true.`` is the modern (5.4.2) spelling per
    Src/diag_option.F90:139.  Asserting the literal value catches a
    typo like ``T``, ``Yes``, etc. -- they're ACCEPTED by fdf_get
    but the run-wrapper detector also has to accept them, and pinning
    one form keeps the contract simple."""
    cfg = SiestaConfig(use_gpu=True, diag_algorithm="ELPA-1STAGE")
    fdf = render_fdf(_mk_struct(), cfg)
    assert_fdf(fdf, "Diag.ELPA.GPU", ".true.")


def test_render_fdf_emits_diag_algorithm_with_gpu():
    """When GPU is on, the generator MUST also emit ``Diag.Algorithm
    ELPA-1STAGE`` -- the GPU keyword alone is silently ignored unless
    SIESTA is routing through ELPA.  Confirmed against SIESTA source
    at Src/diag_option.F90:213-225 (default ScaLAPACK path) and the
    user-visible failure: nvidia-smi at 0% utilisation while SCF is
    iterating happily on CPU."""
    cfg = SiestaConfig(use_gpu=True, diag_algorithm="ELPA-1STAGE")
    fdf = render_fdf(_mk_struct(), cfg)
    assert_fdf(fdf, "Diag.Algorithm", "ELPA-1STAGE")


def test_render_fdf_diag_algorithm_choice_propagates():
    """The ELPA variant is chosen via ``diag_algorithm`` (1STAGE/2STAGE).
    Pin that the choice flows through to the rendered keyword."""
    cfg = SiestaConfig(use_gpu=True, diag_algorithm="ELPA-2STAGE")
    fdf = render_fdf(_mk_struct(), cfg)
    assert_fdf(fdf, "Diag.Algorithm", "ELPA-2STAGE")
    assert "Diag.Algorithm     ELPA-1STAGE" not in fdf


def test_render_fdf_scalapack_default_omits_diag_keywords():
    """ScaLAPACK (the default) emits NEITHER Diag.Algorithm nor
    Diag.ELPA.GPU -- SIESTA falls through to its built-in Divide-and-
    Conquer path.  (Comment lines in the BENCH-MARKS block don't count.)"""
    fdf = render_fdf(_mk_struct(), SiestaConfig())   # default = ScaLAPACK, CPU
    engine_lines = [
        ln for ln in fdf.splitlines()
        if ("Diag.Algorithm" in ln or "Diag.ELPA.GPU" in ln)
        and not ln.lstrip().startswith("#")
    ]
    assert engine_lines == [], (
        f"ScaLAPACK should emit no diag keywords, got: {engine_lines!r}"
    )


def test_render_fdf_cpu_elpa_emits_algorithm_and_gpu_false():
    """CPU-ELPA (engines/siesta.md § 7): selecting an ELPA algorithm
    WITHOUT GPU must emit ``Diag.Algorithm`` AND an EXPLICIT
    ``Diag.ELPA.GPU .false.`` -- the source ELPA build defaults to the
    GPU codepath, so an omitted flag crashes a CPU run (Sol job 57852378).
    This is the behavior the old 'ELPA only when GPU' model wrongly denied."""
    cfg = SiestaConfig(use_gpu=False, diag_algorithm="ELPA-2STAGE")
    fdf = render_fdf(_mk_struct(), cfg)
    assert_fdf(fdf, "Diag.Algorithm", "ELPA-2STAGE")
    assert_fdf(fdf, "Diag.ELPA.GPU", ".false.")
    assert "Diag.ELPA.GPU      .true." not in fdf


def test_render_fdf_gpu_with_scalapack_is_rejected():
    """GPU acceleration only applies to ELPA; GPU + ScaLAPACK is a
    contradiction and must raise rather than emit a nonsensical .fdf."""
    cfg = SiestaConfig(use_gpu=True, diag_algorithm="ScaLAPACK")
    with pytest.raises(ValueError, match="requires an ELPA diagonalizer"):
        render_fdf(_mk_struct(), cfg)


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


# --------------------------------------------------------------------- #
#  L1: _fdf_requests_gpu detector                                        #
# --------------------------------------------------------------------- #


def test_fdf_requests_gpu_unreadable_returns_false(tmp_path):
    """Missing file -> safe default (CPU env).  Routing must never
    raise from inside write_run_wrapper -- the wrapper is on a
    user-input boundary and an OSError here would propagate as a
    500."""
    missing = tmp_path / "absent.fdf"
    assert _runwrap._fdf_requests_gpu(missing) is False


# ---------------------------------------------------------------------------
#  The deck's GPU keyword, read the way SIESTA reads it
# ---------------------------------------------------------------------------
#
#  BOTH keyword spellings, SIESTA fdf_get's truthy set, and the FIRST
#  occurrence of each -- libfdf's `fdf_locate` walks from `file_in%first` and
#  stops at the first matching label, so a later line never overrides an
#  earlier one.  Either keyword being true means the run wants a GPU, so the
#  two are ORed.  (A second reader -- an awk pass inside the wrapper at
#  launch, for a deck edited after prep -- was compared against this one
#  until 2026-10-02; it went with the rank defaults it chose between.)

_DECKS = [
    # (name, deck body, expected)
    ("absent",                 "SystemLabel J\n",                              False),
    ("modern spelling",        "Diag.ELPA.GPU .true.\n",                       True),
    ("older spelling",         "Diag.ELPA.UseGPU .true.\n",                    True),
    ("truthy: true",           "Diag.ELPA.GPU true\n",                         True),
    ("truthy: yes",            "Diag.ELPA.GPU yes\n",                          True),
    ("truthy: t",              "Diag.ELPA.GPU T\n",                            True),
    ("truthy: y",              "Diag.ELPA.GPU y\n",                            True),
    ("truthy: 1",              "Diag.ELPA.GPU 1\n",                            True),
    ("falsy: .false.",         "Diag.ELPA.GPU .false.\n",                      False),
    ("falsy: no",              "Diag.ELPA.GPU no\n",                           False),
    ("falsy: 0",               "Diag.ELPA.GPU 0\n",                            False),
    ("first wins: on then off", "Diag.ELPA.GPU .true.\nDiag.ELPA.GPU .false.\n", True),
    ("first wins: off then on", "Diag.ELPA.GPU .false.\nDiag.ELPA.GPU .true.\n", False),
    ("either keyword true",
     "Diag.ELPA.UseGPU .true.\nDiag.ELPA.GPU .false.\n",                       True),
    ("case-insensitive label", "DIAG.elpa.GpU .TRUE.\n",                       True),
    ("leading whitespace",     "    Diag.ELPA.GPU .true.\n",                   True),
    ("longer token is not it", "Diag.ELPA.GPUX .true.\n",                      False),
    ("commented out",          "# Diag.ELPA.GPU .true.\n",                     False),
    ("no value",               "Diag.ELPA.GPU\n",                              False),
    ("trailing comment",       "Diag.ELPA.GPU .true.   # on purpose\n",        True),
]


@pytest.mark.parametrize("label,body,expected", _DECKS,
                         ids=[d[0] for d in _DECKS])
def test_the_deck_gpu_reader_reads_as_siesta_does(tmp_path, label, body,
                                                  expected):
    """MUTATION THIS MUST FAIL AGAINST: drop a truthy value from
    `_GPU_TRUTHY`, drop the older `Diag.ELPA.UseGPU` spelling, or take the
    LAST occurrence instead of the first."""
    deck = tmp_path / "job.fdf"
    deck.write_text(body, encoding="utf-8")
    assert _runwrap._fdf_requests_gpu(deck) is expected, label

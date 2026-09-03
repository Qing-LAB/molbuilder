"""Tests for the TranSIESTA engine (the TranSIESTA deck emitter).

The engine is one row of `engines/transport.md` § 6 — the NEGF `.fdf`
emitter behind the `TransportEngine` Protocol.  What is pinned here:

* Engine self-registers as ``"transiesta"`` on import of
  :mod:`molbuilder.transport` (§ 6, Registry: a backend that registers
  itself adds its engine choice back to the form).
* ``render_script`` emits a device ``.fdf`` carrying every
  TranSIESTA-required keyword for a zero-bias NEGF run -- the modern
  4.1+/5.x syntax of § 4, *Emitter behavior*: one ``%block TS.Elec.<name>``
  per lead, ``TS.ChemPots`` + per-name ``TS.ChemPot.<name>``, and
  ``SolutionMethod transiesta``.
* ``preflight`` catches missing / empty region labels, and warns on a
  multi-bias attempt: ``TS.Voltage`` is one value per run, and a bias
  scan is multiple runs walked by the chain (§ 4, last callout; § 8).
* ``parse_output`` raises ``NotImplementedError`` with a clear
  deferred-scope message (NOT a stack trace).  Reading a finished run is
  `transport/record.py`'s job today (``summarize run`` -> the shipped
  ``<label>.transport.json``, § 6); the engine gains its own reader with
  the Results-tab transmission inspector (§ 8, Follow-up).
* ``methods_fragment`` returns prose -- a placeholder, which is why no
  document states its shape; the full version lands with parse_output.

The electrode derivation lives in ``transport/wizard.py`` driven by
``transport/compose.py`` (§ 6); the atom-ordering rule the preflight
guards, and why it cannot fire inside the composite, is § 4.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.transport import (
    REGION_BRIDGE,
    REGION_LEFT_ELECTRODE,
    REGION_RIGHT_ELECTRODE,
    TransportConfig,
)
from molbuilder.structure import Structure
from molbuilder.transport import get_engine, registered_engines


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


@pytest.fixture
def labeled_device():
    """A toy Au-S-C-Au junction with the required region labels.

    Geometry is meaningless (atoms on a line); the engine only
    cares about composition + region labels for the .fdf shape.
    """
    return Structure(
        elements=["Au", "Au", "S", "C", "Au", "Au"],
        positions=np.array(
            [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0],
             [8, 0, 0], [10, 0, 0]],
            dtype=float,
        ),
        regions={
            REGION_LEFT_ELECTRODE:  [0, 1],
            REGION_BRIDGE:          [2, 3],
            REGION_RIGHT_ELECTRODE: [4, 5],
        },
    )


@pytest.fixture
def default_cfg():
    return TransportConfig(job_name="testjunc")


# --------------------------------------------------------------------- #
#  Registry                                                             #
# --------------------------------------------------------------------- #


def test_transiesta_registered_on_import():
    """Importing ``molbuilder.transport`` is sufficient to populate
    the engine registry — the package ``__init__`` does the import
    that triggers ``@register_engine``.  Pin so a future cleanup
    that drops the import line silently breaks the dispatch path.
    """
    assert "transiesta" in registered_engines()
    engine = get_engine("transiesta")
    assert engine.name == "transiesta"
    assert "TranSIESTA" in engine.label


# --------------------------------------------------------------------- #
#  render_script — .fdf shape                                           #
# --------------------------------------------------------------------- #


def test_render_script_emits_required_transiesta_keywords(
        labeled_device, default_cfg):
    """The device ``.fdf`` MUST carry every keyword TranSIESTA
    needs to run a zero-bias NEGF calculation.  This test pins the
    minimum set so a refactor that drops one surfaces immediately.
    """
    script = get_engine("transiesta").render_script(
        labeled_device, default_cfg)
    required = [
        "SystemLabel",
        "NumberOfAtoms",
        "NumberOfSpecies",
        "%block ChemicalSpeciesLabel",
        "%block AtomicCoordinatesAndAtomicSpecies",
        "%block LatticeVectors",
        "%block kgrid_Monkhorst_Pack",
        "SolutionMethod",
        # Modern SIESTA 4.1+ / 5.x TranSIESTA syntax (2026-06-18
        # modernization): per-electrode block carries used-atoms +
        # HS.files; per-chempot block carries mu.  Legacy flat
        # TS.HSFileLeft / TS.NumUsedAtomsLeft are no longer emitted.
        "%block TS.Elecs",
        "%block TS.Elec.",        # one per electrode region (L / R / ...)
        "%block TS.ChemPots",
        "%block TS.ChemPot.",     # one per chempot
        "TS.Voltage",
        # NOTE: TS.ComplexContour.Emin / TS.ComplexContour.NumCircle
        # were absorbed into the per-%block TS.ChemPot.<name> blocks
        # under modern SIESTA 4.1+ / 5.x; not asserted at top level
        # any more.  Empirically verified vs. SIESTA 5.4.2 (see
        # tests/test_transiesta_siesta_smoke_l4.py).
        "TS.TBT.NumE",
        "TS.TBT.Emin",
        "TS.TBT.Emax",
    ]
    for k in required:
        assert k in script, (
            f"missing required TranSIESTA keyword {k!r} in the "
            f".fdf — engine cannot produce a runnable script"
        )


def test_render_script_systemlabel_matches_job_name(
        labeled_device, default_cfg):
    """SystemLabel drives the basename for .TSHS / .TBT.* output
    files — must equal cfg.job_name exactly so the user can
    discover outputs by name."""
    script = get_engine("transiesta").render_script(
        labeled_device, default_cfg)
    assert f"SystemLabel            {default_cfg.job_name}" in script


def test_render_script_uses_region_atom_counts(labeled_device,
                                                default_cfg):
    """``used-atoms`` (inside each %block TS.Elec.<name> block)
    comes from struct.regions atom counts.  Pin so a future
    refactor that hard-codes them or reads from a wrong source
    surfaces.  Updated for the 2026-06-18 modern-syntax move:
    legacy TS.NumUsedAtoms{Left,Right} flat keys are no longer
    emitted."""
    script = get_engine("transiesta").render_script(
        labeled_device, default_cfg)
    # Our fixture has 2 + 2 electrode atoms; the modern emitter
    # writes ``used-atoms         2`` inside %block TS.Elec.L
    # AND %block TS.Elec.R (so exactly TWO occurrences).
    assert script.count("used-atoms         2") == 2


def test_render_script_emits_only_first_bias_voltage(labeled_device):
    """B.3 zero-bias scope: only bias_voltages_v[0] is emitted.
    Multi-bias requires multiple .fdfs (deferred scope).  Pin so
    a future change that silently emits a list (which TranSIESTA
    would reject at parse time) is caught."""
    cfg = TransportConfig(
        job_name="biased",
        bias_voltages_v=[0.5, 1.0, 1.5],
    )
    script = get_engine("transiesta").render_script(labeled_device, cfg)
    # Exactly one TS.Voltage line, at the first value.
    voltage_lines = [
        ln for ln in script.split("\n")
        if ln.lstrip().startswith("TS.Voltage")
    ]
    assert len(voltage_lines) == 1
    assert "0.5000 eV" in voltage_lines[0]
    # The other values must NOT appear as TS.Voltage entries.
    assert "1.0000 eV" not in script
    assert "1.5000 eV" not in script


def test_render_script_carries_chemical_species_block(labeled_device,
                                                       default_cfg):
    """The ChemicalSpeciesLabel block must list every unique
    element with its atomic number.  Pin Au=79 + S=16 + C=6 for
    the fixture junction."""
    script = get_engine("transiesta").render_script(
        labeled_device, default_cfg)
    # All three species, with their atomic numbers, in the block.
    block_start = script.index("%block ChemicalSpeciesLabel")
    block_end   = script.index("%endblock ChemicalSpeciesLabel")
    block = script[block_start:block_end]
    for sp, z in [("Au", 79), ("S", 16), ("C", 6)]:
        assert f"  {z}  {sp}" in block or f"{z}  {sp}" in block, (
            f"ChemicalSpeciesLabel missing {sp} (Z={z}); block:\n{block}"
        )


def test_render_script_documents_electrode_file_assumption(
        labeled_device, default_cfg):
    """The banner at the top of the .fdf MUST mention that
    electrode .TSHS files are assumed to exist.  Pin so a future
    refactor that drops the documentation leaves the user lost
    when their first run fails."""
    script = get_engine("transiesta").render_script(
        labeled_device, default_cfg)
    assert ".TSHS" in script.split("# ===")[1], (
        "header banner does not mention .TSHS — users will not "
        "know they need to generate electrode files separately"
    )


# --------------------------------------------------------------------- #
#  preflight                                                            #
# --------------------------------------------------------------------- #


def test_preflight_clean_structure_no_errors(labeled_device, default_cfg):
    """Happy path: labeled junction + default config → no errors."""
    issues = get_engine("transiesta").preflight(labeled_device, default_cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert errs == []


def test_preflight_errors_on_device_transport_kz_gt_1(labeled_device):
    """SCIENTIFIC-AUDIT FIX (transport FN1): the device transport axis
    (A3 = k_mesh_transverse[2]) MUST be 1 -- NEGF treats z as an open
    boundary, so kz>1 imposes a fake Bloch periodicity and gives wrong
    transmission with no runtime error.  The web Generate path gates on
    THIS engine preflight, so the invariant must live here (not only in
    the CLI-only cross-run `transport preflight`)."""
    from molbuilder.config.transport import TransportConfig
    bad = TransportConfig(k_mesh_transverse=(4, 4, 2))
    issues = get_engine("transiesta").preflight(labeled_device, bad)
    kz_errs = [i for i in issues
               if i.where == "config.k_mesh_transverse"
               and i.severity == "error"]
    assert len(kz_errs) == 1, f"kz=2 must error; got {issues}"
    # transverse-only sampling is fine:
    ok = TransportConfig(k_mesh_transverse=(4, 4, 1))
    ok_issues = get_engine("transiesta").preflight(labeled_device, ok)
    assert not [i for i in ok_issues
                if i.where == "config.k_mesh_transverse"
                and i.severity == "error"]


def test_preflight_warns_on_unknown_region_label(labeled_device, default_cfg):
    """No silent absorption (engines/overview.md § 3): a region label
    TranSIESTA does NOT consume (anything beyond L-electrode /
    R-electrode / bridge) is ignored for the calculation, so preflight
    must WARN rather than drop it silently -- and the unknown label
    must not block generation."""
    labeled_device.regions["scatterer-core"] = [3]
    issues = get_engine("transiesta").preflight(labeled_device, default_cfg)
    warns = [i for i in issues if i.severity == "warn"]
    assert any("scatterer-core" in w.message for w in warns), (
        "unknown region label must surface a warn-severity notice"
    )
    errs = [i for i in issues if i.severity == "error"]
    assert errs == [], "an unknown region label must not block generation"


def test_preflight_missing_regions_blocks_generation(default_cfg):
    """No region labels at all → error.  The user must label the
    structure on the Molbuilder tab first."""
    struct = Structure(
        elements=["C", "H"],
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
    )
    issues = get_engine("transiesta").preflight(struct, default_cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert errs, "missing-region structure must surface at least one error"
    assert any("region" in e.message.lower() for e in errs)


def test_preflight_empty_electrode_region_blocks_generation(default_cfg):
    """Region labels present but L-electrode is empty → error.
    The TS.NumUsedAtomsLeft would be 0, breaking the calculation."""
    struct = Structure(
        elements=["C", "H"],
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        regions={
            REGION_LEFT_ELECTRODE:  [],   # empty
            REGION_BRIDGE:          [0],
            REGION_RIGHT_ELECTRODE: [1],
        },
    )
    issues = get_engine("transiesta").preflight(struct, default_cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert any("L-electrode" in e.message for e in errs), (
        "empty left-electrode region must surface a clear error"
    )


def test_preflight_empty_bridge_blocks_generation(default_cfg):
    """Empty bridge = no device region = nothing to compute T(E)
    through.  Must error."""
    struct = Structure(
        elements=["C", "H"],
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        regions={
            REGION_LEFT_ELECTRODE:  [0],
            REGION_BRIDGE:          [],   # empty
            REGION_RIGHT_ELECTRODE: [1],
        },
    )
    issues = get_engine("transiesta").preflight(struct, default_cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert any("bridge" in e.message for e in errs)


def test_preflight_multi_bias_names_the_composite(labeled_device):
    """A single-deck render (the validation surface) emits only
    bias_voltages_v[0]; a SCAN is the composite's job.  Pin the WARN
    so a multi-point config surfaces the road instead of silently
    dropping bias points.
    """
    cfg = TransportConfig(
        job_name="biased",
        bias_voltages_v=[0.0, 0.5, 1.0],
    )
    issues = get_engine("transiesta").preflight(labeled_device, cfg)
    warns = [i for i in issues if i.severity == "warn"]
    assert any(
        "composite" in w.message.lower() and "one deck" in w.message.lower()
        for w in warns
    ), (
        "multi-bias call did not surface the deferred-scope WARN; "
        "user will not know the extra bias points were ignored"
    )


def test_preflight_blocks_out_of_order_atoms(default_cfg):
    """BLOCKER guard from the 2026-06-10 post-ship review:
    TranSIESTA reads TS.NumUsedAtomsLeft = N as "the FIRST N atoms
    in AtomicCoordinates are the left electrode."  If the user's
    XYZ has atoms in any order other than [L][bridge][R], the .fdf
    SILENTLY produces wrong physics — no run-time error.

    Pin so a future refactor that drops the check resurrects the
    silent-science risk.  Reference: Brandbyge 2002 § III.
    """
    # L-electrode at indices [0, 2], bridge at [1], R-electrode at
    # [3, 4] — L atoms are not contiguous (gap at index 1) and
    # bridge atom 1 sits IN THE MIDDLE of the L-electrode range.
    bad_order = Structure(
        elements=["Au", "C", "Au", "Au", "Au"],
        positions=np.array(
            [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0], [8, 0, 0]],
            dtype=float,
        ),
        regions={
            REGION_LEFT_ELECTRODE:  [0, 2],   # NON-CONTIGUOUS — bug seed
            REGION_BRIDGE:          [1],
            REGION_RIGHT_ELECTRODE: [3, 4],
        },
    )
    issues = get_engine("transiesta").preflight(bad_order, default_cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert errs, (
        "out-of-order atoms must surface a preflight error — "
        "silent acceptance would produce chemically wrong "
        "transmission curves"
    )
    assert any(
        "[L-electrode][bridge][R-electrode]" in e.message
        or "ordered" in e.message
        for e in errs
    )


def test_preflight_accepts_correctly_ordered_atoms(default_cfg):
    """Contiguous L→bridge→R ordering = OK.  Pin so the check
    doesn't over-fire and block legitimate structures."""
    good_order = Structure(
        elements=["Au", "Au", "S", "C", "Au", "Au"],
        positions=np.array(
            [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0],
             [8, 0, 0], [10, 0, 0]],
            dtype=float,
        ),
        regions={
            REGION_LEFT_ELECTRODE:  [0, 1],   # contiguous, first
            REGION_BRIDGE:          [2, 3],   # contiguous, middle
            REGION_RIGHT_ELECTRODE: [4, 5],   # contiguous, last
        },
    )
    issues = get_engine("transiesta").preflight(good_order, default_cfg)
    errs = [i for i in issues if i.severity == "error"]
    assert not errs, (
        f"correctly-ordered atoms triggered preflight errors: "
        f"{[e.message[:80] for e in errs]}"
    )


def test_preflight_warns_on_high_bias_outside_linear_response(
        labeled_device):
    """|V| > 2 V is outside the Landauer linear-response regime
    (di Ventra 2008).  Surface a WARN so the user interprets the
    result as a nonlinear-I-V snapshot, not linearized
    conductance.

    Pin so a future change that silently accepts arbitrary bias
    values without surfacing the regime caveat is caught.
    """
    cfg = TransportConfig(job_name="hi_v", bias_voltages_v=[3.5])
    issues = get_engine("transiesta").preflight(labeled_device, cfg)
    warns = [i for i in issues if i.severity == "warn"]
    assert any(
        "linear-response" in w.message.lower()
        or "nonlinear" in w.message.lower()
        for w in warns
    ), (
        "|V| > 2 V did not surface the linear-response-regime "
        "warning — user may misinterpret nonlinear-I-V snapshot "
        "as conductance"
    )


def test_preflight_does_not_fire_high_bias_warning_at_low_v(
        labeled_device):
    """V <= 2 V is in the linear-response regime — no warn.  Pin
    the boundary so the warn doesn't over-fire on legitimate
    low-bias setups."""
    cfg = TransportConfig(job_name="lo_v", bias_voltages_v=[1.5])
    issues = get_engine("transiesta").preflight(labeled_device, cfg)
    warns = [i for i in issues
             if i.severity == "warn" and "linear-response" in i.message.lower()]
    assert not warns, (
        "V = 1.5 V is within linear response; warn should not fire"
    )


def test_preflight_open_shell_metal_routes_through_shared_check(
        labeled_device):
    """Phase 1d single-source contract: the open-shell-metal check
    runs here too via the SHARED ``check_open_shell_metal`` from
    validation.py.  Pin so a future refactor that re-inlines the
    chemistry detection in this engine doesn't break the cross-
    engine consistency rule.

    Add Fe to the fixture (Au→Fe substitution); transport is a
    closed-shell run by default today, so an open-shell metal
    should fire the warn.
    """
    fe_struct = Structure(
        elements=["Fe", "Au", "S", "C", "Au", "Au"],
        positions=np.array(
            [[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0],
             [8, 0, 0], [10, 0, 0]],
            dtype=float,
        ),
        regions={
            REGION_LEFT_ELECTRODE:  [0, 1],
            REGION_BRIDGE:          [2, 3],
            REGION_RIGHT_ELECTRODE: [4, 5],
        },
    )
    issues = get_engine("transiesta").preflight(
        fe_struct, TransportConfig(job_name="fe_junc"))
    warns = [i for i in issues if i.severity == "warn"]
    assert any(
        "open-shell" in w.message.lower() and "Fe" in w.message
        for w in warns
    ), (
        "shared open-shell-metal check did not fire on a Fe-bearing "
        "device — Phase 1d single-source contract broken"
    )


# --------------------------------------------------------------------- #
#  Stub methods                                                         #
# --------------------------------------------------------------------- #


def test_parse_output_raises_and_points_at_the_record():
    """parse_output waits on the /results transmission inspector;
    raising NotImplementedError that NAMES the shipped record is
    BETTER than returning empty results or crashing on a missing key.
    """
    with pytest.raises(NotImplementedError) as ei:
        get_engine("transiesta").parse_output("nonexistent.json")
    msg = str(ei.value).lower()
    assert "transport.json" in msg and "record" in msg.replace("transport/record.py", "record"), (
        f"NotImplementedError message should name the shipped record; got: {ei.value}"
    )


def test_methods_fragment_returns_placeholder_prose(labeled_device,
                                                     default_cfg):
    """The Methods fragment is a placeholder today; full version
    lands with parse_output.  Pin the placeholder includes a
    citation marker AND mentions the deferred status so a reader
    isn't surprised."""
    from molbuilder.transport.results import TransportResults
    # Today's stub doesn't use ``results`` — pass an empty one.
    txt = get_engine("transiesta").methods_fragment(
        default_cfg, TransportResults())
    assert "TranSIESTA" in txt
    assert "[CITE:" in txt or "Brandbyge" in txt
    assert "deferred" in txt.lower()


# --------------------------------------------------------------------- #
#  The engine-frame translation (task #32, 2026-07-29)                   #
# --------------------------------------------------------------------- #


def _coords_from_lines(lines):
    import numpy as _np
    a = next(i for i, l in enumerate(lines)
             if "%block AtomicCoordinatesAndAtomicSpecies" in l)
    out = []
    for l in lines[a + 1:]:
        if "%endblock" in l:
            break
        out.append([float(x) for x in l.split()[:3]])
    return _np.array(out)


def _junction():
    import numpy as _np
    from molbuilder.structure import Structure
    s = Structure(elements=["Au", "S", "Au"],
                  positions=_np.array([[6.0, 6.0, 6.0],
                                       [7.0, 7.0, 7.0],
                                       [8.0, 8.0, 8.0]]))
    s.cell = _np.eye(3) * 10.0
    s.cell_origin = _np.array([5.0, 5.0, 5.0])
    s.__post_init__()
    return s


def test_emit_geometry_applies_the_engine_frame_shift():
    """A junction cell captured around off-origin atoms: the TranSIESTA
    deck must shift atoms by -cell_origin (cell anchored at zero) — it
    used to emit world-frame coordinates, a 5 A mistranslation here."""
    import numpy as _np
    from molbuilder.transport.transiesta import _emit_geometry
    coords = _coords_from_lines(_emit_geometry(_junction()))
    assert _np.allclose(coords[0], [1.0, 1.0, 1.0])
    assert _np.allclose(coords[2], [3.0, 3.0, 3.0])


def test_transiesta_frame_equals_render_fdf_frame():
    """The one frame convention (structure-periodicity.md § 6.1 clause 5):
    both SIESTA emitters produce the SAME coordinates for the same
    structure."""
    import numpy as _np
    from molbuilder.transport.transiesta import _emit_geometry
    from molbuilder.siesta.input import render_fdf
    from molbuilder.config.siesta import SiestaConfig
    s = _junction()
    ts = _coords_from_lines(_emit_geometry(s))
    fdf = _coords_from_lines(
        render_fdf(s, SiestaConfig(verbose_comments=False)).splitlines())
    assert _np.allclose(_np.sort(ts, axis=0), _np.sort(fdf, axis=0))


def test_imported_crystal_stays_verbatim_in_transiesta():
    """Explicit cell, no origin (imported crystal): no shift, matching
    render_fdf's no-shift convention."""
    import numpy as _np
    from molbuilder.structure import Structure
    from molbuilder.transport.transiesta import _emit_geometry
    s = Structure(elements=["Au"], positions=_np.array([[1.0, 2.0, 3.0]]))
    s.cell = _np.eye(3) * 10.0
    s.__post_init__()
    coords = _coords_from_lines(_emit_geometry(s))
    assert _np.allclose(coords[0], [1.0, 2.0, 3.0])


class TestTheReservoirBinding:
    """Which end carries mu = +V/2 follows the REGION LABEL, not the
    geometry (2026-08-29).  TranSIESTA takes chempot names as arbitrary
    strings (Papior 2017); what it enforces positionally is atom order
    and semi-inf-direction.  So the label is the only place the user
    said which lead is "left", and `V = V_left - V_right` must mean
    exactly that -- binding the name by z-order silently inverts the
    I-V sign of a junction whose author put L at the high-z end.
    """

    @staticmethod
    def _junction(l_on_top):
        import numpy as np
        from molbuilder.structure import Structure
        lo = [0.0, 2.5, 5.0]
        hi = [12.0, 14.5, 17.0]
        mid = [7.0, 8.0, 9.0]
        elements = ["Au"] * 3 + ["C"] * 3 + ["Au"] * 3
        zs = lo + mid + hi
        first, last = ((REGION_RIGHT_ELECTRODE, REGION_LEFT_ELECTRODE)
                       if l_on_top
                       else (REGION_LEFT_ELECTRODE,
                             REGION_RIGHT_ELECTRODE))
        regions = {first: [0, 1, 2], REGION_BRIDGE: [3, 4, 5],
                   last: [6, 7, 8]}
        return Structure(
            elements=elements,
            positions=np.array([[0.0, 0.0, z] for z in zs]),
            regions=regions, cell=np.diag([8.0, 8.0, 25.0]))

    def _deck(self, struct):
        from molbuilder.config.transport import TransportConfig
        from molbuilder.transport import get_engine
        return get_engine("transiesta").render_script(
            struct, TransportConfig(job_name="bind_test"))

    def _elec_block(self, text, name):
        head = f"%block TS.Elec.{name}"
        assert head in text, f"{head} missing"
        return text.split(head, 1)[1].split("%endblock", 1)[0]

    def test_l_at_the_low_end_binds_left(self):
        text = self._deck(self._junction(l_on_top=False))
        assert "chem-pot           Left" in self._elec_block(text, "L")
        assert "semi-inf-direction -A3" in self._elec_block(text, "L")

    def test_the_r_electrode_takes_the_minus_reservoir(self):
        text = self._deck(self._junction(l_on_top=False))
        assert "chem-pot           Right" in self._elec_block(text, "R")
        assert "semi-inf-direction +A3" in self._elec_block(text, "R")

    def test_the_deck_states_which_region_is_plus_v_over_2(self):
        """The one fact a reader cannot get from the block names."""
        text = self._deck(self._junction(l_on_top=False))
        assert "carries mu = +V/2" in text
        assert "V = V_left - V_right" in text

    def test_preflight_warns_about_inverted_labels_but_never_blocks(self):
        """CHECK z, WARN, the author decides (user ruling, 2026-08-29).
        The engine door also validates structures that never went
        through prep's sort, so it must SAY that the labels run against
        the usual convention -- and must not stop the run, because
        biasing the other end is a legitimate experiment."""
        from molbuilder.config.transport import TransportConfig
        from molbuilder.transport import get_engine
        issues = get_engine("transiesta").preflight(
            self._junction(l_on_top=True),
            TransportConfig(job_name="bind_test"))
        said = [i for i in issues if "HIGH-z" in i.message]
        assert said, (
            "an inverted junction passes the engine door silently -- "
            "its deck biases the high-z lead and nothing said so")
        assert said[0].severity == "warn", (
            "the convention is a warning, never a refusal")
        assert not [i for i in issues if i.severity == "error"], (
            "an inverted junction must still be runnable")

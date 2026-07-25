"""Pre-emission validation for SIESTA / PySCF / Spectra / Transport.

The check list lives in ``docs/design.md`` § "Validation pass
(pre-emission)" and the machinery in
``docs/protocols/scientific-validation.md``.  Generators
(siesta.input.render_fdf, pyscf.input.render_script, spectra
preflight, transport.transiesta.validate) call :func:`validate`
before writing output; errors block emission, warnings print to
stderr.

Design principles realised by this module:

  * **Principle #1** ("the dataclass is the lingua franca"):
    config-field validators (range / `validate=` callable) are read
    off ``dataclasses.field(metadata=...)`` -- no parallel lookup
    table, so adding a new field with a range is a one-line change
    to the dataclass.

  * **Principle #6** ("pre-emission geometry validation"): every
    structure-side check from the table is enforced here, before any
    SIESTA / PySCF text is emitted.

The output is a ``List[Issue]``.  Callers decide what to do with it;
:func:`report` is the "raise errors, print warnings to stderr"
helper that ``render_fdf`` / ``render_script`` use.

Package layout (split 2026-06-13; see
``docs/protocols/scientific-validation.md`` § 10):

* :mod:`molbuilder.validation.geometry`  — engine-agnostic geometry
* :mod:`molbuilder.validation.metadata`  — dataclass-field driven
* :mod:`molbuilder.validation.chemistry` — analyzer-backed shared rules
* :mod:`molbuilder.validation.sidecar`   — frozen-atoms / region INFO
* :mod:`molbuilder.validation.siesta`    — SIESTA-specific + aggregator
* :mod:`molbuilder.validation.pyscf`     — PySCF-specific + aggregator

Pre-split this all lived in one ``validation.py``.  The flat file
worked while only ``validate()``/``_validate_siesta``/``_validate_pyscf``
were callers; once Spectra + Transport preflights needed
``check_open_shell_metal`` the import path through a private
underscore-name in a 1326-LoC flat module became the smell that
preceded the 2026-06-13 Au-BDT-Au drift incident.  The split is
purely organisational — every function body, signature, and
``_validate_<engine>`` internal call sequence is preserved verbatim
from the pre-split source.  The behaviour-preservation invariant is
pinned by the existing test suite, which imports from this package
exactly as it imported from the flat module.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from ..issues import Issue, ValidationError
from ..structure import Structure

# Re-export the public + cross-module-imported names so the package
# is a drop-in replacement for the pre-split ``validation.py``.  The
# underscore-prefixed names are explicitly part of this surface
# because external modules (spectra/pyscf_engine, transport/transiesta,
# tests/test_validation) imported them by name pre-split; renaming
# them is the follow-up promotion proposed in
# scientific-validation.md § 10 and is out of scope for this commit.
from .chemistry import (_check_metal_basis_adequacy,
                        check_open_shell_metal,
                        _check_peptide_protonation)
from .geometry import (_check_polymer_orientation,
                       _min_image_distance,
                       validate_geometry)
from .metadata import _validate_config_metadata
from .pyscf import _validate_pyscf
from .sidecar import _check_frozen_atoms_consumed
from .siesta import (_check_siesta_charged_makov_payne_notice,
                     _check_siesta_mesh_cutoff,
                     _check_siesta_pseudo_coverage,
                     _check_siesta_spin_polarized_needs_spin_total,
                     _validate_siesta)


# --------------------------------------------------------------------- #
#  Engine-validator registry                                            #
#                                                                        #
#  Type-keyed dispatch.  Each engine config class registers an          #
#  engine-specific validator; `validate()` looks it up by              #
#  isinstance().  Adding a new engine is a `_ENGINE_VALIDATORS[T] = fn` #
#  line, not a string-compare in `validate()`.                          #
# --------------------------------------------------------------------- #


_ENGINE_VALIDATORS: Dict[Type, Callable[[Structure, object, Optional[np.ndarray]], List[Issue]]] = {}


def _register_engine_validator(cfg_cls: Type):
    """Decorator: register a validator for a specific config class.

    The validator receives (struct, cfg, cell) and returns a list of
    Issues.  ``cell`` may be None for engines that don't have a
    periodic cell concept (PySCF gas-phase / PCM); each registered
    validator decides what to do with it.
    """
    def deco(fn):
        _ENGINE_VALIDATORS[cfg_cls] = fn
        return fn
    return deco


# --------------------------------------------------------------------- #
#  Top-level entry point                                                #
# --------------------------------------------------------------------- #


def validate(struct: Structure, cfg, *,
             cell: Optional[np.ndarray] = None,
             dest_dir: "Optional[object]" = None,
             prior: "Optional[object]" = None) -> List[Issue]:
    """Run every applicable validation check and return the findings.

    Parameters
    ----------
    struct
        The Structure about to be emitted.
    cfg
        SiestaConfig or PySCFConfig (or any dataclass; the generic
        config-field metadata pass runs on anything dataclass-shaped).
    cell
        Optional pre-computed (3, 3) lattice the generator is going
        to use.  If None, cell-dependent checks are skipped.  The
        SIESTA generator computes the cell anyway, so it should pass
        the same matrix here.
    dest_dir
        Optional destination directory hint -- the path the user is
        about to save the rendered .fdf into.  Used by the SIESTA
        validator to resolve dest-relative ``cfg.psml_lib`` paths
        (the portable form the web Save handler persists).  When
        None, dest-relative paths fall back to projects/-anchored
        resolution; the file-existence check may then misfire and is
        downgraded to a WARN.
    prior
        Optional prior engine results (Spectra ``SpectraResults`` /
        Transport ``TransportResults``) that the Spectra/Transport
        engine validators use for selector / cross-run checks.  None
        on a first run and on every SIESTA/PySCF Build call (their
        validators ignore it).

    The returned list is in deterministic order: generic geometry
    checks first, generic config-field checks next, then engine-
    specific checks.  Callers can sort / filter as they please.

    This is THE single per-engine validation gate: every engine
    (SIESTA, PySCF, Spectra, Transport) registers ONE validator, so a
    caller runs ``validate(struct, cfg)`` once instead of hand-
    concatenating a separate engine ``preflight()`` (the cross-tab
    silent-skip class the backend-architecture review flagged; V1/V2).
    """
    issues: List[Issue] = []
    issues += validate_geometry(struct, cell)
    issues += _validate_config_metadata(cfg)

    # Engine-specific dispatch via the registry.  isinstance() picks
    # up subclasses too, so a future engine config that subclasses
    # an existing one inherits its validator unless it registers its
    # own.  Extra kwargs (dest_dir / prior) are forwarded only when
    # set; every registered validator accepts **_ and ignores the ones
    # it doesn't use.
    engine_kw = {}
    if dest_dir is not None:
        engine_kw["dest_dir"] = dest_dir
    if prior is not None:
        engine_kw["prior"] = prior
    for cfg_cls, fn in _ENGINE_VALIDATORS.items():
        if isinstance(cfg, cfg_cls):
            issues += fn(struct, cfg, cell, **engine_kw)
            break
    return issues


def report(issues: List[Issue], *,
           raise_on_error: bool = True,
           stream=None) -> None:
    """Print warnings to stderr; raise ValidationError on errors.

    The two-pass shape (warnings first, then maybe-raise) lets the
    user see *all* the warnings even when an error is also present --
    helpful when triaging a misconfigured run.
    """
    if stream is None:
        stream = sys.stderr
    for i in issues:
        if i.severity == "warn":
            tag = f" [{i.where}]" if i.where else ""
            print(f"warn{tag}: {i.message}", file=stream)
    errors = [i for i in issues if i.severity == "error"]
    if errors and raise_on_error:
        raise ValidationError(issues)


# --------------------------------------------------------------------- #
#  Engine-validator registration                                        #
#                                                                        #
#  Done at module bottom rather than via decorators on _validate_siesta #
#  / _validate_pyscf because the config classes import from this        #
#  module in some code paths (lift would create an import cycle).  A   #
#  late lookup in this module is fine; both engines' renderers import  #
#  validation.py before they call validate(), so by then the registry  #
#  is populated.                                                        #
# --------------------------------------------------------------------- #


# --- Spectra / Transport validators ------------------------------------ #
# These engines carry their scientific checks in the engine's own
# ``preflight(struct, cfg, prior=None)`` classmethod (selected by
# ``cfg.engine`` via the engine registry).  Registering thin wrappers here
# makes ``validate(struct, cfg, prior=prior)`` the SINGLE per-engine gate,
# so /spectra and /transport no longer hand-concatenate a second
# ``engine.preflight()`` pass (the silent-skip risk; V1/V2).  The engine
# import is LAZY (call-time) because the engine module imports
# ``check_open_shell_metal`` from this package — a call-time import avoids
# the register-time cycle.


def _validate_spectra(struct: Structure, cfg, cell, *, prior=None, **_) -> List[Issue]:
    # RENDER-gate science only (grid / amplitude / parity / method /
    # open-shell).  The selector-availability check is preflight-only UX
    # (a top_n script is valid to emit), so it is NOT in this gate -- the
    # /spectra preflight endpoint adds engine.selector_checks() on top.
    from ..spectra import get_engine
    return list(get_engine(cfg.engine).render_checks(struct, cfg))


def _validate_transport(struct: Structure, cfg, cell, *, prior=None, **_) -> List[Issue]:
    from ..transport import get_engine
    return list(get_engine(cfg.engine).preflight(struct, cfg, prior=prior))


def _register_default_engines() -> None:
    """Late binding to avoid an import cycle: the engine config classes
    live in modules that themselves import from validation.  Importing
    them here at module-import time would loop; importing inside a
    function called from validate() is safe because by then both modules
    are fully loaded.  Only the config CLASS is imported eagerly (as a
    registry key); the validator BODY for spectra/transport imports its
    engine lazily at call time (see the wrappers above)."""
    try:
        from ..siesta import SiestaConfig
        _ENGINE_VALIDATORS[SiestaConfig] = _validate_siesta
    except ImportError:
        pass
    try:
        from ..pyscf import PySCFConfig
        _ENGINE_VALIDATORS[PySCFConfig] = _validate_pyscf
    except ImportError:
        pass
    try:
        from ..config.spectra import SpectraConfig
        _ENGINE_VALIDATORS[SpectraConfig] = _validate_spectra
    except ImportError:
        pass
    try:
        from ..config.transport import TransportConfig
        _ENGINE_VALIDATORS[TransportConfig] = _validate_transport
    except ImportError:
        pass


_register_default_engines()


__all__ = ["validate", "report"]

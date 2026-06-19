"""Job decoder + monitor for running SIESTA / PySCF jobs.

See ``docs/protocols/job-decoder.md`` for the wire contract this
package implements.  Phase 1 ships ``decode_run_dir`` (the
directory-level decoder).  Phase 2+ adds the background monitor +
REST API + webhook delivery.
"""

from .decoder import (
    SCHEMA_VERSION,
    ENGINE_BODY_KEYS,
    JobTypeAmbiguousError,
    decode_run_dir,
)

__all__ = [
    "SCHEMA_VERSION",
    "ENGINE_BODY_KEYS",
    "JobTypeAmbiguousError",
    "decode_run_dir",
]

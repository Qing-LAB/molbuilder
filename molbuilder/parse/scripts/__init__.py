"""TextParsers for the 6 fdf reserved blocks defined in
``docs/execution/job-contracts.md``.

Unlike FileParsers + DirParsers, TextParsers are NOT registered
in the registry — they're picked explicitly by callers because
there's no path to sniff (text bodies arrive in memory).  Use
:func:`molbuilder.parse.parse_text` to dispatch:

    from molbuilder.parse import parse_text
    from molbuilder.parse.scripts import ScriptSourceTextParser
    result = parse_text(fdf_text, parser=ScriptSourceTextParser)
"""

from .atom_metadata import AtomMetadataTextParser
from .bench_marks import BenchMarksTextParser
from .header import HeaderTextParser
from .provenance import ProvenanceTextParser
from .source import ScriptSourceTextParser
from .user_custom import UserCustomTextParser


__all__ = [
    "AtomMetadataTextParser",
    "BenchMarksTextParser",
    "HeaderTextParser",
    "ProvenanceTextParser",
    "ScriptSourceTextParser",
    "UserCustomTextParser",
]

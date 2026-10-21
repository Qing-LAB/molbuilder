"""A folder's ZIP archive, streamed as its files are read, its length known
before the first byte (`docs/web/projects.md` § 4, *Download*).

Every member is STORED, not compressed -- a calculation's large files are
binary and barely compress, and compressing costs the server minutes per
gigabyte -- and every member is ZIP64, whatever its size: a local header
with a ZIP64 extra, the data as read, a ZIP64 data descriptor carrying its
CRC (computed on the way through, so each file is read once), then the
central directory, each entry with its ZIP64 extra (sizes and offset), and
the ZIP64 end records.  Every byte that is not a file's own is fixed by the
names and the sizes, so :func:`length` is the archive's exact size -- what
the response states as its ``Content-Length``: the browser shows progress
and how long, and a cut connection reads as a failed download, never a
short zip taken for whole.

Standard library alone; nothing here knows a folder, a route or a page.
"""
from __future__ import annotations

import struct
import zlib
from typing import Iterator, List, Sequence, Tuple

#: One member: the file read, its name in the archive, its size when it was
#: measured.
Entry = Tuple[str, str, int]

_CHUNK = 1 << 20
_VERSION = 45                                   # ZIP64 (APPNOTE 4.4.3)
_FLAGS = 0x0808                                 # UTF-8 names; a descriptor
_DOS_TIME, _DOS_DATE = 0, (1 << 5) | 1          # 1980-01-01 00:00
_LOCAL, _DESCRIPTOR, _CENTRAL = 30 + 20, 24, 46 + 28
_END = 56 + 20 + 22                             # ZIP64 end, locator, end


def length(entries: Sequence[Entry]) -> int:
    """The archive's exact size, in bytes, for ``entries``."""
    return _END + sum(_LOCAL + _DESCRIPTOR + _CENTRAL
                      + 2 * len(arc.encode("utf-8")) + size
                      for _src, arc, size in entries)


def stream(entries: Sequence[Entry]) -> Iterator[bytes]:
    """The archive's bytes, ``entries`` in order -- :func:`length` of them.

    A file shorter than its measured size raises ``OSError`` naming it: the
    response ends short of its stated length, and the download reads as
    failed.  A file longer than it was is sent as it was measured."""
    central: List[Tuple[bytes, int, int, int]] = []
    offset = 0
    for src, arc, size in entries:
        name = arc.encode("utf-8")
        yield (struct.pack("<IHHHHHIIIHH", 0x04034B50, _VERSION, _FLAGS, 0,
                           _DOS_TIME, _DOS_DATE, 0, 0xFFFFFFFF, 0xFFFFFFFF,
                           len(name), 20)
               + name + struct.pack("<HHQQ", 0x0001, 16, size, size))
        crc, sent = 0, 0
        with open(src, "rb") as fh:
            while sent < size:
                chunk = fh.read(min(_CHUNK, size - sent))
                if not chunk:
                    raise OSError(f"{src} is shorter than the {size} bytes "
                                  f"it measured when the download was "
                                  f"asked for")
                crc = zlib.crc32(chunk, crc)
                sent += len(chunk)
                yield chunk
        yield struct.pack("<IIQQ", 0x08074B50, crc, size, size)
        central.append((name, crc, size, offset))
        offset += _LOCAL + len(name) + size + _DESCRIPTOR
    directory = b"".join(
        struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, _VERSION, _VERSION,
                    _FLAGS, 0, _DOS_TIME, _DOS_DATE, crc, 0xFFFFFFFF,
                    0xFFFFFFFF, len(name), 28, 0, 0, 0, 0, 0xFFFFFFFF)
        + name + struct.pack("<HHQQQ", 0x0001, 24, size, size, at)
        for name, crc, size, at in central)
    yield directory
    n = len(central)
    yield struct.pack("<IQHHIIQQQQ", 0x06064B50, 44, _VERSION, _VERSION, 0, 0,
                      n, n, len(directory), offset)
    yield struct.pack("<IIQI", 0x07064B50, 0, offset + len(directory), 1)
    yield struct.pack("<IHHHHIIH", 0x06054B50, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
                      0xFFFFFFFF, 0xFFFFFFFF, 0)

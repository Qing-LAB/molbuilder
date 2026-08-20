/* zip-store — several named byte blobs, one STORE-mode `.zip`.
 *
 * Contract: a deterministic, uncompressed archive — local headers, a central
 * directory naming them, and an end record pointing at that directory.  No
 * clock: the same input twice is the same bytes.
 * Owns:     nothing but the format.
 * Called by: VibrationView's animation export (png-zip) and the MolView files
 *           door (a Data download is a PAIR, and a second programmatic
 *           download is exactly what browsers silently swallow — one archive
 *           is the delivery that cannot lose its half).
 *
 * Extracted verbatim from lib/vibrationview/_export.js on 2026-08-19, where
 * it lived private since the animation export shipped — the second consumer
 * is what made it a shared module (one home, not one copy per exporter).
 */
"use strict";

/* ── A zip, with nothing compressed ──────────────────────────────────────────
 *
 * PNGs are already compressed, so storing them costs nothing and saves pulling in
 * a compressor. This is the whole of the format that a reader needs: a local
 * header before each file, a central directory listing them, and an end record
 * pointing at that directory.
 *
 * The timestamp is a constant rather than the clock, so exporting the same
 * animation twice produces the same bytes. Nothing here needs to know when it ran.
 */
const ZIP_TIME = 0;      // 00:00:00
const ZIP_DATE = 0x21;   // 1980-01-01, the epoch the format was born with

const CRC_TABLE = (function () {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
        t[n] = c >>> 0;
    }
    return t;
})();

function crc32(bytes) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) {
        c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
    }
    return (c ^ 0xFFFFFFFF) >>> 0;
}

export function storeZip(entries) {
    const enc   = new TextEncoder();
    const parts = [];
    const dir   = [];
    let offset  = 0;

    function u16(v) { return [v & 0xFF, (v >>> 8) & 0xFF]; }
    function u32(v) {
        return [v & 0xFF, (v >>> 8) & 0xFF, (v >>> 16) & 0xFF, (v >>> 24) & 0xFF];
    }

    for (const e of entries) {
        const name = enc.encode(e.name);
        const crc  = crc32(e.bytes);
        const head = new Uint8Array([].concat(
            u32(0x04034b50), u16(20), u16(0), u16(0),
            u16(ZIP_TIME), u16(ZIP_DATE),
            u32(crc), u32(e.bytes.length), u32(e.bytes.length),
            u16(name.length), u16(0)));
        parts.push(head, name, e.bytes);
        dir.push({ name: name, crc: crc, size: e.bytes.length, offset: offset });
        offset += head.length + name.length + e.bytes.length;
    }

    const cdStart = offset;
    for (const d of dir) {
        const head = new Uint8Array([].concat(
            u32(0x02014b50), u16(20), u16(20), u16(0), u16(0),
            u16(ZIP_TIME), u16(ZIP_DATE),
            u32(d.crc), u32(d.size), u32(d.size),
            u16(d.name.length), u16(0), u16(0), u16(0), u16(0), u32(0),
            u32(d.offset)));
        parts.push(head, d.name);
        offset += head.length + d.name.length;
    }
    parts.push(new Uint8Array([].concat(
        u32(0x06054b50), u16(0), u16(0), u16(dir.length), u16(dir.length),
        u32(offset - cdStart), u32(cdStart), u16(0))));

    return new Blob(parts, { type: "application/zip" });
}

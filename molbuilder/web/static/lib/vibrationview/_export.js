/* VibrationView — getting the animation out. Level 3 of docs/web/vibrationview.md § 7, § 12.
 *
 * Module:    lib/vibrationview/ — INTERNAL.
 * Called by: index.js, from inside the closure that holds the viewer's state.
 *            Everything this file needs arrives as an argument at call time, so
 *            no accessor for it exists on the handle (§ 4).
 *
 * It walks the phase itself and asks level 2 for each frame's positions, exactly
 * as the animation loop does — which is why "what is animated is what is
 * exported" (§ 5.3) is arithmetic here rather than a promise: there is one answer
 * to "where are the atoms on frame n", and the screen and the encoder both read
 * it.
 *
 * NEVER (§ 7 level 3): work out a position itself, or name the drawing library —
 * not in code, and not in a comment either; a guard asserts it, and it caught
 * this line's first draft. This file knows about GIFs, zips and video
 * containers, which is knowledge about formats and not about graphics.
 *
 * ── Carried, not invented ────────────────────────────────────────────────────
 * The two encoders came from the retired embed, where they worked. What is new is
 * the frame sequence they encode (this module's, not the drawing surface's), the
 * PNG-sequence zip, and the fact that all three read a picture that already has
 * the caption composited into it.
 */
"use strict";

const root = (typeof window !== "undefined") ? window : globalThis;

const FORMATS = ["gif", "webm", "png-zip"];

const GIF_VENDOR    = "/static/vendor/gif.min.js";
const GIF_WORKER    = "/static/vendor/gif.worker.min.js";
const WEBM_BITRATE  = 8_000_000;   // the browser default looks muddy on flat 3D


function aborted(signal) {
    return !!(signal && signal.aborted);
}

function report(opts, fraction, label) {
    if (typeof opts.onProgress !== "function") return;
    try { opts.onProgress(fraction, label); } catch (_) {}
}


/* ── The GIF encoder ─────────────────────────────────────────────────────────
 *
 * Loaded only when a GIF is asked for — it is a vendor file most sessions never
 * need. Cached on the global so two viewers exporting at once share one load;
 * the cache holds the WRAPPED promise, so a failed load is cleared for both
 * rather than leaving one caller holding a rejected promise nobody reset.
 */
function gifEncoder() {
    if (root.GIF) return Promise.resolve(root.GIF);
    if (root.__vibrationviewGifLoading) return root.__vibrationviewGifLoading;
    const loading = new Promise(function (resolve, reject) {
        try {
            const s = root.document.createElement("script");
            s.src = GIF_VENDOR;
            s.async = true;
            s.onload = function () {
                root.GIF ? resolve(root.GIF)
                         : reject(new Error("gif.min.js loaded but window.GIF is undefined"));
            };
            s.onerror = function () {
                reject(new Error("could not load " + GIF_VENDOR));
            };
            root.document.head.appendChild(s);
        } catch (e) { reject(e); }
    }).catch(function (e) {
        root.__vibrationviewGifLoading = null;
        throw e;
    });
    root.__vibrationviewGifLoading = loading;
    return loading;
}


/* ── A zip, with nothing compressed ──────────────────────────────────────────
 *
 * PNGs are already compressed, so storing them costs nothing and saves pulling in
 * a compressor. This is the whole of the format that a reader needs: a local
 * header before each file, a central directory listing them, and an end record
 * pointing at that directory.
 *
 * The timestamp is a constant rather than the clock, so exporting the same
 * animation twice produces the same bytes. Nothing here needs to know when it ran.
 *
 * PRIVATE ON PURPOSE: this package is sealed (its module-boundary tests
 * forbid imports in either direction), so it carries its own copy even
 * though lib/zip-store.js now holds the same code for the rest of the app.
 * Consolidating the two is this package's own future migration, not an
 * import through the wall.
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

function zip(entries) {
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

    return new root.Blob(parts, { type: "application/zip" });
}


/* WHAT AN EXPORT IS, written once (§ 12).
 *
 * The caller gets this back and the zip carries it inside; describing the same
 * export twice would be two descriptions to keep in step, and the one nobody
 * looks at is the one that goes stale.
 *
 * The amplitude travels BESIDE the normalization, always, because the two
 * pairings do not share a unit (§ 12.2) — either number alone is an invitation to
 * write the wrong thing in a caption.
 */
function describe(meta, format, count) {
    return {
        format:        format,
        frames:        count,
        cycles:        meta.cycles,
        fps:           meta.fps,
        cycleSeconds:  meta.cycleSec,
        framesPerCycle: meta.framesPerCycle,
        amplitude:     meta.amplitude,
        normalization: meta.norm,
        mode:          meta.index,
    };
}

/* The zip's copy, plus what only a folder of frames needs: its size, the note
 * that keeps the amplitude honest, and the line that turns the frames into
 * whatever a journal asked for. Six months later it still explains itself. */
function manifest(description, width, height) {
    return JSON.stringify(Object.assign({}, description, {
        width:  width,
        height: height,
        note: "amplitude is meaningful only together with normalization; "
            + "they are the two halves of one pairing (vibrationview.md 12.2)",
        ffmpeg: "ffmpeg -framerate " + description.fps + " -i frame_%04d.png"
              + " -c:v libx264 -pix_fmt yuv420p -crf 18 vibration.mp4",
    }), null, 2);
}


/* ── Walking the frames ──────────────────────────────────────────────────────
 *
 * The one loop every format shares: put frame n on the surface, hand the picture
 * to whoever is encoding. Sharing it is what stops three formats from becoming
 * three slightly different animations.
 */
async function eachFrame(ctx, opts, count, take) {
    for (let n = 0; n < count; n++) {
        if (aborted(opts.signal)) throw new Error("the export was cancelled");
        ctx.surface.setAtomCoords(ctx.coordsAt(n));
        const canvas = ctx.surface.compositeCanvas();
        if (!canvas) throw new Error("nothing drawn");
        await take(canvas, n);
        report(opts, (n + 1) / count, "frame " + (n + 1) + "/" + count);
    }
}


async function toPngZip(ctx, opts, count, description) {
    const files = [];
    let width = 0, height = 0;
    await eachFrame(ctx, opts, count, async function (canvas, n) {
        width = canvas.width; height = canvas.height;
        const blob = await new Promise(function (res, rej) {
            canvas.toBlob(function (b) { b ? res(b) : rej(new Error("a frame could not be encoded")); },
                          "image/png");
        });
        files.push({ name: "frame_" + String(n).padStart(4, "0") + ".png",
                     bytes: new Uint8Array(await blob.arrayBuffer()) });
    });
    files.push({ name: "manifest.json",
                 bytes: new TextEncoder().encode(manifest(description, width, height)) });
    return zip(files);
}


async function toGif(ctx, opts, count) {
    const GIF = await gifEncoder();
    const first = ctx.surface.compositeCanvas();
    if (!first) throw new Error("nothing drawn");
    const gif = new GIF({
        workers: 2, quality: 10,
        width: first.width, height: first.height,
        workerScript: GIF_WORKER,
    });
    // Milliseconds, which is what the encoder takes. It writes them into the file
    // in HUNDREDTHS of a second, so only rates dividing 100 come out exact
    // (§ 12.1) — at the default 30 fps the delay rounds and the GIF plays a few
    // percent off the cycle it was asked for. Quantising here as well would round
    // it twice.
    const delay = 1000 / ctx.meta.fps;
    await eachFrame(ctx, opts, count, function (canvas) {
        gif.addFrame(canvas, { copy: true, delay: delay });
    });
    return new Promise(function (resolve, reject) {
        gif.on("finished", resolve);
        try { gif.render(); } catch (e) { reject(e); }
    });
}


/* WebM is recorded rather than assembled: the encoder watches a live surface, so
 * the frames have to be put on screen at wall-clock pace instead of as fast as
 * they can be drawn. A three-second clip therefore takes about three seconds to
 * make, which is a fact about the progress bar and not about the file (§ 12.1). */
async function toWebm(ctx, opts, count) {
    const MR = root.MediaRecorder;
    if (!MR) throw new Error("this browser cannot record video (no MediaRecorder)");
    const canvas = ctx.surface.compositeCanvas();
    if (!canvas || typeof canvas.captureStream !== "function") {
        throw new Error("this browser cannot record a canvas");
    }
    const mime = (typeof MR.isTypeSupported === "function"
                  && MR.isTypeSupported("video/webm;codecs=vp9"))
        ? "video/webm;codecs=vp9" : "video/webm";
    const stream   = canvas.captureStream(ctx.meta.fps);
    const recorder = new MR(stream, { mimeType: mime, videoBitsPerSecond: WEBM_BITRATE });
    const chunks   = [];
    recorder.ondataavailable = function (e) {
        if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    const stopped = new Promise(function (res) { recorder.onstop = res; });
    recorder.start();

    const perFrame = 1000 / ctx.meta.fps;
    try {
        await eachFrame(ctx, opts, count, function () {
            return new Promise(function (res) { root.setTimeout(res, perFrame); });
        });
    } catch (e) {
        try { recorder.stop(); } catch (_) {}
        await stopped;
        throw e;
    }
    try { recorder.stop(); } catch (_) {}
    await stopped;
    return new root.Blob(chunks, { type: "video/webm" });
}


/* ── The door § 12 describes ─────────────────────────────────────────────────
 *
 * `ctx` is assembled by index.js at call time from its own closure and is not
 * reachable from anywhere else — the export needs the viewer's state, and a
 * hatch on the handle would have given it to everyone (§ 4).
 */
export async function exportAnimation(opts, ctx) {
    opts = opts || {};
    const format = opts.format || "png-zip";
    if (FORMATS.indexOf(format) < 0) {
        throw new Error("cannot export '" + format + "'; the formats are "
                      + FORMATS.join(", "));
    }
    if (!ctx.meta.ready) throw new Error("no mode is showing, so there is nothing to export");

    const cycles = Math.max(1, Math.round(Number(opts.cycles) || 1));
    const count  = ctx.meta.framesPerCycle * cycles;

    /* An export changes exactly two things about the picture, and this is the door
     * that takes them (§ 12). Playback stops for the duration: the loop and the
     * encoder would otherwise both be driving the same surface. */
    const wasPlaying = ctx.playing();
    ctx.pause();
    const endCapture = ctx.surface.beginCapture({
        width:      opts.width,
        height:     opts.height,
        background: opts.background,
    });

    let blob, description;
    try {
        const meta = Object.assign({}, ctx.meta, { cycles: cycles });
        const run  = { "png-zip": toPngZip, "gif": toGif, "webm": toWebm }[format];
        description = describe(meta, format, count);
        blob = await run(Object.assign({}, ctx, { meta: meta }), opts, count, description);
    } finally {
        /* On EVERY path out — done, failed, cancelled — the surface goes back to
         * the size and colour it was, and playback to what it was doing. A viewer
         * left at export resolution is a viewer nobody can use. */
        try { endCapture(); } catch (_) {}
        try { ctx.redraw(); } catch (_) {}
        if (wasPlaying) { try { ctx.play(); } catch (_) {} }
    }

    const stem = "vibration"
        + (ctx.meta.index !== null && ctx.meta.index !== undefined
            ? "-mode-" + ctx.meta.index : "");
    return {
        blob:     blob,
        filename: stem + (format === "png-zip" ? ".zip" : "." + format),
        meta:     description,
    };
}

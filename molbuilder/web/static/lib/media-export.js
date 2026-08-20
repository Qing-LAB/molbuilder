/* media-export — frames in, one animation file out.
 *
 * Contract: docs/web/molview.md § 11.3 (Export → Image's range half).
 * Owns:     encoding a sequence of PNG frames into a `.gif` (the vendored
 *           gif.js encoder, worker-backed) or a `.webm` (MediaRecorder over
 *           a scratch canvas).
 * Called by: MolView's Export menu.  (VibrationView keeps its own private
 *           encoder for now — its migration onto this module is a recorded
 *           deferral, not an accident:
 *           docs/plans/molview-and-checkpoint-ui-plan.md.)
 *
 * THE CALLER OWNS THE FRAMES; this module owns the file.  Frames arrive as
 * PNG Blobs — whatever produced them (a sealed 3D window, a chart, a test's
 * fabricated pixels) is not this module's business — and are painted onto a
 * scratch canvas both formats read from.
 *
 * WEBM IS RECORDED, NOT ASSEMBLED: MediaRecorder watches a live surface, so
 * frames are shown at wall-clock pace and a three-second clip takes about
 * three seconds to make.  That is a fact about the progress, not the file.
 */
"use strict";

const GIF_VENDOR = "/static/vendor/gif.min.js";
const GIF_WORKER = "/static/vendor/gif.worker.min.js";
const WEBM_BITRATE = 8_000_000;

let _gifPromise = null;
function _gifEncoder() {
    // One network fetch + one global mount per page (the same lazy shape as
    // the checkpoint graph's vendor load).
    if (_gifPromise) return _gifPromise;
    _gifPromise = new Promise((resolve, reject) => {
        if (window.GIF) { resolve(window.GIF); return; }
        const tag = document.createElement("script");
        tag.src = GIF_VENDOR;
        tag.async = true;
        tag.onload = () => window.GIF
            ? resolve(window.GIF)
            : reject(new Error("gif.min.js loaded but window.GIF is missing"));
        tag.onerror = () => reject(new Error("failed to load gif.min.js"));
        document.head.appendChild(tag);
    });
    return _gifPromise;
}

async function _paint(ctx, blob) {
    const bitmap = await createImageBitmap(blob);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(bitmap, 0, 0, ctx.canvas.width, ctx.canvas.height);
    bitmap.close();
}

/**
 * One encoder for one animation.
 *
 * @param opts {format: "gif"|"webm", width, height, fps}
 * @returns {addFrame(pngBlob): Promise, finish(): Promise<Blob>}
 *          `addFrame` resolves when the frame is consumed; `finish` resolves
 *          to the encoded file.  Either rejects on an encoder failure — the
 *          caller reports, nothing here swallows.
 */
export async function createFrameEncoder(opts) {
    const { format, width, height } = opts || {};
    const fps = (opts && opts.fps) || 12;
    if (format !== "gif" && format !== "webm") {
        throw new Error("media-export: unknown format " + String(format));
    }
    if (!(width > 0) || !(height > 0)) {
        throw new Error("media-export: a frame size is required");
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const delayMs = 1000 / fps;

    if (format === "gif") {
        const GIF = await _gifEncoder();
        const gif = new GIF({
            workers: 2, quality: 10, width, height,
            workerScript: GIF_WORKER,
        });
        return {
            async addFrame(blob) {
                await _paint(ctx, blob);
                gif.addFrame(canvas, { copy: true, delay: delayMs });
            },
            finish() {
                return new Promise((resolve, reject) => {
                    gif.on("finished", resolve);
                    try { gif.render(); } catch (e) { reject(e); }
                });
            },
        };
    }

    // webm
    const MR = window.MediaRecorder;
    if (!MR) {
        throw new Error("this browser cannot record video (no MediaRecorder)");
    }
    if (typeof canvas.captureStream !== "function") {
        throw new Error("this browser cannot record a canvas");
    }
    const mime = (typeof MR.isTypeSupported === "function"
                  && MR.isTypeSupported("video/webm;codecs=vp9"))
        ? "video/webm;codecs=vp9" : "video/webm";
    const recorder = new MR(canvas.captureStream(fps),
                            { mimeType: mime, videoBitsPerSecond: WEBM_BITRATE });
    const chunks = [];
    recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    const stopped = new Promise((res) => { recorder.onstop = res; });
    recorder.start();
    return {
        async addFrame(blob) {
            await _paint(ctx, blob);
            // Wall-clock pace: the recorder watches the canvas live.
            await new Promise((res) => setTimeout(res, delayMs));
        },
        async finish() {
            try { recorder.stop(); } catch (_) {}
            await stopped;
            return new Blob(chunks, { type: "video/webm" });
        },
    };
}

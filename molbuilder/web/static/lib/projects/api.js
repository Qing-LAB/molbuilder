/* projects/api.js -- HTTP wrappers for /api/files/* + /api/projects/*.
 *
 * Pure functions: no DOM, no module-level state.  Each function
 * issues one HTTP request and returns the answer's fields, ALWAYS
 * shaped as ``{ok: bool, ...}``.  Every failure -- a refusal, an
 * answer that is not JSON, a server not reached, a cancel -- comes
 * back as ``{ok: false, error: "..."}`` so callers don't need a
 * try/catch around every call.
 *
 * The backend contract is in ``docs/web/projects.md`` § and
 * ``docs/web/web-api.md`` § ``/api/files/*``.  The uniform-
 * envelope contract is in ``docs/web/projects.md``
 * Principle 6 ("every public method returns {ok, ...} or {ok:false,
 * error}; NEVER throws").
 */

import { ask } from "../server.js";

/* THE ONE READER (ui-contract.md § 12), in this module's envelope: a yes
 * is the body's fields with ``ok: true``; anything else keeps the body's
 * fields (a refusal's ``actual_mtime``) beside ``ok: false``, ``error``,
 * ``status`` and ``aborted`` / ``signedOut``.  ``opts`` is ``ask``'s; its
 * ``cache: "no-store"`` keeps a live listing from a browser-cached answer. */
async function _fetchEnvelope(url, opts) {
  const a = await ask(url, opts);
  if (a.ok) return Object.assign({}, a.body, { ok: true });
  const out = Object.assign({}, a.body, { ok: false, error: a.error,
                                          status: a.status });
  if (a.aborted) out.aborted = true;
  if (a.signedOut) out.signedOut = true;
  return out;
}


export async function apiRoots() {
  // /api/files/roots returns ``{ok, roots: [...]}`` — we normalise
  // here so a missing ``roots`` key never trips callers:
  const body = await _fetchEnvelope("/api/files/roots");
  if (body.ok === false) {
    return { ok: false, error: body.error, roots: [] };
  }
  return { ok: true, roots: body.roots || [] };
}

export async function apiList(path, opts) {
  opts = opts || {};
  return await _fetchEnvelope(
    "/api/files/list?path=" + encodeURIComponent(path),
    { signal: opts.signal }
  );
}

export async function apiStat(path, opts) {
  opts = opts || {};
  return await _fetchEnvelope(
    "/api/files/stat?path=" + encodeURIComponent(path),
    { signal: opts.signal }
  );
}

export async function apiRead(path, opts) {
  opts = opts || {};
  let url = "/api/files/read?path=" + encodeURIComponent(path);
  // ``opts.maxBytes`` lifts the read budget above the server default so a
  // bulk read can pull up to the caller's ceiling (the preview inspector's
  // 16 MB BULK cap) in one shot; omitted -> server default applies.
  if (opts.maxBytes != null) {
    url += "&max_bytes=" + encodeURIComponent(opts.maxBytes);
  }
  return await _fetchEnvelope(url, { signal: opts.signal });
}

/** Read a byte window from ``path`` at ``offset`` (default 0) with a
 *  cap of ``maxBytes`` (server default 256 KB; hard ceiling 16 MB).
 *  Negative ``offset`` reads from EOF (``offset = -N`` returns the
 *  last N bytes, clamped to file size).  Server contract:
 *  ``{ok, path, offset, length, file_size, mtime, text, eof}`` on
 *  success; ``{ok:false, error}`` on bounds / picker-root / non-UTF8
 *  failure.  Powers the paginated source inspector.  ``opts.signal``
 *  honoured for abort.
 */
export async function apiReadRange(path, offset, maxBytes, opts) {
  opts = opts || {};
  let url = "/api/files/read_range?path=" + encodeURIComponent(path);
  if (offset !== undefined && offset !== null) {
    url += "&offset=" + encodeURIComponent(offset);
  }
  if (maxBytes !== undefined && maxBytes !== null) {
    url += "&max_bytes=" + encodeURIComponent(maxBytes);
  }
  return await _fetchEnvelope(url, { signal: opts.signal });
}

export async function apiMkdir(parent, name, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/files/mkdir", {
    method:  "POST",
    body:    {parent: parent, name: name},
    signal:  opts.signal,
  });
}

export async function apiCreateProject(name, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/projects/create", {
    method:  "POST",
    body:    {name: name},
    signal:  opts.signal,
  });
}

/* Every endpoint accepts an ``opts.signal`` AbortSignal: without a
 * signal threaded all the way to fetch, clicking Cancel during a slow
 * read OR write would unlock the UI but leave the request running. */

export async function apiUpload(targetDir, file, opts) {
  opts = opts || {};
  const fd = new FormData();
  fd.append("target_dir", targetDir);
  // Optional filename override.  Used by writeFile(Blob)
  // to set the destination filename when the Blob has no .name
  // (Blobs assembled from a stream / encoder don't carry one).
  if (opts.filename) {
    fd.append("file", file, opts.filename);
  } else {
    fd.append("file", file);
  }
  if (opts.overwrite) fd.append("overwrite", "true");
  if (opts.auto_rename) fd.append("auto_rename", "true");
  return await _fetchEnvelope("/api/files/upload", {
    method: "POST",
    body:   fd,
    signal: opts.signal,
  });
}

export async function apiDelete(path, recursive, opts) {
  opts = opts || {};
  // ``opts.force`` (2026-06-24): bypass the backend's canonical-
  // topic refusal at depth 2.  Callers that pass true MUST have
  // shown the user a confirmation prompt that explicitly warns
  // about wiping the whole subdirectory (sidebar topic-dir kebab
  // does this).  Default false so non-aware callers keep the
  // safety guard.
  return await _fetchEnvelope("/api/files/delete", {
    method:  "DELETE",
    body:    {
      path:      path,
      recursive: !!recursive,
      force:     !!opts.force,
    },
    signal:  opts.signal,
  });
}

export async function apiRename(path, newName, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/files/rename", {
    method:  "POST",
    body:    {path: path, new_name: newName},
    signal:  opts.signal,
  });
}

/** Move ``path`` into ``destDir``.  Sidecar-aware on .xyz/.pdb (the
 *  paired .molstruct.json moves atomically).  Optional ``opts.newName``
 *  renames during the move.  See /api/files/move docstring + the
 *  web/projects.md § 4.1 (the pair moves as one).  Backend rejects
 *  directory sources in v1. */
export async function apiMove(path, destDir, opts) {
  opts = opts || {};
  const body = {path: path, dest_dir: destDir};
  if (opts.newName) body.new_name = opts.newName;
  return await _fetchEnvelope("/api/files/move", {
    method:  "POST",
    body:    body,
    signal:  opts.signal,
  });
}

/** Copy ``path`` to ``destDir``.  Sidecar-aware on .xyz/.pdb.  Optional
 *  ``opts.newName`` (required when destDir == source's parent).
 *  Same v1 directory-source refusal as move. */
export async function apiCopy(path, destDir, opts) {
  opts = opts || {};
  const body = {path: path, dest_dir: destDir};
  if (opts.newName) body.new_name = opts.newName;
  return await _fetchEnvelope("/api/files/copy", {
    method:  "POST",
    body:    body,
    signal:  opts.signal,
  });
}

export async function apiWrite(path, text, opts) {
  opts = opts || {};
  const body = {path: path, text: text};
  if (opts.overwrite) body.overwrite = true;
  // auto_rename parity with
  // /api/files/upload so the export dialog's promise of
  // "auto-renamed to <name>-2 …" is honored for text writes too.
  if (opts.auto_rename) body.auto_rename = true;
  if (opts.expected_mtime != null) body.expected_mtime = opts.expected_mtime;
  return await _fetchEnvelope("/api/files/write", {
    method:  "POST",
    body:    body,
    signal:  opts.signal,
  });
}

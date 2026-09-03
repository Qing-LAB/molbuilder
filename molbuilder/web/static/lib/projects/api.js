/* projects/api.js -- HTTP wrappers for /api/files/* + /api/projects/*.
 *
 * Pure functions: no DOM, no module-level state.  Each function
 * issues one HTTP request and returns the parsed JSON body, ALWAYS
 * shaped as ``{ok: bool, ...}``.  Network errors AND non-JSON
 * responses are caught here and surfaced as
 * ``{ok: false, error: "..."}`` so callers don't need a try/catch
 * around every call.
 *
 * The backend contract is in ``docs/web/projects.md`` § and
 * ``docs/web/web-api.md`` § ``/api/files/*``.  The uniform-
 * envelope contract is in ``docs/web/projects.md``
 * Principle 6 ("every public method returns {ok, ...} or {ok:false,
 * error}; NEVER throws").
 */

/**
 * Issue one HTTP request and return the parsed JSON body.  All
 * failure modes (network drop, browser offline, server returning
 * 5xx with non-JSON, DNS failure, ...) map to
 * ``{ok: false, error: "<reason>"}`` so callers can rely on
 * ``response.ok`` without try/catch.
 *
 * ``fetchInit`` is forwarded verbatim to ``fetch`` so callers can
 * set method / headers / body / credentials.  The function does
 * NOT throw under any condition expected in production; an
 * unexpected throw indicates a programmer error (e.g. an invalid
 * URL).
 */
async function _fetchEnvelope(url, fetchInit) {
  // Default to ``cache: "no-store"`` so two GETs to the same URL during
  // one browser session BOTH reach the server.  Without this, the
  // browser HTTP cache happily serves the previous response for the
  // same URL -- the exact second-half of the 2026-06-02 /results
  // stale-dropdown bug (#192).  ``/api/files/list`` and ``/api/files/
  // read_range`` return live data that changes between requests for
  // the SAME URL (new files, mtime drift, file appended on disk), so
  // browser caching is correctness-breaking, not perf-helping, here.
  //
  // Callers that legitimately want browser caching (e.g. a future
  // immutable basis-set blob) can override by passing an explicit
  // ``cache`` field in fetchInit; our default only fills in the gap.
  //
  // Note ``cache: "no-store"`` is a no-op for POST/DELETE/PUT (which
  // browsers never cache by default), so applying it uniformly via
  // the central wrapper is safe + sweeps every GET caller in one
  // place.
  const init = fetchInit || {};
  if (init.cache === undefined) init.cache = "no-store";
  let resp;
  try {
    resp = await fetch(url, init);
  } catch (e) {
    // Distinguish user-initiated cancellation (AbortError) from
    // genuine network failure.  Both look like exceptions to the
    // caller, but the UI usually wants to silently dismiss an
    // abort (it was the user's choice) while showing a banner for
    // a network drop.  The ``aborted: true`` flag lets callers
    // branch without parsing the error string.
    if (e && e.name === "AbortError") {
      return { ok: false, error: "aborted", aborted: true };
    }
    // Network-level failure: TypeError "Failed to fetch", DNS
    // failure, CORS rejection at preflight, etc.  Surface the
    // error name + message; callers usually just need to know
    // "could not reach server".
    return {
      ok:    false,
      error: "network error: " + (e && e.message
                                  ? e.message
                                  : String(e)),
    };
  }
  // Successful network, but the server might have returned non-JSON
  // (5xx with an HTML error page, 501 from a stubbed endpoint, ...).
  try {
    return await resp.json();
  } catch (_e) {
    return {
      ok:    false,
      error: "server returned non-JSON (status " + resp.status + ")",
    };
  }
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
  // ``missingOk`` -- read of an OPTIONAL file: a missing file returns 200 with
  // ``{ok:true, exists:false, text:null}`` instead of a 404, so the browser logs no
  // failed-resource console error for a normal absent-optional-file (e.g. a bare .xyz
  // with no .molstruct.json sidecar == empty metadata).
  if (opts.missingOk) {
    url += "&missing_ok=1";
  }
  return await _fetchEnvelope(url, { signal: opts.signal });
}

/** Read a byte window from ``path`` at ``offset`` (default 0) with a
 *  cap of ``maxBytes`` (server default 256 KB; hard ceiling 16 MB).
 *  Negative ``offset`` reads from EOF (``offset = -N`` returns the
 *  last N bytes, clamped to file size).  Server contract:
 *  ``{ok, path, offset, length, file_size, mtime, text, eof}`` on
 *  success; ``{ok:false, error}`` on bounds / picker-root / non-UTF8
 *  failure.  Powers the v2 paginated source inspector (task #119,
 *  2026-06-02); promoted here in #189 so the inspector goes through
 *  the uniform envelope instead of raw ``fetch``.  ``opts.signal``
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
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({parent: parent, name: name}),
    signal:  opts.signal,
  });
}

export async function apiCreateProject(name, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/projects/create", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({name: name}),
    signal:  opts.signal,
  });
}

/* Every endpoint accepts an ``opts.signal`` AbortSignal.  The
 * lock's three-layer recovery (timeout + Cancel button +
 * try/finally) relies on this -- without a signal threaded all
 * the way to fetch, clicking Cancel during a slow read OR write
 * would unlock the UI but leave the request running.  Read
 * endpoints (list / stat / read) added signal support 2026-05-31
 * to close the design § C3 + § C5 contract (#175 follow-up).
 * (See docs/web/projects.md Layer B + #174.) */

export async function apiUpload(targetDir, file, opts) {
  opts = opts || {};
  const fd = new FormData();
  fd.append("target_dir", targetDir);
  // Phase 6e: optional filename override.  Used by writeFile(Blob)
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
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({
      path:      path,
      recursive: !!recursive,
      force:     !!opts.force,
    }),
    signal:  opts.signal,
  });
}

export async function apiRename(path, newName, opts) {
  opts = opts || {};
  return await _fetchEnvelope("/api/files/rename", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({path: path, new_name: newName}),
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
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify(body),
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
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify(body),
    signal:  opts.signal,
  });
}

export async function apiWrite(path, text, opts) {
  opts = opts || {};
  const body = {path: path, text: text};
  if (opts.overwrite) body.overwrite = true;
  // Phase 6e second-review BOMB #11: auto_rename parity with
  // /api/files/upload so the export dialog's promise of
  // "auto-renamed to <name>-2 …" is honored for text writes too.
  if (opts.auto_rename) body.auto_rename = true;
  if (opts.expected_mtime != null) body.expected_mtime = opts.expected_mtime;
  return await _fetchEnvelope("/api/files/write", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify(body),
    signal:  opts.signal,
  });
}

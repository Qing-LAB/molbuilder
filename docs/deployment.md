# Deploying molbuilder over a network (LAN · internet · auth · TLS)

> **Scope.** "Deployment" here means **deploying molbuilder itself**. This
> doc covers the **network-access** half: serving the web app + sign-in
> (CAS / OAuth) + TLS + the **serve-time** `molbuilder.json` sections
> (`tls`, `auth`, `envs`, `secret_key_file`). The **install** half
> (bootstrap the envs) is in [README §Deployment](../README.md#deployment)
> + [`README_install.md`](README_install.md).
> **Not deployment:** *using* the script-generator **module** to submit and
> run calculations (job execution) — generating standalone
> `.run.sh`/`.sbatch`, activation, sbatch, benchmarks — is a separate
> concern in [`job-execution.md`](job-execution.md) (the master doc:
> big picture + workflow + detection contract + cookbook),
> [`config.md`](config.md) (the config schema + wrapper contract), and
> [`protocols/slurm-integration.md`](protocols/slurm-integration.md).
> Job execution owns the `script_generation` / `scheduler` keys of the same
> `molbuilder.json`; this doc owns the serve-keys.

Default deployment is **localhost-only**: `molbuilder serve` binds
`127.0.0.1:8000` and is reachable only from the same machine.  No
auth, no TLS, fully open file API — exactly right for a local
research tool.

This doc covers the **two non-default cases**:

  * §1 **Same-LAN access** (your laptop reaches molbuilder on a
    workstation across the office network).
  * §2 **Internet exposure** (you want to reach molbuilder from
    anywhere).  This is the case that needs real care; molbuilder
    itself has only what's covered in §2a.

---

## Fastest path: ``molbuilder auth-setup``

For the common case (ASU CAS, Google OAuth, or both), the wizard
generates everything correctly with zero hand-editing:

```bash
# After ``conda activate molbuilder``:
python -m molbuilder auth-setup
# or, non-interactively:
python -m molbuilder auth-setup --provider asu             # ASURITE = current user
python -m molbuilder auth-setup --provider google          # prompts for client id+secret
python -m molbuilder auth-setup --provider both --asurite <asurite>
```

What the wizard does:

  * Generates the Flask session signing key locally
    (``secrets.token_urlsafe(32)``) and writes it to
    ``~/.config/molbuilder/secret_key`` mode 0600.  Never echoed,
    never logged, never placed into ``molbuilder.json`` -- the
    config holds only the PATH.
  * For Google OAuth, prompts for the OAuth ``client_id`` (visible
    input) and ``client_secret`` (hidden via ``getpass`` -- no
    terminal echo, no shell history).  The secret is stored at
    ``~/.config/molbuilder/google_client_secret`` mode 0600.
  * Defaults the ASU CAS ``allowed_users`` entry to
    ``<system_user>@asu.edu`` -- the OS-level account name is the
    only identifier the wizard assumes.  Override with ``--asurite``
    if the asurite differs from the local username.
  * Writes ``./molbuilder.json`` mode 0600.  Refuses to clobber an
    existing one unless ``--force`` is passed (and even then
    preserves every non-auth top-level key like ``envs`` or
    ``tls``).

The rest of this doc covers the hand-edit + advanced cases (multi-
provider tuning, `hosted_domain`, CAS `email_attribute`, TLS
provisioning, reverse-proxy setups, secret rotation).

---

## Quick start: the config file

Every non-default deployment is driven by a single config file:
`molbuilder.json`.  A fully-commented template ships with the repo
at [`docs/molbuilder.json.example`](molbuilder.json.example).

The recommended workflow is:

```bash
# 1. Copy the template
cp docs/molbuilder.json.example molbuilder.json

# 2. Open it and delete the sections you don't need; fill in the
#    ones you do (TLS cert paths, Google OAuth client_id, allowed
#    user list, etc.).  Each section has a _comment_<name> key
#    above it explaining what it does + when you'd use it.
$EDITOR molbuilder.json

# 3. Start molbuilder
molbuilder serve --host 0.0.0.0 --port 443
```

Important: the live `molbuilder.json` file is **gitignored** because
it contains paths to credentials (TLS private key, Google OAuth
client_secret).  Never commit it.  The committed template
(`.example` suffix) carries only placeholder values + the schema.

See §3 below for the field-by-field schema reference.

---

## 1. Same-LAN access

If you trust everyone on the network (single-user lab workstation,
home network behind a single router), you can simply bind a
non-loopback host:

    molbuilder serve --host 0.0.0.0 --port 8000 --cert ./cert.pem --key ./key.pem

The `--cert` / `--key` flags satisfy the TLS-or-loopback guard
(`molbuilder.cli._enforce_tls_for_remote_bind`).  Without them
molbuilder refuses to bind a non-loopback host -- file ops are
public to whoever can reach the interface and TLS at minimum stops
passive sniffing.

A self-signed cert is fine for LAN:

    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout key.pem -out cert.pem -days 365 \
        -subj "/CN=$(hostname -f)"

Browsers warn on first visit; accept the exception once per device.

**This is still single-trust-zone.** Anyone on the LAN who reaches
the URL can read / write / delete every file in `projects/`.  If
that's not OK for your network, go to §2.

---

## 2. Internet exposure

molbuilder's default file API is fully open: anyone who can connect
to the port has full read/write/delete on `projects/`.  Exposing
that directly is a non-starter.

For an internet-reachable deployment you have **two reasonable
paths**.  The first uses molbuilder's built-in Google sign-in
(simplest; recommended); the second outsources auth entirely to
infrastructure in front of molbuilder.

### 2a. Built-in sign-in (recommended)

molbuilder ships an optional auth layer that supports **five backend
kinds**: Google, GitHub, Microsoft / Azure AD, ORCID (all OAuth 2.0
/ OIDC), and Apereo CAS (e.g. ASU's CAS for ASURITE NetIDs).  When
enabled, molbuilder shows a sign-in page first with **one button
per configured provider** — pick one, the provider handles the
actual login (and any institutional federation, e.g. `@asu.edu`
Google accounts routing through ASU's IdP transparently), molbuilder
gets back only the verified identity, checks it against THAT
provider's `allowed_users` list, and the user lands on the UI.

You can enable multiple providers simultaneously (e.g. Google for
external collaborators + ASU CAS for on-campus users).  Each
provider has its own `allowed_users` list — there is no global
allowlist; an email allowed via Google is *not* implicitly allowed
via GitHub.

The setup work below is for **you, the molbuilder operator** —
done once per deployment per provider you enable.  Google is the
canonical worked example (§ 2a.1); GitHub / Microsoft / ORCID
follow the same OAuth pattern with their own consoles (§ 2a.2);
ASU CAS uses a different protocol (§ 2a.3).

#### 2a.1. Google sign-in

#### Setup walkthrough: Cloud Console → molbuilder.json

##### Why this step exists (read once, then skip)

Google requires every app that uses "Sign in with Google" to be
registered with them first — Slack, Notion, Trello, GitHub
Desktop, every desktop / web app you've used Google login with
went through the same step.  It's not specific to molbuilder, and
**you're not deploying anything to Google's cloud** — despite the
"Cloud Console" name, the registration is just a web form you
fill out once.  Three reasons Google asks for it:

  1. **Identity**.  The screen Google shows users at login —
     "molbuilder wants access to your name and email" — uses the
     app name + logo you register here.  Without registration,
     Google wouldn't know what to call your app.
  2. **Anti-phishing**.  Google will only redirect users back to
     a URL you pre-declared.  A phishing site that copies the
     molbuilder login flow can't trick Google into redirecting
     credentials to its own URL.
  3. **Revocability**.  If something goes wrong (server
     compromised, secret leaked), you can revoke or rotate the
     credentials from the Cloud Console and instantly cut off
     access.

Total time for the registration: ~5 minutes, once per deployment
(or once total if you reuse the same client across reinstalls).
You only revisit Cloud Console if you change the hostname, add
more allowed redirect URIs, or rotate the secret.

##### What you walk away with

The registration produces **three values** you'll need:

  * **Client ID** — a long public string ending in
    `.apps.googleusercontent.com`.  Goes into `molbuilder.json`.
  * **Client secret** — a shorter private string starting with
    `GOCSPX-`.  Goes into a separate file (so the main config
    stays safe to share / commit-the-template).
  * **Authorized redirect URI** — you pre-declare the URL Google
    is allowed to redirect users back to after login.  Must match
    your molbuilder hostname exactly.

You'll also need to **decide upfront** what hostname molbuilder
will be reachable at (e.g. `molbuilder.qlabsrv.physics.asu.edu`).
Google needs this for the redirect URI; molbuilder needs a TLS
cert valid for the same hostname.  Don't pick the hostname after
you've already filled out the Cloud Console form — you'd have to
re-edit the redirect URI list.

##### Step-by-step

| # | In Cloud Console | What you get | Where it goes |
|---|---|---|---|
| 1 | <https://console.cloud.google.com> → Create project (or pick existing). Name it whatever -- "molbuilder" is fine. | A project to scope everything below | (nothing — just the context for the rest) |
| 2 | APIs & Services → **OAuth consent screen** → User type "External" → fill in App name (`molbuilder`), support email, developer email → Save | Consent screen exists | (nothing — required prerequisite for step 3) |
| 3 | APIs & Services → **Credentials** → "Create Credentials" → **OAuth client ID** → Application type: **Web application** | OAuth client form | (the form's outputs go into config) |
| 4 | In the form, set **Authorized redirect URIs** to: `https://<your-molbuilder-hostname>/oauth-callback/google`<br>(e.g., `https://molbuilder.qlabsrv.physics.asu.edu/oauth-callback/google`) | Google now trusts that callback URL | (config-internal — molbuilder generates this URL automatically from the provider's `id`; you just have to declare it to Google) |
| 5 | Click **Create**.  Google shows a modal with two values. | "Client ID" → long `.apps.googleusercontent.com` string<br>"Client secret" → `GOCSPX-xxx...` string | • Client ID → `client_id` field of the Google provider entry in `molbuilder.json`<br>• Client secret → save to a file (next step) |
| 6 | (On the molbuilder server, NOT in Cloud Console) Save the secret to a 0600 file: `echo 'GOCSPX-xxxx' > ~/.molbuilder/google_client_secret && chmod 600 ~/.molbuilder/google_client_secret` | A protected file with just the secret | Path → `client_secret_file` field of the Google provider entry |
| 7 | (Optional, recommended) APIs & Services → OAuth consent screen → **Publishing status: In production** (or leave in "Testing" if your `allowed_users` list is short — see Testing-mode caveat below) | Either Google shows your app to anyone the operator allowed, or limits to the test-user list you set in the consent screen | (no config change) |

##### The full molbuilder.json after the walkthrough

```json
{
  "tls": {
    "cert": "/etc/letsencrypt/live/molbuilder.your-host.edu/fullchain.pem",
    "key":  "/etc/letsencrypt/live/molbuilder.your-host.edu/privkey.pem"
  },

  "auth": {
    "providers": [
      {
        "id":                 "google",
        //   ^^^^ keys the route path /login/google + /oauth-callback/google.
        //        Must be unique across providers + URL-safe (slug).
        "label":              "Sign in with Google",
        //   ^^^^ text on the login-page button (Google requires this exact phrase)
        "kind":               "google",

        "client_id":          "1234567890-abcdef.apps.googleusercontent.com",
        //   ^^^^ from Cloud Console step 5

        "client_secret_file": "~/.molbuilder/google_client_secret",
        //   ^^^^ the file you wrote in step 6 (containing the GOCSPX- value)

        "hosted_domain":      [],
        //   ^^^^ optional IdP-side filter -- e.g. ["asu.edu"] restricts to
        //        Google Workspace accounts in that domain (stricter than
        //        allowed_users alone; check happens before molbuilder sees
        //        the request).  Empty list = no domain filter.

        "allowed_users": [
          "user1@example.edu",
          "user2@example.edu",
          "external-friend@gmail.com"
        ]
        //   ^^^^ you decide this; NOT controlled by Google Cloud Console.
        //        Add/remove emails freely, restart molbuilder.
      }
    ]
  },

  "secret_key_file": "~/.molbuilder/secret.key"
  //   ^^^^ auto-generated on first run (mode 0600)
}
```

(JSON doesn't support `//` comments — they're annotations for this
doc only.  The actual file at [`molbuilder.json.example`](molbuilder.json.example)
uses `_comment_<section>` keys instead, which the parser silently
ignores.)

##### Field-by-field reference (Google provider entry)

| `molbuilder.json` field | What it is | Source |
|---|---|---|
| `auth.providers[i].id` | URL-safe slug; keys `/login/<id>` + `/oauth-callback/<id>` | You write it — pick something stable (renaming breaks the Cloud Console redirect URI) |
| `auth.providers[i].label` | Button text on the login page.  For Google this MUST be "Sign in with Google" (Google brand requirement) | You write it |
| `auth.providers[i].kind` | Backend selector (`google` here) | You write it |
| `auth.providers[i].client_id` | Public identifier Google uses to recognise molbuilder | Cloud Console step 5 |
| `auth.providers[i].client_secret_file` | Path to a 0600 file containing the secret | You create the file (step 6) and put its path here |
| `auth.providers[i].client_secret` (alt) | Literal secret string. Use ONLY when `client_secret_file` is inconvenient (then the whole config file becomes secret-sensitive — gitignore it for real) | Cloud Console step 5 |
| `auth.providers[i].hosted_domain` | Optional list of Google Workspace domains to restrict to (IdP-side filter) | **You decide** |
| `auth.providers[i].allowed_users` | Email allowlist for *this provider* (case-insensitive).  No global list — each provider's gate is independent | **You decide** — Google has no opinion |
| `secret_key_file` | Flask session signing key.  Auto-generated on first run | molbuilder writes it |
| `tls.cert` / `tls.key` | TLS cert + key for the hostname matching Cloud Console step 4 | Let's Encrypt / your institution / etc. |

##### Caveats worth knowing

**Cloud Console "Testing" vs "In production" status.**  Newly-
created OAuth consent screens default to "Testing".  In Testing
mode, only emails in the consent screen's "Test users" list (max
100) can sign in -- the molbuilder `allowed_users` list is a
SECOND filter on top of that.  Simplest for a small lab: leave
the consent screen in Testing, add your colleagues there AND in
`allowed_users`.  For a wider audience: switch to "In production"
(Google may ask a verification questionnaire if you request
sensitive scopes; we only ask for `email + profile`, which is
non-sensitive and doesn't trigger review).

**The hostname has to be stable.**  The redirect URI is registered
in Cloud Console; molbuilder generates the same URL from the
incoming request host.  If you decide later to rename your server
or add another hostname, edit the Cloud Console redirect URIs to
match -- the OAuth handshake fails (with a clear error) when they
don't.

**Rotating the secret.**  If the `GOCSPX-` value ever leaks, go
back to Cloud Console → your client ID → "Reset client secret",
overwrite the file with the new value, restart molbuilder.  No
config-format change.

##### Install + run

```bash
pip install 'molbuilder[auth]'        # pulls Authlib + python-cas
molbuilder serve --host 0.0.0.0 --port 443 \
    --cert /etc/letsencrypt/live/...../fullchain.pem \
    --key  /etc/letsencrypt/live/...../privkey.pem
```

Open the browser to your hostname.  You see the molbuilder
sign-in page with one button per configured provider.  Click
"Sign in with Google".  Google authenticates.  You land on
molbuilder.  Adding a user later = edit that provider's
`allowed_users` list in the config, restart molbuilder.  Google
owns all the password management; molbuilder never sees credentials.

At startup, molbuilder logs the **exact redirect URI** it will
hand to each OAuth provider — copy these from the startup log
into the respective consoles' authorized-redirect-URI lists.
This takes the guesswork out of host/port/scheme/proxy
interactions (the most common cause of `redirect_uri_mismatch`).

#### 2a.2. Other OAuth providers (GitHub, Microsoft, ORCID)

All three follow the **same OAuth 2.0 / OIDC pattern as Google**:
register an app in their respective developer console, get a
client_id + client_secret, declare a redirect URI of the form
`https://<your-host>/oauth-callback/<provider_id>`, drop the values
into a new entry in `auth.providers`.  The differences are which
console you register in and one or two per-provider config fields.

##### GitHub

| | |
|---|---|
| Console | <https://github.com/settings/developers> → **OAuth Apps** → **New OAuth App** |
| Authorization callback URL | `https://<your-host>/oauth-callback/github` |
| Required scopes | `user:email` (always), `read:org` (only when `allowed_organizations` is set) |
| Extra config field | `allowed_organizations` (optional, list of org slugs) — restricts access to members of these GitHub orgs (IdP-side filter) |

```json
{
  "id":                    "github",
  "label":                 "Sign in with GitHub",
  "kind":                  "github",
  "client_id":             "Iv1.abcdef...",
  "client_secret_file":    "~/.molbuilder/github_client_secret",
  "allowed_organizations": ["my-research-lab"],
  "allowed_users":         ["user1@example.edu"]
}
```

GitHub returns the user's **primary verified email** even when
hidden from their public profile (molbuilder requests `user:email`
unconditionally for exactly this reason).

##### Microsoft / Azure AD

| | |
|---|---|
| Console | <https://portal.azure.com> → **Microsoft Entra ID** → **App registrations** → **New registration** |
| Redirect URI (type: Web) | `https://<your-host>/oauth-callback/microsoft` |
| Secret | **Certificates & secrets** → New client secret (copy the *value*, not the ID, when it appears — it's never shown again) |
| Extra config field | `tenant_id` — `"common"` (any Microsoft account inc. personal), `"organizations"` (any work/school), a tenant GUID, or a verified domain like `"asu.onmicrosoft.com"` (restricts to that tenant) |

```json
{
  "id":                 "microsoft",
  "label":              "Sign in with Microsoft",
  "kind":               "microsoft",
  "client_id":          "abc12345-de67-89f0-1234-56789abcdef0",
  "client_secret_file": "~/.molbuilder/microsoft_client_secret",
  "tenant_id":          "asu.onmicrosoft.com",
  "allowed_users":      ["user1@example.edu"]
}
```

##### ORCID

| | |
|---|---|
| Console | <https://orcid.org/developer-tools> → **Register a Public API client** |
| Redirect URI | `https://<your-host>/oauth-callback/orcid` |
| Identity returned | ORCID iD (16-digit string, e.g. `0000-0001-2345-6789`).  Email is included **only if the user has marked it public** in their ORCID profile. |
| Extra config field | (none) |

```json
{
  "id":                 "orcid",
  "label":              "Sign in with ORCID",
  "kind":               "orcid",
  "client_id":          "APP-1234567890ABCDEF",
  "client_secret_file": "~/.molbuilder/orcid_client_secret",
  "allowed_users": [
    "user1@example.edu",
    "0000-0001-2345-6789"
  ]
}
```

When ORCID returns an email, molbuilder matches it against
`allowed_users` like any other OAuth provider.  When it doesn't,
the ORCID iD itself is used as the identity — so `allowed_users`
for ORCID is typically a mix of emails (for collaborators whose
email IS public) and bare ORCID iDs (for those whose isn't).

#### 2a.3. Apereo CAS (e.g. ASURITE)

CAS is a different protocol from OAuth — it's older, simpler, and
used by most institutional SSO systems including ASU's
`weblogin.asu.edu`.  No client_id / secret is required: trust is
established by registering the CAS endpoint URLs.

| | |
|---|---|
| Console | None — CAS uses fixed URLs published by your institution.  ASU's CAS endpoints are public knowledge (the table below). |
| Callback URL | `https://<your-host>/cas-callback/<provider_id>` — auto-registered via the `service_url` parameter on each request (no pre-declaration in any console) |
| Identity returned | The CAS principal (the ASURITE username, e.g. `<asurite>`) plus optional attributes.  ASU CAS releases **only the principal** (no email attribute), so molbuilder synthesises `{principal}@{email_domain}` |

```json
{
  "id":                   "asu_cas",
  "label":                "Sign in with ASURITE ID",
  "kind":                 "cas",
  "version":              3,
  "login_url":            "https://weblogin.asu.edu/cas/login",
  "service_validate_url": "https://weblogin.asu.edu/cas/serviceValidate",
  "email_domain":         "asu.edu",
  "allowed_users": [
    "user1@asu.edu",
    "user2@asu.edu"
  ]
}
```

If your CAS DOES release an email attribute (some institutional
CASes do; ASU's doesn't), set `email_attribute` to its CAS
attribute name (commonly `"mail"`); molbuilder will prefer that
over the synthesised form.  Either `email_attribute` or
`email_domain` must be set — schema validation enforces this at
config-load time.

CAS does NOT use the per-provider redirect URI registration that
OAuth requires (each request carries its own `service_url`
parameter), so you don't need any console-side step for CAS.

### 2b. Reverse proxy with separate auth gateway

If you have an existing auth gateway you'd rather use (Authelia,
oauth2-proxy, Cloudflare Access, an SSO already in front of every
service at your institution), put molbuilder behind it and skip
the Google integration above.  Three patterns work:

#### 2b.i. Tailscale / WireGuard / VPN tunnel

Easiest if you control the client machines.  Put molbuilder behind
a private overlay network; only authenticated devices on the tunnel
reach the host at all.

  * **Tailscale**: install on the molbuilder host + each user's
    laptop.  Bind molbuilder to the tailnet interface (e.g.,
    `--host 100.x.y.z`).  Tailscale handles WireGuard + auth +
    ACLs; molbuilder sees only authenticated requests.
  * **Tailscale Funnel**: if you also want public HTTPS without
    the user installing Tailscale, Funnel publishes a `*.ts.net`
    URL gated by Tailscale auth.

  Effort: 30 minutes.  Cost: free for small teams.
  Threat model: as strong as Tailscale's auth (very strong).

#### 2b.ii. Reverse proxy with auth gateway

Standard pattern: nginx / Caddy / Traefik terminates TLS + an auth
proxy (Authelia / oauth2-proxy / Cloudflare Access) gates requests
before they reach molbuilder.  molbuilder binds loopback only and
trusts every request that reaches it (since the proxy already
authenticated).

Minimal Caddy config (HTTPS + Basic Auth):

    molbuilder.example.com {
        # Caddy auto-provisions Let's Encrypt cert.
        basicauth {
            researcher $2a$14$<bcrypt-hash-here>
        }
        reverse_proxy localhost:8000 {
            header_up X-Forwarded-Proto {scheme}
        }
    }

Molbuilder's `Strict-Transport-Security` header is enabled when
the request carries `X-Forwarded-Proto: https` (handled
automatically in this Caddy config).

  Effort: 1-2 hours including DNS + Let's Encrypt.
  Cost: free; needs a public hostname + a reachable port.
  Threat model: as strong as your auth choice + the proxy's
  patch level.

#### 2b.iii. SSH port-forwarding for occasional access

If you just want to grab one or two files remotely:

    ssh -L 8000:127.0.0.1:8000 user@molbuilder-host

Then open `http://localhost:8000` on your local machine.  Molbuilder
sees only the loopback bind on the host; SSH provides the tunnel +
auth.

  Effort: 0 setup.  Best for ad-hoc remote use, not team sharing.

---

## 3. Config reference (every section of `molbuilder.json`)

Full template with realistic values:
[`docs/molbuilder.json.example`](molbuilder.json.example).

### `tls`

```json
"tls": {
  "cert": "/etc/letsencrypt/live/<hostname>/fullchain.pem",
  "key":  "/etc/letsencrypt/live/<hostname>/privkey.pem"
}
```

Paths to a TLS cert / key pair.  Required for any non-loopback
bind (`molbuilder serve --host` refuses without TLS unless you
pass `--allow-insecure-binding`).  Both keys are strings; paths
are read as-is (no `~` expansion in this section — give absolute
paths).

For Let's Encrypt:

```bash
sudo certbot certonly --standalone -d <hostname>
# → cert path:  /etc/letsencrypt/live/<hostname>/fullchain.pem
# → key path:   /etc/letsencrypt/live/<hostname>/privkey.pem
```

For self-signed (LAN only):

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout key.pem -out cert.pem -days 365 \
    -subj "/CN=$(hostname -f)"
```

Omit the whole `tls` section for localhost-only deployments.

### `envs`

```json
"envs": {
  "siesta":  "molbuilder-siesta",
  "pyscf":   "molbuilder-pySCF",
  "mdtools": "molbuilder-MDtools"
}
```

Conda env name overrides.  molbuilder dispatches PySCF / SIESTA /
MDTools work into the matching env so per-engine deps don't
pollute the host env.  Defaults are the names above; only set
this section if you renamed your envs.

### `auth`

See §2a for per-backend walkthroughs.  Schema:

```json
"auth": {
  "providers": [
    {
      "id":                 "google",
      "label":              "Sign in with Google",
      "kind":               "google",
      "client_id":          "...apps.googleusercontent.com",
      "client_secret_file": "~/.molbuilder/google_client_secret",
      "hosted_domain":      [],
      "allowed_users":      ["user1@example.edu"]
    },
    {
      "id":                   "asu_cas",
      "label":                "Sign in with ASURITE ID",
      "kind":                 "cas",
      "login_url":            "https://weblogin.asu.edu/cas/login",
      "service_validate_url": "https://weblogin.asu.edu/cas/serviceValidate",
      "email_domain":         "asu.edu",
      "allowed_users":        ["user1@asu.edu"]
    }
  ]
}
```

`auth.providers` is a **non-empty list** of provider entries; each
entry becomes a separate button on the login page (rendered in the
declared order).

Top-level optional flag:

| Key | Default | What it does |
|---|---|---|
| `auth.trust_proxy` | `false` | When `true`, molbuilder installs `ProxyFix` and trusts the FIRST upstream proxy hop's `X-Forwarded-Proto` / `X-Forwarded-Host` headers when building OAuth redirect URIs and CAS service URLs. **Only set this when there is an actual reverse proxy in front of molbuilder** that scrubs client-supplied forwarded headers before adding its own (Caddy / nginx / Traefik / Cloudflare Tunnel all do this by default).  In a direct-TLS deploy (`molbuilder serve --cert --key` with nothing in front), leaving this off is critical — an attacker who can reach the bind interface could otherwise send a spoofed `X-Forwarded-Host` and influence the URLs molbuilder generates. |

Common fields (every kind):

| Key | Required? | What it is |
|---|---|---|
| `id` | yes | URL-safe slug; keys `/login/<id>` and the callback path.  Must be unique across all entries |
| `label` | yes | Button text on the login page |
| `kind` | yes | One of `google`, `github`, `microsoft`, `orcid`, `cas` |
| `allowed_users` | yes | Email allowlist for THIS provider (case-insensitive).  Empty list = no one (fail-closed).  No global allowlist — each provider's gate is independent |

OAuth-kind extras (google / github / microsoft / orcid):

| Key | Required? | What it is |
|---|---|---|
| `client_id` | yes | From the provider's developer console |
| `client_secret_file` | one of these two | Path to a 0600 file with just the secret string |
| `client_secret` | one of these two | Literal secret (less safe; whole config becomes secret-sensitive) |
| `hosted_domain` (google only) | optional | List of Google Workspace domains to restrict to (IdP-side) |
| `allowed_organizations` (github only) | optional | List of GitHub org slugs to restrict to (IdP-side) |
| `tenant_id` (microsoft only) | optional, default `"common"` | Tenant scope: `common` / `organizations` / GUID / verified domain |

CAS extras:

| Key | Required? | What it is |
|---|---|---|
| `login_url` | yes | CAS login endpoint (e.g. `https://weblogin.asu.edu/cas/login`) |
| `service_validate_url` | yes | CAS ticket-validation endpoint |
| `version` | optional, default `3` | CAS protocol version (1, 2, or 3) |
| `service_url` | optional | Override for the return URL (default: auto-derive from request) |
| `ca_certs` | optional | Path to CA bundle (default: system trust store) |
| `email_attribute` | at least one of these | CAS attribute name carrying the email, when released |
| `email_domain` | at least one of these | Domain to synthesise `{principal}@{domain}` when no attribute |

Omit the whole `auth` section to run with no authentication
(localhost-only single-user shape).

### `secret_key_file`

```json
"secret_key_file": "~/.molbuilder/secret.key"
```

Path to the file holding the Flask session-signing key.  Auto-
generated on first run with 32 random bytes (mode 0600).  Required
when `auth` is set; quietly ignored otherwise.  If omitted while
auth is on, molbuilder falls back to a per-process random key
(sessions invalidate on every restart — safe but annoying).

`~` expansion IS honoured here (unlike `tls.cert/key` paths).

### Things `molbuilder.json` deliberately does NOT carry

If you find yourself wanting to add any of these, the answer is
"that doesn't belong in molbuilder.json" — see §2b for the
reverse-proxy place to put them instead:

  * User passwords (we never store any — Google owns those)
  * Per-user permission grants beyond `allowed_users` (defer to
    institution SSO groups via the proxy)
  * Rate-limit thresholds (proxy concern)
  * Audit log destinations (proxy concern)
  * API keys for third-party services (out of scope)

---

## What molbuilder does on its own

Even with the reverse proxy doing the heavy lifting, the molbuilder
code itself implements several defense-in-depth measures:

  * **TLS-or-loopback guard** (`molbuilder serve --host` refuses
    non-loopback bind without `--cert` / `--key`, unless you pass
    `--allow-insecure-binding`).
  * **Security headers** on every response (`Content-Security-Policy`,
    `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
    `Referrer-Policy: same-origin`, and `Strict-Transport-Security`
    when the response is HTTPS).  CSP blocks inline JS, restricts
    sources to molbuilder's own origin -- if an XSS slips through
    the audit suite, CSP stops it from loading attacker payloads.
  * **No CDN for 3D viewer**: 3Dmol.js is vendored under
    `static/vendor/` (same pattern as `vendor/plotly.min.js`).  No
    third-party domains in any served page -- a cdnjs compromise
    can't reach molbuilder users.  License + version + upstream
    citation: `molbuilder/web/static/vendor/README.md`
    (BSD-3-Clause; cite Rego & Koes 2015 *Bioinformatics* if you
    publish work using the viewer).
  * **Path validation** on every file-ops endpoint (rejects `..`,
    URL-decoded variants, symlink-escape, canonical-topic dir
    deletion).  Single `_depth_inside_root` helper shared across
    write / upload / delete so the security boundary stays
    consistent.
  * **Filename validation** on upload (basename regex; rejects
    dotfiles + shell metacharacters).
  * **Upload size cap** (`MAX_CONTENT_LENGTH = 50 MB`, JSON
    413 handler).
  * **XSS regression test suite** (`tests/test_xss_audit.py`) that
    forbids unsafe `innerHTML` / `eval` / `javascript:` / `\| safe`
    / event-handler-attribute patterns across all served JS + Jinja
    templates.

## What molbuilder explicitly does NOT do

Each of these is a deliberate non-feature; doing them ALL would
turn molbuilder into a SaaS framework instead of a research tool.
Pick the deployment shape (§2a/b/c) that covers what you need:

  * Authentication, authorization, account management
  * CSRF token issuance + validation
  * Rate limiting
  * Audit log of who-deleted-what-when
  * Per-user `projects/` isolation
  * Backup / undo for destructive operations
  * Brute-force protection
  * Dependency-CVE scanning

If you need any of these, the reverse proxy (§2b) is the right
place to add them, not the molbuilder process.

---

## Deployment checklist

Before exposing molbuilder beyond `127.0.0.1`:

  - [ ] Picked one of §2a / §2b / §2c.
  - [ ] If §2a or §2c: verified the tunnel + authentication.
  - [ ] If §2b: verified the reverse proxy denies unauthenticated
        requests reaching the molbuilder port.
  - [ ] TLS in front of molbuilder (proxy-terminated or
        `--cert`/`--key` direct).
  - [ ] `--host` bind reaches only the proxy / tunnel interface,
        not `0.0.0.0` when proxy is on a different host.
  - [ ] Confirmed the file system user molbuilder runs as has
        read/write only on the `projects/` tree (no `sudo`,
        no home-dir access).
  - [ ] Documented for teammates: what URL to use, how auth works,
        what's destructive (delete is real, no undo).

---

## Reporting security issues

Found a security gap not covered here?  Open a GitHub issue (or
email the maintainer if it's exploitable as-described).  The XSS
regression suite catches everything we've already audited; new
classes of bug land via real-user reports.

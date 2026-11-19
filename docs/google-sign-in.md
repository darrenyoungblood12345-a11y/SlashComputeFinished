# Google sign-in

Accounts are Google accounts. There is no email-and-password sign-up: typing an address into a
form proves nothing about who owns it.

Google is the directory. Each pool keeps its own account table (credits, grants and nodes are per
pool), keyed by Google's stable id for the person (`sub`). The same Google account signs in on any
pool; nothing is shared between pools.

## How a sign-in runs

Google refuses to show its sign-in page inside an embedded web view, and the app's window is one.
So the app uses the flow Google supports for desktop apps: the system browser and a loopback
redirect.

1. **Sign in with Google** in the sidebar asks the local shell (`127.0.0.1:8766`) to start.
2. The shell asks the pool for a single-use nonce (`POST /auth/google/challenge`) and opens Google's
   consent page in your browser.
3. Google sends the browser back to `http://127.0.0.1:<shell port>/api/auth/google/callback` with a
   one-time code.
4. The shell swaps the code for an ID token and posts it to the pool (`POST /auth/google`).
5. The pool checks the token against Google's published signing keys (signature, issuer, audience,
   expiry, verified email, and its own nonce), finds or creates the account, and issues a session.
6. The shell stores the session for that pool and restarts the agent and LLM node with it. The tab
   says you can close it; the window shows the account within a couple of seconds.

The session token never reaches a web page. PKCE ties the code to the shell that asked for it,
`state` ties the browser's return to a sign-in this shell started, and the nonce ties the ID token
to the pool it was requested for (every pool accepts tokens for the same app, so without it a
token handed to one pool could be replayed at another).

## Setting it up (once per build of the app)

Sign-in is off until the build has an OAuth client.

1. In Google Cloud Console, create a project and configure the OAuth consent screen (app name,
   support email; scopes `openid`, `email`, `profile`). While the app is in "Testing", only the
   test users you list can sign in; publish it to let anyone in.
2. Create credentials: **OAuth client ID**, application type **Desktop app**.
3. Give the build the client:
   - for a release, set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in
     `src/slashcompute/common/config.py` (an app opened from Finder has no environment to read);
   - for development, export `SLASHCOMPUTE_GOOGLE_CLIENT_ID` and
     `SLASHCOMPUTE_GOOGLE_CLIENT_SECRET` before `uv run slashcompute`.

Google does not treat a desktop client's secret as confidential ("Desktop apps are public clients
and cannot keep secrets confidential"); PKCE is what protects the flow. The secret is sent only if
one is set.

The pool and the app must use the same client ID: the pool only accepts tokens issued to its own.
A pool hosted from the app inherits the app's. A coordinator started by hand
(`slashcompute-coordinator serve`) needs `SLASHCOMPUTE_GOOGLE_CLIENT_ID` in its environment unless
the value is in `config.py`.

`SLASHCOMPUTE_ADMIN_EMAIL` on the coordinator makes that address an admin the first time Google
verifies it. Otherwise the first account on a pool is its admin.

Not yet checked against a real client: Google's guide shows loopback redirects without a path
(`http://127.0.0.1:port`). This build uses `/api/auth/google/callback` on the shell's port. If
Google answers `redirect_uri_mismatch` on the first real sign-in, add that exact URI to the
client's authorized redirect URIs, or tell us and we move the callback to the root path.

## Accounts from before

Upgrading a pool that had email-and-password accounts:

- Every session opened with a password ends at the first start, and the stored passwords are
  blanked. Agents and LLM nodes still running on such a session keep computing on a LAN pool but
  stop earning for an account until their Mac signs in again.
- The accounts themselves stay, with their credits, grants and nodes. An account is taken over by
  the first Google sign-in whose verified email matches it.
- An account whose email no Google account holds can no longer be signed in to.

## When it does not work

| What you see | Why | Fix |
|---|---|---|
| `Google sign-in is not set up in this build of the app.` | No client ID in `config.py` or the environment | Set it (above) |
| `This pool runs an older version without Google sign-in.` | The coordinator predates this flow | Update the Mac hosting the pool |
| `This pool takes sign-ins from a different build of the app.` | The pool and the app have different client IDs | Use the same build on both, or the same `SLASHCOMPUTE_GOOGLE_CLIENT_ID` |
| `Could not open your browser to sign in.` | macOS has no default browser to hand the page to | Set a default browser and press the button again |
| The tab says `This sign-in link has expired or was already used.` | The browser came back after 10 minutes, twice, or to another copy of the app | Press **Sign in with Google** again |
| The tab says `Google refused the sign-in (…)` | Google rejected the code exchange; the text in brackets is Google's reason | `redirect_uri_mismatch`: see the note above. `invalid_client`: the secret does not belong to the client ID |
| The tab says `The pool refused the sign-in: …` | The pool checked the token and said no: a banned account, an expired sign-in, or a pool that cannot reach Google's keys (`503`) | Follow the message; a pool needs internet access the first time it checks a token |
| `Error 403: disallowed_useragent` on Google's page | The page was opened inside an embedded view | It must open in the system browser; report it if the app's button did this |
| Signed in, but the window still shows the button | The app was pointed at another pool while you were in the browser, so the session was not kept | Sign in again on the pool you are on |

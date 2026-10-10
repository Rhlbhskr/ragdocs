# Streamlit production deployment

Deploy `streamlit_app/app.py` with Python 3.12. The dependency file next to that
entrypoint includes CPU embeddings, PostgreSQL, and Authlib for native OIDC login.
Start with reranking disabled. No separate FastAPI deployment is needed.

## Services and secrets

Provision external PostgreSQL with TLS and backups, Qdrant Cloud, a Gemini project,
and a Google OAuth client of type **Web application**. Register the exact callback URL
`https://YOUR-APP.streamlit.app/oauth2callback` in the identity provider. Local
development uses `http://localhost:8501/oauth2callback` and a separately registered
callback. Generate a random cookie secret, for example with
`python -c "import secrets; print(secrets.token_urlsafe(48))"`.

Copy the placeholder [Secrets template](../.streamlit/secrets.toml.example) into
Streamlit Cloud's Secrets editor and replace every placeholder. Do not commit the
real file. Set `APP_MODE="production"`, your PostgreSQL URL with `sslmode=require`,
both Qdrant credentials, a Gemini key/model accessible to your project, and the
Google login settings. The template allows anyone with a Google account:
`ALLOWED_USERS=[]`, `ALLOW_ALL_AUTHENTICATED=true`, and
`AUTH_ISSUER="https://accounts.google.com"`. Each account gets a separate private
workspace without manual approval. You can optionally restrict access by filling
`ALLOWED_USERS` with verified email addresses. Issuer and subject claims must be present.

In Google Cloud, configure the OAuth consent screen for external users and publish
the app when ready for public access. Google's testing restrictions depend on
requested scopes; basic `openid`, `email`, and `profile` scopes have an exception
to the test-user restriction. Request only these identity scopes. Set the Google
OAuth client ID and client secret under `[auth]` in Streamlit
Secrets. Keep Google's discovery URL from the template. Follow
[Google's OpenID Connect setup](https://developers.google.com/identity/openid-connect).

Production requires semantic embeddings and refuses missing provider credentials
or local database storage. Authentication runs before model loading. Native
Streamlit authentication establishes identity; the application derives a private
workspace from the authenticated issuer and subject and checks authorization.
See [Streamlit authentication](https://docs.streamlit.io/develop/concepts/connections/authentication).

## Data and recovery

PostgreSQL stores extracted source passages, document metadata, the last 50 chat
turns, and daily usage counters. Qdrant stores vectors and workspace/document IDs.
Production vectors do not expire. Original upload binaries are discarded. New
accounts start empty. Documents remain until the owner deletes them; deletion also
clears chat because historical citations contain source text. Clearing chat alone
does not reset usage limits. If you enable an allowlist, removing an entry prevents access on rerun.

Database writes use per-workspace PostgreSQL advisory locks. Each operation reloads
state, preventing stale tabs from overwriting each other's libraries. Question slots
are committed before provider work, so failures still count. SQL is authoritative:
the app re-upserts a workspace's vectors before questions, repairing missing indexes
and embedding collection changes. This bounded strategy prioritizes recovery over
large-corpus throughput. Failed uploads may leave orphan vectors if SQL commit or
cleanup fails; retrieval filters against SQL document IDs so those points cannot
become evidence. Periodically reconcile Qdrant with database backups.

Limits: three active uploaded documents, 10 MB and 50 PDF pages per file, 200 KB
extracted text per document, 200 chunks per account, ten question attempts per UTC
day, one-second question interval, and five retrieved passages. A process-wide
20/minute generation guard supplements persistent account quotas. PDF scanning/OCR
is unsupported. These limits deliberately bound synchronous ingestion on Streamlit.

Generated answers send selected source text, questions, and recent chat to Gemini.
Provider failure returns clearly labeled cited excerpts. Citation IDs are validated;
the lexical evidence heuristic does not verify every claim or prevent all prompt
injection. Inspect source evidence for consequential answers.

## Release acceptance

Run `python -m pytest -q`. Then test the actual deployed URL:

1. Signed-out visitors see Google login; a Google account can sign in without manual approval.
2. Upload a text PDF, ask a question, and inspect page and chunk citations.
3. Sign out, reconnect, and restart the host; your library and history persist.
4. A second Google account cannot view the first account's documents or citations.
5. Delete a document and verify both vectors and chat are removed; clear chat and
   confirm question usage remains.
6. Test exhausted daily quota, invalid uploads, provider quota errors, and unavailable
   Qdrant/PostgreSQL. Test parallel tabs against actual PostgreSQL advisory locks.
7. Restore PostgreSQL and Qdrant backups in staging and repeat the question flow.

Configure provider spending caps, monitor host memory and service errors, and enable
database/Qdrant backups. Keep staging services separate. Community Cloud capacity
and uptime must be measured for your workload; use managed Streamlit infrastructure
with an appropriate service agreement if your availability requirements demand it.

Local tests verify behavior with SQLite, embedded Qdrant, and mocked Gemini. Actual
OIDC login, PostgreSQL locking/TLS, Cloud service connectivity, recovery, and capacity
require deployment credentials and environment testing before a production release.

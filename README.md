# GroundedDocs

A Streamlit RAG app: upload PDF, DOCX, or TXT documents, ask questions, and inspect answers with source citations.

The pipeline parses documents, creates overlapping chunks, embeds them with MiniLM, retrieves from Qdrant, and generates grounded answers with Gemini. Dense, keyword, and hybrid retrieval are available; CrossEncoder reranking is optional. The implementation uses direct Python integrations rather than LangChain.

## Production mode

- Google sign-in, open to anyone with a Google account; an optional allowlist can restrict access.
- Private workspaces derived from authenticated identity.
- PostgreSQL-backed document text, chat history, and daily question quotas.
- Qdrant Cloud vectors with recovery from stored source passages.
- Bounded ingestion, citation validation, low-evidence refusal, and clearly labeled excerpt fallback when Gemini fails.
- No automatic sample documents in private accounts.

Follow the [Streamlit production deployment guide](docs/STREAMLIT_PRODUCTION.md) and use [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) as the configuration template. Deploy **streamlit_app/app.py** with Python 3.12. Production mode requires external PostgreSQL, Qdrant, Gemini, and OIDC credentials.

## Local development

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r streamlit_app/requirements.txt
.venv/Scripts/python.exe -m streamlit run streamlit_app/app.py
```

Without Secrets, the app runs an anonymous temporary demo with sample documents and excerpt answers. For offline testing set `APP_MODE="demo"` and `EMBEDDING_BACKEND="hash"`. For production, copy the template into Streamlit Secrets and replace all placeholders; never commit real credentials.

## Validation and release status

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
```

Automated tests cover ingestion, retrieval, citations, session/account isolation, durable storage, deletion, quota persistence, vector repair, and provider failure behavior. Live authentication, hosted PostgreSQL/Qdrant/Gemini connections, backups, and workload capacity still require deployment verification. There is no verified public deployment yet.

The application is prepared for a bounded authenticated deployment; production readiness depends on the acceptance checks in the deployment guide. Evidence confidence is a lexical heuristic, not a factuality guarantee. Original files are not retained; extracted text remains until deleted. Generated answers send relevant excerpts and recent chat to Gemini.

The [previous architecture reference](docs/LEGACY_README.md) documents the retained FastAPI and other deployment alternatives. Its older Streamlit session-only instructions describe demo mode. Streamlit production instructions above take precedence.

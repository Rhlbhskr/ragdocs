from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4
import logging
import json

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .config import settings
from .grounding import answer_question
from .parsing import parse_document, chunk_pages
from .retrieval import Retriever
from .storage import Base, engine, records, save, Session, Record, workspace_lock
from .limits import BodyLimitMiddleware
from .providers import ProviderError

ROOT = Path(__file__).resolve().parent.parent
retriever = None
logger = logging.getLogger('groundeddocs')


@asynccontextmanager
async def lifespan(app):
    global retriever
    settings.validate_runtime()
    with workspace_lock('__schema__'):
        Base.metadata.create_all(engine)
    retriever = Retriever()
    # SQL records are authoritative. Rebuild the selected collection on startup,
    # including after an embedding model change or a lost local vector directory.
    if not settings.vercel:
        with Session() as db:
            for doc in db.query(Record).filter_by(kind='document'):
                retriever.index(doc.workspace, doc.id, doc.payload['passages'])
    yield
    retriever.client.close()
    engine.dispose()


app = FastAPI(title='GroundedDocs', version='1.0.0', lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_mb * 1024 * 1024 + 64 * 1024)


@app.exception_handler(ProviderError)
async def provider_error(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=502, content={'detail': str(exc)})


@app.middleware('http')
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'same-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'" if not request.url.path.startswith(('/docs', '/redoc')) else "frame-ancestors 'none'"
    if request.url.path.startswith('/api'):
        response.headers['Cache-Control'] = 'no-store'
    return response


def workspace(request: Request, require_initialized=True):
    value = request.cookies.get('grounded_workspace', '')
    try:
        identifier = str(UUID(value, version=4))
    except ValueError:
        raise HTTPException(401, 'Initialize a workspace first.')
    if require_initialized and not records(identifier, 'initialized'):
        raise HTTPException(401, 'Initialize a workspace first.')
    return identifier


def ingest(ws, name, content, size):
    docs = records(ws, 'document')
    if len(docs) >= 20:
        raise ValueError('This workspace is limited to 20 documents. Delete a document first.')
    pages = parse_document(name, content)
    if settings.vercel and sum(len(text.encode('utf-8')) for _, text in pages) > 200_000:
        raise ValueError('The hosted demo supports up to 200 KB of extracted text per document. Split this document into smaller files.')
    chunks = chunk_pages(pages, size=size, overlap=min(64, size // 4))
    if len(chunks) > settings.max_chunks or sum(d['chunks'] for d in docs) + len(chunks) > settings.max_chunks:
        raise ValueError('Workspace chunk limit reached. Use smaller documents or remove old ones.')
    identifier = str(uuid4())
    for c in chunks:
        c.update(id=str(uuid4()), name=name, document_id=identifier)
    payload = {'name': name, 'chunks': len(chunks), 'pages': len(pages) if name.lower().endswith('.pdf') else None, 'chunk_size': size, 'created_at': datetime.now(timezone.utc).isoformat(), 'passages': chunks, 'vector_collection': retriever.collection}
    try:
        retriever.index(ws, identifier, chunks)
        save(ws, 'document', identifier, payload)
    except Exception:
        try:
            retriever.delete(ws, identifier)
        except Exception:
            logger.exception('Failed to clean up a partial vector upload')
        raise
    return dict(payload, id=identifier)


def public_document(doc):
    return {k: v for k, v in doc.items() if k not in ('passages', 'vector_collection')}


def ensure_workspace_index(ws, docs):
    """Only reindex documents whose embedding configuration changed."""
    for doc in docs:
        if doc.get('vector_collection') != retriever.collection:
            retriever.index(ws, doc['id'], doc['passages'])
            if doc.get('vector_collection'):
                retriever.delete(ws, doc['id'], collection=doc['vector_collection'])
            with Session.begin() as db:
                record = db.query(Record).filter_by(id=doc['id'], workspace=ws).one()
                record.payload = dict(record.payload, vector_collection=retriever.collection)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'embedding_backend': settings.embedding_backend, 'reranker': settings.enable_reranker, 'generation': 'llm' if settings.llm_base_url else 'extractive', 'max_upload_mb': settings.max_upload_mb, 'serverless': settings.vercel}


@app.post('/api/workspace')
def initialize(request: Request, response: Response):
    try:
        ws = workspace(request, require_initialized=False)
    except HTTPException:
        ws = str(uuid4())
        response.set_cookie('grounded_workspace', ws, httponly=True, secure=settings.cookie_secure, samesite='strict', max_age=60 * 60 * 24 * 30)
    with workspace_lock('__new_workspaces__'):
        if not records(ws, 'initialized'):
            with Session() as db:
                if db.query(Record).filter_by(kind='initialized').count() >= settings.max_workspaces:
                    raise HTTPException(503, 'The demo has reached its workspace capacity. Please try again later.')
            existing = {d['name'] for d in records(ws, 'document')}
            for path in sorted((ROOT / 'samples').glob('*.txt')):
                if path.name not in existing:
                    ingest(ws, path.name, path.read_bytes(), 512)
            save(ws, 'initialized', str(uuid4()), {})
    return {'ready': True}


@app.get('/api/documents')
def documents(request: Request):
    return [public_document(d) for d in records(workspace(request), 'document')]


@app.post('/api/documents', status_code=201)
def upload(request: Request, file: UploadFile = File(...), chunk_size: int = Form(512)):
    if chunk_size not in (256, 512, 1024):
        raise HTTPException(422, 'Chunk size must be 256, 512, or 1024 words.')
    content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, 'File exceeds upload size limit.')
    name = Path((file.filename or 'document').replace('\\', '/')).name[:200]
    ws = workspace(request)
    try:
        with workspace_lock(ws):
            return public_document(ingest(ws, name, content, chunk_size))
    except HTTPException:
        raise
    except ProviderError:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        logger.exception('Document ingestion failed')
        raise HTTPException(422, 'The document could not be parsed or indexed. Verify the file and retry.')


@app.get('/api/documents/{identifier}/source')
def source(identifier: str, request: Request):
    doc = next((d for d in records(workspace(request), 'document') if d['id'] == identifier), None)
    if not doc:
        raise HTTPException(404, 'Document not found.')
    return {'name': doc['name'], 'passages': doc['passages']}


@app.delete('/api/documents/{identifier}', status_code=204)
def remove(identifier: str, request: Request):
    ws = workspace(request)
    with workspace_lock(ws):
        with Session.begin() as db:
            doc = db.query(Record).filter_by(id=identifier, workspace=ws, kind='document').first()
            if not doc:
                raise HTTPException(404, 'Document not found.')
            collections = {retriever.collection, doc.payload.get('vector_collection', retriever.collection)}
            for collection in collections:
                retriever.delete(ws, identifier, collection=collection)
            db.delete(doc)
            # Historical citations contain excerpts; remove history on document deletion.
            db.query(Record).filter_by(workspace=ws, kind='turn').delete()


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: Literal['dense', 'keyword', 'hybrid'] = 'hybrid'
    top_k: Literal[3, 5, 10] = 5
    rerank: bool = True


@app.post('/api/chat')
def chat(body: Question, request: Request):
    start = monotonic()
    ws = workspace(request)
    query = body.question.strip()
    if len(query) < 2:
        raise HTTPException(422, 'Enter a question with at least two characters.')
    with workspace_lock(ws):
        history = sorted(records(ws, 'turn'), key=lambda t: t['created_at'])
        if history and (datetime.now(timezone.utc) - datetime.fromisoformat(history[-1]['created_at'])).total_seconds() < 1:
            raise HTTPException(429, 'Please wait a moment before asking again.')
        docs = records(ws, 'document')
        ensure_workspace_index(ws, docs)
        chunks = [c for doc in docs for c in doc['passages']]
        # Previous question helps resolve follow-ups, while the current question stays primary.
        retrieval_query = query
        if history and any(word in query.lower().split() for word in ('it', 'they', 'that', 'those', 'them')):
            retrieval_query += ' ' + history[-1]['question']
        passages = retriever.search(ws, retrieval_query, chunks, body.mode, body.top_k, body.rerank)
        try:
            result = answer_question(query, passages, history)
        except Exception:
            logger.exception('Answer generation failed')
            raise HTTPException(502, 'The answer provider is unavailable. Try again or configure extractive mode.')
        result.update(question=query, passages=passages, latency_ms=round((monotonic() - start) * 1000), mode=body.mode, reranked=bool(body.rerank and retriever.reranker), created_at=datetime.now(timezone.utc).isoformat())
        save(ws, 'turn', str(uuid4()), result)
        # Keep bounded server-side conversation history.
        if len(history) >= 50:
            with Session.begin() as db:
                db.query(Record).filter(Record.id.in_([t['id'] for t in history[:-49]])).delete(synchronize_session=False)
        return result


@app.get('/api/history')
def history(request: Request):
    turns = sorted(records(workspace(request), 'turn'), key=lambda t: t['created_at'])
    if settings.vercel:
        # Return the newest complete turns within a safe response-body budget.
        bounded, total = [], 0
        for turn in reversed(turns):
            size = len(json.dumps(turn, ensure_ascii=False).encode('utf-8'))
            if total + size > 3_000_000:
                break
            bounded.append(turn)
            total += size
        return list(reversed(bounded))
    return turns


@app.delete('/api/history', status_code=204)
def clear_history(request: Request):
    ws = workspace(request)
    with workspace_lock(ws), Session.begin() as db:
        db.query(Record).filter_by(workspace=ws, kind='turn').delete()


@app.get('/')
def index():
    return FileResponse(ROOT / 'web' / 'index.html')


app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')

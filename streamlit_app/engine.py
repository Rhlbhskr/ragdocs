"""Session-scoped documents and chat; shared, locked inference/vector resources."""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import monotonic
from uuid import uuid4

from app.config import Settings
from app.grounding import answer_question
from app.parsing import parse_document, chunk_pages
from app.providers import ProviderError

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_MB = 10
MAX_UPLOADS = 3
MAX_QUESTIONS = 10
MAX_CHUNKS = 200
MAX_TEXT_BYTES = 200_000


def config_from_secrets(values):
    def value(key, default=''):
        return values.get(key, default)
    gemini_key = str(value('GEMINI_API_KEY')).strip()
    url = str(value('QDRANT_URL')).strip()
    key = str(value('QDRANT_API_KEY')).strip()
    app_mode = str(value('APP_MODE', 'demo')).lower()
    if app_mode not in ('demo', 'production'):
        raise ValueError('APP_MODE must be demo or production.')
    production = app_mode == 'production'
    if production:
        if not url or not key or not gemini_key:
            raise ValueError('Production requires Qdrant and Gemini credentials.')
        if value('EMBEDDING_BACKEND', 'sentence-transformers') != 'sentence-transformers':
            raise ValueError('Production requires semantic embeddings.')
    if bool(url) != bool(key):
        raise ValueError('Set both QDRANT_URL and QDRANT_API_KEY in Streamlit Secrets, or omit both for local demo mode.')
    if url and not url.startswith('https://'):
        raise ValueError('Use the HTTPS cluster URL from Qdrant Cloud.')
    backend = str(value('EMBEDDING_BACKEND', 'sentence-transformers'))
    if backend not in ('sentence-transformers', 'hash'):
        raise ValueError('For Streamlit, EMBEDDING_BACKEND must be sentence-transformers or hash.')
    enabled = value('ENABLE_RERANKER', False)
    if isinstance(enabled, str):
        enabled = enabled.lower() in ('true', '1', 'yes')
    return Settings(
        _env_file=None, vercel=False, qdrant_url=url, qdrant_api_key=key,
        qdrant_in_memory=not url, qdrant_collection_prefix='groundeddocs_streamlit',
        vector_ttl_seconds=0 if production else 86400, embedding_backend=backend,
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        model_cache_dir='./data/models/hub', enable_reranker=enabled,
        reranker_backend='cross-encoder', llm_provider='gemini' if gemini_key else 'extractive',
        gemini_api_key=gemini_key, gemini_model=str(value('GEMINI_MODEL', 'gemini-3.5-flash-lite')),
        llm_base_url='', llm_api_key='', llm_model='', max_upload_mb=MAX_FILE_MB,
    )


@dataclass
class Resources:
    retriever: object
    lock: RLock = field(default_factory=RLock)
    generated_requests: deque = field(default_factory=deque)
    last_cleanup: float = 0

    def cleanup(self):
        with self.lock:
            if monotonic() - self.last_cleanup > 3600:
                self.retriever.cleanup_expired()
                self.last_cleanup = monotonic()

    def reserve_generation(self):
        with self.lock:
            now = monotonic()
            while self.generated_requests and self.generated_requests[0] < now - 60:
                self.generated_requests.popleft()
            if len(self.generated_requests) >= 20:
                raise ProviderError('The demo is busy. Showing source excerpts instead; try Gemini again in a minute.')
            self.generated_requests.append(now)


@dataclass
class DemoSession:
    workspace: str = field(default_factory=lambda: str(uuid4()))
    documents: list = field(default_factory=list)
    history: list = field(default_factory=list)
    uploads_used: int = 0
    questions_used: int = 0
    last_question_at: float = 0

    def ingest(self, name, content, resources, chunk_size=512, sample=False):
        name = Path(name.replace('\\', '/')).name[:200]
        if not sample and self.uploads_used >= MAX_UPLOADS:
            raise ValueError('This demo permits three uploaded documents per browser session.')
        if any(d['name'] == name for d in self.documents):
            raise ValueError('A document with that name is already in this session.')
        if len(content) > MAX_FILE_MB * 1024 * 1024:
            raise ValueError('Documents must be 10 MB or smaller.')
        pages = parse_document(name, content, max_pages=50)
        if sum(len(text.encode('utf-8')) for _, text in pages) > MAX_TEXT_BYTES:
            raise ValueError('This demo accepts up to 200 KB of extracted text per document. Split larger documents.')
        chunks = chunk_pages(pages, size=chunk_size, overlap=min(64, chunk_size // 4))
        if len(chunks) + sum(len(d['passages']) for d in self.documents) > MAX_CHUNKS:
            raise ValueError('This session has reached its 200-chunk limit.')
        identifier = str(uuid4())
        for chunk in chunks:
            chunk.update(id=str(uuid4()), document_id=identifier, name=name)
        resources.cleanup()
        with resources.lock:
            try:
                resources.retriever.index(self.workspace, identifier, chunks)
            except Exception:
                try:
                    resources.retriever.delete(self.workspace, identifier)
                except Exception:
                    pass
                raise
        self.documents.append({'id': identifier, 'name': name, 'passages': chunks, 'sample': sample})
        if not sample:
            self.uploads_used += 1
        return self.documents[-1]

    def seed_samples(self, resources):
        existing = {d['name'] for d in self.documents}
        for path in sorted((ROOT / 'samples').glob('*.txt')):
            if path.name not in existing:
                self.ingest(path.name, path.read_bytes(), resources, sample=True)

    def remove(self, identifier, resources):
        with resources.lock:
            resources.retriever.delete(self.workspace, identifier)
        self.documents = [d for d in self.documents if d['id'] != identifier]
        # Citations in earlier turns can contain deleted source text.
        self.history.clear()

    def ask(self, question, resources, config, mode='hybrid', rerank=False, generated=True):
        question = question.strip()
        if len(question) < 2 or len(question) > 2000:
            raise ValueError('Enter a question between 2 and 2,000 characters.')
        if self.questions_used >= MAX_QUESTIONS:
            raise ValueError('This demo permits ten questions per browser session.')
        if monotonic() - self.last_question_at < 1:
            raise ValueError('Please wait a moment before asking another question.')
        self.questions_used += 1
        self.last_question_at = monotonic()
        start = monotonic()
        query = question
        if self.history and any(t in question.lower().split() for t in ('it', 'that', 'those', 'them')):
            query += ' ' + self.history[-1]['question']
        chunks = [p for d in self.documents for p in d['passages']]
        with resources.lock:
            passages = resources.retriever.search(self.workspace, query, chunks, mode=mode, top_k=5, rerank=rerank)
        warning = ''
        selected = config if generated else config.model_copy(update={'llm_provider': 'extractive'})
        try:
            if selected.llm_provider == 'gemini':
                resources.reserve_generation()
            result = answer_question(question, passages, self.history, config=selected)
        except ProviderError as exc:
            warning = str(exc)
            result = answer_question(question, passages, self.history, config=config.model_copy(update={'llm_provider': 'extractive'}))
        result.update(question=question, passages=passages, warning=warning, mode=mode,
                      reranked=bool(rerank and resources.retriever.reranker),
                      latency_ms=round((monotonic() - start) * 1000))
        self.history.append(result)
        return result

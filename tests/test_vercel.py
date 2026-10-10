import asyncio
import os
from pathlib import Path
import subprocess
import sys
import httpx
import pytest

from app.config import Settings, settings
from app.providers import CohereReranker, ProviderError, embed_cohere
from app.storage import normalize_database_url


def hosted_settings(**overrides):
    values = dict(vercel=True, database_url='postgresql://user:password@db.example/app?sslmode=require',
                  qdrant_url='https://vectors.example:6333', qdrant_api_key='test-qdrant-key',
                  embedding_backend='cohere', cohere_api_key='test-provider-key',
                  enable_reranker=True, reranker_backend='cohere', llm_base_url='', _env_file=None)
    values.update(overrides)
    return Settings(**values)


def test_vercel_runtime_configuration():
    config = hosted_settings()
    config.validate_runtime()
    assert config.max_upload_mb == 4 and config.max_chunks == 200 and config.cookie_secure
    for field, bad in [('database_url', 'sqlite:///data.db'), ('qdrant_url', ''),
                       ('qdrant_api_key', ''), ('embedding_backend', 'sentence-transformers'),
                       ('reranker_backend', 'cross-encoder'), ('cohere_api_key', '')]:
        with pytest.raises(ValueError):
            hosted_settings(**{field: bad}).validate_runtime()
    hosted_settings(embedding_backend='hash', enable_reranker=False, cohere_api_key='').validate_runtime()


def test_host_database_urls_use_installed_driver():
    assert normalize_database_url('postgres://user@host/db?sslmode=require') == 'postgresql+psycopg://user@host/db?sslmode=require'
    assert normalize_database_url('postgresql://user@host/db') == 'postgresql+psycopg://user@host/db'
    assert normalize_database_url('sqlite:///data.db') == 'sqlite:///data.db'


def test_build_import_does_not_write_data_or_import_models(tmp_path):
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, VERCEL='1', PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
               DATABASE_URL='postgresql://user:password@db.invalid/app', QDRANT_URL='https://vectors.invalid',
               QDRANT_API_KEY='test', EMBEDDING_BACKEND='cohere', COHERE_API_KEY='test',
               ENABLE_RERANKER='true', RERANKER_BACKEND='cohere')
    script = "from pathlib import Path; import sys; import app.main; assert not Path('data').exists(); assert 'sentence_transformers' not in sys.modules; assert app.main.retriever is None; assert any(m.kwargs.get('max_bytes') == 4 * 1024 * 1024 + 64 * 1024 for m in app.main.app.user_middleware)"
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_embed_http_contract_and_batching(monkeypatch):
    import app.providers as providers
    monkeypatch.setattr(settings, 'cohere_api_key', 'test-provider-key')
    monkeypatch.setattr(settings, 'cohere_embedding_dimension', 256)
    requests = []
    def handle(request):
        import json
        payload = json.loads(request.content)
        requests.append(payload)
        assert str(request.url) == 'https://api.cohere.com/v2/embed'
        assert request.headers['authorization'] == 'Bearer test-provider-key'
        return httpx.Response(200, json={'embeddings': {'float': [[.1] * 256 for _ in payload['texts']]}})
    client_class = httpx.Client
    monkeypatch.setattr(providers.httpx, 'Client', lambda **kwargs: client_class(transport=httpx.MockTransport(handle), **kwargs))
    vectors = embed_cohere(['document'] * 33, 'search_document')
    assert len(vectors) == 33 and [len(r['texts']) for r in requests] == [32, 1]
    embed_cohere(['question'], 'search_query')
    assert requests[-1]['input_type'] == 'search_query'
    assert requests[0]['input_type'] == 'search_document'
    assert requests[0]['embedding_types'] == ['float'] and requests[0]['output_dimension'] == 256


def test_embedding_provider_failure_and_invalid_shape(monkeypatch):
    import app.providers as providers
    original_request = providers.cohere_request
    monkeypatch.setattr(providers, 'cohere_request', lambda *a: {'embeddings': {'float': [[1.0]]}})
    with pytest.raises(ProviderError, match='invalid response'):
        embed_cohere(['document'], 'search_document')
    monkeypatch.setattr(providers, 'cohere_request', original_request)
    client_class = httpx.Client
    def fail(request):
        return httpx.Response(429, json={'secret': 'must not appear in user error'})
    monkeypatch.setattr(providers.httpx, 'Client', lambda **kwargs: client_class(transport=httpx.MockTransport(fail), **kwargs))
    with pytest.raises(ProviderError) as exc:
        providers.cohere_request('embed', {'texts': ['private text']})
    assert 'secret' not in str(exc.value) and 'private text' not in str(exc.value)


def test_hosted_rerank_preserves_citations_and_rejects_invalid_indices(monkeypatch):
    import app.providers as providers
    passages = [{'id': 'a', 'text': 'alpha', 'page': 4, 'document_id': 'doc-a'},
                {'id': 'b', 'text': 'beta', 'page': 9, 'document_id': 'doc-b'}]
    monkeypatch.setattr(providers, 'cohere_request', lambda *a: {'results': [
        {'index': 1, 'relevance_score': .9}, {'index': 0, 'relevance_score': .1}]})
    result = CohereReranker().rank('question', passages)
    assert result[0]['id'] == 'b' and result[0]['page'] == 9 and result[0]['document_id'] == 'doc-b'
    monkeypatch.setattr(providers, 'cohere_request', lambda *a: {'results': [
        {'index': 0, 'relevance_score': .9}, {'index': 0, 'relevance_score': .1}]})
    with pytest.raises(ProviderError):
        CohereReranker().rank('question', passages)


def test_serverless_cold_start_does_not_reindex_all_documents(monkeypatch):
    import app.main as main
    monkeypatch.setattr(settings, 'vercel', True)
    monkeypatch.setattr(type(settings), 'validate_runtime', lambda self: None)
    class DummyClient:
        def close(self): pass
    class DummyRetriever:
        client = DummyClient()
        def index(self, *args): raise AssertionError('Cold start must not reindex stored documents')
    monkeypatch.setattr(main, 'Retriever', DummyRetriever)
    async def run():
        async with main.lifespan(main.app):
            assert isinstance(main.retriever, DummyRetriever)
    previous = main.retriever
    try:
        asyncio.run(run())
    finally:
        main.retriever = previous


def test_cohere_retrieval_end_to_end_with_real_vector_store(monkeypatch, tmp_path):
    import uuid
    import app.providers as providers
    from app.retrieval import Retriever
    from app.grounding import answer_question
    monkeypatch.setattr(settings, 'embedding_backend', 'cohere')
    monkeypatch.setattr(settings, 'cohere_embedding_dimension', 256)
    monkeypatch.setattr(settings, 'enable_reranker', True)
    monkeypatch.setattr(settings, 'reranker_backend', 'cohere')
    monkeypatch.setattr(settings, 'qdrant_url', '')
    monkeypatch.setattr(settings, 'qdrant_path', str(tmp_path / 'vectors'))
    calls = []
    def request(endpoint, payload):
        calls.append((endpoint, payload))
        if endpoint == 'embed':
            vectors = []
            for text in payload['texts']:
                vector = [0.] * 256
                vector[0 if 'allowance' in text.lower() else 1] = 1.
                vectors.append(vector)
            return {'embeddings': {'float': vectors}}
        return {'results': [{'index': i, 'relevance_score': .9 if 'allowance' in text.lower() else .1}
                            for i, text in enumerate(payload['documents'])]}
    monkeypatch.setattr(providers, 'cohere_request', request)
    retriever = Retriever()
    doc_id = str(uuid.uuid4())
    chunks = [{'id': str(uuid.uuid4()), 'text': text, 'page': page, 'number': index + 1,
               'name': 'Policy.pdf', 'document_id': doc_id}
              for index, (page, text) in enumerate([(4, 'The home office allowance is $750.'),
                                                    (9, 'The database backup schedule is hourly.')])]
    try:
        retriever.index('workspace-a', doc_id, chunks)
        passages = retriever.search('workspace-a', 'What is the home office allowance?', chunks, top_k=3)
        assert passages[0]['page'] == 4 and passages[0]['rerank_score'] == .9
        result = answer_question('What is the home office allowance?', passages, [])
        assert '$750' in result['answer'] and result['citations'][0]['page'] == 4
        assert calls[0][1]['input_type'] == 'search_document'
        assert calls[1][1]['input_type'] == 'search_query'
        assert calls[2][0] == 'rerank'
        assert retriever.search('workspace-b', 'home office allowance', chunks, mode='dense') == []
    finally:
        retriever.client.close()


def test_remote_qdrant_authentication_and_filter_indexes(monkeypatch):
    import app.retrieval as retrieval
    monkeypatch.setattr(settings, 'embedding_backend', 'hash')
    monkeypatch.setattr(settings, 'enable_reranker', False)
    monkeypatch.setattr(settings, 'qdrant_url', 'https://vectors.example:6333')
    monkeypatch.setattr(settings, 'qdrant_api_key', 'test-qdrant-key')
    captured = {}
    class FakeClient:
        def __init__(self, **kwargs): captured['connection'] = kwargs
        def collection_exists(self, name): return True
        def create_payload_index(self, collection, key, schema, **kwargs):
            captured.setdefault('indexes', []).append(key)
    monkeypatch.setattr(retrieval, 'QdrantClient', FakeClient)
    retrieval.Retriever()
    assert captured['connection']['url'] == 'https://vectors.example:6333'
    assert captured['connection']['api_key'] == 'test-qdrant-key'
    assert captured['indexes'] == ['workspace', 'document_id']


def test_embedding_change_reindexes_workspace_once(monkeypatch):
    import uuid
    import app.main as main
    from app.storage import Base, engine, save, records, Session, Record
    Base.metadata.create_all(engine)
    ws, did, cid = [str(uuid.uuid4()) for _ in range(3)]
    chunks = [{'id': cid, 'text': 'A policy passage', 'page': 4}]
    save(ws, 'document', did, {'passages': chunks, 'vector_collection': 'old_collection'})
    indexed, deleted = [], []
    class DummyRetriever:
        collection = 'new_collection'
        def index(self, workspace, document_id, passages): indexed.append((workspace, document_id, passages))
        def delete(self, workspace, document_id, collection=None): deleted.append(collection)
    monkeypatch.setattr(main, 'retriever', DummyRetriever())
    try:
        main.ensure_workspace_index(ws, records(ws, 'document'))
        main.ensure_workspace_index(ws, records(ws, 'document'))
        assert len(indexed) == 1 and indexed[0][2][0]['id'] == cid
        assert deleted == ['old_collection']
        assert records(ws, 'document')[0]['vector_collection'] == 'new_collection'
    finally:
        with Session.begin() as db:
            db.query(Record).filter_by(id=did).delete()


def test_serverless_history_response_stays_within_budget(monkeypatch):
    import app.main as main
    from starlette.requests import Request
    monkeypatch.setattr(settings, 'vercel', True)
    monkeypatch.setattr(main, 'workspace', lambda request: 'workspace')
    turns = [{'created_at': f'2026-10-06T00:00:0{i}+00:00', 'answer': 'é' * 600_000} for i in range(4)]
    monkeypatch.setattr(main, 'records', lambda *a: turns)
    result = main.history(Request({'type': 'http'}))
    assert len(result) == 2 and result[-1]['created_at'] == turns[-1]['created_at']
    assert result[0]['answer'] == turns[2]['answer']


def test_postgres_workspace_lock_is_transaction_scoped(monkeypatch):
    import app.storage as storage
    from contextlib import contextmanager
    from types import SimpleNamespace
    events = []
    class Connection:
        def execute(self, statement, params=None): events.append((str(statement), params))
    class FakeEngine:
        dialect = SimpleNamespace(name='postgresql')
        @contextmanager
        def begin(self):
            events.append('begin')
            try:
                yield Connection()
            finally:
                events.append('end')
    monkeypatch.setattr(storage, 'engine', FakeEngine())
    with pytest.raises(RuntimeError):
        with storage.workspace_lock('browser-workspace'):
            assert 'pg_advisory_xact_lock' in events[-1][0]
            assert isinstance(events[-1][1]['key'], int)
            raise RuntimeError('failed request')
    assert events[0] == 'begin' and events[-1] == 'end'

import io
import json
from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

from app import gemini
from app.providers import ProviderError
from app.retrieval import Retriever
from streamlit_app.engine import config_from_secrets, DemoSession, Resources


@pytest.fixture
def runtime():
    config = config_from_secrets({'EMBEDDING_BACKEND': 'hash'})
    resources = Resources(Retriever(config))
    yield config, resources
    resources.retriever.client.close()


def test_streamlit_upload_citations_history_and_isolation(runtime):
    config, resources = runtime
    first, second = DemoSession(), DemoSession()
    first.seed_samples(resources)
    doc = first.ingest('Launch.txt', b'The Orchid launch date is November 17, 2026.', resources)
    second.seed_samples(resources)
    answer = first.ask('What is the Orchid launch date?', resources, config)
    assert 'November 17, 2026' in answer['answer']
    assert any(c['document_id'] == doc['id'] for c in answer['citations'])
    assert first.history[-1] == answer
    assert all(p['document_id'] != doc['id'] for p in second.ask('What is the Orchid launch date?', resources, config)['passages'])
    first.remove(doc['id'], resources)
    assert not first.history
    assert first.uploads_used == 1 and first.questions_used == 1


def test_demo_upload_question_and_page_limits(runtime):
    config, resources = runtime
    session = DemoSession()
    for index in range(3):
        session.ingest(f'{index}.txt', b'Home office allowance is 750 dollars.', resources)
    with pytest.raises(ValueError, match='three uploaded'):
        session.ingest('extra.txt', b'Extra source.', resources)
    for _ in range(10):
        session.last_question_at = 0
        session.ask('What is the home office allowance?', resources, config)
    session.history.clear()
    with pytest.raises(ValueError, match='ten questions'):
        session.ask('What is the home office allowance?', resources, config)
    writer = PdfWriter()
    for _ in range(51):
        writer.add_blank_page(width=100, height=100)
    content = io.BytesIO()
    writer.write(content)
    with pytest.raises(ValueError, match='50 pages'):
        DemoSession().ingest('large.pdf', content.getvalue(), resources)


def mock_gemini(monkeypatch, handler):
    client_class = httpx.Client
    monkeypatch.setattr(gemini.httpx, 'Client', lambda **kw: client_class(transport=httpx.MockTransport(handler), **kw))


def test_gemini_header_context_and_valid_citations(runtime, monkeypatch):
    config, resources = runtime
    config = config.model_copy(update={'llm_provider': 'gemini', 'gemini_api_key': 'test-secret'})
    session = DemoSession()
    session.ingest('Policy.txt', b'Home office allowance is 750 dollars.', resources)
    def handler(request):
        assert request.headers['x-goog-api-key'] == 'test-secret'
        assert 'test-secret' not in str(request.url)
        body = json.loads(request.content)
        assert 'DOCUMENT EXCERPTS' in body['contents'][-1]['parts'][0]['text']
        assert '750 dollars' in body['contents'][-1]['parts'][0]['text']
        assert body['systemInstruction']['parts']
        return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'The home office allowance is 750 dollars. [1]'}]}}]})
    mock_gemini(monkeypatch, handler)
    result = session.ask('What is the home office allowance?', resources, config)
    assert result['generation'] == 'llm' and result['citations'][0]['name'] == 'Policy.txt'


@pytest.mark.parametrize('status,payload', [(429, {}), (403, {}), (200, {'candidates': []})])
def test_gemini_failures_fall_back_to_cited_excerpts(runtime, monkeypatch, status, payload):
    config, resources = runtime
    config = config.model_copy(update={'llm_provider': 'gemini', 'gemini_api_key': 'test-secret'})
    session = DemoSession()
    session.ingest('Policy.txt', b'Home office allowance is 750 dollars.', resources)
    mock_gemini(monkeypatch, lambda request: httpx.Response(status, json=payload))
    result = session.ask('What is the home office allowance?', resources, config)
    assert result['generation'] == 'extractive' and result['warning']
    assert '750 dollars' in result['answer'] and result['citations']
    assert 'test-secret' not in result['warning']


def test_invalid_generated_citation_is_rejected(runtime, monkeypatch):
    config, resources = runtime
    config = config.model_copy(update={'llm_provider': 'gemini', 'gemini_api_key': 'test-secret'})
    session = DemoSession()
    session.ingest('Policy.txt', b'Home office allowance is 750 dollars.', resources)
    mock_gemini(monkeypatch, lambda request: httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'Allowance is 750 dollars. [99]'}]}}]}))
    result = session.ask('What is the home office allowance?', resources, config)
    assert result['generation'] == 'abstention' and not result['citations']


def test_expired_vectors_are_removed(runtime, monkeypatch):
    _, resources = runtime
    session = DemoSession()
    session.ingest('Policy.txt', b'Home office allowance is 750 dollars.', resources)
    client = resources.retriever.client
    collection = resources.retriever.collection
    assert client.count(collection).count == 1
    import app.retrieval as retrieval
    now = retrieval.time.time()
    monkeypatch.setattr(retrieval.time, 'time', lambda: now + 86401)
    resources.retriever.cleanup_expired()
    assert client.count(collection).count == 0


def test_secrets_pair_validation():
    with pytest.raises(ValueError, match='both QDRANT'):
        config_from_secrets({'QDRANT_URL': 'https://example.com'})


def test_streamlit_ui_sample_chat_and_sources():
    path = Path(__file__).resolve().parent.parent / 'streamlit_app' / 'app.py'
    ui = AppTest.from_file(str(path), default_timeout=30)
    ui.secrets['EMBEDDING_BACKEND'] = 'hash'
    ui.run()
    assert not ui.exception
    assert len(ui.session_state['demo'].documents) == 3
    next(button for button in ui.button if button.label == 'Home office allowance').click().run()
    assert not ui.exception
    assert any('$750' in element.value for element in ui.markdown)
    assert any('Inspect all' in element.label for element in ui.expander)
    assert ui.session_state['demo'].history[-1]['citations']

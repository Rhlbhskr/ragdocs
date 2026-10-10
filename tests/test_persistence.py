import pytest

from app.retrieval import Retriever
from streamlit_app.engine import Resources, config_from_secrets
from streamlit_app.persistence import WorkspaceStore, PersistentSession, authorize, identity_workspace


def test_identity_and_authorization():
    claims = {'iss': 'https://issuer', 'sub': '123', 'email': 'owner@example.com', 'email_verified': True}
    assert authorize(claims, {'ALLOWED_USERS': ['owner@example.com']}) == identity_workspace(claims)
    google = dict(claims, iss='https://accounts.google.com')
    public = {'ALLOW_ALL_AUTHENTICATED': True, 'AUTH_ISSUER': 'https://accounts.google.com'}
    assert authorize(google, public) == identity_workspace(google)
    with pytest.raises(ValueError, match='identity provider'):
        authorize(claims, public)
    assert identity_workspace(claims) != identity_workspace(dict(claims, iss='https://other'))
    for values in ({}, {'ALLOW_ALL_AUTHENTICATED': 'false'}, {'ALLOWED_USERS': ['other@example.com']}):
        with pytest.raises(ValueError):
            authorize(claims, values)
    with pytest.raises(ValueError):
        authorize(dict(claims, email_verified=False), {'ALLOWED_USERS': ['owner@example.com']})


def test_production_rejects_temporary_configuration():
    with pytest.raises(ValueError, match='credentials'):
        config_from_secrets({'APP_MODE': 'production'})
    with pytest.raises(ValueError, match='PostgreSQL'):
        WorkspaceStore('sqlite:///:memory:')


def test_production_ui_stops_without_authentication():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    ui = AppTest.from_file(str(Path(__file__).resolve().parent.parent / 'streamlit_app/app.py'))
    ui.secrets['APP_MODE'] = 'production'
    ui.run()
    assert not ui.exception
    assert any('authentication' in error.value for error in ui.error)
    assert 'demo' not in ui.session_state


def test_durable_library_isolation_repair_and_deletion(tmp_path):
    config = config_from_secrets({'EMBEDDING_BACKEND': 'hash'})
    resources = Resources(Retriever(config))
    store = WorkspaceStore(f'sqlite:///{tmp_path / "state.db"}', production=False)
    first_id, second_id = 'a' * 64, 'b' * 64
    first = PersistentSession(first_id, store)
    doc = first.ingest('Policy.txt', b'Home office allowance is 750 dollars.', resources)
    restored = PersistentSession(first_id, store)
    assert restored.documents[0]['id'] == doc['id']
    # A second tab created before an upload reloads state before mutation.
    first.ingest('Leave.txt', b'Annual leave is 25 days.', resources)
    restored.refresh()
    assert len(restored.documents) == 2
    assert not PersistentSession(second_id, store).documents
    resources.retriever.delete(first_id, doc['id'])
    result = restored.ask('What is the home office allowance?', resources, config)
    assert result['citations'] and '750 dollars' in result['answer']
    assert PersistentSession(first_id, store).questions_used == 1
    restored.clear_history()
    assert not PersistentSession(first_id, store).history
    assert PersistentSession(first_id, store).questions_used == 1
    with pytest.raises(ValueError, match='not found'):
        PersistentSession(second_id, store).remove(doc['id'], resources)
    restored.remove(doc['id'], resources)
    assert all(d['id'] != doc['id'] for d in PersistentSession(first_id, store).documents)
    resources.retriever.client.close()
    store.engine.dispose()


def test_daily_quota_survives_reconnect_and_resets(tmp_path):
    store = WorkspaceStore(f'sqlite:///{tmp_path / "state.db"}', production=False)
    identifier = 'd' * 64
    with store.transaction(identifier) as session:
        session.questions_used = 10
    restored = PersistentSession(identifier, store)
    with pytest.raises(ValueError, match='Daily question'):
        restored.ask('Question?', None, None)
    from sqlalchemy.orm import Session
    from streamlit_app.persistence import Workspace
    with Session(store.engine) as db, db.begin():
        row = db.get(Workspace, identifier)
        row.payload = dict(row.payload, quota_day='2000-01-01')
    assert PersistentSession(identifier, store).questions_used == 0
    store.engine.dispose()


def test_failed_question_still_consumes_persistent_quota(tmp_path, monkeypatch):
    store = WorkspaceStore(f'sqlite:///{tmp_path / "state.db"}', production=False)
    session = PersistentSession('c' * 64, store)
    config = config_from_secrets({'EMBEDDING_BACKEND': 'hash'})
    resources = Resources(Retriever(config))
    def fail(*args, **kwargs):
        raise RuntimeError('unavailable')
    monkeypatch.setattr(resources.retriever, 'search', fail)
    with pytest.raises(RuntimeError):
        session.ask('What is the policy?', resources, config)
    assert PersistentSession(session.workspace, store).questions_used == 1
    resources.retriever.client.close()
    store.engine.dispose()

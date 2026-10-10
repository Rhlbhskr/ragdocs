"""Durable private workspaces, serialized across app instances.

SQL is the source of truth; vectors are a recoverable index. Never accept a
workspace identifier from query parameters or browser input.
"""
import hashlib
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import RLock
from time import time

from sqlalchemy import Column, JSON, String, create_engine, text
from sqlalchemy.orm import declarative_base, Session

from streamlit_app.engine import DemoSession

Base = declarative_base()


class Workspace(Base):
    __tablename__ = 'streamlit_workspaces'
    id = Column(String(64), primary_key=True)
    payload = Column(JSON, nullable=False)


def identity_workspace(claims):
    issuer, subject = claims.get('iss'), claims.get('sub')
    if not issuer or not subject:
        raise ValueError('The identity provider must supply issuer and subject claims.')
    return hashlib.sha256(f'{issuer}\0{subject}'.encode()).hexdigest()


def authorize(claims, values):
    workspace = identity_workspace(claims)
    issuer = values.get('AUTH_ISSUER')
    if issuer and claims.get('iss') != issuer:
        raise ValueError('Please sign in with the configured identity provider.')
    allowed = values.get('ALLOWED_USERS', [])
    if not isinstance(allowed, list):
        raise ValueError('ALLOWED_USERS must be a TOML array of email addresses.')
    if not allowed and values.get('ALLOW_ALL_AUTHENTICATED', False) is not True:
        raise ValueError('Configure ALLOWED_USERS or explicitly allow all authenticated users.')
    if allowed:
        if claims.get('email_verified') is not True or str(claims.get('email', '')).lower() not in {str(v).lower() for v in allowed}:
            raise ValueError('This account is not authorized to access this application.')
    return workspace


class WorkspaceStore:
    def __init__(self, url, production=True):
        for prefix in ('postgres://', 'postgresql://'):
            if url.startswith(prefix):
                url = 'postgresql+psycopg://' + url[len(prefix):]
        if production and not url.startswith('postgresql+psycopg://'):
            raise ValueError('Production requires an external PostgreSQL DATABASE_URL.')
        options = {'pool_pre_ping': True}
        if url.startswith('sqlite'):
            options['connect_args'] = {'check_same_thread': False}
        else:
            options.update(pool_size=3, max_overflow=2, pool_timeout=15,
                           connect_args={'connect_timeout': 10, 'prepare_threshold': None})
        self.engine = create_engine(url, **options)
        self.lock = RLock()
        Base.metadata.create_all(self.engine)

    @contextmanager
    def transaction(self, identifier):
        with self.lock, Session(self.engine) as db, db.begin():
            if self.engine.dialect.name == 'postgresql':
                key = int.from_bytes(bytes.fromhex(identifier)[:8], 'big', signed=True)
                db.execute(text("SET LOCAL lock_timeout = '15s'"))
                db.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': key})
            row = db.get(Workspace, identifier)
            if row is None:
                row = Workspace(id=identifier, payload={})
                db.add(row)
            state = deepcopy(row.payload)
            session = DemoSession(workspace=identifier, documents=state.get('documents', []),
                                  history=state.get('history', []))
            day = datetime.now(timezone.utc).date().isoformat()
            used = state.get('questions_used', 0) if state.get('quota_day') == day else 0
            session.questions_used = used
            session.reserved_at = state.get('reserved_at', 0)
            yield session
            row.payload = {'documents': session.documents, 'history': session.history[-50:],
                           'questions_used': session.questions_used, 'quota_day': day,
                           'reserved_at': session.reserved_at}


class PersistentSession:
    """Each operation reloads authoritative state to avoid lost writes from tabs."""
    def __init__(self, workspace, store):
        self.workspace, self.store = workspace, store
        self.refresh()

    def refresh(self):
        with self.store.transaction(self.workspace) as session:
            self.documents, self.history = session.documents, session.history
            self.questions_used = session.questions_used
            self.uploads_used = sum(not d.get('sample', False) for d in session.documents)

    def ingest(self, name, content, resources, chunk_size=512):
        with self.store.transaction(self.workspace) as session:
            session.uploads_used = sum(not d.get('sample', False) for d in session.documents)
            result = session.ingest(name, content, resources, chunk_size)
        self.refresh()
        return result

    def ask(self, question, resources, config, **kwargs):
        # Reserve quota in a separate committed transaction, including failures.
        with self.store.transaction(self.workspace) as session:
            if time() - session.reserved_at < 1:
                raise ValueError('Please wait a moment before asking another question.')
            if session.questions_used >= 10:
                raise ValueError('Daily question limit reached. Try again tomorrow (UTC).')
            session.questions_used += 1
            session.reserved_at = time()
        with self.store.transaction(self.workspace) as session:
            session.questions_used -= 1  # ask increments the already reserved slot
            with resources.lock:
                # Repair missing vectors / changed embedding collections from SQL.
                for doc in session.documents:
                    resources.retriever.index(self.workspace, doc['id'], doc['passages'])
            result = session.ask(question, resources, config, **kwargs)
        self.refresh()
        return result

    def remove(self, identifier, resources):
        with self.store.transaction(self.workspace) as session:
            if not any(d['id'] == identifier for d in session.documents):
                raise ValueError('Document not found in your workspace.')
            session.remove(identifier, resources)
        self.refresh()

    def clear_history(self):
        with self.store.transaction(self.workspace) as session:
            session.history.clear()
        self.refresh()

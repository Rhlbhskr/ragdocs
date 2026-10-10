from pathlib import Path
from contextlib import contextmanager
import hashlib
from threading import RLock
from sqlalchemy import JSON, Column, String, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import declarative_base, sessionmaker
from .config import settings

def normalize_database_url(value):
    # Neon and other hosts commonly supply postgres:// or postgresql:// URLs.
    for prefix in ('postgres://', 'postgresql://'):
        if value.startswith(prefix):
            return 'postgresql+psycopg://' + value[len(prefix):]
    return value


database_url = normalize_database_url(settings.database_url)
if database_url.startswith('sqlite') and not settings.vercel:
    filename = make_url(database_url).database
    if filename and filename != ':memory:':
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
engine_options = {'pool_pre_ping': True}
if database_url.startswith('sqlite'):
    engine_options['connect_args'] = {'check_same_thread': False}
elif settings.vercel:
    # Use the database provider's transaction pooler rather than keep idle
    # connections in every serverless instance.
    engine_options['poolclass'] = NullPool
    engine_options['connect_args'] = {'connect_timeout': 10, 'prepare_threshold': None}
engine = create_engine(database_url, **engine_options)
Session = sessionmaker(bind=engine)
Base = declarative_base()
local_lock = RLock()


@contextmanager
def workspace_lock(workspace):
    """Serialize mutations across serverless instances, using PostgreSQL."""
    if engine.dialect.name == 'postgresql':
        key = int.from_bytes(hashlib.sha256(workspace.encode()).digest()[:8], 'big', signed=True)
        with engine.begin() as connection:
            connection.execute(text("SET LOCAL lock_timeout = '30s'"))
            connection.execute(text('SELECT pg_advisory_xact_lock(:key)'), {'key': key})
            yield
    else:
        with local_lock:
            yield


class Record(Base):
    __tablename__ = 'records'
    id = Column(String, primary_key=True)
    workspace = Column(String, index=True, nullable=False)
    kind = Column(String, index=True, nullable=False)
    payload = Column(JSON, nullable=False)


def records(workspace, kind):
    with Session() as db:
        return [dict(r.payload, id=r.id) for r in db.query(Record).filter_by(workspace=workspace, kind=kind).all()]


def save(workspace, kind, identifier, payload):
    with Session.begin() as db:
        db.add(Record(id=identifier, workspace=workspace, kind=kind, payload=payload))

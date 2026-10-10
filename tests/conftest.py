import os
import tempfile

TEST_DATA = tempfile.TemporaryDirectory(prefix='groundeddocs-test-')
os.environ['DATABASE_URL'] = 'sqlite:///' + TEST_DATA.name.replace('\\', '/') + '/test.db'
os.environ['QDRANT_PATH'] = TEST_DATA.name + '/qdrant'
os.environ['EMBEDDING_BACKEND'] = 'hash'
os.environ['ENABLE_RERANKER'] = 'false'
os.environ['LLM_BASE_URL'] = ''
os.environ['VERCEL'] = '0'
os.environ['QDRANT_URL'] = ''


def pytest_sessionfinish(session, exitstatus):
    # Later storage tests can reopen pooled SQLite connections after API teardown.
    # Release them before Windows attempts to remove the temporary database.
    from app.storage import engine
    engine.dispose()
    TEST_DATA.cleanup()

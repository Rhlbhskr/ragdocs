"""Community Cloud entrypoint: deploy this file, not the FastAPI entrypoint."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Streamlit adds the entrypoint directory first; its app.py must not shadow
# the repository's app package when launched with `streamlit run`.
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from app.retrieval import Retriever
from streamlit_app.engine import config_from_secrets, DemoSession, Resources, MAX_QUESTIONS, MAX_UPLOADS
from streamlit_app.persistence import WorkspaceStore, PersistentSession, authorize

st.set_page_config(page_title='GroundedDocs — Answers with evidence', page_icon='📚', layout='wide')


@st.cache_resource(show_spinner='Loading the small embedding model and connecting the vector store…')
def get_resources(config_values):
    config = config_from_secrets(config_values)
    if config.embedding_backend == 'sentence-transformers':
        import torch
        torch.set_num_threads(1)
    return Resources(Retriever(config=config))


def render_turn(turn):
    with st.chat_message('user'):
        st.write(turn['question'])
    with st.chat_message('assistant'):
        if turn.get('warning'):
            st.warning(turn['warning'])
        st.write(turn['answer'])
        ground = turn['grounding']
        st.caption(f"Evidence confidence: {ground['confidence']} · {turn['latency_ms']} ms · {turn['mode']} retrieval · {turn['generation']}")
        if not ground['supported']:
            st.warning('Low evidence found for this answer.')
        st.caption(ground['note'])
        if turn['citations']:
            st.markdown('**Sources**')
            for source in turn['citations']:
                location = f"Page {source['page']} · " if source['page'] else ''
                with st.expander(f"[{source['citation']}] {source['name']} — {location}Chunk {source['number']}"):
                    st.text(source['text'])
        if turn['passages']:
            with st.expander(f"Inspect all {len(turn['passages'])} retrieved passages"):
                for index, source in enumerate(turn['passages'], 1):
                    location = f"Page {source['page']} · " if source['page'] else ''
                    st.markdown(f"**[{index}] {source['name']} · {location}Chunk {source['number']}**")
                    st.text(source['text'])


try:
    secrets = dict(st.secrets)
except (FileNotFoundError, StreamlitSecretNotFoundError):
    secrets = {}

# Only accepted fields affect this app; API/Vercel environment settings are ignored.
accepted = ('GEMINI_API_KEY', 'GEMINI_MODEL', 'QDRANT_URL', 'QDRANT_API_KEY', 'EMBEDDING_BACKEND', 'ENABLE_RERANKER', 'APP_MODE')
values = {key: secrets[key] for key in accepted if key in secrets}
production = str(values.get('APP_MODE', 'demo')).lower() == 'production'
st.title('GroundedDocs')
st.caption('Upload. Ask. Verify. Answers grounded in your documents, with source passages.')

@st.cache_resource
def get_store(url):
    return WorkspaceStore(url)

if production:
    if 'auth' not in secrets:
        st.error('Production requires OpenID Connect authentication in Streamlit Secrets.')
        st.stop()
    if not st.user.is_logged_in:
        st.write('Sign in to access your private document library.')
        if st.button('Sign in with Google'):
            st.login()
        st.stop()
    if st.sidebar.button('Sign out'):
        st.session_state.clear()
        st.logout()
        st.stop()
    try:
        workspace = authorize(dict(st.user), secrets)
        store = get_store(str(secrets.get('DATABASE_URL', '')))
        # Reload on every rerun; no private data goes in a shared cache.
        st.session_state.demo = PersistentSession(workspace, store)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        st.error('Your document library is temporarily unavailable. Please try again later.')
        st.stop()


try:
    config = config_from_secrets(values)
    resources = get_resources(values)
except Exception:
    import logging

    logging.exception(
        "GroundedDocs embedding/Qdrant initialization failed"
    )
    st.error(
        "The embedding model or Qdrant connection could not initialize. "
        "Check the app logs for the underlying error."
    )
    st.stop()


if 'demo' not in st.session_state:
    st.session_state.demo = DemoSession()
demo = st.session_state.demo
if not production and 'seeded' not in st.session_state:
    try:
        with st.spinner('Preparing the three sample documents…'):
            demo.seed_samples(resources)
        st.session_state.seeded = True
    except Exception:
        st.error('The sample documents could not be indexed. Check the Qdrant connection and try again.')
        st.stop()

with st.sidebar:
    st.header('Your document library')
    st.caption('Private, saved workspace' if production else 'Browser-session workspace · no visitor registration')
    if not config.qdrant_url:
        st.info('Local demo vector store is active. Add QDRANT_URL and QDRANT_API_KEY to use Qdrant Cloud.')
    if config.embedding_backend == 'hash':
        st.caption('Offline hash embeddings are active. Enable sentence-transformers for semantic retrieval.')
    chunk_size = st.selectbox('Chunk size (words)', (256, 512, 1024), index=1)
    mode = st.selectbox('Retrieval method', ('hybrid', 'dense', 'keyword'))
    rerank = st.checkbox('CrossEncoder reranking', value=config.enable_reranker, disabled=not config.enable_reranker)
    generated = st.checkbox('Generate answers with Gemini', value=bool(config.gemini_api_key), disabled=not config.gemini_api_key)
    if not config.gemini_api_key:
        st.caption('Exact source excerpts are active. Add GEMINI_API_KEY in app settings → Secrets to enable generated answers.')
    st.caption(f'Uploads: {demo.uploads_used}/{MAX_UPLOADS} · Questions: {demo.questions_used}/{MAX_QUESTIONS}')
    for kind, message in st.session_state.pop('upload_messages', []):
        getattr(st, kind)(message)
    files = st.file_uploader('Upload PDF, DOCX, or TXT', type=['pdf', 'docx', 'txt'], accept_multiple_files=True, help='10 MB, 50 PDF pages, and 200 KB extracted text per file.')
    if st.button('Index selected documents', disabled=not files or demo.uploads_used >= MAX_UPLOADS, use_container_width=True):
        notices = []
        for upload in files or []:
            try:
                with st.spinner(f'Indexing {upload.name}…'):
                    demo.ingest(upload.name, upload.getvalue(), resources, chunk_size)
                notices.append(('success', f'{upload.name} is ready.'))
            except ValueError as exc:
                notices.append(('error', str(exc)))
            except Exception:
                notices.append(('error', 'This file could not be parsed or indexed. Verify the document and Qdrant connection.'))
        st.session_state.upload_messages = notices
        st.rerun()
    for doc in list(demo.documents):
        with st.expander(f"{doc['name']} · {len(doc['passages'])} chunks"):
            st.caption('Fictional sample document' if doc['sample'] else 'Your uploaded document')
            for passage in doc['passages']:
                location = f"Page {passage['page']} · " if passage['page'] else ''
                st.caption(f"{location}Chunk {passage['number']}")
                st.text(passage['text'])
            if st.button('Delete document', key=f"delete-{doc['id']}"):
                try:
                    demo.remove(doc['id'], resources)
                    st.rerun()
                except Exception:
                    st.error('Could not delete the document from the vector store. Try again.')
    if st.button('New conversation', use_container_width=True):
        if production:
            try:
                demo.clear_history()
            except Exception:
                st.error('Could not clear your saved conversation. Please try again later.')
                st.stop()
        else:
            demo.history.clear()
        st.rerun()
    st.caption('Documents and the last 50 chat turns are saved until deleted. Question limits reset daily (UTC).' if production else 'Conversation and source text last for this browser session. Refreshing/reconnecting may reset the session. Quotas do not reset when clearing chat.')
    st.caption('Generated answers send relevant excerpts and recent chat to Gemini. Embeddings are stored in Qdrant. Original uploaded files are not retained.')

if not demo.history:
    st.subheader('A good answer starts with a trusted source.')
    st.write('Upload documents in the sidebar to get started.' if production else 'Three fictional sample documents are ready. Ask about them or upload your own.')
    suggestions = [('Home office allowance', 'What is the home office equipment allowance?'),
                   ('Annual leave', 'How many days of annual leave do employees receive?'),
                   ('Production backups', 'How often are production database backups taken?')]
    for column, (label, question) in zip(st.columns(3), [] if production else suggestions):
        if column.button(label, use_container_width=True, disabled=demo.questions_used >= MAX_QUESTIONS):
            st.session_state.pending_question = question

for turn in demo.history:
    render_turn(turn)

question = st.chat_input('Ask a question about your documents…', disabled=demo.questions_used >= MAX_QUESTIONS)
question = question or st.session_state.pop('pending_question', None)
if question:
    try:
        with st.spinner('Retrieving and grounding the answer…'):
            demo.ask(question, resources, config, mode=mode, rerank=rerank, generated=generated)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error('Retrieval is temporarily unavailable. Check the Qdrant service and try again later.')

if demo.questions_used >= MAX_QUESTIONS:
    st.info('This browser session has used its ten demo questions. You can still inspect every source and retrieved passage above.')

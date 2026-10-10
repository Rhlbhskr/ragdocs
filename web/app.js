const $ = (id) => document.getElementById(id);
let busy = false;
let turns = [];
let currentEvidence = null;
let maxUploadMb = 4;

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { const data = await response.json(); message = typeof data.detail === 'string' ? data.detail : 'Please check your input.'; } catch {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function error(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
function locationLabel(p) { return `${p.page ? `Page ${p.page} · ` : ''}Chunk ${p.number}`; }
function setBusy(value) {
  busy = value;
  $('send').disabled = value;
  $('new-chat').disabled = value;
  $('upload').disabled = value;
  document.querySelectorAll('.delete-doc').forEach(button => { button.disabled = value; });
}
async function loadDocuments() {
  const docs = await api('/documents');
  $('document-count').textContent = docs.length;
  $('documents').replaceChildren();
  for (const doc of docs) {
    const row = element('div', 'document');
    row.append(element('span', 'doc-icon', doc.name.split('.').pop().toUpperCase()));
    const info = element('div', 'doc-info');
    const open = element('button', '', doc.name);
    open.title = doc.name;
    open.addEventListener('click', () => showSource(doc.id));
    info.append(open, element('small', '', `${doc.chunks} chunk${doc.chunks === 1 ? '' : 's'} · Ready to search`));
    const remove = element('button', 'delete-doc', '×');
    remove.setAttribute('aria-label', `Delete ${doc.name}`);
    remove.title = 'Delete document and conversation history';
    remove.disabled = busy;
    remove.addEventListener('click', async () => {
      if (busy || !confirm(`Delete ${doc.name}? Conversation history will also be cleared to remove stored excerpts.`)) return;
      setBusy(true);
      try { await api(`/documents/${doc.id}`, { method: 'DELETE' }); resetConversation(); await loadDocuments(); }
      catch (e) { error(e.message); }
      finally { setBusy(false); }
    });
    row.append(info, remove); $('documents').append(row);
  }
}
async function uploadFiles(files) {
  if (busy) return;
  setBusy(true); error('');
  try {
    for (const file of files) {
      if (file.size > maxUploadMb * 1024 * 1024) throw new Error(`${file.name} exceeds ${maxUploadMb} MB.`);
      if (!/\.(pdf|docx|txt)$/i.test(file.name)) throw new Error('Choose a PDF, DOCX, or TXT file.');
      $('upload-status').textContent = `Indexing ${file.name}…`;
      const data = new FormData(); data.append('file', file); data.append('chunk_size', $('chunk-size').value);
      await api('/documents', { method: 'POST', body: data });
    }
    $('upload-status').textContent = 'Your documents are ready to search.';
  } catch (e) { $('upload-status').textContent = ''; error(e.message); }
  finally { try { await loadDocuments(); } catch (e) { error(e.message); } setBusy(false); $('upload').value = ''; }
}
async function showSource(id, chunkNumber) {
  try {
    const doc = await api(`/documents/${id}/source`);
    $('source-name').textContent = doc.name; $('source-content').replaceChildren();
    for (const p of doc.passages) {
      const section = element('section', 'source-chunk'); section.id = `source-${p.number}`;
      section.append(element('h3', '', locationLabel(p)), element('p', '', p.text));
      $('source-content').append(section);
    }
    if (!$('source-dialog').open) $('source-dialog').showModal();
    if (chunkNumber) $(`source-${chunkNumber}`)?.scrollIntoView({ block: 'center' });
  } catch (e) { error(e.message); }
}
function showEvidence(turn, citation) {
  currentEvidence = turn;
  $('evidence').replaceChildren(); $('evidence-count').textContent = turn.passages.length;
  $('grounding-card').hidden = false;
  const ground = element('div', `grounding ${turn.grounding.confidence === 'Low' ? 'low' : ''}`);
  ground.append(element('strong', '', `${turn.grounding.supported ? '◈' : '⚠'} Evidence confidence: ${turn.grounding.confidence}`), element('p', '', turn.grounding.supported ? 'Relevant evidence found in your documents.' : 'Low evidence found for this answer.'), element('small', '', turn.grounding.note));
  $('grounding-card').replaceChildren(ground);
  turn.passages.forEach((p, index) => {
    const number = index + 1;
    const card = element('article', `passage${citation === number ? ' highlight' : ''}`); card.id = `passage-${number}`;
    const head = element('div', 'passage-header'); head.append(element('span', 'passage-number', number), element('strong', '', p.name));
    const source = element('button', 'source-button', 'Open source ↗'); source.addEventListener('click', () => showSource(p.document_id, p.number));
    const excerpt = element('p', 'passage-text', p.text);
    card.append(head, element('div', 'passage-location', `${locationLabel(p)} · ${turn.reranked ? 'Reranked' : 'Retrieved'}`), excerpt, source);
    $('evidence').append(card);
  });
  if (citation) $(`passage-${citation}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
function renderAnswer(turn) {
  const article = element('article', 'assistant-message');
  const label = element('div', 'answer-label'); label.append(element('span', '', 'g'), document.createTextNode('GroundedDocs'));
  const text = element('div', 'answer-text');
  for (const part of turn.answer.split(/(\[\d+\])/g)) {
    if (/^\[\d+\]$/.test(part)) {
      const citation = Number(part.slice(1, -1));
      const button = element('button', 'citation-link', part); button.setAttribute('aria-label', `Show source ${citation}`);
      button.addEventListener('click', () => showEvidence(turn, citation)); text.append(button);
    } else text.append(document.createTextNode(part));
  }
  const meta = element('div', 'answer-meta');
  meta.append(element('span', '', `${turn.latency_ms} ms`), element('span', '', `${turn.mode} retrieval`), element('span', '', `${turn.generation === 'extractive' ? 'Exact excerpts' : turn.generation === 'llm' ? 'Generated answer' : 'Insufficient evidence'}`));
  const view = element('button', 'view-evidence', `Inspect ${turn.passages.length} passages ↗`); view.addEventListener('click', () => showEvidence(turn));
  article.append(label, text, meta, view); $('conversation').append(article);
}
const welcomeTemplate = $('welcome').cloneNode(true);
const emptyEvidence = $('evidence').cloneNode(true);
function bindSuggestions() {
  document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => { $('question').value = button.dataset.question; $('question-form').requestSubmit(); }));
}
function resetConversation() {
  turns = []; currentEvidence = null;
  $('conversation').replaceChildren(welcomeTemplate.cloneNode(true));
  $('evidence').replaceChildren(...Array.from(emptyEvidence.cloneNode(true).childNodes));
  $('grounding-card').hidden = true; $('evidence-count').textContent = '0'; bindSuggestions();
}
$('question-form').addEventListener('submit', async (event) => {
  event.preventDefault(); if (busy) return;
  const question = $('question').value.trim(); if (question.length < 2) { error('Please enter a question.'); return; }
  setBusy(true); error(''); $('welcome')?.remove();
  const user = element('div', 'user-message', question); $('conversation').append(user);
  const loading = element('div', 'loading', 'Finding the evidence in your documents…'); $('conversation').append(loading);
  $('question').value = ''; $('conversation').scrollTop = $('conversation').scrollHeight;
  try {
    const turn = await api('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, mode: $('mode').value, top_k: Number($('top-k').value), rerank: $('rerank').checked }) });
    turns.push(turn); renderAnswer(turn); showEvidence(turn);
  } catch (e) { user.remove(); $('question').value = question; error(e.message); }
  finally { loading.remove(); setBusy(false); $('conversation').scrollTop = $('conversation').scrollHeight; $('question').focus(); }
});
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('question-form').requestSubmit(); } });
$('upload').addEventListener('change', event => uploadFiles(event.target.files));
$('dropzone').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('upload').click(); } });
for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
for (const type of ['dragleave', 'drop']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); });
$('dropzone').addEventListener('drop', event => uploadFiles(event.dataTransfer.files));
$('close-source').addEventListener('click', () => $('source-dialog').close());
$('new-chat').addEventListener('click', async () => { if (busy) return; setBusy(true); try { await api('/history', { method: 'DELETE' }); resetConversation(); error(''); $('question').focus(); } catch (e) { error(e.message); } finally { setBusy(false); } });
async function initialize() {
  setBusy(true);
  try {
    await api('/workspace', { method: 'POST' });
    const health = await api('/health');
    maxUploadMb = health.max_upload_mb;
    $('upload-limit').textContent = `PDF, DOCX, TXT · up to ${maxUploadMb} MB`;
    $('rerank').checked = health.reranker; $('rerank').disabled = !health.reranker;
    $('rerank-note').textContent = health.reranker ? 'Reranking model is ready.' : 'Enable the reranker on the server to use this setting.';
    $('composer-mode').textContent = health.generation === 'extractive' ? 'Extractive demo · exact source passages' : 'Grounded in your library';
    if (health.embedding_backend === 'hash') $('rerank-note').textContent += ' Offline hash embeddings are active; semantic search requires the model backend.';
    await loadDocuments(); turns = await api('/history');
    if (turns.length) { $('welcome')?.remove(); for (const turn of turns) { $('conversation').append(element('div', 'user-message', turn.question)); renderAnswer(turn); } showEvidence(turns.at(-1)); $('conversation').scrollTop = $('conversation').scrollHeight; }
  } catch (e) { error(`Could not initialize the workspace: ${e.message}. Refresh to retry.`); }
  finally { setBusy(false); }
}
bindSuggestions(); initialize();

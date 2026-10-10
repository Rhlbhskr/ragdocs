"""Run the retrieval grid against a small labeled corpus, without modifying app data."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid
import statistics
import math

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='eval/results-offline.json')
    parser.add_argument('--models', action='store_true', help='Use sentence-transformers and CrossEncoder (downloads models).')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='groundeddocs-eval-') as temp:
        os.environ.update(QDRANT_URL='', QDRANT_PATH=temp, LLM_BASE_URL='', EMBEDDING_BACKEND='sentence-transformers' if args.models else 'hash', ENABLE_RERANKER='true' if args.models else 'false')
        from app.retrieval import Retriever, tokens
        from app.parsing import parse_document, chunk_pages
        from app.grounding import answer_question
        retriever = Retriever()
        questions = json.loads((ROOT / 'eval/questions.json').read_text())
        rows = []
        try:
            for size in (256, 512, 1024):
                ws = str(uuid.uuid4()); corpus = []
                for path in sorted((ROOT / 'samples').glob('*.txt')):
                    did = str(uuid.uuid4())
                    chunks = chunk_pages(parse_document(path.name, path.read_bytes()), size, min(64, size // 4))
                    for chunk in chunks:
                        chunk.update(id=str(uuid.uuid4()), document_id=did, name=path.name)
                    retriever.index(ws, did, chunks); corpus.extend(chunks)
                for mode, rerank, k in itertools.product(('dense', 'hybrid'), (False, True), (3, 5, 10)):
                    if rerank and not args.models:
                        rows.append({'chunk_size': size, 'mode': mode, 'rerank': True, 'top_k': k, 'status': 'skipped: model not enabled'})
                        continue
                    recalls, reciprocal, support, relevance, latency = [], [], [], [], []
                    for q in questions:
                        start = time.perf_counter()
                        passages = retriever.search(ws, q['question'], corpus, mode, k, rerank)
                        result = answer_question(q['question'], passages, [])
                        latency.append((time.perf_counter() - start) * 1000)
                        relevant = {c['id'] for c in corpus if c['name'] == q['document'] and q['evidence'].lower() in c['text'].lower()}
                        ranks = [i for i, c in enumerate(passages, 1) if c['id'] in relevant]
                        recalls.append(len(ranks) / len(relevant) if relevant else 0)
                        reciprocal.append(1 / min(ranks) if ranks else 0)
                        # These are explicitly proxies, not semantic judge scores.
                        answer_terms = set(tokens(result['answer']))
                        context_terms = set(tokens(' '.join(p['text'] for p in result['citations'])))
                        support.append(len(answer_terms & context_terms) / max(len(answer_terms), 1))
                        relevance.append(float(q['answer'].lower() in result['answer'].lower()))
                    rows.append({'chunk_size': size, 'mode': mode, 'rerank': rerank, 'top_k': k, 'status': 'measured', 'recall_at_k': statistics.mean(recalls), 'mrr': statistics.mean(reciprocal), 'faithfulness_token_overlap_proxy': statistics.mean(support), 'answer_expected_phrase_proxy': statistics.mean(relevance), 'mean_latency_ms': statistics.mean(latency), 'p95_latency_ms': sorted(latency)[math.ceil(.95 * len(latency)) - 1]})
        finally:
            retriever.client.close()
        output = {'embedding_backend': 'sentence-transformers' if args.models else 'hash', 'generation': 'extractive', 'questions': len(questions), 'limitations': 'Tiny synthetic corpus; many documents fit one chunk. Not a production benchmark. Faithfulness and relevance are lexical proxies. Latency excludes ingestion, model loading, and remote LLM calls.', 'results': rows}
        target = ROOT / args.output; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, indent=2), encoding='utf-8')
        print(f'Wrote {sum(r["status"] == "measured" for r in rows)} measured configurations to {target}')


if __name__ == '__main__':
    main()

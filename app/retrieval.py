import hashlib
import re
import time
import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from .config import settings
from .providers import embed_cohere, CohereReranker

STOP = set('a an the is are was were do does did what which how when where who of for to in on and or it its this that with can i my me you your please about'.split())


def tokens(text):
    return [x for x in re.findall(r'\w+', text.lower()) if x not in STOP]


class Retriever:
    def __init__(self, config=None):
        self.config = config or settings
        self.model = None
        self.reranker = None
        if self.config.embedding_backend == 'sentence-transformers':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.embedding_model, cache_folder=self.config.model_cache_dir)
        elif self.config.embedding_backend not in ('hash', 'cohere'):
            raise ValueError('Unknown embedding backend')
        if self.config.enable_reranker:
            if self.config.reranker_backend == 'cohere':
                self.reranker = CohereReranker()
            elif self.config.reranker_backend == 'cross-encoder':
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(self.config.reranker_model, cache_folder=self.config.model_cache_dir)
            else:
                raise ValueError('Unknown reranker backend')
        self.dimension = (self.model.get_sentence_embedding_dimension() if self.model else
                          self.config.cohere_embedding_dimension if self.config.embedding_backend == 'cohere' else 384)
        embedding_name = self.config.cohere_embedding_model if self.config.embedding_backend == 'cohere' else self.config.embedding_model
        identity = f'{self.config.embedding_backend}:{embedding_name}:{self.dimension}'
        self.collection = self.config.qdrant_collection_prefix + '_' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        self.client = (QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key or None, timeout=30)
                       if self.config.qdrant_url else QdrantClient(':memory:') if self.config.qdrant_in_memory else QdrantClient(path=self.config.qdrant_path))
        if not self.client.collection_exists(self.collection):
            try:
                self.client.create_collection(self.collection, vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE))
            except Exception:
                # Another serverless instance can create the same collection first.
                if not self.client.collection_exists(self.collection):
                    raise
        if self.config.qdrant_url:
            # Qdrant Cloud strict mode requires indexes for filtered fields.
            for key in ('workspace', 'document_id'):
                self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD, wait=True)
            if self.config.vector_ttl_seconds:
                self.client.create_payload_index(self.collection, 'expires_at', models.PayloadSchemaType.INTEGER, wait=True)

    def embed(self, texts, input_type='search_document'):
        if self.config.embedding_backend == 'cohere':
            return embed_cohere(texts, input_type)
        if self.model:
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension)
            for token in tokens(text):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], 'big') % self.dimension] += 1 if digest[4] % 2 else -1
            vector /= max(float(np.linalg.norm(vector)), 1)
            vectors.append(vector.tolist())
        return vectors

    def index(self, workspace, document_id, chunks):
        vectors = self.embed([c['text'] for c in chunks])
        points = [models.PointStruct(id=c['id'], vector=v, payload={'workspace': workspace, 'document_id': document_id}) for c, v in zip(chunks, vectors)]
        if self.config.vector_ttl_seconds:
            expiry = int(time.time()) + self.config.vector_ttl_seconds
            for point in points:
                point.payload['expires_at'] = expiry
        for start in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[start:start + 128], wait=True)

    def cleanup_expired(self):
        if self.config.vector_ttl_seconds:
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key='expires_at', range=models.Range(lt=int(time.time())))
            ])), wait=True)

    def delete(self, workspace, document_id, collection=None):
        collection = collection or self.collection
        if self.client.collection_exists(collection):
            self.client.delete(collection, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key='workspace', match=models.MatchValue(value=workspace)), models.FieldCondition(key='document_id', match=models.MatchValue(value=document_id))])), wait=True)

    def search(self, workspace, query, chunks, mode='hybrid', top_k=5, rerank=True):
        if not chunks:
            return []
        by_id = {c['id']: c for c in chunks}
        candidates = min(max(top_k * 4, 20), len(chunks))
        lists = []
        if mode in ('dense', 'hybrid'):
            hits = self.client.query_points(self.collection, query=self.embed([query], 'search_query')[0], query_filter=models.Filter(must=[models.FieldCondition(key='workspace', match=models.MatchValue(value=workspace))]), limit=candidates).points
            lists.append([str(h.id) for h in hits if str(h.id) in by_id])
        if mode in ('keyword', 'hybrid'):
            corpus = [tokens(c['text']) or ['_empty_'] for c in chunks]
            scores = BM25Okapi(corpus).get_scores(tokens(query))
            lists.append([chunks[i]['id'] for i in sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:candidates]])
        fusion = {}
        for ranking in lists:
            for rank, cid in enumerate(ranking, 1):
                fusion[cid] = fusion.get(cid, 0) + 1 / (60 + rank)
        ranked = [dict(by_id[cid], score=score) for cid, score in sorted(fusion.items(), key=lambda p: p[1], reverse=True)[:candidates]]
        if rerank and self.reranker and ranked:
            if isinstance(self.reranker, CohereReranker):
                return self.reranker.rank(query, ranked)[:top_k]
            scores = self.reranker.predict([(query, c['text']) for c in ranked])
            for c, score in zip(ranked, scores):
                c['rerank_score'] = float(score)
            ranked.sort(key=lambda c: c['rerank_score'], reverse=True)
        return ranked[:top_k]

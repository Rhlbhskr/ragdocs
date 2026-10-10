"""Small HTTP adapters: no PyTorch or model downloads in serverless functions."""
import math
import httpx
from .config import settings


class ProviderError(RuntimeError):
    pass


def cohere_request(endpoint, payload):
    try:
        with httpx.Client(timeout=45) as client:
            response = client.post(
                f'https://api.cohere.com/v2/{endpoint}',
                headers={'Authorization': f'Bearer {settings.cohere_api_key}'},
                json=payload,
            )
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as exc:
        # Never expose provider bodies, credentials, or submitted document text.
        raise ProviderError('The embedding or reranking provider is unavailable. Check its configuration and quota.') from exc


def embed_cohere(texts, input_type):
    vectors = []
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        data = cohere_request('embed', {
            'model': settings.cohere_embedding_model,
            'texts': batch,
            'input_type': input_type,
            'embedding_types': ['float'],
            'output_dimension': settings.cohere_embedding_dimension,
            'truncate': 'END',
        })
        try:
            result = data['embeddings']['float']
            if len(result) != len(batch) or any(
                len(vector) != settings.cohere_embedding_dimension or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
                for vector in result
            ):
                raise ValueError('Invalid vectors')
            vectors.extend(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError('The embedding provider returned an invalid response.') from exc
    return vectors


class CohereReranker:
    def rank(self, query, passages):
        data = cohere_request('rerank', {
            'model': settings.cohere_reranker_model,
            'query': query,
            'documents': [p['text'] for p in passages],
            'top_n': len(passages),
        })
        try:
            results = data['results']
            indices = [r['index'] for r in results]
            if sorted(indices) != list(range(len(passages))):
                raise ValueError('Invalid result indices')
            ranked = []
            for result in results:
                score = float(result['relevance_score'])
                if not math.isfinite(score):
                    raise ValueError('Invalid score')
                ranked.append(dict(passages[result['index']], rerank_score=score))
            return sorted(ranked, key=lambda p: p['rerank_score'], reverse=True)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ProviderError('The reranking provider returned an invalid response.') from exc

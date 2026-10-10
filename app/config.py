from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    vercel: bool = False
    database_url: str = 'sqlite:///./data/grounded.db'
    qdrant_url: str = ''
    qdrant_api_key: str = ''
    qdrant_path: str = './data/qdrant'
    qdrant_in_memory: bool = False
    qdrant_collection_prefix: str = 'chunks'
    vector_ttl_seconds: int = 0
    embedding_backend: str = 'hash'
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    model_cache_dir: str = './data/models/hub'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L6-v2'
    reranker_backend: str = 'cross-encoder'
    cohere_api_key: str = ''
    cohere_embedding_model: str = 'embed-v4.0'
    cohere_embedding_dimension: Literal[256, 512, 1024, 1536] = 1024
    cohere_reranker_model: str = 'rerank-v3.5'
    enable_reranker: bool = False
    llm_base_url: str = ''
    llm_api_key: str = ''
    llm_model: str = ''
    llm_provider: Literal['compatible', 'gemini', 'extractive'] = 'compatible'
    gemini_api_key: str = ''
    gemini_model: str = 'gemini-3.5-flash-lite'
    cookie_secure: bool = False
    max_upload_mb: int = Field(default=15, ge=1, le=15)
    max_chunks: int = Field(default=2000, ge=1, le=2000)
    max_workspaces: int = 200

    @model_validator(mode='after')
    def serverless_limits(self):
        if self.vercel:
            # Leave room for multipart framing beneath Vercel's 4.5 MB limit.
            self.max_upload_mb = min(self.max_upload_mb, 4)
            self.max_chunks = min(self.max_chunks, 200)
            self.cookie_secure = True
        return self

    def validate_runtime(self):
        """Run at runtime, not during Vercel's build-time app discovery."""
        if self.vercel:
            missing = []
            if not self.database_url.startswith(('postgresql://', 'postgres://', 'postgresql+psycopg://')):
                missing.append('DATABASE_URL (hosted PostgreSQL)')
            if not self.qdrant_url.startswith('https://'):
                missing.append('QDRANT_URL (HTTPS Qdrant service)')
            if not self.qdrant_api_key:
                missing.append('QDRANT_API_KEY')
            if self.embedding_backend == 'sentence-transformers':
                missing.append('EMBEDDING_BACKEND=cohere or hash (local models are not bundled)')
            if self.enable_reranker and self.reranker_backend != 'cohere':
                missing.append('RERANKER_BACKEND=cohere (local reranker is not bundled)')
            if missing:
                raise ValueError('Vercel configuration required: ' + ', '.join(missing))
        if (self.embedding_backend == 'cohere' or (self.enable_reranker and self.reranker_backend == 'cohere')) and not self.cohere_api_key:
            raise ValueError('COHERE_API_KEY is required for hosted embeddings or reranking.')
        if self.llm_base_url and (not self.llm_api_key or not self.llm_model):
            raise ValueError('Set LLM_API_KEY and LLM_MODEL when LLM_BASE_URL is configured.')
        if self.llm_provider == 'gemini' and not self.gemini_api_key:
            raise ValueError('GEMINI_API_KEY is required for Gemini answers.')


settings = Settings()

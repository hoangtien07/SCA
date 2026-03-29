"""
config/settings.py
Central configuration — loaded once at startup.
All modules import from here; never import os.environ directly.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── AI APIs ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Vector DB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/knowledge_base"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "skincare_papers"

    # ── Database ─────────────────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""

    # ── Models ───────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    reasoning_model: str = "claude-sonnet-4-6-20251101"
    vision_model: str = "gpt-4o"

    # ── Pipeline ─────────────────────────────────────────────────────────────
    max_papers_per_query: int = 200
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 8

    # ── LangSmith Tracing ─────────────────────────────────────────────────────
    langsmith_api_key: str = ""
    langsmith_project: str = "skincare-ai"
    langsmith_tracing: bool = False

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: str = "local"          # "local" | "openai"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Semantic Cache ────────────────────────────────────────────────────────
    cache_enabled: bool = False
    cache_retrieval_ttl: int = 86400
    cache_regimen_ttl: int = 43200
    cache_similarity_threshold: float = 0.92

    # ── CV Datasets ───────────────────────────────────────────────────────────
    cv_datasets_dir: str = "./data/cv_datasets"
    vision_eval_sample_size: int = 100

    # ── XAI ──────────────────────────────────────────────────────────────────
    xai_enabled: bool = False
    xai_num_samples: int = 500
    xai_surrogate_model: str = "mobilenet_v2"

    # ── App ──────────────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8501,http://localhost:3000"

    # ── Paths (computed) ─────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent
    @property
    def data_raw_dir(self) -> Path:
        return Path("./data/raw")

    @property
    def data_processed_dir(self) -> Path:
        return Path("./data/processed")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()

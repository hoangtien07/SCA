"""
src/pipeline/embedder.py

Pluggable embedding layer supporting local PubMedBERT and OpenAI models.

Architecture:
    BaseEmbedder (ABC)
        ├── LocalEmbedder  — NeuML/pubmedbert-base-embeddings via sentence-transformers
        └── OpenAIEmbedder — text-embedding-3-small with tenacity retry

Factory:
    get_embedder(settings) -> BaseEmbedder

Usage:
    from config.settings import settings
    from src.pipeline.embedder import get_embedder

    embedder = get_embedder(settings)
    vectors = embedder.embed(["niacinamide for acne", "retinol aging"])
    query_vec = embedder.embed_query("best ingredients for oily skin")
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from config.settings import Settings


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Abstract base class for all embedding implementations."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed

        Returns:
            List of float vectors (one per text)
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text (may use different pooling/prefix).

        Args:
            text: Query string

        Returns:
            Float vector
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimensionality."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier (stored in collection metadata)."""
        ...


# ── Local embedder (PubMedBERT) ───────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Local embedding using NeuML/pubmedbert-base-embeddings.

    Features:
    - Zero API cost
    - Auto-detects device: CUDA > MPS (Apple Silicon) > CPU
    - Batch processing with tqdm progress bar
    - L2-normalized output (cosine similarity via dot product)

    First run downloads ~440MB model to ~/.cache/huggingface/
    """

    MODEL_ID = "NeuML/pubmedbert-base-embeddings"
    _DIMENSION = 768  # PubMedBERT hidden size

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer
        import torch

        # Auto-detect best device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        self._batch_size = batch_size

        logger.info(f"[LocalEmbedder] Loading {self.MODEL_ID} on {device}...")
        self._model = SentenceTransformer(self.MODEL_ID, device=device)
        logger.info(f"[LocalEmbedder] Model ready. Dim={self._DIMENSION}, device={device}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches with L2 normalization."""
        from tqdm import tqdm
        import torch

        all_vecs: list[np.ndarray] = []

        for i in tqdm(range(0, len(texts), self._batch_size),
                      desc="Embedding", leave=False, disable=len(texts) < 10):
            batch = texts[i: i + self._batch_size]
            vecs = self._model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize
            )
            all_vecs.extend(vecs)

        return [v.tolist() for v in all_vecs]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vec = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec[0].tolist()

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    @property
    def model_name(self) -> str:
        return self.MODEL_ID


# ── OpenAI embedder ───────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding using text-embedding-3-small with tenacity retry.

    Costs ~$0.02/1M tokens. Falls back gracefully on rate limits.
    """

    MODEL_ID = "text-embedding-3-small"
    _DIMENSION = 1536

    def __init__(self, api_key: str, model: str = MODEL_ID):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"[OpenAIEmbedder] Using model {model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API with retry on rate limits."""
        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
        from openai import RateLimitError, APIError

        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type((RateLimitError, APIError)),
        )
        def _call(batch: list[str]) -> list[list[float]]:
            resp = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            return [r.embedding for r in resp.data]

        all_vecs: list[list[float]] = []
        batch_size = 100  # OpenAI allows up to 2048 inputs but 100 is safe

        from tqdm import tqdm
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="OpenAI embedding", leave=False, disable=len(texts) < 10):
            batch = texts[i: i + batch_size]
            vecs = _call(batch)
            all_vecs.extend(vecs)

        return all_vecs

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        resp = self._client.embeddings.create(
            model=self._model,
            input=[text],
        )
        return resp.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    @property
    def model_name(self) -> str:
        return self._model


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder(settings: "Settings") -> BaseEmbedder:
    """
    Factory function: create embedder based on settings.

    Args:
        settings: Settings object with embedding_provider, embedding_device,
                  embedding_batch_size, openai_api_key

    Returns:
        BaseEmbedder instance (LocalEmbedder or OpenAIEmbedder)
    """
    provider = getattr(settings, "embedding_provider", "local")

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "embedding_provider='openai' requires OPENAI_API_KEY to be set"
            )
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    elif provider == "local":
        device = getattr(settings, "embedding_device", "auto")
        batch_size = getattr(settings, "embedding_batch_size", 32)
        return LocalEmbedder(device=device, batch_size=batch_size)
    else:
        raise ValueError(
            f"Unknown embedding_provider: '{provider}'. "
            "Valid values: 'local', 'openai'"
        )

"""Local sentence-transformer embeddings for documents and queries."""
import os
from typing import Protocol

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        """Return one query vector per question, in the same order."""
        ...


class MyEmbedder:
    """Encode text locally with a multilingual SentenceTransformer model."""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _BATCH_SIZE = 32

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        load_dotenv()
        if model_name is None:
            selected_model = os.getenv("SENTENCE_TRANSFORMER_MODEL", "").strip()
            selected_model = selected_model or self.DEFAULT_MODEL
        else:
            selected_model = model_name.strip()
        if not selected_model:
            raise ValueError("model_name must not be empty")

        self._model = SentenceTransformer(selected_model, device="cpu")
        model_dimensions = self._model.get_embedding_dimension()
        if model_dimensions is None or model_dimensions <= 0:
            raise RuntimeError(
                f"Could not determine embedding dimensions for {selected_model!r}."
            )
        if dimensions is not None and dimensions != model_dimensions:
            raise ValueError(
                f"Model {selected_model!r} outputs {model_dimensions} dimensions; "
                f"requested {dimensions}."
            )

        self.model_name = selected_model
        self.dimensions = model_dimensions

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must not contain empty or non-string values")

        encoded = self._model.encode(
            texts,
            batch_size=self._BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if len(encoded) != len(texts):
            raise RuntimeError(
                f"Model returned {len(encoded)} vectors for {len(texts)} inputs."
            )

        vectors = [[float(value) for value in vector] for vector in encoded]
        if any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError(
                f"Model output does not match its declared {self.dimensions} dimensions."
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document chunks locally."""
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        """Encode a query with the same model used for document chunks."""
        if not question.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return self._encode([question])[0]

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        """Encode a batch of queries locally."""
        return self._encode(questions)

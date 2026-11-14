"""Persist and search chunks using PostgreSQL and pgvector."""
import json
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    _TABLE_PREFIX = "rag_chunks_"
    _MAX_DIMENSIONS = 16000

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must not be empty")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer")
        if dimensions <= 0 or dimensions > self._MAX_DIMENSIONS:
            raise ValueError(
                f"dimensions must be between 1 and {self._MAX_DIMENSIONS}"
            )

        self.database_url = database_url
        self.dimensions = dimensions
        self._table = f"{self._TABLE_PREFIX}{dimensions}"

    @staticmethod
    def _vector_literal(vector: list[float], dimensions: int) -> str:
        if len(vector) != dimensions:
            raise ValueError(
                f"vector has {len(vector)} dimensions; expected {dimensions}"
            )

        values = []
        for value in vector:
            if isinstance(value, bool):
                raise TypeError("vector values must be numeric, not booleans")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError("vector values must be numeric") from error
            if not math.isfinite(number):
                raise ValueError("vector values must be finite numbers")
            values.append(format(number, ".17g"))

        if not any(float(value) != 0.0 for value in values):
            raise ValueError("zero vectors cannot be used with cosine distance")
        return "[" + ",".join(values) + "]"

    def create_schema(self) -> None:
        """Create the extension and exact-search table if they do not exist."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector({self.dimensions}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = %s::regclass
                      AND attname = 'embedding'
                      AND NOT attisdropped
                    """,
                    (self._table,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(
                        f"Could not inspect the embedding column in {self._table}."
                    )
                if row[0] != self.dimensions:
                    raise ValueError(
                        f"{self._table} already uses {row[0]} dimensions; "
                        f"this model uses {self.dimensions}. Use a separate table "
                        "or migrate the existing table before changing models."
                    )

    def upsert(
        self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Replace the given documents' chunks atomically for one strategy."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy must not be empty")
        if not chunks:
            raise ValueError("chunks must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError(
                f"received {len(chunks)} chunks but {len(vectors)} vectors"
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk IDs must be unique within an upsert")

        rows = []
        for chunk, vector in zip(chunks, vectors):
            if not chunk.chunk_id or not chunk.document_id or not chunk.source:
                raise ValueError("chunk_id, document_id, and source must not be empty")
            if not isinstance(chunk.page, int) or isinstance(chunk.page, bool) or chunk.page < 1:
                raise ValueError("chunk page must be a positive integer")
            if not isinstance(chunk.text, str) or not chunk.text.strip():
                raise ValueError("chunk text must not be empty")
            if not isinstance(chunk.metadata, dict):
                raise TypeError("chunk metadata must be a dictionary")
            json.dumps(chunk.metadata, ensure_ascii=False, allow_nan=False)
            vector_literal = self._vector_literal(vector, self.dimensions)
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    vector_literal,
                    Jsonb(chunk.metadata),
                )
            )

        document_ids = list(dict.fromkeys(chunk.document_id for chunk in chunks))
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._table} "
                    "WHERE strategy = %s AND document_id = ANY(%s)",
                    (strategy, document_ids),
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {self._table}
                        (strategy, chunk_id, document_id, source, page, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    rows,
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks by cosine similarity, applying JSON metadata filters."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")
        query_literal = self._vector_literal(query_vector, self.dimensions)

        filter_clauses = []
        parameters: list[Any] = [query_literal, strategy]
        for key, value in (filters or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata filter keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)) or value is None:
                raise ValueError(
                    f"metadata filter {key!r} must have a non-null scalar value"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"metadata filter {key!r} must be finite")
            if isinstance(value, bool):
                filter_value = "true" if value else "false"
            else:
                filter_value = str(value)
            filter_clauses.append("metadata ->> %s = %s")
            parameters.extend((key, filter_value))

        where_filters = (
            " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
        )
        parameters.extend((query_literal, top_k))
        query = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self._table}
            WHERE strategy = %s{where_filters}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, parameters)
                rows = cursor.fetchall()

        results = []
        for chunk_id, document_id, source, page, text, metadata, score in rows:
            result_metadata = {
                **metadata,
                "document_id": document_id,
                "source": source,
                "page": page,
            }
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source": source,
                    "page": page,
                    "text": text,
                    "score": float(score),
                    "metadata": result_metadata,
                }
            )
        return results

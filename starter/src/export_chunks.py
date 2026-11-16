"""Export stored chunks to a JSON file without embedding vectors."""
import argparse
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Export stored chunks as a JSON array."
    )
    parser.add_argument(
        "--strategy",
        choices=("A", "B", "C", "D", "E"),
        default="A",
        help="Chunking strategy to export (default: A)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=768,
        help="Embedding dimensions and corresponding table (default: 768)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=starter_dir / "results" / "chunks.json",
        help="Output JSON file (default: starter/results/chunks.json)",
    )
    args = parser.parse_args()

    if args.dimensions <= 0 or args.dimensions > 16000:
        parser.error("--dimensions must be between 1 and 16000")

    load_dotenv(starter_dir / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing. Configure it in starter/.env.")

    table = f"rag_chunks_{args.dimensions}"
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT chunk_id, text, document_id, source, page, metadata
                FROM {table}
                WHERE strategy = %s
                ORDER BY page, chunk_id
                """,
                (args.strategy,),
            )
            rows = cursor.fetchall()

    if not rows:
        raise RuntimeError(
            f"No chunks found for strategy {args.strategy} in {table}."
        )

    chunks: list[dict[str, Any]] = [
        {
            "id": chunk_id,
            "text": text,
            "source_id": document_id,
            "source": source,
            "page": page,
            "metadata": metadata,
        }
        for chunk_id, text, document_id, source, page, metadata in rows
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Exported {len(chunks)} chunks from strategy {args.strategy}.")
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()

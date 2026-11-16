"""Evaluate Strategy A using vectors already stored in PostgreSQL."""
import argparse
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .run import _read_questions, _write_json
from .store import PgVectorStore


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval using existing Strategy A vectors."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=starter_dir / "data" / "questions.json",
        help="Benchmark JSON (default: starter/data/questions.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=starter_dir / "results",
        help="Directory for Strategy A traces and summary",
    )
    parser.add_argument(
        "--document-id-alias",
        metavar="STORED_ID=GOLD_SOURCE_ID",
        help="Map a stored document ID to its benchmark source ID for evaluation",
    )
    args = parser.parse_args()

    document_id_alias = None
    if args.document_id_alias:
        stored_id, separator, gold_source_id = args.document_id_alias.partition("=")
        if not separator or not stored_id.strip() or not gold_source_id.strip():
            parser.error("--document-id-alias must be STORED_ID=GOLD_SOURCE_ID")
        document_id_alias = (stored_id.strip(), gold_source_id.strip())

    load_dotenv(starter_dir / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing. Configure it in starter/.env.")

    questions = _read_questions(args.questions)
    embedder = MyEmbedder()
    store = PgVectorStore(database_url, dimensions=embedder.dimensions)

    started_at = time.perf_counter()
    query_vectors = embedder.embed_queries(
        [question["question"] for question in questions]
    )
    traces: list[dict[str, Any]] = []
    for question, query_vector in zip(questions, query_vectors):
        hits = store.search("A", query_vector, top_k=3)
        if not hits:
            raise RuntimeError(
                "No stored chunks were found for Strategy A. "
                "Run ingestion first and verify the database connection."
            )
        evaluation_hits = [
            {
                **hit,
                "document_id": document_id_alias[1],
            }
            if document_id_alias and hit.get("document_id") == document_id_alias[0]
            else hit
            for hit in hits
        ]
        trace = evaluate_query({**question, "strategy": "A"}, evaluation_hits)
        trace["hits"] = hits
        traces.append(trace)

    elapsed_seconds = time.perf_counter() - started_at
    run_info = {
        "benchmark": str(args.questions),
        "strategy": "A",
        "model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "queries": len(questions),
        "elapsed_seconds": elapsed_seconds,
        "document_id_alias": (
            {
                "stored_id": document_id_alias[0],
                "gold_source_id": document_id_alias[1],
            }
            if document_id_alias
            else None
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = args.output_dir / "strategy_a_traces.json"
    summary_path = args.output_dir / "strategy_a_summary.json"
    _write_json(traces_path, {"run": run_info, "traces": traces})
    _write_json(summary_path, {"run": run_info, **summarize(traces)})

    if not any(trace["evidence_hit_at_3"] for trace in traces):
        print(
            "Warning: no evidence matches. Check that stored document IDs "
            "match the benchmark gold_source_id values."
        )
    print(f"Completed Strategy A evaluation for {len(questions)} questions.")
    print(f"Traces: {traces_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

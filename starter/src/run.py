"""Run the ingestion, chunking, retrieval, and evaluation pipeline."""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_heading, by_paragraph, by_sentence, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk, DocumentPage
from .store import PgVectorStore


MIN_CONTENT_PAGES = 30
MIN_DEV_QUESTIONS = 15
MIN_TEST_QUESTIONS = 5


def _read_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)
    if not isinstance(questions, list):
        raise ValueError(f"Benchmark must contain a JSON array: {path}")

    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"Benchmark item {index} must be a JSON object")
        for field in ("id", "split", "question"):
            if not question.get(field):
                raise ValueError(f"Benchmark item {index} is missing {field!r}")

    dev_count = sum(question["split"] == "dev" for question in questions)
    test_count = sum(question["split"] == "test" for question in questions)
    if dev_count < MIN_DEV_QUESTIONS or test_count < MIN_TEST_QUESTIONS:
        raise ValueError(
            "Benchmark requires at least "
            f"{MIN_DEV_QUESTIONS} dev and {MIN_TEST_QUESTIONS} test questions; "
            f"found {dev_count} dev and {test_count} test."
        )
    return questions


def _chunks_for_pages(pages: list[DocumentPage], strategy: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        if strategy == "A":
            chunks.extend(fixed_size(page, size=500, overlap=50))
        elif strategy in ("B", "C"):
            chunks.extend(by_paragraph(page, max_size=1000))
        elif strategy == "D":
            chunks.extend(by_sentence(page, max_size=1000))
        elif strategy == "E":
            chunks.extend(by_heading(page, max_size=1000))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    return chunks


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Ingest a PDF, run five retrieval strategies, and evaluate results."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=starter_dir / "data" / "data.pdf",
        help="Input PDF (default: starter/data/data.pdf)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=starter_dir / "data" / "questions.json",
        help="Benchmark JSON (default: starter/data/questions.json)",
    )
    parser.add_argument(
        "--document-id",
        help="Stable source ID recorded in benchmark gold labels (default: PDF filename stem)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=starter_dir / "results",
        help="Directory for traces and summary (default: starter/results)",
    )
    args = parser.parse_args()

    load_dotenv(starter_dir / ".env")
    if not args.pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    if not args.questions.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {args.questions}")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing. Configure it in starter/.env.")

    questions = _read_questions(args.questions)
    document_id = args.document_id or args.pdf.stem
    pages = load_pdf(args.pdf, document_id=document_id, source=args.pdf.name)
    content_pages = [page for page in pages if page.text.strip()]
    if len(content_pages) < MIN_CONTENT_PAGES:
        raise ValueError(
            f"Corpus has {len(content_pages)} pages with extracted text; "
            f"at least {MIN_CONTENT_PAGES} are required."
        )

    embedder = MyEmbedder()
    store = PgVectorStore(database_url, dimensions=embedder.dimensions)
    store.create_schema()

    query_vectors = embedder.embed_queries(
        [question["question"] for question in questions]
    )
    traces = []
    chunk_counts: dict[str, int] = {}
    strategy_vectors: dict[str, list[list[float]]] = {}
    started_at = time.perf_counter()
    strategies = ("A", "B", "C", "D", "E")
    for strategy in strategies:
        chunks = _chunks_for_pages(pages, strategy)
        if not chunks:
            raise ValueError(f"Strategy {strategy} produced no non-empty chunks.")
        chunk_counts[strategy] = len(chunks)
        if strategy == "C":
            vectors = strategy_vectors["B"]
        else:
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            strategy_vectors[strategy] = vectors
        store.upsert(strategy, chunks, vectors)

        for question, query_vector in zip(questions, query_vectors):
            filters = question.get("filter") if strategy == "C" else None
            hits = store.search(strategy, query_vector, top_k=3, filters=filters)
            trace_question = {**question, "strategy": strategy}
            traces.append(evaluate_query(trace_question, hits))

    elapsed_seconds = time.perf_counter() - started_at
    summary = summarize(traces)
    run_info = {
        "pdf": str(args.pdf),
        "benchmark": str(args.questions),
        "content_pages": len(content_pages),
        "physical_pages": len(pages),
        "model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "chunk_counts": chunk_counts,
        "elapsed_seconds": elapsed_seconds,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "traces.json", {"run": run_info, "traces": traces})
    _write_json(args.output_dir / "summary.json", {"run": run_info, **summary})
    print(
        f"Completed retrieval for {len(questions)} questions "
        f"across {len(strategies)} strategies."
    )
    print(f"Traces: {args.output_dir / 'traces.json'}")
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

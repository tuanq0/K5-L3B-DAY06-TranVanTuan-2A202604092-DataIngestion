"""Evaluate retrieved evidence and preserve per-query traces."""
import unicodedata
from collections import defaultdict
from typing import Any


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    alternatives = question.get("gold_evidence")
    if alternatives is None:
        alternatives = [
            {
                "source_id": question.get("gold_source_id"),
                "page": question.get("gold_page"),
                "quote": question.get("evidence_quote"),
            }
        ]

    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("question must define non-empty gold evidence")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(alternatives):
        if not isinstance(item, dict):
            raise ValueError(f"gold_evidence[{index}] must be an object")

        source_id = (
            item.get("source_id")
            or item.get("gold_source_id")
            or item.get("document_id")
            or item.get("source")
        )
        page = item.get("page", item.get("gold_page"))
        quote = item.get("quote", item.get("evidence_quote"))
        if not source_id or page is None or not isinstance(quote, str) or not quote.strip():
            raise ValueError(
                f"gold_evidence[{index}] requires source_id, page, and a non-empty quote"
            )
        normalized.append(
            {"source_id": str(source_id), "page": page, "quote": _normalize(quote)}
        )

    return normalized


def _matches(
    hit: dict[str, Any], gold_items: list[dict[str, Any]]
) -> dict[str, Any] | None:
    metadata = hit.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("retrieved hit metadata must be an object")

    hit_document_id = hit.get("document_id", metadata.get("document_id"))
    hit_source = hit.get("source", metadata.get("source"))
    hit_page = hit.get("page", metadata.get("page"))
    hit_text = hit.get("text")
    if not isinstance(hit_text, str):
        raise ValueError("each retrieved hit must include text as a string")

    normalized_text = _normalize(hit_text)
    for gold in gold_items:
        source_matches = gold["source_id"] in {
            str(value) for value in (hit_document_id, hit_source) if value is not None
        }
        if (
            source_matches
            and str(hit_page) == str(gold["page"])
            and gold["quote"] in normalized_text
        ):
            return gold
    return None


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check source, page, and quoted evidence; retain the full retrieval trace.

    ``gold_evidence`` may contain multiple acceptable ``source_id``/``page``/
    ``quote`` objects. If omitted, the standard singular gold fields are used.
    """
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    if not isinstance(hits, list):
        raise ValueError("hits must be a list")

    gold_items = _gold_evidence(question)
    checked_hits: list[dict[str, Any]] = []
    evidence_matches: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        if not isinstance(hit, dict):
            raise ValueError(f"hits[{rank - 1}] must be an object")
        matched_gold = _matches(hit, gold_items)
        checked_hits.append(hit)
        if matched_gold is not None and rank <= 3:
            evidence_matches.append(
                {
                    "rank": rank,
                    "chunk_id": hit.get("chunk_id"),
                    "gold_source_id": matched_gold["source_id"],
                    "gold_page": matched_gold["page"],
                }
            )

    return {
        "question_id": question.get("id"),
        "question": question.get("question"),
        "split": question.get("split", "unspecified"),
        "strategy": question.get("strategy", "unspecified"),
        "hits": checked_hits,
        "hit_at_1": bool(checked_hits and _matches(checked_hits[0], gold_items)),
        "evidence_hit_at_3": bool(evidence_matches),
        "evidence_matches": evidence_matches,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            raise ValueError(f"traces[{index}] must be an object")
        strategy = str(trace.get("strategy", "unspecified"))
        split = str(trace.get("split", "unspecified"))
        groups[(strategy, split)].append(trace)

    summaries = []
    for (strategy, split), group in sorted(groups.items()):
        hit_at_1_count = sum(bool(trace.get("hit_at_1")) for trace in group)
        hit_at_3_count = sum(bool(trace.get("evidence_hit_at_3")) for trace in group)
        total = len(group)
        summaries.append(
            {
                "strategy": strategy,
                "split": split,
                "queries": total,
                "hit_at_1": {
                    "hits": hit_at_1_count,
                    "total": total,
                    "rate": hit_at_1_count / total,
                },
                "evidence_hit_at_3": {
                    "hits": hit_at_3_count,
                    "total": total,
                    "rate": hit_at_3_count / total,
                },
            }
        )

    return {"groups": summaries}

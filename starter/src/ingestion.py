"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from docling.document_converter import DocumentConverter
from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract Docling text items back into their physical PDF pages."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    page_keys = doc.pages.keys() if hasattr(doc.pages, "keys") else range(1, len(doc.pages) + 1)
    page_numbers = sorted(int(page_number) for page_number in page_keys)
    page_texts: dict[int, list[str]] = {page_number: [] for page_number in page_numbers}

    for item, _level in doc.iterate_items():
        text = getattr(item, "text", "")
        if not text:
            exporter = getattr(item, "export_to_markdown", None)
            if callable(exporter):
                text = exporter(doc=doc)
        if not isinstance(text, str) or not text.strip():
            continue

        provenance = getattr(item, "prov", [])
        if not isinstance(provenance, (list, tuple)):
            provenance = [provenance]
        for item_provenance in provenance:
            page_number = getattr(item_provenance, "page_no", None)
            if page_number in page_texts:
                page_texts[page_number].append(text)

    pages: list[DocumentPage] = []
    for page_no in page_numbers:
        text = normalize_text("\n".join(page_texts[page_no]))
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_no,
                text=text,
                metadata={
                    "source_path": str(pdf_path),
                    "total_pages": len(page_numbers),
                    "has_text": bool(text),
                },
            )
        )
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving useful paragraph breaks."""
    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\ufeff", "")
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\u200b", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse repeated spaces but keep paragraph breaks meaningful.
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()

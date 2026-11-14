"""Implement fixed-size and structure-aware chunking."""
import re
from typing import Any

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = page.text.strip()
    if not text:
        return [
            Chunk(
                chunk_id=f"{page.document_id}:{page.page}:0",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text="",
                metadata={**page.metadata, "chunk_index": 0},
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + size, len(text))
        window = text[start:end]
        if not window.strip():
            break

        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:{page.page}:{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=window.strip(),
                metadata={**page.metadata, "chunk_index": index, "start_char": start, "end_char": end},
            )
        )

        if end == len(text):
            break
        start = end - overlap
        index += 1

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs and split oversized paragraphs while retaining metadata."""
    if max_size <= 0:
        raise ValueError("max_size must be > 0")

    text = page.text.strip()
    if not text:
        return [
            Chunk(
                chunk_id=f"{page.document_id}:{page.page}:0",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text="",
                metadata={**page.metadata, "chunk_index": 0},
            )
        ]

    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    index = 0

    def flush_current() -> None:
        nonlocal current, index
        if not current.strip():
            return
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:{page.page}:{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=current.strip(),
                metadata={**page.metadata, "chunk_index": index},
            )
        )
        index += 1
        current = ""

    for paragraph in paragraphs:
        if len(paragraph) <= max_size:
            if len((current + "\n" + paragraph).strip()) <= max_size:
                current = (current + "\n" + paragraph).strip()
            else:
                flush_current()
                current = paragraph
        else:
            if current.strip():
                flush_current()
            parts = []
            start = 0
            while start < len(paragraph):
                end = min(start + max_size, len(paragraph))
                parts.append(paragraph[start:end].strip())
                start = end
            for part in parts:
                if part:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{page.document_id}:{page.page}:{index}",
                            document_id=page.document_id,
                            source=page.source,
                            page=page.page,
                            text=part,
                            metadata={**page.metadata, "chunk_index": index},
                        )
                    )
                    index += 1

    flush_current()
    return chunks


def _split_long_text(text: str, max_size: int) -> list[str]:
    parts = []
    remaining = text.strip()
    while len(remaining) > max_size:
        split_at = remaining.rfind(" ", 0, max_size + 1)
        if split_at <= 0:
            split_at = max_size
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _make_chunk(
    page: DocumentPage,
    text: str,
    index: int,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{page.document_id}:{page.page}:{index}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            **(metadata or {}),
            "chunk_index": index,
        },
    )


def by_sentence(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack complete sentences up to a size limit; split only oversized sentences."""
    if max_size <= 0:
        raise ValueError("max_size must be > 0")

    text = page.text.strip()
    if not text:
        return [_make_chunk(page, "", 0)]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", text)
        if sentence.strip()
    ]
    packed: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            packed.append(current)
            current = ""

    for sentence in sentences:
        units = (
            _split_long_text(sentence, max_size)
            if len(sentence) > max_size
            else [sentence]
        )
        for unit in units:
            candidate = f"{current} {unit}".strip() if current else unit
            if current and len(candidate) > max_size:
                flush()
                current = unit
            else:
                current = candidate
    flush()
    return [
        _make_chunk(page, chunk_text, index)
        for index, chunk_text in enumerate(packed)
    ]


_HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s+\S.*|(?:chương|điều|mục|phần)\s+(?:[ivxlcdm]+|\d+)\b.*)$",
    re.IGNORECASE,
)


def by_heading(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Keep Markdown or Vietnamese legal headings with their following content."""
    if max_size <= 0:
        raise ValueError("max_size must be > 0")

    lines = page.text.splitlines()
    if not any(_HEADING_PATTERN.match(line.strip()) for line in lines):
        return by_paragraph(page, max_size)

    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    current_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _HEADING_PATTERN.match(stripped):
            if current_lines:
                sections.append((heading, "\n".join(current_lines).strip()))
            heading = re.sub(r"^#{1,6}\s+", "", stripped)
            current_lines = [stripped]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((heading, "\n".join(current_lines).strip()))

    chunks: list[Chunk] = []
    for section_heading, section_text in sections:
        section_page = DocumentPage(
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=section_text,
            metadata=page.metadata,
        )
        section_chunks = by_paragraph(section_page, max_size)
        for section_chunk in section_chunks:
            if not section_chunk.text.strip():
                continue
            chunk_metadata = {}
            if section_heading:
                chunk_metadata["section_heading"] = section_heading
            chunks.append(
                _make_chunk(
                    page,
                    section_chunk.text,
                    len(chunks),
                    chunk_metadata,
                )
            )
    return chunks

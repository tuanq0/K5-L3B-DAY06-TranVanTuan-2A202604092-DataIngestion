"""Convert a PDF to Markdown with Docling."""
import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def convert_pdf_to_md(pdf_path: str | Path, md_path: str | Path) -> None:
    """Convert ``pdf_path`` to Markdown and write the result to ``md_path``."""
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_file}")

    markdown_file = Path(md_path)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)

    result = DocumentConverter().convert(str(pdf_file))
    markdown = result.document.export_to_markdown()
    markdown_file.write_text(markdown, encoding="utf-8")
    print(f"Converted: {pdf_file} -> {markdown_file}")


def main() -> None:
    starter_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown with Docling.")
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=starter_dir / "data" / "data.pdf",
        help="PDF input (default: starter/data/data.pdf)",
    )
    parser.add_argument(
        "markdown",
        nargs="?",
        type=Path,
        default=starter_dir / "data" / "data.md",
        help="Markdown output (default: starter/data/data.md)",
    )
    args = parser.parse_args()
    convert_pdf_to_md(args.pdf, args.markdown)


if __name__ == "__main__":
    main()

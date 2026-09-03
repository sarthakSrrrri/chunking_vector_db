from pathlib import Path

from docling.document_converter import DocumentConverter


converter = DocumentConverter()


def parse_document(file_path: Path):
    result = converter.convert(str(file_path))

    return result.document




# from pathlib import Path

# from utility.docling_loader import parse_document


document = parse_document(
    Path("../../data/pdfs/structured/edge_cases.pdf")
)

print(document.export_to_markdown())
from pathlib import Path

from docling.document_converter import DocumentConverter


converter = DocumentConverter()


def parse_document(file_path: Path):
    result = converter.convert(str(file_path))

    return result.document

# document = parse_document(
#     Path("")
# )

# print(document.export_to_markdown())
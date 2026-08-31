from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def load_document(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return [
            {
                "text": file_path.read_text(encoding="utf-8"),
                "metadata": {
                    "page": None,
                },
            }
        ]

    if suffix == ".csv":
        return [
            {
                "text": pd.read_csv(file_path).to_csv(index=False),
                "metadata": {
                    "page": None,
                },
            }
        ]

    if suffix == ".pdf":
        reader = PdfReader(file_path)

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text:
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "page": page_number,
                        },
                    }
                )

        return documents

    raise ValueError(f"Unsupported file type: {suffix}")
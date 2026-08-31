from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def load_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    if suffix == ".csv":
        return pd.read_csv(file_path).to_csv(index=False)

    if suffix == ".pdf":
        reader = PdfReader(file_path)

        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {suffix}")
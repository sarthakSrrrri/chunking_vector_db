from pathlib import Path
from pypdf import PdfReader

pdf_path = Path("../data/pdfs/structured/annual_report_2025.pdf")

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print(f"Total pages: {len(reader.pages)}")
print(f"Total characters: {len(text)}")

print("\n--- First 1000 characters ---\n")
print(text[:500])
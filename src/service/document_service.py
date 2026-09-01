from pathlib import Path

from dotenv import load_dotenv


from src.db.schema import MilvusVectorStore
from src.utility.document_loader import load_document
from src.service.embedding_service import EmbeddingService
from src.chunking.fixed_size_chunking import fixed_size_chunking
import os

load_dotenv()

UPLOAD_DIR = Path(os.getenv("DB_UPLOADED_FILE_PATH"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE :int = int(os.getenv("CHUNK_SIZE"))
OVERLAP = int(os.getenv("OVERLAP"))
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv"}

embedding_service = EmbeddingService()
vector_store = MilvusVectorStore()


async def process_document(file):
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    pages = load_document(file_path)

    chunk_records = []  # storing all the chunks here

    for page in pages:
        chunks = fixed_size_chunking(
            text=page["text"],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP,
        )

        for index, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"{file.filename}_{page['metadata']['page']}_{index}",
                    "text": chunk,
                    "source": file.filename,
                    "file_type": extension,
                    "page": page["metadata"]["page"],
                }
            )

    embeddings = embedding_service.embed_documents(
        [chunk["text"] for chunk in chunk_records]
    )

    vector_store.insert(  # Inserting all the entities inside the collection
        chunks=chunk_records,
        embeddings=embeddings,
    )

    return {
        "filename": file.filename,
        "file_type": extension,
        "pages": len(pages),
        "chunks": len(chunk_records),
        "embedding_dimension": embeddings.shape[1],
    }
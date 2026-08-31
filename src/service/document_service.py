from pathlib import Path

# from chunking.fixed_size_chunking import fixed_size_chunking
from src.chunking import fixed_size_chunking
from src.db.vector_db_milvus import MilvusVectorStore
from src.utility.document_loader import load_document
from src.service.embedding_service import EmbeddingService


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

    text = load_document(file_path)

    chunks = fixed_size_chunking(
        text=text,
        chunk_size=500,
        chunk_overlap=50,
    )

    embeddings = embedding_service.embed_documents(
        [chunk for chunk in chunks]
    )

    chunk_records = []

    for index, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "chunk_id": f"{file.filename}_{index}",
                "text": chunk,
                "source": file.filename,
                "file_type": extension,
            }
        )

    vector_store.insert(
        chunks=chunk_records,
        embeddings=embeddings,
    )

    return {
        "filename": file.filename,
        "file_type": extension,
        "characters": len(text),
        "chunks": len(chunks),
        "embedding_dimension": embeddings.shape[1],
    }
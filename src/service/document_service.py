from pathlib import Path

# from chunking.fixed_size_chunking import fixed_size_chunking
# from src.chunking import fixed_size_chunking
from src.db.vector_db_milvus import MilvusVectorStore
from src.utility.document_loader import load_document
from src.service.embedding_service import EmbeddingService
# from src.vector_db.milvus import MilvusVectorStore

from src.chunking.fixed_size_chunking import fixed_size_chunking
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

    pages = load_document(file_path)

    chunk_records = []

    for page in pages:
        chunks = fixed_size_chunking(
            text=page["text"],
            chunk_size=500,
            chunk_overlap=50,
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

    vector_store.insert(
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
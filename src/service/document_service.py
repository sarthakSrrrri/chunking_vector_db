import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking.fixed_size_chunking import fixed_size_chunking
from src.db.schema import MilvusVectorStore
from src.embeddings.registry import get_embedding_model
from src.service.embedding_service import EmbeddingService
from src.utility.document_loader import load_document


load_dotenv()


UPLOAD_DIR = Path(
    os.getenv("DB_UPLOADED_FILE_PATH", "data/uploads")
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "500")
)

OVERLAP = int(
    os.getenv("OVERLAP", "50")
)

ALLOWED_EXTENSIONS = tuple(
    extension.strip()
    for extension in os.getenv(
        "ALLOWED_EXTENSIONS",
        ".pdf,.txt,.csv",
    ).split(",")
)


async def process_document(
    file,
    model_name: str = "bge-small",
):
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    embedding_model = get_embedding_model(
        model_name
    )

    embedding_service = EmbeddingService(
        embedding_model
    )

    vector_store = MilvusVectorStore(
        collection_name=f"documents_{model_name}",
        dimension=embedding_model.dimension,
    )

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    pages = load_document(file_path)

    chunk_records = []

    for page in pages:

        chunks = fixed_size_chunking(
            text=page["text"],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP,
        )

        for index, chunk in enumerate(chunks):

            chunk_records.append(
                {
                    "chunk_id": (
                        f"{file.filename}_"
                        f"{page['metadata']['page']}_"
                        f"{index}"
                    ),
                    "text": chunk,
                    "source": file.filename,
                    "file_type": extension,
                    "page": page["metadata"]["page"],
                }
            )

    embeddings = embedding_service.embed_documents(
        [
            chunk["text"]
            for chunk in chunk_records
        ]
    )

    vector_store.insert(
        chunks=chunk_records,
        embeddings=embeddings,
    )

    return {
        "filename": file.filename,
        "model": embedding_model.model_name,
        "file_type": extension,
        "pages": len(pages),
        "chunks": len(chunk_records),
        "embedding_dimension": embedding_model.dimension,
    }
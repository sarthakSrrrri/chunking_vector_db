import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking.register import get_chunking_strategy
from src.db.schema import MilvusVectorStore
from src.embeddings.registry import get_embedding_model
from src.service.embedding_service import EmbeddingService
from src.utility.document_loader import load_document
from src.utility.docling_loader import parse_document


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
    model_name: str,
    chunking_method: str,
):
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    # Pick the embedding model and its Milvus collection.
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

    # Save the uploaded file before parsing it.
    file_path = UPLOAD_DIR / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    # Chunking is selected through the registry.
    chunking_strategy = get_chunking_strategy(
        chunking_method
    )

    chunk_records = []
    pages = []

    # Structured chunking needs the Docling document.
    if chunking_method == "structured":

        document = parse_document(
            file_path
        )

        chunks = chunking_strategy.chunk(
            document
        )

        for index, chunk in enumerate(chunks):

            chunk_records.append(
                {
                    "chunk_id": (
                        f"{file.filename}_"
                        f"{index}"
                    ),
                    "text": chunk,
                    "source": file.filename,
                    "file_type": extension,
                    "page": None,
                }
            )

    else:

        # Other chunkers currently work with extracted page text.
        pages = load_document(
            file_path
        )

        for page in pages:

            chunks = chunking_strategy.chunk(
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

    if not chunk_records:
        raise ValueError(
            "No chunks were generated from the document"
        )

    # Embed the final chunks and store them in Milvus.
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
        "chunking_method": chunking_method,
        "file_type": extension,
        "pages": len(pages) if pages else None,
        "chunks": len(chunk_records),
        "embedding_dimension": embedding_model.dimension,
    }
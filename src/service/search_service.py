import os
import time

from dotenv import load_dotenv

from src.db.schema import MilvusVectorStore
from src.embeddings.registry import get_embedding_model
from src.service.embedding_service import EmbeddingService


load_dotenv()


DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))


def search_documents(
    query: str,
    model_name: str,
    top_k: int = DEFAULT_TOP_K,
):
    start_time = time.perf_counter()

    embedding_model = get_embedding_model(model_name)

    embedding_service = EmbeddingService(
        embedding_model
    )

    vector_store = MilvusVectorStore(
        collection_name=f"documents_{model_name}",
        dimension=embedding_model.dimension,
    )

    query_embedding = embedding_service.embed_query(
        query
    )

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    retrieved = [] # adding retreival chunks

    for rank, result in enumerate(results[0], start=1):
        retrieved.append({
            "rank": rank,
            "score": result["distance"],
            "text": result["entity"]["text"],
            "source": result["entity"]["source"],
            "page": result["entity"].get("page"),
            "file_type": result["entity"]["file_type"],
        })

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    return {
        "query": query,
        "model": embedding_model.model_name,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "results": retrieved,
    }
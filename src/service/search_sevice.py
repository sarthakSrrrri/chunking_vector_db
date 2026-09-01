import time
from src.db.schema import MilvusVectorStore
from src.service.embedding_service import EmbeddingService
import os
from dotenv import load_dotenv

load_dotenv()
embedding_service = EmbeddingService()
vector_store = MilvusVectorStore()


def search_documents(query: str, top_k: int = os.getenv("TOP_K")):
    start_time = time.perf_counter()

    query_embedding = embedding_service.embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    retrieved = []

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
        "top_k": top_k,
        "latency_ms": latency_ms,
        "results": retrieved,
    }



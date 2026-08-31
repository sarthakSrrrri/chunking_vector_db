from src.service.embedding_service import EmbeddingService
from src.vector_db.milvus import MilvusVectorStore


embedding_service = EmbeddingService()
vector_store = MilvusVectorStore()


def search_documents(query: str, top_k: int = 5):
    query_embedding = embedding_service.embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    retrieved = []

    for result in results[0]:
        retrieved.append({
            "score": result["distance"],
            "text": result["entity"]["text"],
            "source": result["entity"]["source"],
            "file_type": result["entity"]["file_type"],
        })

    return retrieved
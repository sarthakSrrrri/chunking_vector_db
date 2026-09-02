from src.embeddings.base import EmbeddingModel
from src.embeddings.bge_small import BGESmallEmbedding
from src.embeddings.bge_base import BGEBaseEmbedding
from src.embeddings.e5_base import E5BaseEmbedding
from src.embeddings.nomic import NomicEmbedding


EMBEDDING_MODELS: dict[str, type[EmbeddingModel]] = {
    "bge-small": BGESmallEmbedding,
    "bge-base": BGEBaseEmbedding,
    "e5-base": E5BaseEmbedding,
    "nomic": NomicEmbedding,
}


def get_embedding_model(name: str) -> EmbeddingModel:
    model_class = EMBEDDING_MODELS.get(name)

    if model_class is None:
        raise ValueError(
            f"Unknown embedding model: {name}. "
            f"Available models: {list(EMBEDDING_MODELS)}"
        )

    return model_class()
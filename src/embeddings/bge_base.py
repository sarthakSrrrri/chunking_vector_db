import os

from sentence_transformers import SentenceTransformer

from src.embeddings.base import EmbeddingModel



os.environ["HF_HOME"] = os.getenv(
    "HF_HOME",
    "data/models",
)

class BGEBaseEmbedding(EmbeddingModel):

    def __init__(self):
        self.model = SentenceTransformer(
            os.getenv("BGE_BASE")
        )

    @property
    def model_name(self) -> str:
        return os.getenv("BGE_BASE")

    @property
    def dimension(self) -> int:
        return 768

    def embed_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, text: str):
        return self.model.encode(
            text,
            normalize_embeddings=True,
        )
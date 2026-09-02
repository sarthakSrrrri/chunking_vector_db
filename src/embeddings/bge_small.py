from sentence_transformers import SentenceTransformer
# from base import EmbeddingModel
import os
from dotenv import load_dotenv

from src.embeddings.base import EmbeddingModel
load_dotenv()
os.environ["HF_HOME"] = os.getenv(
    "HF_HOME",
    "data/models",
)
class BGESmallEmbedding(EmbeddingModel):

    def __init__(self):
        self.model = SentenceTransformer(
            os.getenv("BGE_SMALL")
        )

    @property
    def model_name(self) -> str:
        return os.getenv("BGE_SMALL")

    @property
    def dimension(self) -> int:
        return 384

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
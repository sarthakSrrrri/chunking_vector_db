from sentence_transformers import SentenceTransformer

from src.embeddings.base import EmbeddingModel
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HOME"] = os.getenv(
    "HF_HOME",
    "data/models",
)
class NomicEmbedding(EmbeddingModel):

    def __init__(self):
        self.model = SentenceTransformer(
            os.getenv("NOMIC_EMBED_TEXT"),
            trust_remote_code=True,
        )

    @property
    def model_name(self) -> str:
        return os.getenv("NOMIC_EMBED_TEXT")

    @property
    def dimension(self) -> int:
        return 768

    def embed_documents(self, texts: list[str]):
        texts = [f"search_document: {text}" for text in texts]

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, text: str):
        return self.model.encode(
            f"search_query: {text}",
            normalize_embeddings=True,
        )
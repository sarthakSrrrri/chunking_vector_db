from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()

class EmbeddingService:

    def __init__(self, model_name: str = os.getenv("EMBEEDING_MODEL")):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )
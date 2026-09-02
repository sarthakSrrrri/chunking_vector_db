from src.embeddings.base import EmbeddingModel


class EmbeddingService:

    def __init__(self, model: EmbeddingModel):
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def dimension(self) -> int:
        return self.model.dimension

    def embed_documents(self, texts: list[str]):
        return self.model.embed_documents(texts)

    def embed_query(self, text: str):
        return self.model.embed_query(text)
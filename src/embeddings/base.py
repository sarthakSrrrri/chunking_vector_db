from abc import ABC, abstractmethod


class EmbeddingModel(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]):
        pass

    @abstractmethod
    def embed_query(self, text: str):
        pass
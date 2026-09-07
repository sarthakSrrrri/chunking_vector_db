from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chunk(self, document):
        pass
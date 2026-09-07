from src.chunking.base import ChunkingStrategy


class FixedSizeChunking(ChunkingStrategy):

    @property
    def name(self) -> str:
        return "fixed"

    def chunk(
        self,
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and smaller than chunk_size"
            )

        chunks = []
        start = 0
        step = chunk_size - chunk_overlap

        while start < len(text):
            chunks.append(
                text[start:start + chunk_size]
            )
            start += step

        return chunks
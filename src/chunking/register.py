from src.chunking.fixed_size_chunking import FixedSizeChunking
from src.chunking.llamaindex_chunking import LlamaIndexChunking
from src.chunking.structure_aware_chunking import StructuredChunking


CHUNKING_STRATEGIES = {
    "fixed": FixedSizeChunking,
    "llamaindex": LlamaIndexChunking,
    "structured": StructuredChunking,
}


def get_chunking_strategy(name: str):
    strategy = CHUNKING_STRATEGIES.get(name)

    if strategy is None:
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Available strategies: {list(CHUNKING_STRATEGIES)}"
        )

    return strategy()
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


def llamaindex_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    document = Document(text=text)

    nodes = splitter.get_nodes_from_documents([document])

    return [node.text for node in nodes]
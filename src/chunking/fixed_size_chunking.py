import os
from dotenv import load_dotenv
load_dotenv()

def fixed_size_chunking(
    text: str,
    chunk_size = int(os.getenv("CHUNK_SIZE")),
    chunk_overlap = int(os.getenv("OVERLAP")),
) -> list[str]:

    if chunk_size < 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    print("total steps : " , step)
    print(len(chunks))
    return chunks


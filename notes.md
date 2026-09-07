
# inside src/chunking folder


    # llamaindex_chunking.py 
        uses text boundaries. LlamaIndex's SentenceSplitter tries to keep sentences and paragraphs together, but it still works mainly on text and a chunk_size limit.LlamaIndex may see this mostly as text and split according to sentence/chunk boundaries.

    # structured_chunking.py 
        uses document structure from Docling. It knows that something is a heading, paragraph, list, or table. So a table can be kept as one unit instead of being treated like ordinary text.
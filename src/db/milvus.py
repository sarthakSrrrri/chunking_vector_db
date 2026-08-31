def search(self, query_embedding, top_k=5):
    results = self.client.search(
        collection_name=self.collection_name,
        data=[query_embedding.tolist()],
        limit=top_k,
        output_fields=[
     "text",
    "source",
    "file_type",
    "page",
        ],
    )

    return results


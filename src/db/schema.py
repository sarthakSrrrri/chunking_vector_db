import os

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient


load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION", "documents")
DIMENSION = int(os.getenv("DIMENSION", "384"))
METRIC_TYPE = os.getenv("METRIC_TYPE", "COSINE")


class MilvusVectorStore:

    def __init__(
        self,
        db_path: str = "data/milvus.db",
        collection_name: str = COLLECTION_NAME,
        dimension: int = DIMENSION,
    ):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.dimension = dimension

        self._create_collection()

    def _create_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(
                collection_name=self.collection_name
            )
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=200,
            is_primary=True,
        )

        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=METRIC_TYPE,
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        self.client.load_collection(
            collection_name=self.collection_name
        )

    def insert(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must be the same"
            )

        data = []

        for chunk, embedding in zip(chunks, embeddings):
            data.append(
                {
                    "id": chunk["chunk_id"],
                    "vector": embedding.tolist(),
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "file_type": chunk["file_type"],
                    "page": chunk.get("page"),
                }
            )

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

        self.client.flush(self.collection_name)

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
from fastapi import FastAPI

from src.routes.document import router as document_router


app = FastAPI(
    # title="Milvus",
    version="1.0.0",
)

app.include_router(document_router)
from fastapi import FastAPI

from src.routes.document import router as document_router


app = FastAPI(
    title="Chunking & Vector DB API",
    version="1.0.0",
)

app.include_router(document_router)
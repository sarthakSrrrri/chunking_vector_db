from fastapi import FastAPI

from src.routes.document import router as document_router

app = FastAPI(
    version="1.0.0",
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.include_router(document_router)
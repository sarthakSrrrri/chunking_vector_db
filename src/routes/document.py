from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.service.document_service import process_document
from src.service.search_service import search_documents


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    model: str = Query(...),
    chunking_method: str = Query(...),
):
    try:
        return await process_document(
            file=file,
            model_name=model,
            chunking_method=chunking_method,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.get("/search")
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    model: str = Query("bge-small"),
):
    try:
        return search_documents(
            query=query,
            top_k=top_k,
            model_name=model,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
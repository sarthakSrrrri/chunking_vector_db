from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.service.document_service import process_document
from src.service.search_sevice import search_documents
# from src.service.search_service import search_documents


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        return await process_document(file)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.get("/search")
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
):
    return search_documents(
        query=query,
        top_k=top_k,
    )
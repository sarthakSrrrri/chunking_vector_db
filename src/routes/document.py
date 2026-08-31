from fastapi import APIRouter, UploadFile, File, HTTPException

from src.service.document_service import process_document


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
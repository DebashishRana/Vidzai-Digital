from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.document_detector import detect_and_classify

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/detect-document")
async def detect_document(file: UploadFile = File(...)):
    """Upload an image → get back document detections with Aadhar/Non-Aadhar labels."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Only JPEG/PNG images accepted")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, "File too large (max 10 MB)")

    try:
        detections = detect_and_classify(contents)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    return {"detections": detections}

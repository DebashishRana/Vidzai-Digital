from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ocr_service import extract_text
from app.services.text_classifier import classify_fields

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/extract-text")
async def extract_text_endpoint(file: UploadFile = File(...)):
    """Upload an image → get back OCR text and classified fields."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Only JPEG/PNG images accepted")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, "File too large (max 10 MB)")

    text = extract_text(contents)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    try:
        classified = classify_fields(lines)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    return {"raw_text": text, "fields": classified}

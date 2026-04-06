from fastapi import APIRouter, UploadFile, File, HTTPException
from app.address_detection import AddressDetector
import tempfile
from pathlib import Path
import uuid
import json
import time

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB

_DEBUG_SESSION_ID = "96abd7"
def _workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    # Fallback: best effort (repo root in this workspace layout)
    return here.parents[4]

_DEBUG_LOG_PATH = _workspace_root() / "debug-96abd7.log"

def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": _DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

# Initialize detector globally (lazy load)
_detector = None

def get_detector():
    """Lazy load address detector to avoid startup delays."""
    global _detector
    if _detector is None:
        try:
            _detector = AddressDetector()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Address detection models not fully loaded: {str(e)}. Please train the models first."
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Address detector initialization failed: {str(e)}"
            )
    return _detector


@router.post("/detect/address")
@router.post("/detect-address")
async def detect_address(file: UploadFile = File(..., description="Document image for address detection")):
    """
    Address Detection Endpoint
    
    Detects address regions in documents using YOLO and extracts text via EasyOCR.
    
    **Returns:**
    - `address`: Extracted address text
    - `confidence`: Detection confidence score
    - `status`: Detection status (success/not_found)
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Only JPEG/PNG images accepted")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, f"File too large (max {MAX_SIZE / 1024 / 1024:.0f} MB)")

    temp_file = None
    try:
        _dbg(
            "H1",
            "main/backend/app/routers/address.py:detect_address",
            "Received upload for address detection",
            {"filename": file.filename, "content_type": file.content_type, "size_bytes": len(contents)},
        )
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            temp_file = tmp.name

        # Load detector and perform inference
        detector = get_detector()
        _dbg(
            "H2",
            "main/backend/app/routers/address.py:detect_address",
            "Detector initialized (lazy-load)",
            {
                "has_yolo_model": bool(getattr(detector, "model", None)),
                "has_ocr_reader": bool(getattr(detector, "reader", None)),
                "temp_file_suffix": Path(temp_file).suffix if temp_file else None,
            },
        )
        result = detector.detect_and_extract(temp_file)

        if result is None:
            _dbg(
                "H3",
                "main/backend/app/routers/address.py:detect_address",
                "No address extracted (detector returned None)",
                {"temp_file": bool(temp_file)},
            )
            return {
                "status": "not_found",
                "success": False,
                "message": "No address region detected in image",
                "address": None,
                "confidence": 0.0,
                "detection_id": f"ADDR{str(uuid.uuid4().hex[:6]).upper()}"
            }

        _dbg(
            "H4",
            "main/backend/app/routers/address.py:detect_address",
            "Address extracted successfully",
            {
                "address_len": len(result.get("address", "") or ""),
                "confidence": result.get("confidence", 0.0),
                "image_name": result.get("image_name"),
            },
        )
        return {
            "status": "success",
            "success": True,
            "detection_id": f"ADDR{str(uuid.uuid4().hex[:6]).upper()}",
            "address": result.get("address", ""),
            "confidence": result.get("confidence", 0.0),
            "image_name": result.get("image_name", file.filename),
            "message": "✅ Address detected and extracted successfully"
        }

    except Exception as e:
        raise HTTPException(500, f"Address detection error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file and Path(temp_file).exists():
            Path(temp_file).unlink()


@router.get("/detect-address/status")
@router.get("/detect/address/status")
async def address_detection_status():
    """Check address detection model status."""
    try:
        models_dir = Path(__file__).parent.parent / "models"
        model_joblib_path = models_dir / "yolo_address_model.joblib"
        model_pt_path = models_dir / "yolov8n.pt"
        
        if model_joblib_path.exists() or model_pt_path.exists():
            available_models = []
            if model_joblib_path.exists():
                available_models.append(str(model_joblib_path.name))
            if model_pt_path.exists():
                available_models.append(str(model_pt_path.name))

            return {
                "status": "ready",
                "model_type": "YOLO Address Detection + EasyOCR",
                "model_files": available_models,
                "message": "Address detection model is ready for inference"
            }
        else:
            return {
                "status": "not_ready",
                "message": "Address detection model not found",
                "checked_paths": [str(model_joblib_path), str(model_pt_path)]
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/detect-address/batch")
@router.post("/detect/address/batch")
async def detect_address_batch(files: list[UploadFile] = File(..., description="Multiple document images")):
    """
    Batch Address Detection
    
    Process multiple images at once.
    
    **Returns:**
    - List of detection results for each image
    """
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 files per request")

    results = []
    
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Invalid file format (JPEG/PNG only)"
            })
            continue

        contents = await file.read()
        if len(contents) > MAX_SIZE:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"File too large (max {MAX_SIZE / 1024 / 1024:.0f} MB)"
            })
            continue

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(contents)
                temp_file = tmp.name

            detector = get_detector()
            result = detector.detect_and_extract(temp_file)

            if result is None:
                results.append({
                    "filename": file.filename,
                    "status": "not_found",
                    "address": None,
                    "confidence": 0.0
                })
            else:
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "address": result.get("address", ""),
                    "confidence": result.get("confidence", 0.0)
                })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
        
        finally:
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    return {
        "total_files": len(files),
        "results": results,
        "summary": {
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "failed": sum(1 for r in results if r.get("status") == "error"),
            "not_found": sum(1 for r in results if r.get("status") == "not_found")
        }
    }

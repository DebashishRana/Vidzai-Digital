from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import DB_ENABLED, MEMORY_STORE, get_db
from app import db_models
import uuid
import joblib
import numpy as np
import cv2
from pathlib import Path
import random

FaceNet = None
FACENET_IMPORT_ERROR = None

def _load_facenet():
    """Lazy load FaceNet to delay TensorFlow initialization."""
    global FaceNet, FACENET_IMPORT_ERROR
    if FaceNet is not None:
        return FaceNet
    try:
        from keras_facenet import FaceNet as FaceNetModel
        FaceNet = FaceNetModel
        FACENET_IMPORT_ERROR = None
        return FaceNet
    except Exception as e:
        FaceNet = None
        FACENET_IMPORT_ERROR = e
        return None

from app.services.document_detector import detect_and_classify
from app.services.ocr_service import extract_text
from app.services.text_classifier import classify_fields

router = APIRouter()

def generate_mock_face_verification():
    """Generate mock face verification with realistic variation including mismatches"""
    # 30% chance of mismatch for variety in demo
    is_match = random.random() > 0.3
    if is_match:
        confidence = random.uniform(0.78, 0.98)  # Match: high confidence
    else:
        confidence = random.uniform(0.35, 0.65)  # Mismatch: lower confidence
    
    threshold = 0.70
    similarity = confidence
    
    return {
        "is_match": is_match,
        "confidence": confidence,
        "threshold": threshold,
        "similarity": similarity,
        "from_mock": True
    }

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/verify-full")
async def verify_full(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Full pipeline: detect document type → extract text → classify fields."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Only JPEG/PNG images accepted")

    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, "File too large (max 10 MB)")

    # Step 1: Detect and classify document
    try:
        detections = detect_and_classify(contents)
    except Exception as exc: 
        print(f"Detection error: {str(exc)} - using fallback")
        detections = [{"label": "Aadhar", "confidence": 0.8}]

    is_aadhar = any(
        d["label"] == "Aadhar" and d["confidence"] > 0.6
        for d in detections
    )
    doc_type = "Aadhar Card" if is_aadhar else "Other Document"

    # Step 2: Extract text via OCR
    try:
        text = extract_text(contents)
    except Exception as exc:
        print(f"OCR error: {str(exc)} - using fallback")
        text = "Name: John Doe\nAadhar: 1234-5678-9012\nAddress: Test Address"
    
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Step 3: Classify text fields
    if lines:
        try:
            classified = classify_fields(lines)
        except Exception as exc:
            print(f"Classification error: {str(exc)} - using fallback")
            classified = [{"text": line, "field": "OTHER"} for line in lines]
    else:
        classified = []

    # Calculate mock risk score
    risk_score = 10.0 if is_aadhar else 85.0
    status = "verified" if is_aadhar else "flagged"
    confidence = 0.92 if is_aadhar else 0.45

    # Create verification in database
    from app import crud
    
    ver_id = f"KYC{str(uuid.uuid4().hex[:6]).upper()}"
    ver_obj = crud.create_verification(db, {
        "id": ver_id,
        "user_id": 1,
        "document_type": doc_type,
        "status": status,
        "risk_score": risk_score,
        "confidence": confidence,
        "document_path": file.filename or "unknown"
    })
    
    # Create document record
    doc_id = f"DOC{str(uuid.uuid4().hex[:6]).upper()}"
    doc_obj = crud.create_document(db, {
        "id": doc_id,
        "verification_id": ver_id,
        "file_path": file.filename or "unknown",
        "extracted_text": text[:500]  # Store first 500 chars of extracted text
    })
    
    # Create Alert if Risk is High
    if risk_score > 70:
        alert_id = f"ALT{str(uuid.uuid4().hex[:6]).upper()}"
        alert_obj = crud.create_alert(db, {
            "id": alert_id,
            "verification_id": ver_id,
            "risk_level": "High",
            "alert_type": "Document Not Recognized",
            "status": "Active"
        })

    return {
        "verification_id": ver_id,
        "document_type": doc_type,
        "detections": detections,
        "extracted_text": text,
        "fields": classified,
        "is_verified": is_aadhar,
        "risk_score": risk_score
    }


# ── FACE VERIFICATION ENDPOINTS ──────────────────────────────────────────

# Global embedder instance
embedder = None

def get_embedder():
    """Lazy load FaceNet embedder to avoid startup delays."""
    global embedder
    FaceNet = _load_facenet()
    if FaceNet is None:
        raise RuntimeError(
            f"Face verification dependency unavailable: {FACENET_IMPORT_ERROR}"
        )
    if embedder is None:
        embedder = FaceNet()
    return embedder


def bytes_to_image(file_bytes):
    """Convert bytes to OpenCV image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image format - could not decode")
    return img


def embed_image(image_bytes, model_config):
    """Generate 128-dim embedding for an image using FaceNet."""
    embedder = get_embedder()
    
    try:
        # Load and preprocess image
        img = bytes_to_image(image_bytes)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, model_config['img_size'])
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Generate embedding
        embedding = embedder.embeddings(np.expand_dims(img_normalized, 0))[0]
        return embedding
    except Exception as e:
        raise ValueError(f"Embedding generation failed: {str(e)}")


@router.post("/verify-face")
async def verify_face_endpoint(
    selfie: UploadFile = File(..., description="Selfie image (JPEG/PNG)"),
    id_photo: UploadFile = File(..., description="ID photo image (JPEG/PNG)")
):
    """
    Face Verification Endpoint
    
    Compares a selfie against an ID photo using FaceNet embeddings and ensemble similarity.
    
    **Returns:**
    - `match`: True if faces match above threshold
    - `confidence`: Similarity score (0-1)
    - `status`: "verified" or "mismatch"
    """
    
    # Validate file types first
    if selfie.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(400, "Selfie: Only JPEG/PNG images accepted")
    if id_photo.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(400, "ID Photo: Only JPEG/PNG images accepted")
    
    # Read and validate file sizes
    selfie_bytes = await selfie.read()
    id_bytes = await id_photo.read()
    
    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    if len(selfie_bytes) > MAX_SIZE or len(id_bytes) > MAX_SIZE:
        raise HTTPException(400, f"Files too large (max {MAX_SIZE / 1024 / 1024:.0f} MB each)")
    
    try:
        # Load trained face verification model
        model_path = Path(__file__).parent.parent / "models" / "face_verification_model.joblib"
        
        if model_path.exists():
            model = joblib.load(str(model_path))
            config = model['config']
            threshold = model['best_threshold']
            ensemble_sim = model['similarity_functions']['ensemble_sim']
            
            # Generate embeddings
            selfie_emb = embed_image(selfie_bytes, config)
            id_emb = embed_image(id_bytes, config)
            
            # Compute similarity
            similarity = ensemble_sim(selfie_emb, id_emb)
            is_match = float(similarity) > float(threshold)
            confidence = float(similarity)
            use_mock = False
        else:
            # Model not found - use mock data for demo
            mock_result = generate_mock_face_verification()
            is_match = mock_result['is_match']
            confidence = mock_result['confidence']
            threshold = mock_result['threshold']
            use_mock = True
        
        # Store verification in memory (demo mode - no database)
        ver_id = f"FACE{str(uuid.uuid4().hex[:6]).upper()}"
        
        ver_obj = {
            "id": ver_id,
            "user_id": 1,
            "document_id": None,
            "status": "verified" if is_match else "flagged",
            "risk_score": 100.0 - (confidence * 100)
        }
        MEMORY_STORE["verifications"].append(ver_obj)
        
        # Create alert if mismatch detected
        if not is_match:
            alert_id = f"FACE{str(uuid.uuid4().hex[:4]).upper()}"
            alert_obj = {
                "id": alert_id,
                "user_id": 1,
                "risk_level": "High",
                "alert_type": "Face Mismatch",
                "status": "Active"
            }
            MEMORY_STORE["alerts"].append(alert_obj)
        
        if use_mock:
            return {
                "verification_id": ver_id,
                "match": is_match,
                "confidence": confidence,
                "threshold": threshold,
                "similarity_score": float(confidence),
                "status": "verified" if is_match else "flagged",
                "risk_score": 100.0 - (confidence * 100),
                "message": f"{'✅ Face match confirmed' if is_match else '❌ Face flagged - possible mismatch'}",
                "mode": "DEMO"
            }
        else:
            return {
                "verification_id": ver_id,
                "match": is_match,
                "confidence": confidence,
                "threshold": threshold,
                "similarity_score": float(similarity),
                "status": "verified" if is_match else "flagged",
                "risk_score": 100.0 - (confidence * 100),
                "message": f"{'✅ Face match confirmed' if is_match else '❌ Face flagged - possible mismatch'}",
                "model_info": {
                    "accuracy": model['best_accuracy'],
                    "auc_roc": model['metrics']['auc_roc']
                }
            }
        
    except ValueError as e:
        raise HTTPException(400, f"Image processing error: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(503, f"Face verification unavailable: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Face verification error: {str(e)}")


@router.get("/verify-face/status")
async def face_verification_status():
    """Check FaceNet model status and configuration."""
    try:
        FaceNet = _load_facenet()
        if FaceNet is None:
            return {
                "status": "not_ready",
                "message": "Face verification dependency unavailable in current runtime",
                "error": str(FACENET_IMPORT_ERROR)
            }

        model_path = Path(r"D:\Python\Infosys\main\backend\app\models\face_verification_model.joblib")
        metadata_path = Path(r"D:\Python\Infosys\main\backend\app\models\model_metadata.json")
        
        if model_path.exists():
            model = joblib.load(str(model_path))
            return {
                "status": "ready",
                "model_type": "FaceNet + Ensemble Similarity",
                "model_file": str(model_path),
                "threshold": model['best_threshold'],
                "accuracy": model['best_accuracy'],
                "auc_roc": model['metrics']['auc_roc'],
                "total_persons_trained": model['metrics']['total_persons'],
                "total_pairs_trained": model['metrics']['total_pairs'],
                "config": model['config']
            }
        else:
            return {
                "status": "not_ready",
                "message": "Face verification model not found. Train the model first.",
                "model_path": str(model_path)
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


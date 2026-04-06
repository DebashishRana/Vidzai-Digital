"""
Router for Verification Logs Endpoints
Handles saving and retrieving verification records
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json
from datetime import datetime

from app.database import SessionLocal, DB_ENABLED, get_db, MEMORY_STORE
from app.models import VerificationLog

router = APIRouter(tags=["verification-logs"])


class SaveVerificationRequest(BaseModel):
    verification_id: str
    document_type: str
    confidence_score: float
    status: str
    detected_address: str
    verification_details: Optional[Dict[str, Any]] = None


@router.post("/save-verification")
async def save_verification(
    request: SaveVerificationRequest,
    db: Session = Depends(get_db)
):
    """
    Save a verification record to the database
    
    Args:
        verification_id: Unique verification ID
        document_type: Type of document verified
        confidence_score: Confidence score (0-1)
        status: Verification status (Verified/Review Required/Failed)
        detected_address: Detected address from document
        verification_details: JSON details of verification steps and scores
    """

    if not DB_ENABLED:
        # Mock saving in memory
        new_record = {
            "id": len(MEMORY_STORE.get("verification_logs", [])) + 1,
            "verification_id": request.verification_id,
            "document_type": request.document_type,
            "confidence_score": request.confidence_score,
            "status": request.status,
            "detected_address": request.detected_address,
            "verification_details": request.verification_details,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if "verification_logs" not in MEMORY_STORE:
            MEMORY_STORE["verification_logs"] = []
            
        # Update if exists
        existing_idx = next((i for i, r in enumerate(MEMORY_STORE["verification_logs"]) if r["verification_id"] == request.verification_id), None)
        if existing_idx is not None:
            MEMORY_STORE["verification_logs"][existing_idx].update(new_record)
        else:
            MEMORY_STORE["verification_logs"].append(new_record)
            
        return {
            "success": True,
            "message": "Verification record saved to memory (DEMO MODE)",
            "verification_id": request.verification_id
        }

    try:
        # Check if verification already exists
        existing = db.query(VerificationLog).filter(
            VerificationLog.verification_id == request.verification_id
        ).first()
        
        if existing:
            # Update existing record
            existing.confidence_score = request.confidence_score
            existing.status = request.status
            existing.detected_address = request.detected_address
            existing.verification_details = request.verification_details
            existing.updated_at = datetime.utcnow()
            db.commit()
            return {
                "success": True,
                "message": "Verification record updated",
                "verification_id": request.verification_id,
                "id": existing.id
            }
        else:
            # Create new record
            verification_log = VerificationLog(
                verification_id=request.verification_id,
                document_type=request.document_type,
                confidence_score=request.confidence_score,
                status=request.status,
                detected_address=request.detected_address,
                verification_details=request.verification_details
            )

            db.add(verification_log)
            db.commit()
            db.refresh(verification_log)
            
            return {
                "success": True,
                "message": "Verification record saved successfully",
                "verification_id": request.verification_id,
                "id": verification_log.id
            }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error saving verification record: {str(e)}"
        )


@router.get("/verification-logs/records")
async def get_verifications(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    status: Optional[str] = None,
    document_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get verification records with optional filtering
    
    Args:
        limit: Number of records to return (default 50, max 1000)
        offset: Number of records to skip (for pagination)
        status: Filter by status (Verified/Review Required/Failed)
        document_type: Filter by document type
        date_from: Filter records from this date (YYYY-MM-DD)
        date_to: Filter records until this date (YYYY-MM-DD)
    """

    if not DB_ENABLED:
        in_memory = MEMORY_STORE.get("verification_logs", [])
        
        # Apply filters (simplified for memory)
        if status:
            in_memory = [r for r in in_memory if r.get("status") == status]
        if document_type:
            in_memory = [r for r in in_memory if r.get("document_type") == document_type]
            
        paginated = sorted(in_memory, key=lambda x: x.get("created_at", ""), reverse=True)[offset:offset+limit]
            
        return {
            "data": paginated,
            "total": len(in_memory),
            "limit": limit,
            "offset": offset,
            "message": "Returned from memory (DEMO MODE)"
        }

    try:
        query = db.query(VerificationLog)
        
        # Apply filters
        if status:
            query = query.filter(VerificationLog.status == status)
        
        if document_type:
            query = query.filter(VerificationLog.document_type == document_type)
        
        if date_from:
            from datetime import datetime
            date_from_obj = datetime.fromisoformat(date_from)
            query = query.filter(VerificationLog.created_at >= date_from_obj)
        
        if date_to:
            from datetime import datetime
            date_to_obj = datetime.fromisoformat(date_to)
            query = query.filter(VerificationLog.created_at <= date_to_obj)
        
        # Get total count before pagination
        total = query.count()
        
        # Apply pagination
        records = query.order_by(VerificationLog.created_at.desc()).offset(offset).limit(limit).all()
        
        # Convert to dictionaries
        data = [
            {
                "id": record.id,
                "verification_id": record.verification_id,
                "document_type": record.document_type,
                "confidence_score": float(record.confidence_score),
                "status": record.status,
                "detected_address": record.detected_address[:100] if record.detected_address else "Not detected",
                "created_at": record.created_at.isoformat()
            }
            for record in records
        ]
        
        return {
            "data": data,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving verification records: {str(e)}"
        )


@router.get("/verifications/{verification_id}")
async def get_verification_detail(
    verification_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific verification
    """
    
    if not DB_ENABLED:
        return {
            "error": "Running in demo mode - record not available",
            "verification_id": verification_id
        }
    
    try:
        record = db.query(VerificationLog).filter(
            VerificationLog.verification_id == verification_id
        ).first()
        
        if not record:
            raise HTTPException(status_code=404, detail="Verification record not found")
        
        return {
            "id": record.id,
            "verification_id": record.verification_id,
            "document_type": record.document_type,
            "confidence_score": float(record.confidence_score),
            "status": record.status,
            "detected_address": record.detected_address,
            "verification_details": record.verification_details,
            "face_verified": record.face_verified,
            "tampering_score": float(record.tampering_score),
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving verification detail: {str(e)}"
        )


@router.get("/verifications/stats/summary")
async def get_verification_stats(
    db: Session = Depends(get_db)
):
    """
    Get summary statistics of verifications
    """
    
    if not DB_ENABLED:
        return {
            "total": 0,
            "verified": 0,
            "review_required": 0,
            "failed": 0,
            "average_confidence": 0
        }
    
    try:
        total = db.query(VerificationLog).count()
        verified = db.query(VerificationLog).filter(
            VerificationLog.status == "Verified"
        ).count()
        review_required = db.query(VerificationLog).filter(
            VerificationLog.status == "Review Required"
        ).count()
        failed = db.query(VerificationLog).filter(
            VerificationLog.status == "Failed"
        ).count()
        
        # Calculate average confidence
        avg_confidence = db.query(VerificationLog).filter(
            VerificationLog.confidence_score > 0
        ).count()
        
        return {
            "total": total,
            "verified": verified,
            "review_required": review_required,
            "failed": failed,
            "verified_percentage": (verified / total * 100) if total > 0 else 0
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving statistics: {str(e)}"
        )

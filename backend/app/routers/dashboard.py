from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app import crud
from app.database import get_db

router = APIRouter()

@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get KYC dashboard statistics"""
    try:
        total_verifications = crud.count_verifications(db)
        verified_count = crud.count_verifications_by_status(db, "verified")
        flagged_count = crud.count_verifications_by_status(db, "flagged")
        pending_count = crud.count_verifications_by_status(db, "pending")
        alerts_count = crud.count_alerts(db)
        
        return {
            "total_verifications": total_verifications,
            "verified": verified_count,
            "flagged": flagged_count,
            "pending": pending_count,
            "alerts": alerts_count
        }
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        # Return default stats on error
        return {
            "total_verifications": 0,
            "verified": 0,
            "flagged": 0,
            "pending": 0,
            "alerts": 0
        }

@router.get("/verifications")
def read_verifications(limit: int = 100, db: Session = Depends(get_db)):
    """Get all verifications"""
    return {"verifications": crud.get_all_verifications(db, limit=limit)}

@router.get("/alerts")
def read_alerts(limit: int = 100, db: Session = Depends(get_db)):
    """Get all alerts"""
    return {"alerts": crud.get_all_alerts(db, limit=limit)}

@router.get("/documents")
def read_documents(limit: int = 100, db: Session = Depends(get_db)):
    """Get all documents"""
    return {"documents": crud.get_all_documents(db, limit=limit)}

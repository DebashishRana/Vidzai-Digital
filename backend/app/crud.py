"""
CRUD Operations for eKYC Database
Supports both MySQL and in-memory fallback
"""

from sqlalchemy.orm import Session
from app.models import Verification, Alert, Document, User
from app.database import DB_ENABLED, MEMORY_STORE
import uuid
from datetime import datetime


# =====================
# Verification CRUD
# =====================

def create_verification(db: Session, verification_data: dict):
    """Create a verification record"""
    ver_id = verification_data.get("id", f"VER{str(uuid.uuid4().hex[:6]).upper()}")
    
    if DB_ENABLED and db:
        try:
            verification = Verification(
                id=ver_id,
                user_id=verification_data.get("user_id", 1),
                document_type=verification_data.get("document_type", "Aadhar"),
                status=verification_data.get("status", "pending"),
                risk_score=float(verification_data.get("risk_score", 0)),
                confidence=float(verification_data.get("confidence", 0)),
                document_path=verification_data.get("document_path", "")
            )
            db.add(verification)
            db.commit()
            db.refresh(verification)
            return verification
        except Exception as e:
            print(f"DB create_verification error: {e}")
            db.rollback()
    
    # Fallback to memory
    ver_obj = {
        **verification_data,
        "id": ver_id,
        "created_at": datetime.utcnow().isoformat()
    }
    MEMORY_STORE["verifications"].append(ver_obj)
    return ver_obj


def get_all_verifications(db: Session, limit: int = 100):
    """Get all verifications sorted by date"""
    if DB_ENABLED and db:
        try:
            verifications = db.query(Verification).order_by(Verification.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": v.id,
                    "type": v.document_type,
                    "status": v.status,
                    "risk_score": v.risk_score,
                    "confidence": v.confidence,
                    "created_at": v.created_at.isoformat()
                }
                for v in verifications
            ]
        except Exception as e:
            print(f"DB get_all_verifications error: {e}")
    
    return MEMORY_STORE["verifications"][-limit:]


def count_verifications(db: Session):
    """Count total verifications"""
    if DB_ENABLED and db:
        try:
            return db.query(Verification).count()
        except:
            pass
    return len(MEMORY_STORE["verifications"])


def count_verifications_by_status(db: Session, status: str):
    """Count verifications by status"""
    if DB_ENABLED and db:
        try:
            return db.query(Verification).filter(Verification.status == status).count()
        except:
            pass
    return sum(1 for v in MEMORY_STORE["verifications"] if v.get("status") == status)


# =====================
# Alert CRUD
# =====================

def create_alert(db: Session, alert_data: dict):
    """Create an alert record"""
    alert_id = alert_data.get("id", f"ALT{str(uuid.uuid4().hex[:4]).upper()}")
    
    if DB_ENABLED and db:
        try:
            alert = Alert(
                id=alert_id,
                verification_id=alert_data.get("verification_id"),
                risk_level=alert_data.get("risk_level", "Medium"),
                alert_type=alert_data.get("alert_type", "Fraud Alert"),
                status=alert_data.get("status", "Active")
            )
            db.add(alert)
            db.commit()
            db.refresh(alert)
            return alert
        except Exception as e:
            print(f"DB create_alert error: {e}")
            db.rollback()
    
    # Fallback to memory
    alert_obj = {
        **alert_data,
        "id": alert_id,
        "created_at": datetime.utcnow().isoformat()
    }
    MEMORY_STORE["alerts"].append(alert_obj)
    return alert_obj


def get_all_alerts(db: Session, limit: int = 100):
    """Get all alerts sorted by date"""
    if DB_ENABLED and db:
        try:
            alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": a.id,
                    "type": a.alert_type,
                    "risk_level": a.risk_level,
                    "status": a.status,
                    "created_at": a.created_at.isoformat()
                }
                for a in alerts
            ]
        except Exception as e:
            print(f"DB get_all_alerts error: {e}")
    
    return MEMORY_STORE["alerts"][-limit:]


def count_alerts(db: Session):
    """Count total alerts"""
    if DB_ENABLED and db:
        try:
            return db.query(Alert).count()
        except:
            pass
    return len(MEMORY_STORE["alerts"])


# =====================
# Document CRUD
# =====================

def create_document(db: Session, document_data: dict):
    """Create a document record"""
    doc_id = document_data.get("id", f"DOC{str(uuid.uuid4().hex[:6]).upper()}")
    
    if DB_ENABLED and db:
        try:
            document = Document(
                id=doc_id,
                verification_id=document_data.get("verification_id"),
                file_path=document_data.get("file_path", ""),
                extracted_text=document_data.get("extracted_text", "")
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            return document
        except Exception as e:
            print(f"DB create_document error: {e}")
            db.rollback()
    
    # Fallback to memory
    doc_obj = {
        **document_data,
        "id": doc_id,
        "created_at": datetime.utcnow().isoformat()
    }
    MEMORY_STORE["documents"].append(doc_obj)
    return doc_obj


def get_all_documents(db: Session, limit: int = 100):
    """Get all documents sorted by date"""
    if DB_ENABLED and db:
        try:
            documents = db.query(Document).order_by(Document.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": d.id,
                    "verification_id": d.verification_id,
                    "extracted_text": d.extracted_text[:200] if d.extracted_text else "",
                    "created_at": d.created_at.isoformat()
                }
                for d in documents
            ]
        except Exception as e:
            print(f"DB get_all_documents error: {e}")
    
    return MEMORY_STORE["documents"][-limit:]

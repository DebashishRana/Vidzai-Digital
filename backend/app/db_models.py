from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
import datetime
from app.database import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255))
    role = Column(String(50), default="User")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    verifications = relationship("Verification", back_populates="user")
    documents = relationship("Document", back_populates="user")
    alerts = relationship("Alert", back_populates="user")

class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String(50), primary_key=True, index=True) # e.g., DOC001
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(String(50)) # Passport, Aadhaar Card, etc.
    storage_url = Column(String(500)) # GCS/MinIO URL
    status = Column(String(50), default="pending") # pending, verified, flagged
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="documents")
    verification = relationship("Verification", back_populates="document", uselist=False)

class Verification(Base):
    __tablename__ = "verifications"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String(50), primary_key=True, index=True) # e.g., KYC1023
    user_id = Column(Integer, ForeignKey("users.id"))
    document_id = Column(String(50), ForeignKey("documents.id"))
    status = Column(String(50), default="pending") # verified, flagged, pending
    risk_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="verifications")
    document = relationship("Document", back_populates="verification")

class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String(50), primary_key=True, index=True) # e.g., ALT001
    user_id = Column(Integer, ForeignKey("users.id"))
    risk_level = Column(String(50)) # High, Medium, Low
    alert_type = Column(String(100)) # Document Forgery, Face Mismatch
    status = Column(String(50), default="Active") # Active, Resolved
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="alerts")

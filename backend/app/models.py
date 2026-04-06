"""
SQLAlchemy Models for eKYC Database
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Verification(Base):
    __tablename__ = "verifications"
    
    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), default=1)
    document_type = Column(String(100), default="Aadhar")
    document_path = Column(String(255), nullable=True)
    status = Column(String(20), default="pending")  # verified, flagged, pending
    risk_score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String(50), primary_key=True, index=True)
    verification_id = Column(String(50), ForeignKey("verifications.id"))
    file_path = Column(String(255), nullable=True)
    extracted_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(String(50), primary_key=True, index=True)
    verification_id = Column(String(50), ForeignKey("verifications.id"), nullable=True)
    risk_level = Column(String(50), default="Medium")
    alert_type = Column(String(100), default="Fraud Alert")
    status = Column(String(20), default="Active")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class VerificationLog(Base):
    __tablename__ = "verification_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    verification_id = Column(String(255), unique=True, nullable=False, index=True)
    document_type = Column(String(100), nullable=True)
    confidence_score = Column(Float, default=0.0, index=True)
    status = Column(String(50), nullable=True, index=True)  # Verified, Review Required, Failed
    detected_address = Column(Text, nullable=True)
    verification_details = Column(JSON, nullable=True)  # Stores steps and scores
    face_verified = Column(String(20), default="Not Provided")
    tampering_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

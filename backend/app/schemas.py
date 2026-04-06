from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    name: str
    email: str
    role: Optional[str] = "User"

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class DocumentBase(BaseModel):
    type: str
    status: str
    storage_url: Optional[str] = None

class Document(DocumentBase):
    id: str
    user_id: int
    created_at: datetime
    class Config:
        from_attributes = True

class VerificationBase(BaseModel):
    status: str
    risk_score: float

class Verification(VerificationBase):
    id: str
    user_id: int
    document_id: str
    created_at: datetime
    class Config:
        from_attributes = True

class AlertBase(BaseModel):
    risk_level: str
    alert_type: str
    status: str

class Alert(AlertBase):
    id: str
    user_id: int
    created_at: datetime
    class Config:
        from_attributes = True

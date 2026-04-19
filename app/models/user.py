# backend/app/models/user.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum, Boolean
from app.core.database import Base

def gen_uuid(): return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    id            = Column(String, primary_key=True, default=gen_uuid)
    name          = Column(String(255), nullable=False)
    email         = Column(String(255), unique=True, nullable=False, index=True)
    phone         = Column(String(50), nullable=True)
    password_hash = Column(String(255), nullable=False)
    role          = Column(Enum("ELDERLY","CAREGIVER","DOCTOR", name="user_role"), nullable=False, default="ELDERLY")
    created_at    = Column(DateTime, default=datetime.utcnow)

class PatientLink(Base):
    """Links Caregiver or Doctor to an Elderly patient."""
    __tablename__ = "patient_links"
    id         = Column(String, primary_key=True, default=gen_uuid)
    patient_id = Column(String, nullable=False, index=True)  # ELDERLY user id
    linked_id  = Column(String, nullable=False, index=True)  # CAREGIVER or DOCTOR id
    role       = Column(String(20), nullable=False)           # "CAREGIVER" or "DOCTOR"
    relation   = Column(String(100), nullable=True)           # "parent","spouse","patient" etc
    created_at = Column(DateTime, default=datetime.utcnow)

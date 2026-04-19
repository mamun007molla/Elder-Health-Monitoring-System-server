# backend/app/models/physical.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base

def gen_uuid():
    return str(uuid.uuid4())

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id         = Column(String, primary_key=True, default=gen_uuid)
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
    logged_by  = Column(String, nullable=True)
    type       = Column(String(100), nullable=False)
    duration   = Column(Integer, nullable=True)
    notes      = Column(Text, nullable=True)
    logged_at  = Column(DateTime, default=datetime.utcnow)

class Routine(Base):
    __tablename__ = "routines"
    id           = Column(String, primary_key=True, default=gen_uuid)
    user_id      = Column(String, ForeignKey("users.id"), nullable=False)
    title        = Column(String(255), nullable=False)
    type         = Column(String(50), nullable=False)
    scheduled_at = Column(String(10), nullable=False)
    days         = Column(Text, nullable=False)
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

class MedicationVerifyLog(Base):
    __tablename__ = "medication_verify_logs"
    id                   = Column(String, primary_key=True, default=gen_uuid)
    user_id              = Column(String, ForeignKey("users.id"), nullable=False)
    prescribed_medication = Column(String(255), nullable=False)
    detected_medication  = Column(String(255), nullable=True)
    matched              = Column(Boolean, default=False)
    confidence           = Column(String(10), nullable=True)
    image_url            = Column(String(500), nullable=True)
    verified_at          = Column(DateTime, default=datetime.utcnow)

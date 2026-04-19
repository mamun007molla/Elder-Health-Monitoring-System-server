# backend/app/models/notification.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text
from app.core.database import Base

def gen_uuid(): return str(uuid.uuid4())

class Notification(Base):
    __tablename__ = "notifications"
    id         = Column(String, primary_key=True, default=gen_uuid)
    user_id    = Column(String, nullable=False, index=True)
    type       = Column(String(50), nullable=False)   # sos/medication/meal/routine/prescription/link/summary
    title      = Column(String(255), nullable=False)
    message    = Column(Text, nullable=False)
    is_read    = Column(Boolean, default=False)
    action_url = Column(String(500), nullable=True)   # frontend URL to navigate
    created_at = Column(DateTime, default=datetime.utcnow)

class SOSAlert(Base):
    __tablename__ = "sos_alerts"
    id           = Column(String, primary_key=True, default=gen_uuid)
    patient_id   = Column(String, nullable=False, index=True)
    triggered_by = Column(String, nullable=False)   # user id who pressed SOS
    location     = Column(String(500), nullable=True)
    message      = Column(Text, nullable=True)
    is_resolved  = Column(Boolean, default=False)
    resolved_at  = Column(DateTime, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

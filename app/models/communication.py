# backend/app/models/communication.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Text, Enum
from app.core.database import Base

def gen_uuid(): return str(uuid.uuid4())

class Message(Base):
    __tablename__ = "messages"
    id          = Column(String, primary_key=True, default=gen_uuid)
    sender_id   = Column(String, nullable=False, index=True)
    receiver_id = Column(String, nullable=False, index=True)
    content     = Column(Text, nullable=False)
    is_read     = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"
    id           = Column(String, primary_key=True, default=gen_uuid)
    patient_id   = Column(String, nullable=False, index=True)
    doctor_id    = Column(String, nullable=True)
    title        = Column(String(255), nullable=False)
    description  = Column(Text, nullable=True)
    scheduled_at = Column(DateTime, nullable=False)
    location     = Column(String(500), nullable=True)
    status       = Column(Enum("upcoming","completed","cancelled", name="appt_status"), default="upcoming")
    created_by   = Column(String, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"
    id           = Column(String, primary_key=True, default=gen_uuid)
    user_id      = Column(String, nullable=False, index=True)
    name         = Column(String(255), nullable=False)
    phone        = Column(String(50), nullable=False)
    relation     = Column(String(100), nullable=False)
    is_primary   = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow)

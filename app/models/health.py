# backend/app/models/health.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from app.core.database import Base

def gen_uuid(): return str(uuid.uuid4())


# ── Feature 1: Medication Reminder ───────────────────────────────────────────
class Medication(Base):
    __tablename__ = "medications"
    id           = Column(String, primary_key=True, default=gen_uuid)
    user_id      = Column(String, ForeignKey("users.id"), nullable=False)
    name         = Column(String(255), nullable=False)
    dosage       = Column(String(100), nullable=False)
    frequency    = Column(String(100), nullable=False)
    times        = Column(Text, nullable=False)       # JSON: ["08:00","20:00"]
    start_date   = Column(DateTime, nullable=False)
    end_date     = Column(DateTime, nullable=True)
    instructions = Column(Text, nullable=True)
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)


# ── Feature 2: Health History & Medical Documents ────────────────────────────
class HealthRecord(Base):
    __tablename__ = "health_records"
    id          = Column(String, primary_key=True, default=gen_uuid)
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    visit_date  = Column(DateTime, nullable=False)
    doctor_name = Column(String(255), nullable=True)
    diagnosis   = Column(Text, nullable=True)
    notes       = Column(Text, nullable=True)
    attachments = Column(Text, nullable=True)   # JSON: list of file paths
    created_at  = Column(DateTime, default=datetime.utcnow)


# ── Feature 3: Prescription Tracker ──────────────────────────────────────────
class Prescription(Base):
    __tablename__ = "prescriptions"
    id          = Column(String, primary_key=True, default=gen_uuid)
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    file_url    = Column(String(500), nullable=False)
    file_name   = Column(String(255), nullable=False)
    doctor_name = Column(String(255), nullable=True)
    issued_date = Column(DateTime, nullable=True)
    notes       = Column(Text, nullable=True)
    ai_summary  = Column(Text, nullable=True)   # AI-generated summary
    uploaded_at = Column(DateTime, default=datetime.utcnow)


# ── Feature 4: Meal & Nutrition Tracker ──────────────────────────────────────
class MealLog(Base):
    __tablename__ = "meal_logs"
    id          = Column(String, primary_key=True, default=gen_uuid)
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    meal_type   = Column(String(50), nullable=False)  # breakfast/lunch/dinner/snack
    description = Column(Text, nullable=False)
    calories    = Column(Integer, nullable=True)
    protein     = Column(Float, nullable=True)
    carbs       = Column(Float, nullable=True)
    fat         = Column(Float, nullable=True)
    logged_at   = Column(DateTime, default=datetime.utcnow)

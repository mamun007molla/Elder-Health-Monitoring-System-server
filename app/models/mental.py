# backend/app/models/mental.py
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, Text, Integer
from app.core.database import Base

def gen_uuid(): return str(uuid.uuid4())

class MoodEntry(Base):
    __tablename__ = "mood_entries"
    id         = Column(String, primary_key=True, default=gen_uuid)
    user_id    = Column(String, nullable=False, index=True)
    text       = Column(Text, nullable=False)        # what user typed/said
    mood       = Column(String(50), nullable=True)   # happy/sad/anxious/calm/angry
    sentiment  = Column(String(20), nullable=True)   # positive/negative/neutral
    score      = Column(Float, nullable=True)         # 0.0 - 1.0
    emotions   = Column(Text, nullable=True)          # JSON: ["happy","relieved"]
    summary    = Column(Text, nullable=True)          # AI summary
    logged_at  = Column(DateTime, default=datetime.utcnow)

class ChecklistItem(Base):
    __tablename__ = "checklist_items"
    id           = Column(String, primary_key=True, default=gen_uuid)
    user_id      = Column(String, nullable=False, index=True)
    title        = Column(String(255), nullable=False)
    category     = Column(String(50), nullable=False)  # medication/meal/exercise/other
    is_done      = Column(Boolean, default=False)
    done_at      = Column(DateTime, nullable=True)
    date         = Column(String(10), nullable=False)  # YYYY-MM-DD
    created_at   = Column(DateTime, default=datetime.utcnow)

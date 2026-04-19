from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ELDERLY   = "ELDERLY"
    CAREGIVER = "CAREGIVER"
    DOCTOR    = "DOCTOR"

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    password: str
    role: UserRole = UserRole.ELDERLY

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str; name: str; email: str; phone: Optional[str]; role: UserRole; created_at: datetime
    class Config: from_attributes = True

class TokenResponse(BaseModel):
    access_token: str; token_type: str = "bearer"; user: UserOut

# Module 1
class ActivityLogCreate(BaseModel):
    type: str; duration: Optional[int] = None; notes: Optional[str] = None; logged_at: Optional[datetime] = None

class ActivityLogOut(BaseModel):
    id: str; user_id: str; logged_by: Optional[str]; type: str
    duration: Optional[int]; notes: Optional[str]; logged_at: datetime
    class Config: from_attributes = True

class RoutineCreate(BaseModel):
    title: str; type: str; scheduled_at: str; days: List[str]; is_active: bool = True

class RoutineOut(BaseModel):
    id: str; user_id: str; title: str; type: str; scheduled_at: str
    days: List[str]; is_active: bool; created_at: datetime
    @field_validator("days", mode="before")
    @classmethod
    def parse_days(cls, v):
        import json
        return json.loads(v) if isinstance(v, str) else v
    class Config: from_attributes = True

class MedicationVerifyResult(BaseModel):
    matched: bool; confidence: float
    detected_medication: Optional[str] = None
    prescribed_medication: Optional[str] = None
    warnings: List[str] = []; raw_response: Optional[str] = None

class MedicationVerifyLogOut(BaseModel):
    id: str; user_id: str; prescribed_medication: str
    detected_medication: Optional[str]; matched: bool
    confidence: Optional[str]; image_url: Optional[str]; verified_at: datetime
    class Config: from_attributes = True

# Module 2
class MedicationCreate(BaseModel):
    name: str; dosage: str; frequency: str; times: List[str]
    start_date: datetime; end_date: Optional[datetime] = None; instructions: Optional[str] = None

class MedicationOut(BaseModel):
    id: str; user_id: str; name: str; dosage: str; frequency: str
    times: List[str]; start_date: datetime; end_date: Optional[datetime]
    instructions: Optional[str]; is_active: bool; created_at: datetime
    @field_validator("times", mode="before")
    @classmethod
    def parse_times(cls, v):
        import json
        return json.loads(v) if isinstance(v, str) else v
    class Config: from_attributes = True

class HealthRecordCreate(BaseModel):
    visit_date: datetime; doctor_name: Optional[str] = None
    diagnosis: Optional[str] = None; notes: Optional[str] = None

class HealthRecordOut(BaseModel):
    id: str; user_id: str; visit_date: datetime; doctor_name: Optional[str]
    diagnosis: Optional[str]; notes: Optional[str]; attachments: Optional[List[str]]; created_at: datetime
    @field_validator("attachments", mode="before")
    @classmethod
    def parse_attachments(cls, v):
        import json
        if isinstance(v, str):
            try: return json.loads(v)
            except: return []
        return v or []
    class Config: from_attributes = True

class PrescriptionOut(BaseModel):
    id: str; user_id: str; file_url: str; file_name: str
    doctor_name: Optional[str]; issued_date: Optional[datetime]
    notes: Optional[str]; ai_summary: Optional[str]; uploaded_at: datetime
    class Config: from_attributes = True

class ReportSummaryResult(BaseModel):
    key_findings: List[str] = []; summary: str
    medications_mentioned: List[str] = []; follow_up_needed: bool = False

class MealLogCreate(BaseModel):
    meal_type: str; description: str; calories: Optional[int] = None
    protein: Optional[float] = None; carbs: Optional[float] = None
    fat: Optional[float] = None; logged_at: Optional[datetime] = None

class MealLogOut(BaseModel):
    id: str; user_id: str; meal_type: str; description: str
    calories: Optional[int]; protein: Optional[float]; carbs: Optional[float]
    fat: Optional[float]; logged_at: datetime
    class Config: from_attributes = True

class FoodAnalysisResult(BaseModel):
    food_name: str; ingredients: List[str] = []; calories: int = 0
    protein: float = 0; carbs: float = 0; fat: float = 0; fiber: float = 0
    serving_size: str = "1 serving"; meal_type_suggestion: str = "meal"; health_notes: str = ""

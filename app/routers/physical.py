# backend/app/routers/physical.py
import json, uuid, os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.physical import ActivityLog, Routine, MedicationVerifyLog
from app.schemas import (
    ActivityLogCreate, ActivityLogOut,
    RoutineCreate, RoutineOut,
    MedicationVerifyResult, MedicationVerifyLogOut,
)

router = APIRouter(prefix="/physical", tags=["Module 1 — Physical Monitoring"])


# ── Activity Log ──────────────────────────────────────────────────────────────
@router.get("/activities", response_model=list[ActivityLogOut])
def get_activities(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    return db.query(ActivityLog).filter(ActivityLog.user_id == cu.id).order_by(ActivityLog.logged_at.desc()).all()

@router.post("/activities", response_model=ActivityLogOut, status_code=201)
def log_activity(body: ActivityLogCreate, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    if cu.role not in ("ELDERLY",):
        raise HTTPException(403, "Only elderly patients can log their own activities")
    log = ActivityLog(
        id=str(uuid.uuid4()), user_id=cu.id, type=body.type,
        duration=body.duration, notes=body.notes,
        logged_at=body.logged_at or datetime.utcnow(),
    )
    db.add(log); db.commit(); db.refresh(log)
    return log

@router.delete("/activities/{activity_id}", status_code=204)
def delete_activity(activity_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    log = db.query(ActivityLog).filter(ActivityLog.id == activity_id, ActivityLog.user_id == cu.id).first()
    if not log: raise HTTPException(404, "Not found")
    db.delete(log); db.commit()

@router.get("/activities/stats", response_model=dict)
def activity_stats(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    logs = db.query(ActivityLog).filter(ActivityLog.user_id == cu.id).all()
    today = datetime.utcnow().date()
    today_logs = [l for l in logs if l.logged_at.date() == today]
    by_type = {}
    for l in logs:
        by_type[l.type] = by_type.get(l.type, 0) + 1
    return {"total":len(logs),"today":len(today_logs),"by_type":by_type,
            "total_duration_min":sum(l.duration or 0 for l in logs)}


# ── Routine Schedule ──────────────────────────────────────────────────────────
@router.get("/routines", response_model=list[RoutineOut])
def get_routines(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    return db.query(Routine).filter(Routine.user_id == cu.id, Routine.is_active == True).all()

@router.post("/routines", response_model=RoutineOut, status_code=201)
def create_routine(body: RoutineCreate, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    routine = Routine(
        id=str(uuid.uuid4()), user_id=cu.id, title=body.title,
        type=body.type, scheduled_at=body.scheduled_at,
        days=json.dumps(body.days), is_active=body.is_active,
    )
    db.add(routine); db.commit(); db.refresh(routine)
    return routine

@router.put("/routines/{routine_id}", response_model=RoutineOut)
def update_routine(routine_id: str, body: RoutineCreate, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    routine = db.query(Routine).filter(Routine.id == routine_id, Routine.user_id == cu.id).first()
    if not routine: raise HTTPException(404, "Not found")
    routine.title=body.title; routine.type=body.type
    routine.scheduled_at=body.scheduled_at; routine.days=json.dumps(body.days)
    routine.is_active=body.is_active
    db.commit(); db.refresh(routine)
    return routine

@router.delete("/routines/{routine_id}", status_code=204)
def delete_routine(routine_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    routine = db.query(Routine).filter(Routine.id == routine_id, Routine.user_id == cu.id).first()
    if not routine: raise HTTPException(404, "Not found")
    db.delete(routine); db.commit()


# ── Medication Verification (Ollama medgemma) ─────────────────────────────────
@router.post("/verify-medication", response_model=MedicationVerifyResult)
async def verify_medication(
    image: UploadFile = File(...),
    prescribed_medication: str = Form(...),
    save_log: bool = Form(True),
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    from app.ai.medication_verify import verify_medication_image
    from app.utils.file_upload import read_upload_bytes
    import aiofiles

    img_bytes, mime_type = await read_upload_bytes(image)
    result = await verify_medication_image(img_bytes, prescribed_medication, mime_type)

    if save_log:
        try:
            upload_dir = "./uploads/medications"
            os.makedirs(upload_dir, exist_ok=True)
            fname = f"{uuid.uuid4().hex}.jpg"
            async with aiofiles.open(f"{upload_dir}/{fname}", "wb") as f:
                await f.write(img_bytes)
            file_url = f"/uploads/medications/{fname}"
        except Exception:
            file_url = None

        log = MedicationVerifyLog(
            id=str(uuid.uuid4()), user_id=cu.id,
            prescribed_medication=prescribed_medication,
            detected_medication=result.detected_medication,
            matched=result.matched,
            confidence=str(round(result.confidence, 2)),
            image_url=file_url,
        )
        db.add(log); db.commit()
    return result

@router.get("/verify-medication/history", response_model=list[MedicationVerifyLogOut])
def get_verify_history(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    return db.query(MedicationVerifyLog).filter(
        MedicationVerifyLog.user_id == cu.id
    ).order_by(MedicationVerifyLog.verified_at.desc()).limit(20).all()


# ── Activity Image Analysis (Ollama medgemma) ─────────────────────────────────
@router.post("/analyze-image", response_model=dict)
async def analyze_activity_image(
    image: UploadFile = File(...),
    question: str = Form("Analyze the person's posture and activity level. Is there any abnormality?"),
    cu: User = Depends(get_current_user),
):
    from app.ai.groq_vision import analyze_activity_image as ai_analyze
    from app.utils.file_upload import read_upload_bytes
    img_bytes, mime_type = await read_upload_bytes(image)
    result = await ai_analyze(img_bytes, question)
    return {"analysis": result, "question": question, "model": "Groq llama-4-maverick"}


# ── Fall Detection (XGBoost + AST) ───────────────────────────────────────────
@router.post("/detect-fall", response_model=dict)
async def detect_fall(
    video: UploadFile = File(...),
    cu: User = Depends(get_current_user),
):
    from app.ai.fall_detection.detector import run_fall_detection
    from app.core.config import settings
    content = await video.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "Video too large (max 100MB)")
    try:
        result = await run_fall_detection(content, settings.UPLOAD_DIR)
        return result
    except Exception as e:
        raise HTTPException(500, f"Fall detection error: {str(e)}")


# ── Linked Patient Activities (Caregiver/Doctor) ──────────────────────────────
@router.get("/activities/patient/{patient_id}", response_model=list[dict])
def get_patient_activities(
    patient_id: str,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Caregiver/Doctor gets linked patient's activities."""
    from app.models.user import PatientLink
    if cu.role == "ELDERLY":
        raise HTTPException(403, "Not allowed")
    link = db.query(PatientLink).filter(
        PatientLink.linked_id == cu.id,
        PatientLink.patient_id == patient_id,
    ).first()
    if not link:
        raise HTTPException(403, "Not linked to this patient")
    logs = db.query(ActivityLog).filter(
        ActivityLog.user_id == patient_id
    ).order_by(ActivityLog.logged_at.desc()).all()
    return [{"id":l.id,"user_id":l.user_id,"logged_by":l.logged_by,"type":l.type,
             "duration":l.duration,"notes":l.notes,"logged_at":l.logged_at.isoformat()} for l in logs]

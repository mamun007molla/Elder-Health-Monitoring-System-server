# backend/app/routers/caregiver.py
"""
Caregiver Actions — যখন Elder নিজে করতে পারবে না।
Caregiver elder এর হয়ে কাজ করতে পারবে।
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User, PatientLink
from app.models.health import MealLog, Medication
from app.models.physical import ActivityLog
from app.models.notification import Notification
from app.schemas import ActivityLogCreate, MealLogCreate

router = APIRouter(prefix="/caregiver", tags=["Caregiver Actions"])


def verify_caregiver_link(db: Session, caregiver_id: str, patient_id: str):
    """Verify caregiver is linked to this patient."""
    link = db.query(PatientLink).filter(
        PatientLink.linked_id == caregiver_id,
        PatientLink.patient_id == patient_id,
    ).first()
    if not link:
        raise HTTPException(403, "You are not linked to this patient")
    return link


# ── Log activity ON BEHALF of elder ──────────────────────────────────────────
@router.post("/log-activity/{patient_id}", response_model=dict)
def log_activity_for_patient(
    patient_id: str,
    body: ActivityLogCreate,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Caregiver logs an activity for their linked elderly patient."""
    if cu.role not in ("CAREGIVER", "DOCTOR"):
        raise HTTPException(403, "Only caregivers can log on behalf of patients")
    verify_caregiver_link(db, cu.id, patient_id)

    log = ActivityLog(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        logged_by=cu.name,   # shows who logged it
        type=body.type,
        duration=body.duration,
        notes=f"[Logged by {cu.name}] {body.notes or ''}".strip(),
        logged_at=body.logged_at or datetime.utcnow(),
    )
    db.add(log)
    # Notify patient
    db.add(Notification(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        type="activity",
        title=f"📋 Activity logged by {cu.name}",
        message=f"{cu.name} logged '{body.type}' activity on your behalf.",
    ))
    db.commit()
    return {"message": f"Activity '{body.type}' logged for patient", "logged_by": cu.name}


# ── Log meal ON BEHALF of elder ───────────────────────────────────────────────
@router.post("/log-meal/{patient_id}", response_model=dict)
def log_meal_for_patient(
    patient_id: str,
    body: MealLogCreate,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Caregiver logs a meal for their linked elderly patient."""
    if cu.role not in ("CAREGIVER", "DOCTOR"):
        raise HTTPException(403, "Only caregivers can log meals on behalf of patients")
    verify_caregiver_link(db, cu.id, patient_id)

    meal = MealLog(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        meal_type=body.meal_type,
        description=f"[By {cu.name}] {body.description}",
        calories=body.calories,
        protein=body.protein,
        carbs=body.carbs,
        fat=body.fat,
        logged_at=body.logged_at or datetime.utcnow(),
    )
    db.add(meal)
    db.add(Notification(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        type="meal",
        title=f"🍽️ Meal logged by {cu.name}",
        message=f"{cu.name} logged '{body.meal_type}' for you: {body.description[:50]}",
    ))
    db.commit()
    return {"message": f"Meal logged for patient by {cu.name}"}


# ── Add observation note ──────────────────────────────────────────────────────
@router.post("/add-note/{patient_id}", response_model=dict)
def add_observation(
    patient_id: str,
    note: str,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Caregiver/Doctor adds an observation note for the patient."""
    if cu.role not in ("CAREGIVER", "DOCTOR"):
        raise HTTPException(403, "Forbidden")
    verify_caregiver_link(db, cu.id, patient_id)

    # Save as activity log with type "observation"
    log = ActivityLog(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        logged_by=cu.name,
        type="observation",
        notes=f"[{cu.role}] {note}",
        logged_at=datetime.utcnow(),
    )
    db.add(log)
    # Notify patient
    db.add(Notification(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        type="activity",
        title=f"📝 Note from {cu.name}",
        message=note[:200],
    ))
    db.commit()
    return {"message": "Observation added"}


# ── Trigger SOS on behalf of elder ───────────────────────────────────────────
@router.post("/sos/{patient_id}", response_model=dict)
async def trigger_sos_for_patient(
    patient_id: str,
    message: str = "Patient needs immediate help!",
    location: str = None,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Caregiver triggers SOS on behalf of elderly patient."""
    if cu.role not in ("CAREGIVER", "DOCTOR"):
        raise HTTPException(403, "Forbidden")
    verify_caregiver_link(db, cu.id, patient_id)

    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")

    from app.models.notification import SOSAlert
    from app.ai.sms_service import send_sos_alert

    sos = SOSAlert(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        triggered_by=cu.id,
        location=location,
        message=f"[Triggered by {cu.name}] {message}",
    )
    db.add(sos)

    # Notify all other linked caregivers/doctors
    all_links = db.query(PatientLink).filter(PatientLink.patient_id == patient_id).all()
    caregivers = []
    for link in all_links:
        if link.linked_id != cu.id:
            linked_user = db.query(User).filter(User.id == link.linked_id).first()
            if linked_user:
                db.add(Notification(
                    id=str(uuid.uuid4()),
                    user_id=linked_user.id,
                    type="sos",
                    title=f"🚨 SOS for {patient.name}",
                    message=f"{cu.name} triggered SOS for {patient.name}. {message}",
                ))
                caregivers.append({"phone": linked_user.phone, "name": linked_user.name})

    # Also notify the patient
    db.add(Notification(
        id=str(uuid.uuid4()),
        user_id=patient_id,
        type="sos",
        title="🚨 SOS Alert Sent",
        message=f"{cu.name} sent an emergency alert on your behalf.",
    ))
    db.commit()

    sms_count = await send_sos_alert(patient.name, caregivers)
    return {"message": "SOS sent", "notified": len(caregivers), "sms": sms_count}


# ── Get patient overview ──────────────────────────────────────────────────────
@router.get("/patient-overview/{patient_id}", response_model=dict)
def get_patient_overview(
    patient_id: str,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Get a quick overview of linked patient's recent data."""
    if cu.role not in ("CAREGIVER", "DOCTOR"):
        raise HTTPException(403, "Forbidden")
    verify_caregiver_link(db, cu.id, patient_id)

    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")

    today = datetime.utcnow().date()

    recent_activities = db.query(ActivityLog).filter(
        ActivityLog.user_id == patient_id
    ).order_by(ActivityLog.logged_at.desc()).limit(5).all()

    today_meals = db.query(MealLog).filter(
        MealLog.user_id == patient_id,
        MealLog.logged_at >= datetime.combine(today, datetime.min.time()),
    ).all()

    active_meds = db.query(Medication).filter(
        Medication.user_id == patient_id,
        Medication.is_active == True,
    ).all()

    return {
        "patient": {"id": patient.id, "name": patient.name, "email": patient.email},
        "today_meals": len(today_meals),
        "today_calories": sum(m.calories or 0 for m in today_meals),
        "recent_activities": [
            {"type": a.type, "duration": a.duration, "logged_by": a.logged_by,
             "logged_at": a.logged_at.isoformat()}
            for a in recent_activities
        ],
        "active_medications": len(active_meds),
        "medication_names": [m.name for m in active_meds],
    }

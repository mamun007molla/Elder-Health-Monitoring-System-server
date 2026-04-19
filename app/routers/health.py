# backend/app/routers/health.py
import json, uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User, PatientLink
from app.models.health import Medication, HealthRecord, Prescription, MealLog
from app.schemas import (
    MedicationCreate,
    MedicationOut,
    HealthRecordCreate,
    HealthRecordOut,
    PrescriptionOut,
    ReportSummaryResult,
    MealLogCreate,
    MealLogOut,
)

router = APIRouter(prefix="/health", tags=["Module 2 — Health Management"])


def get_linked_patient_ids(db: Session, user_id: str) -> list[str]:
    links = db.query(PatientLink).filter(PatientLink.linked_id == user_id).all()
    return [l.patient_id for l in links]


# ══════════════════════════════════════════════════════════════════════════════
# Feature 1 — Medication Reminder
# ELDERLY: add+view own | DOCTOR: view linked | CAREGIVER: view linked
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/medications", response_model=list[MedicationOut])
def get_medications(
    db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role == "ELDERLY":
        return (
            db.query(Medication)
            .filter(Medication.user_id == cu.id, Medication.is_active == True)
            .all()
        )
    patient_ids = get_linked_patient_ids(db, cu.id)
    return (
        db.query(Medication)
        .filter(Medication.user_id.in_(patient_ids), Medication.is_active == True)
        .all()
    )


@router.post("/medications", response_model=MedicationOut, status_code=201)
def create_medication(
    body: MedicationCreate,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    if cu.role != "ELDERLY":
        raise HTTPException(403, "Only patients can add their own medications")
    med = Medication(
        id=str(uuid.uuid4()),
        user_id=cu.id,
        name=body.name,
        dosage=body.dosage,
        frequency=body.frequency,
        times=json.dumps(body.times),
        start_date=body.start_date,
        end_date=body.end_date,
        instructions=body.instructions,
    )
    db.add(med)
    db.commit()
    db.refresh(med)
    return med


@router.delete("/medications/{med_id}", status_code=204)
def delete_medication(
    med_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role != "ELDERLY":
        raise HTTPException(403, "Only patients can delete their medications")
    med = (
        db.query(Medication)
        .filter(Medication.id == med_id, Medication.user_id == cu.id)
        .first()
    )
    if not med:
        raise HTTPException(404, "Not found")
    med.is_active = False
    db.commit()


@router.get("/medications/stats", response_model=dict)
def medication_stats(
    db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    meds = (
        db.query(Medication)
        .filter(Medication.user_id == cu.id, Medication.is_active == True)
        .all()
    )
    return {
        "active_count": len(meds),
        "daily_doses": sum(len(json.loads(m.times)) for m in meds),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Feature 2 — Health History
# DOCTOR: add+view linked patients | ELDERLY: view own | CAREGIVER: view linked
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/records", response_model=list[HealthRecordOut])
def get_health_records(
    db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role == "ELDERLY":
        return (
            db.query(HealthRecord)
            .filter(HealthRecord.user_id == cu.id)
            .order_by(HealthRecord.visit_date.desc())
            .all()
        )
    patient_ids = get_linked_patient_ids(db, cu.id)
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.user_id.in_(patient_ids))
        .order_by(HealthRecord.visit_date.desc())
        .all()
    )


@router.post("/records", response_model=HealthRecordOut, status_code=201)
def create_health_record(
    body: HealthRecordCreate,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    # Only DOCTOR can add health records
    if cu.role != "DOCTOR":
        raise HTTPException(403, "Only doctors can add health records")
    # Doctor must be linked to a patient — add for first linked patient or specify
    patient_ids = get_linked_patient_ids(db, cu.id)
    if not patient_ids:
        raise HTTPException(
            400, "You have no linked patients. Link a patient from Settings first."
        )
    # Add record for the patient (use patient_id from body or first linked)
    record = HealthRecord(
        id=str(uuid.uuid4()),
        user_id=patient_ids[0],  # first linked patient
        visit_date=body.visit_date,
        doctor_name=cu.name,  # auto-fill with doctor's name
        diagnosis=body.diagnosis,
        notes=body.notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    # Notify the patient
    from app.models.notification import Notification

    db.add(
        Notification(
            id=str(uuid.uuid4()),
            user_id=patient_ids[0],
            type="prescription",
            title=f"📋 New Health Record Added",
            message=f"Dr. {cu.name} added a new health record: {body.diagnosis or 'Visit notes'}",
        )
    )
    db.commit()
    return record


@router.delete("/records/{record_id}", status_code=204)
def delete_health_record(
    record_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role != "DOCTOR":
        raise HTTPException(403, "Only doctors can delete health records")
    r = db.query(HealthRecord).filter(HealthRecord.id == record_id).first()
    if not r:
        raise HTTPException(404, "Not found")
    db.delete(r)
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Feature 3 — Prescription Tracker
# DOCTOR: upload+summarize | ALL: view
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/prescriptions", response_model=list[PrescriptionOut])
def get_prescriptions(
    db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role == "ELDERLY":
        return (
            db.query(Prescription)
            .filter(Prescription.user_id == cu.id)
            .order_by(Prescription.uploaded_at.desc())
            .all()
        )
    patient_ids = get_linked_patient_ids(db, cu.id)
    return (
        db.query(Prescription)
        .filter(Prescription.user_id.in_(patient_ids))
        .order_by(Prescription.uploaded_at.desc())
        .all()
    )


@router.post("/prescriptions/upload", response_model=PrescriptionOut, status_code=201)
async def upload_prescription(
    file: UploadFile = File(...),
    doctor_name: str = Form(None),
    issued_date: str = Form(None),
    notes: str = Form(None),
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    if cu.role != "DOCTOR":
        raise HTTPException(403, "Only doctors can upload prescriptions")
    patient_ids = get_linked_patient_ids(db, cu.id)
    if not patient_ids:
        raise HTTPException(
            400, "No linked patients. Link a patient from Settings first."
        )
    from app.utils.file_upload import save_upload

    file_url, _ = await save_upload(file, "prescriptions")
    p = Prescription(
        id=str(uuid.uuid4()),
        user_id=patient_ids[0],
        file_url=file_url,
        file_name=file.filename or "prescription",
        doctor_name=cu.name,
        issued_date=datetime.fromisoformat(issued_date) if issued_date else None,
        notes=notes,
    )
    db.add(p)
    # Notify patient and caregivers
    from app.models.notification import Notification

    for pid in patient_ids:
        db.add(
            Notification(
                id=str(uuid.uuid4()),
                user_id=pid,
                type="prescription",
                title=f"📄 New Prescription from Dr. {cu.name}",
                message=f"Dr. {cu.name} uploaded a new prescription. Check your prescriptions.",
                action_url="/health/prescriptions",
            )
        )
    db.commit()
    db.refresh(p)
    return p


@router.post("/prescriptions/{pres_id}/summarize", response_model=ReportSummaryResult)
async def summarize_prescription(
    pres_id: str,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    if cu.role != "DOCTOR":
        raise HTTPException(403, "Only doctors can summarize prescriptions")
    from app.ai.report_summarizer import summarize_report
    from app.core.config import settings
    import os, aiofiles

    p = db.query(Prescription).filter(Prescription.id == pres_id).first()
    if not p:
        raise HTTPException(404, "Not found")

    text = ""

    # Try to extract text from uploaded image using OpenRouter vision
    if p.file_url:
        try:
            file_path = "." + p.file_url
            if os.path.exists(file_path):
                async with aiofiles.open(file_path, "rb") as f:
                    img_bytes = await f.read()
                from app.ai.groq_vision import extract_text_from_image

                extracted = await extract_text_from_image(img_bytes)
                if extracted and len(extracted.strip()) > 20:
                    text = extracted
        except Exception as e:
            print(f"[Prescription] Image extraction error: {e}")

    # Fallback to notes
    if not text:
        text = (
            p.notes
            or f"Prescription uploaded by Dr. {p.doctor_name or cu.name} on {p.issued_date}"
        )

    result = await summarize_report(text)
    p.ai_summary = result.summary
    db.commit()
    return result


@router.post("/reports/summarize-text", response_model=ReportSummaryResult)
async def summarize_text(
    report_text: str = Form(None),
    file: UploadFile = File(None),
    cu: User = Depends(get_current_user),
):
    """Summarize text or image — all roles allowed. Uses OCR for images."""
    from app.ai.report_summarizer import summarize_report

    text = ""

    # Image upload — extract text via OCR (easyocr)
    if file:
        try:
            from app.ai.ocr_extract import extract_text_from_image  # local OCR
            from app.utils.file_upload import read_upload_bytes

            img_bytes, _ = await read_upload_bytes(file)
            text = await extract_text_from_image(img_bytes)
            if not text or len(text.strip()) < 10:
                raise HTTPException(
                    400,
                    "Could not extract text from image. Make sure the image is clear and contains readable text.",
                )
        except HTTPException:
            raise
        except ImportError:
            raise HTTPException(503, "easyocr not installed. Run: pip install easyocr")
        except Exception as e:
            raise HTTPException(500, f"OCR error: {str(e)}")

    # Text input
    if not text and report_text:
        text = report_text

    if not text:
        raise HTTPException(400, "Provide report_text or upload an image")

    return await summarize_report(text)


@router.delete("/prescriptions/{pres_id}", status_code=204)
def delete_prescription(
    pres_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role != "DOCTOR":
        raise HTTPException(403, "Only doctors can delete prescriptions")
    p = db.query(Prescription).filter(Prescription.id == pres_id).first()
    if not p:
        raise HTTPException(404, "Not found")
    db.delete(p)
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Feature 4 — Meal & Nutrition Tracker
# ELDERLY: add+view own | CAREGIVER+DOCTOR: view linked
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/meals", response_model=list[MealLogOut])
def get_meals(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    if cu.role == "ELDERLY":
        return (
            db.query(MealLog)
            .filter(MealLog.user_id == cu.id)
            .order_by(MealLog.logged_at.desc())
            .limit(50)
            .all()
        )
    patient_ids = get_linked_patient_ids(db, cu.id)
    return (
        db.query(MealLog)
        .filter(MealLog.user_id.in_(patient_ids))
        .order_by(MealLog.logged_at.desc())
        .limit(50)
        .all()
    )


@router.post("/meals", response_model=MealLogOut, status_code=201)
def log_meal(
    body: MealLogCreate,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    if cu.role != "ELDERLY":
        raise HTTPException(403, "Only patients can log their meals")
    meal = MealLog(
        id=str(uuid.uuid4()),
        user_id=cu.id,
        meal_type=body.meal_type,
        description=body.description,
        calories=body.calories,
        protein=body.protein,
        carbs=body.carbs,
        fat=body.fat,
        logged_at=body.logged_at or datetime.utcnow(),
    )
    db.add(meal)
    db.commit()
    db.refresh(meal)
    return meal


@router.delete("/meals/{meal_id}", status_code=204)
def delete_meal(
    meal_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    if cu.role != "ELDERLY":
        raise HTTPException(403, "Only patients can delete their meals")
    m = (
        db.query(MealLog)
        .filter(MealLog.id == meal_id, MealLog.user_id == cu.id)
        .first()
    )
    if not m:
        raise HTTPException(404, "Not found")
    db.delete(m)
    db.commit()


@router.get("/meals/today-stats", response_model=dict)
def today_nutrition_stats(
    db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    from datetime import date

    today_start = datetime.combine(date.today(), datetime.min.time())
    meals = (
        db.query(MealLog)
        .filter(MealLog.user_id == cu.id, MealLog.logged_at >= today_start)
        .all()
    )
    return {
        "total_meals": len(meals),
        "calories": sum(m.calories or 0 for m in meals),
        "protein": sum(m.protein or 0 for m in meals),
        "carbs": sum(m.carbs or 0 for m in meals),
        "fat": sum(m.fat or 0 for m in meals),
    }


# ── Feature 4 Extended: AI Food Image Analysis ────────────────────────────────
@router.post("/meals/analyze-image", response_model=dict)
async def analyze_food_image(
    image: UploadFile = File(...),
    cu: User = Depends(get_current_user),
):
    if cu.role != "ELDERLY":
        raise HTTPException(403, "Only patients can analyze food images")
    from app.ai.groq_vision import analyze_food_image as ai_analyze
    from app.utils.file_upload import read_upload_bytes

    img_bytes, mime_type = await read_upload_bytes(image)
    return await ai_analyze(img_bytes, mime_type)

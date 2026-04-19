# backend/app/routers/links.py
"""Patient Linking System — Connect Caregiver/Doctor to Elderly patient."""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User, PatientLink

router = APIRouter(prefix="/links", tags=["Patient Links"])


@router.post("/link-by-email", response_model=dict)
def link_by_email(
    email: str,
    relation: str = "family",
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    if cu.role not in ("CAREGIVER", "DOCTOR"):
        raise HTTPException(403, "Only CAREGIVER or DOCTOR can link to patients")

    patient = db.query(User).filter(User.email == email.strip().lower(), User.role == "ELDERLY").first()
    if not patient:
        raise HTTPException(404, "No elderly patient found with this email")

    existing = db.query(PatientLink).filter(
        PatientLink.patient_id == patient.id,
        PatientLink.linked_id == cu.id
    ).first()
    if existing:
        raise HTTPException(400, f"Already linked to {patient.name}")

    link = PatientLink(
        id=str(uuid.uuid4()),
        patient_id=patient.id,
        linked_id=cu.id,
        role=cu.role,
        relation=relation,
    )
    db.add(link)

    # Notify the patient
    from app.models.notification import Notification
    n = Notification(
        id=str(uuid.uuid4()),
        user_id=patient.id,
        type="link",
        title=f"🔗 New {'Caregiver' if cu.role=='CAREGIVER' else 'Doctor'} Linked",
        message=f"{cu.name} ({cu.role.lower()}) is now monitoring your health data.",
    )
    db.add(n)
    db.commit()

    return {"message": f"Linked to {patient.name}", "patient_id": patient.id, "patient_name": patient.name}


@router.get("/my-patients", response_model=list[dict])
def get_my_patients(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    links = db.query(PatientLink).filter(PatientLink.linked_id == cu.id).all()
    result = []
    for l in links:
        p = db.query(User).filter(User.id == l.patient_id).first()
        if p:
            result.append({"id": p.id, "name": p.name, "email": p.email, "phone": p.phone, "relation": l.relation})
    return result


@router.get("/my-caregivers", response_model=list[dict])
def get_my_caregivers(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    links = db.query(PatientLink).filter(PatientLink.patient_id == cu.id).all()
    result = []
    for l in links:
        u = db.query(User).filter(User.id == l.linked_id).first()
        if u:
            result.append({"id": u.id, "name": u.name, "email": u.email, "role": u.role, "relation": l.relation})
    return result


@router.delete("/unlink/{patient_id}", status_code=204)
def unlink(patient_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    link = db.query(PatientLink).filter(
        PatientLink.patient_id == patient_id,
        PatientLink.linked_id == cu.id
    ).first()
    if not link: raise HTTPException(404, "Link not found")
    db.delete(link)
    db.commit()

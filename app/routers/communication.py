# backend/app/routers/communication.py
"""Module 3 — Emergency & Communication"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User, PatientLink
from app.models.communication import Message, Appointment, EmergencyContact
from app.models.notification import Notification
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/communication", tags=["Module 3 — Communication"])


# ══════════════════════════════════════════════════════════════════════════════
# Feature 1 — Messaging (Elder ↔ Caregiver ↔ Doctor)
# ══════════════════════════════════════════════════════════════════════════════
class MessageCreate(BaseModel):
    receiver_id: str
    content: str

@router.get("/messages/{other_user_id}", response_model=list[dict])
def get_messages(other_user_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    msgs = db.query(Message).filter(
        ((Message.sender_id == cu.id) & (Message.receiver_id == other_user_id)) |
        ((Message.sender_id == other_user_id) & (Message.receiver_id == cu.id))
    ).order_by(Message.created_at.asc()).all()
    # Mark as read
    for m in msgs:
        if m.receiver_id == cu.id and not m.is_read:
            m.is_read = True
    db.commit()
    return [{"id":m.id,"sender_id":m.sender_id,"receiver_id":m.receiver_id,
             "content":m.content,"is_read":m.is_read,"created_at":m.created_at.isoformat()} for m in msgs]

@router.post("/messages", response_model=dict, status_code=201)
def send_message(body: MessageCreate, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    receiver = db.query(User).filter(User.id == body.receiver_id).first()
    if not receiver:
        raise HTTPException(404, "User not found")
    msg = Message(id=str(uuid.uuid4()), sender_id=cu.id, receiver_id=body.receiver_id, content=body.content)
    db.add(msg)
    db.add(Notification(
        id=str(uuid.uuid4()), user_id=body.receiver_id, type="message",
        title=f"💬 Message from {cu.name}",
        message=body.content[:100],
        action_url="/communication/messages",
    ))
    db.commit()
    return {"id":msg.id,"content":msg.content,"created_at":msg.created_at.isoformat()}

@router.get("/contacts", response_model=list[dict])
def get_contacts(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    """Get all users I can message."""
    if cu.role == "ELDERLY":
        links = db.query(PatientLink).filter(PatientLink.patient_id == cu.id).all()
        user_ids = [l.linked_id for l in links]
    else:
        links = db.query(PatientLink).filter(PatientLink.linked_id == cu.id).all()
        patient_ids = [l.patient_id for l in links]
        # Also get other caregivers/doctors of same patients
        all_links = db.query(PatientLink).filter(PatientLink.patient_id.in_(patient_ids)).all()
        user_ids = list(set([l.patient_id for l in links] + [l.linked_id for l in all_links if l.linked_id != cu.id]))

    contacts = []
    for uid in user_ids:
        u = db.query(User).filter(User.id == uid).first()
        if u:
            unread = db.query(Message).filter(Message.sender_id == uid, Message.receiver_id == cu.id, Message.is_read == False).count()
            contacts.append({"id":u.id,"name":u.name,"role":u.role,"email":u.email,"unread":unread})
    return contacts


# ══════════════════════════════════════════════════════════════════════════════
# Feature 2 — Appointment Management
# ══════════════════════════════════════════════════════════════════════════════
class AppointmentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    scheduled_at: datetime
    location: Optional[str] = None
    patient_id: Optional[str] = None  # Doctor specifies patient

@router.get("/appointments", response_model=list[dict])
def get_appointments(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    if cu.role == "ELDERLY":
        appts = db.query(Appointment).filter(Appointment.patient_id == cu.id).order_by(Appointment.scheduled_at.asc()).all()
    elif cu.role == "DOCTOR":
        appts = db.query(Appointment).filter(Appointment.doctor_id == cu.id).order_by(Appointment.scheduled_at.asc()).all()
    else:
        links = db.query(PatientLink).filter(PatientLink.linked_id == cu.id).all()
        patient_ids = [l.patient_id for l in links]
        appts = db.query(Appointment).filter(Appointment.patient_id.in_(patient_ids)).order_by(Appointment.scheduled_at.asc()).all()
    return [{"id":a.id,"title":a.title,"description":a.description,"scheduled_at":a.scheduled_at.isoformat(),
             "location":a.location,"status":a.status,"patient_id":a.patient_id,"doctor_id":a.doctor_id,"created_by":a.created_by} for a in appts]

@router.post("/appointments", response_model=dict, status_code=201)
def create_appointment(body: AppointmentCreate, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    patient_id = body.patient_id if cu.role == "DOCTOR" and body.patient_id else cu.id
    doctor_id  = cu.id if cu.role == "DOCTOR" else None

    appt = Appointment(
        id=str(uuid.uuid4()), patient_id=patient_id, doctor_id=doctor_id,
        title=body.title, description=body.description,
        scheduled_at=body.scheduled_at, location=body.location,
        created_by=cu.id,
    )
    db.add(appt)

    # Notify relevant users
    notify_ids = set()
    if cu.role == "ELDERLY":
        links = db.query(PatientLink).filter(PatientLink.patient_id == cu.id).all()
        notify_ids = {l.linked_id for l in links}
    elif cu.role == "DOCTOR" and patient_id:
        notify_ids.add(patient_id)
        links = db.query(PatientLink).filter(PatientLink.patient_id == patient_id).all()
        notify_ids.update(l.linked_id for l in links if l.linked_id != cu.id)

    for uid in notify_ids:
        db.add(Notification(
            id=str(uuid.uuid4()), user_id=uid, type="appointment",
            title=f"📅 New Appointment: {body.title}",
            message=f"Scheduled for {body.scheduled_at.strftime('%b %d, %Y at %I:%M %p')}. Location: {body.location or 'TBD'}",
            action_url="/communication/appointments",
        ))
    db.commit()
    return {"id":appt.id,"title":appt.title,"scheduled_at":appt.scheduled_at.isoformat()}

@router.patch("/appointments/{appt_id}/status", response_model=dict)
def update_appointment_status(appt_id: str, status: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    appt = db.query(Appointment).filter(Appointment.id == appt_id).first()
    if not appt: raise HTTPException(404, "Not found")
    if status not in ("upcoming","completed","cancelled"):
        raise HTTPException(400, "Invalid status")
    appt.status = status
    db.commit()
    return {"message": f"Appointment {status}"}

@router.delete("/appointments/{appt_id}", status_code=204)
def delete_appointment(appt_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    appt = db.query(Appointment).filter(Appointment.id == appt_id).first()
    if not appt: raise HTTPException(404, "Not found")
    db.delete(appt); db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Feature 3 — Emergency Contacts
# ══════════════════════════════════════════════════════════════════════════════
class EmergencyContactCreate(BaseModel):
    name: str
    phone: str
    relation: str
    is_primary: bool = False

@router.get("/emergency-contacts", response_model=list[dict])
def get_emergency_contacts(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    contacts = db.query(EmergencyContact).filter(EmergencyContact.user_id == cu.id).order_by(EmergencyContact.is_primary.desc()).all()
    return [{"id":c.id,"name":c.name,"phone":c.phone,"relation":c.relation,"is_primary":c.is_primary} for c in contacts]

@router.post("/emergency-contacts", response_model=dict, status_code=201)
def add_emergency_contact(body: EmergencyContactCreate, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    if body.is_primary:
        db.query(EmergencyContact).filter(EmergencyContact.user_id == cu.id).update({"is_primary": False})
    contact = EmergencyContact(
        id=str(uuid.uuid4()), user_id=cu.id,
        name=body.name, phone=body.phone, relation=body.relation, is_primary=body.is_primary,
    )
    db.add(contact); db.commit()
    return {"id":contact.id,"name":contact.name,"phone":contact.phone}

@router.delete("/emergency-contacts/{contact_id}", status_code=204)
def delete_emergency_contact(contact_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    c = db.query(EmergencyContact).filter(EmergencyContact.id == contact_id, EmergencyContact.user_id == cu.id).first()
    if not c: raise HTTPException(404, "Not found")
    db.delete(c); db.commit()

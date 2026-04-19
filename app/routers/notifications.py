# backend/app/routers/notifications.py
"""
Notification System + SOS Alert
"""
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User, PatientLink
from app.models.notification import Notification, SOSAlert

router = APIRouter(prefix="/notifications", tags=["Notifications"])


def create_notification(
    db: Session,
    user_id: str,
    type: str,
    title: str,
    message: str,
    action_url: str = None,
):
    """Helper to create a notification for a user."""
    n = Notification(
        id=str(uuid.uuid4()),
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        action_url=action_url,
    )
    db.add(n)
    db.commit()
    return n


def notify_linked_users(
    db: Session,
    patient_id: str,
    type: str,
    title: str,
    message: str,
    action_url: str = None,
):
    """Send notification to all caregivers and doctors linked to a patient."""
    links = db.query(PatientLink).filter(PatientLink.patient_id == patient_id).all()
    for link in links:
        create_notification(db, link.linked_id, type, title, message, action_url)


# ── Get my notifications ──────────────────────────────────────────────────────
@router.get("", response_model=list[dict])
def get_notifications(
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    notifs = (
        db.query(Notification)
        .filter(Notification.user_id == cu.id)
        .order_by(Notification.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": n.id,
            "type": n.type,
            "title": n.title,
            "message": n.message,
            "is_read": n.is_read,
            "action_url": n.action_url,
            "created_at": n.created_at.isoformat(),
        }
        for n in notifs
    ]


@router.get("/unread-count", response_model=dict)
def unread_count(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    count = (
        db.query(Notification)
        .filter(Notification.user_id == cu.id, Notification.is_read == False)
        .count()
    )
    return {"count": count}


@router.patch("/{notif_id}/read", response_model=dict)
def mark_read(
    notif_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    n = (
        db.query(Notification)
        .filter(Notification.id == notif_id, Notification.user_id == cu.id)
        .first()
    )
    if not n:
        raise HTTPException(404, "Not found")
    n.is_read = True
    db.commit()
    return {"message": "Marked as read"}


@router.patch("/read-all", response_model=dict)
def mark_all_read(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    db.query(Notification).filter(
        Notification.user_id == cu.id, Notification.is_read == False
    ).update({"is_read": True})
    db.commit()
    return {"message": "All marked as read"}


# ── SOS Alert ─────────────────────────────────────────────────────────────────
@router.post("/sos", response_model=dict)
async def trigger_sos(
    message: str = "I need help!",
    location: str = None,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Elder triggers SOS — notifies all linked caregivers and doctors."""
    from app.ai.sms_service import send_sos_alert

    # Save SOS record
    sos = SOSAlert(
        id=str(uuid.uuid4()),
        patient_id=cu.id,
        triggered_by=cu.id,
        location=location,
        message=message,
    )
    db.add(sos)
    db.commit()

    # Get all linked users
    links = db.query(PatientLink).filter(PatientLink.patient_id == cu.id).all()
    caregivers = []
    sms_count = 0

    for link in links:
        linked_user = db.query(User).filter(User.id == link.linked_id).first()
        if linked_user:
            # In-app notification
            create_notification(
                db,
                linked_user.id,
                "sos",
                f"🚨 SOS Alert from {cu.name}",
                f"{cu.name} needs immediate help! Location: {location or 'Unknown'}",
                "/dashboard",
            )
            caregivers.append({"phone": linked_user.phone, "name": linked_user.name})

    # Also notify the elder themselves
    create_notification(
        db,
        cu.id,
        "sos",
        "🚨 SOS Sent",
        "Your emergency alert has been sent to all caregivers.",
        "/dashboard",
    )

    # Send SMS
    sms_count = await send_sos_alert(cu.name, caregivers)

    return {
        "message": "SOS alert sent!",
        "notified_count": len(caregivers),
        "sms_sent": sms_count,
        "sos_id": sos.id,
    }


@router.get("/sos-alerts", response_model=list[dict])
def get_sos_alerts(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    """Get SOS alerts for current user (elder sees own, caregiver/doctor sees linked patient's)."""
    if cu.role == "ELDERLY":
        alerts = (
            db.query(SOSAlert)
            .filter(SOSAlert.patient_id == cu.id)
            .order_by(SOSAlert.created_at.desc())
            .all()
        )
    else:
        links = db.query(PatientLink).filter(PatientLink.linked_id == cu.id).all()
        patient_ids = [l.patient_id for l in links]
        alerts = (
            db.query(SOSAlert)
            .filter(SOSAlert.patient_id.in_(patient_ids))
            .order_by(SOSAlert.created_at.desc())
            .all()
        )

    return [
        {
            "id": a.id,
            "patient_id": a.patient_id,
            "message": a.message,
            "location": a.location,
            "is_resolved": a.is_resolved,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


@router.patch("/sos/{sos_id}/resolve", response_model=dict)
def resolve_sos(
    sos_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)
):
    alert = db.query(SOSAlert).filter(SOSAlert.id == sos_id).first()
    if not alert:
        raise HTTPException(404, "Not found")
    alert.is_resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    return {"message": "SOS resolved"}


@router.post("/medication-reminder/{med_id}", response_model=dict)
async def create_medication_reminder(
    med_id: str,
    med_name: str,
    dosage: str,
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    """Create a medication reminder notification — called by frontend when time matches."""
    from app.models.user import PatientLink

    # Notify the patient
    n = Notification(
        id=str(uuid.uuid4()),
        user_id=cu.id,
        type="medication",
        title=f"💊 Medication Reminder",
        message=f"Time to take {med_name} — {dosage}",
        action_url="/health/medication-reminder",
    )
    db.add(n)

    # Notify linked caregivers
    links = db.query(PatientLink).filter(PatientLink.patient_id == cu.id).all()
    for link in links:
        db.add(
            Notification(
                id=str(uuid.uuid4()),
                user_id=link.linked_id,
                type="medication",
                title=f"💊 {cu.name}'s Medication Time",
                message=f"{cu.name} should take {med_name} — {dosage} now",
                action_url="/health/medication-reminder",
            )
        )
    db.commit()
    return {"message": "Reminder notification sent"}

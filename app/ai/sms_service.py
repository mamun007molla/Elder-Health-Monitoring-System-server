# backend/app/ai/sms_service.py
"""
Twilio SMS Service
Used for:
  - SOS Alert (Person 1, M1)
  - Missed Medication Alert (Person 1, M2)
"""
from app.core.config import settings


async def send_sms(to_phone: str, message: str) -> bool:
    """Send SMS via Twilio."""
    if not all([settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN, settings.TWILIO_PHONE_NUMBER]):
        print(f"[SMS Mock] To: {to_phone} | Message: {message}")
        return True  # Mock success if not configured

    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=to_phone,
        )
        print(f"[SMS Sent] To: {to_phone}")
        return True
    except Exception as e:
        print(f"[SMS Error] {e}")
        return False


async def send_sos_alert(patient_name: str, caregivers: list[dict]) -> int:
    """Send SOS SMS to all linked caregivers and doctors."""
    message = f"🚨 SOS ALERT from CareAI!\n{patient_name} needs immediate help!\nPlease check on them immediately."
    sent = 0
    for c in caregivers:
        if c.get("phone"):
            success = await send_sms(c["phone"], message)
            if success: sent += 1
    return sent


async def send_medication_reminder_sms(patient_name: str, medication: str, caregiver_phone: str) -> bool:
    """Send missed medication alert to caregiver."""
    message = f"⚠️ CareAI Reminder\n{patient_name} has not logged taking {medication} yet.\nPlease check on them."
    return await send_sms(caregiver_phone, message)

# backend/app/routers/mental.py
import uuid, json
from datetime import datetime, date
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.mental import MoodEntry, ChecklistItem
from app.utils.file_upload import read_upload_bytes

router = APIRouter(prefix="/mental", tags=["Module 3"])


# ── M3-1: Mood Tracking (Groq text) ──────────────────────────────────────────
@router.get("/mood", response_model=list[dict])
def get_mood_entries(limit: int = 30, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    entries = db.query(MoodEntry).filter(MoodEntry.user_id == cu.id).order_by(MoodEntry.logged_at.desc()).limit(limit).all()
    return [{"id":e.id,"text":e.text,"mood":e.mood,"sentiment":e.sentiment,"score":e.score,
             "emotions":json.loads(e.emotions) if e.emotions else [],"summary":e.summary,
             "logged_at":e.logged_at.isoformat()} for e in entries]

@router.post("/mood/analyze", response_model=dict)
async def analyze_mood(
    text: str = Form(...),
    mood_label: str = Form("general"),
    db: Session = Depends(get_db),
    cu: User = Depends(get_current_user),
):
    from app.ai.groq_ai import analyze_mood as ai_analyze
    result = await ai_analyze(text, mood_label)
    entry = MoodEntry(
        id=str(uuid.uuid4()), user_id=cu.id, text=text,
        mood=result["mood"], sentiment=result["sentiment"], score=result["score"],
        emotions=json.dumps(result["emotions"]), summary=result["summary"],
    )
    db.add(entry)
    if result["score"] < 0.3 or result["sentiment"] == "negative":
        from app.models.user import PatientLink
        from app.models.notification import Notification
        links = db.query(PatientLink).filter(PatientLink.patient_id == cu.id).all()
        for l in links:
            db.add(Notification(id=str(uuid.uuid4()), user_id=l.linked_id, type="mood",
                title=f"😔 {cu.name}'s mood needs attention",
                message=f"{cu.name} logged a low mood: {result['summary']}"))
    db.commit()
    return {**result, "id": entry.id, "logged_at": entry.logged_at.isoformat()}

@router.get("/mood/weekly-summary", response_model=dict)
def weekly_mood_summary(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    from datetime import timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    entries = db.query(MoodEntry).filter(MoodEntry.user_id == cu.id, MoodEntry.logged_at >= week_ago).all()
    if not entries:
        return {"total":0,"average_score":0,"mood_distribution":{},"trend":"no data","entries":[]}
    scores = [e.score for e in entries if e.score]
    moods = {}
    for e in entries:
        if e.mood: moods[e.mood] = moods.get(e.mood, 0) + 1
    avg = sum(scores)/len(scores) if scores else 0
    return {
        "total": len(entries), "average_score": round(avg, 2),
        "mood_distribution": moods,
        "trend": "positive" if avg >= 0.6 else "concerning" if avg < 0.4 else "neutral",
        "entries": [{"mood":e.mood,"score":e.score,"logged_at":e.logged_at.isoformat()} for e in entries],
    }


# ── M3-2: Medical VQA (Groq vision) ──────────────────────────────────────────
@router.post("/vqa", response_model=dict)
async def medical_vqa(
    image: UploadFile = File(...),
    question: str = Form(...),
    cu: User = Depends(get_current_user),
):
    from app.ai.groq_vision import medical_vqa as ai_vqa
    img_bytes, mime_type = await read_upload_bytes(image)
    return await ai_vqa(img_bytes, question, mime_type)


# ── M3-3: Disease Diagnostic (Groq vision) ───────────────────────────────────
@router.post("/diagnostic", response_model=dict)
async def disease_diagnostic(
    image: UploadFile = File(...),
    cu: User = Depends(get_current_user),
):
    from app.ai.groq_vision import disease_diagnostic as ai_diag
    img_bytes, mime_type = await read_upload_bytes(image)
    return await ai_diag(img_bytes, mime_type)


# ── M3-4: Report Summarization (Groq text + easyocr for image) ───────────────
@router.post("/summarize-report", response_model=dict)
async def summarize_report(
    report_text: str = Form(None),
    file: UploadFile = File(None),
    cu: User = Depends(get_current_user),
):
    from app.ai.groq_ai import summarize_report as groq_summary
    text = ""
    if file:
        try:
            from app.ai.ocr_extract import extract_text_from_image
            img_bytes, _ = await read_upload_bytes(file)
            text = await extract_text_from_image(img_bytes)
        except Exception as e:
            raise HTTPException(500, f"OCR error: {str(e)}")
    elif report_text:
        text = report_text
    else:
        raise HTTPException(400, "Provide report_text or a file")

    result = await groq_summary(text)
    return {"key_findings":result.key_findings,"summary":result.summary,
            "medications_mentioned":result.medications_mentioned,"follow_up_needed":result.follow_up_needed}


# ── M3-8: Daily Checklist ─────────────────────────────────────────────────────
@router.get("/checklist", response_model=list[dict])
def get_checklist(check_date: str = None, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    today = check_date or date.today().isoformat()
    items = db.query(ChecklistItem).filter(ChecklistItem.user_id == cu.id, ChecklistItem.date == today).all()
    return [{"id":i.id,"title":i.title,"category":i.category,"is_done":i.is_done,
             "done_at":i.done_at.isoformat() if i.done_at else None,"date":i.date} for i in items]

@router.post("/checklist", response_model=dict, status_code=201)
def create_checklist_item(
    title: str = Form(...), category: str = Form("other"), date_str: str = Form(None),
    db: Session = Depends(get_db), cu: User = Depends(get_current_user),
):
    today = date_str or date.today().isoformat()
    item = ChecklistItem(id=str(uuid.uuid4()), user_id=cu.id, title=title, category=category, date=today)
    db.add(item); db.commit()
    return {"id":item.id,"title":item.title,"category":item.category,"is_done":False,"date":today}

@router.patch("/checklist/{item_id}/toggle", response_model=dict)
def toggle_checklist(item_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    item = db.query(ChecklistItem).filter(ChecklistItem.id == item_id, ChecklistItem.user_id == cu.id).first()
    if not item: raise HTTPException(404, "Not found")
    item.is_done = not item.is_done
    item.done_at = datetime.utcnow() if item.is_done else None
    db.commit()
    return {"id":item.id,"is_done":item.is_done,"done_at":item.done_at.isoformat() if item.done_at else None}

@router.delete("/checklist/{item_id}", status_code=204)
def delete_checklist(item_id: str, db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    item = db.query(ChecklistItem).filter(ChecklistItem.id == item_id, ChecklistItem.user_id == cu.id).first()
    if not item: raise HTTPException(404, "Not found")
    db.delete(item); db.commit()

@router.get("/checklist/stats", response_model=dict)
def checklist_stats(db: Session = Depends(get_db), cu: User = Depends(get_current_user)):
    today = date.today().isoformat()
    items = db.query(ChecklistItem).filter(ChecklistItem.user_id == cu.id, ChecklistItem.date == today).all()
    done = [i for i in items if i.is_done]
    return {"total":len(items),"done":len(done),"pending":len(items)-len(done),
            "percentage":round(len(done)/len(items)*100) if items else 0}

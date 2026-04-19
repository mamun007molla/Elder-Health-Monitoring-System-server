# backend/app/ai/groq_ai.py
"""Groq AI — llama-4-scout. Used for text analysis and summarization."""
import json, re
from app.core.config import settings
from app.schemas import ReportSummaryResult


def clean_json(text: str) -> str:
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()


async def _groq_chat(system: str, user_msg: str, max_tokens: int = 800) -> str:
    """Base Groq call."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Get free key at https://console.groq.com")
    from groq import AsyncGroq
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    response = await client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


# ── Mood / Sentiment Analysis ─────────────────────────────────────────────────
async def analyze_mood(text: str, mood_label: str = "general") -> dict:
    """Analyze mood text using Groq. Returns sentiment, emotions, summary."""
    system = """You are a compassionate mental health assistant for elderly care.
Analyze the user's mood description and respond ONLY with valid JSON, no other text:
{
  "sentiment": "positive or negative or neutral",
  "score": 0.0,
  "mood": "happy or sad or anxious or calm or angry or confused or lonely or content",
  "emotions": ["emotion1", "emotion2"],
  "summary": "One caring sentence about their emotional state",
  "recommendations": ["one practical suggestion for the caregiver or patient"]
}
Be empathetic and supportive in your summary."""

    prompt = f"Mood label chosen: {mood_label}\nWhat the person said: {text}"

    try:
        raw  = clean_json(await _groq_chat(system, prompt))
        data = json.loads(raw)
        return {
            "sentiment":       data.get("sentiment", "neutral"),
            "score":           float(data.get("score", 0.5)),
            "mood":            data.get("mood", mood_label),
            "emotions":        data.get("emotions", []),
            "summary":         data.get("summary", ""),
            "recommendations": data.get("recommendations", []),
        }
    except Exception:
        # plain text fallback
        positive = any(w in text.lower() for w in ["happy","good","great","better","fine","okay"])
        return {
            "sentiment": "positive" if positive else "neutral",
            "score": 0.7 if positive else 0.4,
            "mood": mood_label,
            "emotions": [],
            "summary": f"The patient expressed: {text[:100]}",
            "recommendations": [],
        }


# ── Report Summarization ──────────────────────────────────────────────────────
async def summarize_report(report_text: str) -> ReportSummaryResult:
    system = "You are a medical assistant. Respond ONLY with valid JSON, no other text."
    prompt = f"""Summarize this medical report for elderly care:

{{
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "summary": "Simple 2-3 sentence summary that patient can understand",
  "medications_mentioned": ["med1", "med2"],
  "follow_up_needed": false
}}

Report:
{report_text[:3000]}"""

    try:
        raw  = clean_json(await _groq_chat(system, prompt))
        data = json.loads(raw)
        return ReportSummaryResult(
            key_findings=data.get("key_findings", []),
            summary=data.get("summary", ""),
            medications_mentioned=data.get("medications_mentioned", []),
            follow_up_needed=bool(data.get("follow_up_needed", False)),
        )
    except Exception as e:
        return ReportSummaryResult(
            key_findings=[],
            summary=f"AI error: {str(e)[:200]}. Check GROQ_API_KEY.",
            medications_mentioned=[],
            follow_up_needed=False,
        )

# backend/app/ai/groq_vision.py
"""
Vision AI — OpenRouter free models for image features.
Uses openrouter/free which auto-selects best free vision model.
Falls back to Groq llama-4-scout for text-only tasks.
"""
import base64, json, re
from app.core.config import settings


def extract_json(text: str):
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


async def openrouter_image(
    image_bytes: bytes, prompt: str, mime_type: str = "image/jpeg"
) -> str:
    """Send image to OpenRouter free vision model."""
    if not settings.OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Get free key at https://openrouter.ai"
        )

    import httpx

    b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": "openrouter/free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://careai.app",
                "X-Title": "CareAI",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise ValueError(f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""


# ── Medication Verification ───────────────────────────────────────────────────
async def verify_medication(
    image_bytes: bytes, prescribed: str, mime_type: str = "image/jpeg"
) -> dict:
    prompt = f"""You are verifying medication for an elderly patient.
PRESCRIBED: {prescribed}

Look at this medicine image carefully.
1. What do you see? (pill colors, shapes, any text/label on packaging)
2. Does the label say "{prescribed}" or is it a known equivalent?

Respond ONLY with valid JSON:
{{
  "matched": true,
  "confidence": 0.90,
  "detected": "describe what you see",
  "warnings": []
}}"""
    raw = await openrouter_image(image_bytes, prompt, mime_type)
    data = extract_json(raw)
    if data:
        return {
            "matched": bool(data.get("matched", False)),
            "confidence": float(data.get("confidence", 0.5)),
            "detected_medication": str(data.get("detected", "")),
            "warnings": list(data.get("warnings", [])),
            "raw_response": raw,
        }
    matched = any(
        w in raw.lower() for w in ["match", "correct", "verified", "yes", "found"]
    )
    return {
        "matched": matched,
        "confidence": 0.7 if matched else 0.3,
        "detected_medication": raw[:300],
        "warnings": [],
        "raw_response": raw,
    }


# ── Food Image Analysis ───────────────────────────────────────────────────────
async def analyze_food_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    prompt = """Analyze this food image. Identify food items and estimate nutrition.

Respond ONLY with valid JSON:
{
  "food_name": "dish name",
  "ingredients": ["item1", "item2"],
  "calories": 350,
  "protein_g": 20,
  "carbs_g": 45,
  "fat_g": 10,
  "fiber_g": 5,
  "serving_size": "1 plate (300g)",
  "meal_type_suggestion": "lunch",
  "health_notes": "brief note for elderly person"
}"""
    raw = await openrouter_image(image_bytes, prompt, mime_type)
    data = extract_json(raw)
    if data:
        return {
            "food_name": str(data.get("food_name", "Unknown")),
            "ingredients": list(data.get("ingredients", [])),
            "calories": int(data.get("calories", 0)),
            "protein": float(data.get("protein_g", 0)),
            "carbs": float(data.get("carbs_g", 0)),
            "fat": float(data.get("fat_g", 0)),
            "fiber": float(data.get("fiber_g", 0)),
            "serving_size": str(data.get("serving_size", "1 serving")),
            "meal_type_suggestion": str(data.get("meal_type_suggestion", "meal")),
            "health_notes": str(data.get("health_notes", "")),
            "raw_response": raw,
        }
    return {
        "food_name": "Food detected",
        "ingredients": [],
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0,
        "fiber": 0,
        "serving_size": "1 serving",
        "meal_type_suggestion": "meal",
        "health_notes": raw[:200],
        "raw_response": raw,
    }


# ── Activity/Posture Analysis ─────────────────────────────────────────────────
async def analyze_activity_image(
    image_bytes: bytes, question: str, mime_type: str = "image/jpeg"
) -> str:
    prompt = f"""You are monitoring an elderly patient's physical activity.
Analyze this image carefully.

Question: {question}

Focus on: posture, activity, safety concerns, recommendations."""
    return await openrouter_image(image_bytes, prompt, mime_type)


# ── Medical VQA ───────────────────────────────────────────────────────────────
async def medical_vqa(
    image_bytes: bytes, question: str, mime_type: str = "image/jpeg"
) -> dict:
    prompt = f"""You are a medical AI helping elderly patients understand medical images.

Question: {question}

Respond ONLY with valid JSON:
{{
  "answer": "detailed answer",
  "confidence": 0.85,
  "related_findings": ["finding1"],
  "disclaimer": "AI analysis only. Always consult a qualified doctor."
}}"""
    raw = await openrouter_image(image_bytes, prompt, mime_type)
    data = extract_json(raw)
    if data:
        return {
            "answer": str(data.get("answer", raw[:400])),
            "confidence": float(data.get("confidence", 0.5)),
            "related_findings": list(data.get("related_findings", [])),
            "disclaimer": str(
                data.get("disclaimer", "Always consult a qualified doctor.")
            ),
            "raw_response": raw,
        }
    return {
        "answer": raw[:500],
        "confidence": 0.5,
        "related_findings": [],
        "disclaimer": "Always consult a qualified doctor.",
        "raw_response": raw,
    }


# ── Disease Diagnostic ────────────────────────────────────────────────────────
async def disease_diagnostic(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    prompt = """Analyze this medical image and identify possible conditions.

Respond ONLY with valid JSON:
{
  "possible_conditions": [
    {"name": "condition", "probability": 0.75, "description": "brief"}
  ],
  "findings": "key findings summary",
  "recommendations": ["see a doctor"],
  "urgency": "low",
  "disclaimer": "AI analysis only. Consult a qualified doctor."
}"""
    raw = await openrouter_image(image_bytes, prompt, mime_type)
    data = extract_json(raw)
    if data:
        return {
            "possible_conditions": list(data.get("possible_conditions", [])),
            "findings": str(data.get("findings", "")),
            "recommendations": list(data.get("recommendations", [])),
            "urgency": str(data.get("urgency", "low")),
            "disclaimer": str(
                data.get("disclaimer", "AI analysis only. Consult a qualified doctor.")
            ),
            "raw_response": raw,
        }
    return {
        "possible_conditions": [],
        "findings": raw[:400],
        "recommendations": [],
        "urgency": "low",
        "disclaimer": "AI analysis only. Consult a qualified doctor.",
        "raw_response": raw,
    }


# ── Extract Text from Image ───────────────────────────────────────────────────
async def extract_text_from_image(
    image_bytes: bytes, mime_type: str = "image/jpeg"
) -> str:
    prompt = "Extract all text from this medical document image. Return ONLY the extracted text."
    return await openrouter_image(image_bytes, prompt, mime_type)

# backend/app/ai/ollama_vision.py
"""Ollama Vision — gemma3:4b for all image features."""
import base64, json, re, httpx

OLLAMA_URL   = "http://localhost:11434/api/generate"
VISION_MODEL = "gemma3:4b"


def extract_json(text: str):
    text = re.sub(r'```(?:json)?\s*', '', text).replace('```', '').strip()
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None


async def ollama_vision(image_bytes: bytes, prompt: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        if resp.status_code != 200:
            raise ValueError(f"Ollama error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        raw = data.get("response", "")
        # Remove <unused94>thought...thought block if present
        raw = re.sub(r'<unused\d+>.*?<unused\d+>', '', raw, flags=re.DOTALL).strip()
        return raw


async def check_ollama_running() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get("http://localhost:11434/api/tags")
            return r.status_code == 200
    except Exception:
        return False


# ── Medication Verification ───────────────────────────────────────────────────
async def verify_medication(image_bytes: bytes, prescribed: str) -> dict:
    prompt = f"""You are verifying medication for an elderly patient.

PRESCRIBED MEDICATION: {prescribed}

Look at this image very carefully. Do the following:
Step 1: What do you see? Describe the pills/tablets/medicine (color, shape, size, any text written on them, packaging label)
Step 2: Does the label or packaging say "{prescribed}"? Or do the pills look like "{prescribed}"?
Step 3: Make a decision - matched or not.

IMPORTANT: If you can see the medicine name "{prescribed}" written anywhere OR if the pills look like typical "{prescribed}" tablets, set matched=true.

Respond ONLY with valid JSON:
{{
  "matched": true,
  "confidence": 0.85,
  "detected": "I see white round tablets in a blister pack labeled Napa/Paracetamol 500mg",
  "match_reason": "The packaging clearly shows the name matches",
  "warnings": []
}}"""

    raw  = await ollama_vision(image_bytes, prompt)
    data = extract_json(raw)
    if data:
        return {
            "matched": bool(data.get("matched", False)),
            "confidence": float(data.get("confidence", 0.5)),
            "detected_medication": str(data.get("detected", "")),
            "warnings": list(data.get("warnings", [])),
            "match_reason": str(data.get("match_reason", "")),
            "raw_response": raw,
        }
    matched = any(w in raw.lower() for w in ["correct","match","verified","yes","found"])
    return {"matched": matched, "confidence": 0.7 if matched else 0.3,
            "detected_medication": raw[:300], "warnings": [], "match_reason": "", "raw_response": raw}


# ── Food Image Analysis ───────────────────────────────────────────────────────
async def analyze_food_image(image_bytes: bytes) -> dict:
    prompt = """Look at this food image carefully and analyze it.

Identify all food items visible and estimate their nutritional content.

Respond with ONLY this JSON (no other text, no explanation):
{
  "food_name": "name of the main dish",
  "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
  "calories": 350,
  "protein_g": 20,
  "carbs_g": 45,
  "fat_g": 10,
  "fiber_g": 5,
  "serving_size": "1 plate approximately 300g",
  "meal_type_suggestion": "lunch",
  "health_notes": "brief note about this food for elderly person"
}

Use breakfast, lunch, dinner, or snack for meal_type_suggestion."""

    raw  = await ollama_vision(image_bytes, prompt)
    data = extract_json(raw)
    if data:
        return {
            "food_name": str(data.get("food_name", "Food item")),
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
    return {"food_name":"Food detected","ingredients":[],"calories":0,"protein":0,
            "carbs":0,"fat":0,"fiber":0,"serving_size":"1 serving",
            "meal_type_suggestion":"meal","health_notes":raw[:200],"raw_response":raw}


# ── Activity/Posture Analysis ─────────────────────────────────────────────────
async def analyze_activity_image(image_bytes: bytes, question: str) -> str:
    prompt = f"""You are monitoring an elderly patient's physical activity and posture.
Look at this image carefully.

Question: {question}

Provide a clear, specific answer focusing on:
- What activity or posture you observe
- Any concerns about safety or posture
- Recommendations if needed

Give a practical, helpful response."""
    return await ollama_vision(image_bytes, prompt)


# ── Medical VQA ───────────────────────────────────────────────────────────────
async def medical_vqa(image_bytes: bytes, question: str) -> dict:
    prompt = f"""You are a medical AI helping elderly patients understand medical images.
Look at this medical image carefully.

Question: {question}

Respond with ONLY this JSON:
{{
  "answer": "clear detailed answer to the question",
  "confidence": 0.8,
  "related_findings": ["finding1", "finding2"],
  "disclaimer": "This is AI analysis only. Always consult a qualified doctor."
}}"""
    raw  = await ollama_vision(image_bytes, prompt)
    data = extract_json(raw)
    if data:
        return {
            "answer": str(data.get("answer", raw[:400])),
            "confidence": float(data.get("confidence", 0.5)),
            "related_findings": list(data.get("related_findings", [])),
            "disclaimer": str(data.get("disclaimer", "Always consult a qualified doctor.")),
            "raw_response": raw,
        }
    return {"answer": raw[:500], "confidence": 0.5, "related_findings": [],
            "disclaimer": "Always consult a qualified doctor.", "raw_response": raw}


# ── Disease Diagnostic ────────────────────────────────────────────────────────
async def disease_diagnostic(image_bytes: bytes) -> dict:
    prompt = """You are a medical AI. Look at this medical image carefully.
Analyze it and identify possible conditions.

Respond with ONLY this JSON:
{
  "possible_conditions": [
    {"name": "condition name", "probability": 0.75, "description": "brief description"}
  ],
  "findings": "2-3 sentence summary of what you observe",
  "recommendations": ["see a doctor", "monitor the area"],
  "urgency": "low",
  "disclaimer": "AI analysis only. Consult a qualified doctor for diagnosis."
}

urgency must be: low, medium, or high"""
    raw  = await ollama_vision(image_bytes, prompt)
    data = extract_json(raw)
    if data:
        return {
            "possible_conditions": list(data.get("possible_conditions", [])),
            "findings": str(data.get("findings", "")),
            "recommendations": list(data.get("recommendations", [])),
            "urgency": str(data.get("urgency", "low")),
            "disclaimer": str(data.get("disclaimer", "AI analysis only. Consult a qualified doctor.")),
            "raw_response": raw,
        }
    return {"possible_conditions":[],"findings":raw[:400],"recommendations":[],
            "urgency":"low","disclaimer":"AI analysis only. Consult a qualified doctor.","raw_response":raw}


# ── Extract Text from Image ───────────────────────────────────────────────────
async def extract_text_from_image(image_bytes: bytes) -> str:
    prompt = """Look at this medical document image.
Extract ALL text you can read from it.
Return ONLY the extracted text, nothing else."""
    return await ollama_vision(image_bytes, prompt)

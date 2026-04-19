from app.schemas import MedicationVerifyResult


def names_match(prescribed: str, detected_text: str) -> tuple:
    import re
    p = prescribed.lower().strip()
    d = detected_text.lower()
    if p in d: return True, 0.95
    aliases = {
        "napa":        ["paracetamol", "acetaminophen", "panadol", "napa extra"],
        "napa extra":  ["paracetamol", "caffeine", "napa"],
        "paracetamol": ["napa", "ace", "tamen", "panadol", "calpol"],
        "metformin":   ["glucophage", "glycomet"],
        "amlodipine":  ["norvasc", "stamlo"],
        "aspirin":     ["ecotrin", "disprin"],
        "omeprazole":  ["losec", "prilosec"],
    }
    for alias in aliases.get(p, []):
        if alias in d: return True, 0.80
    words = [w for w in re.split(r'\W+', p) if len(w) > 2]
    if words:
        if sum(1 for w in words if w in d) / len(words) >= 0.5:
            return True, 0.65
    if len(p) >= 4 and (p[:4] in d or p[-4:] in d):
        return True, 0.55
    return False, 0.0


async def verify_medication_image(
    image_bytes: bytes,
    prescribed_medication: str,
    mime_type: str = "image/jpeg",
) -> MedicationVerifyResult:
    try:
        from app.ai.groq_vision import verify_medication
        ai_result = await verify_medication(image_bytes, prescribed_medication, mime_type)
        detected_text = ai_result.get("detected_medication", "")

        if detected_text and len(detected_text) >= 5:
            matched, confidence = names_match(prescribed_medication, detected_text)
        else:
            matched    = ai_result.get("matched", False)
            confidence = ai_result.get("confidence", 0.3)

        warnings = list(ai_result.get("warnings", []))
        if not matched:
            warnings.append(f"'{prescribed_medication}' not found. Detected: {detected_text[:80]}")

        return MedicationVerifyResult(
            matched=matched, confidence=confidence,
            detected_medication=detected_text,
            prescribed_medication=prescribed_medication,
            warnings=warnings,
            raw_response=ai_result.get("raw_response"),
        )
    except Exception as e:
        return MedicationVerifyResult(
            matched=False, confidence=0.0,
            detected_medication=None,
            prescribed_medication=prescribed_medication,
            warnings=[f"Error: {str(e)}"],
        )

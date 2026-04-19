# backend/app/ai/ocr_extract.py
"""OCR via OpenRouter vision — no local model needed."""


async def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using OpenRouter vision AI."""
    try:
        from app.ai.groq_vision import extract_text_from_image as vision_extract

        return await vision_extract(image_bytes)
    except Exception as e:
        raise ValueError(f"OCR error: {str(e)}")

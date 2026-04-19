# backend/app/ai/ocr_extract.py
"""
OCR Text Extraction — easyocr for prescription/report images.
No API key needed, runs locally.
"""
import io


async def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using easyocr."""
    try:
        import easyocr
        import numpy as np
        from PIL import Image

        # Convert bytes to numpy array
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        # Run OCR (first time downloads model ~40MB)
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(img_array, detail=0, paragraph=True)

        text = "\n".join(results)
        return text.strip() if text.strip() else ""

    except ImportError:
        raise ValueError("easyocr not installed. Run: pip install easyocr")
    except Exception as e:
        raise ValueError(f"OCR error: {str(e)}")

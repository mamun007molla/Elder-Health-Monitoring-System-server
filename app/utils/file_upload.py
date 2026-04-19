# backend/app/utils/file_upload.py
import os, uuid
import aiofiles
from fastapi import UploadFile, HTTPException
from app.core.config import settings

async def save_upload(file: UploadFile, subfolder: str = "general"):
    upload_dir = os.path.join(settings.UPLOAD_DIR, subfolder)
    os.makedirs(upload_dir, exist_ok=True)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    ext = os.path.splitext(file.filename or "file")[1] or ".bin"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(upload_dir, filename)
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(content)
    return f"/uploads/{subfolder}/{filename}", content

async def read_upload_bytes(file: UploadFile):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    return content, file.content_type or "image/jpeg"

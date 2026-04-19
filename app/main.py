import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.database import create_tables
from app.routers import auth, physical, health, mental, notifications, links, caregiver, communication

app = FastAPI(
    title="CareAI — Elderly Monitoring System",
    version="3.0.0",
    description="M1: Physical | M2: Health | M3: Mental Health & Doctor Support",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000","http://127.0.0.1:3000",
        "http://localhost:3001","http://127.0.0.1:3001",
        "http://localhost:3002","http://127.0.0.1:3002",
        settings.FRONTEND_URL,
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(auth.router,          prefix="/api/v1")
app.include_router(physical.router,      prefix="/api/v1")
app.include_router(health.router,        prefix="/api/v1")
app.include_router(mental.router,        prefix="/api/v1")
app.include_router(notifications.router, prefix="/api/v1")
app.include_router(links.router,         prefix="/api/v1")
app.include_router(caregiver.router,     prefix="/api/v1")
app.include_router(communication.router, prefix="/api/v1")

@app.on_event("startup")
async def startup():
    create_tables()
    print("✅ CareAI v3.0 — All 16 features active")

@app.get("/")
def root():
    return {"message":"CareAI v3.0","features":16,"modules":3}

@app.get("/health-check")
def health_check():
    return {"status":"ok"}

# backend/app/routers/auth.py
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas import RegisterRequest, LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(
        id=str(uuid.uuid4()), name=body.name,
        email=body.email.lower().strip(),
        phone=body.phone,
        password_hash=hash_password(body.password),
        role=body.role.value,
    )
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token({"sub": user.id, "role": user.role})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))

@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower().strip()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": user.id, "role": user.role})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))

@router.get("/me", response_model=UserOut)
def me(cu: User = Depends(get_current_user)):
    return cu

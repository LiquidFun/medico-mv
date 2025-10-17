from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr
from datetime import timedelta
import os
import uuid

from app.models import User, get_db
from app.services import create_access_token, verify_password, get_password_hash, get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    display_name: str | None = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    # Check if username exists
    result = await db.execute(select(User).where(User.username == user_data.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Check if email exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Create access token
    access_token = create_access_token(data={"sub": new_user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(new_user),
    }


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):
    """Login with username and password."""
    # Find user
    result = await db.execute(select(User).where(User.username == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(data={"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse.model_validate(user),
    }


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse.model_validate(user)


@router.get("/config")
async def get_auth_config():
    """Get authentication configuration."""
    return {
        "disable_login": os.getenv("DISABLE_LOGIN", "false").lower() == "true"
    }


@router.post("/anonymous-token", response_model=Token, status_code=status.HTTP_201_CREATED)
async def create_anonymous_token(session_id: str | None = None):
    """
    Create an anonymous session token when DISABLE_LOGIN is enabled.
    If session_id is provided, it will be used; otherwise a new UUID is generated.
    """
    if os.getenv("DISABLE_LOGIN", "false").lower() != "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Anonymous mode is disabled",
        )

    # Use provided session_id or generate new one
    if not session_id:
        session_id = str(uuid.uuid4())

    # Create JWT token with anonymous username
    username = f"anonymous_{session_id}"
    access_token = create_access_token(
        data={"sub": username},
        expires_delta=timedelta(days=365 * 10)  # 10 years for anonymous sessions
    )

    # Return token with virtual user info
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserResponse(
            id=0,  # Will be assigned from DB on first use
            username=username,
            email=f"{session_id}@anonymous.local",
            display_name="Ulrike Schlüter"
        )
    }

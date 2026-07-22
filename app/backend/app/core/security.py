from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.database import get_db
from app.services.identity.user_service import UserService
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user_service = UserService(db)
    user = await user_service.get_user_by_email(email)
    if user is None:
        raise credentials_exception
    # tenant_id 必须来自服务端用户记录，而不是客户端请求参数。
    if not getattr(user, "tenant_id", None):
        raise credentials_exception
    return user


async def require_compliance_user(
    current_user=Depends(get_current_user),
):
    if getattr(current_user, "role", "user") not in {
        "tenant_admin",
        "compliance_operator",
        "platform_admin",
    }:
        raise HTTPException(status_code=403, detail="需要管理员合规权限")
    return current_user

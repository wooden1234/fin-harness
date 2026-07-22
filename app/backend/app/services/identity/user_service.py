from datetime import datetime
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.hashing import get_password_hash, verify_password
from app.core.logger import get_logger
from app.models.identity.user import User
from app.schemas.user import UserCreate

logger = get_logger(service="user_service")


class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(self, user_data: UserCreate) -> User:
        query = select(User).where(
            or_(
                User.email == user_data.email,
                User.username == user_data.username,
            )
        )
        result = await self.db.execute(query)
        existing_user = result.scalar_one_or_none()

        if existing_user:
            if existing_user.email == user_data.email:
                raise ValueError("该邮箱已经被注册！")
            raise ValueError("用户名已被占用！")

        db_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=get_password_hash(user_data.password),
        )
        self.db.add(db_user)
        await self.db.commit()
        await self.db.refresh(db_user)
        return db_user

    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        query = select(User).where(User.email == email)
        result = await self.db.execute(query)
        user = result.scalar_one_or_none()

        if not user:
            logger.warning(f"User not found: {email}")
            return None

        if not verify_password(password, user.password_hash):
            logger.warning(f"Invalid password for user: {email}")
            return None

        user.last_login = datetime.utcnow()
        await self.db.commit()
        return user

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        query = select(User).where(User.id == user_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        query = select(User).where(User.email == email)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

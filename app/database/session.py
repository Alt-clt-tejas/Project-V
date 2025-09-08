# app/database/session.py
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# This import makes it read from your .env file
from app.config.base import settings

logger = logging.getLogger(__name__)

# --- START OF CORRECTION ---

# 1. Check if the DATABASE_URL was actually loaded from the environment.
#    This prevents the application from starting if the .env file is misconfigured.
if not settings.DATABASE_URL:
    raise ValueError("FATAL: DATABASE_URL is not set in the environment configuration. Application cannot start.")

# 2. Convert the Pydantic DSN object to a plain string.
#    SQLAlchemy's create_async_engine function expects a string, not a Pydantic object.
db_url_str = str(settings.DATABASE_URL)

# 3. Create the async engine instance using the corrected string URL.
engine = create_async_engine(
    db_url_str,
    echo=settings.DEBUG, # Log SQL statements if DEBUG is True
    pool_pre_ping=True,  # Checks if connections are alive before using them
)

# --- END OF CORRECTION ---


# This is our session factory for creating new database sessions
AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db_session() -> AsyncSession:
    """Dependency provider for FastAPI to get a database session for a single request."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            # Note: Committing here can be risky. It's often better to commit
            # explicitly in the service layer where the business logic resides.
            # For now, this is acceptable for simplicity.
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
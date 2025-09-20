# app/database/session.py
import logging
import sqlalchemy
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.config.base import settings

logger = logging.getLogger(__name__)

# --- THE DEFINITIVE FIX for Neon/Supabase Poolers ---
connect_args = {}
db_url = settings.DATABASE_URL

# Check if we are connecting to a PostgreSQL database
if db_url.startswith("postgresql"):
    # These settings are ONLY for PostgreSQL and are required for cloud providers
    # like Neon and Supabase that use pgbouncer.
    connect_args = {
        "server_settings": {
            "statement_cache_size": "0"  # Disables prepared statements
        },
        "ssl": "require"  # Enforces SSL, passed to the asyncpg driver
    }
# --- END FIX ---

engine = create_async_engine(
    db_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    connect_args=connect_args
)

# ... (the rest of the file remains the same)

AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db_session() -> AsyncSession:
    # ...
    pass

async def test_connection() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful.")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}", exc_info=False)
        return False
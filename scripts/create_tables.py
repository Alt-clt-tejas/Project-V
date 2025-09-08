# scripts/create_tables.py
import asyncio
import logging

# --- IMPORTS ---
# Since the package is installed in development mode, these imports work from anywhere!
from app.database.session import engine
from app.database.model import Base
from app.config.base import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def create_database_tables():
    logging.info("Connecting to the database...")
    async with engine.begin() as conn:
        logging.info("Dropping all existing tables for a clean slate...")
        await conn.run_sync(Base.metadata.drop_all)
        logging.info("Creating all tables from defined models...")
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    logging.info("Tables created successfully in your Supabase instance.")

async def main():
    if not settings.DATABASE_URL or "YOUR-PASSWORD" in str(settings.DATABASE_URL):
        logging.error("ERROR: DATABASE_URL is not set or is misconfigured in your .env file.")
    else:
        await create_database_tables()

if __name__ == "__main__":
    asyncio.run(main())
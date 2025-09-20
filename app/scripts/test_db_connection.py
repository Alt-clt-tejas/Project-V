# scripts/test_db_connection.py
import asyncio
import logging
from urllib.parse import urlparse

# Test if we can resolve the hostname
import socket

from app.config.base import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dns_resolution():
    """Test if we can resolve the database hostname"""
    try:
        db_url = str(settings.DATABASE_URL)
        parsed_url = urlparse(db_url)
        hostname = parsed_url.hostname
        port = parsed_url.port or 5432
        
        logger.info(f"Testing DNS resolution for: {hostname}:{port}")
        
        # Test DNS resolution
        result = socket.getaddrinfo(hostname, port)
        logger.info(f"✅ DNS Resolution successful: {result[0][4]}")
        return True
        
    except socket.gaierror as e:
        logger.error(f"❌ DNS Resolution failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

async def test_database_connection():
    """Test the actual database connection"""
    try:
        from app.database.session import engine
        
        logger.info("Testing database connection...")
        async with engine.begin() as conn:
            from sqlalchemy import text
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            logger.info(f"✅ Database connection successful! Test query result: {row}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False
    finally:
        try:
            await engine.dispose()
        except:
            pass

async def main():
    logger.info("=== Database Connection Test ===")
    logger.info(f"Database URL: {str(settings.DATABASE_URL)[:50]}...")
    
    # Test 1: DNS Resolution
    dns_ok = await test_dns_resolution()
    
    if dns_ok:
        # Test 2: Database Connection
        db_ok = await test_database_connection()
        
        if db_ok:
            logger.info("🎉 All tests passed! Your database is ready.")
        else:
            logger.error("🚨 Database connection failed even though DNS works.")
    else:
        logger.error("🚨 Cannot resolve database hostname. Check your DATABASE_URL.")
        logger.info("💡 Possible solutions:")
        logger.info("   1. Check your internet connection")
        logger.info("   2. Verify the Supabase hostname is correct")
        logger.info("   3. Try accessing your Supabase dashboard to confirm it's online")

if __name__ == "__main__":
    asyncio.run(main())

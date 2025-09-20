#!/usr/bin/env python3
import asyncio
import asyncpg

async def test_supabase_credentials():
    """Test different credential combinations to find the working one"""
    
    # Test configurations - try these in order
    test_configs = [
        {
            "name": "Original credentials",
            "params": {
                "host": "aws-1-ap-south-1.pooler.supabase.com",
                "port": 6543,
                "database": "postgres",
                "user": "postgres.qpwtrjdlihdpooxclzlk",
                "password": "omop#spma07",
                "statement_cache_size": 0,
                "ssl": "require"
            }
        },
        {
            "name": "With sslmode=require",
            "params": {
                "host": "aws-1-ap-south-1.pooler.supabase.com",
                "port": 6543,
                "database": "postgres",
                "user": "postgres.qpwtrjdlihdpooxclzlk",
                "password": "omop#spma07",
                "statement_cache_size": 0,
                "sslmode": "require"
            }
        },
        {
            "name": "Just username (no prefix)",
            "params": {
                "host": "aws-1-ap-south-1.pooler.supabase.com",
                "port": 6543,
                "database": "postgres", 
                "user": "qpwtrjdlihdpooxclzlk",  # Without postgres. prefix
                "password": "omop#spma07",
                "statement_cache_size": 0,
                "ssl": "require"
            }
        }
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        print(f"Host: {config['params']['host']}")
        print(f"User: {config['params']['user']}")
        print(f"Database: {config['params']['database']}")
        
        try:
            conn = await asyncpg.connect(**config['params'])
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            print(f"✅ SUCCESS! Query result: {result}")
            
            # Test version (this was failing before)
            version = await conn.fetchval("SELECT version()")
            print(f"PostgreSQL version: {version[:50]}...")
            
            await conn.close()
            
            # If we get here, this config works!
            print(f"✅ WORKING CONFIGURATION FOUND: {config['name']}")
            
            # Generate the correct DATABASE_URL
            user = config['params']['user']
            password = "omop%23spma07"  # URL encoded
            host = config['params']['host']
            port = config['params']['port']
            database = config['params']['database']
            
            database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}?sslmode=require&statement_cache_size=0"
            print(f"\nCorrect DATABASE_URL for your .env file:")
            print(f"DATABASE_URL={database_url}")
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    print("\n❌ All configurations failed. Please check:")
    print("1. Your Supabase project is active")
    print("2. Your password is correct")  
    print("3. Your database allows connections from your IP")
    return False

if __name__ == "__main__":
    asyncio.run(test_supabase_credentials())
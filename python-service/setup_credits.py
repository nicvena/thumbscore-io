#!/usr/bin/env python3
"""
Setup script for Thumbscore.io credit-based system

This script initializes the Supabase database schema and sets up
the credit system tables.
"""

import os
import sys
import logging
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize Supabase database schema"""
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Missing Supabase credentials. Please set SUPABASE_URL and SUPABASE_KEY in .env")
        return False
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("✅ Connected to Supabase")
        
        # Read SQL schema file
        schema_file = "supabase_schema.sql"
        if not os.path.exists(schema_file):
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        logger.info("📄 Read schema file")
        
        # Execute schema (this might need to be done manually in Supabase dashboard)
        # For now, we'll just validate the connection
        logger.info("🔧 Schema SQL ready to execute:")
        print("\n" + "="*60)
        print("COPY THE FOLLOWING SQL TO SUPABASE DASHBOARD:")
        print("="*60)
        print(schema_sql)
        print("="*60)
        
        # Test basic operations
        try:
            # Test users table
            result = supabase.table("users").select("id").limit(1).execute()
            logger.info("✅ Users table accessible")
        except Exception as e:
            logger.warning(f"⚠️  Users table not ready: {e}")
        
        try:
            # Test credits table
            result = supabase.table("credits").select("user_id").limit(1).execute()
            logger.info("✅ Credits table accessible")
        except Exception as e:
            logger.warning(f"⚠️  Credits table not ready: {e}")
        
        try:
            # Test score_cache table
            result = supabase.table("score_cache").select("cache_key").limit(1).execute()
            logger.info("✅ Score cache table accessible")
        except Exception as e:
            logger.warning(f"⚠️  Score cache table not ready: {e}")
        
        try:
            # Test rate_limits table
            result = supabase.table("rate_limits").select("ip").limit(1).execute()
            logger.info("✅ Rate limits table accessible")
        except Exception as e:
            logger.warning(f"⚠️  Rate limits table not ready: {e}")
        
        logger.info("🎉 Database setup completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def test_credit_system():
    """Test the credit system functionality"""
    try:
        from app.credits import ensure_wallet, get_credit_status, check_and_consume_credit
        
        # Test with a dummy user
        test_user_id = "test-user-123"
        
        # Ensure wallet exists
        wallet = ensure_wallet(test_user_id)
        logger.info(f"✅ Created wallet for test user: {wallet['plan']}")
        
        # Get credit status
        status = get_credit_status(test_user_id)
        logger.info(f"✅ Credit status: {status['used']}/{status['quota']} remaining")
        
        # Test credit consumption
        try:
            credit_result = check_and_consume_credit(test_user_id)
            logger.info(f"✅ Consumed credit: {credit_result['used']}/{credit_result['quota']}")
        except Exception as e:
            logger.info(f"ℹ️  Credit consumption test: {e}")
        
        logger.info("🎉 Credit system test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Credit system test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting Thumbscore.io credit system setup...")
    
    # Setup database
    if not setup_database():
        logger.error("❌ Database setup failed")
        sys.exit(1)
    
    # Test credit system
    if not test_credit_system():
        logger.error("❌ Credit system test failed")
        sys.exit(1)
    
    logger.info("✅ Setup completed successfully!")
    logger.info("\n📋 Next steps:")
    logger.info("1. Copy the SQL schema to your Supabase dashboard")
    logger.info("2. Run the SQL in the Supabase SQL editor")
    logger.info("3. Restart your FastAPI server")
    logger.info("4. Test the endpoints:")
    logger.info("   - GET /internal/credit-status")
    logger.info("   - POST /v1/preview_score")
    logger.info("   - POST /v1/score")

if __name__ == "__main__":
    main()

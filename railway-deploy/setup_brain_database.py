#!/usr/bin/env python3
"""
Setup script to initialize the YouTube Intelligence Brain database
This script creates all required tables and populates initial data
"""

import os
import asyncio
import logging
from supabase import create_client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_brain_database():
    """Initialize the brain database with all required tables"""
    
    # Load environment variables
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not all([supabase_url, supabase_key]):
        logger.error("Missing Supabase environment variables")
        return False
    
    try:
        # Create Supabase client
        supabase = create_client(supabase_url, supabase_key)
        logger.info("✅ Connected to Supabase")
        
        # Read and execute the schema SQL
        schema_path = os.path.join(os.path.dirname(__file__), "youtube_brain", "schema.sql")
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        logger.info("📋 Creating database tables...")
        
        # Split SQL into individual statements and execute them
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            if statement.upper().startswith(('CREATE', 'INSERT', 'ALTER')):
                try:
                    # Use RPC to execute SQL directly
                    result = supabase.rpc('exec_sql', {'sql': statement + ';'}).execute()
                    logger.info(f"✅ Executed statement {i+1}: {statement[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️  Statement {i+1} failed (may already exist): {str(e)[:100]}...")
        
        logger.info("🧠 YouTube Intelligence Brain database setup complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_brain_database())
    if success:
        print("\n🎉 Brain database setup successful! The YouTube Intelligence Brain is ready to collect data.")
    else:
        print("\n❌ Setup failed. Please check the logs and try again.")
"""
Database Configuration Manager for RAG Migration
Loads all database and system settings from environment variables
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Configuration class to manage all environment variables"""
    
    # Google Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # PostgreSQL Database
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_database")
    
    # Connection String
    POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
    
    # PGVector Configuration
    PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "rag_documents")
    PGVECTOR_USE_JSONB = os.getenv("PGVECTOR_USE_JSONB", "true").lower() == "true"
    PGVECTOR_DISTANCE_STRATEGY = os.getenv("PGVECTOR_DISTANCE_STRATEGY", "cosine")
    
    # RAG System Configuration
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "models/text-embedding-004")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))
    
    @classmethod
    def get_postgres_connection_params(cls) -> Dict[str, Any]:
        """Get PostgreSQL connection parameters as a dictionary"""
        return {
            "host": cls.POSTGRES_HOST,
            "port": cls.POSTGRES_PORT,
            "user": cls.POSTGRES_USER,
            "password": cls.POSTGRES_PASSWORD,
            "database": cls.POSTGRES_DB
        }
    
    @classmethod
    def validate_config(cls) -> tuple[bool, list[str]]:
        """Validate that all required configuration is present"""
        missing = []
        
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not cls.POSTGRES_PASSWORD:
            missing.append("POSTGRES_PASSWORD")
        if not cls.POSTGRES_CONNECTION_STRING:
            missing.append("POSTGRES_CONNECTION_STRING")
        
        return len(missing) == 0, missing
    
    @classmethod
    def print_config(cls):
        """Print current configuration (hiding sensitive data)"""
        print("🔧 Current Configuration:")
        print(f"   PostgreSQL Host: {cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}")
        print(f"   PostgreSQL Database: {cls.POSTGRES_DB}")
        print(f"   PostgreSQL User: {cls.POSTGRES_USER}")
        print(f"   PostgreSQL Password: {'*' * len(cls.POSTGRES_PASSWORD) if cls.POSTGRES_PASSWORD else 'NOT SET'}")
        print(f"   PGVector Collection: {cls.PGVECTOR_COLLECTION_NAME}")
        print(f"   Embeddings Model: {cls.EMBEDDINGS_MODEL}")
        print(f"   LLM Model: {cls.LLM_MODEL}")
        print(f"   Chunk Size: {cls.CHUNK_SIZE}")
        print(f"   Gemini API Key: {'*' * len(cls.GEMINI_API_KEY) if cls.GEMINI_API_KEY else 'NOT SET'}")

# Validate configuration on import
is_valid, missing = Config.validate_config()
if not is_valid:
    print(f"⚠️ Warning: Missing required configuration: {', '.join(missing)}")
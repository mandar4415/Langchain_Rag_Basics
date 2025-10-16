"""
Test PostgreSQL connection and PGVector setup
"""
import os
import psycopg
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import Config

def test_postgres_connection():
    """Test PostgreSQL connection and create database/extension if needed"""
    
    print("🔧 Using configuration from .env file:")
    Config.print_config()
    print()
    
    # Validate configuration
    is_valid, missing = Config.validate_config()
    if not is_valid:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        return False
    
    print("🔍 Testing PostgreSQL connection...")
    
    # Get connection parameters from config
    postgres_config = Config.get_postgres_connection_params()
    # Connect to default postgres database first
    postgres_config["database"] = "postgres"
    
    try:
        # Test basic connection
        with psycopg.connect(
            host=postgres_config["host"],
            port=postgres_config["port"],
            user=postgres_config["user"],
            password=postgres_config["password"],
            dbname=postgres_config["database"]
        ) as conn:
            print("✅ Successfully connected to PostgreSQL!")
            
            with conn.cursor() as cur:
                # Check PostgreSQL version
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"📊 PostgreSQL Version: {version}")
                
                # Check if vector extension exists
                cur.execute("SELECT * FROM pg_available_extensions WHERE name = 'vector';")
                vector_available = cur.fetchone()
                
                if vector_available:
                    print("✅ PGVector extension is available")
                    
                    # Check if vector extension is installed
                    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                    vector_installed = cur.fetchone()
                    
                    if vector_installed:
                        print("✅ PGVector extension is already installed")
                    else:
                        print("⚠️ PGVector extension available but not installed")
                        print("💡 Will install PGVector extension...")
                        
                        try:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                            conn.commit()
                            print("✅ PGVector extension installed successfully!")
                        except Exception as e:
                            print(f"❌ Failed to install PGVector extension: {e}")
                            return False
                else:
                    print("❌ PGVector extension is not available")
                    print("💡 Please install PGVector extension on your PostgreSQL server")
                    return False
                
                # Create RAG database if it doesn't exist
                cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{Config.POSTGRES_DB}';")
                db_exists = cur.fetchone()
                
                if not db_exists:
                    print(f"📦 Creating {Config.POSTGRES_DB} database...")
                    # End the current transaction first
                    conn.commit()
                    # Set autocommit for database creation
                    conn.autocommit = True
                    cur.execute(f"CREATE DATABASE {Config.POSTGRES_DB};")
                    print(f"✅ {Config.POSTGRES_DB} database created successfully!")
                    # Reset autocommit
                    conn.autocommit = False
                else:
                    print(f"✅ {Config.POSTGRES_DB} database already exists")
        
        # Test connection to RAG database and ensure vector extension is enabled
        rag_config = Config.get_postgres_connection_params()
        
        with psycopg.connect(
            host=rag_config["host"],
            port=rag_config["port"],
            user=rag_config["user"],
            password=rag_config["password"],
            dbname=rag_config["database"]
        ) as conn:
            print(f"🔍 Testing {Config.POSTGRES_DB} database connection...")
            
            with conn.cursor() as cur:
                # Ensure vector extension is installed in RAG database
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                
                # Test vector functionality
                cur.execute("SELECT vector_dims('[1,2,3]'::vector);")
                result = cur.fetchone()[0]
                print(f"✅ PGVector test successful: vector dimension = {result}")
        
        return True
        
    except psycopg.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Please check:")
        print("   - PostgreSQL is running")
        print("   - Connection parameters (host, port, user, password)")
        print("   - Database user has necessary permissions")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_langchain_pgvector():
    """Test LangChain PGVector integration"""
    print("\n🧪 Testing LangChain PGVector integration...")
    
    try:
        # Get API key from config
        if not Config.GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY not found in configuration")
            return False
        
        # Initialize embeddings using config
        embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDINGS_MODEL,
            google_api_key=Config.GEMINI_API_KEY
        )
        
        # Use connection string from config
        connection_string = Config.POSTGRES_CONNECTION_STRING
        
        # Test PGVector initialization with config settings
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=Config.PGVECTOR_COLLECTION_NAME,
            connection=connection_string,
            use_jsonb=Config.PGVECTOR_USE_JSONB,
        )
        
        print("✅ LangChain PGVector initialized successfully!")
        
        # Test adding a simple document
        from langchain_core.documents import Document
        
        test_doc = Document(
            page_content="This is a test document for PostgreSQL vector storage.",
            metadata={"source": "test", "type": "test"}
        )
        
        # Add document
        ids = vector_store.add_documents([test_doc], ids=["test_doc_1"])
        print(f"✅ Test document added with ID: {ids[0]}")
        
        # Test similarity search
        results = vector_store.similarity_search("test document", k=1)
        
        if results:
            print(f"✅ Similarity search successful: Found {len(results)} result(s)")
            print(f"   Content: {results[0].page_content[:50]}...")
        else:
            print("⚠️ Similarity search returned no results")
        
        # Clean up test document
        vector_store.delete(ids=["test_doc_1"])
        print("🧹 Test document cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain PGVector test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PostgreSQL and PGVector Setup Test")
    print("=" * 50)
    
    # Test basic PostgreSQL connection
    postgres_ok = test_postgres_connection()
    
    if postgres_ok:
        # Test LangChain integration
        langchain_ok = test_langchain_pgvector()
        
        if langchain_ok:
            print("\n🎉 All tests passed! PostgreSQL with PGVector is ready for RAG migration.")
        else:
            print("\n⚠️ PostgreSQL connection works, but LangChain integration needs attention.")
    else:
        print("\n❌ PostgreSQL connection failed. Please fix connection issues first.")
    
    print("\n💡 Next steps:")
    print("   1. Update connection parameters if needed")
    print("   2. Ensure PostgreSQL credentials are correct")
    print("   3. Run this script again until all tests pass")
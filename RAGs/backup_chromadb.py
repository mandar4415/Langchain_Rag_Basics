"""
ChromaDB Data Backup Script
Export all documents and metadata from existing ChromaDB before migration to PostgreSQL
"""
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import Config

def backup_chromadb_data():
    """Export all data from ChromaDB for backup and migration"""
    
    print("📦 ChromaDB Data Backup Script")
    print("=" * 50)
    
    # Validate configuration
    is_valid, missing = Config.validate_config()
    if not is_valid:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        return False
    
    try:
        # Define paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
        
        # Check if ChromaDB exists
        if not os.path.exists(persistent_directory):
            print(f"❌ ChromaDB not found at: {persistent_directory}")
            print("💡 No existing ChromaDB to backup")
            return False
        
        print(f"🔍 Found ChromaDB at: {persistent_directory}")
        
        # Initialize embeddings (same as original system)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDINGS_MODEL,
            google_api_key=Config.GEMINI_API_KEY
        )
        
        # Load existing ChromaDB
        print("🔄 Loading existing ChromaDB...")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
        
        # Get all documents from ChromaDB
        print("📊 Extracting all documents...")
        
        # Get documents in batches to handle large datasets
        batch_size = 1000
        all_documents = []
        offset = 0
        
        while True:
            try:
                # Use similarity search with a generic query to get documents
                batch_docs = db.similarity_search("", k=batch_size)
                
                if not batch_docs:
                    break
                    
                all_documents.extend(batch_docs)
                offset += batch_size
                
                print(f"   Extracted {len(all_documents)} documents so far...")
                
                # If we got fewer than batch_size, we've reached the end
                if len(batch_docs) < batch_size:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error extracting batch at offset {offset}: {e}")
                # Try a different approach
                try:
                    # Alternative method: get all documents at once
                    all_documents = db.similarity_search("test", k=10000)
                    break
                except Exception as e2:
                    print(f"❌ Failed to extract documents: {e2}")
                    return False
        
        print(f"✅ Extracted {len(all_documents)} total documents from ChromaDB")
        
        if not all_documents:
            print("⚠️ No documents found in ChromaDB")
            return True
        
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(current_dir, "backup", f"chromadb_backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        print(f"💾 Creating backup in: {backup_dir}")
        
        # Prepare data for export
        backup_data = {
            "metadata": {
                "backup_timestamp": timestamp,
                "total_documents": len(all_documents),
                "source_db": "ChromaDB",
                "embeddings_model": Config.EMBEDDINGS_MODEL,
                "backup_version": "1.0"
            },
            "documents": []
        }
        
        # Extract document data
        source_files = set()
        chunk_types = {}
        
        for i, doc in enumerate(all_documents):
            doc_data = {
                "id": i,
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            backup_data["documents"].append(doc_data)
            
            # Collect statistics
            source = doc.metadata.get('source', 'unknown')
            source_files.add(source)
            
            chunk_type = doc.metadata.get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        # Save backup data as JSON
        json_file = os.path.join(backup_dir, "chromadb_backup.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        # Save backup data as pickle (for exact Python object preservation)
        pickle_file = os.path.join(backup_dir, "chromadb_backup.pkl")
        with open(pickle_file, 'wb') as f:
            pickle.dump(backup_data, f)
        
        # Create summary report
        summary = {
            "backup_summary": {
                "timestamp": timestamp,
                "total_documents": len(all_documents),
                "source_files": sorted(list(source_files)),
                "chunk_types": chunk_types,
                "backup_files": {
                    "json": json_file,
                    "pickle": pickle_file
                }
            }
        }
        
        summary_file = os.path.join(backup_dir, "backup_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n📋 Backup Summary:")
        print(f"   📁 Backup Directory: {backup_dir}")
        print(f"   📄 Total Documents: {len(all_documents)}")
        print(f"   📚 Source Files: {len(source_files)}")
        print(f"   🏷️  Chunk Types: {len(chunk_types)}")
        print(f"   💾 JSON Backup: {json_file}")
        print(f"   💾 Pickle Backup: {pickle_file}")
        print(f"   📊 Summary Report: {summary_file}")
        
        print("\n📚 Source Files:")
        for source in sorted(source_files):
            count = sum(1 for doc in all_documents if doc.metadata.get('source') == source)
            print(f"   - {source}: {count} chunks")
        
        print("\n🏷️  Chunk Types:")
        for chunk_type, count in chunk_types.items():
            print(f"   - {chunk_type}: {count} chunks")
        
        print("\n✅ ChromaDB backup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_backup(backup_dir: str):
    """Verify the integrity of the backup"""
    print(f"\n🔍 Verifying backup in: {backup_dir}")
    
    try:
        # Check if files exist
        json_file = os.path.join(backup_dir, "chromadb_backup.json")
        pickle_file = os.path.join(backup_dir, "chromadb_backup.pkl")
        summary_file = os.path.join(backup_dir, "backup_summary.json")
        
        files_exist = all(os.path.exists(f) for f in [json_file, pickle_file, summary_file])
        
        if not files_exist:
            print("❌ Backup verification failed: Missing files")
            return False
        
        # Load and verify JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Load and verify pickle data
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
        
        # Compare data integrity
        json_count = len(json_data.get("documents", []))
        pickle_count = len(pickle_data.get("documents", []))
        
        if json_count != pickle_count:
            print(f"⚠️ Data mismatch: JSON has {json_count} docs, Pickle has {pickle_count} docs")
            return False
        
        print(f"✅ Backup verification successful: {json_count} documents verified")
        return True
        
    except Exception as e:
        print(f"❌ Backup verification failed: {e}")
        return False

if __name__ == "__main__":
    success = backup_chromadb_data()
    
    if success:
        print("\n🎉 Ready for migration to PostgreSQL!")
    else:
        print("\n⚠️ Backup process encountered issues. Please review and retry.")
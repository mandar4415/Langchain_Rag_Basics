"""
FastAPI wrapper for RAG Question Answering System
This preserves all the tested logic from 3_rag_one_off_question.py
"""

import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Question Answering API with PostgreSQL",
    description="AI-powered question answering using Retrieval Augmented Generation with PostgreSQL and PGVector",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for database and models (initialized once)
vector_store = None
llm = None
embeddings = None

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = 10

class DocumentMetadata(BaseModel):
    source: str
    chunk_index: str
    chunk_id: str
    similarity_score: float
    similarity_percentage: float
    first_words: Optional[str] = None
    last_words: Optional[str] = None
    start_line: Optional[str] = None
    end_line: Optional[str] = None
    document_percentage: Optional[str] = None
    chunk_size_chars: Optional[str] = None
    chunk_size_words: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None

class RelevantDocument(BaseModel):
    content: str
    metadata: DocumentMetadata

class QueryResponse(BaseModel):
    question: str
    answer: str
    relevant_documents: List[RelevantDocument]
    total_documents_found: int

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int
    total_chunks_in_database: int
    file_size_bytes: int
    file_already_existed: bool
    file_storage_path: str
    file_existed_physically: bool

class DatabaseInfo(BaseModel):
    total_chunks: int
    source_files: List[str]
    chunk_types: Dict[str, int]
    database_status: str
    embeddings_model: str
    database_type: str

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system components with PostgreSQL"""
    global vector_store, llm, embeddings
    
    try:
        # Validate configuration
        is_valid, missing = Config.validate_config()
        if not is_valid:
            raise ValueError(f"Missing configuration: {', '.join(missing)}")
        
        print("🔧 Using PostgreSQL configuration:")
        Config.print_config()
        
        # Create uploaded files directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        uploaded_files_dir = os.path.join(current_dir, "uploaded_files")
        os.makedirs(uploaded_files_dir, exist_ok=True)
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDINGS_MODEL,
            google_api_key=Config.GEMINI_API_KEY
        )
        
        # Initialize PostgreSQL vector store
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=Config.PGVECTOR_COLLECTION_NAME,
            connection=Config.POSTGRES_CONNECTION_STRING,
            use_jsonb=Config.PGVECTOR_USE_JSONB,
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, 
            google_api_key=Config.GEMINI_API_KEY
        )
        
        print("✅ RAG system initialized successfully with PostgreSQL!")
        print(f"📁 Uploaded files directory: {uploaded_files_dir}")
        print(f"🗄️ PostgreSQL database: {Config.POSTGRES_DB}")
        print(f"📊 PGVector collection: {Config.PGVECTOR_COLLECTION_NAME}")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        raise e

def clean_text_for_postgresql(text: str) -> str:
    """Clean text to remove NULL bytes and other problematic characters for PostgreSQL"""
    import re
    
    # Remove NULL bytes (0x00) which PostgreSQL cannot handle
    text = text.replace('\x00', '')
    
    # Remove other problematic control characters except newlines, tabs, and carriage returns
    cleaned_text = ""
    for char in text:
        # Keep printable characters, newlines, tabs, and carriage returns
        if char.isprintable() or char in ['\n', '\t', '\r']:
            cleaned_text += char
        else:
            # Replace other control characters with space
            cleaned_text += ' '
    
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean text for PostgreSQL compatibility
        text = clean_text_for_postgresql(text)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file"""
    try:
        text = file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = file_content.decode('latin-1')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")
    
    # Clean text for PostgreSQL compatibility
    text = clean_text_for_postgresql(text)
    return text

def process_uploaded_file(filename: str, file_content: bytes) -> str:
    """Process uploaded file and extract text based on file type"""
    file_extension = filename.lower().split('.')[-1]
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['txt', 'text']:
        return extract_text_from_txt(file_content)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported types: PDF, TXT"
        )

def check_file_exists_in_database(filename: str) -> bool:
    """Check if a file with the same name already exists in the PostgreSQL database"""
    try:
        # Use PostgreSQL metadata filtering to check for existing source
        results = vector_store.similarity_search(
            query="", 
            k=1,
            filter={"source": {"$eq": filename}}
        )
        return len(results) > 0
    except:
        return False

def get_file_storage_path(filename: str) -> str:
    """Get the full path where uploaded file should be stored"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    uploaded_files_dir = os.path.join(current_dir, "uploaded_files")
    return os.path.join(uploaded_files_dir, filename)

def check_file_exists_physically(filename: str) -> bool:
    """Check if a file exists physically in the uploaded_files directory"""
    file_path = get_file_storage_path(filename)
    return os.path.exists(file_path)

def create_chunks_with_metadata(filename: str, full_text: str) -> List[Document]:
    """
    Create document chunks with metadata using the same logic as 2a_rag_basics_metadata.py
    """
    # RecursiveCharacterTextSplitter is better at respecting chunk size limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs, then lines, then spaces, then characters
    )
    
    # Split this document's text into chunks
    chunks = text_splitter.split_text(full_text)
    
    # Create enhanced chunks with position metadata
    documents = []
    current_pos = 0
    
    for i, chunk_text in enumerate(chunks):
        # Find where this chunk starts in the original text
        chunk_start_pos = full_text.find(chunk_text, current_pos)
        
        if chunk_start_pos != -1:
            # Calculate end position
            chunk_end_pos = chunk_start_pos + len(chunk_text)
            
            # Calculate line numbers for start and end
            lines_before_start = full_text[:chunk_start_pos].count('\n')
            chunk_start_line = lines_before_start + 1
            
            lines_before_end = full_text[:chunk_end_pos].count('\n')
            chunk_end_line = lines_before_end + 1
            
            # Calculate word positions (more precise than character positions)
            words_before_start = len(full_text[:chunk_start_pos].split())
            chunk_start_word = words_before_start + 1
            
            words_in_chunk = len(chunk_text.split())
            chunk_end_word = chunk_start_word + words_in_chunk - 1
            
            # Calculate percentage through document
            chunk_percentage = (chunk_start_pos / len(full_text)) * 100
            
            # Get preview of surrounding text for better context
            preview_start = max(0, chunk_start_pos - 50)
            preview_end = min(len(full_text), chunk_end_pos + 50)
            context_before = full_text[preview_start:chunk_start_pos]
            context_after = full_text[chunk_end_pos:preview_end]
            
            # Get first and last few words of the chunk for easy identification
            chunk_words = chunk_text.split()
            first_words = " ".join(chunk_words[:5]) if len(chunk_words) >= 5 else chunk_text
            last_words = " ".join(chunk_words[-5:]) if len(chunk_words) >= 5 else chunk_text
            
            # Create a new document with enhanced metadata
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": filename,
                    "start_line": chunk_start_line,
                    "end_line": chunk_end_line,
                    "start_char_pos": chunk_start_pos,
                    "end_char_pos": chunk_end_pos,
                    "start_word_pos": chunk_start_word,
                    "end_word_pos": chunk_end_word,
                    "document_percentage": round(chunk_percentage, 1),
                    "chunk_size_chars": len(chunk_text),
                    "chunk_size_words": words_in_chunk,
                    "first_words": first_words,
                    "last_words": last_words,
                    "context_before": context_before.strip(),
                    "context_after": context_after.strip(),
                    "chunk_index": len(documents),  # Will be updated globally later
                    "chunk_id": f"chunk_{len(documents):04d}"  # Will be updated globally later
                }
            )
            documents.append(chunk_doc)
            
            # Update position for next search
            current_pos = chunk_end_pos
        else:
            # Fallback if exact position not found
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": filename,
                    "start_line": "Unknown",
                    "end_line": "Unknown",
                    "start_char_pos": "Unknown",
                    "end_char_pos": "Unknown",
                    "start_word_pos": "Unknown",
                    "end_word_pos": "Unknown",
                    "document_percentage": "Unknown",
                    "chunk_size_chars": len(chunk_text),
                    "chunk_size_words": len(chunk_text.split()),
                    "first_words": chunk_text[:50] + "..." if len(chunk_text) > 50 else chunk_text,
                    "last_words": "..." + chunk_text[-50:] if len(chunk_text) > 50 else chunk_text,
                    "context_before": "",
                    "context_after": "",
                    "chunk_index": len(documents),  # Will be updated globally later
                    "chunk_id": f"chunk_{len(documents):04d}"  # Will be updated globally later
                }
            )
            documents.append(chunk_doc)
    
    return documents

def filter_and_process_documents(query: str, k: int = 15) -> List[Dict[str, Any]]:
    """
    Filter and process documents using PostgreSQL similarity search with scores
    """
    # Use PostgreSQL's similarity search with scores method
    result_docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    
    # Filter and organize the results (same logic as ChromaDB version)
    relevant_docs = []
    for doc, score in result_docs_with_scores:
        # Filter out extremely small chunks (likely formatting/artifacts)
        chunk_text = (doc.page_content or "").strip()
        
        # More sophisticated filtering:
        # 1. Skip very small chunks
        if len(chunk_text) < 50:
            continue
        
        # 2. Skip chunks that are mostly punctuation/symbols (like "* * * * *")
        if len(chunk_text.replace(" ", "").replace("*", "").replace("-", "").replace("_", "")) < 10:
            continue
        
        # 3. Prefer chunks from specific sources for technical queries
        source = doc.metadata.get('source', '')
        
        relevant_docs.append({"doc": doc, "score": score})
        
        # Stop once we have enough good results
        if len(relevant_docs) >= 10:
            break
    
    return relevant_docs

def generate_answer(query: str, relevant_docs: List[Dict[str, Any]]) -> str:
    """
    Generate answer using the exact same logic as 3_rag_one_off_question.py
    """
    # Combine the query and the relevant document contents (exact same logic)
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([entry['doc'].page_content for entry in relevant_docs])
        + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    
    # Define the messages for the model (exact same logic)
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]
    
    # Invoke the model with the combined input
    result = llm.invoke(messages)
    return result.content

# API Endpoints

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (TXT or PDF) and add it to the vector database
    Includes duplicate detection and file storage
    """
    if not vector_store or not embeddings:
        raise HTTPException(status_code=500, detail="RAG system not properly initialized")
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['pdf', 'txt', 'text']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported types: PDF, TXT"
            )
        
        # Check if file already exists in database and physically
        file_already_existed = check_file_exists_in_database(file.filename)
        file_existed_physically = check_file_exists_physically(file.filename)
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Get file storage path
        file_storage_path = get_file_storage_path(file.filename)
        
        if file_already_existed:
            # File already exists in database - don't process or save again
            try:
                # Count total documents in PostgreSQL (more efficient than retrieving all)
                sample_results = vector_store.similarity_search("", k=1)
                # For now, we'll use a simple count - in production you'd want a proper count query
                current_total = len(vector_store.similarity_search("", k=10000))
            except:
                current_total = 0
            
            return UploadResponse(
                message=f"File '{file.filename}' already exists in database. No new chunks added.",
                filename=file.filename,
                chunks_added=0,
                total_chunks_in_database=current_total,
                file_size_bytes=file_size,
                file_already_existed=True,
                file_storage_path=file_storage_path,
                file_existed_physically=file_existed_physically
            )
        
        # Process new file
        # Extract text from file
        full_text = process_uploaded_file(file.filename, file_content)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Save file to storage directory (only for new files)
        with open(file_storage_path, "wb") as f:
            f.write(file_content)
        
        # Create chunks with metadata (same logic as existing system)
        new_chunks = create_chunks_with_metadata(file.filename, full_text)
        
        if not new_chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from file")
        
        # Get current total chunks for global indexing
        try:
            # For PostgreSQL, we'll use a simple count approach
            current_total = len(vector_store.similarity_search("", k=10000))
        except:
            current_total = 0
        
        # Update global chunk indices for new chunks
        for i, chunk in enumerate(new_chunks):
            global_index = current_total + i
            chunk.metadata["chunk_index"] = global_index
            chunk.metadata["chunk_id"] = f"chunk_{global_index:04d}"
        
        # Add new chunks to PostgreSQL database (preserves all existing content)
        vector_store.add_documents(new_chunks)
        
        # Get updated total count
        try:
            updated_total = len(vector_store.similarity_search("", k=10000))
        except:
            updated_total = current_total + len(new_chunks)
        
        return UploadResponse(
            message=f"File '{file.filename}' uploaded and processed successfully",
            filename=file.filename,
            chunks_added=len(new_chunks),
            total_chunks_in_database=updated_total,
            file_size_bytes=file_size,
            file_already_existed=False,
            file_storage_path=file_storage_path,
            file_existed_physically=file_existed_physically
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "RAG API is running with PostgreSQL",
        "database_connected": vector_store is not None,
        "llm_initialized": llm is not None,
        "database_type": "PostgreSQL with PGVector"
    }

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main RAG endpoint - ask a question and get an AI-generated answer
    """
    if not vector_store or not llm:
        raise HTTPException(status_code=500, detail="RAG system not properly initialized")
    
    try:
        query = request.question.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process documents using exact same logic as working script
        relevant_docs = filter_and_process_documents(query, k=15)
        
        if not relevant_docs:
            return QueryResponse(
                question=query,
                answer="I'm not sure. No relevant documents were found for your question.",
                relevant_documents=[],
                total_documents_found=0
            )
        
        # Generate answer using exact same logic
        answer = generate_answer(query, relevant_docs)
        
        # Format response documents
        response_docs = []
        for entry in relevant_docs[:request.max_results]:
            doc = entry.get("doc")
            score = entry.get("score")
            
            # Calculate similarity percentage (same logic as working script)
            similarity_percentage = (1 - score) * 100 if score <= 1 else 0
            
            doc_metadata = DocumentMetadata(
                source=doc.metadata.get('source', 'Unknown'),
                chunk_index=str(doc.metadata.get('chunk_index', 'N/A')),
                chunk_id=doc.metadata.get('chunk_id', 'N/A'),
                similarity_score=round(score, 3),
                similarity_percentage=round(similarity_percentage, 1),
                first_words=doc.metadata.get('first_words'),
                last_words=doc.metadata.get('last_words'),
                start_line=str(doc.metadata.get('start_line', 'N/A')),
                end_line=str(doc.metadata.get('end_line', 'N/A')),
                document_percentage=str(doc.metadata.get('document_percentage', 'N/A')),
                chunk_size_chars=str(doc.metadata.get('chunk_size_chars', 'N/A')),
                chunk_size_words=str(doc.metadata.get('chunk_size_words', 'N/A')),
                context_before=doc.metadata.get('context_before'),
                context_after=doc.metadata.get('context_after')
            )
            
            response_docs.append(RelevantDocument(
                content=doc.page_content,
                metadata=doc_metadata
            ))
        
        return QueryResponse(
            question=query,
            answer=answer,
            relevant_documents=response_docs,
            total_documents_found=len(relevant_docs)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/database/info", response_model=DatabaseInfo)
async def get_database_info():
    """Get information about the PostgreSQL vector database"""
    if not vector_store:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Get all documents from PostgreSQL database
        all_docs = vector_store.similarity_search("", k=10000)  # Get all documents
        
        # Count total chunks
        total_chunks = len(all_docs)
        
        # Extract unique source files from metadata
        unique_sources = set()
        chunk_types = {}
        
        for doc in all_docs:
            source = doc.metadata.get('source', 'unknown')
            unique_sources.add(source)
            
            # Count chunk types
            chunk_type = doc.metadata.get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        source_files = sorted(list(unique_sources))
        
        return DatabaseInfo(
            total_chunks=total_chunks,
            source_files=source_files,
            chunk_types=chunk_types,
            database_status="active" if total_chunks > 0 else "empty",
            embeddings_model=Config.EMBEDDINGS_MODEL,
            database_type="PostgreSQL with PGVector"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving database info: {str(e)}")

@app.get("/files/uploaded")
async def list_uploaded_files():
    """List all files stored in the uploaded_files directory"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        uploaded_files_dir = os.path.join(current_dir, "uploaded_files")
        
        if not os.path.exists(uploaded_files_dir):
            return {"uploaded_files": [], "total_files": 0, "storage_directory": uploaded_files_dir}
        
        files = []
        for filename in os.listdir(uploaded_files_dir):
            file_path = os.path.join(uploaded_files_dir, filename)
            if os.path.isfile(file_path):
                file_stats = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": file_stats.st_size,
                    "upload_time": file_stats.st_mtime,
                    "file_path": file_path
                })
        
        # Sort by upload time (newest first)
        files.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return {
            "uploaded_files": files,
            "total_files": len(files),
            "storage_directory": uploaded_files_dir
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing uploaded files: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Welcome message with API information"""
    return {
        "message": "Welcome to RAG Question Answering API with PostgreSQL & PGVector",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/query",
        "upload_endpoint": "/upload", 
        "database_info": "/database/info",
        "uploaded_files": "/files/uploaded",
        "supported_file_types": ["PDF", "TXT"],
        "database_type": "PostgreSQL with PGVector",
        "features": [
            "Query documents with similarity search using PostgreSQL",
            "Upload new files (PDF/TXT) to expand database",
            "Duplicate file detection",
            "Physical file storage in uploaded_files directory",
            "Search across all uploaded files",
            "Contextual responses with metadata",
            "PostgreSQL vector storage with PGVector extension",
            "High-performance vector operations"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
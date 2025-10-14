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
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Question Answering API",
    description="AI-powered question answering using Retrieval Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for database and models (initialized once)
db = None
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

class DatabaseInfo(BaseModel):
    total_chunks: int
    source_files: List[str]
    chunk_types: Dict[str, int]
    database_status: str
    embeddings_model: str

# Initialize RAG components on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system components"""
    global db, llm, embeddings
    
    try:
        # Define paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")
        
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        
        # Load vector database
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=api_key
        )
        
        print("✅ RAG system initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        raise e

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file"""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_content.decode('latin-1')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading text file: {str(e)}")

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
    Filter and process documents using the exact same logic as 3_rag_one_off_question.py
    """
    # Use Chroma's direct similarity search with scores method to get actual similarity scores
    result_docs_with_scores = db.similarity_search_with_score(query, k=k)
    
    # Filter and organize the results (exact same logic as working script)
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
        
        # 3. Prefer chunks from the AI/ML guide for technical queries
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
    Preserves all existing documents and adds new chunks to the same database
    """
    if not db or not embeddings:
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
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Extract text from file
        full_text = process_uploaded_file(file.filename, file_content)
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text content found in file")
        
        # Create chunks with metadata (same logic as existing system)
        new_chunks = create_chunks_with_metadata(file.filename, full_text)
        
        if not new_chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from file")
        
        # Get current total chunks for global indexing
        try:
            current_docs = db.similarity_search("test", k=10000)  # Get all existing docs
            current_total = len(current_docs)
        except:
            current_total = 0
        
        # Update global chunk indices for new chunks
        for i, chunk in enumerate(new_chunks):
            global_index = current_total + i
            chunk.metadata["chunk_index"] = global_index
            chunk.metadata["chunk_id"] = f"chunk_{global_index:04d}"
        
        # Add new chunks to existing database (preserves all existing content)
        db.add_documents(new_chunks)
        
        # Get updated total count
        try:
            updated_docs = db.similarity_search("test", k=10000)
            updated_total = len(updated_docs)
        except:
            updated_total = current_total + len(new_chunks)
        
        return UploadResponse(
            message=f"File '{file.filename}' uploaded and processed successfully",
            filename=file.filename,
            chunks_added=len(new_chunks),
            total_chunks_in_database=updated_total,
            file_size_bytes=file_size
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
        "message": "RAG API is running",
        "database_connected": db is not None,
        "llm_initialized": llm is not None
    }

@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main RAG endpoint - ask a question and get an AI-generated answer
    """
    if not db or not llm:
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
    """Get information about the vector database"""
    if not db:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    try:
        # Get all documents from database
        all_docs = db.similarity_search("test", k=10000)  # Get all documents
        
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
            embeddings_model="text-embedding-004"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving database info: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Welcome message with API information"""
    return {
        "message": "Welcome to RAG Question Answering API with File Upload",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/query",
        "upload_endpoint": "/upload", 
        "database_info": "/database/info",
        "supported_file_types": ["PDF", "TXT"],
        "features": [
            "Query documents with similarity search",
            "Upload new files (PDF/TXT) to expand database",
            "Search across all uploaded files",
            "Contextual responses with metadata"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
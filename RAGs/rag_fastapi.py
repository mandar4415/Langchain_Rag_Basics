"""
FastAPI wrapper for RAG Question Answering System
This preserves all the tested logic from 3_rag_one_off_question.py
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

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

class DatabaseInfo(BaseModel):
    status: str
    total_chunks: int
    database_path: str

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
        # Get database statistics
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "db", "chroma_db_with_metadata")
        
        # Try to get a sample to count total documents
        sample_docs = db.similarity_search("test", k=1000)  # Get many docs to estimate total
        
        return DatabaseInfo(
            status="connected",
            total_chunks=len(sample_docs),
            database_path=db_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database info: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Welcome message with API information"""
    return {
        "message": "Welcome to RAG Question Answering API",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/query",
        "database_info": "/database/info"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
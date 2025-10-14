#Metdata; gives more relevant and exact details about chunks retrived.
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with enhanced metadata
    documents = []
    # RecursiveCharacterTextSplitter is better at respecting chunk size limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs, then lines, then spaces, then characters
    )
    
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding='utf-8')
        book_docs = loader.load()
        
        for doc in book_docs:
            # Get the full text content
            full_text = doc.page_content
            
            # Split this document's text into chunks
            chunks = text_splitter.split_text(full_text)
            
            # Create enhanced chunks with position metadata
            current_pos = 0
            for chunk_text in chunks:
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
                    from langchain_core.documents import Document
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "source": book_file,
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
                            "context_after": context_after.strip()
                        }
                    )
                    documents.append(chunk_doc)
                    
                    # Update position for next search
                    current_pos = chunk_end_pos
                else:
                    # Fallback if exact position not found
                    from langchain_core.documents import Document
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            "source": book_file,
                            "start_line": "Unknown",
                            "end_line": "Unknown",
                            "start_char_pos": "Unknown",
                            "end_char_pos": "Unknown",
                            "start_word_pos": "Unknown",
                            "end_word_pos": "Unknown",
                            "document_percentage": "Unknown",
                            "chunk_size_chars": len(chunk_text),
                            "chunk_size_words": len(chunk_text.split())
                        }
                    )
                    documents.append(chunk_doc)

    # Add global chunk index to each document
    for i, doc in enumerate(documents):
        doc.metadata["chunk_index"] = i
        doc.metadata["chunk_id"] = f"chunk_{i:04d}"  # Format: chunk_0001, chunk_0002, etc.
    
    # Now docs is our documents list
    docs = documents

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print("Sample chunk metadata:")
    if docs:
        print(f"  Chunk 0: {docs[0].metadata}")
        if len(docs) > 1:
            print(f"  Chunk 1: {docs[1].metadata}")

    # Create embeddings
    print("\n--- Creating embeddings ---")
     # Get Gemini API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )  # Using Google Gemini embeddings for consistency with other project files
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
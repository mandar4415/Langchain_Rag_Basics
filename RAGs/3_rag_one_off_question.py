import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# Define the embedding model
api_key = os.getenv("GEMINI_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
    )  # Using Google Gemini embeddings for consistency with other project files

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
print("=== RAG Question Answering System ===")
print("Type your question below (or 'quit' to exit):")
query = input("\nEnter your question: ").strip()

if query.lower() in ['quit', 'exit', 'q']:
    print("Goodbye!")
    exit()

if not query:
    print("No question provided. Exiting.")
    exit()

print(f"\nProcessing query: '{query}'")
print("-" * 50)

# Use Chroma's direct similarity search with scores method to get actual similarity scores
# This returns tuples of (document, score) where score represents similarity
result_docs_with_scores = db.similarity_search_with_score(query, k=15)  # Get more results to account for filtering

# Filter and organize the results
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

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, entry in enumerate(relevant_docs, 1):
    doc = entry.get("doc")
    score = entry.get("score")
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")
    
    # Display similarity score with interpretation
    if score is not None:
        # Chroma uses cosine distance (lower = more similar)
        # Convert to similarity percentage for better understanding
        similarity_percentage = (1 - score) * 100 if score <= 1 else 0
        print(f"Similarity Score: {score:.3f} (Distance from query)")
        print(f"Similarity Percentage: {similarity_percentage:.1f}% (Higher = more relevant)")
    
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
    print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")

    # Show human-readable chunk boundaries
    first_words = doc.metadata.get('first_words', 'N/A')
    last_words = doc.metadata.get('last_words', 'N/A')
    if first_words != 'N/A' and last_words != 'N/A':
        print(f"Chunk Content: \"{first_words}...\" → \"...{last_words}\"")

    # Show precise technical details
    print(f"Position: Lines {doc.metadata.get('start_line', 'N/A')}-{doc.metadata.get('end_line', 'N/A')}, "
          f"Chars {doc.metadata.get('start_char_pos', 'N/A')}-{doc.metadata.get('end_char_pos', 'N/A')}")
    print(f"Document Progress: {doc.metadata.get('document_percentage', 'N/A')}%")
    print(f"Size: {doc.metadata.get('chunk_size_chars', 'N/A')} chars, {doc.metadata.get('chunk_size_words', 'N/A')} words")

    # Show context for better understanding
    context_before = doc.metadata.get('context_before', '')
    context_after = doc.metadata.get('context_after', '')
    if context_before or context_after:
        print(f"Context: \"...{context_before}\" [CHUNK] \"{context_after}...\"")
    print()

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([entry['doc'].page_content for entry in relevant_docs])
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Create a ChatOpenAI model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = llm.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
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
query = input("Enter your question: ")

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
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
    + "\n\n".join([doc.page_content for doc in relevant_docs])
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
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

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
query = "Where is Dracula's castle located?"

# Retrieve relevant documents based on the query
# Using similarity_score_threshold to filter by relevance score
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)
relevant_docs = retriever.invoke(query)

# If no documents found, let's check what's in the database
if not relevant_docs:
    print("No documents found with similarity search. Checking database content...")
    # Get some documents to see what's actually in the database
    all_retriever = db.as_retriever(search_kwargs={"k": 5})
    sample_docs = all_retriever.invoke("castle")
    if sample_docs:
        print(f"Found {len(sample_docs)} documents with 'castle' search")
    else:
        print("Database appears to be empty or incompatible")

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}")
    print(f"Chunk Index: {doc.metadata['chunk_index']}")
    print(f"Chunk ID: {doc.metadata['chunk_id']}")
    
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


# combined_input = (
#     "Here are some documents that might help answer the question: "
#     + query
#     + "\n\nRelevant Documents:\n"
#     + "\n\n".join([doc.page_content for doc in relevant_docs])
#     + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
# )

# # Create a ChatOpenAI model
# model = ChatOpenAI(model="gpt-4o")

# # Define the messages for the model
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content=combined_input),
# ]

# # print(messages, "messages")

# # Invoke the model with the combined input
# result = model.invoke(messages)

# # Display the full result and content only
# print("\n--- Generated Response ---")
# print("Full result:")
# # print(result)
# print("Content only:")
# print(result.content)
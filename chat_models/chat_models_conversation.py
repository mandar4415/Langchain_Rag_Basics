from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key from environment
api_key = os.getenv("GEMINI_API_KEY")

# Instantiate Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
messages = [
    SystemMessage("You are an expert in social media marketing."),
    HumanMessage("Give a short tip to create engaging posts on Instagram.")
    
]

# Ask Gemini a question
result = llm.invoke(messages)
print(result.content)
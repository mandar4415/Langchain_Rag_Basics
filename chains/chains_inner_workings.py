from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

# Get Gemini API key from environment
api_key = os.getenv("GEMINI_API_KEY")

# Instantiate Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
     [
        ("system", "You love facts and you tell facts about {animal}"),
        ("human", "Tell me {count} facts."),
    ]
)

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x)) 
#here RunnableLamda is ; integrate custom Python functions or lambda expressions into a chain.
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)
#RunnableSequence class chains multiple RunnableLambda together in a sequence.
# Run the chain
response = chain.invoke({"animal": "alligator", "count": 1})

# Output
print(response)
# Langchain Project

This repository contains my hands-on work with Langchain basics in Python. The project demonstrates prompt templates, chains, conditional logic, parallel chains, and integration with Gemini (Google Generative AI).

## What’s Included
- Langchain basics: prompt templates, chains, output parsers, conditional and parallel chains
- Gemini API integration for LLM tasks
- Example scripts for chat, prompt engineering, and chain composition

## Next Steps
- I will add Retrieval-Augmented Generation (RAG) basics and advanced workflows
- All RAG test files will be placed in the `documents/` folder (ignored from version control)

## Usage
1. Clone the repository
2. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your-gemini-api-key
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run any example script:
   ```
   python chains/chains_basics.py
   ```

## Folder Structure
```
Langchain_T/
├── chains/
├── chat_models/
├── prompt_templates/
├── documents/        # For RAG test files (ignored)
├── .env              # Your API keys (ignored)
├── .gitignore
├── README.md
```

## License
MIT

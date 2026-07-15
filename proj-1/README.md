# University Handbook RAG App

This project uses:
- Groq for the LLM
- Hugging Face embeddings
- Chroma as the vector store
- LangChain for the RAG pipeline

## Setup
1. Create a .env file based on .env.example and add your Groq API key.
2. Install the required packages:
   pip install python-dotenv langchain langchain-community langchain-core langchain-groq langchain-huggingface langchain-chroma langchain-text-splitters pypdf sentence-transformers chroma
3. Run the app:
   python app1.py --pdf path/to/students_handbook.pdf

## Usage
- Run interactively:
  python app1.py --pdf students_handbook.pdf
- Ask a single question:
  python app1.py --pdf students_handbook.pdf --query "What is the attendance policy?"

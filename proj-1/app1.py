# importings

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

# loading the file

loader = PyPDFLoader(r"C:\Users\\Ahmed tariq\\OneDrive\\Documents\\langchain\\proj-1\\docs\\Student-Handbook-2024_compressed.pdf") 
  # same PDF you used before
docs = loader.load()

# Check if documents have content
if not docs or not docs[0].page_content.strip():
    print("Warning: No content extracted from the PDF. Please ensure the PDF is text-searchable.")
    if docs:
        print(f"First document's page_content: '{docs[0].page_content}'")
else:
    print(f"Successfully loaded PDF. First 100 characters of first document: {docs[0].page_content[:100]}")


# --- 2. Split into chunks ---
# Analogy: you don't hand the LLM the whole textbook, just the relevant paragraphs.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# embeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create a vectore store from the chunks and embeddings 

vectorstore = Chroma.from_documents(
    documents = chunks, 
    embedding = embeddings, 
    persist_directory = "db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# llm 

llm = ChatGroq(
    groq_api_key = os.environ.get("Api_key").strip(),
    model="llama-3.3-70b-versatile",
    temperature=0.1
)


# --- 6. Format retrieved docs into a single string ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



# this is for simple rage without memory so i just
# comment out it for now.

# --- 5. Prompt template ---
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about the University of Lahore
using ONLY the context provided below. If the answer isn't in the context, say
you don't have that information — do not make it up.

Context:
{context}

Question: {question}

Answer:
""")

# now form the chain 

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}

    | prompt 
    | llm 
    | StrOutputParser()
)



# main 


if __name__ == "__main__":
    question = "What is the attendance policy?"
    answer = rag_chain.invoke(question)
    print(answer)







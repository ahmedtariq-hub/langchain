
# imprting 
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# for memory keeping 
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
# fro an agent
from langchain_core.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# --- 1. Build the UOL retriever (same pattern as before) ---
loader = PyPDFLoader(r"C:\Users\\Ahmed tariq\\OneDrive\\Documents\\langchain\\proj-1\\docs\\Student-Handbook-2024_compressed.pdf")
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Tools 

@tool
def lookup_uol_docs(query: str) -> str:
    """Searchs for the University of lahore student handbook for info about UOL specifically 
    — fees, programs, admissions, policies."""""

    results = retriever.invoke(query)
    return format_docs(results)

web_search = DuckDuckGoSearchRun()

web_search.description = (
        "Searches the live web. Use this to find current information about "
    "OTHER universities (not UOL) — their fee structures, BSCS programs, "
    "admission requirements, etc. Call this once per university you need "
    "information about."
)

tools = [lookup_uol_docs, web_search]

# LLM 
llm = C=ChatGroq(
    groq_api_key = os.environ.get("Api_key").strip(),
    model_name = "llama-3.1-8b-instant",
    temperature = 0.1
)


# --- 4. System prompt guiding the comparison behavior ---
SYSTEM_PROMPT = """You are a university research assistant that helps students
compare programs across universities in Pakistan.

When asked to compare universities on ANY criteria (fees, attendance policy,
admission requirements, etc.):
1. Use lookup_uol_info for anything about University of Lahore specifically.
2. Use web_search separately for EACH other university mentioned, using a
   query that matches what the user actually asked about.
3. Present a clear comparison — a table if useful, or plain text if a table
   doesn't fit naturally.
4. If you cannot find reliable information for a university on the requested
   topic, say so explicitly rather than guessing or forcing irrelevant data
   into the answer.
"""

# A gent with memory
checkpoint = InMemorySaver()
agent = create_agent(
    llm, tools, system_prompt = SYSTEM_PROMPT, checkpointer=checkpoint
)



# main 
# --- 6. Test ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "compare_thread_1"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content":
            "Compare the BSCS program fees at University of Lahore, FAST-NUCES, and NUST."}]},

        config=config
    )
    print(result["messages"][-1].content)












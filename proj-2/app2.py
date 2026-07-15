import os
os.environ["HF_HUB_OFFLINE"] = "1"
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder





load_dotenv()

docs = PyPDFLoader(r"C:\Users\\Ahmed tariq\\OneDrive\\Documents\\langchain\\proj-1\\docs\\Student-Handbook-2024_compressed.pdf").load()

if not docs or not docs[0].page_content.strip():
    print("no content extracted from pdf")

else:
    print(f"successfully loaded first 100 charcter of pdf : {docs[0].page_content[:100]}")


# --- 2. Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)


# --- 3. Embeddings + vectorstore ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# --- 4. LLM ---
llm = ChatGroq(
    groq_api_key=os.environ.get("Api_key").strip(),
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

# formate the docs 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# question rewriting chain (uses history to reslve vauges)

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, "
               "rewrite it as a standalone question that makes sense "
               "without the chat history. Do NOT answer it, just rewrite it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
contextualize_chain = contextualize_prompt | llm | StrOutputParser()


# --- 6. Main QA prompt ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant answering questions about the "
               "University of Lahore using ONLY the context below. If the "
               "answer isn't in the context, say you don't know.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])



def get_context(inputs):
    standalone_question = contextualize_chain.invoke(inputs)
    docs = retriever.invoke(standalone_question)
    return format_docs(docs)



# --- 7. Full chain (parallel dict + sequential pipe) ---
rag_chain_with_memory = (
    {
        "context": get_context,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

# --- 8. Memory wrapper ---
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain_with_memory,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# --- 9. Test ---
if __name__ == "__main__":
    session_id = "test_session_1"

    r1 = conversational_rag.invoke(
        {"question": "What is the attendance policy?"},
        config={"configurable": {"session_id": session_id}}
    )
    print("Answer 1:", r1)

    r2 = conversational_rag.invoke(
        {"question": "What happens if I violate it?"},
        config={"configurable": {"session_id": session_id}}
    )
    print("Answer 2:", r2)



import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

DB_PATH = "faiss_db"
st.set_page_config(page_title="AI Research Assistant", page_icon="🤖")
st.title("🤖 AI Research Assistant")


# Sidebar--------------------------------------------------------

st.sidebar.header("📂 Upload Sources")

# URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

# PDFs
pdf_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Excel
excel_files = st.sidebar.file_uploader(
    "Upload Excel Files",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

# CSV
csv_files = st.sidebar.file_uploader(
    "Upload CSV Files",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=True
)

process_btn = st.sidebar.button("Process Data")

# Embeddings------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Detect CSV Separator

def detect_separator(file):

    sample = file.read(4096).decode("utf-8", errors="ignore")
    file.seek(0)

    if "\t" in sample:
        return "\t"
    elif ";" in sample:
        return ";"
    else:
        return ","


# Process Documents

if process_btn:

    documents = []

    with st.spinner("Processing documents..."):

        # -------- URLs --------
        if urls:
            loader = UnstructuredURLLoader(urls=urls)
            documents.extend(loader.load())

        # -------- PDFs --------
        if pdf_files:
            for file in pdf_files:

                with open(file.name, "wb") as f:
                    f.write(file.getbuffer())

                loader = PyPDFLoader(file.name)
                documents.extend(loader.load())

        # -------- Excel --------
        if excel_files:
            for file in excel_files:

                df = pd.read_excel(file)

                for index, row in df.iterrows():

                    row_text = " | ".join(
                        [f"{col}: {row[col]}" for col in df.columns]
                    )

                    documents.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "source": file.name,
                                "row": index + 1
                            }
                        )
                    )

        # -------- CSV--------
        if csv_files:
            for file in csv_files:

                sep = detect_separator(file)

                df = pd.read_csv(
                    file,
                    sep=sep,
                    encoding="utf-8",
                    engine="python",
                    on_bad_lines="skip"
                )

                for index, row in df.iterrows():

                    row_text = " | ".join(
                        [f"{col}: {row[col]}" for col in df.columns]
                    )

                    documents.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "source": file.name,
                                "row": index + 1
                            }
                        )
                    )

        # -------- Vectorize --------
        if documents:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            embeddings = get_embeddings()

            db = FAISS.from_documents(chunks, embeddings)

            db.save_local(DB_PATH)

            st.success("All data indexed successfully")

        else:
            st.warning("Please upload some files first.")

# Load FAISS DB
def load_db():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    return None


db = load_db()

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# Chat Memory

if "history_store" not in st.session_state:
    st.session_state.history_store = {}


def get_history(session_id):

    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = InMemoryChatMessageHistory()

    return st.session_state.history_store[session_id]


# Helper
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# Build RAG Chain

def build_rag_chain():

    retriever = db.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([

        ("system",
         "You are a helpful AI assistant. "
         "Answer only from context. "
         "If answer is missing, say you don't know."),

        MessagesPlaceholder("chat_history"),

        ("human", "{question}"),

        ("system", "Context:\n{context}")
    ])

    get_question = RunnableLambda(lambda x: x["question"])

    get_context = get_question | retriever | format_docs

    rag_chain = (

        {
            "question": get_question,
            "context": get_context,
            "chat_history": lambda x: x["chat_history"]
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = build_rag_chain() if db else None


# -----------------------
# Chat UI
# -----------------------
st.subheader("💬 Chat")

query = st.chat_input("Ask anything from your data...")

if query and rag_chain:

    session_id = "user_session"

    rag_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # User
    with st.chat_message("user"):
        st.write(query)

    # Assistant
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = rag_with_memory.invoke(
                {"question": query},
                config={"configurable": {"session_id": session_id}}
            )

            st.write(answer)

elif query and not db:

    st.warning("⚠️ Upload and process files first.")

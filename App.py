import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "docs/movies.md"
DB_DIR = "vector_db"

# -------------------------------
# LOAD & SPLIT DATA
# -------------------------------
def load_documents():
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    docs = splitter.split_documents(documents)
    return docs

# -------------------------------
# CREATE VECTOR DB
# -------------------------------
def create_vector_db():
    docs = load_documents()
    
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=DB_DIR
    )
    
    db.persist()
    return db

# -------------------------------
# LOAD VECTOR DB
# -------------------------------
def load_vector_db():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    return db

# -------------------------------
# RAG CHAIN
# -------------------------------
def get_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    llm = OpenAI(temperature=0.5)  # requires API key
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="RAG Recommendation System")

st.title("🎬 AI Recommendation System (RAG)")

st.write("Ask for recommendations like:")
st.write("- Action movies")
st.write("- Emotional movies")
st.write("- Indian movies")

# Initialize DB
if "db_ready" not in st.session_state:
    if not os.path.exists(DB_DIR):
        with st.spinner("Creating vector database..."):
            db = create_vector_db()
    else:
        db = load_vector_db()
    
    st.session_state.db = db
    st.session_state.qa = get_qa_chain(db)
    st.session_state.db_ready = True

# Input
query = st.text_input("Enter your preference:")

# Output
if query:
    with st.spinner("Finding recommendations..."):
        result = st.session_state.qa(query)
        
        st.subheader("🎯 Recommendations:")
        st.write(result["result"])
        
        st.subheader("📄 Source Data:")
        for doc in result["source_documents"]:
            st

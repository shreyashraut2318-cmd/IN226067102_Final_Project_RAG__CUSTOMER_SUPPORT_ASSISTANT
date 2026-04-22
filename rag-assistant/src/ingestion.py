import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Define the directory where data will be stored
DB_PATH = "./chromadb_storage"

def ingest_docs(file_path: str):
    """
    Step 1 & 2: Load PDF and convert to searchable Vector Store
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 1. Load the PDF 
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Chunking strategy [cite: 37, 61]
    # We use 500 characters with 50 overlap to keep context intact 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Embedding Model 
    # This turns text into numbers the computer understands
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Store in ChromaDB 
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print(f"✅ Ingestion Complete! {len(chunks)} chunks stored in {DB_PATH}")
    return vector_db

def get_retriever():
    """
    Helper function to load the existing database for queries
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 3}) # Retrieves top 3 most relevant chunks 
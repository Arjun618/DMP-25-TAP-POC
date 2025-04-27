"""
Vector database management for TAP AI Assistant
"""
import glob
import os
import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

from config import DATA_DIR, INDEX_PATH, EMBEDDING_MODEL

def ingest_data(data_dir: str = DATA_DIR, index_path: str = INDEX_PATH):
    """
    Load and process CSV files, create embeddings, and build a vector store
    
    Args:
        data_dir: Directory containing CSV data files
        index_path: Path to save the FAISS index
        
    Returns:
        FAISS vector store object
    """
    # Load and split CSV documents from all files in data directory
    all_docs = []
    for csv_file in glob.glob(os.path.join(data_dir, "*.csv")):
        loader = CSVLoader(file_path=csv_file)
        all_docs.extend(loader.load())
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def load_vectorstore(data_dir: str = DATA_DIR, index_path: str = INDEX_PATH):
    """
    Load an existing vector store or create a new one if it doesn't exist
    
    Args:
        data_dir: Directory containing CSV data files
        index_path: Path to the FAISS index
        
    Returns:
        FAISS vector store object
    """
    # Ensure absolute paths for both parameters
    data_dir = os.path.abspath(data_dir)
    index_path = os.path.abspath(index_path)
    
    try:
        # Check if the index path exists
        if os.path.exists(index_path):
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
                return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.warning(f"Error loading existing vectorstore: {e}")
                st.info("Creating a new vector index...")
                return ingest_data(data_dir, index_path)
        else:
            st.info("Vector index not found. Creating a new one...")
            return ingest_data(data_dir, index_path)
    except Exception as e:
        st.error(f"Unexpected error in vectorstore handling: {e}")
        # Last resort - recreate the index
        return ingest_data(data_dir, index_path)
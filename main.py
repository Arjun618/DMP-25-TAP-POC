"""
TAP AI Assistant - Main entry point
"""
import traceback
import streamlit as st

# Import modules from our modular structure
from config import DATA_DIR
from vectorstore import load_vectorstore
from llm import init_language_model
from ui import setup_page, verify_environment, display_error, create_query_interface

def main():
    """Main entry point for the TAP AI Assistant application"""
    try:
        # Set up the page and UI elements
        setup_page()
        
        # Verify environment and show errors if any
        is_valid, error_msg = verify_environment()
        if not is_valid:
            suggestion = ""
            if "HUGGINGFACEHUB_API_TOKEN" in error_msg:
                suggestion = "Please add your Hugging Face API token to your environment variables or .env file."
            elif "data directory" in error_msg:
                suggestion = "Please make sure your data directory exists at the correct location."
            elif "Missing required dataset" in error_msg:
                suggestion = f"Please ensure these files exist in {DATA_DIR}"
                
            display_error(error_msg, suggestion)
            return
        
        # Try to initialize vector store and give clear error if it fails
        try:
            with st.spinner('Loading vector store...'):
                # Initialize vector store
                vectorstore = load_vectorstore()
                if vectorstore is None:
                    display_error(
                        "Failed to load or create vector store.",
                        "Check the faiss_index directory and ensure it's properly set up."
                    )
                    return
                
                # Create retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            display_error(
                f"Error initializing vector store: {str(e)}",
                "Please check console logs for more details."
            )
            return
            
        # Initialize language model
        llm = init_language_model()
        if llm is None:
            return  # Error already displayed in init_language_model
        
        # Create the query interface
        create_query_interface(retriever, llm)
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error(traceback.format_exc())
        st.info("Please try reloading the page. If the issue persists, check the console logs for details.")

if __name__ == "__main__":
    main()
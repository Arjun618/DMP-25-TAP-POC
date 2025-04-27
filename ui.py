"""
Streamlit UI components for TAP AI Assistant
"""
import os
import streamlit as st
from config import DATA_DIR, HUGGINGFACE_API_TOKEN
from query_processing import process_query
from data_processing import analyze_dataset

def setup_page():
    """Set up the page configuration and header"""
    # Set page config first (must be called before any other Streamlit command)
    st.set_page_config(
        page_title="TAP AI Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    # Display app header
    st.title("TAP AI Assistant")
    st.markdown("Ask questions about your student and teacher data in natural language")

def verify_environment():
    """
    Verify that the environment is properly set up
    
    Returns:
        Tuple of (bool, str) indicating success and error message if any
    """
    # Check for the Hugging Face API token
    if not HUGGINGFACE_API_TOKEN:
        return False, "HUGGINGFACEHUB_API_TOKEN not set in environment."
    
    # Check if data directory exists
    abs_data_dir = os.path.abspath(DATA_DIR)
    if not os.path.exists(abs_data_dir) or not os.path.isdir(abs_data_dir):
        return False, f"Data directory not found: {abs_data_dir}"
    
    # Check if dataset files exist
    student_file = os.path.join(abs_data_dir, "student_dataset.csv")
    teacher_file = os.path.join(abs_data_dir, "teacher_dataset.csv")
    
    missing_files = []
    if not os.path.exists(student_file):
        missing_files.append("student_dataset.csv")
    
    if not os.path.exists(teacher_file):
        missing_files.append("teacher_dataset.csv")
        
    if missing_files:
        return False, f"Missing required dataset files: {', '.join(missing_files)}"
    
    return True, ""

def display_error(message, suggestion=""):
    """Display an error message with a suggestion if provided"""
    st.error(f"⚠️ {message}")
    if suggestion:
        st.info(suggestion)

def create_query_interface(retriever, llm):
    """
    Create the query interface for user interaction
    
    Args:
        retriever: Document retriever object
        llm: Language model for generating responses
    """
    # Display dataset summary at the top
    dataset_summary = analyze_dataset("")
    with st.expander("📊 Dataset Summary & Details", expanded=True):
        st.markdown(dataset_summary)

    # Create a response container early to handle any errors
    response_container = st.container()
    
    # Create a form that allows Enter key to submit
    with st.form(key='query_form'):
        query = st.text_input("Enter your query:", placeholder="E.g., How many students whose name starts with A?")
        submit_button = st.form_submit_button(label='Send')
    
    # Example queries that users can click
    st.markdown("### Example queries:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Count students with name starting with A"):
            query = "What is the total number of students whose name starts with A?"
            process_query(query, retriever, llm, response_container)
            
        if st.button("List students in Coding Advanced"):
            query = "How many students are enrolled in Coding Advanced program?"
            process_query(query, retriever, llm, response_container)
    
    with col2:
        if st.button("Teachers who haven't submitted feedback"):
            query = "Which teachers haven't submitted feedback yet?"
            process_query(query, retriever, llm, response_container)
            
        if st.button("Student performance summary"):
            query = "Give me a summary of student performance across coding and math scores"
            process_query(query, retriever, llm, response_container)
    
    # Handle form submission
    if submit_button:
        if not query.strip():
            st.warning("Please enter a query")
        else:
            process_query(query, retriever, llm, response_container)
            
    return response_container
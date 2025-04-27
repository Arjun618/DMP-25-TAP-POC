"""
Query processing and RAG pipeline for TAP AI Assistant
"""
import re
import streamlit as st
import traceback

from data_processing import analyze_dataset, filter_students_by_name_start
from config import REFERENCE_DATE

def create_qa_prompt(context, query):
    """
    Create a comprehensive prompt for the RAG system with detailed instructions
    
    Args:
        context: The context information to provide to the LLM
        query: The user's query
        
    Returns:
        Formatted prompt string for the LLM
    """
    return (
        "You are a helpful AI assistant for The Apprentice Project (TAP). "
        "Your task is to answer questions about student and teacher data using the provided dataset information. "
        f"The current date is {REFERENCE_DATE}.\n\n"
        
        "Important instructions:\n"
        "1. When answering questions, be detailed and precise.\n"
        "2. If asked to filter or count data based on specific conditions (like names starting with a letter, "
        "scores above/below a threshold, etc.), perform the filtering mentally and give the exact count.\n"
        "3. If asked for statistics or aggregations, calculate them based on the information provided.\n"
        "4. When answering 'how many' questions, always provide a specific number.\n"
        "5. For complex queries, explain your reasoning step by step.\n"
        "6. If the data doesn't contain information to answer the question, clearly state that.\n\n"
        
        f"Context Information:\n{context}\n\n"
        
        f"Question: {query}\n"
        "Answer:"
    )

def process_query(query, retriever, llm, response_container):
    """
    Process a user query using the RAG pipeline and display results
    
    Args:
        query: The user's query string
        retriever: The document retriever object
        llm: The language model for generating responses
        response_container: Streamlit container for displaying the response
    """
    with st.spinner('Processing your query...'):
        try:
            # Check for common name-based queries that we can handle directly
            q_lower = query.lower()
            
            # Handle name starting with specific letter queries efficiently
            if ("name" in q_lower and "start" in q_lower) or ("whose name begins with" in q_lower) or ("name starting with" in q_lower):
                # Try to extract the letter
                letter_candidates = []
                # Look for patterns like "starts with A" or "starting with A"
                name_pattern = r"start(?:s|ing)?\s+with\s+([a-zA-Z])"
                match = re.search(name_pattern, q_lower)
                if match:
                    letter_candidates.append(match.group(1).upper())
                
                # Look for "with A" pattern
                name_pattern2 = r"with\s+([a-zA-Z])[^a-zA-Z]"
                match = re.search(name_pattern2, q_lower)
                if match:
                    letter_candidates.append(match.group(1).upper())
                    
                # Look for letter within quotes
                name_pattern3 = r"['\"]([a-zA-Z])['\"]"
                match = re.search(name_pattern3, q_lower)
                if match:
                    letter_candidates.append(match.group(1).upper())
                
                # If we found a candidate letter to filter by
                if letter_candidates:
                    count, names = filter_students_by_name_start(letter_candidates[0])
                    if count > 0:
                        response = f"There are {count} students whose names start with '{letter_candidates[0]}'."
                        if names:
                            if len(names) <= 10:
                                response += f" Their names are: {', '.join(names)}."
                            else:
                                response += f" The first 10 names are: {', '.join(names)}."
                        
                        # Display the response in a read-only text area inside the container
                        with response_container:
                            st.text_area("Response", value=response, height=200, disabled=True)
                        return
            
            # Check for other specific patterns that we can handle directly
            if "how many students" in q_lower and "in" in q_lower and "program" in q_lower:
                # Try to identify program name
                programs = ["Coding Advanced", "Coding Basics", "Science Club", "Math Explorers"]
                for program in programs:
                    if program.lower() in q_lower:
                        try:
                            import os
                            import pandas as pd
                            from config import DATA_DIR
                            
                            student_file = os.path.join(os.path.abspath(DATA_DIR), "student_dataset.csv")
                            student_df = pd.read_csv(student_file)
                            count = len(student_df[student_df["program"] == program])
                            response = f"There are {count} students enrolled in the {program} program."
                            
                            # Display the response
                            with response_container:
                                st.text_area("Response", value=response, height=200, disabled=True)
                            return
                        except Exception as e:
                            # If direct handling fails, fall through to general approach
                            st.warning(f"Direct filtering attempt failed: {e}")
                            pass
            
            # Enhanced RAG pipeline for all other queries
            # Step 1: Get dataset information and format it for the LLM
            analytics_context = analyze_dataset(query)
            
            # Step 2: Also retrieve relevant document chunks for additional context
            try:
                docs = retriever.get_relevant_documents(query)
                st.session_state.retrieved_docs = [doc.page_content for doc in docs]
            except Exception as retriever_error:
                st.warning(f"Error retrieving documents: {retriever_error}")
                docs = []
                st.session_state.retrieved_docs = []
            
            # Step 3: Combine contexts if documents were found
            if docs and len(docs) > 0:
                # Limit the size of each document to prevent context overload
                max_doc_chars = 1000
                trimmed_docs = [d.page_content[:max_doc_chars] + ("..." if len(d.page_content) > max_doc_chars else "") for d in docs]
                doc_context = "\n\n".join(trimmed_docs)
                combined_context = f"{analytics_context}\n\n## Additional Relevant Context:\n{doc_context}"
            else:
                combined_context = analytics_context
            
            # Step 4: Create a comprehensive prompt for the LLM with improved instructions
            prompt = create_qa_prompt(combined_context, query)
            
            # Step 5: Let the LLM generate the answer with error handling
            try:
                response = llm(prompt)
                
                # Clean up the response if necessary
                response = response.strip()
                
                # Remove any generic prefixes that the model might add
                prefixes_to_remove = ["Answer:", "I'll help you answer this question:", "Based on the information provided:"]
                for prefix in prefixes_to_remove:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()
            except Exception as llm_error:
                st.error(f"Error generating response: {llm_error}")
                response = "I'm sorry, I encountered an error while processing your query. Please try again or rephrase your question."
            
            # Display the response in a read-only text area inside the container
            with response_container:
                st.text_area("Response", value=response, height=200, disabled=True)
                
                # Add an expandable section to show the retrieved documents (for debugging)
                if st.session_state.get("retrieved_docs"):
                    with st.expander("Debug: View retrieved documents"):
                        for i, doc in enumerate(st.session_state.retrieved_docs):
                            st.markdown(f"**Document {i+1}:**")
                            st.text(doc[:500] + ("..." if len(doc) > 500 else ""))
                
        except Exception as e:
            st.error(f"Error processing your query: {str(e)}")
            st.error(traceback.format_exc())
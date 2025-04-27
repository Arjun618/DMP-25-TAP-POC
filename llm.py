"""
Language model initialization and configuration for TAP AI Assistant
"""
import os
import streamlit as st
from langchain.llms import HuggingFaceHub, CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    LLM_MODEL, 
    HUGGINGFACE_API_TOKEN, 
    USE_LOCAL_MODEL, 
    LOCAL_MODEL_PATH,
    LOCAL_MODEL_TYPE,
    MODEL_MAX_TOKENS
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def create_remote_llm_with_retry():
    """Create LLM with retry logic for handling connection issues"""
    return HuggingFaceHub(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
        model_kwargs={
            "temperature": 0.6,
            "max_length": MODEL_MAX_TOKENS
        },
        # Add callback manager for better output handling
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        # Add request timeout parameters
        huggingfacehub_api_timeout=120  # Increase timeout to 120 seconds
    )

def create_local_llm():
    """Create a local LLM using CTransformers backend"""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

    # Check if model file exists, if not provide instructions
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.error(f"Local model file not found at: {LOCAL_MODEL_PATH}")
        st.info(
            "Please download the model file manually from Hugging Face and place it in "
            f"the correct location: {LOCAL_MODEL_PATH}\n\n"
            f"You can download a compatible model from: https://huggingface.co/{LLM_MODEL}"
        )
        return None
    
    # Create LLM instance with local model
    return CTransformers(
        model=LOCAL_MODEL_PATH,
        model_type=LOCAL_MODEL_TYPE,
        config={
            "temperature": 0.6,
            "max_new_tokens": MODEL_MAX_TOKENS,
            "context_length": 2048,  # Adjust based on your model and available memory
        },
        callbacks=[StreamingStdOutCallbackHandler()]
    )

def init_language_model():
    """
    Initialize the language model, either local or remote based on configuration
    
    Returns:
        Initialized language model or None if initialization fails
    """
    try:
        if USE_LOCAL_MODEL:
            with st.spinner('Loading local language model...'):
                st.info("Using local model for inferencing. This may take some time to initialize.")
                
                try:
                    llm = create_local_llm()
                    if llm:
                        st.success("Local model loaded successfully!")
                        return llm
                except Exception as local_error:
                    st.error(f"Error initializing local language model: {local_error}")
                    st.info(
                        "Make sure you have downloaded the model file and placed it in the correct location.\n"
                        "You may need to allocate more memory or use a smaller quantized model."
                    )
                    return None
        else:
            # Use remote Hugging Face API (original implementation)
            with st.spinner('Connecting to Hugging Face API...'):
                if not HUGGINGFACE_API_TOKEN:
                    st.error("Missing Hugging Face API token. Please set HUGGINGFACEHUB_API_TOKEN in your .env file or environment variables.")
                    return None
                    
                try:
                    llm = create_remote_llm_with_retry()
                    return llm
                except Exception as retry_error:
                    st.error(f"Connection to Hugging Face API failed after multiple attempts: {retry_error}")
                    st.info("The Hugging Face servers might be experiencing high load or there might be network connectivity issues.")
                    return None
                
    except Exception as e:
        st.error(f"Error initializing language model: {e}")
        if USE_LOCAL_MODEL:
            st.info("Error loading local model. Check your model configuration and file paths.")
        else:
            st.info(f"Could not connect to model {LLM_MODEL}. Check your internet connection and API token validity.")
        return None
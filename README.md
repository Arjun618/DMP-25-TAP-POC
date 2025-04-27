# TAP AI Assistant

This project is a POC for an internal AI assistant for The Apprentice Project (TAP). It answers natural language queries about student and teacher data using a hybrid Retrieval-Augmented Generation (RAG) pipeline with FAISS vector search and a Large Language Model (LLM).

---

## 🚀 Features

- **Natural Language Queries**: Ask questions about student and teacher data in plain English
- **Local or Cloud LLM**: Use Llama-2-7B locally or Falcon-7B-Instruct via Hugging Face API
- **Robust RAG Pipeline**: Combines vector search with structured data analysis for accurate responses
- **Interactive UI**: User-friendly Streamlit interface with example queries and detailed responses
- **Modular Architecture**: Well-organized code structure for easy maintenance and extension
- **Debug Mode**: Optional expandable section to see retrieved documents (for troubleshooting)

---

## 📁 Project Structure

```
TAP_ONLINE/
├── data/                    # CSV datasets directory
│   ├── student_dataset.csv  # Student information
│   └── teacher_dataset.csv  # Teacher information
├── faiss_index/             # Vector embeddings storage
├── models/                  # Local LLM model files directory (optional)
├── config.py                # Configuration settings
├── data_processing.py       # Data analysis functions
├── llm.py                   # Language model setup
├── main.py                  # Application entry point
├── query_processing.py      # RAG pipeline implementation
├── ui.py                    # Streamlit UI components
├── vectorstore.py           # FAISS vector database management
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ⚙️ Prerequisites

- Python 3.8+ (Windows recommended)
- Minimum 8GB RAM (16GB+ recommended for local LLM)
- Stable internet connection (for cloud LLM or model download)
- Hugging Face API token (required for cloud LLM mode)
- (Optional) GPU for faster local inferencing

---

## 🔧 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Arjun618/DMP-25-TAP-POC.git
   cd tap-ai-assistant
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv env
   # On Windows:
   .\env\Scripts\activate
   # On Linux/Mac:
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Hugging Face API token**
   - Create a `.env` file in the project root with this line:
     ```
     HUGGINGFACEHUB_API_TOKEN=your_token_here
     ```
   - You can get a free token from https://huggingface.co/settings/tokens

5. **(Optional) Use a local LLM**
   - Download a GGUF model file (e.g., Llama-2-7B-Chat) from Hugging Face
   - Place it in the `models/` directory
   - Set `USE_LOCAL_MODEL = True` in `config.py`

6. **Verify data files**
   - Ensure `data/student_dataset.csv` and `data/teacher_dataset.csv` are present

7. **Run the assistant**
   ```bash
   python -m streamlit run main.py
   ```
   - The web interface will open in your browser.

**That's it! Just clone, add your Hugging Face token, and it works out of the box.**

---

## 🖥️ Usage

- Type your question in the search box (e.g., "How many students are enrolled in Coding Advanced?")
- Click example queries for quick access
- View responses and expand the debug section for details

### Example Queries
- "How many students are enrolled in the Coding Advanced program?"
- "List all teachers who haven't submitted feedback in the past 2 weeks?"
- "What is the average math score for female students?"
- "How many students have names starting with A?"
- "Which location has the highest student attendance percentage?"

---

## 🛠️ Troubleshooting

- **API connection errors**: Check your internet and Hugging Face token
- **Local model loading failures**: Ensure the model file is in `models/` and path is correct
- **Out of memory errors**: Use a more quantized model or increase swap space
- **No module errors**: Activate your environment and install dependencies
- **Empty responses**: Try a simpler query or check the debug section
- **Slow responses**: First query may be slow (model loading); subsequent queries are faster

---

## 🧠 Implementation Details

- **Vector search**: CSV data is embedded using SentenceTransformers and indexed with FAISS
- **Structured data analysis**: SQL-like operations on dataframes for specific queries
- **LLM Options**:
  - Local: Llama-2-7B-Chat (CTransformers)
  - Cloud: Falcon-7B-Instruct (Hugging Face API)
- **Modular architecture**: Separate modules for data, UI, vector storage, and query handling

---

## 📊 Performance Metrics

- Tested with 500+ student and 150+ teacher records
- Average response time:
  - Cloud LLM: ~3-5 seconds
  - Local LLM: ~5-10 seconds initial load, ~2-4 seconds after
- ~95% accuracy on common data queries

---

## 🔮 Future Scope

- **Upgrade to Advanced LLMs**: Integrate more powerful language models such as GPT-4, Claude, or Llama-3 to drastically improve answer quality, reasoning, and processing speed.
- **Custom Analytics**: Add more advanced analytics and visualization features for deeper insights.
- **Role-based Access**: Implement user authentication and role-based data access.
- **Multi-language Support**: Enable queries and responses in multiple languages.
- **Deployment**: Package as a cloud or on-premise solution for broader TAP use.

---


## 📄 License

Internal use only. Not for public distribution.

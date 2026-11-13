# PDF Chat AI 📚

A Streamlit-based web application that allows you to chat seamlessly with multiple PDF documents using **LangChain**, **Google Gemini**, and a **FAISS Vector Database**.

---

## 📝 Overview

The Multi-PDF Chat Agent allows users to upload multiple PDF documents, extract and index their contents, and engage in real-time conversational question-answering based on the uploaded material.

---

## 🎯 How It Works

1. **PDF Text Extraction**: Reads uploaded PDF files and extracts text page-by-page.
2. **Text Chunking**: Breaks large documents into semantic chunks (`chunk_size=1000`, `chunk_overlap=200`).
3. **Embeddings & Vector Storage**: Computes vector embeddings using Google Gemini Embeddings (`models/gemini-embedding-001`) and indexes them locally with FAISS.
4. **Similarity Search**: Performs semantic vector similarity search to find the most relevant chunks for user questions.
5. **Conversational QA**: Generates detailed, contextual answers using Google Gemini (`gemini-3.5-flash-lite`).

---

## 🌟 Tech Stack & Requirements

- **Streamlit**: Web application framework and UI.
- **LangChain**: Retrieval and conversational QA pipeline.
- **Google GenAI / LangChain Google GenAI**: Embeddings and LLM inference.
- **FAISS (CPU)**: Local vector database for fast similarity search.
- **PyPDF2**: PDF reading and text extraction.
- **python-dotenv**: Environment configuration.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/MultiPDFChatApp.git
cd MultiPDFChatApp
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_actual_google_api_key_here
```
> Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 💡 How to Use

1. Launch the application in your browser.
2. In the sidebar, upload one or more PDF documents.
3. Click **"Submit & Process"** to extract text and generate the vector index.
4. Once processing is complete, type your question in the main chat prompt to get answers based on your documents.

---

## ©️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

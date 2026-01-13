# 🤖 LLM Memory Chatbot (RAG & History Aware)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![LangChain](https://img.shields.io/badge/Orchestration-LangChain%20v0.2-green)
![License](https://img.shields.io/badge/License-MIT-purple)

A comprehensive **Conversational AI Assistant** capable of answering questions based on your own documents (PDF, TXT, DOCX). 

This project implements a robust **RAG (Retrieval-Augmented Generation)** pipeline with **Memory**, allowing the AI to understand context from previous turns in the conversation. It is engineered to support multiple LLM providers seamlessly, including **OpenAI**, **Groq (Llama 3)**, **HuggingFace**, and local models via **Ollama**.

## 🧠 Key Features

* **📚 Multi-Document RAG:** Upload multiple PDF, DOCX, or TXT files simultaneously. The system vectorizes them locally using FAISS.
* **🗣️ Context-Aware Chat:** Unlike simple Q&A bots, this assistant reformulates your questions based on chat history. (e.g., If you ask "Who is he?" after discussing "Elon Musk", the bot understands who you refer to).
* **🔌 Multi-Provider Support:**
    * **Groq:** For ultra-fast inference (Llama 3 70B/8B).
    * **OpenAI:** For standard GPT-4o models.
    * **HuggingFace Hub:** For accessing open-source models remotely.
    * **Ollama:** For running models completely locally and offline.
* **🔎 Transparency:** Every answer includes **citations** (source file and page number) so you can verify the information.

## 🏗️ Architecture

The application follows a modern RAG architecture:

1.  **Ingestion:** Documents are loaded using `PyPDFLoader`, `TextLoader`, or `Docx2txtLoader`.
2.  **Splitting:** Text is divided into chunks (1000 chars) with overlap (200 chars) using `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Chunks are converted to vectors using **HuggingFace Embeddings** (`BAAI/bge-m3` or similar).
4.  **Storage:** Vectors are stored in a transient **FAISS** index.
5.  **Retrieval & Generation:**
    * *Step 1 (History):* The user's query is reformulated to include context from the chat history.
    * *Step 2 (Retrieve):* The system fetches relevant chunks from FAISS.
    * *Step 3 (Answer):* The LLM generates the final answer using the retrieved chunks.

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher.
* Git.

### 1. Install Dependencies

```bash
git clone [https://github.com/your-username/llm-memory-chatbot.git](https://github.com/your-username/llm-memory-chatbot.git)
cd llm-memory-chatbot
```

### 2. Clone the Repository

```
pip install -r requirements.txt
```

### 3. Environment Configuration

```
# .env

# [Option 1] OpenAI (GPT-4o, GPT-3.5)
OPENAI_API_KEY="sk-..."

# [Option 2] Groq (Llama 3 - Recommended for speed)
GROQ_API_KEY="gsk_..."

# [Option 3] HuggingFace Hub (Mistral, Phi-3, etc.)
HUGGINGFACEHUB_API_TOKEN="hf_..."
```

### Start the application using Strealit

```
streamlit run chat.py 
```

# 📚 PDF Chat – Chat with Your Documents using AI

This project is a Streamlit-based web application that allows users to upload PDF files and chat with them intelligently using a **Retrieval-Augmented Generation (RAG)** pipeline. It uses embedding-based vector search to retrieve relevant document chunks and generate context-aware answers with an LLM (Language Model).

---

## 🚀 Features

- 📁 Upload and parse multiple PDF documents
- 🔍 Semantic search using vector embeddings (Jina)
- 🧠 Intelligent Q&A with document context using LLM (Groq)
- 🧾 Session-based memory with temporary file storage
- 🔐 Uses `.env` for secure API key management
- 🧹 One-click session cleanup
- 📄 Shows sources used in responses for transparency

---

## 🧱 Architecture Overview
<pre>
                        ┌────────────────────┐
                        │   Streamlit Front  │
                        │     (UI + Chat)    │
                        └────────┬───────────┘
                                 │
                      Upload PDF │ Ask question
                                 ▼
                     ┌────────────────────┐
                     │ File Saved to Disk │
                     └────────┬───────────┘
                              │
                              ▼
      ┌──────────────────────────────────────────────┐
      │ Load & Chunk PDF using LangChain TextSplitter│
      └────────┬───────────────────────────▲─────────┘
               │                           │
     Store as Embeddings          Retrieve Relevant Chunks
 (Jina + Chroma Vectorstore)         using Retriever
               │                           │
               ▼                           ▼
     ┌────────────────┐      ┌────────────────────────┐
     │ Vector Database│ ◄────┤ Retrieval Chain (RAG)  │
     └────────────────┘      └──────────┬─────────────┘
                                        │
                                        ▼
                           Generate Answer via Groq LLM
                                        │
                                        ▼
                           Return Answer + Source Docs
</pre>
---

## 🧰 Tech Stack

| Component         | Tool/Library                                       | Purpose                           |
| ----------------- | -------------------------------------------------- | --------------------------------- |
| 🧠 LLM            | Groq LLaMA 3 via `langchain_groq`                  | Text generation                   |
| 📄 PDF Parsing    | `langchain_community.document_loaders.PyPDFLoader` | Load PDFs                         |
| ✂️ Chunking       | `RecursiveCharacterTextSplitter`                   | Split docs into meaningful chunks |
| 🔍 Embeddings     | Jina AI Embeddings v2                              | Generate vector representations   |
| 📦 Vector Store   | ChromaDB via `langchain.vectorstores`              | Store and retrieve chunks         |
| 💬 Interface      | Streamlit                                          | Web UI                            |
| 🔑 Env Management | python-dotenv                                      | Secure API key handling           |

---

## ⚙️ Setup Instructions

```bash

# Clone the Repo
git clone https://github.com/your-username/pdf-chat-rag.git
cd pdf-chat-rag

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

#Install dependencies
pip install -r requirements.txt

#Run the App
streamlit run app.py


```

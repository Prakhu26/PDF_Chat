import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import threading
import time

import streamlit as st
from dotenv import load_dotenv
import chromadb
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RAG Configuration
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# File upload configuration
ALLOWED_EXTENSIONS = {'pdf'}
BASE_UPLOAD_FOLDER = './uploads'
BASE_VECTORSTORE_FOLDER = './vectorstores'

# Create base directories
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BASE_VECTORSTORE_FOLDER, exist_ok=True)

@st.cache_resource
def init_models():
    """Initialize the embedding model and LLM"""
    try:
        if not os.environ.get("JINA_API_KEY"):
            raise ValueError("JINA_API_KEY not found in environment variables")
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment variables")
            
        embedding_model = JinaEmbeddings(
            jina_api_key=os.environ.get("JINA_API_KEY"),
            model_name=EMBED_MODEL_NAME,
        )
        
        llm = ChatGroq(
            temperature=LLM_TEMPERATURE,
            model_name=LLM_NAME,
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )
        
        return embedding_model, llm
        
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

def create_session_directories(session_id: str):
    """Create session-specific directories"""
    session_doc_dir = os.path.join(BASE_UPLOAD_FOLDER, session_id)
    session_vector_dir = os.path.join(BASE_VECTORSTORE_FOLDER, session_id)
    
    os.makedirs(session_doc_dir, exist_ok=True)
    os.makedirs(session_vector_dir, exist_ok=True)
    
    return session_doc_dir, session_vector_dir

def save_uploaded_file(uploaded_file, session_id: str):
    """Save uploaded file to session directory"""
    try:
        session_doc_dir, _ = create_session_directories(session_id)
        file_path = os.path.join(session_doc_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

@st.cache_resource
def load_and_process_documents(session_id: str, _embedding_model, _llm):
    """Load documents and create RAG chain for a session"""
    try:
        session_doc_dir = os.path.join(BASE_UPLOAD_FOLDER, session_id)
        session_vector_dir = os.path.join(BASE_VECTORSTORE_FOLDER, session_id)
        collection_name = f"collection_{session_id}"
        
        # Load documents
        documents = DirectoryLoader(
            session_doc_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        ).load()
        
        if not documents:
            return None, "No documents found"
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            chunks,
            embedding=_embedding_model,
            collection_name=collection_name,
            persist_directory=session_vector_dir,
        )
        
        # Create RAG chain
        template = """Answer the question based only on the following context.
        Think step by step before providing a detailed answer.
        
        <context>
        {context}
        </context>

        Question: {input}
        
        Answer: """
        
        prompt = ChatPromptTemplate.from_template(template)
        document_chain = create_stuff_documents_chain(llm=_llm, prompt=prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain, f"Successfully processed {len(chunks)} document chunks"
        
    except Exception as e:
        return None, f"Error processing documents: {str(e)}"

def cleanup_session(session_id: str):
    """Clean up session files and directories"""
    try:
        session_doc_dir = os.path.join(BASE_UPLOAD_FOLDER, session_id)
        session_vector_dir = os.path.join(BASE_VECTORSTORE_FOLDER, session_id)
        
        # Remove directories
        if os.path.exists(session_doc_dir):
            shutil.rmtree(session_doc_dir)
        if os.path.exists(session_vector_dir):
            shutil.rmtree(session_vector_dir)
            
        st.success(f"Session {session_id[:8]}... cleaned up successfully")
        
    except Exception as e:
        st.error(f"Error cleaning up session: {str(e)}")

def main():
    st.title("📚 PDF Chat")
    st.markdown("Upload PDF documents and chat with them using AI-powered retrieval")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    
    # Initialize models
    embedding_model, llm = init_models()
    
    if embedding_model is None or llm is None:
        st.error("Failed to initialize models. Please check your API keys.")
        st.stop()
    
    # Sidebar for file upload and session management
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to start chatting with it"
        )
        
        if uploaded_file is not None:
            if st.session_state.processed_file != uploaded_file.name:
                if st.button("Process Document", type="primary"):
                    with st.spinner("Processing document..."):
                        # Create new session
                        session_id = str(uuid.uuid4())
                        
                        # Save uploaded file
                        file_path = save_uploaded_file(uploaded_file, session_id)
                        
                        if file_path:
                            # Process document
                            rag_chain, message = load_and_process_documents(
                                session_id, embedding_model, llm
                            )
                            
                            if rag_chain:
                                st.session_state.session_id = session_id
                                st.session_state.rag_chain = rag_chain
                                st.session_state.processed_file = uploaded_file.name
                                st.session_state.chat_history = []
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                                cleanup_session(session_id)
        
        # Session info
        if st.session_state.session_id:
            st.header("📊 Session Info")
            st.info(f"**File:** {st.session_state.processed_file}")
            st.info(f"**Session ID:** {st.session_state.session_id[:8]}...")
            
            if st.button("🗑️ End Session", type="secondary"):
                cleanup_session(st.session_state.session_id)
                st.session_state.session_id = None
                st.session_state.rag_chain = None
                st.session_state.chat_history = []
                st.session_state.processed_file = None
                st.rerun()
        
        # Model info
        st.header("🤖 Model Info")
        st.caption(f"**Embedding:** {EMBED_MODEL_NAME}")
        st.caption(f"**LLM:** {LLM_NAME}")
        st.caption(f"**Temperature:** {LLM_TEMPERATURE}")
    
    # Main chat interface
    if st.session_state.rag_chain is None:
        st.info("👈 Please upload and process a PDF document to start chatting")
        
        # Show sample questions
        st.subheader("💡 What you can do:")
        st.markdown("""
        - **Upload PDF documents** and ask questions about their content
        - **Get contextual answers** based on the document content
        - **See source references** for transparency
        - **Chat naturally** with your documents
        """)
        
    else:
        # Chat interface
        st.subheader(f"💬 Chat with: {st.session_state.processed_file}")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    st.write(answer)
                    
                    if sources:
                        with st.expander("📄 Sources"):
                            for j, source in enumerate(sources):
                                st.markdown(f"**Source {j+1}:**")
                                st.caption(f"Page: {source.get('page', 'Unknown')}")
                                st.text(source.get('content_preview', ''))
                                st.divider()
        
        # Chat input
        question = st.chat_input("Ask a question about your document...")
        
        if question:
            # Add user question to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Process question
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke({"input": question})
                        answer = response['answer']
                        
                        # Extract source information
                        sources = []
                        for doc in response.get("context", []):
                            source_info = {
                                'source': doc.metadata.get('source', 'Unknown'),
                                'page': doc.metadata.get('page', 'Unknown'),
                                'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                            }
                            sources.append(source_info)
                        
                        st.write(answer)
                        
                        if sources:
                            with st.expander("📄 Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.caption(f"Page: {source.get('page', 'Unknown')}")
                                    st.text(source.get('content_preview', ''))
                                    st.divider()
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer, sources))
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()
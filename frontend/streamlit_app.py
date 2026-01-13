"""Streamlit UI for Agentic RAG System"""

import streamlit as st
import sys
import os

# --- 1. ROBUST PATH SETUP ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(page_title="🤖 UET Dept Agent", page_icon="🏫")

# --- 2. INITIALIZATION & MEMORY ---
def init_session_state():
    """Initialize session state for chat history"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    # Store chat history (Messages: role, content, citations)
    if "messages" not in st.session_state:
        st.session_state.messages = []

@st.cache_resource
def initialize_rag():
    """Initialize RAG with PDF ONLY"""
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        vector_store = VectorStore()
        
        # Load PDF from data folder
        pdf_path = os.path.join(project_root, "data")
        
        if not os.path.exists(pdf_path) or not any(f.endswith('.pdf') for f in os.listdir(pdf_path)):
            st.error(f"❌ No PDF found in {pdf_path}. Please add 'UET Prospectus.pdf'.")
            return None, 0
            
        documents = doc_processor.load_from_pdf_dir(pdf_path)
        chunks = doc_processor.split_documents(documents)
        vector_store.create_vectorstore(chunks)
        
        graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
        graph_builder.build()
        
        return graph_builder, len(chunks)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    init_session_state()
    st.title("🏫 UET Department Agent")
    st.markdown("Ask questions about the UET Prospectus.")
    
    # Initialize System
    if not st.session_state.rag_system:
        with st.spinner("Loading UET Prospectus..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.success(f"✅ System ready! ({num_chunks} chunks loaded)")

    # --- 3. DISPLAY CHAT HISTORY ---
    # Loop through history and display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If there are citations stored, display them
            if "citations" in message and message["citations"]:
                with st.expander("📄 Source Citations"):
                    for cit in message["citations"]:
                        st.caption(f"**Source:** {cit['source']}")
                        st.text(cit['text'])

    # --- 4. HANDLE NEW INPUT ---
    if prompt := st.chat_input("Ask a department-related question..."):
        
        # A. Display User Message Immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # B. Add User Message to History
        st.session_state.messages.append({"role": "user", "content": prompt})

        # C. Generate Response
        if st.session_state.rag_system:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_system.run(prompt)
                    answer_text = result['answer']
                    
                    # Display Answer
                    st.markdown(answer_text)
                    
                    # Process Citations
                    citation_data = []
                    if "I only answer" not in answer_text and result.get('retrieved_docs'):
                        with st.expander("📄 Source Citations"):
                            for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                                meta = doc.metadata
                                source = meta.get('source', 'Unknown')
                                page = meta.get('page', 'N/A')
                                
                                # Show in UI
                                st.caption(f"**Source {i}:** {source} (Page {page})")
                                st.text(doc.page_content[:200] + "...")
                                
                                # Save for History
                                citation_data.append({
                                    "source": f"{source} (Page {page})",
                                    "text": doc.page_content[:200] + "..."
                                })

                    # D. Add Assistant Message (with citations) to History
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer_text,
                        "citations": citation_data
                    })

if __name__ == "__main__":
    main()
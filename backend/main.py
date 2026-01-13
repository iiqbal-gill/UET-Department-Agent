import sys
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Path Setup to find 'src' ---
# Add the project root to python path so we can import 'src' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# --- 2. Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

class Citation(BaseModel):
    source: str
    page_content: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

# --- 3. App Definition ---
app = FastAPI(title="NLP Project RAG API")

# Global variable to hold the initialized graph
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on server startup"""
    global rag_system
    print("🚀 Initializing RAG System...")
    
    try:
        # Initialize LLM
        llm = Config.get_llm()
        
        # Initialize Processor & Vector Store
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Load Documents
        # Note: Guidelines say "Use only ONE PDF: UET Prospectus"
        # Ensure your PDF is in the 'data' folder or update this logic
        print("📂 Loading documents...")
        path_to_data = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Determine source: Prefer PDF in data/, fallback to Config URLs
        documents = []
        if os.path.exists(path_to_data) and os.listdir(path_to_data):
            # Load from 'data' folder (PDFs)
            documents = doc_processor.load_from_pdf_dir(path_to_data)
        else:
            # Fallback to URLs from config
            documents = doc_processor.process_urls(Config.DEFAULT_URLS)
            
        # Split and Embed
        chunks = doc_processor.split_documents(documents)
        vector_store.create_vectorstore(chunks)
        
        # Build Graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        # Use .build() to compile the graph
        rag_system = graph_builder
        # Pre-build the graph internally
        rag_system.build()
        
        print(f"✅ System Ready! Loaded {len(chunks)} chunks.")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        raise e

# --- 4. Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Takes user message and returns answer with citations.
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="System is still initializing")

    try:
        # Run the graph
        # Note: rag_system is the GraphBuilder instance, we call .run() on it
        result = rag_system.run(request.message)
        
        # Check if the guardrail blocked it (graph ends early)
        answer_text = result.get('answer', "")
        
        # If the answer is the strict refusal, we might not have docs
        citations = []
        if "retrieved_docs" in result and result['retrieved_docs']:
            for doc in result['retrieved_docs']:
                # Extract source metadata (e.g., page number, source file)
                meta = doc.metadata
                source_name = meta.get('source', 'Unknown Source')
                page_num = meta.get('page', '')
                src_label = f"{source_name} (Page {page_num})" if page_num else source_name
                
                citations.append(Citation(
                    source=src_label,
                    page_content=doc.page_content[:200] + "..." # Snippet
                ))
        
        return ChatResponse(
            answer=answer_text,
            citations=citations
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
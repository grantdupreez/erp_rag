import streamlit as st
import anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import uuid
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import os
from sentence_transformers import SentenceTransformer
import json
import re

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", 6333)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For Qdrant Cloud
COLLECTION_NAME = "erp_projects"

class ERPProjectRAG:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Initialize Qdrant client
        if QDRANT_API_KEY:
            self.qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
        else:
            self.qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
        
        # Initialize embedding model (lightweight alternative to OpenAI)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Initialize collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                st.success(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            st.error(f"Error with Qdrant collection: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if end >= len(text):
                break
        
        return chunks
    
    def add_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Add a document to the RAG system"""
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Chunk the document
            chunks = self.chunk_text(content)
            
            # Process each chunk
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Insert into Qdrant
            self.qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            return True
        except Exception as e:
            st.error(f"Error adding document: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Qdrant
            results = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.payload["content"],
                    "metadata": result.payload["metadata"],
                    "score": result.score,
                    "doc_id": result.payload["doc_id"]
                })
            
            return formatted_results
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Claude 4 with retrieved context"""
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Document {i+1} - {doc['metadata'].get('title', 'Untitled')}]\n{doc['content']}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Create the prompt
        prompt = f"""You are an expert ERP project management assistant. You help project managers by providing guidance based on previous project experiences and documentation.

CONTEXT FROM PREVIOUS PROJECTS:
{context}

USER QUESTION: {query}

Based on the provided context from previous ERP projects, please provide a helpful and specific response. If the context doesn't contain relevant information, acknowledge this and provide general best practices for ERP project management.

Guidelines:
- Reference specific examples from the context when relevant
- Provide actionable advice
- Consider common ERP implementation challenges
- Focus on practical project management insights
- If discussing timelines, risks, or resources, be specific based on the context

Response:"""

        try:
            # Call Claude 4
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

def main():
    st.set_page_config(
        page_title="ERP Project Management Assistant",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ ERP Project Management RAG Assistant")
    st.markdown("*Powered by Claude 4 and your project knowledge base*")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ERPProjectRAG()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Knowledge Base Management")
        
        # Document upload section
        st.subheader("Add New Project Document")
        
        uploaded_file = st.file_uploader(
            "Upload project document",
            type=['txt', 'md', 'csv'],
            help="Upload project documentation, lessons learned, or project reports"
        )
        
        if uploaded_file:
            # Metadata inputs
            doc_title = st.text_input("Document Title", value=uploaded_file.name)
            project_name = st.text_input("Project Name")
            project_phase = st.selectbox(
                "Project Phase",
                ["Planning", "Analysis", "Design", "Development", "Testing", "Deployment", "Post-Implementation"]
            )
            document_type = st.selectbox(
                "Document Type",
                ["Lessons Learned", "Project Plan", "Risk Assessment", "Requirements", "Technical Specification", "Meeting Notes", "Status Report"]
            )
            
            if st.button("Add to Knowledge Base"):
                try:
                    # Read file content
                    if uploaded_file.type == "text/plain":
                        content = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        content = df.to_string()
                    else:
                        content = str(uploaded_file.read(), "utf-8")
                    
                    # Prepare metadata
                    metadata = {
                        "title": doc_title,
                        "project_name": project_name,
                        "project_phase": project_phase,
                        "document_type": document_type,
                        "upload_date": datetime.now().isoformat()
                    }
                    
                    # Add to RAG system
                    if st.session_state.rag_system.add_document(content, metadata):
                        st.success("Document added successfully!")
                    else:
                        st.error("Failed to add document")
                        
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Knowledge Base Stats")
        try:
            collection_info = st.session_state.rag_system.qdrant_client.get_collection(COLLECTION_NAME)
            st.metric("Total Documents", collection_info.points_count)
        except:
            st.metric("Total Documents", "N/A")
    
    # Main chat interface
    st.header("💬 Ask Your Project Management Assistant")
    
    # Sample questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - *What were the main risks identified in similar ERP implementations?*
        - *How long did the testing phase typically take in previous projects?*
        - *What lessons were learned from the data migration process?*
        - *What change management strategies worked best?*
        - *What were common issues during go-live?*
        """)
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your ERP projects..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating response..."):
                # Search for relevant documents
                relevant_docs = st.session_state.rag_system.search_documents(prompt, top_k=5)
                
                # Generate response
                response = st.session_state.rag_system.generate_response(prompt, relevant_docs)
                
                st.markdown(response)
                
                # Show sources
                if relevant_docs:
                    with st.expander("📖 Sources Used"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {i+1}:** {doc['metadata'].get('title', 'Untitled')}")
                            st.markdown(f"*Project:* {doc['metadata'].get('project_name', 'N/A')}")
                            st.markdown(f"*Type:* {doc['metadata'].get('document_type', 'N/A')}")
                            st.markdown(f"*Relevance Score:* {doc['score']:.3f}")
                            st.markdown("---")
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    # Check for required environment variables
    if not ANTHROPIC_API_KEY:
        st.error("Please set your ANTHROPIC_API_KEY environment variable")
        st.stop()
    
    main()
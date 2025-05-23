import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import hmac
import uuid
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import os
import json
import re
import PyPDF2
import io
import requests
import numpy as np

# Configuration
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", None)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
QDRANT_URL = st.secrets.get("QDRANT_URL", None)
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", None)
COLLECTION_NAME = "erp_projects"

class ERPProjectRAG:
    def __init__(self):
        # Store API keys
        self.anthropic_api_key = ANTHROPIC_API_KEY
        self.openai_api_key = OPENAI_API_KEY
        
        # Initialize with a default embedding dimension
        # Will be updated based on collection settings if needed
        self.embedding_dim = 384  # Default to text-embedding-3-small
        self.embedding_model = "text-embedding-3-small"
        
        # Initialize Qdrant client
        try:
            if not QDRANT_URL:
                st.error("QDRANT_URL not found in secrets")
                self.qdrant_client = None
            elif QDRANT_API_KEY:
                self.qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY,
                    timeout=30
                )
            else:
                # Parse URL to extract host and port if needed
                if ":" in QDRANT_URL and not QDRANT_URL.startswith("http"):
                    host, port = QDRANT_URL.split(":")
                    self.qdrant_client = QdrantClient(host=host, port=int(port))
                else:
                    self.qdrant_client = QdrantClient(url=QDRANT_URL)
        except Exception as e:
            st.error(f"Error connecting to Qdrant: {e}")
            self.qdrant_client = None
        
        # Initialize collection if it doesn't exist
        if self.qdrant_client:
            self._ensure_collection_exists()
            
        # Validate Claude model
        self.claude_model = self.validate_claude_model()
    
    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist, or validate dimensions if it does"""
        if not self.qdrant_client:
            return
            
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if COLLECTION_NAME not in collection_names:
                # Create new collection with the correct dimensions
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                st.success(f"Created new collection: {COLLECTION_NAME} with dimension {self.embedding_dim}")
            else:
                # Collection exists, check if dimensions match
                collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
                actual_dim = collection_info.config.params.vectors.size
                
                if actual_dim != self.embedding_dim:
                    st.warning(f"Dimension mismatch! Collection expects {actual_dim} but code is using {self.embedding_dim}")
                    st.info(f"Adjusting embedding dimension to match collection: {actual_dim}")
                    self.embedding_dim = actual_dim
                    
                    # Update the model based on the dimension
                    if actual_dim == 1536:
                        st.info("Using text-embedding-ada-002 model (1536 dimensions)")
                        self.embedding_model = "text-embedding-ada-002"
                    elif actual_dim == 384:
                        st.info("Using text-embedding-3-small model (384 dimensions)")
                        self.embedding_model = "text-embedding-3-small"
                    elif actual_dim == 3072:
                        st.info("Using text-embedding-3-large model (3072 dimensions)")
                        self.embedding_model = "text-embedding-3-large"
                    else:
                        st.warning(f"Unknown dimension: {actual_dim}. Using random embeddings for testing.")
                        self.embedding_model = None
                
        except Exception as e:
            st.error(f"Error with Qdrant collection: {e}")
            # Provide more helpful error message
            if "dimension" in str(e).lower():
                st.error("Vector dimension mismatch. You may need to recreate your collection or adjust your embedding model.")
    
    def validate_claude_model(self) -> str:
        """Validate which Claude models are available and return a working model name"""
        if not self.anthropic_api_key:
            return None
            
        # List of models to try, from newest to oldest
        model_candidates = [
            "claude-3-sonnet-20240530",  # May 2025 hypothetical model
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1"  # Fallback to older model if needed
        ]
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Simple test prompt
        data = {
            "max_tokens": 10,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        # Try each model until one works
        working_model = None
        for model in model_candidates:
            try:
                data["model"] = model
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=5  # Short timeout for testing
                )
                
                if response.status_code == 200:
                    working_model = model
                    st.success(f"Successfully connected to Claude API using model: {model}")
                    break
                else:
                    st.warning(f"Model {model} not available: {response.status_code}")
                    
            except Exception as e:
                st.warning(f"Error testing model {model}: {e}")
        
        if not working_model:
            st.error("No Claude models are working. Please check your API key and configuration.")
        
        return working_model
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI API"""
        if not self.openai_api_key or not self.embedding_model:
            # Fallback to random embedding for testing
            st.warning("OpenAI API key not found or model not set. Using random embeddings (for testing only).")
            return np.random.rand(self.embedding_dim).tolist()
        
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "input": text,
                "model": self.embedding_model
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                st.error(f"OpenAI API Error: {response.status_code} - {response.text}")
                # Fallback to random embedding
                return np.random.rand(self.embedding_dim).tolist()
                
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            # Fallback to random embedding
            return np.random.rand(self.embedding_dim).tolist()
    
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
            if not self.qdrant_client:
                st.error("Qdrant client not available")
                return False
                
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Chunk the document
            chunks = self.chunk_text(content)
            
            # Process each chunk
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.get_embedding(chunk)
                
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
            if not self.qdrant_client:
                st.error("Qdrant client not available")
                return []
                
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
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
        """Generate response using Claude API directly"""
        
        if not self.anthropic_api_key or not self.claude_model:
            return "I apologize, but the AI service is not configured properly. Please contact your administrator."
        
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
            # Direct API call without using the anthropic library
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Use the validated model
            data = {
                "model": self.claude_model,
                "max_tokens": 1500,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Make the API request
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                error_msg = f"API Error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                st.error(error_msg)
                return "I apologize, but I encountered an error while generating a response. Please check your API key and model configuration and try again."
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return "The request took too long to process. Please try again."
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            # Store the username for later use
            st.session_state["current_user"] = st.session_state["username"]
            del st.session_state["password"]  # Don't store the password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def check_system_status():
    """Check the status of all system components and display in Streamlit"""
    st.subheader("🔧 System Status")
    
    # Check Qdrant Connection
    if st.session_state.rag_system.qdrant_client:
        try:
            collection_info = st.session_state.rag_system.qdrant_client.get_collection(COLLECTION_NAME)
            st.success("✅ Qdrant Connection: **Connected**")
            st.info(f"Collection: {COLLECTION_NAME}, Vector Dimension: {collection_info.config.params.vectors.size}")
            st.info(f"Documents: {collection_info.points_count}")
        except Exception as e:
            st.error(f"❌ Qdrant Connection Error: {e}")
    else:
        st.error("❌ Qdrant Connection: **Not Connected**")
    
    # Check OpenAI API
    if st.session_state.rag_system.openai_api_key:
        try:
            # Test embedding
            test_embedding = st.session_state.rag_system.get_embedding("Test embedding")
            if len(test_embedding) == st.session_state.rag_system.embedding_dim:
                st.success("✅ OpenAI Embedding API: **Connected**")
                st.info(f"Embedding Model: {getattr(st.session_state.rag_system, 'embedding_model', 'Unknown')}")
                st.info(f"Embedding Dimension: {st.session_state.rag_system.embedding_dim}")
            else:
                st.warning(f"⚠️ OpenAI Embedding API: Dimension mismatch ({len(test_embedding)} vs {st.session_state.rag_system.embedding_dim})")
        except Exception as e:
            st.error(f"❌ OpenAI Embedding API Error: {e}")
    else:
        st.warning("⚠️ OpenAI Embedding API: **Not Configured** (using random embeddings)")
    
    # Check Claude API
    if st.session_state.rag_system.anthropic_api_key:
        if hasattr(st.session_state.rag_system, 'claude_model') and st.session_state.rag_system.claude_model:
            st.success("✅ Claude API: **Connected**")
            st.info(f"Claude Model: {st.session_state.rag_system.claude_model}")
        else:
            st.error("❌ Claude API: **No Working Model Found**")
    else:
        st.error("❌ Claude API: **Not Configured**")

def main():
    st.set_page_config(
        page_title="ERP Project Management Assistant",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ ERP Project Management RAG Assistant")
    st.markdown("*Powered by Claude API and your project knowledge base*")
    
    # Display current user in the top right
    col1, col2 = st.columns([6, 1])
    with col2:
        st.markdown(f"**User:** {st.session_state.get('current_user', 'Unknown')}")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ERPProjectRAG()
    
    # Sidebar for document management - ONLY FOR ADMIN
    if st.session_state.get('current_user') == 'admin':
        with st.sidebar:
            st.header("📚 Knowledge Base Management")
            st.markdown("*Admin access only*")
            
            # System status check
            if st.button("🔄 Check System Status"):
                pass  # Will trigger the check_system_status() function below
                
            # Check configuration status
            if not OPENAI_API_KEY:
                st.warning("⚠️ OpenAI API key not configured. Using random embeddings for testing.")
                st.info("Add OPENAI_API_KEY to secrets for production use.")
            
            # Document upload section
            st.subheader("Add New Project Document")
            
            uploaded_file = st.file_uploader(
                "Upload project document",
                type=['txt', 'md', 'csv', 'pdf'],
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
                        # Read file content based on type
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                        elif uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            content = df.to_string()
                        elif uploaded_file.type == "application/pdf":
                            content = extract_text_from_pdf(uploaded_file)
                            if content is None:
                                st.error("Failed to extract text from PDF")
                                st.stop()
                        else:
                            content = str(uploaded_file.read(), "utf-8")
                        
                        # Prepare metadata
                        metadata = {
                            "title": doc_title,
                            "project_name": project_name,
                            "project_phase": project_phase,
                            "document_type": document_type,
                            "upload_date": datetime.now().isoformat(),
                            "uploaded_by": st.session_state.get('current_user', 'Unknown')
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
            if st.session_state.rag_system.qdrant_client:
                try:
                    collection_info = st.session_state.rag_system.qdrant_client.get_collection(COLLECTION_NAME)
                    st.metric("Total Documents", collection_info.points_count)
                except:
                    st.metric("Total Documents", "N/A")
            else:
                st.metric("Total Documents", "Not connected")
        
        # Display system status for admin users
        check_system_status()
    
    # Main chat interface (visible to all users)
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
    # Check for required configuration
    missing_keys = []
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not QDRANT_URL:
        missing_keys.append("QDRANT_URL")
    
    if missing_keys:
        st.error(f"Missing required secrets: {', '.join(missing_keys)}")
        st.info("Please configure these in your Streamlit secrets.")
        st.stop()
    
    # Check password before starting the app
    if not check_password():
        st.stop()
        
    main()

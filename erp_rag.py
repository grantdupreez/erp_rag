import streamlit as st
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import openai
from datetime import datetime
import uuid
import PyPDF2
import docx
import pandas as pd
import json
import os
from typing import List, Dict, Any
import hashlib
import hmac

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Configuration
QDRANT_URL = st.secrets["QDRANT_URL"]
COLLECTION_NAME = "project_documents"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize clients
@st.cache_resource
def get_qdrant_client():
    return AsyncQdrantClient(url=QDRANT_URL)

@st.cache_resource
def get_openai_client():
    return openai.Client(api_key=OPENAI_API_KEY)

qdrant_client = get_qdrant_client()
openai_client = get_openai_client()

# Document processing functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    return file.read().decode("utf-8")

def process_file(file) -> Dict[str, Any]:
    """Process uploaded file and extract text"""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # Generate file hash for deduplication
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    
    return {
        "text": text,
        "filename": file.name,
        "file_type": file_extension,
        "file_hash": file_hash,
        "upload_date": datetime.now().isoformat()
    }

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# Embedding functions
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI with correct dimensions"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            dimensions=384  # Specify dimension to match Qdrant collection
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return None

# Qdrant functions
async def ensure_collection_exists():
    """Ensure the Qdrant collection exists"""
    try:
        collections = await qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            await qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            st.success(f"Created collection: {COLLECTION_NAME}")
    except Exception as e:
        st.error(f"Error checking/creating collection: {e}")

async def upload_to_qdrant(chunks: List[str], metadata: Dict[str, Any], username: str):
    """Upload document chunks to Qdrant"""
    embeddings = await get_embeddings(chunks)
    if not embeddings:
        return False
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk,
                    "chunk_index": i,
                    "username": username,
                    **metadata
                }
            )
        )
    
    try:
        await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to Qdrant: {e}")
        return False

async def search_documents(query: str, username: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search documents in Qdrant"""
    query_embedding = await get_embeddings([query])
    if not query_embedding:
        return []
    
    try:
        results = await qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding[0],
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="username",
                        match=MatchValue(value=username)
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "filename": hit.payload["filename"],
                "chunk_index": hit.payload["chunk_index"]
            }
            for hit in results
        ]
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

async def generate_rag_response(query: str, context: List[Dict[str, Any]]) -> str:
    """Generate response using RAG"""
    context_text = "\n\n".join([f"[{doc['filename']}]: {doc['text']}" for doc in context])
    
    prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

# Main application
def main():
    # Set page config
    st.set_page_config(
        page_title="ERP Project Manager",
        page_icon="📊",
        layout="wide"
    )
    
    # Login page
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
                del st.session_state["password"]  # Don't store the username or password.
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
    
    if not check_password():
        st.stop()    
        
    # App header with user info and logout
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.title("ERP Project Manager")
    with col2:
        st.write("")  # Empty space
    with col3:
        user_col1, user_col2 = st.columns([3, 1])
        with user_col1:
            st.markdown(f"**User:** {st.session_state.username}")
        with user_col2:
            if st.button("Logout", key="logout_btn", type="secondary"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Ensure collection exists
    asyncio.run(ensure_collection_exists())
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🔍 Search & Query", "📚 My Documents"])
    
    # Upload Documents Tab
    with tab1:
        st.header("Upload Project Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("Process and Upload", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    try:
                        # Process file
                        file_data = process_file(file)
                        
                        # Chunk text
                        chunks = chunk_text(file_data["text"])
                        
                        # Upload to Qdrant
                        success = asyncio.run(upload_to_qdrant(
                            chunks, 
                            file_data, 
                            st.session_state.username
                        ))
                        
                        if success:
                            st.success(f"✅ Successfully uploaded {file.name}")
                            st.session_state.uploaded_files.append({
                                "filename": file.name,
                                "upload_date": file_data["upload_date"],
                                "chunks": len(chunks)
                            })
                        else:
                            st.error(f"❌ Failed to upload {file.name}")
                            
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                status_text.text("Upload complete!")
                progress_bar.progress(1.0)
    
    # Search & Query Tab
    with tab2:
        st.header("Search and Query Documents")
        
        query = st.text_input("Enter your question or search query:")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            num_results = st.number_input("Results to retrieve:", min_value=1, max_value=20, value=5)
        
        if st.button("Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    # Search documents
                    search_results = asyncio.run(search_documents(
                        query, 
                        st.session_state.username, 
                        num_results
                    ))
                    
                    if search_results:
                        # Generate RAG response
                        with st.spinner("Generating response..."):
                            response = asyncio.run(generate_rag_response(query, search_results))
                        
                        # Display response
                        st.markdown("### 💡 Answer")
                        st.markdown(response)
                        
                        # Display sources
                        st.markdown("### 📄 Sources")
                        for idx, result in enumerate(search_results):
                            with st.expander(f"{result['filename']} (Score: {result['score']:.3f})"):
                                st.text(result['text'])
                    else:
                        st.warning("No relevant documents found. Try uploading more documents or refining your query.")
            else:
                st.warning("Please enter a search query.")
    
    # My Documents Tab
    with tab3:
        st.header("My Uploaded Documents")
        
        if st.session_state.uploaded_files:
            df = pd.DataFrame(st.session_state.uploaded_files)
            st.dataframe(df, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(st.session_state.uploaded_files))
            with col2:
                total_chunks = sum(doc['chunks'] for doc in st.session_state.uploaded_files)
                st.metric("Total Chunks", total_chunks)
            with col3:
                st.metric("User", st.session_state.username)
        else:
            st.info("No documents uploaded yet. Go to the Upload tab to add documents.")

if __name__ == "__main__":
    main()

import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="ERP Document Assistant",
    page_icon="📊",
    layout="wide"
)

from pinecone import Pinecone, ServerlessSpec
import openai
from datetime import datetime
import uuid
import PyPDF2
import docx
import pandas as pd
import hashlib
import hmac
from typing import List, Dict, Any
import time
import json
import re

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Configuration
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "erp-documents"
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,  # OpenAI embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
            # Wait for index to be ready
            time.sleep(1)
        
        return pc.Index(INDEX_NAME)
    except Exception as e:
        st.error(f"Failed to initialize Pinecone: {e}")
        return None

# Document processing functions
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file) -> str:
    """Extract text from TXT file"""
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # Avoid empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap if end < text_length else text_length
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI"""
    try:
        # Remove empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []
            
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return []

def upload_to_pinecone(chunks: List[str], metadata: Dict[str, Any], index):
    """Upload document chunks to Pinecone"""
    if not chunks:
        return False
        
    embeddings = get_embeddings(chunks)
    if not embeddings:
        return False
    
    # Prepare vectors for upload
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{metadata['file_hash']}_{i}"
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk[:1000],  # Limit metadata size
                "chunk_index": i,
                "total_chunks": len(chunks),
                "filename": metadata["filename"],
                "file_type": metadata["file_type"],
                "file_hash": metadata["file_hash"],
                "upload_date": metadata["upload_date"],
                "uploaded_by": metadata["uploaded_by"]
            }
        })
    
    try:
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        return True
    except Exception as e:
        st.error(f"Error uploading to Pinecone: {e}")
        return False

def delete_document(filename: str, file_hash: str, index):
    """Delete all chunks of a document from Pinecone"""
    try:
        # Get all vector IDs for this document
        vector_ids = []
        
        # Query to find the document's chunks
        results = index.query(
            vector=[0] * 1536,  # Dummy vector
            filter={
                "filename": {"$eq": filename},
                "file_hash": {"$eq": file_hash}
            },
            top_k=10000,
            include_metadata=False
        )
        
        # Extract vector IDs
        vector_ids = [match["id"] for match in results["matches"]]
        
        if vector_ids:
            # Delete in batches
            batch_size = 100
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i + batch_size]
                index.delete(ids=batch)
            return True, len(vector_ids)
        else:
            return False, 0
            
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False, 0

def search_documents(query: str, index, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant documents"""
    query_embedding = get_embeddings([query])
    if not query_embedding:
        return []
    
    try:
        results = index.query(
            vector=query_embedding[0],
            top_k=limit,
            include_metadata=True
        )
        
        return [
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "filename": match["metadata"]["filename"],
                "chunk_index": match["metadata"]["chunk_index"],
                "total_chunks": match["metadata"].get("total_chunks", "?")
            }
            for match in results["matches"]
        ]
    except Exception as e:
        st.error(f"Error searching: {e}")
        return []

def generate_response_with_citations(query: str, context: List[Dict[str, Any]]) -> str:
    """Generate response using OpenAI with inline citations"""
    if not context:
        return "I couldn't find any relevant information in the uploaded documents."
    
    # Prepare context with citation markers
    context_text = ""
    citation_map = {}
    
    for idx, doc in enumerate(context):
        citation_key = f"[{idx + 1}]"
        citation_map[citation_key] = f"{doc['filename']} (chunk {doc['chunk_index'] + 1}/{doc['total_chunks']})"
        context_text += f"{citation_key} {doc['text']}\n\n"
    
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that answers questions based on the provided document context. 
            When you use information from the context, include the citation number in square brackets (e.g., [1], [2]) 
            immediately after the relevant information. You may use multiple citations if information comes from multiple sources.
            Only use information from the provided context."""
        },
        {
            "role": "user",
            "content": f"""Based on the following context from uploaded documents, please answer the question. 
            Include citations in square brackets when using information from specific sources.

Context:
{context_text}

Question: {query}

Answer with citations:"""
        }
    ]
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Add citation references at the end
        citation_text = "\n\n**Sources:**\n"
        for key, value in citation_map.items():
            if key in answer:
                citation_text += f"{key} {value}\n"
        
        return answer + citation_text
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."

def export_chat_history(messages: List[Dict[str, str]], username: str) -> str:
    """Export chat history to markdown format"""
    export_text = f"# ERP Document Assistant - Chat Export\n\n"
    export_text += f"**User:** {username}\n"
    export_text += f"**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    export_text += "---\n\n"
    
    for message in messages:
        role = "**You:**" if message["role"] == "user" else "**Assistant:**"
        export_text += f"{role}\n{message['content']}\n\n"
    
    return export_text

def get_all_documents(index) -> pd.DataFrame:
    """Get list of all uploaded documents with file hash"""
    try:
        # Query with a dummy vector to get metadata
        results = index.query(
            vector=[0] * 1536,
            top_k=10000,
            include_metadata=True
        )
        
        # Extract unique documents
        documents = {}
        for match in results["matches"]:
            metadata = match["metadata"]
            filename = metadata.get("filename", "Unknown")
            file_hash = metadata.get("file_hash", "")
            
            if filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "file_hash": file_hash,
                    "upload_date": metadata.get("upload_date", "Unknown"),
                    "uploaded_by": metadata.get("uploaded_by", "Unknown"),
                    "file_type": metadata.get("file_type", "Unknown"),
                    "chunks": 0
                }
            documents[filename]["chunks"] += 1
        
        if documents:
            return pd.DataFrame(list(documents.values()))
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return pd.DataFrame()

# Authentication
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
            st.session_state["authenticated_username"] = st.session_state["username"]
            del st.session_state["password"]
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

# Main application
def main():
    if not check_password():
        st.stop()
    
    # Initialize Pinecone
    index = init_pinecone()
    if not index:
        st.error("Failed to connect to Pinecone. Please check your configuration.")
        st.stop()
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📊 ERP Document Assistant")
    with col2:
        st.write(f"**User:** {st.session_state.authenticated_username}")
        if st.button("Logout", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Check if user is admin
    admin_users_config = st.secrets.get("admin_users", "admin")
    if isinstance(admin_users_config, str):
        admin_users = [user.strip() for user in admin_users_config.split(',')]
    elif isinstance(admin_users_config, list):
        admin_users = admin_users_config
    else:
        admin_users = ["admin"]
    
    is_admin = st.session_state.authenticated_username in admin_users
    
    # Tabs
    if is_admin:
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload Documents", "📚 Document Library"])
    else:
        tab1, tab3 = st.tabs(["💬 Chat", "📚 Document Library"])
        tab2 = None
    
    # Chat Tab
    with tab1:
        st.header("Chat with Documents")
        
        # Export button
        if st.session_state.messages:
            col1, col2 = st.columns([6, 1])
            with col2:
                export_text = export_chat_history(
                    st.session_state.messages, 
                    st.session_state.authenticated_username
                )
                st.download_button(
                    label="📥 Export Chat",
                    data=export_text,
                    file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    # Search for relevant content
                    search_results = search_documents(prompt, index, limit=5)
                    
                    if search_results:
                        # Generate response with citations
                        response = generate_response_with_citations(prompt, search_results)
                        st.markdown(response)
                        
                        # Show detailed sources
                        with st.expander("📄 View Source Documents"):
                            for i, result in enumerate(search_results):
                                st.markdown(f"**[{i+1}] {result['filename']}** (Chunk {result['chunk_index']+1}/{result['total_chunks']}, Relevance: {result['score']:.2f})")
                                st.text(result['text'][:300] + "...")
                                if i < len(search_results) - 1:
                                    st.divider()
                    else:
                        response = "I couldn't find any relevant information in the documents. Please make sure documents have been uploaded."
                        st.markdown(response)
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
    
    # Upload Tab (Admin only)
    if tab2 and is_admin:
        with tab2:
            st.header("Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Upload", type="primary"):
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_files):
                        with st.spinner(f"Processing {file.name}..."):
                            try:
                                # Extract text based on file type
                                file_extension = file.name.split('.')[-1].lower()
                                
                                if file_extension == 'pdf':
                                    text = extract_text_from_pdf(file)
                                elif file_extension == 'docx':
                                    text = extract_text_from_docx(file)
                                elif file_extension == 'txt':
                                    text = extract_text_from_txt(file)
                                else:
                                    st.error(f"Unsupported file type: {file_extension}")
                                    continue
                                
                                if not text:
                                    st.error(f"No text extracted from {file.name}")
                                    continue
                                
                                # Generate file hash
                                file.seek(0)
                                file_hash = hashlib.md5(file.read()).hexdigest()
                                
                                # Chunk text
                                chunks = chunk_text(text)
                                
                                if not chunks:
                                    st.error(f"No chunks created from {file.name}")
                                    continue
                                
                                # Prepare metadata
                                metadata = {
                                    "filename": file.name,
                                    "file_type": file_extension,
                                    "file_hash": file_hash,
                                    "upload_date": datetime.now().isoformat(),
                                    "uploaded_by": st.session_state.authenticated_username
                                }
                                
                                # Upload to Pinecone
                                success = upload_to_pinecone(chunks, metadata, index)
                                
                                if success:
                                    st.success(f"✅ {file.name} - {len(chunks)} chunks uploaded")
                                else:
                                    st.error(f"❌ Failed to upload {file.name}")
                                    
                            except Exception as e:
                                st.error(f"Error processing {file.name}: {str(e)}")
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    st.balloons()
    
    # Document Library Tab
    with tab3:
        st.header("Document Library")
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with st.spinner("Loading documents..."):
            df = get_all_documents(index)
        
        if not df.empty:
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(df))
            with col2:
                st.metric("Total Chunks", df['chunks'].sum())
            with col3:
                unique_uploaders = df['uploaded_by'].nunique()
                st.metric("Contributors", unique_uploaders)
            
            # Document management for admins
            if is_admin:
                st.subheader("Document Management")
                
                # Create a container for each document
                for idx, row in df.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                        
                        with col1:
                            st.write(f"📄 **{row['filename']}**")
                        with col2:
                            st.write(f"{row['chunks']} chunks")
                        with col3:
                            st.write(f"by {row['uploaded_by']}")
                        with col4:
                            upload_date = datetime.fromisoformat(row['upload_date'])
                            st.write(upload_date.strftime('%Y-%m-%d'))
                        with col5:
                            if st.button("🗑️ Delete", key=f"delete_{idx}"):
                                with st.spinner(f"Deleting {row['filename']}..."):
                                    success, chunks_deleted = delete_document(
                                        row['filename'], 
                                        row['file_hash'], 
                                        index
                                    )
                                    if success:
                                        st.success(f"Deleted {chunks_deleted} chunks")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete document")
                        
                        st.divider()
            else:
                # Non-admin view
                st.dataframe(
                    df[['filename', 'chunks', 'uploaded_by', 'upload_date']].sort_values('upload_date', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No documents uploaded yet.")
            if is_admin:
                st.write("Go to the Upload Documents tab to add documents.")

if __name__ == "__main__":
    main()

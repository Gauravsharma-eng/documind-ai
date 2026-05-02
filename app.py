import streamlit as st
import time
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🤖 DocuMind AI", layout="wide")

# Groq API Key Setup
try:
    # Dashboard -> Settings -> Secrets mein "GROQ_API_KEY" save karna zaroori hai
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Error: Please add GROQ_API_KEY in Streamlit Secrets!")

# ---------------- CSS FOR UI ----------------
st.markdown("""
    <style>
    .stApp { background:#0e1117; color: white; }
    .title { font-size:40px; color:#00f7ff; text-shadow:0 0 15px #00f7ff; text-align: center; }
    .user-msg { background:#00c6ff; padding:12px; border-radius:15px; color:black; margin:10px 0; display: inline-block; }
    .bot-msg { background:#7b2ff7; padding:12px; border-radius:15px; margin:10px 0; color:white; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None

# ---------------- UI LAYOUT ----------------
st.markdown('<h1 class="title">🚀 DocuMind AI</h1>', unsafe_allow_html=True)
st.caption("<p style='text-align: center;'>Your Intelligent PDF Assistant powered by Llama 3</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your PDF document", type="pdf")

if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process PDF only once
    if st.session_state.db is None:
        with st.spinner("Reading and Indexing PDF..."):
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Create Embeddings using CPU-friendly model
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.db = FAISS.from_documents(chunks, embeddings)
            st.success("✅ PDF Ready for Chat!")

# Chat Interface
query = st.chat_input("Ask something about the PDF...")

if query:
    st.session_state.chat.append(("user", query))
    
    # Get Context from Vector Store (RAG)
    context = ""
    if st.session_state.db:
        docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
    
    # Prepare Prompt
    system_prompt = f"""
    You are a helpful AI assistant. Answer the question based ONLY on the provided context.
    If the answer is not in the context, say you don't know, but you can answer generally if asked.
    
    Context:
    {context}
    """
    
    try:
        # Call Groq Llama 3 API (Cloud based)
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ API Error: {str(e)}"
    
    st.session_state.chat.append(("bot", answer))

# Display Chat History
for role, msg in st.session_state.chat:
    if role == "user":
        st.write(f"🧑 **You:** {msg}")
    else:
        st.write(f"🤖 **DocuMind:** {msg}")

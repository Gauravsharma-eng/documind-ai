import streamlit as st
import time
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- ⚙️ CONFIG ----------------
st.set_page_config(page_title="🤖 DocuMind AI", layout="wide", page_icon="🚀")

# ---------------- 🎨 CUSTOM CSS ----------------
st.markdown("""
    <style>
    .stApp { background:#0e1117; color: white; }
    .title-text { font-size:50px; color:#00f7ff; text-shadow:0 0 20px #00f7ff; font-weight: bold; margin-bottom: 0px; text-align: center; }
    .subtitle-text { color: #b0b0b0; font-size: 18px; margin-top: -10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# Groq Free API Client Setup via OpenAI compatibility
try:
    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )
except Exception as e:
    st.error("❌ Error: Please add GROQ_API_KEY in Streamlit Secrets!")

# ---------------- 🧠 SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None

# ---------------- 🚀 MAIN UI LAYOUT ----------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("project images.jpeg", width='stretch')
    st.markdown('<p class="title-text">DocuMind AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Your Intelligent PDF Assistant</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")
    
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.session_state.db is None:
            with st.spinner("🔍 Indexing PDF..."):
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.db = FAISS.from_documents(chunks, embeddings)
                st.success("✅ PDF Ready!")

# 💬 Chat Interface
query = st.chat_input("💬 Ask your question...")

if query:
    st.session_state.chat.append(("user", query))
    
    context = ""
    if st.session_state.db:
        docs = st.session_state.db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"Answer based ONLY on context: \n{context}"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            temperature=0.5,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"⚠️ Error: {str(e)}"
    
    st.session_state.chat.append(("bot", answer))

# 📜 Display Chat History
for role, msg in st.session_state.chat:
    if role == "user":
        st.chat_message("user", avatar="🧑‍💻").write(msg)
    else:
        st.chat_message("assistant", avatar="🤖").write(msg)

# Sidebar
with st.sidebar:
    st.title("🛠️ Project Info")
    st.info("Developed by **Gaurav** 🧑‍💻")
    st.write("📍 Based in Gwalior")
    st.write("🚀 Tech: Llama 3.3 + FAISS")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()

import streamlit as st
import time
from groq import Groq  # Native Groq client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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

# Native Groq Client Setup
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("❌ Error: Please add GROQ_API_KEY in Streamlit Secrets!")

# ---------------- 🧠 SESSION STATE ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ---------------- 🚀 MAIN UI LAYOUT ----------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("project images.jpeg", use_container_width=True)
    st.markdown('<p class="title-text">DocuMind AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Your Intelligent PDF Assistant</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")
    
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.session_state.db is None:
            with st.spinner("🔍 Indexing PDF & Generating Summary..."):
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # 1. Persistent Vector DB (Chroma)
                st.session_state.db = Chroma.from_documents(
                    chunks, embeddings, persist_directory="./chroma_db"
                )
                
                # 2. Auto-Summary Generation
                sample_text = "\n".join([doc.page_content for doc in chunks[:3]])
                sum_completion = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": "Provide a concise 3-bullet executive summary of this document excerpt."},
                        {"role": "user", "content": sample_text}
                    ],
                    temperature=0.3
                )
                st.session_state.summary = sum_completion.choices[0].message.content
                st.success("✅ PDF Ready & Summarized!")

    # Display Auto-Summary in an expander
    if st.session_state.summary:
        with st.expander("📌 Document Executive Summary", expanded=False):
            st.write(st.session_state.summary)

# 📜 Display Chat History First
for role, msg in st.session_state.chat:
    if role == "user":
        st.chat_message("user", avatar="🧑‍💻").write(msg)
    else:
        st.chat_message("assistant", avatar="🤖").write(msg)

# 💬 Chat Interface with Streaming & Citations
query = st.chat_input("💬 Ask your question...")

if query:
    st.session_state.chat.append(("user", query))
    st.chat_message("user", avatar="🧑‍💻").write(query)
    
    context = ""
    citations = set()
    if st.session_state.db:
        results = st.session_state.db.similarity_search_with_score(query, k=3)
        context_parts = []
        for doc, score in results:
            page_num = doc.metadata.get("page", 0) + 1
            citations.add(page_num)
            context_parts.append(f"[Page {page_num}]: {doc.page_content}")
        context = "\n".join(context_parts)
    
    system_prompt = f"Answer based ONLY on the provided context. Be precise and helpful.\n\nContext:\n{context}"
    
    messages = [{"role": "system", "content": system_prompt}]
    for role, msg in st.session_state.chat[:-1]:
        api_role = "user" if role == "user" else "assistant"
        messages.append({"role": api_role, "content": msg})
    messages.append({"role": "user", "content": query})
    
    with st.chat_message("assistant", avatar="🤖"):
        try:
            stream_response = client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=messages,
                temperature=0.5,
                stream=True,
            )
            
            def response_generator():
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            
            answer = st.write_stream(response_generator())
            
            if citations:
                citation_str = f"\n\n*📚 Source Pages: {', '.join(map(str, sorted(citations)))}*"
                st.markdown(citation_str)
                answer += citation_str
                
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
            st.error(answer)
    
    st.session_state.chat.append(("bot", answer))
    st.rerun()

# Sidebar
with st.sidebar:
    st.title("🛠️ Project Info")
    st.info("Developed by **Gaurav** 🧑‍💻")
    st.write("📍 Based in Gwalior")
    st.write("🚀 Tech: ChromaDB + Groq + Streaming")
    
    # Download Chat History Button
    if st.session_state.chat:
        chat_export = "\n\n".join([f"{role.upper()}: {msg}" for role, msg in st.session_state.chat])
        st.download_button(
            label="📥 Download Chat History",
            data=chat_export,
            file_name="documind_chat_history.txt",
            mime="text/plain"
        )
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.session_state.summary = ""
        st.session_state.db = None
        st.rerun()

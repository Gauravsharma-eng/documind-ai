import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

import streamlit as st
import ollama
import time
import speech_recognition as sr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🤖 PDF Assistant", layout="wide")

# ---------------- DARK / LIGHT TOGGLE ----------------
dark_mode = st.sidebar.checkbox("🌑 Dark Mode", value=True)

# ---------------- CSS ----------------
if dark_mode:
    st.markdown("""
    <style>
    body { background:#000000; color: white; font-family: 'Segoe UI', sans-serif; }
    .title { font-size:40px; color:#00f7ff; text-shadow:0 0 20px #00f7ff; }
    .card { background: rgba(255,255,255,0.05); padding:20px; border-radius:15px; }
    .user { background:#00c6ff; padding:10px; border-radius:10px; color:black; margin:5px; }
    .bot { background:#7b2ff7; padding:10px; border-radius:10px; margin:5px; color:white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body { background:#f5f5f5; color: black; font-family: 'Segoe UI', sans-serif; }
    .title { font-size:40px; color:#0077ff; text-shadow:0 0 20px #0077ff; }
    .card { background: rgba(0,0,0,0.05); padding:20px; border-radius:15px; }
    .user { background:#00aaff; padding:10px; border-radius:10px; color:white; margin:5px; }
    .bot { background:#aa00ff; padding:10px; border-radius:10px; margin:5px; color:white; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- MEMORY ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<h1 class="title">🚀 DocuMind AI</h1>', unsafe_allow_html=True)
    st.caption("Your Intelligent PDF Assistant")
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png")

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

    if st.button("🎤 Speak"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
            try:
                query = recognizer.recognize_google(audio)
                st.success(f"You said: {query}")
            except:
                query = ""
                st.error("Voice not recognized")
    else:
        query = st.text_input("💬 Ask your question")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.db = FAISS.from_documents(chunks, embeddings)
        st.success("✅ PDF Ready!")

    def stream_text(text):
        result = ""
        for word in text.split():
            result += word + " "
            yield result
            time.sleep(0.02)

    if query:
        st.session_state.chat.append(("user", query))
        context = ""
        if st.session_state.db:
            docs = st.session_state.db.similarity_search(query)
            context = " ".join([doc.page_content for doc in docs])
        prompt = f"""
Answer using PDF context if available.
If not found in context, answer generally using AI knowledge.

Context:
{context}

Question:
{query}
"""
        try:
            response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
            answer = response['message']['content']
        except Exception as e:
            answer = f"⚠️ Ollama not running: {str(e)}"
        st.session_state.chat.append(("bot", answer))

    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f'<div class="user">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            placeholder = st.empty()
            for chunk in stream_text(msg):
                placeholder.markdown(f'<div class="bot">🤖 {chunk}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
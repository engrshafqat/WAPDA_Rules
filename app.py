import os
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq
import streamlit as st

# Initialize Groq Client
# 1. Access Groq API key from environment variables
groq_api_key = os.environ.get("GROQ_API_KEY")

# 2. Initialize Groq API client (Conditionally)
client = None
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        st.stop()
else:
    st.error("GROQ_API_KEY environment variable not set. Please check your Hugging Face Secrets.")
    st.stop()

# Function to download PDF from Google Drive link
def download_pdf(google_drive_link):
    file_id = google_drive_link.split("/d/")[1].split("/view")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    with open("document.pdf", "wb") as pdf_file:
        pdf_file.write(response.content)
    return "document.pdf"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to create embeddings and store in FAISS
def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(index, chunks, query, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

# Function to interact with Groq's API
def interact_with_groq(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant using provided context."},
        {"role": "user", "content": f"Context: {context}"},
        {"role": "user", "content": f"Query: {query}"}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Hardcoded Google Drive links
google_drive_links = [
    "https://drive.google.com/file/d/1aiEUaUy_rrMj9ypIBvf5mBBbWXtNidfJ/view?usp=sharing"
]

# Streamlit UI - Enhanced Look
st.set_page_config(page_title="WAPDA Chatbot", layout="wide")
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background: linear-gradient(to bottom, #f0f4f8, #d9e6f2);
    }
    .header {
        text-align: center;
        font-size: 36px;
        color: #2E86C1;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        font-size: 20px;
        color: #2E86C1;
        margin-bottom: 20px;
    }
    .developer {
        text-align: center;
        font-size: 14px;
        color: #808080;
        margin-top: 40px;
    }
    .logo {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 150px;
    }
    .response-box {
        background-color: #ffffff;
        border: 1px solid #d9d9d9;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    <img src="logo.png" class="logo" alt="Organization Logo" onerror="this.src='https://via.placeholder.com/150';">
    <div class="header">Pakistan WAPDA Manual of General Rules</div>
    <div class="subheader">Ask any query related to the official rules and regulations.</div>
    """,
    unsafe_allow_html=True
)

# Automatically process PDF links
if "index" not in st.session_state:
    all_chunks = []
    for link in google_drive_links:
        pdf_path = download_pdf(link)
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    index, embeddings, chunks = create_embeddings(all_chunks)
    st.session_state.index = index
    st.session_state.chunks = chunks

# Query Input
query = st.text_input("Enter your query here:")
if st.button("Get Answer"):
    if query:
        with st.spinner("Processing..."):
            if st.session_state.index and st.session_state.chunks:
                relevant_chunks = retrieve_relevant_chunks(
                    st.session_state.index, st.session_state.chunks, query
                )
                context = " ".join(relevant_chunks)
                response = interact_with_groq(query, context)
                st.markdown("### Response:")
                st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
            else:
                st.error("PDF links are still being processed. Please try again later.")

st.markdown('<div class="developer">Developed by Shafqat Ali Memon</div>', unsafe_allow_html=True)
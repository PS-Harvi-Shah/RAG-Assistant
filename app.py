import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap
import google.generativeai as genai
import os
import pickle
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("⚠️ GEMINI_API_KEY not found in .env file. Please add it.")
    st.stop()

genai.configure(api_key=api_key)


def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# Extracting YouTube Transcript
def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript_list = YouTubeTranscriptApi().list(video_id)
    
    for transcript in transcript_list:
        if transcript.language_code.startswith('en'):
            transcript_data = transcript.fetch()
            text = " ".join([snippet.text for snippet in transcript_data if snippet.text.strip()])
            return text
    
    if transcript_list:
        transcript_data = transcript_list[0].fetch()
        text = " ".join([snippet.text for snippet in transcript_data if snippet.text.strip()])
        return text
    
    raise ValueError("No transcript available for this video")


# Embedding transcripts 
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(model, texts):
    return model.encode(texts)

# Create and store vector in FAISS database
def create_vector_store(embeddings, texts, save_path="data_faiss.pkl"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    with open(save_path, "wb") as f:
        pickle.dump({"index": index, "texts": texts}, f)


def load_vector_store(save_path="data_faiss.pkl"):
    if not os.path.exists(save_path):
        raise FileNotFoundError("No video has been processed yet. Please process a video first.")
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["texts"]


# Searching similar chunks
def search_similar(query_embedding, index, texts, top_k=3):
    D, I = index.search(query_embedding, top_k)
    return [texts[i] for i in I[0]]


# RAG pipeline
def process_video(video_url):
    st.info("📥 Fetching transcript...")
    text = get_transcript(video_url)

    st.info("🧩 Splitting transcript into chunks...")
    chunks = textwrap.wrap(text, 1000)

    st.info("🔢 Generating embeddings...")
    model = get_embedding_model()
    embeddings = embed_texts(model, chunks)

    create_vector_store(embeddings, chunks)
    return "Transcript processed and indexed successfully!"



def ask_question(query):
    embedding_model = get_embedding_model()
    index, texts = load_vector_store()
    query_embedding = embedding_model.encode([query])
    relevant_chunks = search_similar(query_embedding, index, texts, top_k=3)

    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question based on the following YouTube transcript context:\n\n{context}\n\nQuestion: {query}"

    try:
        # Using Gemini 2.5 Flash - fast and efficient for RAG
        gemini_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# UI using Streamlit

st.set_page_config(page_title="🎬 YouTube Video Chat Assistant", page_icon="🎥", layout="wide")

st.title("🎬 YouTube Video Summarizer & Chat Assistant")
st.markdown("Ask questions or get summaries from any YouTube video 🚀")

video_url = st.text_input("Enter a YouTube Video URL:")

if st.button("Process Video"):
    if not video_url.strip():
        st.error("⚠️ Please enter a valid YouTube URL")
    else:
        try:
            with st.spinner("Processing video..."):
                msg = process_video(video_url)
            st.success(msg)
        except Exception as e:
            st.error(f"Error: {e}")

query = st.text_input("Ask a question about the video:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                answer = ask_question(query)
                st.markdown("### 📝 Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")

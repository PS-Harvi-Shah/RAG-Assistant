# 🎬 YouTube Video Summarizer & Chat Assistant

A **Streamlit-based local RAG (Retrieval-Augmented Generation)** application that allows users to **chat with YouTube videos** — ask questions

---

## 🚀 Features

- 🔗 **Fetch YouTube transcripts** automatically  
- 🧠 **Embed & index transcripts locally** using FAISS  
- 💬 **Ask questions** and get contextual answers from the video  

---

## 🧩 Tech Stack

| Component         | Technology Used |
|------------------:|----------------|
| Frontend          | Streamlit |
| Transcript Fetch  | YouTube Transcript API |
| Embeddings        | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database   | FAISS |
| LLM Model         | Gemini 2.5 Flash (via Google GenAI SDK) |
| Language          | Python 3.9+ |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/PS-Harvi-Shah/RAG-Assistant.git
cd RAG-Assistant

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate       

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

4️⃣ Get a Gemini API Key
```bash
- Go to https://ai.google.dev/
- Sign in → Get API Key from Google AI Studio
- Copy your key.

### 5️⃣ Set Your API Key
```bash
- Make .evn folder and set the api key: 
  GEMINI_API_KEY "your_api_key_here"

▶️ Run the Application
```bash
streamlit run app.py


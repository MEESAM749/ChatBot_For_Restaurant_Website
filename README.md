# 🍛 Karachi Bites — AI Customer Support Chatbot

A production-ready **RAG (Retrieval-Augmented Generation) chatbot** that answers customer questions using a restaurant's own knowledge base. Built with FastAPI, ChromaDB, and Groq (Llama 3.3 70B).

**[Live Demo →](https://web-production-36259.up.railway.app/health)**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LLM](https://img.shields.io/badge/LLM-Llama_3.3_70B-orange)
![Deployed](https://img.shields.io/badge/Deployed-Railway-purple)

---

## What It Does

A customer visits a restaurant's website, clicks the chat bubble, and asks:

> *"Do you guys deliver to G-9?"*
> *"What's your cheapest biryani?"*
> *"Are you halal?"*

The chatbot retrieves relevant information from the restaurant's own documents (menu, policies, FAQ) and generates a natural, accurate answer — no hallucination, no made-up facts.

If the answer isn't in the knowledge base, it gracefully says so and redirects to WhatsApp.

---

## How It Works

The system has two phases:

### Preparation (runs once on startup)
```
Documents → Chunking → Embedding → Vector Database
```
1. **Collect** — text files from `knowledge_base/` (menu, policies, FAQ, about)
2. **Chunk** — split documents into meaningful sections (menu categories, individual Q&A pairs, policy sections)
3. **Embed** — convert each chunk into a 384-dimensional vector using all-MiniLM-L6-v2
4. **Store** — save vectors + text in ChromaDB for fast similarity search

### Runtime (every customer message)
```
Question → Embed → Search → Build Prompt → LLM → Answer
```
1. **Embed** the customer's question into a vector
2. **Search** ChromaDB for the 3 most similar chunks
3. **Build** a prompt with the retrieved context + system instructions
4. **Generate** an answer using Llama 3.3 70B via Groq
5. **Return** the answer through the REST API

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | **FastAPI** | Async, fast, auto-generates API docs |
| Vector DB | **ChromaDB** | Free, local, built-in embeddings |
| Embeddings | **all-MiniLM-L6-v2** (via ChromaDB) | Lightweight, no PyTorch needed |
| LLM | **Llama 3.3 70B** via **Groq** | Free API, very fast inference |
| Frontend | **Vanilla HTML/CSS/JS** | Zero dependencies, embeddable anywhere |
| Deployment | **Railway** | Simple, auto-deploy from GitHub |

---

## Project Structure

```
├── knowledge_base/          # Client's documents (the data source)
│   ├── menu.txt
│   ├── about.txt
│   ├── policies.txt
│   └── faq.txt
├── chatbot_server.py        # FastAPI server (retrieval + LLM)
├── chat_widget.html         # Embeddable frontend chat widget
├── test_chatbot.py          # Test script for API endpoints
├── setup_db.py              # Standalone DB builder (optional)
├── chunker.py               # Document chunking logic (educational)
├── embedder.py              # Embedding logic (educational)
├── requirements.txt
├── Procfile                 # Railway deployment config
└── .gitignore
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/karachi-bites-chatbot.git
cd karachi-bites-chatbot
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Go to [console.groq.com](https://console.groq.com), sign up, create an API key.

```bash
# Linux/Mac
export GROQ_API_KEY="gsk_your_key_here"

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your_key_here"
```

### 3. Run the server

```bash
uvicorn chatbot_server:app --reload --port 8000
```

The server auto-builds the vector database from `knowledge_base/` on first startup.

### 4. Test

```bash
python test_chatbot.py
```

Or open `chat_widget.html` in your browser and chat directly.

---

## API Endpoints

### `POST /chat`

Send a customer message, get an AI-generated answer.

**Request:**
```json
{
  "message": "What time do you close on Friday?"
}
```

**Response:**
```json
{
  "answer": "We're open until 1:00 AM on Fridays! If you'd like to order delivery, our delivery service runs until 11:30 PM.",
  "sources": [
    {"source": "about.txt", "section": "HOURS", "chunk_id": "about_002"},
    {"source": "policies.txt", "section": "DELIVERY", "chunk_id": "policies_025"}
  ]
}
```

### `GET /health`

Health check endpoint.

```json
{
  "status": "ok",
  "collection": "karachi_bites",
  "chunks_in_db": 30,
  "llm": "llama-3.3-70b-versatile"
}
```

---

## Chat Widget Integration

The chat widget is a self-contained HTML/CSS/JS component. To embed it on any website:

1. Copy the widget CSS and JS from `chat_widget.html`
2. Change the `API_URL` to your deployed server:
   ```javascript
   const API_URL = "https://your-app.up.railway.app";
   ```
3. Paste into the client's website HTML

Features:
- Floating chat bubble with open/close animation
- Typing indicator while waiting for response
- Mobile responsive
- Graceful error handling (falls back to WhatsApp contact)
- Customizable colors via CSS variables

---

## Adapting for Other Businesses

This is designed to be reused. To build a chatbot for a different client:

1. **Replace the knowledge base** — swap the files in `knowledge_base/` with the new client's documents
2. **Update the system prompt** — change the restaurant name, tone, and fallback contact in `SYSTEM_PROMPT` inside `chatbot_server.py`
3. **Customize the widget** — update colors (`--kb-primary`), name, avatar, and greeting in `chat_widget.html`
4. **Deploy** — push to GitHub, deploy on Railway/Render

The entire pipeline (chunking, embedding, retrieval, generation) works automatically with any text-based knowledge base.

---

## Deployment

### Railway (recommended)

1. Push to GitHub
2. Connect repo on [railway.app](https://railway.app)
3. Add environment variable: `GROQ_API_KEY`
4. Deploy — the server auto-builds the DB on first startup

### Render (free tier)

1. Push to GitHub
2. Create Web Service on [render.com](https://render.com)
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn chatbot_server:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `GROQ_API_KEY`

---

## Limitations & Future Improvements

**Current limitations:**
- Single-turn only (no conversation memory across messages)
- Small embedding model occasionally retrieves imperfect chunks
- No admin dashboard for document management

**Planned improvements:**
- Conversation memory for multi-turn chat
- Admin dashboard for uploading/managing documents
- Reranker for better retrieval accuracy
- Support for PDF and DOCX document ingestion
- Streaming responses for better UX

---

## License

MIT
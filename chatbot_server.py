"""
Chatbot server — auto-builds vector DB on first startup.

REQUIREMENTS:
    pip install fastapi uvicorn chromadb groq

TO RUN:
    $env:GROQ_API_KEY = "gsk_..."
    uvicorn chatbot_server:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "karachi_bites"
KB_FOLDER = "knowledge_base"
TOP_K = 3
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 10          # keep last 10 messages (5 user + 5 assistant)
SESSION_TTL_SECONDS = 30 * 60   # 30 minutes

SYSTEM_PROMPT = """You are the friendly customer support assistant for Karachi Bites, 
a Pakistani restaurant in F-7 Markaz, Islamabad.

RULES:
- Answer ONLY based on the provided context. Do not make up information.
- If the context doesn't contain the answer, say: "I'm not sure about that. 
  Please contact us on WhatsApp at 0311-5551234 for more details."
- Keep answers concise and helpful — 2-3 sentences max unless the question 
  requires a detailed answer (like listing menu items).
- Be warm and friendly. Use "we" when referring to the restaurant.
- If someone asks about something not related to the restaurant, 
  politely redirect them.
- When mentioning prices, always include "Rs." before the amount.
- If a question could be answered by multiple chunks, synthesize the
  information into one coherent answer.
- You may receive conversation history. Use it to understand follow-up
  questions and resolve references like "it", "that", "how much", etc.
"""


# ============================================================
# AUTO-BUILD: Chunk and embed documents if DB doesn't exist
# ============================================================

def build_database():
    """Build the vector DB from knowledge_base/ folder."""
    print("Building vector database from knowledge_base/...")
    
    chunks = []
    for filename in sorted(os.listdir(KB_FOLDER)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(KB_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.split("\n")
        current_section = "General"
        current_lines = []
        doc_title = ""
        
        if "faq" in filename.lower():
            current_qa = []
            for line in lines:
                stripped = line.strip()
                if not doc_title and stripped and not stripped.startswith("Q:") and not stripped.startswith("==="):
                    doc_title = stripped
                    continue
                if stripped.startswith("==="):
                    continue
                if stripped.startswith("Q:") and current_qa:
                    qa_text = "\n".join(current_qa).strip()
                    if qa_text:
                        chunks.append({"text": qa_text, "source": filename, "section": "FAQ", "doc_title": doc_title})
                    current_qa = [line]
                else:
                    current_qa.append(line)
            qa_text = "\n".join(current_qa).strip()
            if qa_text:
                chunks.append({"text": qa_text, "source": filename, "section": "FAQ", "doc_title": doc_title})
        else:
            for line in lines:
                stripped = line.strip()
                if not doc_title and stripped and not stripped.startswith("==="):
                    doc_title = stripped
                    continue
                if stripped.startswith("===") and stripped.endswith("==="):
                    section_text = "\n".join(current_lines).strip()
                    if section_text and len(section_text) > 20:
                        chunks.append({"text": section_text, "source": filename, "section": current_section, "doc_title": doc_title})
                    current_section = stripped.replace("=", "").strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            section_text = "\n".join(current_lines).strip()
            if section_text and len(section_text) > 20:
                chunks.append({"text": section_text, "source": filename, "section": current_section, "doc_title": doc_title})
    
    for i, chunk in enumerate(chunks):
        source = chunk["source"].replace(".txt", "")
        chunk["chunk_id"] = f"{source}_{i:03d}"
    
    print(f"Created {len(chunks)} chunks")
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"], "doc_title": c["doc_title"]} for c in chunks],
    )
    
    print(f"Database built! {len(chunks)} chunks stored.")
    return client, collection


# ============================================================
# STARTUP
# ============================================================

app = FastAPI(title="Karachi Bites Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to ChromaDB — build it if it doesn't exist
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"Found existing DB with {collection.count()} chunks")
except Exception:
    print("No existing DB found — building from scratch...")
    chroma_client, collection = build_database()

# Groq client
print("Initializing Groq client...")
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not set!")
groq_client = Groq(api_key=groq_api_key)

print(f"Model: {GROQ_MODEL}")
print(f"Chunks: {collection.count()}")
print("Server ready!\n")


# ============================================================
# CONVERSATION MEMORY
# ============================================================

# { session_id: { "messages": [...], "last_active": timestamp } }
conversations: dict[str, dict] = {}


def get_or_create_session(conversation_id: str | None) -> str:
    """Return an existing session ID or create a new one."""
    if conversation_id and conversation_id in conversations:
        conversations[conversation_id]["last_active"] = time.time()
        return conversation_id
    new_id = str(uuid.uuid4())
    conversations[new_id] = {"messages": [], "last_active": time.time()}
    return new_id


def cleanup_expired_sessions():
    """Remove sessions that have been inactive longer than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [sid for sid, s in conversations.items()
               if now - s["last_active"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del conversations[sid]


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    conversation_id: str


# ============================================================
# CORE LOGIC
# ============================================================

def retrieve_context(query: str, top_k: int = TOP_K) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "section": results["metadatas"][0][i]["section"],
            "distance": results["distances"][0][i],
        })
    return chunks


def build_prompt(user_message: str, context_chunks: list[dict]) -> str:
    context_text = ""
    for chunk in context_chunks:
        context_text += f"\n--- Source: {chunk['source']} | Section: {chunk['section']} ---\n"
        context_text += chunk["text"]
        context_text += "\n"
    return f"""Here is the relevant information from our knowledge base:

{context_text}

Customer question: {user_message}

Answer the customer's question based on the context above."""


def call_llm(system_prompt: str, history: list[dict], user_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.3,
    )
    return response.choices[0].message.content


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Housekeeping: prune stale sessions
    cleanup_expired_sessions()

    # Get or create conversation session
    session_id = get_or_create_session(request.conversation_id)
    history = conversations[session_id]["messages"]

    # RAG retrieval (always based on current message)
    context_chunks = retrieve_context(request.message)
    user_prompt = build_prompt(request.message, context_chunks)

    # Send history + new message to LLM
    answer = call_llm(SYSTEM_PROMPT, history[-MAX_HISTORY_TURNS:], user_prompt)

    # Save this turn to session history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})

    sources = [
        {"source": c["source"], "section": c["section"], "chunk_id": c["chunk_id"]}
        for c in context_chunks
    ]
    return ChatResponse(answer=answer, sources=sources, conversation_id=session_id)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "chunks_in_db": collection.count(),
        "llm": GROQ_MODEL,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat widget demo page at the root URL."""
    html_path = os.path.join(os.path.dirname(__file__), "chat_widget.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

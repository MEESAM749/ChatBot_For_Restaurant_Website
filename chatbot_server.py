"""
Chatbot server — auto-builds vector DB on first startup.

REQUIREMENTS:
    pip install fastapi uvicorn chromadb groq

TO RUN:
    $env:GROQ_API_KEY = "gsk_..."
    uvicorn chatbot_server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import uuid
import time
from urllib.parse import quote
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
WHATSAPP_NUMBER = "923115551234"


# ============================================================
# MENU PARSING
# ============================================================

def parse_menu(filepath: str) -> list[dict]:
    """Parse menu.txt into a list of {name, price, category}."""
    items = []
    category = "General"
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("===") and line.endswith("==="):
                category = line.replace("=", "").strip()
                continue
            match = re.match(r"^(.+?)\s*[—\-]\s*Rs\.\s*([\d,]+)", line)
            if match:
                name = match.group(1).strip()
                price = int(match.group(2).replace(",", ""))
                items.append({"name": name, "price": price, "category": category})
    return items


MENU_ITEMS = parse_menu(os.path.join(KB_FOLDER, "menu.txt"))
# Build a lookup dict for fast matching (lowercase name → item)
MENU_LOOKUP = {item["name"].lower(): item for item in MENU_ITEMS}
print(f"Parsed {len(MENU_ITEMS)} menu items")

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
- Customers can order directly through this chat! When listing menu items,
  always include the item name and price in the format "Item Name — Rs. Price"
  so the ordering buttons appear. If a customer wants to order, let them know
  they can use the cart button in the chat header and checkout via WhatsApp.
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
    conversations[new_id] = {"messages": [], "cart": [], "last_active": time.time()}
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


class CartAddRequest(BaseModel):
    conversation_id: str
    item: str
    quantity: int = 1

class CartRemoveRequest(BaseModel):
    conversation_id: str
    item: str

class CartResponse(BaseModel):
    cart: list[dict]
    total: int
    message: str

class CheckoutRequest(BaseModel):
    conversation_id: str
    name: str
    phone: str
    address: str

class CheckoutResponse(BaseModel):
    whatsapp_url: str
    order_summary: str


# ============================================================
# CART HELPERS
# ============================================================

def find_menu_item(query: str) -> dict | None:
    """Find a menu item by case-insensitive exact or substring match."""
    query_lower = query.lower().strip()
    # Exact match first
    if query_lower in MENU_LOOKUP:
        return MENU_LOOKUP[query_lower]
    # Substring match — return first item whose name contains the query
    for name, item in MENU_LOOKUP.items():
        if query_lower in name or name in query_lower:
            return item
    return None


def cart_total(cart: list[dict]) -> int:
    return sum(item["qty"] * item["unit_price"] for item in cart)


def build_whatsapp_url(cart: list[dict], name: str, phone: str, address: str) -> str:
    lines = ["🍛 *New Order — Karachi Bites*", ""]
    lines.append(f"*Customer:* {name}")
    lines.append(f"*Phone:* {phone}")
    lines.append(f"*Address:* {address}")
    lines.append("")
    lines.append("*Order:*")
    for item in cart:
        line_total = item["qty"] * item["unit_price"]
        lines.append(f"• {item['qty']}x {item['name']} — Rs. {line_total:,}")
    lines.append("")
    lines.append(f"*Total: Rs. {cart_total(cart):,}*")
    lines.append("")
    lines.append("Sent via Karachi Bites Chatbot")
    message = "\n".join(lines)
    return f"https://wa.me/{WHATSAPP_NUMBER}?text={quote(message)}"


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


# ============================================================
# CART ENDPOINTS
# ============================================================

@app.post("/cart/add", response_model=CartResponse)
async def cart_add(request: CartAddRequest):
    if request.conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    menu_item = find_menu_item(request.item)
    if not menu_item:
        raise HTTPException(status_code=404, detail=f"Menu item '{request.item}' not found")

    cart = conversations[request.conversation_id]["cart"]
    # If item already in cart, increment qty
    for entry in cart:
        if entry["name"].lower() == menu_item["name"].lower():
            entry["qty"] += request.quantity
            return CartResponse(
                cart=cart, total=cart_total(cart),
                message=f"Updated {menu_item['name']} to {entry['qty']}x"
            )
    # New item
    cart.append({"name": menu_item["name"], "qty": request.quantity, "unit_price": menu_item["price"]})
    return CartResponse(
        cart=cart, total=cart_total(cart),
        message=f"Added {request.quantity}x {menu_item['name']} (Rs. {menu_item['price']})"
    )


@app.post("/cart/remove", response_model=CartResponse)
async def cart_remove(request: CartRemoveRequest):
    if request.conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")

    cart = conversations[request.conversation_id]["cart"]
    menu_item = find_menu_item(request.item)
    item_name_lower = menu_item["name"].lower() if menu_item else request.item.lower()

    original_len = len(cart)
    conversations[request.conversation_id]["cart"] = [
        e for e in cart if e["name"].lower() != item_name_lower
    ]
    cart = conversations[request.conversation_id]["cart"]

    if len(cart) == original_len:
        raise HTTPException(status_code=404, detail="Item not in cart")
    return CartResponse(cart=cart, total=cart_total(cart), message=f"Removed {request.item}")


@app.get("/cart/{conversation_id}", response_model=CartResponse)
async def cart_get(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    cart = conversations[conversation_id]["cart"]
    return CartResponse(cart=cart, total=cart_total(cart), message=f"{len(cart)} item(s) in cart")


@app.post("/cart/checkout", response_model=CheckoutResponse)
async def cart_checkout(request: CheckoutRequest):
    if request.conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    cart = conversations[request.conversation_id]["cart"]
    if not cart:
        raise HTTPException(status_code=400, detail="Cart is empty")

    whatsapp_url = build_whatsapp_url(cart, request.name, request.phone, request.address)
    summary_lines = [f"{e['qty']}x {e['name']} — Rs. {e['qty'] * e['unit_price']:,}" for e in cart]
    summary = "\n".join(summary_lines) + f"\nTotal: Rs. {cart_total(cart):,}"
    return CheckoutResponse(whatsapp_url=whatsapp_url, order_summary=summary)


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

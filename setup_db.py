"""
setup_db.py — Run this ONCE after deployment to build the vector DB on the server.

Railway (and similar platforms) give you persistent storage, so you only 
need to run this once. After that, the chatbot_server.py connects to 
the existing DB on every restart.

Usage:
    python setup_db.py
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


def build_database():
    KB_FOLDER = "knowledge_base"
    CHROMA_PATH = "./chroma_db"
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Check if knowledge_base folder exists
    if not os.path.exists(KB_FOLDER):
        print(f"ERROR: '{KB_FOLDER}' folder not found!")
        print("Make sure your knowledge_base/ folder is in the project.")
        return
    
    # Load and chunk documents
    print("Loading documents...")
    chunks = []
    for filename in sorted(os.listdir(KB_FOLDER)):
        if not filename.endswith(".txt"):
            continue
        
        filepath = os.path.join(KB_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Simple section-based chunking
        lines = content.split("\n")
        current_section = "General"
        current_lines = []
        doc_title = ""
        
        if "faq" in filename.lower():
            # FAQ: split on Q: patterns
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
                        chunks.append({
                            "text": qa_text,
                            "source": filename,
                            "section": "FAQ",
                            "doc_title": doc_title,
                        })
                    current_qa = [line]
                else:
                    current_qa.append(line)
            qa_text = "\n".join(current_qa).strip()
            if qa_text:
                chunks.append({
                    "text": qa_text,
                    "source": filename,
                    "section": "FAQ",
                    "doc_title": doc_title,
                })
        else:
            # Section-based chunking
            for line in lines:
                stripped = line.strip()
                if not doc_title and stripped and not stripped.startswith("==="):
                    doc_title = stripped
                    continue
                if stripped.startswith("===") and stripped.endswith("==="):
                    section_text = "\n".join(current_lines).strip()
                    if section_text and len(section_text) > 20:
                        chunks.append({
                            "text": section_text,
                            "source": filename,
                            "section": current_section,
                            "doc_title": doc_title,
                        })
                    current_section = stripped.replace("=", "").strip()
                    current_lines = []
                else:
                    current_lines.append(line)
            section_text = "\n".join(current_lines).strip()
            if section_text and len(section_text) > 20:
                chunks.append({
                    "text": section_text,
                    "source": filename,
                    "section": current_section,
                    "doc_title": doc_title,
                })
    
    # Add IDs
    for i, chunk in enumerate(chunks):
        source = chunk["source"].replace(".txt", "")
        chunk["chunk_id"] = f"{source}_{i:03d}"
    
    print(f"Created {len(chunks)} chunks from {KB_FOLDER}/")
    
    # Embed
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Store in ChromaDB
    print(f"Storing in ChromaDB at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        client.delete_collection("karachi_bites")
    except:
        pass
    
    collection = client.create_collection(
        name="karachi_bites",
        metadata={"description": "Karachi Bites restaurant knowledge base"}
    )
    
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[emb.tolist() for emb in embeddings],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "source": c["source"],
            "section": c["section"],
            "doc_title": c["doc_title"]
        } for c in chunks],
    )
    
    print(f"\nDone! {len(chunks)} chunks stored in ChromaDB.")
    print("The chatbot server will now use this database.")


if __name__ == "__main__":
    build_database()

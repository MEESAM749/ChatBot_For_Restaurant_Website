"""
setup_db.py — Build the vector DB using ChromaDB's built-in embeddings.

No sentence-transformers, no PyTorch. ChromaDB uses its own lightweight
embedding model (all-MiniLM-L6-v2 via onnxruntime) which is much smaller.

Run once after deployment:
    python setup_db.py
"""

import os
import chromadb


def build_database():
    KB_FOLDER = "knowledge_base"
    CHROMA_PATH = "./chroma_db"
    
    if not os.path.exists(KB_FOLDER):
        print(f"ERROR: '{KB_FOLDER}' folder not found!")
        return
    
    # Load and chunk documents
    print("Loading and chunking documents...")
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
    
    # Store in ChromaDB — let ChromaDB handle embeddings automatically
    # ChromaDB's default embedding function uses all-MiniLM-L6-v2 via onnxruntime
    # No PyTorch needed!
    print("Building ChromaDB (embedding + storing)...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        client.delete_collection("karachi_bites")
    except:
        pass
    
    # When you DON'T pass an embedding_function, ChromaDB uses its default
    # which is the same all-MiniLM-L6-v2 model but via onnxruntime (lightweight)
    collection = client.create_collection(name="karachi_bites")
    
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"], "doc_title": c["doc_title"]} for c in chunks],
    )
    
    print(f"\nDone! {len(chunks)} chunks embedded and stored.")
    print(f"Database at: {CHROMA_PATH}")


if __name__ == "__main__":
    build_database()
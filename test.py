"""
Test script for the chatbot server.

Run the server first:
    uvicorn chatbot_server:app --reload --port 8000

Then run this:
    python test_chatbot.py
"""

import requests
import time

BASE_URL = "http://localhost:8000"

DELAY = 5  # seconds between requests — avoids Gemini free tier rate limits


def ask(question: str):
    """Send a question to the chatbot and print the response."""
    print(f"\n{'='*60}")
    print(f"YOU: {question}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": question}
    )
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code} — {response.text}")
        return
    
    data = response.json()
    
    print(f"\nBOT: {data['answer']}")
    print(f"\n  Sources used:")
    for src in data["sources"]:
        print(f"    - {src['source']} > {src['section']} ({src['chunk_id']})")


if __name__ == "__main__":
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health")
        print(f"Server status: {health.json()}")
    except requests.ConnectionError:
        print("ERROR: Server not running!")
        print("Start it with: uvicorn chatbot_server:app --reload --port 8000")
        exit(1)
    
    questions = [
        "Do you deliver to G-9?",
        "What's the cheapest biryani you have?",
        "Are you guys halal?",
        "I want to book a table for 8 people this Friday",
        "Do you accept JazzCash?",
        "What time do you close on weekends?",
        "Do you have wifi?",
        "What's the weather today?",
    ]
    
    for i, q in enumerate(questions):
        ask(q)
        if i < len(questions) - 1:
            print(f"\n  [waiting {DELAY}s to avoid rate limit...]")
            time.sleep(DELAY)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
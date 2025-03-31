from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import streamlit as st

st.title("🧠 Sherlock Bot is loading...")

# Step 1: Load and Chunk the Book
def load_book(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def split_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Step 2: Load and Process the Book
text = load_book("data.txt")
chunks = split_text(text)

# Step 3: Embed Chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Step 4: Create the FAISS Index
index = faiss.IndexFlatL2(embeddings.shape[1])  # Make sure dimensions match
index.add(np.array(embeddings))

# Step 5: Define the Query Function
def query_book(question, top_k=3):
    question_vec = model.encode([question])
    D, I = index.search(np.array(question_vec), k=top_k)
    results = [chunks[i] for i in I[0]]
    return "\n\n---\n\n".join(results)

# Step 6: Chat Loop
while True:
    query = input("Ask Sherlock Bot (or type 'exit'): ")
    if query.lower() in ['exit', 'quit']:
        break
    answer = query_book(query)
    print("\n🔎 Answer:\n", answer)
    print("\n=========================\n")
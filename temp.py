import google.generativeai as genai
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Configure Gemini API (Use only once)
genai.configure(api_key="AIzaSyCOlqTKvlOSHqV9r91ahNhfkXmmSiFZRhE")

# ✅ Load knowledge base (text file)
with open("info.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# ✅ Split text into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(text_data)

# ✅ Function to get embeddings from Gemini API
def get_gemini_embeddings(text):
    """Generate text embeddings using Google Gemini."""
    model = genai.GenerativeModel("models/embedding-001")  # Gemini Embedding Model
    response = genai.embed_content(model =("models/embedding-001"), content=text)  
    return response["embedding"]  # Extract embeddings from the response

# ✅ Generate embeddings for all text chunks
embeddings = np.array([get_gemini_embeddings(chunk) for chunk in text_chunks])

# ✅ Store embeddings in FAISS for fast retrieval
dimension = embeddings.shape[1]  # Ensure dimension matches
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# ✅ Save FAISS index and text chunks
faiss.write_index(faiss_index, "faiss_index.bin")
np.save("text_chunks.npy", text_chunks)

print("✅ Embeddings stored successfully!")

# ✅ Load FAISS and text chunks for retrieval
faiss_index = faiss.read_index("faiss_index.bin")
text_chunks = np.load("text_chunks.npy", allow_pickle=True)

# ✅ Function to retrieve relevant document
def retrieve_relevant_data(query):
    """Retrieve the most relevant document using FAISS search."""
    query_embedding = np.array(get_gemini_embeddings(query)).reshape(1, -1)  # Ensure correct shape
    distances, indices = faiss_index.search(query_embedding, k=2)  # Retrieve top 2 matches
    return " ".join([text_chunks[i] for i in indices[0] if i < len(text_chunks)])  # Avoid out-of-range error

# ✅ Example: Retrieve relevant info for a user query
query = "Which university is Model Engineering College affiliated to?"
context = retrieve_relevant_data(query)
print("Retrieved Context:", context)

import json

# Load JSON knowledge base
with open("collegedata.json", "r", encoding="utf-8") as file:
    knowledge_data = json.load(file)

# Extract text content for embeddings
text_chunks = [entry["content"] for entry in knowledge_data]
import google.generativeai as genai
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Configure Gemini API
genai.configure(api_key="your_gemini_api_key")

# ✅ Function to get embeddings from Gemini API
def get_gemini_embeddings(text):
    """Generate text embeddings using Google Gemini."""
    try:
        model = genai.GenerativeModel("models/embedding-001")
        response = model.embed_content(content=text, task_type="retrieval_document")
        return np.array(response["embedding"])
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return np.zeros(768)  # Assuming 768-dim embedding

# ✅ Load JSON knowledge base
with open("collegedata.json", "r", encoding="utf-8") as file:
    knowledge_data = json.load(file)

# ✅ Extract text content for embeddings
text_chunks = [entry["content"] for entry in knowledge_data]

# ✅ Generate embeddings for all text chunks
embeddings = np.array([get_gemini_embeddings(chunk) for chunk in text_chunks])

# ✅ Store embeddings in FAISS
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# ✅ Save FAISS index
faiss.write_index(faiss_index, "faiss_index.bin")
np.save("text_chunks.npy", text_chunks)

print("✅ JSON-based embeddings stored successfully!")

# ✅ Function to retrieve relevant document
def retrieve_relevant_data(query):
    """Retrieve the most relevant document using FAISS search."""
    query_embedding = get_gemini_embeddings(query).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, k=2)
    return " ".join([text_chunks[i] for i in indices[0] if i < len(text_chunks)])

# ✅ Function to generate chatbot response
def generate_response(query):
    """Retrieve context from FAISS and generate a final answer using Gemini."""
    context = retrieve_relevant_data(query)
    model = genai.GenerativeModel("gemini-pro")
    
    # Provide context and query to Gemini
    response = model.generate_content(f"Context: {context}\n\nQuestion: {query}\n\nAnswer:")
    
    return response.text  # Return Gemini-generated answer

# ✅ Example: User input and chatbot response
while True:
    user_query = input("\n You: ")
    
    if user_query.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! 👋")
        break
    
    response = generate_response(user_query)
    print(f"Chatbot: {response}")




# Import necessary libraries
import os
import pickle
import faiss
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document as LCDocument

# Import RAG utility functions
from rag_utils import retrieve, rag

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"

# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")

# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings( 
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ---------------------------------------------------------
# Reconstruct and Load the FAISS vector store safely
# ---------------------------------------------------------

# Read raw native FAISS C++ index
index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
pkl_file = os.path.join(VECTORSTORE_PATH, "index.pkl")

index = faiss.read_index(index_file)

# Load raw document data and reconstitute LangChain objects to bypass pickle errors
with open(pkl_file, "rb") as f:
    raw_docstore_dict, index_to_docstore_id = pickle.load(f)

reconstructed_dict = {}
for doc_id, doc_data in raw_docstore_dict.items():
    reconstructed_dict[doc_id] = LCDocument(
        page_content=doc_data["page_content"],
        metadata=doc_data["metadata"]
    )

docstore = InMemoryDocstore(reconstructed_dict)

# Build valid LangChain FAISS instance with correct schema
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

print(f"Vectors successfully loaded and verified: {vectorstore.index.ntotal}")

# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"

# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.
    """
    data = request.get_json()
    query = data["query"]
    k = data.get("k", 2)

    chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore
    )

    return jsonify({
        "query": query,
        "k": k,
        "relevant_chunks": chunks
    })

# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.
    """
    data = request.get_json()
    query = data["query"]
    k = data.get("k", 2)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    answer, chunks = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    return jsonify({
        "query": query,
        "k": k,
        "model_name": model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })

# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.
    """
    data = request.get_json()
    query = data["query"]
    k = data.get("k", 2)
    model_name = data.get("model_name", "gpt-4o-mini")
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 512)

    generated_answer, _ = rag(
        query=query,
        k=k,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        vectorstore=vectorstore
    )

    return jsonify({
        "answer": generated_answer
    })

if __name__ == "__main__":
    rag_api.run(debug=True)

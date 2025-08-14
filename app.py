import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import re
import nltk
import gdown

# --- Configuration ---
# Ensure these variables match the ones in your notebook
# Google Drive file ID for your VectorDB.json
FILE_ID = "17vqN_pUH3jov1E_PQfs4YrwhSvmfLYoo"
# Where to put the downloaded file in Streamlit Cloud (ephemeral, but fine)
DEST_PATH = "/tmp/VectorDB.json"
EMBEDDING_MODEL = "sujet-ai/Marsilia-Embeddings-EN-Large"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct" # Specify your LLM model

# --- Load Data and Model ---
@st.cache_resource(show_spinner=False)
def download_once_from_drive(file_id: str, dest_path: str) -> str:
    """
    Downloads the file from Google Drive exactly once per app process.
    Uses gdown with a public link; if the file already exists, it skips.
    """
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Downloading vector DB from Google Drive..."):
            gdown.download(url, dest_path, quiet=False)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("Download failed or produced an empty file.")
    return dest_path


@st.cache_resource # Cache the model and data loading
def load_resources():
    # Load Vector DB
    path = DEST_PATH
    with open(path, "r") as file:
        Vector_db = json.load(file)

    flat_chunks = []
    all_embeddings = []
    chunk_metadata = []
    seen_chunks = {}  # set of chunk texts

    for doc_name, chunks in Vector_db.items():
        for chunk in chunks:
            chunk_text = chunk['chunk_text']

            # Check if we've seen exact text before
            if chunk_text not in seen_chunks:
                # First occurrence: add to set
                seen_chunks[chunk_text] = len(flat_chunks)

                # normal processing
                flat_chunks.append(chunk_text)
                all_embeddings.append(chunk['embed'])
                chunk_metadata.append({
                    'source_document': doc_name,
                    'chunk_id': chunk['chunk_id'],
                    'original_index': len(flat_chunks) - 1
                })

    # Build FAISS index
    vecs = np.asarray(all_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Load Embedding Model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize Inference Client for LLM
    client = InferenceClient(model=LLM_MODEL)

    return flat_chunks, index, chunk_metadata, model, client

flat_chunks, index, chunk_metadata, model, client = load_resources()


# --- RAG Functions ---
def retrieve(query, top_k=100):
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = query_vector.astype(np.float32)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = scores[0][i]

        if idx < len(flat_chunks):
            results.append({
                "text": flat_chunks[idx].strip(),
                "score": float(score),
                "source": f"{chunk_metadata[idx]['source_document']}_chunk_{chunk_metadata[idx]['chunk_id']}"
            })

    return results

def extract_source(text):
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, text)
    return matches

def rag_implementation(input_query):
    retrieved_knowledge = retrieve(input_query)

    # build the system prompt with source tags
    context_lines = [f' - [{h["source"]}] {h["text"]}' for h in retrieved_knowledge]

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Cite the source ID in square brackets after each fact:\n"
        + "\n".join(context_lines)
    )

    messages = [
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ]

    response = client.chat_completion(messages=messages, max_tokens=512)
    response_text = response.choices[0].message.content
    relevant_docs = extract_source(response_text)

    return [h["source"] for h in retrieved_knowledge], relevant_docs, [h["text"] for h in retrieved_knowledge], response_text


# --- Streamlit App ---
st.title("InvestIQ: Finance Based RAG System")

query = st.text_input("Enter your query:")

if query:
    st.write("Running RAG implementation...")

    # Call your existing rag_implementation function
    retrieved_docs, relevant_docs, retrieved_text, response_text = rag_implementation(query)

    st.subheader("Retrieved Knowledge:")
    for i, doc in enumerate(retrieved_text):
        st.write(f"- {doc}")

    st.subheader("Generated Response:")
    st.write(response_text)

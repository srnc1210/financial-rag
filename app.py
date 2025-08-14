import os
import json
import re
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import gdown

# ========= Config =========
FILE_ID = "17vqN_pUH3jov1E_PQfs4YrwhSvmfLYoo"   # <-- your Drive file id
DEST_PATH = "/tmp/VectorDB.json"
EMBEDDING_MODEL = "sujet-ai/Marsilia-Embeddings-EN-Large"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ========= Helpers =========
@st.cache_resource(show_spinner=False)
def download_once_from_drive(file_id: str, dest_path: str) -> str:
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Downloading vector DB from Google Drive..."):
            gdown.download(url, dest_path, quiet=False)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            raise RuntimeError("Download failed or produced an empty file.")
    return dest_path

@st.cache_data(show_spinner=True)
def load_vector_db(json_path: str):
    # Memory-aware loader: avoid holding the whole dict any longer than needed
    with open(json_path, "r") as f:
        vector_db = json.load(f)

    flat_chunks, all_embeddings, chunk_metadata = [], [], []
    seen = set()
    for doc_name, chunks in vector_db.items():
        for ch in chunks:
            txt = ch["chunk_text"]
            if txt in seen:
                continue
            seen.add(txt)
            flat_chunks.append(txt)
            all_embeddings.append(ch["embed"])
            chunk_metadata.append({
                "source_document": doc_name,
                "chunk_id": ch.get("chunk_id"),
            })

    vecs = np.asarray(all_embeddings, dtype=np.float32)
    # free original dict ASAP
    del vector_db, all_embeddings, seen
    return flat_chunks, vecs, chunk_metadata

@st.cache_resource(show_spinner=True)
def build_faiss_index(vectors: np.ndarray):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

@st.cache_resource(show_spinner=True)
def load_models():
    emb_model = SentenceTransformer(EMBEDDING_MODEL)
    llm_client = InferenceClient(model=LLM_MODEL)
    return emb_model, llm_client

@st.cache_resource(show_spinner=True)
def get_resources():
    """Lazy-init everything. Called only when the user interacts."""
    json_path = download_once_from_drive(FILE_ID, DEST_PATH)
    flat_chunks, vecs, meta = load_vector_db(json_path)
    index = build_faiss_index(vecs)
    emb_model, llm_client = load_models()
    return flat_chunks, index, meta, emb_model, llm_client

# ========= RAG =========
def retrieve(query: str, model, index, flat_chunks, chunk_metadata, top_k: int = 20):
    if index is None or index.ntotal == 0:
        return []

    top_k = int(min(top_k, index.ntotal))
    qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(qvec, top_k)

    results = []
    for r in range(len(indices[0])):
        idx = int(indices[0][r])
        score = float(scores[0][r])
        if 0 <= idx < len(flat_chunks):
            results.append({
                "text": flat_chunks[idx].strip(),
                "score": score,
                "source": f"{chunk_metadata[idx]['source_document']}_chunk_{chunk_metadata[idx]['chunk_id']}",
            })
    return results

def extract_source(text: str):
    return re.findall(r"\[([^\]]+)\]", text)

def rag_implementation(input_query: str, model, client, index, flat_chunks, chunk_metadata):
    retrieved = retrieve(input_query, model, index, flat_chunks, chunk_metadata)
    context_lines = [f' - [{h["source"]}] {h["text"]}' for h in retrieved]
    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Cite the source ID in square brackets after each fact:\n" + "\n".join(context_lines)
    )
    messages = [
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": input_query},
    ]
    resp = client.chat_completion(messages=messages, max_tokens=512)
    text = resp.choices[0].message.content
    return [h["source"] for h in retrieved], extract_source(text), [h["text"] for h in retrieved], text

# ========= UI =========
st.title("InvestIQ: Finance-Based RAG System")

# Optional explicit init button (helps avoid surprise cold-start cost)
with st.sidebar:
    if st.button("Initialize / Warm up", type="primary"):
        with st.spinner("Setting up vector DB and models..."):
            flat_chunks, index, chunk_metadata, model, client = get_resources()
        st.success("Ready!")

query = st.text_input("Enter your query:")

if query:
    with st.spinner("Loading resources..."):
        flat_chunks, index, chunk_metadata, model, client = get_resources()

    with st.spinner("Running RAG..."):
        retrieved_docs, relevant_docs, retrieved_text, response_text = rag_implementation(
            query, model, client, index, flat_chunks, chunk_metadata
        )

    st.caption(f"Loaded {len(flat_chunks)} chunks • FAISS size: {index.ntotal}")
    st.subheader("Retrieved Knowledge (top snippets):")
    for doc in retrieved_text:
        st.write(f"- {doc}")

    st.subheader("Generated Response:")
    st.write(response_text)

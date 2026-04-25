import os
import uuid
import logging
import hashlib
import math
from typing import List

import chromadb
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Initialize ChromaDB client to store data in the artifacts directory
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "chromadb")
os.makedirs(CHROMA_PATH, exist_ok=True)

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="patient_history")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    chroma_client = None
    collection = None

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits text into chunks of roughly `chunk_size` characters."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep overlap words
            overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def _local_embeddings(texts: List[str], dim: int = 64) -> List[List[float]]:
    """Create deterministic local embeddings so RAG never downloads a model at runtime."""

    embeddings: List[List[float]] = []
    for text in texts:
        vector = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            embeddings.append(vector)
            continue
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        embeddings.append([round(value / norm, 6) for value in vector])
    return embeddings

def index_patient_document(patient_id: str, file_path: str, filename: str) -> int:
    """Parses a document, chunks it, and indexes it into ChromaDB."""
    if not collection:
        return 0
        
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    else:
        # Assuming plain text for non-PDFs or we can add more parsers
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return 0
            
    if not text.strip():
        return 0
        
    chunks = chunk_text(text, chunk_size=800, overlap=50)
    
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        ids.append(f"{patient_id}_{uuid.uuid4().hex[:8]}")
        documents.append(chunk)
        metadatas.append({
            "patient_id": patient_id,
            "filename": filename,
            "chunk_index": i
        })
        
    try:
        collection.add(
            documents=documents,
            embeddings=_local_embeddings(documents),
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        logger.error(f"RAG Indexing failed: {e}")
        return 0
    
    return len(chunks)

def index_patient_text(patient_id: str, text: str, source_name: str) -> int:
    """Indexes processed model/fusion summaries without exposing raw image bytes."""
    if not collection or not text.strip():
        return 0
    chunks = chunk_text(text, chunk_size=800, overlap=50)
    ids = [f"{patient_id}_{uuid.uuid4().hex[:8]}" for _ in chunks]
    metadatas = [
        {"patient_id": patient_id, "filename": source_name, "chunk_index": index}
        for index, _ in enumerate(chunks)
    ]
    try:
        collection.add(
            documents=chunks,
            embeddings=_local_embeddings(chunks),
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        logger.error(f"RAG Text Indexing failed: {e}")
        return 0
    return len(chunks)

def retrieve_patient_history(patient_id: str, query: str, top_k: int = 3) -> str:
    """Retrieves relevant chunks from the patient's history."""
    if not collection:
        return ""
        
    try:
        results = collection.query(
            query_embeddings=_local_embeddings([query]),
            n_results=top_k,
            where={"patient_id": patient_id}
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            return ""
            
        retrieved_text = "\n\n---\n\n".join(
            f"From {meta['filename']}:\n{doc}" 
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        )
        return retrieved_text
    except Exception as e:
        logger.error(f"RAG Retrieval failed: {e}")
        return ""

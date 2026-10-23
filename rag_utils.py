from typing import List, Dict, Tuple
import numpy as np
from pypdf import PdfReader


def extract_pdf_pages(pdf_file) -> List[Dict]:
    """Extract text from a PDF while retaining page numbers."""
    reader = PdfReader(pdf_file)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_num, "text": text})
    return pages


def chunk_pages(pages: List[Dict], chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    """Split page text into overlapping character chunks with source page metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks = []
    for page in pages:
        text = page["text"].strip()
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"page": page["page"], "text": chunk})
            if end == len(text):
                break
            start = end - overlap
    return chunks


def retrieve_chunks(question: str, chunks: List[Dict], model, top_k: int = 4) -> List[Tuple[float, Dict]]:
    """Retrieve top-k chunks by cosine similarity over normalized sentence embeddings."""
    if not question.strip() or not chunks:
        return []
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    query_vector = model.encode([question], normalize_embeddings=True)[0]
    scores = np.dot(vectors, query_vector)
    indices = np.argsort(scores)[::-1][: min(top_k, len(chunks))]
    return [(float(scores[i]), chunks[int(i)]) for i in indices]

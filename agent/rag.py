"""Local Document RAG (Retrieval-Augmented Generation) Engine.

Processes uploaded PDF/TXT documents, creates text chunks, generates vector embeddings,
and performs similarity search for semantic context retrieval.
"""

import math
import re
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from config import Config

class DocumentChunk:
    def __init__(self, text: str, source: str, page: int, chunk_id: int):
        self.text = text
        self.source = source
        self.page = page
        self.chunk_id = chunk_id
        self.embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "text": self.text
        }

class LocalRAGEngine:
    """Lightweight in-memory RAG engine for local document processing."""

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[DocumentChunk] = []

    def load_pdf(self, file_path_or_bytes, filename: str = "document.pdf") -> int:
        """Extract text from a PDF file and chunk it."""
        reader = PdfReader(file_path_or_bytes)
        new_chunks = []
        global_chunk_counter = len(self.chunks)

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text:
                continue

            # Split text into overlapping windows
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                if len(chunk_text.strip()) > 20: # Ignore tiny noise
                    chunk = DocumentChunk(
                        text=chunk_text,
                        source=filename,
                        page=page_idx + 1,
                        chunk_id=global_chunk_counter
                    )
                    new_chunks.append(chunk)
                    global_chunk_counter += 1
                
                start += self.chunk_size - self.overlap

        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def load_text(self, text_content: str, filename: str = "note.txt") -> int:
        """Process raw text content into chunks."""
        text = re.sub(r'\s+', ' ', text_content).strip()
        new_chunks = []
        global_chunk_counter = len(self.chunks)

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if len(chunk_text.strip()) > 20:
                chunk = DocumentChunk(
                    text=chunk_text,
                    source=filename,
                    page=1,
                    chunk_id=global_chunk_counter
                )
                new_chunks.append(chunk)
                global_chunk_counter += 1
            
            start += self.chunk_size - self.overlap

        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def _tfidf_vectorize(self, text: str) -> Dict[str, float]:
        """Simple local term-frequency vectorizer fallback."""
        words = re.findall(r'\w+', text.lower())
        if not words:
            return {}
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        length = math.sqrt(sum(c*c for c in counts.values()))
        return {w: c / length for w, c in counts.items()}

    def _cosine_similarity_tf(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Cosine similarity for local keyword vectors."""
        common = set(vec1.keys()) & set(vec2.keys())
        return sum(vec1[w] * vec2[w] for w in common)

    def query(self, query_text: str, top_k: int = Config.MAX_RAG_RESULTS) -> List[Dict[str, Any]]:
        """Retrieve most relevant document chunks for a given query."""
        if not self.chunks:
            return []

        query_vec = self._tfidf_vectorize(query_text)
        scored_chunks = []

        for chunk in self.chunks:
            chunk_vec = self._tfidf_vectorize(chunk.text)
            score = self._cosine_similarity_tf(query_vec, chunk_vec)
            scored_chunks.append((score, chunk))

        # Sort by relevance score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_chunks[:top_k]

        return [
            {
                "score": round(score, 4),
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id
            }
            for score, chunk in top_results if score > 0.05
        ]

    def clear(self):
        """Reset the knowledge base."""
        self.chunks = []

"""Unit tests for Local RAG Engine chunking and retrieval."""

from agent.rag import LocalRAGEngine

def test_rag_text_ingestion_and_query():
    rag = LocalRAGEngine(chunk_size=100, overlap=20)
    text = "Artificial Intelligence and Machine Learning are transforming modern software development. Gemini API provides powerful multimodal models."
    
    num_chunks = rag.load_text(text, filename="sample.txt")
    assert num_chunks > 0
    assert len(rag.chunks) == num_chunks
    
    # Query test
    results = rag.query("Machine Learning")
    assert len(results) > 0
    assert results[0]["source"] == "sample.txt"

def test_rag_clear():
    rag = LocalRAGEngine()
    rag.load_text("Sample data content for testing.")
    assert len(rag.chunks) > 0
    rag.clear()
    assert len(rag.chunks) == 0

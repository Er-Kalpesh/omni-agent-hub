"""FastAPI REST API Service for Omni-Agent Hub.

Provides programmatically accessible endpoints for agent interaction,
document indexing, and tool execution.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import io

from config import Config
from agent.core import OmniAgent
from agent.tools import ALL_AGENT_TOOLS

app = FastAPI(
    title="Omni-Agent Hub REST API",
    description="Production-grade API endpoints for Multimodal Agent Execution, RAG Ingestion, and Tool Tracing.",
    version="1.0.0"
)

# Global Agent Instance
agent = OmniAgent()

class ChatRequest(BaseModel):
    prompt: str = Field(..., example="What is the stock price of GOOGL and calculate sqrt(144) * 5?")
    model: str = Field(default=Config.DEFAULT_MODEL, example="gemini-2.5-flash")
    use_rag: bool = Field(default=True, description="Enable local document knowledge search")
    use_tools: bool = Field(default=True, description="Enable Gemini tool execution")

class ChatResponse(BaseModel):
    status: str
    response: str
    tool_traces: List[Dict[str, Any]]
    rag_citations: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    gemini_api_configured: bool
    model: str
    documents_indexed_chunks: int

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health and agent status endpoint."""
    return HealthResponse(
        status="active",
        gemini_api_configured=agent.is_configured(),
        model=Config.DEFAULT_MODEL,
        documents_indexed_chunks=len(agent.rag_engine.chunks)
    )

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Query the OmniAgent with prompt, tool selection, and RAG search."""
    try:
        text_out, traces, citations = agent.process_query(
            prompt=request.prompt,
            use_rag=request.use_rag,
            use_tools=request.use_tools,
            model_name=request.model
        )
        return ChatResponse(
            status="success",
            response=text_out,
            tool_traces=traces,
            rag_citations=citations
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file to index into the local RAG knowledge base."""
    contents = await file.read()
    filename = file.filename or "uploaded_file"

    if filename.lower().endswith(".pdf"):
        num_chunks = agent.rag_engine.load_pdf(io.BytesIO(contents), filename=filename)
    else:
        text_str = contents.decode("utf-8", errors="ignore")
        num_chunks = agent.rag_engine.load_text(text_str, filename=filename)

    return {
        "status": "success",
        "filename": filename,
        "chunks_indexed": num_chunks,
        "total_knowledge_chunks": len(agent.rag_engine.chunks)
    }

@app.get("/api/tools")
def list_available_tools():
    """List all registered tools available for agent function calling."""
    tools_info = []
    for tool in ALL_AGENT_TOOLS:
        tools_info.append({
            "name": tool.__name__,
            "description": tool.__doc__.strip() if tool.__doc__ else "No documentation."
        })
    return {"status": "success", "count": len(tools_info), "tools": tools_info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.API_PORT)

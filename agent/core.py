"""Core OmniAgent Orchestrator using Google GenAI SDK.

Handles multimodal chat, automatic tool calling execution loops,
and local RAG knowledge retrieval.
"""

import os
from typing import List, Dict, Any, Optional, Generator, Tuple
from google import genai
from google.genai import types
from PIL import Image

from config import Config
from agent.tools import ALL_AGENT_TOOLS
from agent.rag import LocalRAGEngine

class OmniAgent:
    """Multimodal AI Agent orchestrator."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.rag_engine = LocalRAGEngine()
        self.client: Optional[genai.Client] = None
        
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as err:
                print(f"Failed to initialize Gemini Client: {err}")

    def is_configured(self) -> bool:
        """Check if Gemini API is ready to use."""
        return self.client is not None

    def process_query(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        use_rag: bool = True,
        use_tools: bool = True,
        model_name: str = Config.DEFAULT_MODEL
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process a user query with multimodal context, RAG docs, and tool execution.

        Returns:
            Tuple of (response_text, tool_calls_made, rag_citations)
        """
        tool_traces: List[Dict[str, Any]] = []
        rag_citations: List[Dict[str, Any]] = []
        augmented_prompt = prompt

        # Step 1: Query Local RAG Knowledge Base if active
        if use_rag and self.rag_engine.chunks:
            rag_results = self.rag_engine.query(prompt)
            if rag_results:
                rag_citations = rag_results
                context_str = "\n---\n".join(
                    [f"[Doc: {item['source']} (Pg {item['page']})]: {item['text']}" for item in rag_results]
                )
                augmented_prompt = (
                    f"Use the following document context to help answer the user question:\n"
                    f"{context_str}\n\n"
                    f"User Question: {prompt}"
                )

        # Fallback offline answer if API key missing
        if not self.is_configured():
            offline_msg = (
                "⚠️ **Offline Demo Mode**: GEMINI_API_KEY is not configured in `.env`.\n\n"
                f"**Question Processed**: {prompt}\n"
            )
            if rag_citations:
                offline_msg += f"\n**Retrieved {len(rag_citations)} Document Chunks from Local RAG Engine**:\n"
                for cite in rag_citations:
                    offline_msg += f"- *{cite['source']} (Page {cite['page']})*: \"{cite['text'][:120]}...\"\n"
            else:
                offline_msg += "\nTo enable live multimodal responses and Gemini 2.5 Flash execution, add your free key to `.env`."
            return offline_msg, tool_traces, rag_citations

        # Step 2: Build contents list (multimodal image + text)
        contents = []
        if image:
            contents.append(image)
        contents.append(augmented_prompt)

        # Step 3: Configure Tools if enabled
        config = None
        if use_tools:
            config = types.GenerateContentConfig(
                tools=ALL_AGENT_TOOLS,
                system_instruction="You are OmniAgent, a helpful multimodal AI assistant equipped with real-time tools and document knowledge."
            )

        try:
            # Generate content using official Google GenAI SDK
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            # Record any automatic function calls made by SDK
            if hasattr(response, 'function_calls') and response.function_calls:
                for fc in response.function_calls:
                    tool_traces.append({
                        "tool": getattr(fc, 'name', 'unknown_tool'),
                        "args": getattr(fc, 'args', {})
                    })

            response_text = response.text if response.text else "Response generated successfully."
            return response_text, tool_traces, rag_citations

        except Exception as err:
            return f"❌ Error communicating with Gemini API: {str(err)}", tool_traces, rag_citations

"""Streamlit Interactive Web Application for Omni-Agent Hub.

Provides a rich visual UI for interacting with Multimodal AI Agents,
uploading documents for RAG, inspecting tool execution traces, and testing custom tools.
"""

import streamlit as st
from PIL import Image
import os
import io
from dotenv import load_dotenv

# Force reload .env
load_dotenv(override=True)

from config import Config
from agent.core import OmniAgent
from agent.tools import (
    ALL_AGENT_TOOLS,
    search_web_information,
    get_weather_forecast,
    fetch_financial_stock_quote,
    calculate_expression,
    execute_python_code_sandbox
)

# Page Setup & Theme Styling
st.set_page_config(
    page_title="Omni-Agent Hub | Multimodal AI Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .stBadge {
        background-color: #E3F2FD;
        color: #0D47A1;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .citation-box {
        background-color: #F8F9FA;
        border-left: 4px solid #1E88E5;
        padding: 10px;
        margin-top: 5px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Always initialize or refresh OmniAgent instance
if "agent" not in st.session_state or not st.session_state.agent.is_configured():
    st.session_state.agent = OmniAgent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am **OmniAgent**. I can analyze images, search document PDFs, run math calculations, and fetch real-time data using Gemini tools. How can I assist you today?"}
    ]

agent: OmniAgent = st.session_state.agent

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # API Key Configuration Notice
    if agent.is_configured():
        st.success("✅ Gemini API: Connected")
    else:
        st.warning("⚠️ Running in Local/Offline Mode")
        user_key = st.text_input("Enter Gemini API Key (Free):", type="password", help="Get free API key at aistudio.google.com")
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
            st.session_state.agent = OmniAgent(api_key=user_key)
            st.rerun()

    st.markdown("---")
    st.subheader("🤖 Model Settings")
    model_choice = st.selectbox(
        "Select Gemini Model:",
        [Config.DEFAULT_MODEL, Config.REASONING_MODEL],
        index=0
    )
    
    use_rag_toggle = st.toggle("Enable Document RAG Context", value=True)
    use_tools_toggle = st.toggle("Enable Gemini Function Tools", value=True)
    
    st.markdown("---")
    st.subheader("📄 Local RAG Ingestion")
    uploaded_file = st.file_uploader("Upload PDF or TXT Document:", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("📥 Index Document into Knowledge Base", use_container_width=True):
            with st.spinner("Processing text and creating chunks..."):
                file_bytes = io.BytesIO(uploaded_file.getvalue())
                if uploaded_file.name.endswith(".pdf"):
                    num_chunks = agent.rag_engine.load_pdf(file_bytes, filename=uploaded_file.name)
                else:
                    text_str = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    num_chunks = agent.rag_engine.load_text(text_str, filename=uploaded_file.name)
                
                st.success(f"Successfully indexed {num_chunks} chunks!")
    
    chunks_count = len(agent.rag_engine.chunks)
    st.info(f"📚 **Indexed Chunks in Memory**: {chunks_count}")
    
    if chunks_count > 0:
        if st.button("🗑️ Clear Knowledge Base", type="secondary"):
            agent.rag_engine.clear()
            st.rerun()

# --- MAIN INTERACTIVE TABS ---
st.markdown("<div class='main-header'>🤖 Omni-Agent Hub</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Multimodal AI Agent Platform featuring Function Calling, Document RAG & Tool Execution</div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Multimodal Agent Chat",
    "📄 Document RAG Inspector",
    "🛠️ Function & Tool Studio",
    "⚡ REST API & Architecture"
])

# --- TAB 1: MULTIMODAL AGENT CHAT ---
with tab1:
    col_chat, col_inspect = st.columns([2, 1])
    
    with col_chat:
        st.subheader("Chat Interface")
        
        # Display existing message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], caption="Attached Input Image", width=250)
                if "traces" in msg and msg["traces"]:
                    for trace in msg["traces"]:
                        st.caption(f"🛠️ **Tool Used**: `{trace['tool']}` | Args: `{trace['args']}`")
                if "citations" in msg and msg["citations"]:
                    with st.expander("📚 Source Document Citations"):
                        for cite in msg["citations"]:
                            st.markdown(f"**[{cite['source']} - Page {cite['page']}]**: {cite['text']}")

        # User Image Upload Attachment
        uploaded_img = st.file_uploader("Attach Image for Visual Analysis (Optional):", type=["jpg", "png", "jpeg"], key="chat_img")
        pil_image = Image.open(uploaded_img) if uploaded_img else None
        
        if pil_image:
            st.image(pil_image, caption="Preview of Attached Image", width=200)

        # Chat Input Box
        if prompt := st.chat_input("Ask a question, upload a PDF in sidebar, or request calculations/search..."):
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "image": pil_image
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
                if pil_image:
                    st.image(pil_image, width=200)

            # Generate Agent Response
            with st.chat_message("assistant"):
                with st.spinner("OmniAgent is processing..."):
                    response_text, tool_traces, rag_citations = agent.process_query(
                        prompt=prompt,
                        image=pil_image,
                        use_rag=use_rag_toggle,
                        use_tools=use_tools_toggle,
                        model_name=model_choice
                    )
                    
                    st.markdown(response_text)
                    
                    if tool_traces:
                        for trace in tool_traces:
                            st.caption(f"🛠️ **Tool Used**: `{trace['tool']}` | Args: `{trace['args']}`")
                            
                    if rag_citations:
                        with st.expander("📚 Retrieved Knowledge Citations"):
                            for cite in rag_citations:
                                st.markdown(f"**[{cite['source']} - Page {cite['page']}]**: {cite['text']}")

            # Save assistant message to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "traces": tool_traces,
                "citations": rag_citations
            })

    with col_inspect:
        st.subheader("💡 Example Prompts")
        st.markdown("""
        Try testing these capabilities:
        - **Function Calling**: *"What is the stock price of GOOGL and what is sqrt(256) * 12?"*
        - **Web Search Tool**: *"Search the latest updates on quantum computing breakthroughs."*
        - **Weather Tool**: *"Get the current weather forecast for Tokyo."*
        - **Python Code Sandbox**: *"Run a python snippet to generate the first 10 Fibonacci numbers."*
        - **Multimodal Vision**: Upload an image and ask *"Describe this image and analyze its key elements."*
        - **PDF RAG**: Upload a PDF in sidebar and ask *"Summarize key points from the uploaded document."*
        """)

# --- TAB 2: DOCUMENT RAG INSPECTOR ---
with tab2:
    st.subheader("🔍 Local Vector RAG Knowledge Base Inspector")
    st.write("Inspect how the local similarity engine chunks and matches query text with uploaded documents.")
    
    if len(agent.rag_engine.chunks) == 0:
        st.info("ℹ️ No documents indexed yet. Upload a PDF or TXT file using the sidebar control panel.")
    else:
        test_query = st.text_input("Enter test query to evaluate similarity scores:", "summary of document")
        if test_query:
            results = agent.rag_engine.query(test_query)
            st.write(f"### Top Matched Chunks ({len(results)})")
            for res in results:
                st.markdown(f"""
                <div class="citation-box">
                    <strong>Score: {res['score']}</strong> | Source: <code>{res['source']}</code> (Page {res['page']})<br/>
                    <em>"{res['text']}"</em>
                </div>
                """, unsafe_allow_html=True)

# --- TAB 3: FUNCTION & TOOL STUDIO ---
with tab3:
    st.subheader("🛠️ Custom Tool Execution Studio")
    st.write("Test individual Python tool functions registered with Gemini Function Calling.")
    
    tool_choice = st.selectbox("Select Tool to Test:", [
        "search_web_information",
        "get_weather_forecast",
        "fetch_financial_stock_quote",
        "calculate_expression",
        "execute_python_code_sandbox"
    ])
    
    col_in, col_out = st.columns(2)
    
    with col_in:
        st.write("#### Tool Input Parameters")
        if tool_choice == "search_web_information":
            q = st.text_input("Query:", "Gemini 3 Flash developer features")
            if st.button("Run Tool"):
                res = search_web_information(q)
                st.session_state.tool_res = res
                
        elif tool_choice == "get_weather_forecast":
            city = st.text_input("City:", "San Francisco")
            if st.button("Run Tool"):
                res = get_weather_forecast(city)
                st.session_state.tool_res = res
                
        elif tool_choice == "fetch_financial_stock_quote":
            symbol = st.text_input("Ticker Symbol:", "GOOGL")
            if st.button("Run Tool"):
                res = fetch_financial_stock_quote(symbol)
                st.session_state.tool_res = res
                
        elif tool_choice == "calculate_expression":
            expr = st.text_input("Math Expression:", "sqrt(144) * 10 + round(pi, 2)")
            if st.button("Run Tool"):
                res = calculate_expression(expr)
                st.session_state.tool_res = res
                
        elif tool_choice == "execute_python_code_sandbox":
            code = st.text_area("Python Script:", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n\nprint(list(fib(10)))")
            if st.button("Run Tool"):
                res = execute_python_code_sandbox(code)
                st.session_state.tool_res = res

    with col_out:
        st.write("#### Output Response (JSON)")
        if "tool_res" in st.session_state:
            st.json(st.session_state.tool_res)

# --- TAB 4: REST API & ARCHITECTURE ---
with tab4:
    st.subheader("⚡ REST API Architecture & OpenAPI Showcase")
    st.markdown("""
    This project features a decoupled **FastAPI REST API** server (`api.py`) allowing external applications to query the agent programmatically.
    
    #### 🚀 Start Local API Server
    ```bash
    python api.py
    ```
    Once started, interactive OpenAPI docs are accessible at `http://localhost:8000/docs`.
    
    #### 📡 Sample cURL Query
    ```bash
    curl -X POST "http://localhost:8000/api/chat" \\
         -H "Content-Type: application/json" \\
         -d '{
               "prompt": "What is the stock price of GOOGL?",
               "use_rag": true,
               "use_tools": true
             }'
    ```
    """)

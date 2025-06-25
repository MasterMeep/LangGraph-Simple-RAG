# RAG-Langchain

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph that intelligently routes questions between vectorstore retrieval and web search to provide accurate, grounded responses.

## Overview

This RAG system uses a graph-based workflow to ensure high-quality answers by:
- **Intelligent Routing**: Automatically determines whether to use vectorstore or web search
- **Document Grading**: Evaluates relevance of retrieved documents
- **Hallucination Detection**: Ensures responses are grounded in source material
- **Answer Quality Assessment**: Validates that responses actually answer the question
- **Adaptive Retry Logic**: Falls back to web search or retries generation when needed

The system follows this decision flow:

1. **Start** → Question is routed to either vectorstore or web search
2. **Vectorstore Path**:
   - **Retrieve** → Get relevant documents from vectorstore
   - **Grade Documents** → Assess document relevance
   - If documents are relevant → **Generate** answer
   - If documents aren't relevant → **Web Search**
3. **Generate** → Create response using retrieved context
4. **Quality Checks**:
   - Check for hallucinations (grounded in facts?)
   - Assess answer usefulness (answers the question?)
   - **Useful** → **End**
   - **Not Useful** → **Web Search** (fallback)
   - **Not Supported** → Retry generation
   - **Max Retries** → **End**

## Features

- **Local LLM Support**: Uses Ollama with Llama 3.2:3b model
- **Hybrid Retrieval**: Combines vectorstore search with web search via Tavily
- **Document Processing**: Automatically loads and chunks web content
- **Embedding**: Uses Nomic embeddings for document vectorization
- **Quality Assurance**: Multiple validation layers to ensure response quality
- **Configurable Retries**: Adjustable retry limits for answer generation

## Setup
1. Install and set up Ollama with Llama 3.2:3b model:
```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
ollama pull llama3.2:3b
ollama serve
```

2. Set up environment variables:
Create a `.env` file in the project root:
```
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

Run the main script:
```bash
python main.py
```

The system will automatically:
1. Load and process documents from predefined URLs
2. Create a vectorstore with embeddings
3. Process the default question: "what is an agent"
4. Display the generated answer

### Customizing Questions

To ask different questions, modify the `inputs` dictionary in `main.py`:
```python
inputs = {"question": "Your question here", "max_retries": 3}
```

### Adding More Documents

Add URLs to the `urls` list in `main.py`:
```python
urls = [
    "https://your-document-url-1.com",
    "https://your-document-url-2.com",
    # Add more URLs as needed
]
```

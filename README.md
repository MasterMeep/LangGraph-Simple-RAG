# RAG-Langchain

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph that intelligently routes questions between vectorstore retrieval and web search to provide accurate, grounded responses.

## Overview

This RAG system uses a graph-based workflow to ensure high-quality answers by:
- **Intelligent Routing**: Automatically determines whether to use vectorstore or web search
- **Document Grading**: Evaluates relevance of retrieved documents
- **Hallucination Detection**: Ensures responses are grounded in source material
- **Answer Quality Assessment**: Validates that responses actually answer the question
- **Adaptive Retry Logic**: Falls back to web search or retries generation when needed

## Workflow Architecture

![Workflow Diagram](workflow-diagram.png)

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

## Prerequisites

- Python 3.8+
- Ollama installed with Llama 3.2:3b model
- Tavily API key for web search

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG-Langchain
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install langchain langchain-community langchain-ollama langchain-nomic langgraph tavily-python scikit-learn tiktoken python-dotenv
```

4. Set up environment variables:
Create a `.env` file in the project root:
```
TAVILY_API_KEY=your_tavily_api_key_here
```

5. Ensure Ollama is running with Llama 3.2:3b:
```bash
ollama pull llama3.2:3b
ollama serve
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

## Configuration

### Model Settings
- **Local LLM**: Llama 3.2:3b via Ollama
- **Temperature**: 0 (deterministic responses)
- **JSON Mode**: Enabled for structured outputs
- **Chunk Size**: 1000 tokens with 200 token overlap
- **Retrieval**: Top 3 most relevant documents

### Retry Logic
- **Max Retries**: Configurable (default: 3)
- **Fallback Strategy**: Web search when vectorstore fails
- **Quality Gates**: Hallucination detection and usefulness assessment

## Project Structure

```
RAG-Langchain/
├── main.py          # Main application logic and workflow
├── prompts.py       # All prompt templates
├── README.md        # This file
├── .env            # Environment variables (create this)
└── venv/           # Virtual environment
```

## Key Components

### Prompts (`prompts.py`)
Contains all prompt templates for:
- Question routing (vectorstore vs web search)
- Document relevance grading
- RAG response generation
- Hallucination detection
- Answer usefulness assessment

### Main Workflow (`main.py`)
Implements the LangGraph state machine with nodes for:
- Document retrieval
- Document grading
- Answer generation
- Quality assessment
- Web search fallback

## Troubleshooting

**Ollama Connection Issues**:
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`

**Tavily API Errors**:
- Check your API key in `.env` file
- Verify API key is valid and has sufficient credits

**Memory Issues**:
- Reduce chunk size or number of documents
- Use a smaller embedding model if needed

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## License

This project is open source and available under the MIT License. 
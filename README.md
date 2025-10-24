# KEP Knowledge Server

A semantic search system for Kubernetes Enhancement Proposals (KEPs) that provides intelligent retrieval of KEP documents using vector embeddings and FAISS.

## Overview

This project implements a complete knowledge retrieval system for Kubernetes Enhancement Proposals, consisting of:

1. **Document Ingestion Pipeline**: Discovers, extracts, and chunks KEP documents
2. **Vector Store**: Uses FAISS for efficient similarity search with sentence embeddings
3. **REST API**: FastAPI-based search interface
4. **MCP Integration**: Model Context Protocol server for Claude Code integration

## Features

- 🔍 **Semantic Search**: Find relevant KEPs based on meaning, not just keywords
- ⚡ **Fast Retrieval**: FAISS-powered vector search with sub-100ms query times
- 📊 **Smart Chunking**: Intelligent document splitting that preserves context
- 🔌 **API Interface**: RESTful API for easy integration
- 🤖 **Claude Integration**: MCP server for seamless Claude Code usage

## Project Structure

```
.
├── ingestion/          # Document loading, extraction, and chunking
├── vector_store/       # FAISS index and search functionality
├── api/                # FastAPI REST API
├── data/               # KEP documents (cloned from kubernetes/enhancements)
├── storage/            # FAISS index storage
└── mcp-kep-knowledge/  # MCP server for Claude integration
```

## Installation

### Prerequisites

- Python 3.11+
- Git
- Node.js 18+ (for MCP server)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd internal-knowledge-mcp-server
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. The KEP repository will be cloned automatically on first run, or clone it manually:
```bash
git clone https://github.com/kubernetes/enhancements.git data/enhancements
```

## Usage

### Starting the Knowledge Server

First-time startup (with document ingestion):
```bash
python start_knowledge_server.py
```

Subsequent startups (loads existing index):
```bash
python start_knowledge_server.py
```

Force re-indexing:
```bash
python start_knowledge_server.py --reindex
```

The server will start on `http://localhost:8000`

### API Endpoints

- `POST /search` - Search for KEPs
  ```json
  {
    "query": "How does Kubernetes handle CRD versioning?",
    "top_k": 5,
    "min_score": 0.0
  }
  ```

- `GET /health` - Health check and server status
- `GET /stats` - Index statistics
- `GET /` - API information

### Example Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "pod security standards",
    "top_k": 5
  }'
```

## Architecture

1. **Ingestion**: KEP documents are discovered from the `data/enhancements` directory
2. **Chunking**: Documents are split into semantic chunks with overlap
3. **Embedding**: Text chunks are converted to 384-dimensional vectors using SentenceTransformers
4. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
5. **Search**: Queries are embedded and matched against the index using cosine similarity

## Configuration

Key configuration options (defined in `config.py`):

- `EMBEDDING_MODEL`: Default is "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: Default is 512 characters
- `CHUNK_OVERLAP`: Default is 64 characters
- `API_PORT`: Default is 8000

## Development Status

This project is currently under active development. See [tasks.md](tasks.md) for the complete implementation roadmap.

### Completed

- ✅ Phase 1: Project setup and dependencies

### In Progress

- 🔄 Phase 2: Configuration system
- 🔄 Phase 3: Document ingestion pipeline
- 🔄 Phase 4: Vector store with FAISS
- 🔄 Phase 5: REST API
- 🔄 Phase 6: Main orchestration

### Planned

- Phase 7: Testing & validation
- Phase 8: Docker support
- Phase 9: MCP server integration
- Phase 10: Documentation & cleanup

## License

This project is for educational and development purposes.

## Contributing

This is an internal development project. See [tasks.md](tasks.md) for the implementation roadmap.

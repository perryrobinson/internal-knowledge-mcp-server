# Quick Reference

## Project Structure

```
kep-knowledge-server/
├── start_knowledge_server.py          # Main entry point
├── config.py                           # All configuration constants
├── requirements.txt                    # Python dependencies
├── ingestion/
│   ├── document_loader.py              # Discover KEP documents
│   ├── text_extractor.py               # Extract text from files
│   ├── chunker.py                      # Split documents into chunks
│   └── embedder.py                     # Generate embeddings
├── vector_store/
│   └── faiss_store.py                  # FAISS index management
├── api/
│   ├── server.py                       # FastAPI application
│   └── models.py                       # Pydantic models
├── data/
│   └── enhancements/                   # Cloned KEPs repo
└── storage/
    ├── faiss_index/                    # Persisted index
    └── metadata.json                   # Document metadata

mcp-kep-knowledge/
├── package.json                        # Node.js package config
├── tsconfig.json                       # TypeScript config
├── src/
│   └── index.ts                        # MCP server implementation
└── build/                              # Compiled JS output
```

## Key Commands

### Knowledge Server
```bash
# First run (with ingestion)
python start_knowledge_server.py

# Force re-index
python start_knowledge_server.py --reindex

# Test search endpoint
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CRD versioning", "top_k": 5}'

# Health check
curl http://localhost:8000/health

# Statistics
curl http://localhost:8000/stats
```

### MCP Server
```bash
# Build MCP server
cd mcp-kep-knowledge
npm install
npm run build

# Install globally
npm link

# Test (should start and wait for stdio input)
mcp-kep-knowledge
```

## Key Configuration Values

### config.py
- `EMBEDDING_MODEL`: "sentence-transformers/all-MiniLM-L6-v2"
- `EMBEDDING_DIMENSION`: 384
- `CHUNK_SIZE`: 512 tokens
- `CHUNK_OVERLAP`: 50 tokens
- `API_PORT`: 8000

## Data Models

### KEPDocument
```python
kep_id: str              # "sig-api-machinery/0001"
file_path: Path          # Full path
file_type: str           # .md, .yaml, .txt
sig_name: str            # "sig-api-machinery"
kep_number: str          # "0001"
title: str               # Extracted from file
file_size: int           # Bytes
```

### Chunk
```python
chunk_id: str            # "sig-api-machinery/0001_5"
text: str                # Actual content
kep_id: str              # Parent KEP ID
chunk_index: int         # Position (0-indexed)
metadata: dict           # All KEP metadata
```

### SearchResult
```python
chunk_id: str
kep_id: str
text: str
score: float             # Cosine similarity (0-1)
metadata: dict
chunk_index: int
```

## API Endpoints

### POST /search
**Request:**
```json
{
  "query": "How does Kubernetes handle CRD versioning?",
  "top_k": 5,
  "min_score": 0.0
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "sig-api-machinery/0001_5",
      "kep_id": "sig-api-machinery/0001",
      "text": "...",
      "score": 0.87,
      "metadata": {...},
      "chunk_index": 5
    }
  ],
  "query": "How does Kubernetes handle CRD versioning?",
  "total_results": 5,
  "search_time_ms": 45.3
}
```

### GET /health
```json
{
  "status": "healthy",
  "indexed_documents": 156,
  "indexed_chunks": 5432,
  "model_loaded": true
}
```

### GET /stats
```json
{
  "total_keps": 156,
  "total_chunks": 5432,
  "total_vectors": 5432,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dimension": 384
}
```

## MCP Tool Definition

### search_keps
**Purpose:** Search Kubernetes Enhancement Proposals for architecture and design information

**Parameters:**
- `query` (string, required): The search query
- `top_k` (number, optional): Number of results to return (default: 5)

**Example Usage in Claude:**
```
User: "How does Kubernetes handle CRD versioning?"
Claude: [calls search_keps with query="CRD versioning"]
```

## Claude Code Configuration

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac)

```json
{
  "mcpServers": {
    "kep-knowledge": {
      "command": "mcp-kep-knowledge"
    }
  }
}
```

## Troubleshooting

### Ingestion fails with encoding errors
→ Update `text_extractor.py` to handle more encodings

### Search returns irrelevant results
→ Adjust `CHUNK_SIZE` (try 256 or 1024)
→ Increase `CHUNK_OVERLAP` (try 100)
→ Filter by `min_score` (e.g., 0.5)

### Out of memory during embedding
→ Reduce `batch_size` in `embedder.embed_batch()` (try 16 or 8)

### MCP server can't connect to Knowledge Server
→ Ensure Knowledge Server is running on localhost:8000
→ Check firewall settings

### Claude doesn't call the tool
→ Verify MCP config is valid JSON
→ Restart Claude Code
→ Make tool description more specific

## Implementation Phases

1. **Project Setup** - Structure, dependencies, KEPs clone
2. **Configuration** - config.py with all constants
3. **Ingestion** - Loader, extractor, chunker, embedder
4. **Vector Store** - FAISS index with search
5. **REST API** - FastAPI with endpoints
6. **Main** - Orchestration and startup
7. **Testing** - Validate all components
8. **Docker** (Optional) - Containerization
9. **MCP Integration** - Connect to Claude Code
10. **Documentation** - Final docs and cleanup

## Performance Expectations

- **First Ingestion**: 3-5 minutes
- **Index Load**: <30 seconds
- **Search Latency**: 20-50ms
- **Memory Usage**: ~500MB serving
- **Disk Usage**: ~60MB total

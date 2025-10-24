# KEP Knowledge Server - Project Context

## Project Overview

This project builds a **semantic search system for Kubernetes Enhancement Proposals (KEPs)** that integrates with Claude Code through the Model Context Protocol (MCP).

## Architecture

The system consists of two main components:

### 1. Knowledge Server (Python Backend)
- **Technology**: Python, FastAPI, FAISS, sentence-transformers
- **Purpose**: Ingest, index, and search KEP documents using semantic search
- **API**: REST endpoints on `localhost:8000`
- **Vector Store**: FAISS (Facebook AI Similarity Search) with all-MiniLM-L6-v2 embeddings
- **Key Features**:
  - Automatic document ingestion from Kubernetes enhancements repository
  - Intelligent text chunking (preserves markdown structure)
  - Fast vector similarity search (<100ms)
  - Persistent index storage

### 2. MCP Server (TypeScript Wrapper)
- **Technology**: TypeScript/Node.js, MCP SDK
- **Purpose**: Bridge between Claude Code and Knowledge Server
- **Communication**: stdio with Claude Code, HTTP with Knowledge Server
- **Key Features**:
  - Exposes `search_keps` tool to Claude
  - Translates tool calls to HTTP requests
  - Formats results for Claude's consumption

## Data Flow

```
User Question
    ↓
Claude Code (built-in MCP client)
    ↓
MCP Server (search_keps tool via stdio)
    ↓
Knowledge Server (POST /search via HTTP)
    ↓
FAISS Vector Search
    ↓
Results returned up the chain
    ↓
Claude uses context to answer
```

## Key Files

- `start_knowledge_server.py` - Main entry point for Knowledge Server
- `config.py` - Configuration constants
- `ingestion/` - Document loading, extraction, chunking, embedding
- `vector_store/` - FAISS index management
- `api/` - FastAPI server and request/response models
- `mcp-kep-knowledge/` - MCP server implementation

## Implementation Status

See [tasks.md](../tasks.md) for detailed task breakdown and progress tracking.

## Technology Stack

### Python Backend
- **FastAPI**: Modern REST API framework
- **sentence-transformers**: For generating embeddings (all-MiniLM-L6-v2)
- **FAISS**: Facebook's vector similarity search library
- **uvicorn**: ASGI server for FastAPI
- **pyyaml**: YAML parsing for KEP metadata
- **gitpython**: For cloning KEPs repository

### MCP Server
- **TypeScript**: Type-safe implementation
- **@modelcontextprotocol/sdk**: Official MCP SDK
- **Node.js**: Runtime environment

## Performance Targets

- **Ingestion Time**: 3-5 minutes for ~150 KEPs (first run)
- **Index Load Time**: <30 seconds (subsequent runs)
- **Search Latency**: 20-50ms for top-5 results
- **Memory Usage**: ~500MB during serving
- **Disk Usage**: ~50MB for index + ~10MB metadata

## Data Source

**Kubernetes Enhancement Proposals (KEPs)**
- Repository: https://github.com/kubernetes/enhancements
- Structure: `keps/sig-name/number/README.md`
- Formats: Markdown (.md), YAML (.yaml), Text (.txt)
- Size: ~150-200 KEPs, multiple files per KEP

## Configuration

### Knowledge Server
Port: `8000` (configurable in `config.py`)

### MCP Server
Configured in Claude Code's settings:
```json
{
  "mcpServers": {
    "kep-knowledge": {
      "command": "mcp-kep-knowledge"
    }
  }
}
```

## Development Approach

1. **Build Knowledge Server First** - This is the core functionality and can be tested independently
2. **Test with curl** - Validate search quality before adding MCP layer
3. **Add MCP Server** - Thin wrapper to expose to Claude Code
4. **End-to-End Testing** - Test the full integration

## Success Criteria

- ✅ Server starts automatically and ingests KEPs on first run
- ✅ Subsequent starts are fast (load existing index)
- ✅ Search returns semantically relevant results
- ✅ Claude Code can call the tool successfully
- ✅ Results improve Claude's answers about Kubernetes architecture

## Next Steps After Completion

1. Adapt for your real internal documentation
2. Replace KEPs with your company's architecture docs
3. Add authentication if exposing beyond localhost
4. Consider hybrid search (vector + keyword) for better results
5. Add incremental updates instead of full re-indexing

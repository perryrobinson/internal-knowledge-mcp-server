# KEP Knowledge Server - Implementation Tasks

## Phase 1: Project Setup & Dependencies
- [ ] 1.1 Create project directory structure
  - [ ] Create `ingestion/` directory with `__init__.py`
  - [ ] Create `vector_store/` directory with `__init__.py`
  - [ ] Create `api/` directory with `__init__.py`
  - [ ] Create `data/` directory
  - [ ] Create `storage/faiss_index/` directory
- [ ] 1.2 Create `requirements.txt` with all dependencies
- [ ] 1.3 Create `.gitignore` file
- [ ] 1.4 Clone KEPs repository to `data/enhancements`
- [ ] 1.5 Create basic README.md

## Phase 2: Configuration System
- [ ] 2.1 Implement `config.py` with all constants
  - [ ] Define path constants (BASE_DIR, DATA_DIR, STORAGE_DIR, etc.)
  - [ ] Define model configuration (EMBEDDING_MODEL, EMBEDDING_DIMENSION)
  - [ ] Define chunking configuration (CHUNK_SIZE, CHUNK_OVERLAP)
  - [ ] Define API configuration (API_HOST, API_PORT)
  - [ ] Define KEPs configuration (REPO_URL, BASE_PATH, SUPPORTED_FILE_TYPES)

## Phase 3: Document Ingestion Pipeline

### 3.1 Document Loader
- [ ] 3.1.1 Create `ingestion/document_loader.py`
- [ ] 3.1.2 Implement `KEPDocument` dataclass
  - [ ] Add fields: kep_id, file_path, file_type, sig_name, kep_number, title, file_size
- [ ] 3.1.3 Implement `discover_keps()` function
  - [ ] Walk directory structure (keps/sig-name/number/)
  - [ ] Extract metadata from path structure
  - [ ] Filter by SUPPORTED_FILE_TYPES
  - [ ] Return list of KEPDocument objects
- [ ] 3.1.4 Implement `extract_title_from_file()` function
  - [ ] Handle .md files (extract first # heading)
  - [ ] Handle .yaml files (extract 'title' field)
  - [ ] Handle .txt files (extract first non-empty line)

### 3.2 Text Extractor
- [ ] 3.2.1 Create `ingestion/text_extractor.py`
- [ ] 3.2.2 Implement `extract_text()` function
  - [ ] Handle .md files (read as markdown, preserve structure)
  - [ ] Handle .yaml files (convert to readable text)
  - [ ] Handle .txt files (read with encoding detection)
  - [ ] Handle encoding errors (utf-8 → latin-1 → ignore)
  - [ ] Skip binary files
  - [ ] Return empty string for empty files
- [ ] 3.2.3 Implement `clean_markdown()` function
  - [ ] Remove excessive whitespace
  - [ ] Normalize line endings
  - [ ] Keep headers and structure intact
- [ ] 3.2.4 Implement `yaml_to_text()` function
  - [ ] Convert YAML dict to readable key: value format

### 3.3 Document Chunker
- [ ] 3.3.1 Create `ingestion/chunker.py`
- [ ] 3.3.2 Implement `Chunk` dataclass
  - [ ] Add fields: chunk_id, text, kep_id, chunk_index, metadata
- [ ] 3.3.3 Implement `chunk_document()` function
  - [ ] Split on markdown headers first (##, ###, ####)
  - [ ] Split on paragraph breaks if sections too large
  - [ ] Split on sentence boundaries as fallback
  - [ ] Add overlap between chunks
  - [ ] Generate unique chunk IDs
  - [ ] Preserve metadata in each chunk
- [ ] 3.3.4 Implement `split_by_headers()` helper function
- [ ] 3.3.5 Implement `split_by_paragraphs()` helper function
- [ ] 3.3.6 Implement `split_by_sentences()` helper function
- [ ] 3.3.7 Implement `add_overlap()` helper function

### 3.4 Embedder
- [ ] 3.4.1 Create `ingestion/embedder.py`
- [ ] 3.4.2 Implement `Embedder` class
  - [ ] Initialize with model name (all-MiniLM-L6-v2)
  - [ ] Load SentenceTransformer model
  - [ ] Get embedding dimension (384)
- [ ] 3.4.3 Implement `embed_batch()` method
  - [ ] Accept list of texts and batch_size parameter
  - [ ] Process in batches for efficiency
  - [ ] Show progress bar with tqdm
  - [ ] Return numpy array of embeddings
- [ ] 3.4.4 Implement `embed_query()` method
  - [ ] Embed single query string
  - [ ] Return numpy array

## Phase 4: Vector Store with FAISS

### 4.1 FAISS Store
- [ ] 4.1.1 Create `vector_store/faiss_store.py`
- [ ] 4.1.2 Implement `SearchResult` dataclass
  - [ ] Add fields: chunk_id, kep_id, text, score, metadata, chunk_index
- [ ] 4.1.3 Implement `FAISSStore` class
  - [ ] Initialize with dimension parameter
  - [ ] Create IndexFlatIP (inner product) index
  - [ ] Initialize chunks list and chunk_map dict
- [ ] 4.1.4 Implement `add_embeddings()` method
  - [ ] Normalize embeddings for cosine similarity
  - [ ] Add to FAISS index
  - [ ] Store chunk metadata
  - [ ] Update chunk_map
- [ ] 4.1.5 Implement `search()` method
  - [ ] Normalize query embedding
  - [ ] Search FAISS index
  - [ ] Filter by min_score
  - [ ] Attach chunk metadata
  - [ ] Return list of SearchResult objects
- [ ] 4.1.6 Implement `save()` method
  - [ ] Create directories if needed
  - [ ] Save FAISS index to disk
  - [ ] Save metadata as JSON
- [ ] 4.1.7 Implement `load()` method
  - [ ] Load FAISS index from disk
  - [ ] Load metadata from JSON
  - [ ] Reconstruct chunks and chunk_map
- [ ] 4.1.8 Implement `get_stats()` method
  - [ ] Return total_chunks, total_vectors, dimension, unique_keps

## Phase 5: REST API with FastAPI

### 5.1 API Models
- [ ] 5.1.1 Create `api/models.py`
- [ ] 5.1.2 Implement `SearchRequest` model
  - [ ] Add query field (required)
  - [ ] Add top_k field (default: 5, range: 1-50)
  - [ ] Add min_score field (default: 0.0, range: 0.0-1.0)
  - [ ] Add example in Config
- [ ] 5.1.3 Implement `SearchResultModel` model
  - [ ] Add all result fields
- [ ] 5.1.4 Implement `SearchResponse` model
  - [ ] Add results list
  - [ ] Add query, total_results, search_time_ms
- [ ] 5.1.5 Implement `HealthResponse` model
- [ ] 5.1.6 Implement `StatsResponse` model

### 5.2 FastAPI Server
- [ ] 5.2.1 Create `api/server.py`
- [ ] 5.2.2 Initialize FastAPI app with metadata
- [ ] 5.2.3 Add CORS middleware
- [ ] 5.2.4 Create global state variables (vector_store, embedder)
- [ ] 5.2.5 Implement `set_globals()` function
- [ ] 5.2.6 Implement GET `/` root endpoint
- [ ] 5.2.7 Implement POST `/search` endpoint
  - [ ] Validate server is ready
  - [ ] Embed query
  - [ ] Search vector store
  - [ ] Calculate search time
  - [ ] Return formatted results
  - [ ] Handle errors gracefully
- [ ] 5.2.8 Implement GET `/health` endpoint
  - [ ] Return server status
  - [ ] Return index statistics
- [ ] 5.2.9 Implement GET `/stats` endpoint
  - [ ] Return detailed statistics

## Phase 6: Main Orchestration

### 6.1 Main Entry Point
- [ ] 6.1.1 Create `start_knowledge_server.py`
- [ ] 6.1.2 Implement `clone_keps_repo()` function
  - [ ] Check if data directory exists
  - [ ] Clone repository if needed
  - [ ] Handle clone errors
- [ ] 6.1.3 Implement `ingest_and_index()` function
  - [ ] Step 1: Discover KEP documents
  - [ ] Step 2: Extract and chunk text
  - [ ] Step 3: Initialize embedder
  - [ ] Step 4: Generate embeddings in batches
  - [ ] Step 5: Build FAISS index
  - [ ] Save index to disk
  - [ ] Print statistics and summary
  - [ ] Return vector_store and embedder
- [ ] 6.1.4 Implement `load_existing_index()` function
  - [ ] Load FAISS index from disk
  - [ ] Load embedder
  - [ ] Print statistics
  - [ ] Return vector_store and embedder
- [ ] 6.1.5 Implement `start_api_server()` function
  - [ ] Inject dependencies into FastAPI app
  - [ ] Print server information
  - [ ] Start uvicorn server
- [ ] 6.1.6 Implement `main()` function
  - [ ] Parse command-line arguments (--reindex flag)
  - [ ] Ensure KEPs repo exists
  - [ ] Check if index exists
  - [ ] Load or rebuild index based on conditions
  - [ ] Start API server
- [ ] 6.1.7 Add if __name__ == "__main__" block

## Phase 7: Testing & Validation

### 7.1 Manual Testing
- [ ] 7.1.1 Test first-time startup (with ingestion)
  - [ ] Verify KEPs are cloned
  - [ ] Verify documents are discovered
  - [ ] Verify chunks are created
  - [ ] Verify embeddings are generated
  - [ ] Verify FAISS index is built
  - [ ] Verify index is saved to disk
  - [ ] Verify server starts successfully
- [ ] 7.1.2 Test subsequent startup (load existing index)
  - [ ] Verify index loads quickly (<30 seconds)
  - [ ] Verify server starts successfully
- [ ] 7.1.3 Test `--reindex` flag
  - [ ] Verify forced re-ingestion works

### 7.2 API Testing
- [ ] 7.2.1 Test POST `/search` endpoint
  - [ ] Test with various queries
  - [ ] Test with different top_k values
  - [ ] Test with different min_score values
  - [ ] Verify results are relevant
  - [ ] Verify response format
  - [ ] Verify search time is reasonable (<100ms)
- [ ] 7.2.2 Test GET `/health` endpoint
  - [ ] Verify status is "healthy"
  - [ ] Verify statistics are accurate
- [ ] 7.2.3 Test GET `/stats` endpoint
  - [ ] Verify all statistics are present
- [ ] 7.2.4 Test error handling
  - [ ] Test with invalid requests
  - [ ] Test with server not ready

### 7.3 Quality Validation
- [ ] 7.3.1 Test search quality with sample queries
  - [ ] Query: "CRD versioning"
  - [ ] Query: "pod security"
  - [ ] Query: "API deprecation"
  - [ ] Query: "storage volume expansion"
  - [ ] Query: "ingress controller"
- [ ] 7.3.2 Verify top results are relevant
- [ ] 7.3.3 Document any quality issues

## Phase 8: Docker Support (Optional)

- [ ] 8.1 Create `Dockerfile`
  - [ ] Use Python 3.11-slim base image
  - [ ] Install git
  - [ ] Copy requirements.txt and install dependencies
  - [ ] Copy application code
  - [ ] Create directories
  - [ ] Expose port 8000
  - [ ] Set CMD to run server
- [ ] 8.2 Create `docker-compose.yml`
  - [ ] Define knowledge-server service
  - [ ] Map port 8000
  - [ ] Add volume mounts for data and storage
  - [ ] Set environment variables
  - [ ] Add restart policy
- [ ] 8.3 Test Docker deployment
  - [ ] Build image
  - [ ] Start container
  - [ ] Verify server works
  - [ ] Test volume persistence

## Phase 9: MCP Server Integration

### 9.1 MCP Server Setup
- [ ] 9.1.1 Create `mcp-kep-knowledge/` directory
- [ ] 9.1.2 Create `package.json`
  - [ ] Set package name and version
  - [ ] Set type to "module"
  - [ ] Define bin entry point
  - [ ] Add build scripts
  - [ ] Add dependencies (@modelcontextprotocol/sdk)
  - [ ] Add devDependencies (typescript, @types/node)
- [ ] 9.1.3 Create `tsconfig.json`
  - [ ] Configure TypeScript compiler options
  - [ ] Set output directory to build/
  - [ ] Enable ES modules
- [ ] 9.1.4 Create `src/` directory

### 9.2 MCP Server Implementation
- [ ] 9.2.1 Create `src/index.ts`
- [ ] 9.2.2 Import MCP SDK components
  - [ ] Server, StdioServerTransport
  - [ ] Request schemas (CallToolRequestSchema, ListToolsRequestSchema)
- [ ] 9.2.3 Define constants (KNOWLEDGE_SERVER_URL)
- [ ] 9.2.4 Create MCP server instance
  - [ ] Set name and version
  - [ ] Define capabilities (tools)
- [ ] 9.2.5 Implement ListToolsRequestSchema handler
  - [ ] Define search_keps tool
  - [ ] Add description
  - [ ] Define input schema (query, top_k)
- [ ] 9.2.6 Implement CallToolRequestSchema handler
  - [ ] Handle search_keps tool calls
  - [ ] Extract query and top_k from arguments
  - [ ] Make HTTP POST to Knowledge Server
  - [ ] Format results for Claude
  - [ ] Handle errors gracefully
- [ ] 9.2.7 Implement main() function
  - [ ] Create stdio transport
  - [ ] Connect server to transport
  - [ ] Log startup message
- [ ] 9.2.8 Add error handling for main()

### 9.3 MCP Server Build & Install
- [ ] 9.3.1 Run `npm install`
- [ ] 9.3.2 Run `npm run build`
- [ ] 9.3.3 Run `npm link` for global installation
- [ ] 9.3.4 Verify build output exists in `build/`

### 9.4 Claude Code Configuration
- [ ] 9.4.1 Locate Claude Code config file
  - [ ] Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
  - [ ] Windows: `%APPDATA%\Claude\claude_desktop_config.json`
  - [ ] Linux: `~/.config/Claude/claude_desktop_config.json`
- [ ] 9.4.2 Add MCP server configuration
  - [ ] Add "kep-knowledge" to mcpServers
  - [ ] Set command to "mcp-kep-knowledge"
- [ ] 9.4.3 Validate JSON syntax

### 9.5 End-to-End Testing
- [ ] 9.5.1 Start Knowledge Server
  - [ ] Verify server is running on port 8000
  - [ ] Verify health check returns "healthy"
- [ ] 9.5.2 Restart Claude Code
  - [ ] Verify MCP server loads
  - [ ] Check for errors in logs
- [ ] 9.5.3 Test tool availability
  - [ ] Verify search_keps tool appears in Claude's tools
- [ ] 9.5.4 Test tool functionality
  - [ ] Ask: "How does Kubernetes handle CRD versioning?"
  - [ ] Verify Claude calls search_keps tool
  - [ ] Verify Knowledge Server receives request
  - [ ] Verify results are returned
  - [ ] Verify Claude uses results in response
- [ ] 9.5.5 Test error scenarios
  - [ ] Stop Knowledge Server and test error handling
  - [ ] Test with various query types

## Phase 10: Documentation & Cleanup

- [ ] 10.1 Update README.md
  - [ ] Add project overview
  - [ ] Add installation instructions
  - [ ] Add usage instructions
  - [ ] Add configuration guide
  - [ ] Add troubleshooting section
- [ ] 10.2 Add code comments
  - [ ] Document complex functions
  - [ ] Add docstrings to all public functions
  - [ ] Add type hints where missing
- [ ] 10.3 Create example queries document
  - [ ] List sample queries and expected results
- [ ] 10.4 Create architecture diagram
  - [ ] Document component interactions
  - [ ] Document data flow
- [ ] 10.5 Final cleanup
  - [ ] Remove debug print statements
  - [ ] Remove unused imports
  - [ ] Format code consistently
  - [ ] Run linting if configured

## Performance Benchmarks (After Completion)

- [ ] Measure first-run ingestion time
- [ ] Measure index load time
- [ ] Measure search latency (average, p50, p95, p99)
- [ ] Measure memory usage during ingestion
- [ ] Measure memory usage during serving
- [ ] Measure disk usage (index + metadata)
- [ ] Document all metrics

## Future Enhancements (Optional)

- [ ] Add hybrid search (vector + keyword/BM25)
- [ ] Add query expansion
- [ ] Add re-ranking of results
- [ ] Add authentication/API keys
- [ ] Add rate limiting
- [ ] Add logging and monitoring
- [ ] Add metrics collection (Prometheus)
- [ ] Add search analytics
- [ ] Create web UI for testing
- [ ] Add support for incremental updates
- [ ] Add support for multiple document sources
- [ ] Optimize chunking strategy based on testing
- [ ] Experiment with different embedding models

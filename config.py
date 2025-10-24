"""
Configuration file for KEP Knowledge Server.

This module contains all configuration constants for the knowledge server,
including paths, model settings, chunking parameters, API settings, and
KEP repository configuration.
"""

import os
from pathlib import Path

# ============================================================================
# Path Constants
# ============================================================================

# Base directory (project root)
BASE_DIR = Path(__file__).parent.absolute()

# Data directory (where KEPs are stored)
DATA_DIR = BASE_DIR / "data"

# Storage directory (where FAISS index and metadata are stored)
STORAGE_DIR = BASE_DIR / "storage"

# FAISS index directory
FAISS_INDEX_DIR = STORAGE_DIR / "faiss_index"

# KEPs directory (cloned repository)
KEPS_DIR = DATA_DIR / "enhancements"

# KEPs base path (where KEP documents are located within the repo)
KEPS_BASE_PATH = KEPS_DIR / "keps"


# ============================================================================
# Model Configuration
# ============================================================================

# Embedding model from SentenceTransformers
# all-MiniLM-L6-v2 produces 384-dimensional embeddings
# It's fast, efficient, and provides good quality for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Dimension of the embedding vectors
# This matches the all-MiniLM-L6-v2 model output dimension
EMBEDDING_DIMENSION = 384

# Batch size for embedding generation (adjust based on available memory)
EMBEDDING_BATCH_SIZE = 32


# ============================================================================
# Chunking Configuration
# ============================================================================

# Maximum size of each text chunk in characters
# 512 chars provides a good balance between context and granularity
CHUNK_SIZE = 512

# Number of characters to overlap between consecutive chunks
# Overlap helps maintain context across chunk boundaries
CHUNK_OVERLAP = 64

# Minimum chunk size (chunks smaller than this will be merged or discarded)
MIN_CHUNK_SIZE = 50


# ============================================================================
# API Configuration
# ============================================================================

# API server host
API_HOST = "0.0.0.0"

# API server port
API_PORT = 8000

# API title and version
API_TITLE = "KEP Knowledge Server API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Semantic search API for Kubernetes Enhancement Proposals (KEPs)"

# CORS settings (allow all origins for development)
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# Default search parameters
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
DEFAULT_MIN_SCORE = 0.0


# ============================================================================
# KEPs Configuration
# ============================================================================

# KEPs repository URL
REPO_URL = "https://github.com/kubernetes/enhancements.git"

# Supported file types for KEP documents
# We'll process markdown, YAML (for metadata), and text files
SUPPORTED_FILE_TYPES = {".md", ".yaml", ".yml", ".txt"}

# Files and directories to exclude during ingestion
EXCLUDED_PATTERNS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "venv",
    "env",
    ".venv",
}

# KEP path structure: keps/sig-<name>/<number>/
# This regex helps extract metadata from the path
KEP_PATH_PATTERN = r"keps/sig-([^/]+)/(\d+)"


# ============================================================================
# FAISS Index Configuration
# ============================================================================

# Index file name
FAISS_INDEX_FILE = "kep_knowledge.index"

# Metadata file name (stores chunk information as JSON)
FAISS_METADATA_FILE = "kep_metadata.json"

# Full paths
FAISS_INDEX_PATH = FAISS_INDEX_DIR / FAISS_INDEX_FILE
FAISS_METADATA_PATH = FAISS_INDEX_DIR / FAISS_METADATA_FILE


# ============================================================================
# Logging Configuration
# ============================================================================

# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# Performance Configuration
# ============================================================================

# Number of CPU cores to use for parallel processing (None = use all available)
NUM_WORKERS = None

# Whether to normalize embeddings for cosine similarity (True recommended)
NORMALIZE_EMBEDDINGS = True


# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories():
    """
    Create necessary directories if they don't exist.

    This function should be called during server initialization to ensure
    all required directories are present.
    """
    directories = [
        DATA_DIR,
        STORAGE_DIR,
        FAISS_INDEX_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_summary():
    """
    Return a dictionary with current configuration settings.

    Useful for debugging and logging startup configuration.
    """
    return {
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "storage_dir": str(STORAGE_DIR),
            "keps_dir": str(KEPS_DIR),
            "faiss_index_dir": str(FAISS_INDEX_DIR),
        },
        "model": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE,
        },
        "api": {
            "host": API_HOST,
            "port": API_PORT,
            "title": API_TITLE,
            "version": API_VERSION,
        },
        "keps": {
            "repo_url": REPO_URL,
            "supported_file_types": list(SUPPORTED_FILE_TYPES),
        },
    }


# ============================================================================
# Environment Variable Overrides
# ============================================================================

# Allow overriding settings via environment variables
if os.getenv("API_PORT"):
    API_PORT = int(os.getenv("API_PORT"))

if os.getenv("API_HOST"):
    API_HOST = os.getenv("API_HOST")

if os.getenv("EMBEDDING_MODEL"):
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

if os.getenv("CHUNK_SIZE"):
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))

if os.getenv("CHUNK_OVERLAP"):
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

if os.getenv("LOG_LEVEL"):
    LOG_LEVEL = os.getenv("LOG_LEVEL")

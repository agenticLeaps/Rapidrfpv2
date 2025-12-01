import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import streamlit config helper, fallback to os.getenv if not available
try:
    from .streamlit_config import get_config_value, get_int_config, get_float_config, get_bool_config
except ImportError:
    # Fallback functions when streamlit_config is not available
    def get_config_value(key: str, default=None):
        return os.getenv(key, default)
    
    def get_int_config(key: str, default: int = 0) -> int:
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_float_config(key: str, default: float = 0.0) -> float:
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    def get_bool_config(key: str, default: bool = False) -> bool:
        value = os.getenv(key, str(default))
        return str(value).lower() in ('true', '1', 'yes', 'on')

class Config:
    # LLM Provider Settings
    LLM_PROVIDER = get_config_value("LLM_PROVIDER", "gemini")  # "openai" or "gemini"
    
    # OpenAI API Settings (for LLM)
    OPENAI_API_KEY = get_config_value("OPENAI_API_KEY", "")
    
    # Gemini API Settings (for LLM)
    GEMINI_API_KEY = get_config_value("GEMINI_API_KEY", "") or get_config_value("GOOGLE_API_KEY", "")
    GEMINI_MODEL = get_config_value("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_PROJECT_ID = get_config_value("GEMINI_PROJECT_ID", "")
    GEMINI_LOCATION = get_config_value("GEMINI_LOCATION", "global")
    
    # LlamaParse API Settings
    LLAMA_CLOUD_API_KEY = get_config_value("LLAMA_CLOUD_API_KEY", "")
    USE_LLAMAPARSE = get_bool_config("USE_LLAMAPARSE", True)
    LLAMAPARSE_RESULT_TYPE = get_config_value("LLAMAPARSE_RESULT_TYPE", "markdown")
    LLAMAPARSE_LANGUAGE = get_config_value("LLAMAPARSE_LANGUAGE", "en")
    LLAMAPARSE_NUM_WORKERS = get_int_config("LLAMAPARSE_NUM_WORKERS", 1)
    LLAMAPARSE_MAX_WAIT_TIME = get_int_config("LLAMAPARSE_MAX_WAIT_TIME", 300)
    LLAMAPARSE_POLL_INTERVAL = get_int_config("LLAMAPARSE_POLL_INTERVAL", 5)
    LLAMAPARSE_PARSING_METHOD = get_config_value("LLAMAPARSE_PARSING_METHOD", "job_monitoring")  # sync, async, job_monitoring
    
    # HuggingFace Embedding Endpoint
    QWEN_EMBEDDING_ENDPOINT = get_config_value("QWEN_EMBEDDING_ENDPOINT", "mahendraVarmaGokaraju/qwen3-embeddings")
    HF_CLIENT_TIMEOUT = get_int_config("HF_CLIENT_TIMEOUT", 30)
    HF_CLIENT_CONNECT_TIMEOUT = get_int_config("HF_CLIENT_CONNECT_TIMEOUT", 10)
    HF_CLIENT_MAX_RETRIES = get_int_config("HF_CLIENT_MAX_RETRIES", 3)
    
    # Document Processing Settings (Optimized for speed)
    CHUNK_SIZE = get_int_config("CHUNK_SIZE", 1536)  # Increased from 512 for 3x fewer chunks
    CHUNK_OVERLAP = get_int_config("CHUNK_OVERLAP", 75)   # Proportional increase
    
    # Graph Settings (Adjusted for larger chunks)
    MAX_ENTITIES_PER_CHUNK = get_int_config("MAX_ENTITIES_PER_CHUNK", 35)     # Increased for larger chunks
    MAX_RELATIONSHIPS_PER_CHUNK = get_int_config("MAX_RELATIONSHIPS_PER_CHUNK", 25)  # Increased for larger chunks
    IMPORTANT_ENTITY_PERCENTAGE = get_float_config("IMPORTANT_ENTITY_PERCENTAGE", 0.2)
    
    # Community Detection Settings
    LEIDEN_RESOLUTION = get_float_config("LEIDEN_RESOLUTION", 1.0)
    LEIDEN_RANDOM_STATE = get_int_config("LEIDEN_RANDOM_STATE", 42)
    
    # LLM Generation Settings (Optimized)
    DEFAULT_MAX_LENGTH = get_int_config("DEFAULT_MAX_LENGTH", 1536)  # Increased for better context with larger chunks
    DEFAULT_TEMPERATURE = get_float_config("DEFAULT_TEMPERATURE", 0.7)
    DEFAULT_TOP_P = get_float_config("DEFAULT_TOP_P", 0.9)
    
    # Batch Processing (Optimized for performance)
    DEFAULT_BATCH_SIZE = get_int_config("DEFAULT_BATCH_SIZE", 500)  # Dramatically increased for embedding optimization
    
    # Performance Optimization Settings
    LLM_BATCH_SIZE = get_int_config("LLM_BATCH_SIZE", 8)              # Batch LLM calls for efficiency
    MAX_CONCURRENT_CHUNKS = get_int_config("MAX_CONCURRENT_CHUNKS", 4)  # Conservative parallel processing
    PARALLEL_EMBEDDING_BATCH = get_int_config("PARALLEL_EMBEDDING_BATCH", 16)  # Batch embeddings
    ENABLE_PARALLEL_PROCESSING = get_bool_config("ENABLE_PARALLEL_PROCESSING", True)  # Enable parallel chunk processing
    
    # Embedding Optimization
    USE_OPENAI_EMBEDDINGS_ONLY = get_bool_config("USE_OPENAI_EMBEDDINGS_ONLY", False)  # Skip HF embeddings
    
    # Storage
    DATA_DIR = get_config_value("DATA_DIR", "data")
    GRAPH_DB_PATH = get_config_value("GRAPH_DB_PATH", "data/processed/graph.gpickle")
    INDEX_PATH = get_config_value("INDEX_PATH", "data/processed/embeddings")
    HNSW_INDEX_PATH = get_config_value("HNSW_INDEX_PATH", "data/processed/hnsw_index")
    
    # HNSW Settings
    HNSW_DIMENSION = get_int_config("HNSW_DIMENSION", 1536)  # Default embedding dimension (OpenAI/HF compatible)
    HNSW_MAX_ELEMENTS = get_int_config("HNSW_MAX_ELEMENTS", 100000)
    HNSW_EF_CONSTRUCTION = get_int_config("HNSW_EF_CONSTRUCTION", 200)
    HNSW_M = get_int_config("HNSW_M", 50)
    HNSW_SPACE = get_config_value("HNSW_SPACE", "cosine")  # cosine, l2, ip
    
    # Search Settings
    DEFAULT_SEARCH_K = get_int_config("DEFAULT_SEARCH_K", 10)
    PPR_ALPHA = get_float_config("PPR_ALPHA", 0.85)
    PPR_MAX_ITERATIONS = get_int_config("PPR_MAX_ITERATIONS", 100)
    SIMILARITY_WEIGHT = get_float_config("SIMILARITY_WEIGHT", 1.0)
    ACCURACY_WEIGHT = get_float_config("ACCURACY_WEIGHT", 1.0)
    
    # Logging
    LOG_LEVEL = get_config_value("LOG_LEVEL", "INFO")
    
    # API Settings
    API_HOST = get_config_value("API_HOST", "0.0.0.0")
    API_PORT = get_int_config("API_PORT", 5001)
    API_DEBUG = get_bool_config("API_DEBUG", True)
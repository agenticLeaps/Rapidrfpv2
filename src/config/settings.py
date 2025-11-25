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
    # OpenAI API Settings (for LLM)
    OPENAI_API_KEY = get_config_value("OPENAI_API_KEY", "")
    
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
    
    # Document Processing Settings
    CHUNK_SIZE = get_int_config("CHUNK_SIZE", 512)
    CHUNK_OVERLAP = get_int_config("CHUNK_OVERLAP", 50)
    
    # Graph Settings
    MAX_ENTITIES_PER_CHUNK = get_int_config("MAX_ENTITIES_PER_CHUNK", 20)
    MAX_RELATIONSHIPS_PER_CHUNK = get_int_config("MAX_RELATIONSHIPS_PER_CHUNK", 15)
    IMPORTANT_ENTITY_PERCENTAGE = get_float_config("IMPORTANT_ENTITY_PERCENTAGE", 0.2)
    
    # Community Detection Settings
    LEIDEN_RESOLUTION = get_float_config("LEIDEN_RESOLUTION", 1.0)
    LEIDEN_RANDOM_STATE = get_int_config("LEIDEN_RANDOM_STATE", 42)
    
    # LLM Generation Settings
    DEFAULT_MAX_LENGTH = get_int_config("DEFAULT_MAX_LENGTH", 1024)
    DEFAULT_TEMPERATURE = get_float_config("DEFAULT_TEMPERATURE", 0.7)
    DEFAULT_TOP_P = get_float_config("DEFAULT_TOP_P", 0.9)
    
    # Batch Processing
    DEFAULT_BATCH_SIZE = get_int_config("DEFAULT_BATCH_SIZE", 16)
    
    # Storage
    DATA_DIR = get_config_value("DATA_DIR", "data")
    GRAPH_DB_PATH = get_config_value("GRAPH_DB_PATH", "data/processed/graph.gpickle")
    INDEX_PATH = get_config_value("INDEX_PATH", "data/processed/embeddings")
    HNSW_INDEX_PATH = get_config_value("HNSW_INDEX_PATH", "data/processed/hnsw_index")
    
    # HNSW Settings
    HNSW_DIMENSION = get_int_config("HNSW_DIMENSION", 1536)  # OpenAI embedding dimension
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
    LOG_FILE = get_config_value("LOG_FILE", "logs/rapidrfp_rag.log")
    
    # API Settings
    API_HOST = get_config_value("API_HOST", "0.0.0.0")
    API_PORT = get_int_config("API_PORT", 5001)
    API_DEBUG = get_bool_config("API_DEBUG", True)
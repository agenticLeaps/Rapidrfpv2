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
    
    # Graph Settings - Ultra-optimized for maximum speed with GPT-5-nano
    MAX_ENTITIES_PER_CHUNK = get_int_config("MAX_ENTITIES_PER_CHUNK", 4)
    MAX_RELATIONSHIPS_PER_CHUNK = get_int_config("MAX_RELATIONSHIPS_PER_CHUNK", 3)
    IMPORTANT_ENTITY_PERCENTAGE = get_float_config("IMPORTANT_ENTITY_PERCENTAGE", 0.05)
    
    # Community Detection Settings
    LEIDEN_RESOLUTION = get_float_config("LEIDEN_RESOLUTION", 1.0)
    LEIDEN_RANDOM_STATE = get_int_config("LEIDEN_RANDOM_STATE", 42)
    
    # LLM Generation Settings - Ultra-reduced for maximum speed
    DEFAULT_MAX_LENGTH = get_int_config("DEFAULT_MAX_LENGTH", 256)
    DEFAULT_TEMPERATURE = get_float_config("DEFAULT_TEMPERATURE", 0.7)
    DEFAULT_TOP_P = get_float_config("DEFAULT_TOP_P", 0.9)
    
    # Environment Detection
    IS_RENDER = get_bool_config("IS_RENDER", False) or "render" in get_config_value("HOSTNAME", "").lower()
    IS_CLOUD = IS_RENDER or get_bool_config("IS_CLOUD", False)
    
    # Batch Processing - Environment Aware
    DEFAULT_BATCH_SIZE = get_int_config("DEFAULT_BATCH_SIZE", 16)
    RENDER_BATCH_SIZE = get_int_config("RENDER_BATCH_SIZE", 8)
    CLOUD_BATCH_SIZE = get_int_config("CLOUD_BATCH_SIZE", 10)
    
    # Rate Limiting & Delays
    BASE_DELAY = get_float_config("BASE_DELAY", 0.1)
    RENDER_BASE_DELAY = get_float_config("RENDER_BASE_DELAY", 0.5)
    MAX_DELAY = get_float_config("MAX_DELAY", 3.0)
    
    # Memory Management
    GC_EVERY_N_BATCHES = get_int_config("GC_EVERY_N_BATCHES", 10)
    MEMORY_CHECK_INTERVAL = get_int_config("MEMORY_CHECK_INTERVAL", 5)
    
    # Retry Configuration
    MAX_RETRIES = get_int_config("MAX_RETRIES", 3)
    RETRY_DELAY_BASE = get_float_config("RETRY_DELAY_BASE", 1.0)
    
    # Checkpoint Configuration
    ENABLE_CHECKPOINTS = get_bool_config("ENABLE_CHECKPOINTS", True)
    CHECKPOINT_INTERVAL = get_int_config("CHECKPOINT_INTERVAL", 20)
    
    @classmethod
    def get_adaptive_batch_size(cls, total_items: int = None) -> int:
        """Get adaptive batch size based on environment and total items."""
        if cls.IS_RENDER:
            base_size = cls.RENDER_BATCH_SIZE
        elif cls.IS_CLOUD:
            base_size = cls.CLOUD_BATCH_SIZE
        else:
            base_size = cls.DEFAULT_BATCH_SIZE
        
        # For very large datasets, reduce batch size to prevent memory issues
        if total_items and total_items > 1000:
            if cls.IS_RENDER:
                return max(4, min(base_size, 6))  # 4-6 for Render on large datasets
            elif cls.IS_CLOUD:
                return max(6, min(base_size, 8))  # 6-8 for other cloud on large datasets
            else:
                return max(8, min(base_size, 12))  # 8-12 for local on large datasets
        
        return base_size
    
    @classmethod
    def get_adaptive_delay(cls, batch_num: int, total_batches: int = None) -> float:
        """Get adaptive delay based on environment and progress."""
        if cls.IS_RENDER:
            base_delay = cls.RENDER_BASE_DELAY
            # Progressive delay for Render - more aggressive
            progress_factor = (batch_num / max(total_batches or 100, 1)) if total_batches else 0.1
            return min(base_delay + (progress_factor * 2.0), cls.MAX_DELAY)
        elif cls.IS_CLOUD:
            # Moderate delay for other cloud platforms
            progress_factor = (batch_num / max(total_batches or 100, 1)) if total_batches else 0.1
            return min(cls.BASE_DELAY + (progress_factor * 1.0), cls.MAX_DELAY * 0.8)
        else:
            # Minimal delay for local
            return cls.BASE_DELAY
    
    # Storage - migrated from NeonDB to Neo4j
    DATA_DIR = get_config_value("DATA_DIR", "data")
    GRAPH_DB_PATH = get_config_value("GRAPH_DB_PATH", "data/processed/graph.gpickle")
    INDEX_PATH = get_config_value("INDEX_PATH", "data/processed/embeddings")
    HNSW_INDEX_PATH = get_config_value("HNSW_INDEX_PATH", "data/processed/hnsw_index")
    
    # Neo4j Settings
    NEO4J_URI = get_config_value("NEO4J_URI", "neo4j+s://af7c3a71.databases.neo4j.io")
    NEO4J_USERNAME = get_config_value("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = get_config_value("NEO4J_PASSWORD", "")
    
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
    LOG_LEVEL = get_config_value("LOG_LEVEL", "CRITICAL")  # Disabled for performance
    LOG_FILE = get_config_value("LOG_FILE", "logs/rapidrfp_rag.log")
    
    # API Settings
    API_HOST = get_config_value("API_HOST", "0.0.0.0")
    API_PORT = get_int_config("API_PORT", 5001)
    API_DEBUG = get_bool_config("API_DEBUG", True)
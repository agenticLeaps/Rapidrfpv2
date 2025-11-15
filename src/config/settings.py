import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API Settings (for LLM)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
    
    # LlamaParse API Settings
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "llx-39hXdpTryilYqVOHcd3nu9FXaNRTIZS625IZNR11idRfqk1u")
    USE_LLAMAPARSE = os.getenv("USE_LLAMAPARSE", "True").lower() == "true"
    LLAMAPARSE_RESULT_TYPE = os.getenv("LLAMAPARSE_RESULT_TYPE", "markdown")
    LLAMAPARSE_LANGUAGE = os.getenv("LLAMAPARSE_LANGUAGE", "en")
    LLAMAPARSE_NUM_WORKERS = int(os.getenv("LLAMAPARSE_NUM_WORKERS", "1"))
    LLAMAPARSE_MAX_WAIT_TIME = int(os.getenv("LLAMAPARSE_MAX_WAIT_TIME", "300"))
    LLAMAPARSE_POLL_INTERVAL = int(os.getenv("LLAMAPARSE_POLL_INTERVAL", "5"))
    LLAMAPARSE_PARSING_METHOD = os.getenv("LLAMAPARSE_PARSING_METHOD", "job_monitoring")  # sync, async, job_monitoring
    
    # HuggingFace Embedding Endpoint
    QWEN_EMBEDDING_ENDPOINT = os.getenv("QWEN_EMBEDDING_ENDPOINT", "mahendraVarmaGokaraju/qwen3-embeddings")
    
    # Document Processing Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Graph Settings
    MAX_ENTITIES_PER_CHUNK = int(os.getenv("MAX_ENTITIES_PER_CHUNK", "20"))
    MAX_RELATIONSHIPS_PER_CHUNK = int(os.getenv("MAX_RELATIONSHIPS_PER_CHUNK", "15"))
    IMPORTANT_ENTITY_PERCENTAGE = float(os.getenv("IMPORTANT_ENTITY_PERCENTAGE", "0.2"))
    
    # Community Detection Settings
    LEIDEN_RESOLUTION = float(os.getenv("LEIDEN_RESOLUTION", "1.0"))
    LEIDEN_RANDOM_STATE = int(os.getenv("LEIDEN_RANDOM_STATE", "42"))
    
    # LLM Generation Settings
    DEFAULT_MAX_LENGTH = int(os.getenv("DEFAULT_MAX_LENGTH", "1024"))
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    
    # Batch Processing
    DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "16"))
    
    # Storage
    DATA_DIR = os.getenv("DATA_DIR", "data")
    GRAPH_DB_PATH = os.getenv("GRAPH_DB_PATH", "data/processed/graph.gpickle")
    INDEX_PATH = os.getenv("INDEX_PATH", "data/processed/embeddings")
    HNSW_INDEX_PATH = os.getenv("HNSW_INDEX_PATH", "data/processed/hnsw_index")
    
    # HNSW Settings
    HNSW_DIMENSION = int(os.getenv("HNSW_DIMENSION", "2560"))  # HuggingFace Qwen embedding dimension
    HNSW_MAX_ELEMENTS = int(os.getenv("HNSW_MAX_ELEMENTS", "100000"))
    HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    HNSW_M = int(os.getenv("HNSW_M", "50"))
    HNSW_SPACE = os.getenv("HNSW_SPACE", "cosine")  # cosine, l2, ip
    
    # Search Settings
    DEFAULT_SEARCH_K = int(os.getenv("DEFAULT_SEARCH_K", "10"))
    PPR_ALPHA = float(os.getenv("PPR_ALPHA", "0.85"))
    PPR_MAX_ITERATIONS = int(os.getenv("PPR_MAX_ITERATIONS", "100"))
    SIMILARITY_WEIGHT = float(os.getenv("SIMILARITY_WEIGHT", "1.0"))
    ACCURACY_WEIGHT = float(os.getenv("ACCURACY_WEIGHT", "1.0"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/rapidrfp_rag.log")
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "5001"))
    API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"
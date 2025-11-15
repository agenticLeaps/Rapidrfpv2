# System Endpoints

System-level endpoints for health monitoring and configuration management.

## Health Check

### `GET /health`

Check if the API server is running and healthy.

**Request:**
```bash
curl http://localhost:5001/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "RapidRFP RAG API"
}
```

**Status Codes:**
- `200 OK` - Service is healthy

---

## Get Configuration

### `GET /api/config`

Retrieve current system configuration settings.

**Request:**
```bash
curl http://localhost:5001/api/config
```

**Response:**
```json
{
  "chunk_size": 512,
  "chunk_overlap": 50,
  "max_entities_per_chunk": 20,
  "max_relationships_per_chunk": 15,
  "important_entity_percentage": 0.2,
  "leiden_resolution": 1.0,
  "default_batch_size": 16,
  "llm_endpoints": {
    "qwen_llm": "mahendraVarmaGokaraju/qwen2.5",
    "qwen_embedding": "mahendraVarmaGokaraju/qwen3-embeddings"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_size` | integer | Token size for document chunks |
| `chunk_overlap` | integer | Overlap tokens between chunks |
| `max_entities_per_chunk` | integer | Maximum entities extracted per chunk |
| `max_relationships_per_chunk` | integer | Maximum relationships per chunk |
| `important_entity_percentage` | float | Percentage of entities considered important |
| `leiden_resolution` | float | Community detection resolution parameter |
| `default_batch_size` | integer | Default batch size for LLM calls |
| `llm_endpoints` | object | LLM service endpoint configuration |

**Status Codes:**
- `200 OK` - Configuration retrieved successfully
- `500 Internal Server Error` - Server error

**Example Usage:**

```python
import requests

# Get current configuration
response = requests.get('http://localhost:5001/api/config')
config = response.json()

print(f"Chunk size: {config['chunk_size']}")
print(f"LLM endpoint: {config['llm_endpoints']['qwen_llm']}")
```

**Environment Variables:**

These configuration values are controlled by environment variables in `.env`:

```bash
# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Graph Settings  
MAX_ENTITIES_PER_CHUNK=20
MAX_RELATIONSHIPS_PER_CHUNK=15
IMPORTANT_ENTITY_PERCENTAGE=0.2

# Community Detection
LEIDEN_RESOLUTION=1.0
LEIDEN_RANDOM_STATE=42

# LLM Settings
DEFAULT_BATCH_SIZE=16
QWEN_LLM_ENDPOINT=mahendraVarmaGokaraju/qwen2.5
QWEN_EMBEDDING_ENDPOINT=mahendraVarmaGokaraju/qwen3-embeddings
```
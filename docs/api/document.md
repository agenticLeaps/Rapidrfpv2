# Document Processing Endpoints

Endpoints for document indexing and processing cost estimation.

## Index Document (File Path)

### `POST /api/index/document`

Index a document from an existing file path through the complete pipeline (Phase I + Phase II).

**Request:**
```bash
curl -X POST http://localhost:5001/api/index/document \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'
```

**Request Body:**
```json
{
  "file_path": "/absolute/path/to/document.pdf"
}
```

---

## Upload and Index Document (File Upload)

### `POST /api/upload/document`

Upload a document file and index it through the complete pipeline. Uses multipart form data.

**Request:**
```bash
curl -X POST http://localhost:5001/api/upload/document \
  -F "file=@/path/to/document.pdf"
```

**Form Data:**
- **file**: The document file to upload (required)

**Content-Type**: `multipart/form-data`

**Response:**
```json
{
  "success": true,
  "processing_time": 45.32,
  "uploaded_file": "uuid123_document.pdf",
  "original_filename": "document.pdf",
  "document_metadata": {
    "file_name": "uuid123_document.pdf",
    "file_extension": ".pdf",
    "file_size": 2048576,
    "total_chars": 15420
  },
  "graph_stats": {
    "total_nodes": 157,
    "total_edges": 298,
    "node_type_counts": {
      "T": 12,
      "S": 45,
      "N": 67,
      "R": 23,
      "A": 8,
      "H": 2,
      "O": 2
    },
    "num_communities": 3,
    "is_connected": true
  },
  "chunks_processed": 12,
  "graph_saved": true
}
```

**Python Example:**
```python
import requests

# Upload and index a file
with open('/path/to/document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5001/api/upload/document', files=files)
    result = response.json()
```

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5001/api/upload/document', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Features:**
- Automatic file cleanup after processing
- Unique filename generation to prevent conflicts
- File format validation
- Temporary storage in `data/raw/` directory

---

## Common Fields

**Request Fields (File Path Method):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Yes | Absolute path to the document file |

**Supported File Formats:**
- PDF (`.pdf`)
- Microsoft Word (`.docx`) 
- Plain Text (`.txt`)
- Markdown (`.md`)

**Response:**
```json
{
  "success": true,
  "processing_time": 45.32,
  "document_metadata": {
    "file_path": "/path/to/document.pdf",
    "file_name": "document.pdf",
    "file_extension": ".pdf",
    "file_size": 2048576,
    "total_chars": 15420
  },
  "graph_stats": {
    "total_nodes": 157,
    "total_edges": 298,
    "node_type_counts": {
      "T": 12,
      "S": 45,
      "N": 67,
      "R": 23,
      "A": 8,
      "H": 2,
      "O": 2
    },
    "num_communities": 3,
    "is_connected": true
  },
  "chunks_processed": 12,
  "graph_saved": true
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether indexing completed successfully |
| `processing_time` | float | Total processing time in seconds |
| `document_metadata` | object | Document file information |
| `graph_stats` | object | Statistics about the generated graph |
| `chunks_processed` | integer | Number of document chunks processed |
| `graph_saved` | boolean | Whether graph was saved to disk |

**Graph Statistics Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_nodes` | integer | Total number of nodes in graph |
| `total_edges` | integer | Total number of edges in graph |
| `node_type_counts` | object | Count of each node type (T, S, N, R, A, H, O) |
| `num_communities` | integer | Number of detected communities |
| `is_connected` | boolean | Whether the graph is connected |

**Status Codes:**
- `200 OK` - Document indexed successfully
- `400 Bad Request` - Missing or invalid file_path
- `404 Not Found` - File not found
- `500 Internal Server Error` - Processing error

**Error Response:**
```json
{
  "error": "File not found: /path/to/document.pdf"
}
```

---

## Estimate Processing Cost

### `POST /api/estimate/cost`

Estimate the processing cost and resource requirements for a document before indexing.

**Request:**
```bash
curl -X POST http://localhost:5001/api/estimate/cost \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'
```

**Request Body:**
```json
{
  "file_path": "/absolute/path/to/document.pdf"
}
```

**Response:**
```json
{
  "estimated_chars": 15420,
  "estimated_tokens": 3855,
  "estimated_chunks": 8,
  "estimated_llm_calls": 8,
  "estimated_embedding_calls": 24,
  "file_size_mb": 2.0
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `estimated_chars` | integer | Estimated character count |
| `estimated_tokens` | integer | Estimated token count |
| `estimated_chunks` | integer | Estimated number of chunks |
| `estimated_llm_calls` | integer | Estimated LLM API calls needed |
| `estimated_embedding_calls` | integer | Estimated embedding API calls needed |
| `file_size_mb` | float | File size in megabytes |

**Status Codes:**
- `200 OK` - Cost estimation completed
- `400 Bad Request` - Missing or invalid file_path
- `404 Not Found` - File not found
- `500 Internal Server Error` - Estimation error

**Example Usage:**

```python
import requests

# Estimate cost before processing
response = requests.post('http://localhost:5001/api/estimate/cost', 
                        json={'file_path': '/path/to/large_document.pdf'})
estimate = response.json()

print(f"Estimated processing time: ~{estimate['estimated_llm_calls'] * 2} seconds")
print(f"File size: {estimate['file_size_mb']:.1f} MB")
print(f"Will create ~{estimate['estimated_chunks']} chunks")

# Proceed with indexing if estimates are acceptable
if estimate['file_size_mb'] < 10:  # Only process files under 10MB
    response = requests.post('http://localhost:5001/api/index/document',
                           json={'file_path': '/path/to/large_document.pdf'})
```

**Processing Pipeline:**

1. **Document Loading**: Parse file based on extension
2. **Chunking**: Split into overlapping token-based chunks
3. **Phase I - Decomposition**: Extract T, S, N, R nodes per chunk
4. **Entity Deduplication**: Merge duplicate entities across chunks
5. **Phase II - Augmentation**: Generate A, H, O nodes
6. **Community Detection**: Identify semantic clusters
7. **Graph Storage**: Save complete graph to disk

**Performance Notes:**

- Processing time scales roughly linearly with document size
- PDF files may take longer due to text extraction overhead
- Large documents (>100 pages) may require several minutes
- LLM calls are the primary bottleneck in processing time
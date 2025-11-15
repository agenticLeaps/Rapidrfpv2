# Error Handling

Comprehensive guide to API error codes, messages, and troubleshooting.

## HTTP Status Codes

| Code | Description | When It Occurs |
|------|-------------|----------------|
| `200` | OK | Request successful |
| `400` | Bad Request | Invalid request parameters |
| `404` | Not Found | Resource not found |
| `405` | Method Not Allowed | Wrong HTTP method |
| `500` | Internal Server Error | Server-side error |

## Error Response Format

All errors follow this consistent format:

```json
{
  "error": "Human-readable error description",
  "details": "Additional technical details (optional)",
  "traceback": "Full Python traceback (debug mode only)"
}
```

## Common Errors

### 1. File Not Found (404)

**Occurs when:**
- Document file path doesn't exist
- Graph file not found during load operation

**Example:**
```json
{
  "error": "File not found: /path/to/nonexistent.pdf"
}
```

**Solutions:**
- Verify file path is correct and absolute
- Check file permissions
- Ensure file exists before API call

### 2. Invalid Request Parameters (400)

**Occurs when:**
- Missing required fields
- Invalid node type codes
- Malformed JSON

**Examples:**

Missing file_path:
```json
{
  "error": "file_path is required"
}
```

Invalid node type:
```json
{
  "error": "Invalid node type: X. Valid types: ['T', 'S', 'N', 'R', 'A', 'H', 'O']"
}
```

**Solutions:**
- Check request body contains all required fields
- Validate node type codes (T, S, N, R, A, H, O)
- Ensure JSON is properly formatted

### 3. Processing Failures (500)

**Occurs when:**
- LLM service unavailable
- Document parsing errors
- Graph operation failures
- Memory or resource constraints

**Example:**
```json
{
  "error": "Failed to connect to LLM service",
  "details": "Connection timeout to mahendraVarmaGokaraju/qwen2.5"
}
```

## Endpoint-Specific Errors

### Document Indexing Errors

**Unsupported File Format:**
```json
{
  "error": "Unsupported file format: .xlsx"
}
```

**Document Processing Error:**
```json
{
  "error": "Error in Phase I decomposition",
  "details": "LLM extraction failed for chunk 5: API rate limit exceeded"
}
```

**Solutions:**
- Use supported formats: PDF, DOCX, TXT, MD
- Check LLM service availability
- Reduce document size if memory issues occur
- Implement retry logic for rate limits

### Graph Operation Errors

**Node Not Found:**
```json
{
  "error": "Node not found: invalid_node_id"
}
```

**Community Detection Failed:**
```json
{
  "error": "Community detection failed",
  "details": "python-louvain library not available"
}
```

**Graph Save/Load Errors:**
```json
{
  "error": "Error saving graph: Permission denied",
  "details": "Cannot write to data/processed/graph.gpickle"
}
```

**Solutions:**
- Verify node IDs exist before querying
- Install required dependencies
- Check file system permissions
- Ensure sufficient disk space

### LLM Service Errors

**Connection Errors:**
```json
{
  "error": "Failed to initialize LLM clients",
  "details": "HTTPError: 403 Forbidden"
}
```

**Rate Limiting:**
```json
{
  "error": "LLM service rate limit exceeded",
  "details": "Too many requests to embedding service"
}
```

**Invalid Responses:**
```json
{
  "error": "Invalid LLM response format",
  "details": "Expected JSON array but got string"
}
```

**Solutions:**
- Check LLM endpoint configuration in .env
- Implement backoff/retry logic
- Verify API access permissions
- Add response validation

## Debugging Tips

### 1. Enable Debug Mode

Set in `.env`:
```bash
API_DEBUG=True
LOG_LEVEL=DEBUG
```

This provides:
- Detailed error tracebacks
- Request/response logging
- Step-by-step processing logs

### 2. Check Logs

View application logs:
```bash
tail -f logs/rapidrfp_rag.log
```

Common log patterns:
```
INFO - Starting Phase I: Graph Decomposition
WARNING - LLM extraction failed for chunk 3
ERROR - Error processing chunk 5: Connection timeout
```

### 3. Validate Environment

Check configuration:
```bash
curl http://localhost:5001/api/config
```

Verify endpoints are accessible:
```python
from gradio_client import Client
client = Client("mahendraVarmaGokaraju/qwen2.5")
# Should not raise exceptions
```

### 4. Test with Small Documents

If processing fails:
1. Try a small test document first
2. Check if specific file formats cause issues
3. Verify basic functionality before large documents

## Error Recovery

### Automatic Retry Logic

For transient errors, implement retry with exponential backoff:

```python
import time
import requests

def index_document_with_retry(file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post('http://localhost:5001/api/index/document',
                                   json={'file_path': file_path})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            time.sleep(wait_time)
```

### Graceful Degradation

Handle partial failures:

```python
def safe_index_document(file_path):
    # First estimate cost
    try:
        cost_response = requests.post('http://localhost:5001/api/estimate/cost',
                                    json={'file_path': file_path})
        cost_response.raise_for_status()
        estimate = cost_response.json()
        
        # Skip very large files
        if estimate['file_size_mb'] > 50:
            return {'skipped': True, 'reason': 'File too large'}
            
    except requests.exceptions.RequestException:
        # Continue anyway if estimation fails
        pass
    
    # Attempt indexing
    try:
        response = requests.post('http://localhost:5001/api/index/document',
                               json={'file_path': file_path})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e), 'file_path': file_path}
```

## Monitoring and Alerting

### Health Checks

Implement regular health monitoring:

```python
def check_api_health():
    try:
        response = requests.get('http://localhost:5001/health', timeout=5)
        return response.status_code == 200
    except:
        return False
```

### Performance Monitoring

Track processing times and error rates:

```python
import time

def monitor_processing():
    start_time = time.time()
    try:
        result = index_document(file_path)
        processing_time = time.time() - start_time
        log_success(processing_time, result)
    except Exception as e:
        processing_time = time.time() - start_time
        log_error(processing_time, str(e))
```

## Support and Reporting

When reporting issues, include:

1. **Request Details**: Full request body and headers
2. **Response**: Complete error response
3. **Environment**: API version, Python version, OS
4. **Logs**: Relevant log entries with timestamps
5. **Reproduction**: Minimal example to reproduce the issue

**Example Bug Report:**

```
Environment:
- API Version: 1.0.0
- Python: 3.13
- OS: macOS 14.0

Request:
POST /api/index/document
{"file_path": "/Users/test/document.pdf"}

Response:
{"error": "LLM extraction failed", "details": "Connection timeout"}

Logs:
2024-01-01 10:30:15 - ERROR - Error extracting entities: HTTPSConnectionPool timeout
```
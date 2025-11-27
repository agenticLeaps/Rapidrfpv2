# Chunk Processing Endpoint Documentation

## Overview

The `/api/v1/process-chunk` endpoint allows you to process individual text chunks through the complete NodeRAG pipeline without storing results in the database. Instead, all processing results and performance metrics are saved to workspace files for inspection and analysis.

## Endpoint Details

- **URL**: `/api/v1/process-chunk`
- **Method**: `POST`
- **Content-Type**: `application/json`

## Request Format

```json
{
    "chunk_text": "Your text content to process",
    "chatgpt_model": "gpt-4",
    "chunk_id": "optional-custom-chunk-id"
}
```

### Required Fields

- `chunk_text` (string): The text content to process through the NodeRAG pipeline
- `chatgpt_model` (string): The model identifier (e.g., "gpt-4", "gpt-3.5-turbo")

### Optional Fields

- `chunk_id` (string): Custom identifier for the chunk. If not provided, a timestamp-based ID is generated

## Response Format

### Success Response (200)

```json
{
    "message": "Chunk processing completed",
    "chunk_id": "chunk_1701234567",
    "session_id": "chunk_chunk_1701234567_1701234567",
    "output_file": "/path/to/workspace/chunk_outputs/chunk_output_chunk_1701234567_1701234567.json",
    "output_filename": "chunk_output_chunk_1701234567_1701234567.json",
    "workspace_location": "/path/to/workspace/chunk_outputs",
    "success": true,
    "processing_summary": {
        "total_duration": 12.34,
        "total_duration_formatted": "12.3s",
        "graph_nodes": 15,
        "graph_edges": 8,
        "errors_count": 0
    }
}
```

### Error Response (500)

```json
{
    "message": "Chunk processing completed",
    "chunk_id": "chunk_1701234567",
    "session_id": "chunk_chunk_1701234567_1701234567",
    "output_file": "/path/to/workspace/chunk_outputs/chunk_output_chunk_1701234567_1701234567.json",
    "success": false,
    "processing_summary": {
        "total_duration": 5.67,
        "errors_count": 1
    },
    "errors": [
        "Phase 1 failed: Some error message"
    ]
}
```

## Processing Pipeline

The endpoint processes the chunk through the complete NodeRAG pipeline:

### Phase 1: Graph Decomposition
- Converts the text chunk into structured graph nodes
- Extracts entities, relationships, and text segments
- Creates semantic chunks if the text is large

### Phase 2: Graph Augmentation  
- Identifies important entities using graph metrics
- Generates attribute nodes for key entities
- Detects communities in the graph
- Creates high-level summary nodes
- Generates overview nodes

### Phase 3: Embedding Generation & Vector Storage
- Generates embeddings for all graph nodes
- Builds HNSW index for vector similarity search
- Prepares vector storage (but doesn't persist to database)

### Phase 4: Graph Data Extraction
- Extracts all graph nodes organized by type
- Collects graph statistics and metadata
- Prepares structured output data

## Output File Structure

The processing results are saved to a JSON file in the workspace with the following structure:

```json
{
    "input": {
        "chunk_id": "chunk_1701234567",
        "chunk_text": "Your input text...",
        "chatgpt_model": "gpt-4",
        "text_length": 256,
        "processing_timestamp": 1701234567.123,
        "session_id": "chunk_chunk_1701234567_1701234567"
    },
    "processing_results": {
        "phase1_decomposition": {
            "success": true,
            "chunks_processed": 1,
            "chunks_failed": 0,
            "processing_time": 3.45
        },
        "phase2_augmentation": {
            "success": true,
            "important_entities": 5,
            "attribute_nodes": 3,
            "communities_detected": 2,
            "high_level_nodes": 2,
            "overview_nodes": 1
        },
        "phase3_embedding": {
            "success": true,
            "embeddings_generated": 15,
            "hnsw_indexed": 15
        },
        "graph_stats": {
            "total_nodes": 15,
            "total_edges": 8,
            "node_type_counts": {
                "T": 4,
                "N": 5,
                "R": 3,
                "A": 2,
                "H": 1
            }
        },
        "graph_data": {
            "total_nodes": 15,
            "total_edges": 8,
            "nodes_by_type": {
                "T": [
                    {
                        "id": "T_abc123",
                        "content": "Text content...",
                        "metadata": {...}
                    }
                ],
                "N": [...],
                "R": [...],
                "A": [...],
                "H": [...]
            }
        }
    },
    "performance_metrics": {
        "session_id": "chunk_chunk_1701234567_1701234567",
        "total_duration": 12.34,
        "total_duration_formatted": "12.3s",
        "processing_rate": 20.73,
        "steps_summary": {
            "total_steps": 5,
            "completed_steps": 5,
            "failed_steps": 0,
            "step_details": [...]
        },
        "performance_report_path": "/path/to/performance/report.json",
        "success": true
    },
    "errors": []
}
```

## Usage Examples

### Basic Usage

```bash
curl -X POST http://localhost:8000/api/v1/process-chunk \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_text": "This is a sample text about artificial intelligence and machine learning concepts.",
    "chatgpt_model": "gpt-4"
  }'
```

### Python Example

```python
import requests
import json

def process_chunk(text, model="gpt-4", chunk_id=None):
    url = "http://localhost:8000/api/v1/process-chunk"
    
    data = {
        "chunk_text": text,
        "chatgpt_model": model
    }
    
    if chunk_id:
        data["chunk_id"] = chunk_id
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Processing completed!")
        print(f"📄 Output file: {result['output_filename']}")
        print(f"⏱️ Duration: {result['processing_summary']['total_duration_formatted']}")
        print(f"📊 Nodes: {result['processing_summary']['graph_nodes']}")
        return result
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

# Usage
text = "Your text content here..."
result = process_chunk(text)
```

## Testing the Endpoint

Use the provided test script to verify the endpoint is working:

```bash
python test_chunk_endpoint.py
```

This script will:
1. Check if the API service is running
2. Send a test chunk for processing
3. Display the results and output file location
4. Show performance metrics

## Output File Location

All output files are saved to: `workspace/chunk_outputs/`

Each file is named with the pattern: `chunk_output_{chunk_id}_{timestamp}.json`

## Performance Monitoring

The endpoint includes comprehensive performance logging:

- **Session tracking**: Each request gets a unique session ID
- **Step-by-step timing**: Detailed timing for each processing phase
- **Memory usage**: Tracks memory consumption during processing  
- **Error tracking**: Captures and logs any processing errors
- **Performance reports**: Generates detailed performance reports

## Node Types in Output

The graph data includes different types of nodes:

- **T (Text)**: Original text segments and semantic chunks
- **N (Named Entity)**: Entities like companies, people, products
- **R (Relationship)**: Relationships between entities
- **A (Attribute)**: Attributes and properties of entities  
- **H (High-level)**: High-level summaries and overviews

## Error Handling

The endpoint handles errors gracefully:

- Processing errors are captured and included in the output file
- Performance logging continues even on failures
- Detailed error messages help with debugging
- The API returns appropriate HTTP status codes

## Performance Considerations

- Processing time depends on text length and complexity
- Typical processing time: 10-30 seconds for medium-sized chunks
- Memory usage scales with the number of extracted entities
- HNSW indexing time increases with node count

## Integration Notes

- No database storage - purely file-based output
- Stateless processing - each request is independent
- Compatible with existing NodeRAG infrastructure
- Uses the same pipeline as `/api/v1/process-document`
- Performance logging integrates with existing monitoring
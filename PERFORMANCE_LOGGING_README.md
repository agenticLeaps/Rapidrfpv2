# Performance Logging System

This document describes the comprehensive performance logging system implemented for tracking file upload and processing times in the RapidRFP RAG system.

## Overview

The performance logging system provides detailed timing analysis for every step of the document processing pipeline, from file upload to graph storage. It generates comprehensive reports that help analyze bottlenecks and optimize processing performance.

## Key Features

### 📊 Comprehensive Step Tracking
- **Hierarchical Logging**: Tracks main phases and sub-steps with nested timing
- **Metadata Collection**: Captures relevant data for each processing step
- **Real-time Monitoring**: Monitor active processing sessions
- **Error Tracking**: Logs failures and error messages at each step

### ⏱️ Detailed Timing Analysis
- **Step Duration**: Precise timing for each processing step
- **Processing Rate**: Calculates bytes/second processing rate
- **Progress Tracking**: Real-time progress updates during processing
- **Performance Trends**: Historical analysis of processing performance

### 📁 Multiple Output Formats
- **JSON Logs**: Machine-readable logs for automated analysis
- **Text Reports**: Human-readable performance reports
- **API Endpoints**: Real-time access to performance data
- **Session Tracking**: Isolated logging per session/user

## Processing Steps Tracked

### 1. Document Loading
- File reading and validation
- Chunking and tokenization
- Parsing method selection (LlamaParse vs fallback)

### 2. Graph Decomposition
- **Chunk Processing**: Individual chunk processing times
- **Text Node Creation**: Creating text nodes (T)
- **LLM Extraction**: Entity, relationship, and semantic extraction
- **Semantic Nodes**: Creating semantic unit nodes (S)
- **Entity Nodes**: Creating entity nodes (N) with deduplication
- **Relationship Nodes**: Creating relationship nodes (R)

### 3. Graph Augmentation
- **Important Entity Identification**: Finding high-importance entities
- **Attribute Node Generation**: Creating attribute nodes (A)
- **Community Detection**: Graph clustering analysis
- **High-Level Summaries**: Creating high-level nodes (H)
- **Overview Generation**: Creating overview nodes (O)

### 4. Embedding Generation & Storage
- **HNSW Initialization**: Vector search index setup
- **Node Collection**: Gathering nodes for embedding
- **Batch Processing**: Embedding generation in batches
- **Vector Indexing**: Adding embeddings to HNSW index
- **Index Storage**: Saving vector index to disk

### 5. Graph Storage
- Final graph persistence to disk

## Usage

### Basic Integration

The performance logger is automatically integrated into the indexing pipeline:

```python
# The indexing pipeline automatically starts performance logging
result = indexing_pipeline.index_document(file_path, session_id)

# Performance data is included in the result
if result.get('performance_report_available'):
    session_id = result['session_id']
    report = performance_logger.get_session_report(session_id)
```

### Manual Step Tracking

For custom processing steps:

```python
from src.utils.performance_logger import performance_logger

# Start a session
session_id = performance_logger.start_session("my_session", "file.txt", 1024)

# Track a processing step
with performance_logger.step("Custom Processing", custom_param="value"):
    # Your processing code here
    process_data()
    
    # Add metadata during processing
    performance_logger.add_step_metadata(
        items_processed=100,
        success_rate=95.5
    )

# End the session
completed_session = performance_logger.end_session('completed')
```

### Nested Steps

Track sub-steps within main processing phases:

```python
with performance_logger.step("Main Processing"):
    with performance_logger.step("Sub-step 1"):
        # Sub-processing code
        pass
    
    with performance_logger.step("Sub-step 2"):
        # More sub-processing code
        pass
```

## API Endpoints

### Get Performance Report
```http
GET /api/performance/report/<session_id>
```
Returns detailed performance analysis for a session.

### Export Report to File
```http
POST /api/performance/export/<session_id>
Content-Type: application/json

{
    "output_path": "/optional/custom/path.txt"
}
```

### Current Session Status
```http
GET /api/performance/current
```
Returns real-time status of the currently active processing session.

### Stream Performance Logs
```http
GET /api/performance/logs/stream?lines=100
```
Returns recent performance log entries for monitoring.

## Example Output

### Session Summary
```
🎉 Completed processing session: session_12345 (2.45s)
📊 Session Summary for session_12345:
   • Total Duration: 2.45s
   • File Size: 15,420 bytes
   • Processing Rate: 6,294.3 bytes/sec
   • Steps: 5/5 successful
   • Step Breakdown:
     ✅ Document Loading: 245ms (10.0%)
     ✅ Graph Decomposition: 1.20s (49.0%)
     ✅ Graph Augmentation: 685ms (28.0%)
     ✅ Embedding Generation & Storage: 285ms (11.6%)
     ✅ Graph Storage: 35ms (1.4%)
```

### Detailed Text Report
```
================================================================================
PERFORMANCE ANALYSIS REPORT
================================================================================

Session ID: session_12345
File: document.pdf
File Size: 15.4 KB (15,420 bytes)
Status: completed
Start Time: 2024-01-15T10:30:45.123
End Time: 2024-01-15T10:30:47.573
Total Duration: 2.45s

SUMMARY
----------------------------------------
Processing Rate: 6294.3 bytes/sec
Total Steps: 5
Successful Steps: 5
Failed Steps: 0

STEP BREAKDOWN
----------------------------------------
✓ Document Loading          245ms
  └─ Sub-steps: 3
✓ Graph Decomposition       1.20s
  └─ Sub-steps: 15
✓ Graph Augmentation        685ms
  └─ Sub-steps: 8
✓ Embedding Generation      285ms
  └─ Sub-steps: 6
✓ Graph Storage             35ms
```

## Log Files

### Performance Logs Location
- **Main Directory**: `data/performance_logs/`
- **JSON Log**: `performance.log` - Real-time event logging
- **Session Archive**: `performance_sessions.jsonl` - Complete session data
- **Text Reports**: `performance_report_<session_id>_<timestamp>.txt`

### Log File Structure

**performance.log** (JSON Lines format):
```json
{"timestamp": "2024-01-15T10:30:45.123", "event_type": "SESSION_START", "session_id": "session_12345", "file_name": "document.pdf", "file_size": 15420}
{"timestamp": "2024-01-15T10:30:45.125", "event_type": "STEP_START", "session_id": "session_12345", "step_name": "Document Loading", "metadata": {}}
{"timestamp": "2024-01-15T10:30:45.370", "event_type": "STEP_END", "session_id": "session_12345", "step_name": "Document Loading", "duration": 0.245, "status": "completed"}
```

## Testing

Run the performance logging demo:

```bash
python test_performance_logging.py
```

This will simulate a complete file processing workflow and generate sample performance reports.

## Configuration

Performance logging settings in `Config`:

```python
# Log directory (default: data/performance_logs)
PERFORMANCE_LOG_DIR = "data/performance_logs"

# Enable/disable performance logging
ENABLE_PERFORMANCE_LOGGING = True

# Log level for performance events
PERFORMANCE_LOG_LEVEL = "INFO"
```

## Benefits

### 🔍 Performance Analysis
- **Bottleneck Identification**: Quickly identify slow processing steps
- **Optimization Targets**: Focus optimization efforts on high-impact areas
- **Trend Analysis**: Track performance improvements over time
- **Resource Planning**: Estimate processing times for capacity planning

### 🐛 Debugging and Monitoring
- **Error Tracking**: Detailed error logging with context
- **Progress Monitoring**: Real-time processing status
- **Session Isolation**: Track performance per user/session
- **Historical Analysis**: Compare processing performance across sessions

### 📈 Business Intelligence
- **Usage Patterns**: Understand document processing patterns
- **Performance Metrics**: Generate performance KPIs
- **Capacity Planning**: Plan infrastructure based on processing demands
- **Cost Analysis**: Track processing costs and efficiency

## Integration with Monitoring Tools

The performance logs can be integrated with monitoring and observability tools:

- **Prometheus**: Export metrics for monitoring
- **Grafana**: Create performance dashboards
- **ELK Stack**: Centralized log analysis
- **Custom Analytics**: Parse JSON logs for custom analysis

## Future Enhancements

Planned improvements to the performance logging system:

1. **Real-time WebSocket Streaming**: Live progress updates
2. **Performance Alerting**: Automatic alerts for slow processing
3. **Comparison Analysis**: Compare performance across different files
4. **Resource Usage Tracking**: Monitor CPU, memory, and disk usage
5. **Machine Learning Integration**: Predict processing times based on file characteristics
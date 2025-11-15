# RapidRFP RAG API Documentation

Complete API reference for the RapidRFP RAG system. The API provides endpoints for document indexing, graph operations, search, and analysis.

## Base URL
```
http://localhost:5001
```

## API Endpoints Overview

| Category | Endpoint | Method | Description |
|----------|----------|---------|-------------|
| **System** | `/health` | GET | Health check |
| **System** | `/api/config` | GET | Get configuration |
| **Document** | `/api/index/document` | POST | Index a document (file path) |
| **Document** | `/api/upload/document` | POST | Upload and index a document |
| **Document** | `/api/estimate/cost` | POST | Estimate processing cost |
| **Graph** | `/api/graph/stats` | GET | Get graph statistics |
| **Graph** | `/api/graph/nodes/<type>` | GET | Get nodes by type |
| **Graph** | `/api/graph/node/<id>` | GET | Get specific node |
| **Graph** | `/api/graph/save` | POST | Save graph to disk |
| **Graph** | `/api/graph/load` | POST | Load graph from disk |
| **Graph** | `/api/graph/communities` | GET | Get communities |
| **Graph** | `/api/graph/important-entities` | GET | Get important entities |
| **Search** | `/api/search/entities` | POST | Search entities |

## Quick Start

### 1. Check System Health
```bash
curl http://localhost:5001/health
```

### 2. Upload and Index Your First Document
```bash
# Method 1: Upload file directly
curl -X POST http://localhost:5001/api/upload/document \
  -F "file=@/path/to/your/document.pdf"

# Method 2: Index existing file by path  
curl -X POST http://localhost:5001/api/index/document \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/your/document.pdf"}'
```

### 3. View Graph Statistics
```bash
curl http://localhost:5001/api/graph/stats
```

## Detailed Documentation

- [System Endpoints](./system.md) - Health check and configuration
- [Document Processing](./document.md) - Indexing and cost estimation  
- [Graph Operations](./graph.md) - Graph management and analysis
- [Search & Query](./search.md) - Entity search and retrieval
- [Error Handling](./errors.md) - Error codes and troubleshooting
- [Examples](./examples.md) - Complete usage examples

## Response Format

All API responses follow this structure:

### Success Response
```json
{
  "success": true,
  "data": {...},
  "message": "Optional success message"
}
```

### Error Response
```json
{
  "error": "Error description",
  "details": "Additional error details (optional)"
}
```

## Node Types Reference

| Type | Code | Description | Example |
|------|------|-------------|---------|
| Text | `T` | Original document chunks | Raw text segments |
| Semantic | `S` | Independent concepts | "John discovered the artifact" |
| Entity | `N` | Named entities | "John Smith", "New York" |
| Relationship | `R` | Entity connections | "John works_for Company" |
| Attribute | `A` | Entity summaries | Detailed entity descriptions |
| High-Level | `H` | Community summaries | Topic overviews |
| Overview | `O` | Keyword titles | "Project Management Topics" |

## Authentication

Currently, the API does not require authentication. In production, consider implementing:
- API keys
- JWT tokens
- Rate limiting
- IP whitelisting

## Rate Limiting

No rate limiting is currently implemented. For production use, consider:
- Request per minute limits
- Concurrent request limits
- Resource-based throttling
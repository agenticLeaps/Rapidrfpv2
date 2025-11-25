# NodeRAG Integration Guide

## Overview

This document describes the integration of NodeRAG as a standalone microservice with the main RapidRFP AI server. NodeRAG provides advanced graph-based RAG capabilities as an alternative to the existing naive RAG system.

## Architecture

```
Main AI Server (RapidRFPAI)          NodeRAG Service (Rapidrfpv2)
├── v3/upload endpoint               ├── Standalone Flask API service
├── LlamaParse processing            ├── Independent NeonDB storage
├── API calls to NodeRAG             ├── Graph processing pipeline
├── Webhook receivers                ├── Real-time status webhooks
└── Unified search interface         └── Search API endpoints
```

## Features

### NodeRAG v2 Processing Pipeline
1. **Phase 1: Graph Decomposition** - Extract T, S, N, R nodes
2. **Phase 2: Graph Augmentation** - Generate A, H, O nodes  
3. **Phase 3: Embedding Generation** - Create embeddings for all nodes
4. **Storage** - Store graph and embeddings in NeonDB

### API Endpoints

#### NodeRAG Service (Port 5001)
- `POST /api/v1/process-document` - Process document chunks
- `POST /api/v1/search` - Search NodeRAG data
- `GET /api/v1/status/{file_id}` - Get processing status
- `DELETE /api/v1/delete-file` - Delete file data
- `GET /api/v1/health` - Health check

#### Main AI Server (Port 5000)
- `POST /api/v3/upload` - Upload with `ragversion=v2` parameter
- `POST /api/v3/search` - Unified search across v1 and v2
- `POST /api/webhook/noderag` - Receive NodeRAG status updates

## Setup Instructions

### 1. Environment Configuration

Create `.env` file in Rapidrfpv2 directory:
```bash
# Database
NEON_DATABASE_URL=postgresql://username:password@host/database

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Optional: LlamaParse
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# API Configuration
API_HOST=0.0.0.0
API_PORT=5001
API_DEBUG=false
```

Main server environment variables:
```bash
# NodeRAG Service URL
NODERAG_SERVICE_URL=http://localhost:5001

# Main Server URL (for callbacks)
MAIN_SERVER_URL=http://localhost:5000
```

### 2. Database Setup

The NodeRAG service will automatically create required tables:
- `noderag_embeddings` - Store node embeddings
- `noderag_graphs` - Store graph data

### 3. Start Services

#### Start NodeRAG Service:
```bash
cd Rapidrfpv2
pip install -r requirements.txt
python start_noderag_service.py
```

#### Start Main AI Server:
```bash
cd RapidRFPAI
python app.py
```

## Usage

### Document Processing

#### V1 (Naive RAG):
```bash
curl -X POST http://localhost:5000/api/v3/upload \
  -F "file=@document.pdf" \
  -F "orgId=org123" \
  -F "fileId=file456" \
  -F "userId=user789" \
  -F "ragversion=v1"
```

#### V2 (NodeRAG):
```bash
curl -X POST http://localhost:5000/api/v3/upload \
  -F "file=@document.pdf" \
  -F "orgId=org123" \
  -F "fileId=file456" \
  -F "userId=user789" \
  -F "ragversion=v2"
```

### Search

#### Unified Search (Both v1 and v2):
```json
POST /api/v3/search
{
  "query": "What are the key findings?",
  "orgId": "org123",
  "top_k": 10,
  "ragversion": "both"
}
```

#### V1 Only:
```json
POST /api/v3/search
{
  "query": "What are the key findings?",
  "orgId": "org123", 
  "top_k": 10,
  "ragversion": "v1"
}
```

#### V2 Only:
```json
POST /api/v3/search
{
  "query": "What are the key findings?",
  "orgId": "org123",
  "top_k": 10, 
  "ragversion": "v2"
}
```

## Processing Flow

### V2 (NodeRAG) Processing Flow:
1. File uploaded to main server with `ragversion=v2`
2. Main server extracts chunks using LlamaParse
3. Chunks sent to NodeRAG service via API
4. NodeRAG processes through 3-phase pipeline:
   - Phase 1: Graph decomposition (T, S, N, R nodes)
   - Phase 2: Graph augmentation (A, H, O nodes) 
   - Phase 3: Embedding generation
5. NodeRAG stores graph and embeddings in NeonDB
6. Real-time status updates sent via webhooks
7. Processing completion notification

### Search Flow:
1. Query sent to unified search endpoint
2. If `ragversion="both"`, searches both v1 and v2 data
3. V1 results from existing naive RAG system
4. V2 results from NodeRAG service API
5. Combined and ranked results returned

## Node Types in NodeRAG

- **T (Text)**: Original text chunks
- **S (Semantic)**: Semantic units extracted from text
- **N (Entity)**: Named entities with deduplication
- **R (Relationship)**: Relationships between entities
- **A (Attribute)**: Attributes for important entities
- **H (High-Level)**: High-level summaries of communities
- **O (Overview)**: Overview titles for communities

## Monitoring

### Status Updates
The main server receives real-time updates via webhooks:
- `processing` - Initial processing started
- `phase1_started` - Graph decomposition phase
- `phase2_started` - Graph augmentation phase  
- `phase3_started` - Embedding generation phase
- `completed` - Processing finished successfully
- `failed` - Processing failed with error

### Health Checks
```bash
# Check NodeRAG service health
curl http://localhost:5001/api/v1/health

# Check processing status
curl http://localhost:5001/api/v1/status/{file_id}
```

## Troubleshooting

### Common Issues

1. **NodeRAG service unavailable**
   - Ensure service is running on correct port
   - Check NODERAG_SERVICE_URL environment variable

2. **Database connection errors**
   - Verify NEON_DATABASE_URL is correct
   - Ensure database is accessible

3. **Webhook delivery failures** 
   - Check MAIN_SERVER_URL environment variable
   - Ensure main server webhook endpoint is accessible

4. **Processing timeouts**
   - Large files may take several minutes to process
   - Check NodeRAG service logs for detailed error messages

### Logs
- NodeRAG service logs: Console output when running `start_noderag_service.py`
- Main server logs: Check Flask app logs
- Database logs: Check NeonDB logs if needed

## Performance Considerations

- NodeRAG processing is more computationally intensive than naive RAG
- Processing time scales with document complexity and length
- Graph operations require more memory and CPU resources
- Consider running NodeRAG service on dedicated infrastructure for production

## Future Enhancements

- [ ] Implement caching for frequently accessed graphs
- [ ] Add batch processing for multiple documents
- [ ] Implement graph visualization endpoints
- [ ] Add performance metrics and monitoring
- [ ] Support for incremental graph updates
#!/bin/bash
# Start NodeRAG API Server

echo "🚀 Starting NodeRAG API Server..."
export REDIS_URL=redis://localhost:6379/0
export NODERAG_SERVICE_TYPE=api
export PYTHONPATH=.
python start_noderag_service.py
#!/bin/bash
# Start NodeRAG Celery Worker

echo "⚙️ Starting NodeRAG Celery Worker..."
export REDIS_URL=redis://localhost:6379/0
export NODERAG_SERVICE_TYPE=worker
export PYTHONPATH=.
export CELERY_OPTIMIZATION=1
python start_noderag_service.py
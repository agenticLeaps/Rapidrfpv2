# NodeRAG Deployment Guide for Render.com

## 🎯 Overview
This guide shows how to deploy the NodeRAG queue system on Render.com with memory-efficient processing for 2GB RAM environments.

---

## 🏗️ Architecture on Render

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Service   │    │ Worker Service  │    │ Worker Service  │    │  Redis Service  │
│   (noderag-api) │    │(noderag-worker) │    │(noderag-sched)  │    │ (noderag-redis) │
│                 │    │                 │    │                 │    │                 │
│ Flask API       │───▶│ Celery Worker   │◄───│ Celery Beat     │◄───│ Redis Queue     │
│ Queue Manager   │    │ Doc Processing  │    │ Cleanup Tasks   │    │ Task Storage    │
│ Status Monitor  │    │ Memory Limited  │    │ Periodic Jobs   │    │ Result Backend  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Service Types:
1. **Web Service** (`noderag-api`): Handles API requests and queues tasks
2. **Worker Service** (`noderag-worker`): Processes documents with memory limits
3. **Worker Service** (`noderag-scheduler`): Runs cleanup and maintenance tasks
4. **Redis Service** (`noderag-redis`): Message queue and result storage

---

## 🚀 Quick Deployment

### Method 1: Using render.yaml (Recommended)

1. **Push to GitHub**:
```bash
git add .
git commit -m "Add NodeRAG queue system for Render deployment"
git push origin main
```

2. **Connect to Render**:
- Go to [Render Dashboard](https://dashboard.render.com)
- Click "New" → "Blueprint"
- Connect your GitHub repository
- Select the repository containing `render.yaml`
- Click "Apply"

3. **Render will automatically create**:
- ✅ **noderag-api** (Web Service)
- ✅ **noderag-worker** (Worker Service) 
- ✅ **noderag-scheduler** (Worker Service)
- ✅ **noderag-redis** (Redis Service)

### Method 2: Manual Service Creation

#### Step 1: Create Redis Service
```yaml
Name: noderag-redis
Type: Redis
Plan: Starter (25MB)
```

#### Step 2: Create Web Service
```yaml
Name: noderag-api
Type: Web Service
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python start_noderag_service.py
Environment Variables:
  - NODERAG_SERVICE_TYPE=api
  - REDIS_URL=[Connect to noderag-redis]
  - PYTHONPATH=.
  - CELERY_OPTIMIZATION=1
```

#### Step 3: Create Worker Service
```yaml
Name: noderag-worker
Type: Worker
Environment: Python  
Build Command: pip install -r requirements.txt
Start Command: python start_noderag_service.py
Environment Variables:
  - NODERAG_SERVICE_TYPE=worker
  - REDIS_URL=[Connect to noderag-redis]
  - PYTHONPATH=.
  - CELERY_OPTIMIZATION=1
```

#### Step 4: Create Scheduler Service (Optional)
```yaml
Name: noderag-scheduler
Type: Worker
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python start_noderag_service.py
Environment Variables:
  - NODERAG_SERVICE_TYPE=scheduler
  - REDIS_URL=[Connect to noderag-redis]
  - PYTHONPATH=.
```

---

## 🔧 Configuration Details

### Main Startup Script
The `start_noderag_service.py` script automatically detects the service type and starts the appropriate service:

```python
# Environment variable determines service type
NODERAG_SERVICE_TYPE = os.getenv('NODERAG_SERVICE_TYPE', 'api')

# Service types:
# - 'api': Flask API server with gunicorn
# - 'worker': Celery worker for document processing  
# - 'scheduler': Celery beat scheduler for cleanup
```

### Memory Optimization
The startup script automatically detects available RAM and sets appropriate limits:

```python
# Memory limits based on available RAM:
if total_ram_mb < 2048:
    max_memory_kb = 1200000  # 1.2GB (conservative)
elif total_ram_mb < 4096:
    max_memory_kb = 1500000  # 1.5GB (standard)
else:
    max_memory_kb = 2000000  # 2GB (high RAM)
```

### Environment Variables

#### Required for All Services:
```bash
REDIS_URL=redis://...              # Automatically set by Render
PYTHONPATH=.                       # Python module path
PYTHONUNBUFFERED=1                 # Unbuffered Python output
```

#### Service-Specific:
```bash
# API Service
NODERAG_SERVICE_TYPE=api
PORT=8000                          # Set automatically by Render

# Worker Service  
NODERAG_SERVICE_TYPE=worker
CELERY_OPTIMIZATION=1

# Scheduler Service
NODERAG_SERVICE_TYPE=scheduler
```

#### Optional:
```bash
# Database credentials (add these in Render dashboard)
DATABASE_URL=postgresql://...      # Your NeonDB connection string
OPENAI_API_KEY=sk-...             # Your OpenAI API key
GOOGLE_API_KEY=...                # Your Google Gemini API key

# Admin access
ADMIN_KEY=your-secure-key         # For queue purge operations
```

---

## 📊 Service Plans & Scaling

### Recommended Plans:

#### For Standard Usage (20+ concurrent requests):
```yaml
# API Service
plan: standard  # 512MB RAM, sufficient for API handling
scaling:
  minInstances: 1
  maxInstances: 3

# Worker Service  
plan: standard  # 512MB RAM, memory-optimized processing
scaling:
  minInstances: 1
  maxInstances: 2

# Scheduler Service
plan: starter   # 128MB RAM, minimal resource usage
scaling:
  minInstances: 1
  maxInstances: 1

# Redis Service
plan: starter   # 25MB storage, sufficient for queue data
```

#### For High Volume (50+ concurrent requests):
```yaml
# API Service
plan: pro       # 1GB RAM
scaling:
  minInstances: 2
  maxInstances: 5

# Worker Service
plan: pro       # 1GB RAM, can handle larger documents
scaling:
  minInstances: 2
  maxInstances: 4

# Redis Service  
plan: pro       # 100MB storage
```

---

## 🔍 Monitoring & Debugging

### Service Status
Check service health via API:
```bash
# API Health Check
curl https://your-app.onrender.com/api/v1/health

# Queue Statistics
curl https://your-app.onrender.com/api/v1/queue/stats
```

### Render Dashboard Monitoring
1. **Logs**: View real-time logs for each service
2. **Metrics**: Monitor CPU, RAM, and network usage
3. **Events**: Track deployments and restarts
4. **Shell**: Access service shell for debugging

### Common Log Messages
```bash
# Successful Startup
🚀 NodeRAG Service Manager starting...
✅ Redis connected successfully
🌐 Starting NodeRAG API Service

# Memory Optimization
📊 System RAM: 512MB
⚠️ Low RAM environment - setting conservative memory limit: 1.2GB

# Processing Status
📥 Submitting document processing task: org_id=..., file_id=...
⚙️ Starting NodeRAG Celery Worker
✅ Task completed successfully
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Service Won't Start"
```bash
# Check logs for:
❌ REDIS_URL environment variable not set
❌ Redis connection timeout after 60 seconds

# Solution: Ensure Redis service is running and connected
```

#### 2. "Worker Out of Memory"
```bash
# Check logs for:
⚠️ Worker consuming too much memory
💀 Worker killed due to memory limit

# Solution: 
# - Upgrade to higher plan (standard → pro)
# - Reduce max_memory_per_child in startup script
# - Process smaller document batches
```

#### 3. "Tasks Stay in PENDING State"
```bash
# Check logs for:
🔍 No workers available
❌ Worker failed to start

# Solution:
# - Check worker service is running
# - Verify Redis connection
# - Restart worker service
```

#### 4. "API Timeouts"
```bash
# Check logs for:
⏰ Task timed out
🕐 Request timeout after 300s

# Solution:
# - Increase gunicorn timeout in startup script
# - Process smaller document chunks
# - Add more worker instances
```

### Debug Commands
```bash
# Access service shell in Render dashboard
python -c "import redis; print(redis.Redis.from_url('$REDIS_URL').ping())"

# Check Python packages
pip list | grep -E "(celery|redis|gunicorn)"

# Monitor system resources
python -c "import psutil; print(f'RAM: {psutil.virtual_memory()}')"
```

---

## 🚀 Testing the Deployment

### 1. Health Check
```bash
curl https://your-app.onrender.com/api/v1/health
# Expected: {"status": "healthy", "service": "noderag", ...}
```

### 2. Queue Status
```bash
curl https://your-app.onrender.com/api/v1/queue/stats
# Expected: {"active_tasks": 0, "workers_available": 1, ...}
```

### 3. Document Processing
```bash
curl -X POST https://your-app.onrender.com/api/v1/process-document \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "test-org",
    "file_id": "test-file-1",
    "user_id": "test-user", 
    "chunks": [{"content": "Test document content for queue processing"}],
    "use_queue": true
  }'

# Expected: {"message": "Document processing queued", "task_id": "...", ...}
```

### 4. Task Status
```bash
curl https://your-app.onrender.com/api/v1/status/test-file-1
# Expected: {"status": "processing", "phase": "embedding_generation", ...}
```

### 5. Load Testing (Optional)
```bash
# Test multiple concurrent requests
for i in {1..20}; do
  curl -X POST https://your-app.onrender.com/api/v1/process-document \
    -H "Content-Type: application/json" \
    -d "{\"org_id\":\"test-org\",\"file_id\":\"test-file-$i\",\"user_id\":\"test-user\",\"chunks\":[{\"content\":\"Test document $i\"}]}" &
done
wait

# Monitor queue: should handle all 20 requests without memory issues
curl https://your-app.onrender.com/api/v1/queue/stats
```

---

## 📈 Performance Optimization

### 1. Scaling Strategy
```yaml
# Start small, scale as needed
Initial: 1 API + 1 Worker + 1 Scheduler + Redis
Growth:  2 API + 2 Worker + 1 Scheduler + Redis  
High:    3 API + 4 Worker + 1 Scheduler + Redis Pro
```

### 2. Memory Management
- Monitor RAM usage in Render dashboard
- Adjust memory limits in `start_noderag_service.py` if needed
- Use smaller document chunks for processing
- Enable automatic worker restarts after each task

### 3. Queue Optimization
```python
# Tune these in src/queue/celery_config.py:
worker_max_tasks_per_child=1      # Restart frequency
worker_prefetch_multiplier=1      # Queue prefetch
task_time_limit=1800              # Task timeout (30min)
worker_max_memory_per_child=1500000  # Memory limit
```

### 4. Database Optimization
- Use connection pooling for NeonDB
- Store embeddings in PostgreSQL (not local files)
- Implement cleanup for old completed tasks

---

## 🎯 Summary

The Render deployment provides:

✅ **Memory Safety**: Handles 20+ concurrent requests on 512MB-1GB RAM  
✅ **Auto-Scaling**: Scales workers based on queue load  
✅ **Reliability**: Automatic restarts and health monitoring  
✅ **Production Ready**: Proper WSGI server (gunicorn) and process management  
✅ **Cost Effective**: Starts with $7/month for starter plan services  
✅ **Easy Monitoring**: Built-in logs, metrics, and debugging tools  

**Total Monthly Cost (Estimated)**:
- Standard Plan: ~$25-50/month (API + Worker + Redis)
- Pro Plan: ~$75-150/month (Higher performance)

The queue system ensures your NodeRAG deployment can handle high-concurrency scenarios without running into memory issues on Render's infrastructure!
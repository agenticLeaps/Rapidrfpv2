# Render Deployment Optimizations

## Problem Solved
Fixed embedding process failing at batch 91/110 on Render due to memory/timeout constraints.

## Key Improvements

### 1. Environment-Aware Configuration
- **Adaptive Batch Sizing**: Automatically reduces batch size on Render (6 vs 16 locally)
- **Progressive Delays**: Implements smart rate limiting that increases with progress
- **Environment Detection**: Auto-detects Render/Cloud environments via `IS_RENDER` flag

### 2. Enhanced Error Handling
- **Exponential Backoff**: Handles OpenAI rate limits with increasing delays
- **Specific Error Types**: Different retry strategies for timeouts, rate limits, connection errors
- **Comprehensive Logging**: Detailed error reporting for debugging

### 3. Memory Management
- **Periodic Garbage Collection**: Cleans memory every 5-10 batches
- **Memory Monitoring**: Tracks usage on cloud platforms (requires `psutil`)
- **Smaller Batches for Large Datasets**: Auto-reduces batch size for 1000+ items

### 4. Checkpoint System
- **Resume Capability**: Can resume from last successful batch after failures
- **Progress Tracking**: Saves every 10-20 batches (configurable)
- **Auto-cleanup**: Removes checkpoint files on successful completion

### 5. Production Settings

#### For Render (set these environment variables):
```bash
IS_RENDER=true
RENDER_BATCH_SIZE=6
RENDER_BASE_DELAY=1.0
MAX_RETRIES=5
CHECKPOINT_INTERVAL=10
GC_EVERY_N_BATCHES=5
```

#### For Local Development:
```bash
DEFAULT_BATCH_SIZE=16
BASE_DELAY=0.1
MAX_RETRIES=3
CHECKPOINT_INTERVAL=20
```

## Usage

The system automatically detects the environment and applies appropriate settings. No code changes needed - just set the environment variables.

### Manual Override
You can force specific batch sizes:
```python
# In your code
batch_size = Config.get_adaptive_batch_size(total_items=1500)
delay = Config.get_adaptive_delay(batch_num=50, total_batches=100)
```

## Expected Performance

| Environment | Batch Size | Delay Range | Memory Usage |
|-------------|------------|-------------|--------------|
| Local       | 16         | 0.1s        | Normal       |
| Cloud       | 8-10       | 0.1-2.4s    | Monitored    |
| Render      | 4-6        | 0.5-3.0s    | Aggressive GC|

## Monitoring

Look for these log messages:
- `📊 Environment: Render/Cloud/Local`
- `📏 Batch size: X (adaptive)`
- `💾 Checkpoint saved after batch X`
- `🧹 Memory cleanup after batch X`
- `⚠️ High memory usage: X%`

## Troubleshooting

### Still Failing on Render?
1. Reduce `RENDER_BATCH_SIZE` to 4
2. Increase `RENDER_BASE_DELAY` to 2.0
3. Enable more frequent checkpoints (`CHECKPOINT_INTERVAL=5`)
4. Check Render logs for specific error patterns

### Memory Issues?
1. Reduce `GC_EVERY_N_BATCHES` to 3
2. Set `MEMORY_CHECK_INTERVAL=1` for constant monitoring
3. Consider reducing overall dataset size

### Rate Limiting?
1. Increase `MAX_DELAY` to 10.0
2. Set `MAX_RETRIES=10` for more patience
3. Check OpenAI API usage dashboard
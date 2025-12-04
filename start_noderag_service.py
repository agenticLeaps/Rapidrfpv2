#!/usr/bin/env python3
"""
NodeRAG Service Startup Script for Render Deployment
This is the main entry point that determines which service to start
"""

import os
import sys
import time
import subprocess
import logging
import signal
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NodeRAGServiceManager:
    """Main service manager for NodeRAG deployment"""
    
    def __init__(self):
        self.service_type = os.getenv('NODERAG_SERVICE_TYPE', 'api')
        self.port = os.getenv('PORT', '8000')
        self.redis_url = os.getenv('REDIS_URL')
        self.process: Optional[subprocess.Popen] = None
        
        # Set Python path
        os.environ['PYTHONPATH'] = '/app'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        logger.info(f"🚀 NodeRAG Service Manager starting...")
        logger.info(f"Service Type: {self.service_type}")
        logger.info(f"Port: {self.port}")
        logger.info(f"Redis URL: {self.redis_url[:20]}..." if self.redis_url else "Redis URL: Not set")
    
    def wait_for_redis(self, timeout: int = 60) -> bool:
        """Wait for Redis to be available"""
        if not self.redis_url:
            logger.error("❌ REDIS_URL environment variable not set")
            return False
            
        logger.info("⏳ Waiting for Redis connection...")
        
        for attempt in range(timeout):
            try:
                import redis
                client = redis.Redis.from_url(self.redis_url)
                client.ping()
                logger.info("✅ Redis connected successfully")
                return True
            except Exception as e:
                if attempt < 5:  # Only log first few attempts
                    logger.debug(f"Redis connection attempt {attempt + 1}: {e}")
                elif attempt % 10 == 0:  # Log every 10th attempt after that
                    logger.info(f"Still waiting for Redis... (attempt {attempt + 1}/{timeout})")
                time.sleep(1)
        
        logger.error(f"❌ Redis connection timeout after {timeout} seconds")
        return False
    
    def start_api_service(self):
        """Start the Flask API server"""
        logger.info("🌐 Starting NodeRAG API Service")
        
        if not self.wait_for_redis():
            sys.exit(1)
        
        # Check if gunicorn is available, fallback to flask dev server
        try:
            subprocess.check_call(['which', 'gunicorn'], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            use_gunicorn = True
        except subprocess.CalledProcessError:
            use_gunicorn = False
            logger.warning("⚠️ gunicorn not found, using Flask dev server")
        
        if use_gunicorn:
            # Production WSGI server
            cmd = [
                'gunicorn',
                '--bind', f'0.0.0.0:{self.port}',
                '--workers', '1',  # Single worker for 2GB RAM
                '--worker-class', 'sync',
                '--worker-connections', '1000',
                '--max-requests', '100',
                '--max-requests-jitter', '10',
                '--timeout', '300',  # 5 minutes for long document processing
                '--keep-alive', '5',
                '--log-level', 'info',
                '--access-logfile', '-',
                '--error-logfile', '-',
                'api_service:app'
            ]
        else:
            # Development server fallback
            cmd = [
                'python', 'api_service.py'
            ]
            os.environ['FLASK_ENV'] = 'production'
            os.environ['FLASK_DEBUG'] = '0'
        
        logger.info(f"📋 Command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
        return self.process.wait()
    
    def start_worker_service(self):
        """Start the Celery worker"""
        logger.info("⚙️ Starting NodeRAG Celery Worker")
        
        if not self.wait_for_redis():
            sys.exit(1)
        
        # Get system memory info for optimization
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            total_ram_kb = int([line for line in meminfo.split('\n') 
                              if line.startswith('MemTotal')][0].split()[1])
            total_ram_mb = total_ram_kb // 1024
        except:
            total_ram_mb = 2048  # Default assumption
        
        logger.info(f"📊 System RAM: {total_ram_mb}MB")
        
        # Set memory limit based on available RAM
        if total_ram_mb < 2048:
            max_memory_kb = 1200000  # 1.2GB for low RAM
            logger.warning("⚠️ Low RAM environment - setting conservative memory limit: 1.2GB")
        elif total_ram_mb < 4096:
            max_memory_kb = 1500000  # 1.5GB for standard
            logger.info("📊 Standard RAM environment - setting memory limit: 1.5GB")
        else:
            max_memory_kb = 2000000  # 2GB for high RAM
            logger.info("🚀 High RAM environment - setting memory limit: 2GB")
        
        # Start Celery worker
        cmd = [
            'celery', '-A', 'src.queue.celery_config', 'worker',
            '--loglevel=info',
            '--concurrency=1',
            '--max-tasks-per-child=1',
            f'--max-memory-per-child={max_memory_kb}',
            '--prefetch-multiplier=1',
            '--pool=solo',
            '--queues=document_processing,cleanup',
            f'--hostname=noderag-worker@{os.getenv("RENDER_SERVICE_NAME", "render")}'
        ]
        
        logger.info(f"📋 Command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
        return self.process.wait()
    
    def start_scheduler_service(self):
        """Start the Celery beat scheduler"""
        logger.info("🕒 Starting NodeRAG Celery Beat Scheduler")
        
        if not self.wait_for_redis():
            sys.exit(1)
        
        cmd = [
            'celery', '-A', 'src.queue.celery_config', 'beat',
            '--loglevel=info',
            '--scheduler=celery.beat:PersistentScheduler'
        ]
        
        logger.info(f"📋 Command: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
        return self.process.wait()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ Process didn't terminate gracefully, killing...")
                    self.process.kill()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def run(self):
        """Main entry point"""
        self.setup_signal_handlers()
        
        try:
            if self.service_type == 'api':
                return self.start_api_service()
            elif self.service_type == 'worker':
                return self.start_worker_service()
            elif self.service_type == 'scheduler':
                return self.start_scheduler_service()
            else:
                logger.error(f"❌ Unknown service type: {self.service_type}")
                logger.error("Valid service types: api, worker, scheduler")
                return 1
                
        except KeyboardInterrupt:
            logger.info("🛑 Received interrupt, shutting down...")
            return 0
        except Exception as e:
            logger.error(f"❌ Service failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

def main():
    """Main function"""
    try:
        # Install missing dependencies if needed
        logger.info("🔍 Checking dependencies...")
        
        # Check for required packages
        required_packages = ['celery', 'redis', 'gunicorn']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.info(f"📦 Installing missing packages: {missing_packages}")
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
            subprocess.check_call(cmd)
            logger.info("✅ Dependencies installed")
        
        # Start the service manager
        manager = NodeRAGServiceManager()
        return manager.run()
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
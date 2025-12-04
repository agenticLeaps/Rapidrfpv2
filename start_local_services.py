#!/usr/bin/env python3
"""
Local Service Starter for NodeRAG Queue System
Starts all required services for local testing
"""

import subprocess
import time
import os
import signal
import sys
import threading
from typing import List, Optional

class ServiceManager:
    """Manages local services for NodeRAG testing"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.services = {
            "redis": None,
            "api": None,
            "worker": None,
            "scheduler": None
        }
        
    def start_redis(self) -> bool:
        """Start Redis server"""
        try:
            print("🔄 Starting Redis server...")
            
            # Check if Redis is already running
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                print("✅ Redis is already running")
                return True
            except:
                pass
            
            # Start Redis
            process = subprocess.Popen(
                ['redis-server', '--daemonize', 'yes'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a bit and check if it started
            time.sleep(2)
            
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0)
                r.ping()
                print("✅ Redis started successfully")
                self.services["redis"] = process
                return True
            except Exception as e:
                print(f"❌ Redis failed to start: {e}")
                return False
                
        except FileNotFoundError:
            print("❌ Redis not found. Please install Redis:")
            print("   macOS: brew install redis")
            print("   Ubuntu: sudo apt-get install redis-server")
            return False
        except Exception as e:
            print(f"❌ Failed to start Redis: {e}")
            return False
    
    def start_api_service(self) -> bool:
        """Start the API service"""
        try:
            print("🔄 Starting API service...")
            
            env = os.environ.copy()
            env['NODERAG_SERVICE_TYPE'] = 'api'
            env['REDIS_URL'] = 'redis://localhost:6379/0'
            env['PYTHONPATH'] = '.'
            env['PORT'] = '8000'
            
            process = subprocess.Popen(
                [sys.executable, 'start_noderag_service.py'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            self.services["api"] = process
            self.processes.append(process)
            
            # Start log reader thread
            threading.Thread(
                target=self._log_reader,
                args=(process, "API"),
                daemon=True
            ).start()
            
            # Wait for service to start
            time.sleep(5)
            
            # Check if it's running
            if process.poll() is None:
                print("✅ API service started")
                return True
            else:
                print("❌ API service failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start API service: {e}")
            return False
    
    def start_worker_service(self) -> bool:
        """Start the Celery worker"""
        try:
            print("🔄 Starting Celery worker...")
            
            env = os.environ.copy()
            env['NODERAG_SERVICE_TYPE'] = 'worker'
            env['REDIS_URL'] = 'redis://localhost:6379/0'
            env['PYTHONPATH'] = '.'
            env['CELERY_OPTIMIZATION'] = '1'
            
            process = subprocess.Popen(
                [sys.executable, 'start_noderag_service.py'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            self.services["worker"] = process
            self.processes.append(process)
            
            # Start log reader thread
            threading.Thread(
                target=self._log_reader,
                args=(process, "WORKER"),
                daemon=True
            ).start()
            
            # Wait for service to start
            time.sleep(5)
            
            # Check if it's running
            if process.poll() is None:
                print("✅ Celery worker started")
                return True
            else:
                print("❌ Celery worker failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start Celery worker: {e}")
            return False
    
    def start_scheduler_service(self) -> bool:
        """Start the Celery beat scheduler (optional)"""
        try:
            print("🔄 Starting Celery scheduler...")
            
            env = os.environ.copy()
            env['NODERAG_SERVICE_TYPE'] = 'scheduler'
            env['REDIS_URL'] = 'redis://localhost:6379/0'
            env['PYTHONPATH'] = '.'
            
            process = subprocess.Popen(
                [sys.executable, 'start_noderag_service.py'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            self.services["scheduler"] = process
            self.processes.append(process)
            
            # Start log reader thread
            threading.Thread(
                target=self._log_reader,
                args=(process, "SCHEDULER"),
                daemon=True
            ).start()
            
            # Wait for service to start
            time.sleep(3)
            
            # Check if it's running
            if process.poll() is None:
                print("✅ Celery scheduler started")
                return True
            else:
                print("❌ Celery scheduler failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start Celery scheduler: {e}")
            return False
    
    def _log_reader(self, process: subprocess.Popen, service_name: str):
        """Read and display logs from a service"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[{service_name}] {line.strip()}")
        except:
            pass
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        # Stop our processes
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                except:
                    pass
        
        # Stop Redis if we started it
        if self.services["redis"]:
            try:
                subprocess.run(['redis-cli', 'shutdown'], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
                print("✅ Redis stopped")
            except:
                print("⚠️ Could not stop Redis (may have been running before)")
        
        print("✅ All services stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}")
            self.stop_all_services()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def start_all_services(self, include_scheduler: bool = False) -> bool:
        """Start all required services"""
        print("🚀 Starting NodeRAG Local Services")
        print("=" * 40)
        
        self.setup_signal_handlers()
        
        # Start Redis
        if not self.start_redis():
            return False
        
        time.sleep(2)
        
        # Start API service
        if not self.start_api_service():
            self.stop_all_services()
            return False
        
        time.sleep(3)
        
        # Start worker service
        if not self.start_worker_service():
            self.stop_all_services()
            return False
        
        time.sleep(2)
        
        # Start scheduler if requested
        if include_scheduler:
            if not self.start_scheduler_service():
                print("⚠️ Scheduler failed to start (optional)")
        
        print("\n✅ All services started successfully!")
        print("\n📋 Services running:")
        print("   🌐 API Server: http://localhost:8000")
        print("   ⚙️ Celery Worker: Processing queue tasks")
        if include_scheduler:
            print("   🕒 Celery Scheduler: Running cleanup tasks")
        print("   📊 Redis: localhost:6379")
        
        print("\n💡 Ready for testing!")
        print("   Run: python test_queue_local.py")
        print("   Or test API: curl http://localhost:8000/api/v1/health")
        
        return True
    
    def run_interactive(self):
        """Run in interactive mode"""
        try:
            include_scheduler = input("🤔 Start scheduler service? (y/n): ").lower().strip() == 'y'
            
            if self.start_all_services(include_scheduler=include_scheduler):
                print("\n🎯 Services are running. Press Ctrl+C to stop.")
                
                # Keep running until interrupted
                while True:
                    time.sleep(1)
                    
                    # Check if any service died
                    for service_name, process in self.services.items():
                        if process and process.poll() is not None:
                            print(f"⚠️ Service {service_name} stopped unexpectedly")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
        finally:
            self.stop_all_services()

def main():
    """Main function"""
    manager = ServiceManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        # Non-interactive mode
        success = manager.start_all_services(include_scheduler=False)
        if success:
            try:
                print("\n🎯 Services running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        manager.stop_all_services()
        return 0 if success else 1
    else:
        # Interactive mode
        manager.run_interactive()
        return 0

if __name__ == "__main__":
    sys.exit(main())
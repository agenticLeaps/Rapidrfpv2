#!/usr/bin/env python3
"""
RapidRFP RAG System Launcher

This script provides an easy way to start the complete system:
- Flask API server
- Streamlit web interface  
- System health monitoring

Usage:
    python run_system.py [--mode api|web|both] [--host 0.0.0.0] [--api-port 5000] [--web-port 8501]
"""

import os
import sys
import time
import argparse
import subprocess
import threading
import signal
import requests
from typing import List

class SystemLauncher:
    """Launch and manage RapidRFP RAG system components."""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all processes."""
        self.running = False
        
        print("🔄 Stopping all processes...")
        for process in self.processes:
            try:
                process.terminate()
                # Give process a moment to terminate gracefully
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing process {process.pid}")
                process.kill()
            except:
                pass
        
        print("✅ All processes stopped.")
    
    def check_api_health(self, api_url: str, max_retries: int = 30) -> bool:
        """Check if API is responding."""
        for i in range(max_retries):
            try:
                response = requests.get(f"{api_url}/health", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        return True
            except:
                pass
            
            if not self.running:
                return False
            
            print(f"⏳ Waiting for API... ({i+1}/{max_retries})")
            time.sleep(2)
        
        return False
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 5001) -> subprocess.Popen:
        """Start Flask API server."""
        print(f"🚀 Starting API server on {host}:{port}...")
        
        env = os.environ.copy()
        env.update({
            'API_HOST': host,
            'API_PORT': str(port),
            'FLASK_ENV': 'development'
        })
        
        process = subprocess.Popen(
            [sys.executable, 'app.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes.append(process)
        
        # Monitor output in a separate thread
        def monitor_api_output():
            while self.running and process.poll() is None:
                try:
                    line = process.stdout.readline()
                    if line:
                        print(f"[API] {line.strip()}")
                except:
                    break
        
        threading.Thread(target=monitor_api_output, daemon=True).start()
        
        return process
    
    def start_web_interface(self, host: str = "0.0.0.0", port: int = 8501, 
                          api_url: str = "http://localhost:5001") -> subprocess.Popen:
        """Start Streamlit web interface."""
        print(f"🌐 Starting web interface on {host}:{port}...")
        
        # Set environment variables for Streamlit
        env = os.environ.copy()
        env.update({
            'RAPIDRFP_API_URL': api_url
        })
        
        process = subprocess.Popen(
            [
                sys.executable, '-m', 'streamlit', 'run', 
                'web_ui.py',
                '--server.address', host,
                '--server.port', str(port),
                '--browser.gatherUsageStats', 'false',
                '--theme.base', 'dark'
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.processes.append(process)
        
        # Monitor output in a separate thread  
        def monitor_web_output():
            while self.running and process.poll() is None:
                try:
                    line = process.stdout.readline()
                    if line:
                        print(f"[WEB] {line.strip()}")
                except:
                    break
        
        threading.Thread(target=monitor_web_output, daemon=True).start()
        
        return process
    
    def run_api_only(self, host: str = "0.0.0.0", port: int = 5001):
        """Run API server only."""
        print("🔧 Mode: API Only")
        print("=" * 50)
        
        api_process = self.start_api_server(host, port)
        api_url = f"http://{host}:{port}"
        
        # Wait for API to be ready
        if self.check_api_health(api_url):
            print(f"✅ API server is ready at {api_url}")
            print(f"📋 API Documentation: {api_url}/health")
            print(f"🔗 Graph Stats: {api_url}/api/graph/stats")
        else:
            print("❌ API server failed to start")
            self.shutdown()
            return
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.shutdown()
    
    def run_web_only(self, host: str = "0.0.0.0", port: int = 8501, 
                    api_url: str = "http://localhost:5001"):
        """Run web interface only (assumes API is already running)."""
        print("🌐 Mode: Web Interface Only")
        print("=" * 50)
        
        # Check if API is available
        print(f"🔍 Checking API availability at {api_url}...")
        if not self.check_api_health(api_url, max_retries=3):
            print(f"❌ API not available at {api_url}")
            print("   Please start the API server first or use --mode both")
            return
        
        web_process = self.start_web_interface(host, port, api_url)
        
        # Give Streamlit a moment to start
        time.sleep(3)
        
        web_url = f"http://{host}:{port}"
        print(f"✅ Web interface is ready at {web_url}")
        print(f"🎨 Access the dashboard in your browser")
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.shutdown()
    
    def run_both(self, api_host: str = "0.0.0.0", api_port: int = 5001,
                web_host: str = "0.0.0.0", web_port: int = 8501):
        """Run both API and web interface."""
        print("🚀 Mode: Complete System")
        print("=" * 50)
        
        # Start API server
        api_process = self.start_api_server(api_host, api_port)
        api_url = f"http://{api_host}:{api_port}"
        
        # Wait for API to be ready
        if not self.check_api_health(api_url):
            print("❌ API server failed to start")
            self.shutdown()
            return
        
        print(f"✅ API server is ready at {api_url}")
        
        # Start web interface
        web_process = self.start_web_interface(web_host, web_port, api_url)
        
        # Give Streamlit a moment to start
        time.sleep(3)
        
        web_url = f"http://{web_host}:{web_port}"
        
        print("\n" + "=" * 50)
        print("🎉 RapidRFP RAG System is Ready!")
        print("=" * 50)
        print(f"🔧 API Server:     {api_url}")
        print(f"🌐 Web Interface:  {web_url}")
        print("=" * 50)
        print("\n📋 Quick Start:")
        print("1. Open the web interface in your browser")
        print("2. Upload a document using the sidebar")
        print("3. Explore the graph and search capabilities")
        print("4. Generate visualizations")
        print("\n🛑 Press Ctrl+C to stop the system")
        
        # Keep running until interrupted
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        self.shutdown()

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description='Launch RapidRFP RAG System components',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system.py                          # Start both API and web interface
  python run_system.py --mode api               # Start API server only  
  python run_system.py --mode web               # Start web interface only
  python run_system.py --api-port 5001          # Use custom API port
  python run_system.py --host 127.0.0.1         # Bind to localhost only
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['api', 'web', 'both'],
        default='both',
        help='Which components to start (default: both)'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        default=5001,
        help='API server port (default: 5001)'
    )
    
    parser.add_argument(
        '--web-port',
        type=int,
        default=8501,
        help='Web interface port (default: 8501)'
    )
    
    parser.add_argument(
        '--api-url',
        default='http://localhost:5001',
        help='API URL for web interface (default: http://localhost:5001)'
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists('app.py') or not os.path.exists('web_ui.py'):
        print("❌ Error: Please run this script from the RapidRFP RAG project directory")
        print("   Make sure app.py and web_ui.py are in the current directory")
        sys.exit(1)
    
    # Create launcher and run
    launcher = SystemLauncher()
    
    try:
        if args.mode == 'api':
            launcher.run_api_only(args.host, args.api_port)
        elif args.mode == 'web':
            launcher.run_web_only(args.host, args.web_port, args.api_url)
        else:  # both
            launcher.run_both(args.host, args.api_port, args.host, args.web_port)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        launcher.shutdown()
        sys.exit(1)

if __name__ == '__main__':
    main()
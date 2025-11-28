#!/usr/bin/env python3
"""
Startup script for NodeRAG API Service
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_environment():
    """Check required environment variables"""
    required_env_vars = [
        "NEO4J_URI",
        "NEO4J_USERNAME", 
        "NEO4J_PASSWORD",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True

def main():
    """Main startup function"""
    logger.info("🚀 Starting NodeRAG API Service...")
    
    # Check environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        sys.exit(1)
    
    # Import and start the API service
    try:
        from api_service import app
        from src.config.settings import Config
        
        host = Config.API_HOST
        port = Config.API_PORT
        debug = Config.API_DEBUG
        
        logger.info(f"✅ Environment check passed")
        logger.info(f"🌐 Starting server on {host}:{port} (debug={debug})")
        
        # Run the Flask app
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start NodeRAG service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
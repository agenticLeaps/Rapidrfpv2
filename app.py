#!/usr/bin/env python3
"""
RapidRFP RAG System - Main Application Entry Point

A Graph-based Retrieval Augmented Generation system for document processing.
Implements a multi-phase indexing pipeline with LLM-powered node extraction.
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logging_config import setup_logging
from src.api.routes import app

def main():
    """Main application entry point."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting RapidRFP RAG System")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Import config for API settings
    from src.config.settings import Config
    
    # Start Flask application
    logger.info(f"Starting Flask API server on http://{Config.API_HOST}:{Config.API_PORT}")
    app.run(debug=Config.API_DEBUG, host=Config.API_HOST, port=Config.API_PORT)

if __name__ == '__main__':
    main()
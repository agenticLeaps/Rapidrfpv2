import logging
import os
from logging.handlers import RotatingFileHandler
from ..config.settings import Config

def setup_logging():
    """Set up minimal logging configuration for performance."""
    
    # Set up root logger with minimal configuration for performance
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)  # Only log critical errors
    
    # Only critical console output for errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
    
    # Disable all non-critical logging for performance
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('requests').setLevel(logging.CRITICAL)
    logging.getLogger('gradio_client').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    
    # Disable our application loggers for performance
    logging.getLogger('src.llm.llm_service').setLevel(logging.CRITICAL)
    logging.getLogger('src.document_processing.indexing_pipeline').setLevel(logging.CRITICAL)
    logging.getLogger('src.storage.neon_storage').setLevel(logging.CRITICAL)
    logging.getLogger('src.graph.graph_manager').setLevel(logging.CRITICAL)
    
    return root_logger
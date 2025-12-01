import logging
from ..config.settings import Config

def setup_logging():
    """Set up logging configuration for the application."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler only (no file logging to save space)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('gradio_client').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('google_genai.models').setLevel(logging.WARNING)
    
    return root_logger
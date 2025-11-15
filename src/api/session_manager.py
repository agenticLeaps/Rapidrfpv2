"""
Session Manager for isolating user data and services.
Ensures each session has its own graph, search system, and data storage.
"""

import os
import shutil
import threading
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..document_processing.indexing_pipeline import IndexingPipeline
from ..document_processing.document_loader import DocumentLoader
from ..incremental.incremental_pipeline import IncrementalIndexingPipeline
from ..search.advanced_search import AdvancedSearchSystem
from ..vector.hnsw_service import HNSWService
from ..visualization.graph_visualizer import GraphVisualizer
from ..llm.llm_service import LLMService
from ..config.settings import Config

logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    """Data container for a user session."""
    session_id: str
    created_at: datetime
    last_accessed: datetime
    data_dir: str
    indexing_pipeline: IndexingPipeline
    incremental_pipeline: IncrementalIndexingPipeline
    document_loader: DocumentLoader
    advanced_search: Optional[AdvancedSearchSystem] = None
    hnsw_service: Optional[HNSWService] = None
    graph_visualizer: Optional[GraphVisualizer] = None
    llm_service: Optional[LLMService] = None

class SessionManager:
    """Manages isolated sessions for multi-user support."""
    
    def __init__(self, base_data_dir: str = "data/sessions", session_timeout_hours: int = 24):
        self.base_data_dir = base_data_dir
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        
        # Create base directory
        os.makedirs(self.base_data_dir, exist_ok=True)
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Session manager initialized with {session_timeout_hours}h timeout")
    
    def get_or_create_session(self, session_id: str) -> SessionData:
        """Get existing session or create new one."""
        with self._lock:
            if session_id in self.sessions:
                # Update last accessed time
                self.sessions[session_id].last_accessed = datetime.now()
                logger.debug(f"Retrieved existing session: {session_id}")
                return self.sessions[session_id]
            
            # Create new session
            session_data = self._create_session(session_id)
            self.sessions[session_id] = session_data
            logger.info(f"Created new session: {session_id}")
            return session_data
    
    def _create_session(self, session_id: str) -> SessionData:
        """Create a new isolated session."""
        # Create session-specific data directory
        session_data_dir = os.path.join(self.base_data_dir, session_id)
        os.makedirs(session_data_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['raw', 'processed', 'graphs', 'hnsw', 'visualizations']
        for subdir in subdirs:
            os.makedirs(os.path.join(session_data_dir, subdir), exist_ok=True)
        
        # Initialize session-specific services
        indexing_pipeline = IndexingPipeline()
        
        # Override graph database path for session isolation
        session_graph_path = os.path.join(session_data_dir, 'graphs', 'knowledge_graph.graphml')
        indexing_pipeline.graph_manager.graph_db_path = session_graph_path
        
        incremental_pipeline = IncrementalIndexingPipeline()
        incremental_pipeline.indexing_pipeline = indexing_pipeline
        
        document_loader = DocumentLoader()
        
        session_data = SessionData(
            session_id=session_id,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            data_dir=session_data_dir,
            indexing_pipeline=indexing_pipeline,
            incremental_pipeline=incremental_pipeline,
            document_loader=document_loader
        )
        
        logger.info(f"Session {session_id} created with data dir: {session_data_dir}")
        return session_data
    
    def get_advanced_search(self, session_id: str) -> AdvancedSearchSystem:
        """Get or initialize advanced search for session."""
        session_data = self.get_or_create_session(session_id)
        
        if session_data.advanced_search is None:
            # Initialize HNSW service first
            hnsw_service = self.get_hnsw_service(session_id)
            
            # Initialize LLM service
            if session_data.llm_service is None:
                session_data.llm_service = LLMService()
            
            # Initialize advanced search
            session_data.advanced_search = AdvancedSearchSystem(
                graph_manager=session_data.indexing_pipeline.graph_manager,
                hnsw_service=hnsw_service,
                llm_service=session_data.llm_service
            )
            logger.info(f"Advanced search initialized for session: {session_id}")
        
        return session_data.advanced_search
    
    def get_hnsw_service(self, session_id: str) -> HNSWService:
        """Get or initialize HNSW service for session."""
        session_data = self.get_or_create_session(session_id)
        
        if session_data.hnsw_service is None:
            # Create session-specific HNSW service
            session_data.hnsw_service = HNSWService()
            
            # Override index path for session isolation
            hnsw_dir = os.path.join(session_data.data_dir, 'hnsw')
            session_data.hnsw_service.index_path = os.path.join(hnsw_dir, 'hnsw_index.bin')
            session_data.hnsw_service.metadata_path = os.path.join(hnsw_dir, 'hnsw_metadata.json')
            
            # Try to load existing index
            try:
                loaded = session_data.hnsw_service.load_index()
                if loaded:
                    logger.info(f"HNSW index loaded for session: {session_id}")
                else:
                    logger.info(f"New HNSW index created for session: {session_id}")
            except Exception as e:
                logger.warning(f"Could not load HNSW index for session {session_id}: {e}")
        
        return session_data.hnsw_service
    
    def get_graph_visualizer(self, session_id: str) -> GraphVisualizer:
        """Get or initialize graph visualizer for session."""
        session_data = self.get_or_create_session(session_id)
        
        if session_data.graph_visualizer is None:
            session_data.graph_visualizer = GraphVisualizer(
                session_data.indexing_pipeline.graph_manager
            )
            
            # Override output directory
            viz_dir = os.path.join(session_data.data_dir, 'visualizations')
            session_data.graph_visualizer.output_dir = viz_dir
            
            logger.info(f"Graph visualizer initialized for session: {session_id}")
        
        return session_data.graph_visualizer
    
    def clear_session(self, session_id: str) -> bool:
        """Clear all data for a specific session."""
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session_data = self.sessions[session_id]
            
            # Remove session data directory
            try:
                if os.path.exists(session_data.data_dir):
                    shutil.rmtree(session_data.data_dir)
                    logger.info(f"Removed data directory for session: {session_id}")
            except Exception as e:
                logger.error(f"Error removing session data directory {session_id}: {e}")
            
            # Remove session from memory
            del self.sessions[session_id]
            logger.info(f"Session cleared: {session_id}")
            return True
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session."""
        session_data = self.get_or_create_session(session_id)
        
        # Get graph stats
        graph_stats = session_data.indexing_pipeline.get_indexing_stats()
        
        # Get data directory size
        data_dir_size = self._get_directory_size(session_data.data_dir)
        
        return {
            'session_id': session_id,
            'created_at': session_data.created_at.isoformat(),
            'last_accessed': session_data.last_accessed.isoformat(),
            'data_directory_size_mb': round(data_dir_size / (1024 * 1024), 2),
            'graph_stats': graph_stats
        }
    
    def list_active_sessions(self) -> Dict:
        """List all active sessions."""
        with self._lock:
            return {
                'total_sessions': len(self.sessions),
                'sessions': [
                    {
                        'session_id': session_id,
                        'created_at': session_data.created_at.isoformat(),
                        'last_accessed': session_data.last_accessed.isoformat(),
                        'data_dir': session_data.data_dir
                    }
                    for session_id, session_data in self.sessions.items()
                ]
            }
    
    def _cleanup_expired_sessions(self):
        """Background cleanup of expired sessions."""
        while True:
            try:
                time.sleep(3600)  # Check every hour
                
                with self._lock:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session_data in self.sessions.items():
                        if current_time - session_data.last_accessed > self.session_timeout:
                            expired_sessions.append(session_id)
                
                # Clean up expired sessions outside the lock
                for session_id in expired_sessions:
                    self.clear_session(session_id)
                    logger.info(f"Automatically cleaned up expired session: {session_id}")
                    
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def _get_directory_size(self, directory: str) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")
        return total_size

# Global session manager instance
session_manager = SessionManager()
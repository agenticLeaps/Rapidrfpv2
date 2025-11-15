import os
import json
import hashlib
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from ..config.settings import Config

logger = logging.getLogger(__name__)

@dataclass
class DocumentState:
    """State information for a processed document."""
    file_path: str
    file_hash: str
    file_size: int
    last_modified: float
    processed_at: datetime
    chunks_processed: int
    node_ids_created: List[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class PipelineState:
    """Overall pipeline state information."""
    current_phase: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    last_update: datetime
    is_incremental: bool
    error_count: int
    
class StateManager:
    """
    Manages incremental updates and pipeline state.
    Tracks processed documents and enables resuming from failures.
    """
    
    def __init__(self, state_dir: str = None):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = state_dir or os.path.join(os.path.dirname(Config.GRAPH_DB_PATH), "state")
        os.makedirs(self.state_dir, exist_ok=True)
        
        # State file paths
        self.pipeline_state_path = os.path.join(self.state_dir, "pipeline_state.json")
        self.documents_state_path = os.path.join(self.state_dir, "documents_state.json")
        self.error_log_path = os.path.join(self.state_dir, "error_log.jsonl")
        
        # Load existing state
        self.pipeline_state = self._load_pipeline_state()
        self.document_states: Dict[str, DocumentState] = self._load_document_states()
        
        logger.info(f"State manager initialized: {len(self.document_states)} documents tracked")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file content."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""
    
    def _load_pipeline_state(self) -> PipelineState:
        """Load pipeline state from disk."""
        try:
            if os.path.exists(self.pipeline_state_path):
                with open(self.pipeline_state_path, 'r') as f:
                    data = json.load(f)
                    data['last_update'] = datetime.fromisoformat(data['last_update'])
                    return PipelineState(**data)
            else:
                return PipelineState(
                    current_phase="INIT",
                    total_documents=0,
                    processed_documents=0,
                    failed_documents=0,
                    last_update=datetime.now(),
                    is_incremental=False,
                    error_count=0
                )
        except Exception as e:
            logger.error(f"Error loading pipeline state: {e}")
            return PipelineState(
                current_phase="ERROR",
                total_documents=0,
                processed_documents=0,
                failed_documents=0,
                last_update=datetime.now(),
                is_incremental=False,
                error_count=1
            )
    
    def _load_document_states(self) -> Dict[str, DocumentState]:
        """Load document states from disk."""
        document_states = {}
        try:
            if os.path.exists(self.documents_state_path):
                with open(self.documents_state_path, 'r') as f:
                    data = json.load(f)
                    for file_path, state_data in data.items():
                        state_data['processed_at'] = datetime.fromisoformat(state_data['processed_at'])
                        document_states[file_path] = DocumentState(**state_data)
        except Exception as e:
            logger.error(f"Error loading document states: {e}")
        
        return document_states
    
    def _save_pipeline_state(self):
        """Save pipeline state to disk."""
        try:
            state_dict = asdict(self.pipeline_state)
            state_dict['last_update'] = self.pipeline_state.last_update.isoformat()
            
            with open(self.pipeline_state_path, 'w') as f:
                json.dump(state_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pipeline state: {e}")
    
    def _save_document_states(self):
        """Save document states to disk."""
        try:
            states_dict = {}
            for file_path, state in self.document_states.items():
                state_dict = asdict(state)
                state_dict['processed_at'] = state.processed_at.isoformat()
                states_dict[file_path] = state_dict
            
            with open(self.documents_state_path, 'w') as f:
                json.dump(states_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving document states: {e}")
    
    def check_document_needs_processing(self, file_path: str) -> bool:
        """
        Check if a document needs processing (new or modified).
        
        Args:
            file_path: Path to document file
            
        Returns:
            True if document needs processing
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return False
            
            # Get current file info
            current_hash = self._compute_file_hash(file_path)
            current_size = os.path.getsize(file_path)
            current_mtime = os.path.getmtime(file_path)
            
            # Check if we have previous state
            if file_path not in self.document_states:
                logger.info(f"New document detected: {file_path}")
                return True
            
            prev_state = self.document_states[file_path]
            
            # Check if file was successfully processed before
            if not prev_state.success:
                logger.info(f"Previously failed document: {file_path}")
                return True
            
            # Check if file has changed
            if (current_hash != prev_state.file_hash or 
                current_size != prev_state.file_size or
                current_mtime != prev_state.last_modified):
                logger.info(f"Modified document detected: {file_path}")
                return True
            
            logger.debug(f"Document up to date: {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"Error checking document state for {file_path}: {e}")
            return True  # Process if unsure
    
    def get_documents_to_process(self, file_paths: List[str]) -> List[str]:
        """
        Filter list of documents to only those needing processing.
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            List of file paths that need processing
        """
        needs_processing = []
        
        for file_path in file_paths:
            if self.check_document_needs_processing(file_path):
                needs_processing.append(file_path)
        
        logger.info(f"Documents needing processing: {len(needs_processing)}/{len(file_paths)}")
        return needs_processing
    
    def start_document_processing(self, file_path: str):
        """Mark document as being processed."""
        try:
            current_hash = self._compute_file_hash(file_path)
            current_size = os.path.getsize(file_path)
            current_mtime = os.path.getmtime(file_path)
            
            # Create initial state
            self.document_states[file_path] = DocumentState(
                file_path=file_path,
                file_hash=current_hash,
                file_size=current_size,
                last_modified=current_mtime,
                processed_at=datetime.now(),
                chunks_processed=0,
                node_ids_created=[],
                processing_time=0.0,
                success=False
            )
            
            logger.debug(f"Started processing: {file_path}")
            
        except Exception as e:
            logger.error(f"Error starting document processing for {file_path}: {e}")
    
    def complete_document_processing(self, 
                                   file_path: str,
                                   chunks_processed: int,
                                   node_ids_created: List[str],
                                   processing_time: float,
                                   success: bool,
                                   error_message: Optional[str] = None):
        """Mark document processing as complete."""
        try:
            if file_path in self.document_states:
                state = self.document_states[file_path]
                state.chunks_processed = chunks_processed
                state.node_ids_created = node_ids_created.copy()
                state.processing_time = processing_time
                state.success = success
                state.error_message = error_message
                state.processed_at = datetime.now()
                
                if success:
                    self.pipeline_state.processed_documents += 1
                    logger.info(f"Document processing completed: {file_path}")
                else:
                    self.pipeline_state.failed_documents += 1
                    self._log_error(file_path, error_message or "Unknown error")
                    logger.warning(f"Document processing failed: {file_path}")
                
                self._save_document_states()
                self._save_pipeline_state()
            
        except Exception as e:
            logger.error(f"Error completing document processing for {file_path}: {e}")
    
    def _log_error(self, file_path: str, error_message: str):
        """Log error to error log file."""
        try:
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'error_message': error_message,
                'phase': self.pipeline_state.current_phase
            }
            
            with open(self.error_log_path, 'a') as f:
                f.write(json.dumps(error_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def update_pipeline_phase(self, phase: str):
        """Update current pipeline phase."""
        self.pipeline_state.current_phase = phase
        self.pipeline_state.last_update = datetime.now()
        self._save_pipeline_state()
        logger.info(f"Pipeline phase updated to: {phase}")
    
    def set_incremental_mode(self, is_incremental: bool):
        """Set incremental processing mode."""
        self.pipeline_state.is_incremental = is_incremental
        self._save_pipeline_state()
        logger.info(f"Incremental mode: {is_incremental}")
    
    def get_failed_documents(self) -> List[str]:
        """Get list of documents that failed processing."""
        failed = []
        for file_path, state in self.document_states.items():
            if not state.success:
                failed.append(file_path)
        return failed
    
    def reset_failed_documents(self):
        """Reset failed documents for reprocessing."""
        count = 0
        for file_path, state in self.document_states.items():
            if not state.success:
                del self.document_states[file_path]
                count += 1
        
        if count > 0:
            self.pipeline_state.failed_documents -= count
            self._save_document_states()
            self._save_pipeline_state()
            logger.info(f"Reset {count} failed documents for reprocessing")
    
    def cleanup_deleted_documents(self, current_file_paths: Set[str]):
        """Remove state for documents that no longer exist."""
        to_remove = []
        for file_path in self.document_states:
            if file_path not in current_file_paths:
                to_remove.append(file_path)
        
        for file_path in to_remove:
            del self.document_states[file_path]
            logger.info(f"Cleaned up state for deleted document: {file_path}")
        
        if to_remove:
            self._save_document_states()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        total_processing_time = sum(
            state.processing_time for state in self.document_states.values()
            if state.success
        )
        
        total_nodes_created = sum(
            len(state.node_ids_created) for state in self.document_states.values()
            if state.success
        )
        
        return {
            'pipeline_phase': self.pipeline_state.current_phase,
            'total_documents': len(self.document_states),
            'processed_documents': self.pipeline_state.processed_documents,
            'failed_documents': self.pipeline_state.failed_documents,
            'success_rate': (self.pipeline_state.processed_documents / 
                           max(1, len(self.document_states))) * 100,
            'total_processing_time': total_processing_time,
            'total_nodes_created': total_nodes_created,
            'is_incremental': self.pipeline_state.is_incremental,
            'last_update': self.pipeline_state.last_update.isoformat(),
            'error_count': self.pipeline_state.error_count
        }
    
    def can_resume_processing(self) -> bool:
        """Check if processing can be resumed from previous state."""
        return (self.pipeline_state.current_phase != "INIT" and 
                self.pipeline_state.current_phase != "FINISHED" and
                len(self.document_states) > 0)
    
    def clear_all_state(self):
        """Clear all state (for fresh processing)."""
        try:
            self.document_states.clear()
            self.pipeline_state = PipelineState(
                current_phase="INIT",
                total_documents=0,
                processed_documents=0,
                failed_documents=0,
                last_update=datetime.now(),
                is_incremental=False,
                error_count=0
            )
            
            # Remove state files
            for path in [self.pipeline_state_path, self.documents_state_path, self.error_log_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info("All state cleared")
            
        except Exception as e:
            logger.error(f"Error clearing state: {e}")
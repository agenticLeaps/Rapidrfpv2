#!/usr/bin/env python3
"""
Queue Manager for NodeRAG Document Processing
"""

import logging
import time
import json
from typing import Dict, Any, Optional
from celery.result import AsyncResult
from .celery_config import celery_app
from .tasks import process_document_task

logger = logging.getLogger(__name__)

class NodeRAGQueueManager:
    """
    Queue Manager for handling NodeRAG document processing with memory constraints
    """
    
    def __init__(self):
        self.celery_app = celery_app
        
    def submit_document_processing(self, org_id: str, file_id: str, user_id: str, 
                                 chunks: list, callback_url: str = None) -> Dict[str, Any]:
        """
        Submit a document processing task to the queue
        
        Args:
            org_id: Organization ID
            file_id: File ID
            user_id: User ID  
            chunks: List of document chunks
            callback_url: Optional webhook URL
            
        Returns:
            Dict with task information
        """
        try:
            logger.info(f"📥 Submitting document processing task: org_id={org_id}, file_id={file_id}")
            
            # Submit task to Celery queue
            task_result = process_document_task.delay(
                org_id=org_id,
                file_id=file_id,
                user_id=user_id,
                chunks=chunks,
                callback_url=callback_url
            )
            
            logger.info(f"✅ Task submitted with ID: {task_result.id}")
            
            return {
                "success": True,
                "task_id": task_result.id,
                "status": "queued",
                "message": "Document processing task submitted to queue",
                "file_id": file_id,
                "org_id": org_id,
                "estimated_time": "2-10 minutes depending on queue"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to submit task: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to submit document processing task"
            }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a processing task
        
        Args:
            task_id: Celery task ID
            
        Returns:
            Dict with task status information
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            
            if result.state == 'PENDING':
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "message": "Task is waiting in queue"
                }
            
            elif result.state == 'PROCESSING':
                meta = result.result or {}
                return {
                    "task_id": task_id,
                    "status": "processing",
                    "phase": meta.get('phase', 'unknown'),
                    "progress": meta.get('progress', 0),
                    "memory_mb": meta.get('memory_mb', 0),
                    "file_id": meta.get('file_id'),
                    "org_id": meta.get('org_id'),
                    "incremental_mode": meta.get('incremental_mode')
                }
            
            elif result.state == 'SUCCESS':
                meta = result.result or {}
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "phase": meta.get('phase', 'completed'),
                    "progress": 100,
                    "processing_time": meta.get('processing_time'),
                    "memory_mb": meta.get('memory_mb'),
                    "file_id": meta.get('file_id'),
                    "org_id": meta.get('org_id'),
                    "results": meta.get('results', {})
                }
            
            elif result.state == 'FAILURE':
                meta = result.result or {}
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "phase": meta.get('phase', 'error'),
                    "error": meta.get('error', str(result.result)),
                    "file_id": meta.get('file_id'),
                    "org_id": meta.get('org_id'),
                    "memory_mb": meta.get('memory_mb')
                }
            
            else:
                return {
                    "task_id": task_id,
                    "status": result.state.lower(),
                    "message": f"Task is in {result.state} state"
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "message": "Failed to get task status"
            }
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        Cancel a processing task
        
        Args:
            task_id: Celery task ID
            
        Returns:
            Dict with cancellation result
        """
        try:
            result = AsyncResult(task_id, app=self.celery_app)
            result.revoke(terminate=True)
            
            logger.info(f"🛑 Task {task_id} cancelled")
            
            return {
                "success": True,
                "task_id": task_id,
                "message": "Task cancelled successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Error cancelling task {task_id}: {e}")
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "message": "Failed to cancel task"
            }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics
        
        Returns:
            Dict with queue statistics
        """
        try:
            # Get active tasks
            inspect = self.celery_app.control.inspect()
            active_tasks = inspect.active()
            scheduled_tasks = inspect.scheduled()
            reserved_tasks = inspect.reserved()
            
            # Count tasks
            total_active = sum(len(tasks) for tasks in (active_tasks or {}).values())
            total_scheduled = sum(len(tasks) for tasks in (scheduled_tasks or {}).values())
            total_reserved = sum(len(tasks) for tasks in (reserved_tasks or {}).values())
            
            return {
                "queue_name": "document_processing",
                "active_tasks": total_active,
                "scheduled_tasks": total_scheduled,
                "reserved_tasks": total_reserved,
                "total_pending": total_active + total_scheduled + total_reserved,
                "workers_available": len(active_tasks or {}),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting queue stats: {e}")
            return {
                "error": str(e),
                "message": "Failed to get queue statistics"
            }
    
    def purge_queue(self) -> Dict[str, Any]:
        """
        Purge all pending tasks from queue (use with caution)
        
        Returns:
            Dict with purge result
        """
        try:
            purged_count = self.celery_app.control.purge()
            
            logger.warning(f"🗑️ Purged {purged_count} tasks from queue")
            
            return {
                "success": True,
                "purged_tasks": purged_count,
                "message": f"Purged {purged_count} tasks from queue"
            }
            
        except Exception as e:
            logger.error(f"❌ Error purging queue: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to purge queue"
            }

# Global instance
queue_manager = NodeRAGQueueManager()
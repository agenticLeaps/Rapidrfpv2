#!/usr/bin/env python3
"""
Celery Tasks for NodeRAG Document Processing
"""

import logging
import time
import gc
import psutil
import os
from typing import List, Dict, Any
from celery import current_task
from celery.exceptions import SoftTimeLimitExceeded
import requests

from .celery_config import celery_app
from src.document_processing.indexing_pipeline import IndexingPipeline
from src.document_processing.document_loader import DocumentChunk, ProcessedDocument

logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def memory_cleanup():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    logger.info(f"Memory after cleanup: {get_memory_usage():.2f} MB")

@celery_app.task(bind=True, name='src.queue.tasks.process_document_task')
def process_document_task(self, org_id: str, file_id: str, user_id: str, 
                         chunks: List[Dict], callback_url: str = None):
    """
    Celery task for processing documents with memory management
    
    Args:
        org_id: Organization ID
        file_id: File ID  
        user_id: User ID
        chunks: List of document chunks
        callback_url: Optional webhook URL
    """
    start_time = time.time()
    pipeline = None
    
    try:
        logger.info(f"🚀 Starting Celery task for file_id={file_id}, org_id={org_id}")
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
        
        # Log initialization (avoid update_state serialization issues)
        logger.info(f"🔄 Phase: initialization, Progress: 0%, Memory: {get_memory_usage():.2f} MB")
        
        # Send initial webhook
        if callback_url:
            send_webhook_safe(callback_url, "processing", {
                "file_id": file_id,
                "user_id": user_id,
                "phase": "initialization", 
                "progress": 0,
                "celery_task_id": current_task.request.id
            })
        
        # 🔒 Import and run the same org-level processing logic
        import sys
        import os
        # Ensure the root directory is in Python path
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from api_service import get_or_create_org_lock, NodeRAGService
        
        # Create service instance for this worker
        noderag_service = NodeRAGService()
        
        # Use org-level locking
        org_lock = get_or_create_org_lock(org_id)
        
        with org_lock:
            logger.info(f"🔒 Acquired org lock for {org_id} in Celery worker")
            
            # 🔍 STEP 1: Check if org already has a graph
            storage = noderag_service.get_neon_storage()
            existing_graph = storage.load_org_graph_sync(org_id)
            
            if existing_graph:
                logger.info(f"📊 Found existing org graph: {len(existing_graph['processed_files'])} files processed")
                
                # Check if this file was already processed
                if file_id in existing_graph['processed_files']:
                    logger.info(f"⚠️ File {file_id} already processed for org {org_id}")
                    
                    # Log skipped processing (avoid update_state serialization issues)
                    logger.info(f"⏭️ Phase: already_processed, Progress: 100%, File already processed: {file_id}")
                    
                    if callback_url:
                        send_webhook_safe(callback_url, "skipped", {
                            "file_id": file_id,
                            "user_id": user_id,
                            "message": "File already processed"
                        })
                    return {"status": "skipped", "message": "File already processed"}
                
                # Load existing graph into pipeline
                pipeline = noderag_service.get_pipeline()
                pipeline.set_incremental_mode(org_id, file_id, existing_graph['graph_data'])
                is_incremental = True
                
            else:
                logger.info(f"🆕 Creating first graph for org {org_id}")
                pipeline = noderag_service.get_pipeline()
                pipeline.set_incremental_mode(org_id, file_id)
                is_incremental = False
            
            logger.info(f"Memory after graph loading: {get_memory_usage():.2f} MB")
            
            # Log progress
            logger.info(f"🔄 Phase: chunk_conversion, Progress: 10%, Memory: {get_memory_usage():.2f} MB")
            
            # Convert chunks to NodeRAG format
            noderag_chunks = []
            for i, chunk_data in enumerate(chunks):
                chunk_text = chunk_data.get('content', chunk_data.get('text', ''))
                if chunk_text.strip():
                    noderag_chunk = DocumentChunk(
                        content=chunk_text,
                        chunk_index=i,
                        start_char=0,
                        end_char=len(chunk_text),
                        metadata={
                            'file_id': file_id,
                            'org_id': org_id,
                            'user_id': user_id,
                            'chunk_index': i,
                            'original_metadata': chunk_data.get('metadata', {})
                        },
                        token_count=len(chunk_text.split())
                    )
                    noderag_chunks.append(noderag_chunk)
            
            processed_doc = ProcessedDocument(
                chunks=noderag_chunks,
                metadata={
                    'file_id': file_id,
                    'org_id': org_id,
                    'user_id': user_id,
                    'total_chunks': len(noderag_chunks),
                    'source': 'celery_queue',
                    'incremental_mode': is_incremental
                },
                total_tokens=sum(chunk.token_count for chunk in noderag_chunks)
            )
            
            logger.info(f"Memory after chunk conversion: {get_memory_usage():.2f} MB")
            
            # Phase 1: Graph Decomposition
            logger.info(f"🔄 Phase: decomposition, Progress: 25%, Memory: {get_memory_usage():.2f} MB")
            
            logger.info(f"🔄 Phase 1: Graph Decomposition ({'Incremental' if is_incremental else 'Initial'})")
            result = pipeline._phase_1_decomposition(processed_doc)
            if not result['success']:
                logger.error(f"❌ Phase 1 FAILED: {result.get('error')}")
                raise Exception(f"Phase 1 failed: {result.get('error')}")
            
            logger.info(f"✅ Phase 1 COMPLETED - Graph Decomposition successful")
            logger.info(f"Memory after decomposition: {get_memory_usage():.2f} MB")
            memory_cleanup()  # Force cleanup after phase 1
            
            # Phase 2: Graph Augmentation  
            logger.info(f"🔄 Phase: augmentation, Progress: 50%, Memory: {get_memory_usage():.2f} MB")
            
            if is_incremental:
                logger.info("🔄 Phase 2: Incremental Graph Augmentation")
                new_entities = pipeline.get_new_entities_from_current_file()
                augmentation_result = pipeline._phase_2_incremental_augmentation(new_entities)
            else:
                logger.info("🔄 Phase 2: Full Graph Augmentation")
                augmentation_result = pipeline._phase_2_augmentation()
                
            if not augmentation_result['success']:
                logger.error(f"❌ Phase 2 FAILED: {augmentation_result.get('error')}")
                raise Exception(f"Phase 2 failed: {augmentation_result.get('error')}")
            
            logger.info(f"✅ Phase 2 COMPLETED - Graph Augmentation successful")
            logger.info(f"Memory after augmentation: {get_memory_usage():.2f} MB")
            memory_cleanup()  # Force cleanup after phase 2
            
            # Phase 3: Embedding Generation
            logger.info(f"🔄 Phase: embedding_generation, Progress: 75%, Memory: {get_memory_usage():.2f} MB")
            
            if is_incremental:
                logger.info("🔄 Phase 3: Incremental Embedding Generation")
                embedding_result = pipeline._phase_3_incremental_embeddings()
            else:
                logger.info("🔄 Phase 3: Full Embedding Generation")
                embedding_result = pipeline._phase_3_embedding_generation()
                
            if not embedding_result['success']:
                logger.error(f"❌ Phase 3 FAILED: {embedding_result.get('error')}")
                raise Exception(f"Phase 3 failed: {embedding_result.get('error')}")
            
            logger.info(f"✅ Phase 3 COMPLETED - Embedding Generation successful")
            logger.info(f"Memory after embedding generation: {get_memory_usage():.2f} MB")
            memory_cleanup()  # Force cleanup after phase 3
            
            # 🗄️ STEP 4: Store updated org graph
            logger.info(f"🔄 Phase: storage, Progress: 90%, Memory: {get_memory_usage():.2f} MB")
            
            logger.info("💾 Storing org-level graph in NeonDB")
            
            # Get updated processed files list
            updated_processed_files = existing_graph.get('processed_files', []) if existing_graph else []
            if file_id not in updated_processed_files:
                updated_processed_files.append(file_id)
            
            # Get org stats
            org_stats = pipeline.graph_manager.get_org_stats(updated_processed_files)
            
            # Store unified org graph
            import pickle
            graph_data = pickle.dumps({
                'graph': pipeline.graph_manager.graph,
                'entity_index': dict(pipeline.graph_manager.entity_index),
                'community_assignments': pipeline.graph_manager.community_assignments
            })
            
            version = (existing_graph.get('version', 0) + 1) if existing_graph else 1
            
            storage_result = storage.store_org_graph_sync(
                org_id=org_id,
                graph_data=graph_data,
                processed_files=updated_processed_files,
                version=version,
                last_file_added=file_id,
                stats=org_stats,
                user_id=user_id
            )
            
            if not storage_result.get("success"):
                logger.error(f"❌ Phase 4 FAILED: Org graph storage failed: {storage_result.get('error')}")
                raise Exception(f"Org graph storage failed: {storage_result.get('error')}")
            
            logger.info(f"✅ Phase 4 COMPLETED - Graph Storage successful")
            
            # Complete processing
            processing_time = time.time() - start_time
            final_memory = get_memory_usage()
            
            result_data = {
                'status': 'completed',
                'phase': 'completed',
                'progress': 100,
                'processing_time': processing_time,
                'memory_mb': final_memory,
                'file_id': file_id,
                'org_id': org_id,
                'results': {
                    'chunks_processed': len(noderag_chunks),
                    'incremental_mode': is_incremental,
                    'org_version': version,
                    'org_total_files': len(updated_processed_files),
                    'org_graph_nodes': org_stats.get("total_nodes", 0),
                    'org_graph_edges': org_stats.get("total_edges", 0)
                }
            }
            
            # Log final success (avoid update_state serialization issues)
            logger.info(f"✅ Phase: completed, Progress: 100%, Task completed successfully")
            
            if callback_url:
                send_webhook_safe(callback_url, "completed", {
                    "file_id": file_id,
                    "user_id": user_id,
                    "results": result_data["results"],
                    "processing_time": processing_time,
                    "celery_task_id": current_task.request.id
                })
            
            logger.info(f"✅ Celery task completed for file_id={file_id}, org_id={org_id} (v{version})")
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            
            # Final cleanup
            pipeline = None
            memory_cleanup()
            
            return result_data
            
    except SoftTimeLimitExceeded:
        logger.error(f"⏰ Task timed out for file_id={file_id}")
        
        # Log timeout error (avoid update_state serialization issues)
        logger.error(f"⏰ Phase: timeout, Task timed out for file_id={file_id}")
        
        if callback_url:
            send_webhook_safe(callback_url, "failed", {
                "file_id": file_id,
                "user_id": user_id,
                "error": "Task timed out"
            })
        
        raise
        
    except Exception as e:
        logger.error(f"❌ Celery task error for file_id={file_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Log general error (avoid update_state serialization issues)
        logger.error(f"❌ Phase: error, Task failed for file_id={file_id}: {str(e)}, Memory: {get_memory_usage():.2f} MB")
        
        if callback_url:
            send_webhook_safe(callback_url, "failed", {
                "file_id": file_id,
                "user_id": user_id,
                "error": str(e)
            })
        
        # Cleanup on error
        pipeline = None
        memory_cleanup()
        
        raise
    
    finally:
        # Aggressive cleanup to prevent memory accumulation
        if pipeline:
            # Clear pipeline components
            pipeline.graph_manager = None
            pipeline.llm_service = None
            pipeline.hnsw_service = None
            pipeline = None
        
        # Force multiple garbage collections
        for _ in range(3):
            gc.collect()
        
        # Clear any remaining references
        import sys
        if 'noderag_service' in locals():
            noderag_service = None
        if 'storage' in locals():
            storage = None
            
        final_memory = get_memory_usage()
        logger.info(f"🧹 Task cleanup complete - Final memory: {final_memory:.2f} MB")

@celery_app.task(name='src.queue.tasks.cleanup_task')
def cleanup_task():
    """Periodic cleanup task to free memory"""
    logger.info("🧹 Running periodic cleanup task")
    initial_memory = get_memory_usage()
    
    # Force garbage collection
    gc.collect()
    
    final_memory = get_memory_usage()
    logger.info(f"Cleanup completed: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
    
    return {
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "freed_mb": initial_memory - final_memory
    }

def send_webhook_safe(callback_url: str, status: str, data: Dict[str, Any]):
    """Send webhook notification safely (no exceptions)"""
    try:
        if not callback_url:
            return
            
        payload = {
            "status": status,
            "timestamp": time.time(),
            "data": data
        }
        
        response = requests.post(
            callback_url,
            json=payload,
            timeout=10,  # Shorter timeout for webhooks
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Webhook sent successfully: {status}")
        else:
            logger.warning(f"⚠️ Webhook failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Webhook error (ignored): {e}")

# Periodic cleanup every 10 minutes
celery_app.conf.beat_schedule = {
    'memory-cleanup': {
        'task': 'src.queue.tasks.cleanup_task',
        'schedule': 600.0,  # Every 10 minutes
    },
}

celery_app.conf.timezone = 'UTC'
#!/usr/bin/env python3
"""
NodeRAG API Service - Fixed version with proper indentation and org-level processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import uuid
import time
import requests
import threading
import asyncio
from typing import List, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NodeRAG components
from src.document_processing.indexing_pipeline import IndexingPipeline
from src.document_processing.document_loader import DocumentChunk, ProcessedDocument
from src.search.advanced_search import AdvancedSearchSystem
from src.config.settings import Config
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "DELETE", "OPTIONS", "PUT"])

# Global storage for processing status
processing_status = {}
processing_lock = threading.Lock()

# Org-level processing locks for concurrent protection
org_processing_locks = {}

def get_or_create_org_lock(org_id: str) -> threading.Lock:
    """Get or create a processing lock for an organization"""
    if org_id not in org_processing_locks:
        org_processing_locks[org_id] = threading.Lock()
    return org_processing_locks[org_id]

class NodeRAGService:
    def __init__(self):
        self.pipeline = None
        self.neon_storage = None
        self.advanced_search = None
    
    def get_pipeline(self):
        """Lazy initialization of pipeline"""
        if self.pipeline is None:
            self.pipeline = IndexingPipeline()
        return self.pipeline
    
    def get_neon_storage(self):
        """Lazy initialization of NeonDB storage"""
        if self.neon_storage is None:
            from src.storage.neon_storage import NeonDBStorage
            self.neon_storage = NeonDBStorage()
        return self.neon_storage
    
    def get_advanced_search(self):
        """Lazy initialization of AdvancedSearchSystem"""
        if self.advanced_search is None:
            pipeline = self.get_pipeline()
            
            # Check if graph has nodes directly without calling get_stats()
            # which might trigger PPR initialization
            graph = pipeline.graph_manager.graph
            if graph is None or graph.number_of_nodes() == 0:
                return None
                
            self.advanced_search = AdvancedSearchSystem(
                graph_manager=pipeline.graph_manager,
                hnsw_service=pipeline._get_hnsw_service(),
                llm_service=pipeline.llm_service
            )
        return self.advanced_search

noderag_service = NodeRAGService()

def send_webhook(callback_url: str, status: str, data: Dict[str, Any]):
    """Send webhook notification to main server"""
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
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"✅ Webhook sent successfully: {status}")
        else:
            logger.warning(f"⚠️ Webhook failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")

@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "noderag",
        "version": "1.0.0",
        "timestamp": time.time()
    })

@app.route("/api/v1/process-document", methods=["POST"])
def process_document():
    """Process document chunks through NodeRAG pipeline"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["org_id", "file_id", "user_id", "chunks"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        file_id = data["file_id"]
        user_id = data["user_id"]
        chunks = data["chunks"]
        callback_url = data.get("callback_url")
        
        logger.info(f"🚀 Starting NodeRAG processing: file_id={file_id}, org_id={org_id}, chunks={len(chunks)}")
        
        # Store initial status
        with processing_lock:
            processing_status[file_id] = {
                "status": "processing",
                "phase": "initialization",
                "progress": 0,
                "started_at": time.time(),
                "org_id": org_id,
                "user_id": user_id
            }
        
        # Send initial webhook
        if callback_url:
            send_webhook(callback_url, "processing", {
                "file_id": file_id,
                "user_id": user_id,
                "phase": "initialization",
                "progress": 0
            })
        
        # Start background processing
        def process_async():
            process_document_pipeline(org_id, file_id, user_id, chunks, callback_url)
        
        threading.Thread(target=process_async, daemon=True).start()
        
        return jsonify({
            "message": "Document processing started",
            "file_id": file_id,
            "status": "processing",
            "estimated_time": "2-5 minutes"
        }), 202
        
    except Exception as e:
        logger.error(f"❌ Process document error: {e}")
        return jsonify({"error": str(e)}), 500

def process_document_pipeline(org_id: str, file_id: str, user_id: str, chunks: List[Dict], callback_url: str = None):
    """UPDATED: Org-level incremental graph processing with concurrent protection"""
    
    # 🔒 CONCURRENT PROTECTION: Use org-level lock
    org_lock = get_or_create_org_lock(org_id)
    
    with org_lock:
        try:
            logger.info(f"🔒 Acquired org lock for {org_id}, processing file {file_id}")
            
            # 🔍 STEP 1: Check if org already has a graph
            storage = noderag_service.get_neon_storage()
            existing_graph = storage.load_org_graph_sync(org_id)
            
            if existing_graph:
                logger.info(f"📊 Found existing org graph: {len(existing_graph['processed_files'])} files processed")
                
                # Check if this file was already processed
                if file_id in existing_graph['processed_files']:
                    logger.info(f"⚠️ File {file_id} already processed for org {org_id}")
                    with processing_lock:
                        processing_status[file_id].update({
                            "status": "skipped",
                            "phase": "already_processed",
                            "progress": 100,
                            "message": "File already processed in org graph"
                        })
                    if callback_url:
                        send_webhook(callback_url, "skipped", {
                            "file_id": file_id,
                            "user_id": user_id,
                            "message": "File already processed"
                        })
                    return
                
                # Load existing graph into pipeline
                pipeline = noderag_service.get_pipeline()
                pipeline.set_incremental_mode(org_id, file_id, existing_graph['graph_data'])
                is_incremental = True
                
            else:
                logger.info(f"🆕 Creating first graph for org {org_id}")
                pipeline = noderag_service.get_pipeline()
                pipeline.set_incremental_mode(org_id, file_id)  # Fresh pipeline for new org
                is_incremental = False
            
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
                    'source': 'api_service',
                    'incremental_mode': is_incremental
                },
                total_tokens=sum(chunk.token_count for chunk in noderag_chunks)
            )
            
            # Phase 1: Graph Decomposition (always needed for new content)
            logger.info(f"🔄 Phase 1: Graph Decomposition ({'Incremental' if is_incremental else 'Initial'})")
            result = pipeline._phase_1_decomposition(processed_doc)
            if not result['success']:
                raise Exception(f"Phase 1 failed: {result.get('error')}")
            
            # Phase 2: Graph Augmentation (incremental or full)
            if is_incremental:
                logger.info("🔄 Phase 2: Incremental Graph Augmentation")
                new_entities = pipeline.get_new_entities_from_current_file()
                augmentation_result = pipeline._phase_2_incremental_augmentation(new_entities)
            else:
                logger.info("🔄 Phase 2: Full Graph Augmentation")
                augmentation_result = pipeline._phase_2_augmentation()
                
            if not augmentation_result['success']:
                raise Exception(f"Phase 2 failed: {augmentation_result.get('error')}")
            
            # Phase 3: Embedding Generation (incremental or full)
            if is_incremental:
                logger.info("🔄 Phase 3: Incremental Embedding Generation")
                embedding_result = pipeline._phase_3_incremental_embeddings()
            else:
                logger.info("🔄 Phase 3: Full Embedding Generation")
                embedding_result = pipeline._phase_3_embedding_generation()
                
            if not embedding_result['success']:
                raise Exception(f"Phase 3 failed: {embedding_result.get('error')}")
            
            # 🗄️ STEP 4: Store updated org graph
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
                raise Exception(f"Org graph storage failed: {storage_result.get('error')}")
            
            # Complete processing
            with processing_lock:
                processing_status[file_id].update({
                    "status": "completed",
                    "phase": "completed",
                    "progress": 100,
                    "completed_at": time.time(),
                    "results": {
                        "chunks_processed": len(noderag_chunks),
                        "incremental_mode": is_incremental,
                        "org_version": version,
                        "org_total_files": len(updated_processed_files),
                        "org_graph_nodes": org_stats.get("total_nodes", 0),
                        "org_graph_edges": org_stats.get("total_edges", 0)
                    }
                })
            
            if callback_url:
                send_webhook(callback_url, "completed", {
                    "file_id": file_id,
                    "user_id": user_id,
                    "results": processing_status[file_id]["results"]
                })
            
            logger.info(f"✅ Org-level processing completed for file_id={file_id}, org_id={org_id} (v{version})")
            
        except Exception as e:
            logger.error(f"❌ Org-level processing pipeline error: {e}")
            import traceback
            traceback.print_exc()
            
            # Update status: Failed
            with processing_lock:
                processing_status[file_id].update({
                    "status": "failed",
                    "phase": "failed",
                    "error": str(e),
                    "failed_at": time.time()
                })
            
            if callback_url:
                send_webhook(callback_url, "failed", {
                    "file_id": file_id,
                    "user_id": user_id,
                    "error": str(e)
                })

@app.route("/api/v1/generate-response", methods=["POST"])
def generate_response():
    """Generate a response using NodeRAG's advanced search system"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if "query" not in data or "org_id" not in data:
            return jsonify({"error": "Missing required fields: query, org_id"}), 400
        
        query = data["query"]
        org_id = data["org_id"]
        max_results = data.get("max_results", 5)
        max_tokens = data.get("max_tokens", 2048)
        temperature = data.get("temperature", 0.7)
        conversation_history = data.get("conversation_history", "")
        
        logger.info(f"🔍 NodeRAG query request: org_id={org_id}, query='{query[:100]}...'")
        
        # Check if org has any data by loading org graph first
        storage = noderag_service.get_neon_storage()
        existing_graph = storage.load_org_graph_sync(org_id)
        
        if not existing_graph:
            logger.warning(f"No graph data found for org {org_id}")
            return jsonify({
                "error": "No knowledge base found for organization",
                "message": f"No processed documents found for org {org_id}. Please process some documents first.",
                "org_id": org_id
            }), 404
        
        # Load the org graph into the pipeline
        pipeline = noderag_service.get_pipeline()
        logger.info(f"Loading org graph for {org_id} into search system")
        pipeline.graph_manager.load_from_data(existing_graph['graph_data'])
        
        # Reset and get the advanced search system with loaded graph
        noderag_service.advanced_search = None  # Reset to force reinitialization
        advanced_search = noderag_service.get_advanced_search()
        
        if not advanced_search:
            logger.error("Failed to initialize AdvancedSearchSystem after loading graph")
            return jsonify({"error": "Failed to initialize search system"}), 500
        
        # Set org_id for database vector search
        advanced_search._current_org_id = org_id
        
        # Perform the search and generate response
        search_result = advanced_search.answer_query(
            query=query,
            use_structured_prompt=True
        )
        
        if search_result.get('error'):
            logger.error(f"NodeRAG search failed: {search_result.get('answer', 'Unknown error')}")
            return jsonify({
                "error": "Search failed",
                "message": search_result.get('answer', 'Unknown error')
            }), 500
        
        logger.info(f"✅ NodeRAG response generated successfully for org {org_id}")
        
        return jsonify({
            "success": True,
            "response": search_result['answer'],
            "context": {
                "retrieved_nodes": search_result.get('retrieved_nodes', 0),
                "context_length": search_result.get('context_length', 0),
                "retrieval_metadata": search_result.get('retrieval_metadata', {}),
                "org_id": org_id,
                "query": query
            },
            "metadata": search_result.get('retrieval_metadata', {})
        })
        
    except Exception as e:
        logger.error(f"❌ Generate response error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route("/api/v1/delete-embeddings", methods=["DELETE"])
def delete_embeddings():
    """Delete embeddings for an organization or specific file"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if "org_id" not in data:
            return jsonify({"error": "Missing required field: org_id"}), 400
        
        org_id = data["org_id"]
        file_id = data.get("file_id")  # Optional - if not provided, delete all org embeddings
        
        logger.info(f"🗑️ Delete request: org_id={org_id}, file_id={file_id}")
        
        # Get storage service
        storage = noderag_service.get_neon_storage()
        
        if file_id:
            # Delete specific file embeddings
            result = storage.delete_file_data(org_id, file_id)
            
            if result.get("success"):
                logger.info(f"✅ Deleted embeddings for file {file_id} in org {org_id}")
                return jsonify({
                    "success": True,
                    "message": f"Deleted embeddings for file {file_id}",
                    "org_id": org_id,
                    "file_id": file_id,
                    "embeddings_deleted": result.get("deleted_count", 0),
                    "graphs_deleted": result.get("graphs_deleted", 0)
                })
            else:
                return jsonify({
                    "error": "Failed to delete file embeddings", 
                    "message": result.get("error", "Unknown error"),
                    "org_id": org_id,
                    "file_id": file_id
                }), 500
        else:
            # Delete all org embeddings - use existing delete_file_data for now
            # TODO: Add delete_org_embeddings method to storage
            logger.warning("Full org deletion not implemented yet, use file_id parameter")
            return jsonify({
                "error": "Full organization deletion not implemented",
                "message": "Please provide file_id parameter to delete specific file",
                "org_id": org_id
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Delete embeddings error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route("/api/v1/status/<file_id>", methods=["GET"])
def get_status(file_id: str):
    """Get processing status for a file"""
    try:
        with processing_lock:
            status = processing_status.get(file_id)
        
        if not status:
            return jsonify({"error": "File not found"}), 404
        
        return jsonify({
            "file_id": file_id,
            **status
        })
        
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get configuration
    host = Config.API_HOST
    port = Config.API_PORT
    debug = Config.API_DEBUG
    
    logger.info(f"🚀 Starting NodeRAG API Service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
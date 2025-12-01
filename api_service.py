#!/usr/bin/env python3
"""
NodeRAG API Service - Standalone microservice for graph-based document processing
Handles document processing, graph generation, and search operations
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
    """Background processing pipeline"""
    try:
        pipeline = noderag_service.get_pipeline()
        
        # Update status: Phase 1
        with processing_lock:
            processing_status[file_id].update({
                "phase": "graph_decomposition",
                "progress": 10
            })
        
        if callback_url:
            send_webhook(callback_url, "phase1_started", {
                "file_id": file_id,
                "user_id": user_id,
                "phase": "graph_decomposition",
                "progress": 10
            })
        
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
                'source': 'api_service'
            },
            total_tokens=sum(chunk.token_count for chunk in noderag_chunks)
        )
        
        # Phase 1: Graph Decomposition
        logger.info("🔄 Phase 1: Graph Decomposition")
        result = pipeline._phase_1_decomposition(processed_doc)
        if not result['success']:
            raise Exception(f"Phase 1 failed: {result.get('error')}")
        
        # Update status: Phase 2
        with processing_lock:
            processing_status[file_id].update({
                "phase": "graph_augmentation",
                "progress": 40
            })
        
        if callback_url:
            send_webhook(callback_url, "phase2_started", {
                "file_id": file_id,
                "user_id": user_id,
                "phase": "graph_augmentation",
                "progress": 40
            })
        
        # Phase 2: Graph Augmentation
        logger.info("🔄 Phase 2: Graph Augmentation")
        augmentation_result = pipeline._phase_2_augmentation()
        if not augmentation_result['success']:
            raise Exception(f"Phase 2 failed: {augmentation_result.get('error')}")
        
        # Update status: Phase 3
        with processing_lock:
            processing_status[file_id].update({
                "phase": "embedding_generation",
                "progress": 70
            })
        
        if callback_url:
            send_webhook(callback_url, "phase3_started", {
                "file_id": file_id,
                "user_id": user_id,
                "phase": "embedding_generation",
                "progress": 70
            })
        
        # Phase 3: Embedding Generation
        logger.info("🔄 Phase 3: Embedding Generation")
        embedding_result = pipeline._phase_3_embedding_generation()
        if not embedding_result['success']:
            raise Exception(f"Phase 3 failed: {embedding_result.get('error')}")
        
        # Update status: Storage
        with processing_lock:
            processing_status[file_id].update({
                "phase": "storage",
                "progress": 90
            })
        
        # Store in NeonDB
        logger.info("💾 Storing in NeonDB")
        storage = noderag_service.get_neon_storage()
        storage_result = storage.store_noderag_data(
            org_id=org_id,
            file_id=file_id,
            user_id=user_id,
            pipeline=pipeline
        )
        
        if not storage_result.get("success"):
            raise Exception(f"Storage failed: {storage_result.get('error')}")
        
        # Complete processing
        with processing_lock:
            processing_status[file_id].update({
                "status": "completed",
                "phase": "completed",
                "progress": 100,
                "completed_at": time.time(),
                "results": {
                    "chunks_processed": len(noderag_chunks),
                    "embeddings_stored": storage_result.get("embeddings_stored", 0),
                    "graph_nodes": storage_result.get("graph_nodes", 0)
                }
            })
        
        if callback_url:
            send_webhook(callback_url, "completed", {
                "file_id": file_id,
                "user_id": user_id,
                "results": processing_status[file_id]["results"]
            })
        
        logger.info(f"✅ NodeRAG processing completed for file_id={file_id}")
        
    except Exception as e:
        logger.error(f"❌ Processing pipeline error: {e}")
        
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

@app.route("/api/v1/search", methods=["POST"])
def search_documents():
    """Search documents using NodeRAG"""
    try:
        data = request.get_json()
        
        required_fields = ["org_id", "query"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        query = data["query"]
        top_k = data.get("top_k", 10)
        filters = data.get("filters", {})
        
        logger.info(f"🔍 NodeRAG advanced search: org_id={org_id}, query='{query[:50]}...'")
        
        # Use advanced search system for complete NodeRAG algorithm
        advanced_search = noderag_service.get_advanced_search()
        retrieval_result = advanced_search.search(
            query=query,
            k_final=top_k
        )
        
        # Convert retrieval result to search results format
        search_results = []
        for node_id in retrieval_result.final_nodes:
            node = advanced_search.graph_manager.get_node(node_id)
            if node and node.metadata.get('org_id') == org_id:  # Filter by org_id
                search_results.append({
                    'node_id': node_id,
                    'content': node.content,
                    'node_type': node.type.value if hasattr(node.type, 'value') else str(node.type),
                    'file_id': node.metadata.get('file_id', ''),
                    'user_id': node.metadata.get('user_id', ''),
                    'similarity_score': 1.0,  # Advanced search doesn't directly provide scores
                    'metadata': node.metadata
                })
        
        return jsonify({
            "query": query,
            "results": search_results,
            "count": len(search_results)
        })
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/inspect-data", methods=["POST"])
def inspect_data():
    """Inspect stored data in Neon DB for debugging"""
    try:
        data = request.get_json()
        org_id = data.get("org_id", "")
        
        storage = noderag_service.get_neon_storage()
        result = asyncio.run(inspect_data_async(storage, org_id))
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Inspect error: {e}")
        return jsonify({"error": str(e)}), 500

async def inspect_data_async(storage, org_id):
    """Async implementation of data inspection"""
    conn = await storage._get_connection()
    
    try:
        # Check what data exists for this org_id
        if org_id:
            embeddings_query = """
            SELECT file_id, org_id, node_type, COUNT(*) as count, 
                   MIN(created_at) as first_created, MAX(created_at) as last_created
            FROM noderag_embeddings 
            WHERE org_id = $1
            GROUP BY file_id, org_id, node_type
            ORDER BY last_created DESC
            LIMIT 20
            """
            rows = await conn.fetch(embeddings_query, org_id)
        else:
            all_query = """
            SELECT file_id, org_id, node_type, COUNT(*) as count,
                   MIN(created_at) as first_created, MAX(created_at) as last_created
            FROM noderag_embeddings 
            GROUP BY file_id, org_id, node_type
            ORDER BY last_created DESC
            LIMIT 20
            """
            rows = await conn.fetch(all_query)
        
        results = []
        for row in rows:
            results.append({
                "file_id": row['file_id'],
                "org_id": row['org_id'], 
                "node_type": row['node_type'],
                "count": row['count'],
                "first_created": str(row['first_created']),
                "last_created": str(row['last_created'])
            })
        
        return {
            "org_id_filter": org_id,
            "results": results,
            "total_groups": len(results)
        }
        
    finally:
        await conn.close()

@app.route("/api/v1/debug-db", methods=["GET"])
def debug_db():
    """Debug database contents"""
    try:
        storage = noderag_service.get_neon_storage()
        result = storage.inspect_all_data()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Debug error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/delete-file", methods=["DELETE"])
def delete_file():
    """Delete all data for a specific file"""
    try:
        data = request.get_json()
        
        required_fields = ["org_id", "file_id"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        file_id = data["file_id"]
        
        logger.info(f"🗑️ Deleting NodeRAG data: org_id={org_id}, file_id={file_id}")
        
        storage = noderag_service.get_neon_storage()
        delete_result = storage.delete_file_data(org_id=org_id, file_id=file_id)
        
        # Clean up processing status
        with processing_lock:
            if file_id in processing_status:
                del processing_status[file_id]
        
        return jsonify({
            "message": "File data deleted successfully",
            "file_id": file_id,
            "deleted_count": delete_result.get("deleted_count", 0)
        })
        
    except Exception as e:
        logger.error(f"❌ Delete error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/delete-embeddings", methods=["DELETE"])
def delete_embeddings():
    """Delete embeddings and graph data for multiple files (bulk delete)"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["org_id", "file_ids"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        file_ids = data["file_ids"]
        callback_url = data.get("callback_url")
        
        # Validate file_ids
        if not isinstance(file_ids, list) or len(file_ids) == 0:
            return jsonify({"error": "file_ids must be a non-empty list"}), 400
        
        # Generate deletion task ID
        deletion_id = str(uuid.uuid4())
        
        logger.info(f"🗑️ Starting bulk deletion: org_id={org_id}, file_count={len(file_ids)}, deletion_id={deletion_id}")
        
        # Store initial status
        with processing_lock:
            processing_status[deletion_id] = {
                "status": "deleting",
                "phase": "initialization",
                "progress": 0,
                "started_at": time.time(),
                "org_id": org_id,
                "file_ids": file_ids,
                "total_files": len(file_ids),
                "deleted_files": 0,
                "failed_files": 0,
                "operation": "bulk_delete"
            }
        
        # Send initial webhook
        if callback_url:
            send_webhook(callback_url, "deletion_started", {
                "deletion_id": deletion_id,
                "org_id": org_id,
                "file_count": len(file_ids),
                "phase": "initialization",
                "progress": 0
            })
        
        # Start background deletion
        def delete_async():
            bulk_delete_pipeline(org_id, file_ids, deletion_id, callback_url)
        
        threading.Thread(target=delete_async, daemon=True).start()
        
        return jsonify({
            "message": "Bulk deletion started",
            "deletion_id": deletion_id,
            "org_id": org_id,
            "file_count": len(file_ids),
            "status": "deleting",
            "estimated_time": f"{len(file_ids) * 2}-{len(file_ids) * 5} seconds"
        }), 202
        
    except Exception as e:
        logger.error(f"❌ Bulk delete error: {e}")
        return jsonify({"error": str(e)}), 500

def bulk_delete_pipeline(org_id: str, file_ids: List[str], deletion_id: str, callback_url: str = None):
    """Background bulk deletion pipeline"""
    try:
        storage = noderag_service.get_neon_storage()
        
        # Update status: Phase 1 - Database cleanup
        with processing_lock:
            processing_status[deletion_id].update({
                "phase": "database_cleanup",
                "progress": 10
            })
        
        if callback_url:
            send_webhook(callback_url, "phase1_database_cleanup", {
                "deletion_id": deletion_id,
                "org_id": org_id,
                "phase": "database_cleanup",
                "progress": 10
            })
        
        deleted_count = 0
        failed_count = 0
        deleted_files = []
        failed_files = []
        
        # Delete each file's data
        for i, file_id in enumerate(file_ids):
            try:
                logger.info(f"🗑️ Deleting file {i+1}/{len(file_ids)}: {file_id}")
                
                delete_result = storage.delete_file_data(org_id=org_id, file_id=file_id)
                
                if delete_result.get("success", True):
                    deleted_count += 1
                    deleted_files.append(file_id)
                    logger.info(f"✅ Deleted {delete_result.get('deleted_count', 0)} records for file_id={file_id}")
                else:
                    failed_count += 1
                    failed_files.append(file_id)
                    logger.error(f"❌ Failed to delete file_id={file_id}: {delete_result.get('error', 'Unknown error')}")
                
                # Update progress
                progress = 10 + (70 * (i + 1) / len(file_ids))
                with processing_lock:
                    processing_status[deletion_id].update({
                        "progress": int(progress),
                        "deleted_files": deleted_count,
                        "failed_files": failed_count
                    })
                
                # Clean up processing status for this file
                with processing_lock:
                    if file_id in processing_status:
                        del processing_status[file_id]
                
            except Exception as e:
                failed_count += 1
                failed_files.append(file_id)
                logger.error(f"❌ Exception deleting file_id={file_id}: {e}")
        
        # Update status: Phase 2 - Memory cleanup
        with processing_lock:
            processing_status[deletion_id].update({
                "phase": "memory_cleanup",
                "progress": 80
            })
        
        if callback_url:
            send_webhook(callback_url, "phase2_memory_cleanup", {
                "deletion_id": deletion_id,
                "org_id": org_id,
                "phase": "memory_cleanup",
                "progress": 80,
                "deleted_files": deleted_count,
                "failed_files": failed_count
            })
        
        # Clean up in-memory graph data if available
        pipeline = noderag_service.get_pipeline()
        if pipeline and pipeline.graph_manager:
            try:
                # Remove nodes associated with deleted files from in-memory graph
                nodes_to_remove = []
                for node_id in pipeline.graph_manager.graph.nodes():
                    node = pipeline.graph_manager.get_node(node_id)
                    if node and node.metadata.get('org_id') == org_id and node.metadata.get('file_id') in deleted_files:
                        nodes_to_remove.append(node_id)
                
                for node_id in nodes_to_remove:
                    pipeline.graph_manager.graph.remove_node(node_id)
                
                logger.info(f"🧹 Removed {len(nodes_to_remove)} nodes from in-memory graph")
                
                # Update HNSW index if available
                hnsw_service = pipeline._get_hnsw_service()
                if hnsw_service:
                    # Note: HNSW doesn't support deletion, would need rebuild for production
                    logger.info("⚠️ HNSW index may contain stale data - consider rebuilding")
                
            except Exception as e:
                logger.warning(f"⚠️ Memory cleanup warning: {e}")
        
        # Complete deletion
        with processing_lock:
            processing_status[deletion_id].update({
                "status": "completed",
                "phase": "completed",
                "progress": 100,
                "completed_at": time.time(),
                "deleted_files": deleted_count,
                "failed_files": failed_count,
                "results": {
                    "total_files": len(file_ids),
                    "successfully_deleted": deleted_files,
                    "failed_deletions": failed_files,
                    "deleted_count": deleted_count,
                    "failed_count": failed_count
                }
            })
        
        if callback_url:
            send_webhook(callback_url, "deletion_completed", {
                "deletion_id": deletion_id,
                "org_id": org_id,
                "results": processing_status[deletion_id]["results"]
            })
        
        logger.info(f"✅ Bulk deletion completed: deletion_id={deletion_id}, deleted={deleted_count}, failed={failed_count}")
        
    except Exception as e:
        logger.error(f"❌ Bulk deletion pipeline error: {e}")
        
        # Update status: Failed
        with processing_lock:
            processing_status[deletion_id].update({
                "status": "failed",
                "phase": "failed",
                "error": str(e),
                "failed_at": time.time()
            })
        
        if callback_url:
            send_webhook(callback_url, "deletion_failed", {
                "deletion_id": deletion_id,
                "org_id": org_id,
                "error": str(e)
            })

@app.route("/api/v1/delete-status/<deletion_id>", methods=["GET"])
def get_deletion_status(deletion_id: str):
    """Get deletion status for a bulk deletion operation"""
    try:
        with processing_lock:
            status = processing_status.get(deletion_id)
        
        if not status:
            return jsonify({"error": "Deletion operation not found"}), 404
        
        if status.get("operation") != "bulk_delete":
            return jsonify({"error": "Invalid operation type"}), 400
        
        return jsonify({
            "deletion_id": deletion_id,
            **status
        })
        
    except Exception as e:
        logger.error(f"❌ Deletion status check error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/generate-response", methods=["POST"])
def generate_response():
    """Generate a response using NodeRAG's advanced search and LLM - same approach as web UI"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["org_id", "query"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        query = data["query"]
        conversation_history = data.get("conversation_history", "")
        max_tokens = data.get("max_tokens", 2048)
        temperature = data.get("temperature", 0.7)
        
        logger.info(f"🤖 Generating NodeRAG response for query: '{query[:50]}...'")
        
        # 🧠 AGENTIC ENHANCEMENT: Detect if this is a knowledge discovery query
        knowledge_discovery_keywords = [
            "data you have", "data u have", "tell about", "what do you know", 
            "what information", "what data", "overview", "summary", "available data",
            "knowledge base", "tell me about", "what's in", "content you have"
        ]
        
        is_knowledge_discovery = any(keyword in query.lower() for keyword in knowledge_discovery_keywords)
        
        if is_knowledge_discovery:
            logger.info("🧠 Detected knowledge discovery query - using agentic exploration")
        
        # Check if we have data in memory graph first
        advanced_search = noderag_service.get_advanced_search()
        
        # If no data in memory graph, use Neon storage search directly
        if advanced_search is None:
            logger.info("📊 No data in memory graph, using Neon storage search")
            storage = noderag_service.get_neon_storage()
            
            # 🔍 AGENTIC EXPLORATION: For knowledge discovery, use smart multi-query approach
            if is_knowledge_discovery:
                logger.info("🔍 Performing agentic knowledge exploration")
                
                # Get overview nodes (H, O types) with high priority
                overview_results = storage.search_noderag_data(
                    org_id=org_id,
                    query="overview summary main topics key information",
                    top_k=15,
                    filters={"node_type": "H"}
                )
                
                # Get key entities (N types)
                entity_results = storage.search_noderag_data(
                    org_id=org_id, 
                    query="companies organizations entities names",
                    top_k=10,
                    filters={"node_type": "N"}
                )
                
                # Get semantic chunks (S types) for diverse content
                semantic_results = storage.search_noderag_data(
                    org_id=org_id,
                    query=query,
                    top_k=15,
                    filters={"node_type": "S"}
                )
                
                # Combine results intelligently
                search_results = []
                
                # Prioritize overview content
                search_results.extend(overview_results)
                
                # Add key entities
                added_nodes = {r['node_id'] for r in search_results}
                for result in entity_results:
                    if result['node_id'] not in added_nodes:
                        search_results.append(result)
                        added_nodes.add(result['node_id'])
                
                # Add semantic content
                for result in semantic_results:
                    if result['node_id'] not in added_nodes and len(search_results) < 25:
                        search_results.append(result)
                        added_nodes.add(result['node_id'])
                
                logger.info(f"🧠 Agentic exploration found: {len(overview_results)} overview + {len(entity_results)} entities + {len(semantic_results)} semantic = {len(search_results)} total")
                
            else:
                # Standard search for specific queries
                search_results = storage.search_noderag_data(
                    org_id=org_id,
                    query=query,
                    top_k=20
                )
            
            if not search_results:
                return jsonify({
                    "query": query,
                    "response": "I couldn't find relevant information in your organization's knowledge base for this query.",
                    "sources": [],
                    "node_types": {},
                    "confidence": 0.0,
                    "context_used": 0,
                    "search_results": 0,
                    "algorithm_used": "Neon Storage Vector Search"
                })
            
            # Build context from search results
            filtered_context_parts = []
            source_files = set()
            node_types = {}
            
            logger.info(f"🔍 Processing {len(search_results)} search results:")
            for i, result in enumerate(search_results):
                # Log the first few results for debugging
                if i < 3:
                    logger.info(f"  Result {i}: node_type={result['node_type']}, file_id={result.get('file_id', 'N/A')}, content_preview={result['content'][:100]}...")
                
                # Build context with structured format
                filtered_context_parts.append(f"[{result['node_type']}] {result['content']}")
                
                # Track sources and node types
                file_id = result.get('file_id', '')
                if file_id:
                    source_files.add(file_id)
                
                node_type = result['node_type']
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1
            
            retrieved_info = "\n\n".join(filtered_context_parts)
            logger.info(f"📝 Built context with {len(retrieved_info)} characters from {len(filtered_context_parts)} nodes")
            
            # Generate answer using filtered context
            llm_service = noderag_service.get_pipeline().llm_service
            
            # 🧠 AGENTIC PROMPTING: Use different prompts for knowledge discovery vs specific queries
            if is_knowledge_discovery:
                # Cognitive prompt for exploring and summarizing available knowledge
                answer_prompt = f"""You are an intelligent knowledge assistant exploring the user's data repository. The user wants to understand what information is available in their knowledge base.

CONTEXT INFORMATION:
{retrieved_info}

USER QUERY: {query}

TASK: Provide a comprehensive, well-structured overview of the available knowledge. Focus on:

1. **Main Topics & Themes**: What are the primary subjects covered?
2. **Key Entities**: What organizations, companies, people, or products are mentioned?
3. **Important Concepts**: What key processes, programs, or relationships are described?
4. **Data Scope**: What types of information and knowledge areas are represented?

Structure your response to give the user a clear understanding of their knowledge base content. Be specific about what information is available and how it could be useful.

RESPONSE:"""
            else:
                # Standard NodeRAG prompt for specific queries
                answer_prompt = llm_service.prompt_manager.answer_generation.format(
                    info=retrieved_info,
                    query=query
                )
            
            logger.info(f"🤖 Generating {'agentic' if is_knowledge_discovery else 'standard'} answer with prompt length: {len(answer_prompt)} characters")
            
            # Generate initial answer
            final_response = llm_service._chat_completion(
                prompt=answer_prompt,
                temperature=0.6 if is_knowledge_discovery else temperature,  # Slightly lower temp for exploration
                max_tokens=max_tokens
            )
            
            logger.info(f"✅ Generated response length: {len(final_response)} characters")
            logger.info(f"📄 Response preview: {final_response[:200]}...")
            
            # If we have conversation history, enhance the answer
            if conversation_history and final_response:
                enhanced_prompt = f"""Given the previous conversation context and the answer below, please refine the answer to be more contextual and helpful:

Previous conversation:
{conversation_history}

Current question: {query}

Generated answer: {final_response}

Please provide a refined response that takes into account the conversation history:"""
                
                final_response = llm_service._chat_completion(
                    prompt=enhanced_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Calculate confidence based on retrieval quality
            confidence = min(0.9, 0.6 + (len(search_results) * 0.02))
            
            logger.info(f"✅ Generated Neon storage response with {len(search_results)} sources")
            
            return jsonify({
                "query": query,
                "response": final_response.strip(),
                "sources": list(source_files),
                "node_types": node_types,
                "confidence": confidence,
                "context_used": len(search_results),
                "search_results": len(search_results),
                "context_length": len(retrieved_info),
                "algorithm_used": "Agentic Knowledge Discovery" if is_knowledge_discovery else "Neon Storage Vector Search",
                "agentic_mode": is_knowledge_discovery,
                "retrieval_metadata": {
                    "storage_search_count": len(search_results),
                    "unique_sources": len(source_files),
                    "node_type_distribution": node_types,
                    "exploration_strategy": "multi_query_node_type_prioritization" if is_knowledge_discovery else "standard_vector_search"
                }
            })
        
        # If we have data in memory, use the advanced search system
        # First, perform search with optimal settings (matching web UI)
        retrieval_result = advanced_search.search(
            query=query,
            k_hnsw=15,
            k_final=30,
            entity_nodes_limit=15,
            relationship_nodes_limit=40,
            high_level_nodes_limit=15
        )
        
        # Filter nodes by org_id BEFORE generating answer
        org_filtered_nodes = []
        filtered_context_parts = []
        source_files = set()
        node_types = {}
        
        for node_id in retrieval_result.final_nodes:
            node = advanced_search.graph_manager.get_node(node_id)
            if node and node.metadata.get('org_id') == org_id:
                org_filtered_nodes.append(node_id)
                
                # Build context with structured format
                filtered_context_parts.append(f"[{node.type.value}] {node.content}")
                
                # Track sources and node types
                file_id = node.metadata.get('file_id', '')
                if file_id:
                    source_files.add(file_id)
                
                node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1
        
        if not org_filtered_nodes:
            return jsonify({
                "query": query,
                "response": "I couldn't find relevant information in your organization's knowledge base for this query.",
                "sources": [],
                "node_types": {},
                "confidence": 0.0,
                "context_used": 0,
                "search_results": 0,
                "algorithm_used": "NodeRAG Advanced Search (HNSW + PPR + Entity Matching)"
            })
        
        # Generate answer using filtered context (matching web UI approach)
        retrieved_info = "\n\n".join(filtered_context_parts)
        
        # Use the same prompt as the advanced search system
        llm_service = noderag_service.get_pipeline().llm_service
        answer_prompt = llm_service.prompt_manager.answer_generation.format(
            info=retrieved_info,
            query=query
        )
        
        # Generate initial answer
        final_response = llm_service._chat_completion(
            prompt=answer_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # If we have conversation history, enhance the answer
        if conversation_history and final_response:
            enhanced_prompt = f"""Given the previous conversation context and the answer below, please refine the answer to be more contextual and helpful:

Previous conversation:
{conversation_history}

Current question: {query}

Generated answer: {final_response}

Please provide a refined response that takes into account the conversation history:"""
            
            final_response = llm_service._chat_completion(
                prompt=enhanced_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Calculate confidence based on retrieval quality
        confidence = min(0.9, 0.6 + (len(org_filtered_nodes) * 0.02))  # Scale with number of sources
        
        logger.info(f"✅ Generated NodeRAG response with {len(org_filtered_nodes)} org-filtered sources")
        
        return jsonify({
            "query": query,
            "response": final_response.strip(),
            "sources": list(source_files),
            "node_types": node_types,
            "confidence": confidence,
            "context_used": len(org_filtered_nodes),
            "search_results": len(org_filtered_nodes),
            "context_length": len(retrieved_info),
            "algorithm_used": "NodeRAG Advanced Search (HNSW + PPR + Entity Matching)",
            "retrieval_metadata": retrieval_result.search_metadata,
            "search_components": {
                "hnsw_results": len(retrieval_result.hnsw_results),
                "exact_matches": len(retrieval_result.accurate_results), 
                "ppr_results": len(retrieval_result.ppr_results),
                "entity_nodes": len(retrieval_result.entity_nodes),
                "relationship_nodes": len(retrieval_result.relationship_nodes),
                "high_level_nodes": len(retrieval_result.high_level_nodes)
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Response generation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get configuration
    host = Config.API_HOST
    port = Config.API_PORT
    debug = Config.API_DEBUG
    
    logger.info(f"🚀 Starting NodeRAG API Service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
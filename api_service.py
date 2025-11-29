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
        self.neo4j_storage = None
        self.advanced_search = None
    
    def get_pipeline(self):
        """Lazy initialization of pipeline"""
        if self.pipeline is None:
            self.pipeline = IndexingPipeline()
        return self.pipeline
    
    def get_neo4j_storage(self):
        """Lazy initialization of Neo4j storage"""
        if self.neo4j_storage is None:
            from src.storage.neo4j_storage import Neo4jStorage
            self.neo4j_storage = Neo4jStorage()
        return self.neo4j_storage
    
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
            print("⚠️ No callback URL provided, skipping webhook")
            return
            
        payload = {
            "status": status,
            "timestamp": time.time(),
            "data": data
        }
        
        print(f"📤 Sending webhook to {callback_url}")
        print(f"   Status: {status}")
        print(f"   Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            callback_url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"✅ Webhook sent successfully: {status}")
            logger.info(f"✅ Webhook sent successfully: {status}")
        else:
            print(f"⚠️ Webhook failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            logger.warning(f"⚠️ Webhook failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        logger.error(f"❌ Webhook error: {e}")

def send_webhook_direct(callback_url: str, data: Dict[str, Any]):
    """Send webhook data directly without nesting in 'data' field"""
    try:
        if not callback_url:
            print("⚠️ No callback URL provided, skipping webhook")
            return
        
        print(f"📤 Sending direct webhook to {callback_url}")
        print(f"   Payload: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            callback_url,
            json=data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print(f"✅ Direct webhook sent successfully")
            logger.info(f"✅ Direct webhook sent successfully")
        else:
            print(f"⚠️ Direct webhook failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            logger.warning(f"⚠️ Direct webhook failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Direct webhook error: {e}")
        logger.error(f"❌ Direct webhook error: {e}")

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
        
        # Use the optimized index_document method instead of calling phases separately
        print(f"🚀 Processing document with {len(noderag_chunks)} chunks...")
        
        # Create a temporary file for the optimized pipeline
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            # Write all chunk content to temp file
            content = "\n\n".join([chunk.content for chunk in noderag_chunks])
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Use the optimized async pipeline
            pipeline_result = pipeline.index_document(temp_file_path)
            if not pipeline_result['success']:
                raise Exception(f"Pipeline failed: {pipeline_result.get('error')}")
            
            print(f"✅ Pipeline completed successfully!")
            
        finally:
            # Clean up temp file
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Update status: Storage
        with processing_lock:
            processing_status[file_id].update({
                "phase": "storage",
                "progress": 90
            })
        
        # Store in NeonDB
        print("💾 Storing data in Neo4j...")
        storage = noderag_service.get_neo4j_storage()
        
        # Get token usage from pipeline's LLM service
        llm_stats = pipeline.llm_service.get_usage_stats()
        
        storage_result = storage.store_noderag_data(
            org_id=org_id,
            file_id=file_id,
            user_id=user_id,
            pipeline=pipeline,
            input_tokens=llm_stats.get('input_tokens', 0),
            output_tokens=llm_stats.get('output_tokens', 0),
            api_calls=llm_stats.get('api_calls', 0)
        )
        
        if storage_result.get("success"):
            print(f"✅ Database storage complete: {storage_result.get('embeddings_stored', 0)} embeddings stored")
        else:
            print(f"❌ Database storage failed: {storage_result.get('error', 'Unknown error')}")
        
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
                    "chunks_processed": pipeline_result.get("chunks_processed", len(noderag_chunks)),
                    "embeddings_stored": storage_result.get("embeddings_stored", 0),
                    "graph_nodes": storage_result.get("graph_nodes", 0),
                    "processing_time": pipeline_result.get("processing_time", 0),
                    "graph_stats": pipeline_result.get("graph_stats", {}),
                    "phase_timing": pipeline_result.get("phase_timing", {})
                }
            })
        
        if callback_url:
            send_webhook(callback_url, "completed", {
                "file_id": file_id,
                "user_id": user_id,
                "results": processing_status[file_id]["results"]
            })
        
        print(f"\n🎉 NodeRAG processing completed for file_id={file_id}")
        print(f"📊 Final Results:")
        print(f"   • Processing Time: {pipeline_result.get('processing_time', 0):.2f}s")
        
        # Graph verification options
        print("\n" + "="*60)
        print("🕸️ GRAPH VERIFICATION & VISUALIZATION READY!")
        print("🔍 Quick verification: python verify_graph.py")
        print("📊 Analytics dashboard: python simple_graph_viewer.py")
        print("🌐 Visual network graph:")
        try:
            from visual_graph import start_network_graph
            visual_url = start_network_graph()
            print(f"   {visual_url}")
            print("   ⚡ Interactive network with node filtering and layouts")
        except Exception as e:
            print(f"   ⚠️ Visual graph issue: {e}")
            print("   Run: python visual_graph.py")
        print("="*60)
        print(f"   • Total Nodes: {pipeline_result.get('graph_stats', {}).get('total_nodes', 0)}")
        print(f"   • Total Edges: {pipeline_result.get('graph_stats', {}).get('total_edges', 0)}")
        print(f"   • Embeddings Stored: {storage_result.get('embeddings_stored', 0)}")
        
        # Print LLM usage statistics
        if hasattr(pipeline, 'llm_service'):
            pipeline.llm_service.print_usage_summary()
        
        # Print phase timing if available
        if 'phase_timing' in pipeline_result:
            timing = pipeline_result['phase_timing']
            print(f"\n⏱️  Phase Performance:")
            print(f"   • Decomposition: {timing.get('phase1_time', 0):.2f}s")
            print(f"   • Augmentation:  {timing.get('phase2_time', 0):.2f}s")
            print(f"   • Embeddings:    {timing.get('phase3_time', 0):.2f}s")
        
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
    """Enhanced search using both in-memory graph and Neo4j storage"""
    try:
        data = request.get_json()
        
        required_fields = ["org_id", "query"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        query = data["query"]
        top_k = data.get("top_k", 20)
        filters = data.get("filters", {})
        search_strategy = data.get("strategy", "hybrid")  # hybrid, memory, storage
        
        logger.info(f"🔍 Enhanced search: org_id={org_id}, query='{query[:50]}...', strategy={search_strategy}")
        
        search_results = []
        metadata = {"sources": [], "strategies_used": []}
        
        # Strategy 1: Try in-memory advanced search first
        if search_strategy in ["hybrid", "memory"]:
            try:
                advanced_search = noderag_service.get_advanced_search()
                if advanced_search is not None:
                    logger.info("🧠 Using in-memory advanced search")
                    retrieval_result = advanced_search.search(
                        query=query,
                        k_final=top_k,
                        k_hnsw=15,
                        entity_nodes_limit=10,
                        relationship_nodes_limit=15,
                        high_level_nodes_limit=10
                    )
                    
                    # Convert retrieval result to search results format
                    for node_id in retrieval_result.final_nodes:
                        node = advanced_search.graph_manager.get_node(node_id)
                        if node and node.metadata.get('org_id') == org_id:
                            search_results.append({
                                'node_id': node_id,
                                'content': node.content,
                                'node_type': node.type.value if hasattr(node.type, 'value') else str(node.type),
                                'file_id': node.metadata.get('file_id', ''),
                                'user_id': node.metadata.get('user_id', ''),
                                'similarity_score': 0.95,  # High score for advanced search
                                'metadata': node.metadata,
                                'source': 'memory_graph'
                            })
                    
                    metadata["strategies_used"].append("in_memory_advanced_search")
                    metadata["memory_results"] = len(search_results)
                    logger.info(f"🧠 Memory search found {len(search_results)} results")
                    
            except Exception as e:
                logger.warning(f"⚠️ Memory search failed: {e}")
        
        # Strategy 2: Use Neo4j storage search (especially if memory search failed or insufficient results)
        if search_strategy in ["hybrid", "storage"] and len(search_results) < top_k:
            try:
                logger.info("💾 Using Neo4j storage search")
                storage = noderag_service.get_neo4j_storage()
                
                # Multi-strategy storage search
                storage_results = []
                
                # 1. Semantic search
                semantic_results = storage.search_noderag_data(
                    org_id=org_id,
                    query=query,
                    top_k=min(15, top_k),
                    filters={**filters, "node_type": "S"} if filters else {"node_type": "S"}
                )
                storage_results.extend(semantic_results)
                
                # 2. Entity search (if not enough semantic results)
                if len(semantic_results) < 10:
                    entity_results = storage.search_noderag_data(
                        org_id=org_id,
                        query=query,
                        top_k=8,
                        filters={**filters, "node_type": "N"} if filters else {"node_type": "N"}
                    )
                    storage_results.extend(entity_results)
                
                # 3. High-level overview search 
                overview_results = storage.search_noderag_data(
                    org_id=org_id,
                    query=query,
                    top_k=5,
                    filters={**filters, "node_type": "H"} if filters else {"node_type": "H"}
                )
                storage_results.extend(overview_results)
                
                # Convert storage results and avoid duplicates
                existing_node_ids = {r['node_id'] for r in search_results}
                
                for result in storage_results:
                    if result['node_id'] not in existing_node_ids and len(search_results) < top_k:
                        search_results.append({
                            'node_id': result['node_id'],
                            'content': result['content'],
                            'node_type': result['node_type'],
                            'file_id': result['file_id'],
                            'user_id': result['user_id'],
                            'similarity_score': result['similarity_score'],
                            'metadata': result['metadata'],
                            'source': 'neo4j_storage',
                            'token_info': result.get('token_info', {})
                        })
                        existing_node_ids.add(result['node_id'])
                
                metadata["strategies_used"].append("neo4j_vector_search")
                metadata["storage_results"] = len(storage_results)
                logger.info(f"💾 Storage search found {len(storage_results)} results")
                
            except Exception as e:
                logger.warning(f"⚠️ Storage search failed: {e}")
        
        # Sort by similarity score
        search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Limit to top_k
        search_results = search_results[:top_k]
        
        # Collect metadata
        file_ids = list(set([r['file_id'] for r in search_results if r['file_id']]))
        node_types = {}
        for result in search_results:
            node_type = result['node_type']
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        metadata.update({
            "file_ids": file_ids,
            "node_types": node_types,
            "total_results": len(search_results),
            "search_strategy": search_strategy
        })
        
        logger.info(f"✅ Enhanced search completed: {len(search_results)} results, strategies: {metadata['strategies_used']}")
        
        return jsonify({
            "query": query,
            "results": search_results,
            "count": len(search_results),
            "metadata": metadata,
            "search_strategy": search_strategy
        })
        
    except Exception as e:
        logger.error(f"❌ Enhanced search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/inspect-data", methods=["POST"])
def inspect_data():
    """Inspect stored data in Neon DB for debugging"""
    try:
        data = request.get_json()
        org_id = data.get("org_id", "")
        
        storage = noderag_service.get_neo4j_storage()
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
        storage = noderag_service.get_neo4j_storage()
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
        
        storage = noderag_service.get_neo4j_storage()
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

@app.route("/api/v1/generate-response", methods=["POST"])
def generate_response():
    """Generate a response using NodeRAG's advanced search and LLM with comprehensive token tracking"""
    start_time = time.time()
    input_tokens = 0
    output_tokens = 0
    api_calls = 0
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["org_id", "query"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        org_id = data["org_id"]
        query = data["query"]
        user_id = data.get("user_id", "api_user")  # Track user for analytics
        conversation_history = data.get("conversation_history", "")
        max_tokens = data.get("max_tokens", 2048)
        temperature = data.get("temperature", 0.7)
        
        # Validate input parameters
        if not query.strip():
            return jsonify({"error": "Query cannot be empty"}), 400
        if len(query) > 5000:
            return jsonify({"error": "Query too long (max 5000 characters)"}), 400
        if max_tokens < 100 or max_tokens > 4096:
            return jsonify({"error": "max_tokens must be between 100 and 4096"}), 400
        if temperature < 0.0 or temperature > 2.0:
            return jsonify({"error": "temperature must be between 0.0 and 2.0"}), 400
        
        logger.info(f"📥 INCOMING REQUEST: org_id={org_id}, user_id={user_id}, query='{query}', max_tokens={max_tokens}, temperature={temperature}")
        
        # Diagnostic logging for Neo4j connection
        neo4j_uri = os.getenv("NEO4J_URI", "NOT_SET")
        neo4j_user = os.getenv("NEO4J_USERNAME", "NOT_SET")
        neo4j_pass = "SET" if os.getenv("NEO4J_PASSWORD") else "NOT_SET"
        logger.info(f"🔧 NEO4J CONFIG: URI={neo4j_uri[:20]}..., USER={neo4j_user}, PASS={neo4j_pass}")
        
        logger.info(f"🤖 Generating NodeRAG response for org_id: {org_id}, user_id: {user_id}, query: '{query[:50]}...'")
        
        # Get LLM service for token tracking
        llm_service = noderag_service.get_pipeline().llm_service
        initial_input_tokens = llm_service.total_input_tokens
        initial_output_tokens = llm_service.total_output_tokens
        initial_api_calls = llm_service.api_calls_count
        
        # 🧠 AGENTIC ENHANCEMENT: Detect if this is a knowledge discovery query or greeting
        knowledge_discovery_keywords = [
            "data you have", "data u have", "tell about", "what do you know", 
            "what information", "what data", "overview", "summary", "available data",
            "knowledge base", "tell me about", "what's in", "content you have"
        ]
        
        greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"]
        
        is_knowledge_discovery = any(keyword in query.lower() for keyword in knowledge_discovery_keywords)
        is_greeting = query.lower().strip() in greeting_keywords or len(query.strip()) <= 3
        
        if is_knowledge_discovery:
            logger.info("🧠 Detected knowledge discovery query - using agentic exploration")
        elif is_greeting:
            logger.info("👋 Detected greeting query")
            
            # For greetings, provide a helpful response and try to show available data
            try:
                storage = noderag_service.get_neo4j_storage()
                
                # Quick check if any data exists for this org
                quick_search = storage.search_noderag_data(
                    org_id=org_id,
                    query="overview summary",
                    top_k=3
                )
                
                if quick_search:
                    greeting_response = f"Hello! I'm your AI assistant with access to your organization's knowledge base. I have information from {len(set(r.get('file_id', '') for r in quick_search if r.get('file_id')))} documents. You can ask me questions like:\n\n• 'What data do you have available?'\n• 'Give me an overview of the information'\n• 'Tell me about [specific topic]'\n\nHow can I help you today?"
                else:
                    greeting_response = "Hello! I'm your AI assistant. It looks like your knowledge base is empty or I don't have access to any documents for your organization. Please upload some documents first, then I'll be able to help answer questions about your content. How can I assist you?"
                
                # Track the greeting interaction
                session_input_tokens = len(query) // 4  # Rough estimate
                session_output_tokens = len(greeting_response) // 4
                processing_time = time.time() - start_time
                
                try:
                    storage.store_api_usage(
                        org_id=org_id,
                        user_id=user_id,
                        endpoint="generate-response-greeting",
                        input_tokens=session_input_tokens,
                        output_tokens=session_output_tokens,
                        total_tokens=session_input_tokens + session_output_tokens,
                        api_calls=0,
                        metadata={"greeting": True, "data_available": len(quick_search) > 0}
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store greeting usage: {e}")
                
                response_data = {
                    "query": query,
                    "response": greeting_response,
                    "sources": [],
                    "node_types": {},
                    "confidence": 1.0,  # High confidence for greetings
                    "context_used": 0,
                    "search_results": len(quick_search),
                    "context_length": 0,
                    "algorithm_used": "Greeting Handler",
                    "processing_time": round(processing_time, 2),
                    "token_usage": {
                        "input_tokens": session_input_tokens,
                        "output_tokens": session_output_tokens,
                        "total_tokens": session_input_tokens + session_output_tokens,
                        "api_calls": 0
                    },
                    "quality_metrics": {
                        "quality_score": 1.0,
                        "response_length_score": 1.0,
                        "source_diversity_score": 0.0,
                        "retrieval_coverage": 0.0,
                        "processing_efficiency": 1.0,
                        "token_efficiency": session_output_tokens / max(session_input_tokens, 1)
                    },
                    "greeting_mode": True,
                    "data_available": len(quick_search) > 0
                }
                
                logger.info(f"📤 GREETING RESPONSE SENT: {json.dumps(response_data, indent=2)}")
                return jsonify(response_data)
            
            except Exception as e:
                logger.warning(f"⚠️ Greeting handler error: {e}")
                # Fall through to normal processing if greeting handler fails
        
        # PRODUCTION FIX: Always use Neo4j storage for consistent, reliable data access
        # This ensures all instances serve the same data regardless of in-memory state
        logger.info("📊 Using Neo4j storage search for production reliability")
        storage = noderag_service.get_neo4j_storage()
        
        # 🔍 AGENTIC EXPLORATION: For knowledge discovery, use smart multi-query approach
        if is_knowledge_discovery:
            logger.info("🔍 Performing optimized agentic knowledge exploration")
            
            # Get overview nodes (H, O types) - prioritize high-level summaries
            overview_results = storage.search_noderag_data(
                org_id=org_id,
                query="overview summary main topics key information executive summary conclusions",
                top_k=20,  # Increased for better coverage
                filters={"node_type": "H"}
            )
            
            # Get key entities (N types) - focus on important names and organizations
            entity_results = storage.search_noderag_data(
                org_id=org_id, 
                query="companies organizations entities names people products services key players",
                top_k=15,
                filters={"node_type": "N"}
            )
            
            # Get semantic chunks (S types) - get diverse content based on original query
            semantic_results = storage.search_noderag_data(
                org_id=org_id,
                query=f"{query} content details information data",
                top_k=20,
                filters={"node_type": "S"}
            )
            
            # Get relationship insights (R types) - capture connections and dependencies
            relationship_results = storage.search_noderag_data(
                org_id=org_id,
                query="relationships connections dependencies interactions processes workflows",
                top_k=10,
                filters={"node_type": "R"}
            )
            
            # Combine results intelligently with score-based prioritization
            search_results = []
            
            # Prioritize overview content (highest priority for knowledge discovery)
            search_results.extend(overview_results)
            
            # Add key entities with deduplication
            added_nodes = {r['node_id'] for r in search_results}
            for result in entity_results:
                if result['node_id'] not in added_nodes:
                    search_results.append(result)
                    added_nodes.add(result['node_id'])
            
            # Add relationship insights for context
            for result in relationship_results:
                if result['node_id'] not in added_nodes and len(search_results) < 30:
                    search_results.append(result)
                    added_nodes.add(result['node_id'])
            
            # Add semantic content (fill remaining slots)
            for result in semantic_results:
                if result['node_id'] not in added_nodes and len(search_results) < 35:
                    search_results.append(result)
                    added_nodes.add(result['node_id'])
            
            # Sort by similarity score for final ranking
            search_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            logger.info(f"🧠 Optimized agentic exploration found: {len(overview_results)} overview + {len(entity_results)} entities + {len(relationship_results)} relationships + {len(semantic_results)} semantic = {len(search_results)} total")
            
        else:
            # Enhanced standard search for specific queries
            logger.info("🎯 Performing enhanced targeted search")
            
            # Primary search with the original query
            primary_results = storage.search_noderag_data(
                org_id=org_id,
                query=query,
                top_k=25
            )
            
            # Secondary search for context with expanded query
            context_query = f"{query} related information context background details"
            context_results = storage.search_noderag_data(
                org_id=org_id,
                query=context_query,
                top_k=15
            )
            
            # Combine and deduplicate
            search_results = primary_results.copy()
            added_nodes = {r['node_id'] for r in search_results}
            
            for result in context_results:
                if result['node_id'] not in added_nodes and len(search_results) < 30:
                    search_results.append(result)
                    added_nodes.add(result['node_id'])
            
            # Sort by relevance
            search_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            logger.info(f"🎯 Enhanced search found: {len(primary_results)} primary + {len(context_results)} context = {len(search_results)} total")
        
        if not search_results:
                # Try to provide a helpful response even when no data is found
                try:
                    stats = storage.get_storage_stats(org_id)
                    total_files = stats.get('stats', {}).get('total_files', 0)
                    
                    if total_files == 0:
                        helpful_response = f"I don't have any documents in the knowledge base for your organization (org_id: {org_id}). Please upload some documents first using the upload API, then I'll be able to answer questions about your content."
                    else:
                        helpful_response = f"I found {total_files} documents in your knowledge base, but couldn't find information relevant to '{query}'. Try asking more specific questions or use broader terms like 'overview', 'summary', or 'what data do you have available?'"
                except:
                    helpful_response = f"I couldn't find relevant information for '{query}' in your knowledge base. This might be because there are no documents uploaded for your organization, or the query doesn't match the available content. Try asking 'what data do you have available?' to explore what's available."
                
                response_data = {
                    "query": query,
                    "response": helpful_response,
                    "sources": [],
                    "node_types": {},
                    "confidence": 0.0,
                    "context_used": 0,
                    "search_results": 0,
                    "algorithm_used": "Neo4j Storage Vector Search",
                    "helpful_suggestion": True,
                    "org_id": org_id
                }
                
                logger.info(f"📤 EMPTY RESULTS RESPONSE SENT (Neo4j Search): {json.dumps(response_data, indent=2)}")
                return jsonify(response_data)
            
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
        
        # Calculate token usage for this session
        session_input_tokens = llm_service.total_input_tokens - initial_input_tokens
        session_output_tokens = llm_service.total_output_tokens - initial_output_tokens
        session_api_calls = llm_service.api_calls_count - initial_api_calls
        session_total_tokens = session_input_tokens + session_output_tokens
        processing_time = time.time() - start_time
        
        # Calculate quality metrics
        quality_metrics = {
            "response_length_score": min(1.0, len(final_response) / 500),  # Normalized response length
            "source_diversity_score": len(source_files) / max(len(search_results), 1),  # Source diversity
            "retrieval_coverage": min(1.0, len(search_results) / 20),  # Retrieval effectiveness
            "processing_efficiency": max(0.1, 1.0 / max(processing_time, 0.1)),  # Speed efficiency
            "token_efficiency": session_output_tokens / max(session_input_tokens, 1)  # Output/input ratio
        }
        
        # Composite quality score
        quality_score = (
            quality_metrics["response_length_score"] * 0.2 +
            quality_metrics["source_diversity_score"] * 0.3 +
            quality_metrics["retrieval_coverage"] * 0.2 +
            quality_metrics["processing_efficiency"] * 0.15 +
            quality_metrics["token_efficiency"] * 0.15
        )
        
        # Store token usage and response metadata in Neo4j
        try:
            response_metadata = {
                "query": query[:500],  # Truncate for storage
                "response_length": len(final_response),
                "retrieval_count": len(search_results),
                "unique_sources": len(source_files),
                "processing_time": round(processing_time, 2),
                "algorithm": "Agentic Knowledge Discovery" if is_knowledge_discovery else "Neo4j Storage Vector Search",
                "confidence": confidence,
                "quality_score": round(quality_score, 3),
                "quality_metrics": quality_metrics
            }
            
            storage.store_api_usage(
                org_id=org_id,
                user_id=user_id,
                endpoint="generate-response",
                input_tokens=session_input_tokens,
                output_tokens=session_output_tokens,
                total_tokens=session_total_tokens,
                api_calls=session_api_calls,
                metadata=response_metadata
            )
            logger.info(f"📊 Stored token usage: input={session_input_tokens}, output={session_output_tokens}, api_calls={session_api_calls}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to store token usage: {e}")
        
        logger.info(f"✅ Generated Neo4j storage response with {len(search_results)} sources in {processing_time:.2f}s")
        
        response_data = {
            "query": query,
            "response": final_response.strip(),
            "sources": list(source_files),
            "node_types": node_types,
            "confidence": confidence,
            "context_used": len(search_results),
            "search_results": len(search_results),
            "context_length": len(retrieved_info),
            "algorithm_used": "Agentic Knowledge Discovery" if is_knowledge_discovery else "Neo4j Storage Vector Search",
            "agentic_mode": is_knowledge_discovery,
            "processing_time": round(processing_time, 2),
            "token_usage": {
                "input_tokens": session_input_tokens,
                "output_tokens": session_output_tokens,
                "total_tokens": session_total_tokens,
                "api_calls": session_api_calls
            },
            "quality_metrics": {
                "quality_score": round(quality_score, 3),
                "response_length_score": round(quality_metrics["response_length_score"], 3),
                "source_diversity_score": round(quality_metrics["source_diversity_score"], 3),
                "retrieval_coverage": round(quality_metrics["retrieval_coverage"], 3),
                "processing_efficiency": round(quality_metrics["processing_efficiency"], 3),
                "token_efficiency": round(quality_metrics["token_efficiency"], 3)
            },
            "retrieval_metadata": {
                "storage_search_count": len(search_results),
                "unique_sources": len(source_files),
                "node_type_distribution": node_types,
                "exploration_strategy": "multi_query_node_type_prioritization" if is_knowledge_discovery else "standard_vector_search"
            }
        }
        
        logger.info(f"📤 SUCCESSFUL RESPONSE SENT (Neo4j Search): Query='{query}', Sources={len(source_files)}, Confidence={confidence:.3f}")
        logger.debug(f"📤 FULL RESPONSE DATA: {json.dumps(response_data, indent=2)}")
        return jsonify(response_data)
        
    except ValueError as e:
        logger.error(f"❌ Invalid input: {e}")
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except ConnectionError as e:
        logger.error(f"❌ Neo4j connection error: {e}")
        return jsonify({"error": "Database connection failed. Please try again later."}), 503
    except TimeoutError as e:
        logger.error(f"❌ Request timeout: {e}")
        return jsonify({"error": "Request timeout. Please try with a shorter query."}), 408
    except Exception as e:
        logger.error(f"❌ Response generation error: {e}")
        # Calculate partial token usage even on failure
        try:
            if 'llm_service' in locals() and 'initial_input_tokens' in locals():
                session_input_tokens = llm_service.total_input_tokens - initial_input_tokens
                session_output_tokens = llm_service.total_output_tokens - initial_output_tokens
                session_api_calls = llm_service.api_calls_count - initial_api_calls
                processing_time = time.time() - start_time if 'start_time' in locals() else 0
                
                # Store failed request metrics
                storage = noderag_service.get_neo4j_storage()
                storage.store_api_usage(
                    org_id=data.get("org_id", "unknown"),
                    user_id=data.get("user_id", "api_user"),
                    endpoint="generate-response-failed",
                    input_tokens=session_input_tokens,
                    output_tokens=session_output_tokens,
                    total_tokens=session_input_tokens + session_output_tokens,
                    api_calls=session_api_calls,
                    metadata={"error": str(e), "processing_time": round(processing_time, 2)}
                )
        except Exception as track_error:
            logger.warning(f"⚠️ Failed to track error usage: {track_error}")
            
        return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.route("/api/v1/delete-embeddings", methods=["DELETE"])
def delete_embeddings():
    """Delete embeddings and graphs for multiple files with webhook callback"""
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
        
        if not isinstance(file_ids, list) or not file_ids:
            return jsonify({"error": "file_ids must be a non-empty list"}), 400
        
        logger.info(f"🗑️ Starting bulk delete for org_id={org_id}, {len(file_ids)} files")
        
        # Start delete operation in background
        delete_id = str(uuid.uuid4())
        
        def delete_in_background():
            """Background task to delete embeddings and send webhook"""
            try:
                print(f"🔄 Starting background delete for delete_id={delete_id}")
                storage = noderag_service.get_neo4j_storage()
                
                total_deleted_embeddings = 0
                total_deleted_graphs = 0
                successful_files = []
                failed_files = []
                
                for file_id in file_ids:
                    try:
                        print(f"🗑️ Deleting data for file_id={file_id}")
                        logger.info(f"🗑️ Deleting data for file_id={file_id}")
                        
                        # Delete file data (both embeddings and graphs)
                        delete_result = storage.delete_file_data(org_id=org_id, file_id=file_id)
                        print(f"   Delete result: {delete_result}")
                        
                        if delete_result.get("success"):
                            successful_files.append(file_id)
                            total_deleted_embeddings += delete_result.get("deleted_count", 0)
                            total_deleted_graphs += delete_result.get("graphs_deleted", 0)
                            
                            print(f"   ✅ Successfully deleted: {delete_result.get('deleted_count', 0)} embeddings, {delete_result.get('graphs_deleted', 0)} graphs")
                            
                            # Clean up processing status
                            with processing_lock:
                                if file_id in processing_status:
                                    del processing_status[file_id]
                                    
                            logger.info(f"✅ Successfully deleted data for file_id={file_id}")
                        else:
                            failed_files.append({"file_id": file_id, "error": delete_result.get("error", "Unknown error")})
                            logger.error(f"❌ Failed to delete data for file_id={file_id}: {delete_result.get('error')}")
                            
                    except Exception as e:
                        failed_files.append({"file_id": file_id, "error": str(e)})
                        logger.error(f"❌ Exception deleting file_id={file_id}: {e}")
                
                # Prepare webhook response
                webhook_data = {
                    "status": "completed",
                    "org_id": org_id,
                    "file_ids": successful_files,
                    "deleted_embeddings": total_deleted_embeddings,
                    "deleted_graphs": total_deleted_graphs,
                    "success": len(failed_files) == 0,
                    "failed_files": failed_files,
                    "delete_id": delete_id,
                    "timestamp": time.time()
                }
                
                # Send webhook callback if provided
                if callback_url:
                    # Send webhook data directly (not nested in 'data')
                    send_webhook_direct(callback_url, webhook_data)
                    logger.info(f"✅ Webhook sent to {callback_url}")
                else:
                    logger.info("✅ Delete completed (no webhook configured)")
                    
                logger.info(f"🎉 Bulk delete completed: {len(successful_files)}/{len(file_ids)} files deleted, {total_deleted_embeddings} embeddings, {total_deleted_graphs} graphs")
                
            except Exception as e:
                logger.error(f"❌ Background delete failed: {e}")
                
                # Send error webhook
                if callback_url:
                    error_data = {
                        "status": "failed",
                        "org_id": org_id,
                        "file_ids": [],
                        "deleted_embeddings": 0,
                        "deleted_graphs": 0,
                        "success": False,
                        "error": str(e),
                        "delete_id": delete_id,
                        "timestamp": time.time()
                    }
                    send_webhook_direct(callback_url, error_data)
        
        # Start background task
        thread = threading.Thread(target=delete_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Delete operation started",
            "delete_id": delete_id,
            "org_id": org_id,
            "file_count": len(file_ids),
            "status": "processing",
            "callback_url": callback_url
        }), 202  # HTTP 202 Accepted
        
    except Exception as e:
        logger.error(f"❌ Delete operation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/delete-status/<delete_id>", methods=["GET"])
def delete_status(delete_id):
    """Get status of delete operation"""
    # In a production environment, you would store this in a database
    # For now, return a simple response
    return jsonify({
        "delete_id": delete_id,
        "status": "processing",
        "message": "Check webhook for completion status"
    })

if __name__ == "__main__":
    # Get configuration
    host = Config.API_HOST
    port = Config.API_PORT
    debug = Config.API_DEBUG
    
    logger.info(f"🚀 Starting NodeRAG API Service on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
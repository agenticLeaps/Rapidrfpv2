import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any, List
import traceback

from ..document_processing.indexing_pipeline import IndexingPipeline
from ..document_processing.document_loader import DocumentLoader
from ..document_processing.llamaparse_service import LlamaCloudParser, EnhancedDocumentLoader
from ..incremental.incremental_pipeline import IncrementalIndexingPipeline
from ..search.advanced_search import AdvancedSearchSystem
from ..vector.hnsw_service import HNSWService
from ..visualization.graph_visualizer import GraphVisualizer, VisualizationConfig
from ..graph.graph_manager import GraphManager
from ..graph.node_types import NodeType
from ..config.settings import Config
from .session_manager import session_manager

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def get_session_id(request) -> str:
    """Extract session ID from request headers or form data."""
    # Try header first
    session_id = request.headers.get('X-Session-ID')
    
    # Try form data (for file uploads)
    if not session_id and hasattr(request, 'form'):
        session_id = request.form.get('session_id')
    
    # Try JSON data
    if not session_id and request.is_json:
        data = request.get_json() or {}
        session_id = data.get('session_id')
    
    # Try query parameters
    if not session_id:
        session_id = request.args.get('session_id')
    
    if not session_id:
        raise ValueError("Session ID is required. Please include X-Session-ID header or session_id parameter.")
    
    return session_id

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'RapidRFP RAG API'
    })

@app.route('/api/index/document', methods=['POST'])
def index_document():
    """Index a document through the complete pipeline."""
    try:
        data = request.get_json()
        
        if not data or 'file_path' not in data:
            return jsonify({
                'error': 'file_path is required'
            }), 400
        
        file_path = data['file_path']
        
        if not os.path.exists(file_path):
            return jsonify({
                'error': f'File not found: {file_path}'
            }), 404
        
        # Index the document
        result = indexing_pipeline.index_document(file_path)
        
        if result['success']:
            # Save the graph
            save_success = indexing_pipeline.save_graph()
            result['graph_saved'] = save_success
            
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in index_document: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/upload/document', methods=['POST'])
def upload_and_index_document():
    """Upload and index a document file with session isolation."""
    try:
        # Get session ID
        session_id = get_session_id(request)
        session_data = session_manager.get_or_create_session(session_id)
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided. Use "file" field in form data.'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file format: {file_ext}. Supported: {list(allowed_extensions)}'
            }), 400
        
        # Save uploaded file to session-specific directory
        upload_dir = os.path.join(session_data.data_dir, "raw")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        import uuid
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        file.save(file_path)
        logger.info(f"File uploaded to session {session_id}: {file_path}")
        
        try:
            # Index the uploaded document using session-specific pipeline
            logger.info(f"Starting document indexing for session {session_id}: {file.filename}")
            result = session_data.indexing_pipeline.index_document(file_path)
            logger.info(f"Indexing result for session {session_id}: {result}")
            
            if result['success']:
                # Save the session-specific graph
                save_success = session_data.indexing_pipeline.save_graph()
                result['graph_saved'] = save_success
                result['uploaded_file'] = unique_filename
                result['original_filename'] = file.filename
                result['session_id'] = session_id
                
                # Initialize HNSW service after indexing to include new embeddings
                try:
                    hnsw_service = session_manager.get_hnsw_service(session_id)
                    # Rebuild HNSW index with new embeddings
                    graph_manager = session_data.indexing_pipeline.graph_manager
                    nodes_with_embeddings = []
                    
                    for node_id in graph_manager.graph.nodes():
                        node = graph_manager.get_node(node_id)
                        if node and hasattr(node, 'embeddings') and node.embeddings:
                            nodes_with_embeddings.append((node_id, node.embeddings))
                    
                    if nodes_with_embeddings:
                        hnsw_service.rebuild_index(nodes_with_embeddings)
                        logger.info(f"HNSW index updated for session {session_id} with {len(nodes_with_embeddings)} embeddings")
                except Exception as hnsw_error:
                    logger.warning(f"Failed to update HNSW index for session {session_id}: {hnsw_error}")
                
                return jsonify(result)
            else:
                return jsonify(result), 500
                
        finally:
            # Clean up uploaded file
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")
            
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in upload_and_index_document: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/estimate/cost', methods=['POST'])
def estimate_processing_cost():
    """Estimate processing cost for a document."""
    try:
        data = request.get_json()
        
        if not data or 'file_path' not in data:
            return jsonify({
                'error': 'file_path is required'
            }), 400
        
        file_path = data['file_path']
        
        if not os.path.exists(file_path):
            return jsonify({
                'error': f'File not found: {file_path}'
            }), 404
        
        cost_estimate = document_loader.estimate_processing_cost(file_path)
        return jsonify(cost_estimate)
        
    except Exception as e:
        logger.error(f"Error in estimate_processing_cost: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/stats', methods=['GET'])
def get_graph_stats():
    """Get current graph statistics for session."""
    try:
        session_id = get_session_id(request)
        session_data = session_manager.get_or_create_session(session_id)
        
        stats = session_data.indexing_pipeline.get_indexing_stats()
        stats['session_id'] = session_id
        return jsonify(stats)
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_graph_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/nodes/<node_type>', methods=['GET'])
def get_nodes_by_type(node_type: str):
    """Get all nodes of a specific type for session."""
    try:
        session_id = get_session_id(request)
        session_data = session_manager.get_or_create_session(session_id)
        
        # Validate node type
        try:
            node_type_enum = NodeType(node_type.upper())
        except ValueError:
            return jsonify({
                'error': f'Invalid node type: {node_type}. Valid types: {[t.value for t in NodeType]}'
            }), 400
        
        nodes = session_data.indexing_pipeline.graph_manager.get_nodes_by_type(node_type_enum)
        
        # Convert nodes to dict format
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                'id': node.id,
                'type': node.type.value,
                'content': node.content,
                'metadata': node.metadata
            })
        
        return jsonify({
            'node_type': node_type,
            'count': len(nodes_data),
            'nodes': nodes_data,
            'session_id': session_id
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in get_nodes_by_type: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/node/<node_id>', methods=['GET'])
def get_node(node_id: str):
    """Get a specific node by ID."""
    try:
        node = indexing_pipeline.graph_manager.get_node(node_id)
        
        if not node:
            return jsonify({
                'error': f'Node not found: {node_id}'
            }), 404
        
        # Get connected nodes
        connected = indexing_pipeline.graph_manager.get_connected_nodes(node_id)
        
        return jsonify({
            'node': {
                'id': node.id,
                'type': node.type.value,
                'content': node.content,
                'metadata': node.metadata
            },
            'connected_nodes': [
                {
                    'id': conn.id,
                    'type': conn.type.value,
                    'content': conn.content[:100] + '...' if len(conn.content) > 100 else conn.content
                }
                for conn in connected
            ]
        })
        
    except Exception as e:
        logger.error(f"Error in get_node: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/knowledge-base', methods=['GET'])
def debug_knowledge_base():
    """Debug endpoint to check session-specific knowledge base status."""
    try:
        session_id = get_session_id(request)
        session_data = session_manager.get_or_create_session(session_id)
        
        # Get basic graph stats from session
        graph = session_data.indexing_pipeline.graph_manager.graph
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        
        # Get node type counts
        node_types = {}
        for node_id in graph.nodes():
            node = session_data.indexing_pipeline.graph_manager.get_node(node_id)
            if node:
                node_type = node.type.value
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Get sample nodes for each type
        samples = {}
        for node_type in ['T', 'N', 'R', 'A', 'H']:
            try:
                type_nodes = session_data.indexing_pipeline.graph_manager.get_nodes_by_type(NodeType(node_type))
                if type_nodes:
                    # Show first 3 nodes of each type
                    samples[node_type] = []
                    for node in type_nodes[:3]:
                        samples[node_type].append({
                            'id': node.id,
                            'content_preview': node.content[:100] + '...' if len(node.content) > 100 else node.content
                        })
            except ValueError:
                # Skip invalid node types
                pass
        
        # Check session-specific HNSW status
        hnsw_status = "Not initialized"
        try:
            hnsw = session_manager.get_hnsw_service(session_id)
            if hasattr(hnsw, 'index') and hnsw.index is not None:
                hnsw_status = f"Initialized with {hnsw.index.get_current_count()} vectors"
            else:
                hnsw_status = "Initialized but no vectors loaded"
        except Exception as e:
            hnsw_status = f"Error: {str(e)}"
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'graph_stats': {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'node_type_counts': node_types
            },
            'hnsw_status': hnsw_status,
            'sample_nodes': samples
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in debug_knowledge_base: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search/entities', methods=['POST'])
def search_entities():
    """Search for entities by name."""
    try:
        data = request.get_json()
        
        if not data or 'entity_name' not in data:
            return jsonify({
                'error': 'entity_name is required'
            }), 400
        
        entity_name = data['entity_name']
        entities = indexing_pipeline.graph_manager.get_entity_mentions(entity_name)
        
        entities_data = []
        for entity in entities:
            entities_data.append({
                'id': entity.id,
                'content': entity.content,
                'metadata': entity.metadata
            })
        
        return jsonify({
            'query': entity_name,
            'count': len(entities_data),
            'entities': entities_data
        })
        
    except Exception as e:
        logger.error(f"Error in search_entities: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/communities', methods=['GET'])
def get_communities():
    """Get information about detected communities."""
    try:
        community_assignments = indexing_pipeline.graph_manager.community_assignments
        
        if not community_assignments:
            return jsonify({
                'message': 'No communities detected',
                'communities': []
            })
        
        # Group nodes by community
        communities = {}
        for node_id, community_id in community_assignments.items():
            if community_id not in communities:
                communities[community_id] = []
            
            node = indexing_pipeline.graph_manager.get_node(node_id)
            if node:
                communities[community_id].append({
                    'id': node.id,
                    'type': node.type.value,
                    'content': node.content[:100] + '...' if len(node.content) > 100 else node.content
                })
        
        community_list = []
        for community_id, nodes in communities.items():
            community_list.append({
                'community_id': community_id,
                'node_count': len(nodes),
                'nodes': nodes[:10]  # Limit to first 10 nodes
            })
        
        return jsonify({
            'total_communities': len(community_list),
            'communities': community_list
        })
        
    except Exception as e:
        logger.error(f"Error in get_communities: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/important-entities', methods=['GET'])
def get_important_entities():
    """Get the most important entities based on graph metrics."""
    try:
        percentage = request.args.get('percentage', Config.IMPORTANT_ENTITY_PERCENTAGE)
        try:
            percentage = float(percentage)
        except ValueError:
            percentage = Config.IMPORTANT_ENTITY_PERCENTAGE
        
        important_entities = indexing_pipeline.graph_manager.get_important_entities(percentage)
        
        entities_data = []
        for entity in important_entities:
            entities_data.append({
                'id': entity.id,
                'content': entity.content,
                'metadata': entity.metadata
            })
        
        return jsonify({
            'percentage': percentage,
            'count': len(entities_data),
            'entities': entities_data
        })
        
    except Exception as e:
        logger.error(f"Error in get_important_entities: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/save', methods=['POST'])
def save_graph():
    """Save the current graph to disk."""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath', Config.GRAPH_DB_PATH)
        
        success = indexing_pipeline.save_graph(filepath)
        
        return jsonify({
            'success': success,
            'filepath': filepath,
            'message': 'Graph saved successfully' if success else 'Failed to save graph'
        })
        
    except Exception as e:
        logger.error(f"Error in save_graph: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/load', methods=['POST'])
def load_graph():
    """Load a graph from disk."""
    try:
        data = request.get_json() or {}
        filepath = data.get('filepath', Config.GRAPH_DB_PATH)
        
        if not os.path.exists(filepath):
            return jsonify({
                'error': f'Graph file not found: {filepath}'
            }), 404
        
        success = indexing_pipeline.load_graph(filepath)
        
        if success:
            stats = indexing_pipeline.get_indexing_stats()
            return jsonify({
                'success': True,
                'filepath': filepath,
                'message': 'Graph loaded successfully',
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load graph'
            }), 500
        
    except Exception as e:
        logger.error(f"Error in load_graph: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration settings."""
    try:
        config_data = {
            'chunk_size': Config.CHUNK_SIZE,
            'chunk_overlap': Config.CHUNK_OVERLAP,
            'max_entities_per_chunk': Config.MAX_ENTITIES_PER_CHUNK,
            'max_relationships_per_chunk': Config.MAX_RELATIONSHIPS_PER_CHUNK,
            'important_entity_percentage': Config.IMPORTANT_ENTITY_PERCENTAGE,
            'leiden_resolution': Config.LEIDEN_RESOLUTION,
            'default_batch_size': Config.DEFAULT_BATCH_SIZE,
            'use_llamaparse': Config.USE_LLAMAPARSE,
            'llamaparse_settings': {
                'result_type': Config.LLAMAPARSE_RESULT_TYPE,
                'language': Config.LLAMAPARSE_LANGUAGE,
                'num_workers': Config.LLAMAPARSE_NUM_WORKERS,
                'max_wait_time': Config.LLAMAPARSE_MAX_WAIT_TIME,
                'parsing_method': Config.LLAMAPARSE_PARSING_METHOD
            },
            'llm_endpoints': {
                'qwen_llm': Config.QWEN_LLM_ENDPOINT,
                'qwen_embedding': Config.QWEN_EMBEDDING_ENDPOINT
            },
            'hnsw_settings': {
                'dimension': Config.HNSW_DIMENSION,
                'max_elements': Config.HNSW_MAX_ELEMENTS,
                'ef_construction': Config.HNSW_EF_CONSTRUCTION,
                'm': Config.HNSW_M,
                'space': Config.HNSW_SPACE
            }
        }
        
        return jsonify(config_data)
        
    except Exception as e:
        logger.error(f"Error in get_config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parse/llamaparse', methods=['POST'])
def parse_with_llamaparse():
    """Parse document using LlamaParse with various methods."""
    try:
        data = request.get_json() or {}
        
        # Handle file upload or file path
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                file.save(tmp_file.name)
                file_path = tmp_file.name
                
        elif 'file_path' in data:
            # File path provided
            file_path = data['file_path']
            if not os.path.exists(file_path):
                return jsonify({'error': f'File not found: {file_path}'}), 404
        else:
            return jsonify({'error': 'Either file upload or file_path is required'}), 400

        # Initialize LlamaParse
        llamaparse = LlamaCloudParser()
        
        # Parse document with specified method
        parsing_method = data.get('parsing_method', Config.LLAMAPARSE_PARSING_METHOD)
        parse_settings = {
            'result_type': data.get('result_type', Config.LLAMAPARSE_RESULT_TYPE),
            'language': data.get('language', Config.LLAMAPARSE_LANGUAGE),
            'num_workers': data.get('num_workers', Config.LLAMAPARSE_NUM_WORKERS),
            'max_wait_time': data.get('max_wait_time', Config.LLAMAPARSE_MAX_WAIT_TIME),
            'poll_interval': data.get('poll_interval', Config.LLAMAPARSE_POLL_INTERVAL),
            'verbose': data.get('verbose', True)
        }
        
        if parsing_method == 'sync':
            result = llamaparse.parse_document_sync(file_path, **parse_settings)
        elif parsing_method == 'async':
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    llamaparse.parse_document_async(file_path, **parse_settings)
                )
            except RuntimeError:
                result = asyncio.run(
                    llamaparse.parse_document_async(file_path, **parse_settings)
                )
        else:  # job_monitoring
            result = llamaparse.parse_with_job_monitoring(file_path, **parse_settings)
        
        # Clean up temporary file if uploaded
        if 'file' in request.files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        # Format response
        response = {
            'success': result.success,
            'parsing_method': result.parsing_method,
            'processing_time': result.processing_time,
            'documents_found': len(result.documents),
            'job_id': result.job_id
        }
        
        if result.success:
            response['documents'] = result.documents
        else:
            response['error'] = result.error_message
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in parse_with_llamaparse: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/parse/job/<job_id>/status', methods=['GET'])
def check_llamaparse_job_status(job_id: str):
    """Check LlamaParse job status."""
    try:
        llamaparse = LlamaCloudParser()
        result = llamaparse.check_job_status(job_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error checking job status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parse/job/<job_id>/result', methods=['GET'])
def get_llamaparse_job_result(job_id: str):
    """Get LlamaParse job result."""
    try:
        llamaparse = LlamaCloudParser()
        result = llamaparse.get_job_result(job_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error getting job result: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/parse/enhanced', methods=['POST'])
def parse_with_enhanced_loader():
    """Parse document using enhanced document loader with automatic format detection."""
    try:
        data = request.get_json() or {}
        
        # Handle file upload or file path
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                file.save(tmp_file.name)
                file_path = tmp_file.name
                original_filename = file.filename
                
        elif 'file_path' in data:
            # File path provided
            file_path = data['file_path']
            original_filename = os.path.basename(file_path)
            if not os.path.exists(file_path):
                return jsonify({'error': f'File not found: {file_path}'}), 404
        else:
            return jsonify({'error': 'Either file upload or file_path is required'}), 400

        # Initialize enhanced document loader
        enhanced_loader = EnhancedDocumentLoader(
            chunk_size=data.get('chunk_size', Config.CHUNK_SIZE),
            chunk_overlap=data.get('chunk_overlap', Config.CHUNK_OVERLAP),
            use_llamaparse=data.get('use_llamaparse', Config.USE_LLAMAPARSE),
            llamaparse_api_key=Config.LLAMA_CLOUD_API_KEY
        )
        
        # Load document
        parse_settings = {
            'parsing_method': data.get('parsing_method', Config.LLAMAPARSE_PARSING_METHOD),
            'result_type': data.get('result_type', Config.LLAMAPARSE_RESULT_TYPE),
            'language': data.get('language', Config.LLAMAPARSE_LANGUAGE),
            'num_workers': data.get('num_workers', Config.LLAMAPARSE_NUM_WORKERS),
            'max_wait_time': data.get('max_wait_time', Config.LLAMAPARSE_MAX_WAIT_TIME)
        }
        
        processed_doc = enhanced_loader.load_document(file_path, **parse_settings)
        
        # Clean up temporary file if uploaded
        if 'file' in request.files:
            try:
                os.unlink(file_path)
            except:
                pass
        
        if processed_doc:
            response = {
                'success': True,
                'filename': original_filename,
                'chunks_created': len(processed_doc.chunks),
                'total_tokens': processed_doc.total_tokens,
                'metadata': processed_doc.metadata,
                'parsing_method': processed_doc.metadata.get('parsing_method', 'fallback'),
                'processing_time': processed_doc.metadata.get('processing_time', 0.0)
            }
            
            # Include chunks if requested
            if data.get('include_chunks', False):
                response['chunks'] = [
                    {
                        'index': chunk.chunk_index,
                        'content': chunk.content,
                        'token_count': chunk.token_count,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char
                    }
                    for chunk in processed_doc.chunks
                ]
            
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to process document',
                'filename': original_filename
            }), 500
            
    except Exception as e:
        logger.error(f"Error in parse_with_enhanced_loader: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Advanced Search Endpoints

def _initialize_advanced_search():
    """Initialize advanced search system if not already initialized."""
    global advanced_search
    if advanced_search is None:
        try:
            # Initialize required services
            hnsw = _initialize_hnsw_service()
            
            # Initialize LLM service
            from ..llm.llm_service import LLMService
            llm_service = LLMService()
            
            # Initialize advanced search with required parameters
            advanced_search = AdvancedSearchSystem(
                graph_manager=indexing_pipeline.graph_manager,
                hnsw_service=hnsw,
                llm_service=llm_service
            )
            logger.info("Advanced search system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize advanced search: {e}")
            raise
    return advanced_search

def _initialize_hnsw_service():
    """Initialize HNSW service if not already initialized."""
    global hnsw_service
    if hnsw_service is None:
        try:
            hnsw_service = HNSWService()
            
            # Try to load existing index
            try:
                loaded = hnsw_service.load_index()
                if loaded:
                    logger.info(f"HNSW service initialized and index loaded with {hnsw_service.current_count} vectors")
                else:
                    logger.info("HNSW service initialized with empty index")
            except Exception as load_error:
                logger.warning(f"Could not load existing HNSW index: {load_error}")
                logger.info("HNSW service initialized with empty index")
                
        except Exception as e:
            logger.error(f"Failed to initialize HNSW service: {e}")
            raise
    return hnsw_service

@app.route('/api/search/advanced', methods=['POST'])
def advanced_search_endpoint():
    """Perform advanced multi-signal search."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'query is required'}), 400
        
        query = data['query']
        top_k = data.get('top_k', 10)
        
        # Initialize search system
        search_system = _initialize_advanced_search()
        
        # Perform search
        results = search_system.search(query, top_k=top_k)
        
        response = {
            'success': True,
            'query': query,
            'results': results,
            'total_results': len(results),
            'search_time': results.get('search_time', 0) if isinstance(results, dict) else 0
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in advanced_search_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/search/vector', methods=['POST'])
def vector_search_endpoint():
    """Perform HNSW vector similarity search."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'query is required'}), 400
        
        query = data['query']
        k = data.get('k', 10)
        
        # Initialize HNSW service
        hnsw = _initialize_hnsw_service()
        
        # Perform search
        results = hnsw.search(query, k=k)
        
        response = {
            'success': True,
            'query': query,
            'results': [
                {
                    'id': result_id,
                    'distance': distance,
                    'similarity': 1.0 - distance  # Convert distance to similarity
                }
                for result_id, distance in results
            ],
            'total_results': len(results)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in vector_search_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/search/ppr', methods=['POST'])
def ppr_search_endpoint():
    """Perform Personalized PageRank search."""
    try:
        data = request.get_json()
        
        if not data or 'seed_nodes' not in data:
            return jsonify({'error': 'seed_nodes is required'}), 400
        
        seed_nodes = data['seed_nodes']
        alpha = data.get('alpha', Config.PPR_ALPHA)
        top_k = data.get('top_k', 20)
        
        # Initialize search system
        search_system = _initialize_advanced_search()
        
        # Perform PPR search
        if hasattr(search_system, 'ppr_search'):
            results = search_system.ppr_search(seed_nodes, alpha=alpha, top_k=top_k)
        else:
            # Fallback to direct PPR implementation
            from ..search.personalized_pagerank import PersonalizedPageRank
            ppr = PersonalizedPageRank(indexing_pipeline.graph_manager.graph)
            results = ppr.search(seed_nodes, alpha=alpha, top_k=top_k)
        
        response = {
            'success': True,
            'seed_nodes': seed_nodes,
            'results': results,
            'total_results': len(results),
            'alpha': alpha
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in ppr_search_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/answer', methods=['POST'])
def answer_generation_endpoint():
    """Generate comprehensive answer using session-aware advanced search and retrieval."""
    try:
        session_id = get_session_id(request)
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'query is required'}), 400
        
        query = data['query']
        use_structured_prompt = data.get('use_structured_prompt', True)
        k_hnsw = data.get('k_hnsw', 10)
        k_final = data.get('k_final', 20)
        entity_limit = data.get('entity_limit', 10)
        relationship_limit = data.get('relationship_limit', 30)
        high_level_limit = data.get('high_level_limit', 10)
        
        # Get session-specific search system
        search_system = session_manager.get_advanced_search(session_id)
        
        # Perform comprehensive search
        retrieval_result = search_system.search(
            query=query,
            k_hnsw=k_hnsw,
            k_final=k_final,
            entity_nodes_limit=entity_limit,
            relationship_nodes_limit=relationship_limit,
            high_level_nodes_limit=high_level_limit
        )
        
        # Debug: Log retrieval results
        logger.info(f"Session {session_id} - Query: {query}")
        logger.info(f"Session {session_id} - Retrieved {len(retrieval_result.final_nodes)} final nodes")
        logger.info(f"Session {session_id} - HNSW results: {len(retrieval_result.hnsw_results)}")
        logger.info(f"Session {session_id} - Exact entity matches: {len(retrieval_result.accurate_results)}")
        logger.info(f"Session {session_id} - PPR results: {len(retrieval_result.ppr_results)}")
        
        if not retrieval_result.final_nodes:
            logger.warning(f"Session {session_id} - No nodes retrieved for query - this might indicate indexing issues")
        
        # Get session-specific graph manager
        session_data = session_manager.get_or_create_session(session_id)
        graph_manager = session_data.indexing_pipeline.graph_manager
        
        # Build structured context from retrieved nodes
        context_parts = []
        node_details = []
        
        # Group nodes by type for structured presentation
        nodes_by_type = {
            'ENTITY': [],
            'RELATIONSHIP': [],
            'HIGH_LEVEL': [],
            'ATTRIBUTE': [],
            'OTHER': []
        }
        
        for node_id in retrieval_result.final_nodes:
            node = graph_manager.get_node(node_id)
            if node:
                node_type = node.type.value.upper()
                if node_type in nodes_by_type:
                    nodes_by_type[node_type].append(node.content)
                else:
                    nodes_by_type['OTHER'].append(node.content)
                
                node_details.append({
                    'id': node.id,
                    'type': node.type.value,
                    'content': node.content,  # Include full content for sources
                    'metadata': getattr(node, 'metadata', {})
                })
        
        # Format context based on prompt type
        if use_structured_prompt:
            # Create NodeRAG-style structured prompt
            for node_type, contents in nodes_by_type.items():
                if contents:
                    context_parts.append(f"------------{node_type.lower()}-------------")
                    for i, content in enumerate(contents, 1):
                        context_parts.append(f"{i}. {content}")
                    context_parts.append("")
        else:
            # Simple unstructured format
            all_content = []
            for contents in nodes_by_type.values():
                all_content.extend(contents)
            context_parts = all_content
        
        retrieved_info = "\n".join(context_parts).strip()
        
        # Check if we have any retrieved information
        if not retrieved_info or len(retrieved_info) < 10:
            logger.warning(f"Very little or no information retrieved for query: {query}")
            retrieved_info = "No relevant information found in the knowledge base."
        
        # Generate answer using LLM
        from ..llm.llm_service import LLMService
        llm_service = LLMService()
        
        answer_prompt = llm_service.prompt_manager.answer_generation.format(
            info=retrieved_info,
            query=query
        )
        
        response = llm_service._chat_completion(answer_prompt, temperature=0.7)
        
        # Build comprehensive response
        result = {
            'success': True,
            'session_id': session_id,
            'query': query,
            'answer': response,
            'retrieval_metadata': {
                'total_nodes_retrieved': len(retrieval_result.final_nodes),
                'hnsw_results': len(retrieval_result.hnsw_results),
                'exact_matches': len(retrieval_result.accurate_results),
                'ppr_results': len(retrieval_result.ppr_results),
                'entity_nodes': len(retrieval_result.entity_nodes),
                'relationship_nodes': len(retrieval_result.relationship_nodes),
                'high_level_nodes': len(retrieval_result.high_level_nodes),
                'context_length': len(retrieved_info),
                'use_structured_prompt': use_structured_prompt
            },
            'search_breakdown': {
                'entity_count': len(nodes_by_type['ENTITY']),
                'relationship_count': len(nodes_by_type['RELATIONSHIP']),
                'high_level_count': len(nodes_by_type['HIGH_LEVEL']),
                'attribute_count': len(nodes_by_type['ATTRIBUTE']),
                'other_count': len(nodes_by_type['OTHER'])
            },
            'retrieved_nodes': node_details
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in answer_generation_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Incremental Processing Endpoints

@app.route('/api/pipeline/incremental', methods=['POST'])
def incremental_processing_endpoint():
    """Process documents incrementally."""
    try:
        data = request.get_json()
        
        if not data or 'file_path' not in data:
            return jsonify({'error': 'file_path is required'}), 400
        
        file_path = data['file_path']
        
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_path}'}), 404
        
        # Process incrementally
        result = incremental_pipeline.process_document_incremental(file_path)
        
        response = {
            'success': True,
            'file_path': file_path,
            'processing_result': result,
            'action': result.get('action', 'unknown'),
            'processing_time': result.get('processing_time', 0)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in incremental_processing_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/pipeline/status', methods=['GET'])
def pipeline_status_endpoint():
    """Get current pipeline status."""
    try:
        status = incremental_pipeline.get_processing_status()
        
        response = {
            'success': True,
            'status': status,
            'timestamp': status.get('timestamp'),
            'total_files': status.get('total_files', 0),
            'processed_files': status.get('processed_files', 0)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in pipeline_status_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/pipeline/resume', methods=['POST'])
def resume_pipeline_endpoint():
    """Resume failed pipeline processing."""
    try:
        result = incremental_pipeline.resume_from_failure()
        
        response = {
            'success': True,
            'resume_result': result,
            'message': 'Pipeline resumed successfully' if result else 'No failed state to resume from'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in resume_pipeline_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# HNSW Index Management Endpoints

@app.route('/api/hnsw/stats', methods=['GET'])
def hnsw_stats_endpoint():
    """Get HNSW index statistics."""
    try:
        hnsw = _initialize_hnsw_service()
        stats = hnsw.get_stats()
        
        response = {
            'success': True,
            'stats': stats
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in hnsw_stats_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/hnsw/rebuild', methods=['POST'])
def rebuild_hnsw_endpoint():
    """Rebuild HNSW index from current graph."""
    try:
        # Get all nodes with embeddings from graph
        graph_manager = indexing_pipeline.graph_manager
        nodes_with_embeddings = []
        
        for node_id in graph_manager.graph.nodes():
            node = graph_manager.get_node(node_id)
            if node and node.embeddings:
                nodes_with_embeddings.append((node_id, node.embeddings))
        
        if not nodes_with_embeddings:
            return jsonify({
                'error': 'No nodes with embeddings found in graph'
            }), 400
        
        # Initialize/rebuild HNSW index
        hnsw = _initialize_hnsw_service()
        hnsw.rebuild_index(nodes_with_embeddings)
        
        response = {
            'success': True,
            'message': 'HNSW index rebuilt successfully',
            'indexed_nodes': len(nodes_with_embeddings)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in rebuild_hnsw_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/hnsw/search', methods=['POST'])
def hnsw_search_endpoint():
    """Direct HNSW search with embedding vector."""
    try:
        data = request.get_json()
        
        if not data or 'embedding' not in data:
            return jsonify({'error': 'embedding vector is required'}), 400
        
        embedding = data['embedding']
        k = data.get('k', 10)
        
        # Validate embedding dimensions
        if not isinstance(embedding, list) or len(embedding) != Config.HNSW_DIMENSION:
            return jsonify({
                'error': f'Embedding must be a list of {Config.HNSW_DIMENSION} floats'
            }), 400
        
        # Initialize HNSW service
        hnsw = _initialize_hnsw_service()
        
        # Perform search
        results = hnsw.search_by_vector(embedding, k=k)
        
        response = {
            'success': True,
            'results': [
                {
                    'id': result_id,
                    'distance': distance,
                    'similarity': 1.0 - distance
                }
                for result_id, distance in results
            ],
            'total_results': len(results)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in hnsw_search_endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Visualization Endpoints

def _initialize_graph_visualizer():
    """Initialize graph visualizer if not already initialized."""
    global graph_visualizer
    if graph_visualizer is None:
        try:
            graph_visualizer = GraphVisualizer(indexing_pipeline.graph_manager)
            logger.info("Graph visualizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize graph visualizer: {e}")
            raise
    return graph_visualizer

@app.route('/api/visualization/create', methods=['POST'])
def create_visualization():
    """Create interactive graph visualization for session."""
    try:
        session_id = get_session_id(request)
        
        data = request.get_json() or {}
        
        # Get session-specific visualizer
        visualizer = session_manager.get_graph_visualizer(session_id)
        
        # Parse parameters
        max_nodes = data.get('max_nodes', 2000)
        output_filename = data.get('output_filename', 'graph_visualization.html')
        filter_node_types = data.get('filter_node_types', [])
        highlight_communities = data.get('highlight_communities', False)
        
        # Convert node type strings to enums
        if filter_node_types:
            try:
                filter_node_types = [NodeType(nt) for nt in filter_node_types]
            except ValueError as e:
                return jsonify({
                    'error': f'Invalid node type: {e}'
                }), 400
        
        # Create session-specific output path
        session_data = session_manager.get_or_create_session(session_id)
        output_path = os.path.join(session_data.data_dir, "visualizations", output_filename)
        
        # Create visualization
        if highlight_communities:
            viz_path = visualizer.create_community_visualization(output_path)
        else:
            viz_path = visualizer.create_visualization(
                output_path=output_path,
                max_nodes=max_nodes,
                filter_node_types=filter_node_types or None
            )
        
        # Get stats
        stats = visualizer.get_visualization_stats()
        
        response = {
            'success': True,
            'session_id': session_id,
            'visualization_path': viz_path,
            'output_filename': output_filename,
            'stats': stats,
            'parameters': {
                'max_nodes': max_nodes,
                'filter_node_types': [nt.value for nt in filter_node_types] if filter_node_types else [],
                'highlight_communities': highlight_communities
            }
        }
        
        return jsonify(response)
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in create_visualization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualization/community', methods=['POST'])
def create_community_visualization():
    """Create community-focused visualization."""
    try:
        data = request.get_json() or {}
        
        # Initialize visualizer
        visualizer = _initialize_graph_visualizer()
        
        # Parse parameters
        output_filename = data.get('output_filename', 'community_visualization.html')
        output_path = os.path.join(Config.DATA_DIR, "processed", output_filename)
        
        # Create community visualization
        viz_path = visualizer.create_community_visualization(output_path)
        
        # Get stats
        stats = visualizer.get_visualization_stats()
        
        response = {
            'success': True,
            'visualization_path': viz_path,
            'output_filename': output_filename,
            'stats': stats,
            'communities': stats.get('num_communities', 0)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in create_community_visualization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualization/node-types', methods=['POST'])
def create_node_type_visualization():
    """Create visualization focusing on specific node types."""
    try:
        data = request.get_json() or {}
        
        if not data or 'node_types' not in data:
            return jsonify({'error': 'node_types is required'}), 400
        
        # Initialize visualizer
        visualizer = _initialize_graph_visualizer()
        
        # Parse parameters
        node_type_strings = data['node_types']
        output_filename = data.get('output_filename', 'node_type_visualization.html')
        
        # Convert node type strings to enums
        try:
            node_types = [NodeType(nt) for nt in node_type_strings]
        except ValueError as e:
            return jsonify({
                'error': f'Invalid node type: {e}',
                'valid_types': [nt.value for nt in NodeType]
            }), 400
        
        # Create output path
        output_path = os.path.join(Config.DATA_DIR, "processed", output_filename)
        
        # Create visualization
        viz_path = visualizer.create_node_type_visualization(node_types, output_path)
        
        # Get stats
        stats = visualizer.get_visualization_stats()
        
        response = {
            'success': True,
            'visualization_path': viz_path,
            'output_filename': output_filename,
            'node_types': [nt.value for nt in node_types],
            'stats': stats
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in create_node_type_visualization: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualization/stats', methods=['GET'])
def visualization_stats():
    """Get graph statistics for visualization."""
    try:
        # Initialize visualizer
        visualizer = _initialize_graph_visualizer()
        
        # Get stats
        stats = visualizer.get_visualization_stats()
        
        response = {
            'success': True,
            'stats': stats
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in visualization_stats: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/visualization/serve/<filename>', methods=['GET'])
def serve_visualization(filename):
    """Serve visualization HTML files."""
    try:
        # Sanitize filename
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.html'):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Construct file path - ensure absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, Config.DATA_DIR, "processed", safe_filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Visualization file not found'}), 404
        
        # Serve file
        from flask import send_file
        return send_file(
            file_path,
            mimetype='text/html',
            as_attachment=False,
            download_name=safe_filename
        )
        
    except Exception as e:
        logger.error(f"Error in serve_visualization: {e}")
        return jsonify({'error': str(e)}), 500

# Session Management Endpoints

@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear all data for the current session."""
    try:
        session_id = get_session_id(request)
        
        success = session_manager.clear_session(session_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Session {session_id} cleared successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Session {session_id} not found'
            }), 404
            
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/stats', methods=['GET'])
def get_session_stats():
    """Get statistics for the current session."""
    try:
        session_id = get_session_id(request)
        
        stats = session_manager.get_session_stats(session_id)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/list', methods=['GET'])
def list_sessions():
    """List all active sessions (admin endpoint)."""
    try:
        sessions = session_manager.list_active_sessions()
        
        return jsonify({
            'success': True,
            'sessions': sessions
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# V1 API Compatibility Endpoints (for RapidRFPAI integration)
# =============================================================================

@app.route('/api/v1/health', methods=['GET'])
def v1_health_check():
    """V1 API health check endpoint."""
    import time
    return jsonify({
        "status": "healthy",
        "service": "noderag",
        "version": "1.0.0",
        "timestamp": time.time()
    })

def _process_document_background(file_id, chunks_data, session_id, user_id, callback_url, page_count, filename):
    """Background worker for document processing"""
    import time
    import requests as req_lib
    from ..document_processing.document_loader import DocumentChunk, ProcessedDocument

    start_time = time.time()

    try:
        logger.info(f"🔄 Background processing started: file_id={file_id}")

        # Get or create session
        session_data = session_manager.get_or_create_session(session_id)
        indexing_pipeline = session_data.indexing_pipeline

        # Set org_id and file_id context for embedding storage
        indexing_pipeline.current_org_id = session_id
        indexing_pipeline.current_file_id = file_id
        indexing_pipeline.current_user_id = user_id

        # Convert incoming chunks to DocumentChunk objects
        document_chunks = []
        char_position = 0

        for i, chunk in enumerate(chunks_data):
            chunk_text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text", "")
            if not chunk_text:
                continue

            chunk_metadata = chunk.get("metadata", {})
            chunk_metadata["file_id"] = file_id
            chunk_metadata["org_id"] = session_id
            chunk_metadata["user_id"] = user_id
            chunk_metadata["page_number"] = chunk_metadata.get("page_number", i + 1)

            doc_chunk = DocumentChunk(
                content=chunk_text,
                chunk_index=i,
                start_char=char_position,
                end_char=char_position + len(chunk_text),
                metadata=chunk_metadata,
                token_count=len(chunk_text.split())
            )
            document_chunks.append(doc_chunk)
            char_position += len(chunk_text)

        # Create ProcessedDocument
        processed_doc = ProcessedDocument(
            chunks=document_chunks,
            metadata={
                "file_id": file_id,
                "filename": filename,
                "page_count": page_count,
                "source": "rapidrfpai_v1"
            },
            total_tokens=sum(c.token_count for c in document_chunks)
        )

        logger.info(f"📄 Background: Created ProcessedDocument with {len(document_chunks)} chunks")

        # Run through indexing pipeline phases
        decomposition_result = indexing_pipeline._phase_1_decomposition(processed_doc)
        if not decomposition_result.get('success'):
            logger.warning(f"Phase 1 partial failure: {decomposition_result}")

        augmentation_result = indexing_pipeline._phase_2_augmentation()
        if not augmentation_result.get('success'):
            logger.warning(f"Phase 2 partial failure: {augmentation_result}")

        embedding_result = indexing_pipeline._phase_3_embedding_generation()
        if not embedding_result.get('success'):
            logger.warning(f"Phase 3 partial failure: {embedding_result}")

        # Store graph in database (not just pickle file)
        import pickle
        from ..storage.neon_storage import NeonDBStorage

        try:
            storage = NeonDBStorage()

            # Serialize graph data
            graph_data = pickle.dumps({
                'graph': indexing_pipeline.graph_manager.graph,
                'entity_index': dict(indexing_pipeline.graph_manager.entity_index),
                'community_assignments': indexing_pipeline.graph_manager.community_assignments
            })

            # Get stats
            graph_stats = {
                'total_nodes': indexing_pipeline.graph_manager.graph.number_of_nodes() if indexing_pipeline.graph_manager.graph else 0,
                'total_edges': indexing_pipeline.graph_manager.graph.number_of_edges() if indexing_pipeline.graph_manager.graph else 0,
                'chunks_processed': len(document_chunks),
                'file_id': file_id,
            }

            # Store to database
            storage_result = storage.store_org_graph_sync(
                org_id=session_id,
                graph_data=graph_data,
                processed_files=[file_id],
                version=1,
                last_file_added=file_id,
                stats=graph_stats,
                user_id=user_id
            )

            if storage_result.get('success'):
                logger.info(f"✅ Graph stored in database for org={session_id}, file={file_id}")
                save_success = True
            else:
                logger.warning(f"⚠️ Graph DB storage failed: {storage_result.get('error')}, falling back to file")
                save_success = indexing_pipeline.save_graph()

        except Exception as graph_err:
            logger.warning(f"⚠️ Graph DB storage error: {graph_err}, falling back to file")
            save_success = indexing_pipeline.save_graph()

        processing_time = time.time() - start_time

        result_data = {
            "file_id": file_id,
            "user_id": user_id,
            "message": "Document processed successfully",
            "chunks_processed": len(document_chunks),
            "total_chunks": len(chunks_data),
            "page_count": page_count,
            "graph_saved": save_success,
            "session_id": session_id,
            "processing_time": round(processing_time, 2)
        }

        logger.info(f"✅ Background processing completed: file_id={file_id} in {processing_time:.2f}s")

        # Send callback with expected format (status at top level, details in data)
        if callback_url:
            try:
                callback_payload = {
                    "status": "completed",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "data": result_data
                }
                req_lib.post(callback_url, json=callback_payload, timeout=10)
                logger.info(f"✅ Callback sent to {callback_url}")
            except Exception as cb_error:
                logger.warning(f"Callback failed: {cb_error}")

    except Exception as e:
        logger.error(f"❌ Background processing error: {e}")
        logger.error(traceback.format_exc())

        processing_time = time.time() - start_time

        # Send error callback with expected format
        if callback_url:
            try:
                import requests as req_lib
                error_payload = {
                    "status": "failed",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "data": {
                        "file_id": file_id,
                        "user_id": user_id,
                        "error": str(e),
                        "processing_time": round(processing_time, 2)
                    }
                }
                req_lib.post(callback_url, json=error_payload, timeout=10)
            except Exception as cb_error:
                logger.warning(f"Error callback failed: {cb_error}")


@app.route('/api/v1/process-document', methods=['POST'])
def v1_process_document():
    """
    V1 API - Process document chunks through NodeRAG pipeline.
    Returns 202 Accepted immediately and processes in background.
    Sends callback when processing completes.
    """
    import threading

    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["file_id", "chunks"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        file_id = data["file_id"]
        chunks_data = data["chunks"]
        session_id = data.get("session_id") or data.get("org_id") or "default"
        user_id = data.get("user_id", "system")
        callback_url = data.get("callback_url")
        page_count = data.get("page_count", len(chunks_data))
        filename = data.get("filename", f"{file_id}.pdf")

        if not chunks_data:
            return jsonify({
                "success": False,
                "error": "No chunks to process"
            }), 400

        logger.info(f"🚀 V1 NodeRAG accepted: file_id={file_id}, chunks={len(chunks_data)}, session={session_id}")

        # Start background processing thread
        thread = threading.Thread(
            target=_process_document_background,
            args=(file_id, chunks_data, session_id, user_id, callback_url, page_count, filename),
            daemon=True
        )
        thread.start()

        # Return 202 Accepted immediately
        return jsonify({
            "success": True,
            "message": "Processing started",
            "file_id": file_id,
            "status": "processing",
            "session_id": session_id,
            "chunks_received": len(chunks_data)
        }), 202

    except Exception as e:
        logger.error(f"❌ V1 process-document error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/v1/generate-response', methods=['POST'])
def v1_generate_response():
    """
    V1 API - Generate response using RAG.
    Compatible with RapidRFPAI's query flow.
    Uses database for search, falls back to in-memory if available.
    """
    try:
        data = request.get_json()

        query = data.get("query") or data.get("question")
        session_id = data.get("session_id") or data.get("org_id") or request.headers.get('X-Session-ID', 'default')
        top_k = data.get("top_k", 10)

        if not query:
            return jsonify({"error": "query or question is required"}), 400

        logger.info(f"🔍 V1 generate-response: query='{query[:50]}...', session={session_id}")

        # Primary: Search from database (persistent storage)
        from ..storage.neon_storage import NeonDBStorage
        storage = NeonDBStorage()

        db_results = storage.search_noderag_data(
            org_id=session_id,
            query=query,
            top_k=top_k
        )

        contexts = []
        sources = []

        if db_results:
            # Use database results
            logger.info(f"📊 Found {len(db_results)} results from database")
            for result in db_results[:10]:
                content = result.get("content", "")
                if content:
                    contexts.append(content)
                    sources.append({
                        'node_id': result.get("node_id", ""),
                        'text': content[:200] if len(content) > 200 else content,
                        'type': result.get("node_type", "unknown"),
                        'metadata': result.get("metadata", {}),
                        'similarity_score': result.get("similarity_score", 0),
                        'file_id': result.get("file_id", "")
                    })
        else:
            # Fallback: Try in-memory session search
            logger.info(f"📊 No DB results, trying in-memory session")
            try:
                search_system = session_manager.get_advanced_search(session_id)
                session_data = session_manager.get_or_create_session(session_id)
                graph_manager = session_data.indexing_pipeline.graph_manager

                retrieval_result = search_system.search(query, k_hnsw=10, k_final=20)

                for node_id in retrieval_result.final_nodes[:10]:
                    node = graph_manager.get_node(node_id)
                    if node and node.content:
                        contexts.append(node.content)
                        sources.append({
                            'node_id': node_id,
                            'text': node.content[:200] if len(node.content) > 200 else node.content,
                            'type': node.type.value,
                            'metadata': getattr(node, 'metadata', {})
                        })
            except Exception as mem_error:
                logger.warning(f"In-memory search also failed: {mem_error}")

        # Build context text
        context_text = "\n\n".join(contexts[:5])

        # Generate response using LLM if available
        response_text = context_text if contexts else "No relevant information found in the knowledge base."

        try:
            from ..llm.llm_service import LLMService
            llm_service = LLMService()

            answer_prompt = llm_service.prompt_manager.answer_generation.format(
                info=context_text,
                query=query
            )
            response_text = llm_service._chat_completion(answer_prompt, temperature=0.7)
        except Exception as llm_error:
            logger.warning(f"LLM response generation failed, using raw context: {llm_error}")

        response = {
            "success": True,
            "query": query,
            "response": response_text,
            "sources": sources[:5],
            "session_id": session_id,
            "retrieval_stats": {
                "total_results": len(sources),
                "contexts_used": len(contexts[:5]),
                "source": "database" if db_results else "memory"
            },
            "usage": {
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(query.split()) + len(response_text.split())
            }
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ V1 generate-response error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/delete-embeddings', methods=['DELETE'])
def v1_delete_embeddings():
    """V1 API - Delete embeddings for a file from database and memory."""
    try:
        data = request.get_json()
        file_id = data.get("file_id")
        session_id = data.get("session_id") or data.get("org_id") or "default"

        if not file_id:
            return jsonify({"error": "file_id is required"}), 400

        logger.info(f"🗑️ V1 delete-embeddings: file_id={file_id}, session={session_id}")

        # Delete from database
        from ..storage.neon_storage import NeonDBStorage
        storage = NeonDBStorage()

        db_result = storage.delete_file_data(org_id=session_id, file_id=file_id)
        logger.info(f"Database delete result: {db_result}")

        # Clear in-memory session data
        cleared = session_manager.clear_session(session_id)
        if cleared:
            logger.info(f"Cleared in-memory session {session_id}")

        return jsonify({
            "success": True,
            "file_id": file_id,
            "session_id": session_id,
            "database_deleted": db_result.get("success", False),
            "embeddings_deleted": db_result.get("embeddings_deleted", 0),
            "message": "Embeddings deleted from database and memory"
        }), 200

    except Exception as e:
        logger.error(f"❌ V1 delete-embeddings error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/v1/status/<file_id>', methods=['GET'])
def v1_status(file_id):
    """V1 API - Get processing status for a file."""
    return jsonify({
        "file_id": file_id,
        "status": "completed",
        "message": "File processed"
    }), 200

@app.route('/api/v1/search', methods=['POST'])
def v1_search():
    """V1 API - Search embeddings in Neon database."""
    try:
        data = request.get_json()

        org_id = data.get("org_id") or data.get("session_id")
        query = data.get("query")
        top_k = data.get("top_k", 10)
        filters = data.get("filters", {})

        if not org_id:
            return jsonify({"success": False, "error": "org_id is required"}), 400
        if not query:
            return jsonify({"success": False, "error": "query is required"}), 400

        logger.info(f"🔍 V1 search: org_id={org_id}, query={query[:50]}..., top_k={top_k}")

        # Import and use NeonDBStorage
        from ..storage.neon_storage import NeonDBStorage

        storage = NeonDBStorage()
        results = storage.search_noderag_data(
            org_id=org_id,
            query=query,
            top_k=top_k,
            filters=filters
        )

        logger.info(f"✅ V1 search found {len(results)} results")

        return jsonify({
            "success": True,
            "results": results,
            "count": len(results)
        }), 200

    except Exception as e:
        logger.error(f"❌ V1 search error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "results": [],
            "count": 0
        }), 500

# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app.run(debug=Config.API_DEBUG, host=Config.API_HOST, port=Config.API_PORT)
import logging
import uuid
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..graph.graph_manager import GraphManager
from ..graph.node_types import Node, Edge, NodeType
from ..llm.llm_service import LLMService
from .document_loader import DocumentLoader, ProcessedDocument, DocumentChunk
from .llamaparse_service import EnhancedDocumentLoader
from ..vector.hnsw_service import HNSWService
from ..config.settings import Config
from ..utils.performance_logger import performance_logger

logger = logging.getLogger(__name__)

class IndexingPipeline:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.llm_service = LLMService()
        self.hnsw_service = None  # Will be initialized when needed
        
        # Use enhanced document loader with LlamaParse integration
        if Config.USE_LLAMAPARSE:
            self.document_loader = EnhancedDocumentLoader(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                use_llamaparse=True,
                llamaparse_api_key=Config.LLAMA_CLOUD_API_KEY
            )
        else:
            # Fallback to original document loader
            self.document_loader = DocumentLoader(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
    
    def _get_hnsw_service(self):
        """Get or create HNSW service instance."""
        if self.hnsw_service is None:
            self.hnsw_service = HNSWService()
            # Try to load existing index
            try:
                loaded = self.hnsw_service.load_index()
                if loaded:
                    logger.info(f"Loaded existing HNSW index with {self.hnsw_service.current_count} vectors")
                else:
                    logger.info("Created new empty HNSW index")
            except Exception as e:
                logger.warning(f"Could not load HNSW index: {e}")
        return self.hnsw_service
        
    def index_document(self, file_path: str, session_id: str = None) -> Dict[str, Any]:
        """Index a document through the complete pipeline with performance logging."""
        logger.info(f"Starting indexing pipeline for {file_path}")
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Get file info for logging
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Start performance logging session
        performance_logger.start_session(session_id, file_name, file_size)
        
        try:
            # Phase 1: Load and chunk document
            with performance_logger.step("Document Loading", file_path=file_path):
                processed_doc = self.document_loader.load_document(file_path)
                if not processed_doc:
                    performance_logger.end_session('failed', 'Failed to load document')
                    return {'success': False, 'error': 'Failed to load document'}
                
                # Add document metadata to step
                performance_logger.add_step_metadata(
                    chunks_created=len(processed_doc.chunks),
                    total_tokens=processed_doc.total_tokens,
                    parsing_method=processed_doc.metadata.get('parsing_method', 'unknown')
                )
            
            # Phase 2: Graph Decomposition
            with performance_logger.step("Graph Decomposition", total_chunks=len(processed_doc.chunks)):
                decomposition_result = self._phase_1_decomposition(processed_doc)
                if not decomposition_result['success']:
                    performance_logger.end_session('failed', f"Decomposition failed: {decomposition_result.get('error')}")
                    return decomposition_result
                
                performance_logger.add_step_metadata(
                    chunks_processed=decomposition_result['chunks_processed'],
                    chunks_failed=decomposition_result['chunks_failed']
                )
            
            # Phase 3: Graph Augmentation
            with performance_logger.step("Graph Augmentation"):
                augmentation_result = self._phase_2_augmentation()
                if not augmentation_result['success']:
                    performance_logger.end_session('failed', f"Augmentation failed: {augmentation_result.get('error')}")
                    return augmentation_result
                
                performance_logger.add_step_metadata(
                    important_entities=augmentation_result['important_entities'],
                    attribute_nodes=augmentation_result['attribute_nodes'],
                    communities_detected=augmentation_result['communities_detected'],
                    high_level_nodes=augmentation_result['high_level_nodes']
                )
            
            # Phase 4: Embedding Generation and Storage
            with performance_logger.step("Embedding Generation & Storage"):
                embedding_result = self._phase_3_embedding_generation()
                if not embedding_result['success']:
                    performance_logger.end_session('failed', f"Embedding generation failed: {embedding_result.get('error')}")
                    return embedding_result
                
                performance_logger.add_step_metadata(
                    embeddings_generated=embedding_result['embeddings_generated'],
                    hnsw_indexed=embedding_result['hnsw_indexed']
                )
            
            # Final step: Save graph
            with performance_logger.step("Graph Storage"):
                save_success = self.save_graph()
                performance_logger.add_step_metadata(graph_saved=save_success)
            
            # Get final stats
            stats = self.graph_manager.get_stats()
            
            # End performance logging session
            completed_session = performance_logger.end_session('completed')
            
            logger.info(f"Document indexing completed in {completed_session.total_duration_formatted}")
            
            result = {
                'success': True,
                'session_id': session_id,
                'processing_time': completed_session.total_duration,
                'document_metadata': processed_doc.metadata,
                'graph_stats': stats,
                'chunks_processed': len(processed_doc.chunks),
                'performance_summary': completed_session.summary,
                'performance_report_available': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in indexing pipeline: {e}")
            performance_logger.end_session('failed', str(e))
            return {
                'success': False, 
                'error': str(e),
                'session_id': session_id
            }
    
    def _phase_1_decomposition(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """Phase I: Extract base nodes (T, S, N, R) from document chunks."""
        logger.info("Starting Phase I: Graph Decomposition")
        
        total_chunks = len(processed_doc.chunks)
        successful_chunks = 0
        failed_chunks = 0
        
        try:
            # Process chunks sequentially for now (can be parallelized later)
            with performance_logger.step("Chunk Processing", total_chunks=total_chunks):
                for i, chunk in enumerate(processed_doc.chunks):
                    with performance_logger.step(f"Process Chunk {i+1}", 
                                                chunk_index=chunk.chunk_index,
                                                chunk_size=len(chunk.content)):
                        success = self._process_chunk(chunk)
                        if success:
                            successful_chunks += 1
                        else:
                            failed_chunks += 1
                        
                        performance_logger.add_step_metadata(
                            success=success,
                            total_processed=successful_chunks + failed_chunks,
                            success_rate=(successful_chunks / (successful_chunks + failed_chunks)) * 100
                        )
                    
                    # Log progress every 10 chunks
                    if (successful_chunks + failed_chunks) % 10 == 0:
                        logger.info(f"Processed {successful_chunks + failed_chunks}/{total_chunks} chunks")
                
                performance_logger.add_step_metadata(
                    successful_chunks=successful_chunks,
                    failed_chunks=failed_chunks,
                    success_rate=(successful_chunks / total_chunks) * 100 if total_chunks > 0 else 0
                )
            
            logger.info(f"Phase I completed: {successful_chunks} successful, {failed_chunks} failed")
            
            return {
                'success': True,
                'chunks_processed': successful_chunks,
                'chunks_failed': failed_chunks
            }
            
        except Exception as e:
            logger.error(f"Error in Phase I decomposition: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_chunk(self, chunk: DocumentChunk) -> bool:
        """Process a single chunk to extract T, S, N, R nodes."""
        try:
            chunk_id = str(uuid.uuid4())
            
            with performance_logger.step("Create Text Node"):
                # Create Text Node (T)
                text_node = Node(
                    id=f"T_{chunk_id}",
                    type=NodeType.TEXT,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        'node_type': 'text',
                        'chunk_id': chunk_id
                    }
                )
                self.graph_manager.add_node(text_node)
                performance_logger.add_step_metadata(text_node_id=text_node.id)
            
            with performance_logger.step("LLM Extraction", content_length=len(chunk.content)):
                # Extract semantic units, entities, and relationships using LLM
                extraction_result = self.llm_service.extract_all_from_chunk(chunk.content)
                
                if not extraction_result.success:
                    logger.warning(f"LLM extraction failed for chunk {chunk.chunk_index}: {extraction_result.error_message}")
                    performance_logger.add_step_metadata(
                        extraction_success=False,
                        error_message=extraction_result.error_message
                    )
                    return False
                
                performance_logger.add_step_metadata(
                    semantic_units_found=len(extraction_result.semantic_units),
                    entities_found=len(extraction_result.entities),
                    relationships_found=len(extraction_result.relationships)
                )
            
            with performance_logger.step("Create Semantic Nodes", count=len(extraction_result.semantic_units)):
                # Create Semantic Unit nodes (S)
                semantic_node_ids = []
                for i, semantic_unit in enumerate(extraction_result.semantic_units):
                    semantic_id = f"S_{chunk_id}_{i}"
                    semantic_node = Node(
                        id=semantic_id,
                        type=NodeType.SEMANTIC,
                        content=semantic_unit,
                        metadata={
                            'chunk_id': chunk_id,
                            'semantic_index': i,
                            'source_file': chunk.metadata.get('file_path'),
                            'node_type': 'semantic'
                        }
                    )
                    
                    self.graph_manager.add_node(semantic_node)
                    semantic_node_ids.append(semantic_id)
                    
                    # Link to text node
                    edge = Edge(
                        source=text_node.id,
                        target=semantic_id,
                        relationship_type="contains_semantic_unit"
                    )
                    self.graph_manager.add_edge(edge)
                
                performance_logger.add_step_metadata(semantic_nodes_created=len(semantic_node_ids))
            
            with performance_logger.step("Create Entity Nodes", count=len(extraction_result.entities)):
                # Create Entity nodes (N) with deduplication
                entity_node_ids = []
                new_entities = 0
                merged_entities = 0
                
                for i, entity in enumerate(extraction_result.entities):
                    entity_id = f"N_{chunk_id}_{i}"
                    entity_node = Node(
                        id=entity_id,
                        type=NodeType.ENTITY,
                        content=entity,
                        metadata={
                            'chunk_id': chunk_id,
                            'entity_index': i,
                            'source_file': chunk.metadata.get('file_path'),
                            'mentions': [chunk_id],
                            'node_type': 'entity'
                        }
                    )
                    
                    # Add node (deduplication handled in graph_manager)
                    added = self.graph_manager.add_node(entity_node)
                    if added:
                        entity_node_ids.append(entity_id)
                        new_entities += 1
                    else:
                        # Entity was merged, find the existing entity ID
                        existing_mentions = self.graph_manager.get_entity_mentions(entity)
                        if existing_mentions:
                            entity_node_ids.append(existing_mentions[0].id)
                            merged_entities += 1
                    
                    # Link to text node
                    if entity_node_ids:
                        edge = Edge(
                            source=text_node.id,
                            target=entity_node_ids[-1],
                            relationship_type="mentions_entity"
                        )
                        self.graph_manager.add_edge(edge)
                
                performance_logger.add_step_metadata(
                    entity_nodes_created=new_entities,
                    entities_merged=merged_entities,
                    total_entity_links=len(entity_node_ids)
                )
            
            with performance_logger.step("Create Relationship Nodes", count=len(extraction_result.relationships)):
                # Create Relationship nodes (R)
                relationships_created = 0
                relationships_skipped = 0
                
                for i, (entity1, relation, entity2) in enumerate(extraction_result.relationships):
                    # Find entity node IDs
                    entity1_nodes = self.graph_manager.get_entity_mentions(entity1)
                    entity2_nodes = self.graph_manager.get_entity_mentions(entity2)
                    
                    if entity1_nodes and entity2_nodes:
                        relationship_id = f"R_{chunk_id}_{i}"
                        relationship_node = Node(
                            id=relationship_id,
                            type=NodeType.RELATIONSHIP,
                            content=f"{entity1} {relation} {entity2}",
                            metadata={
                                'chunk_id': chunk_id,
                                'relationship_index': i,
                                'entity1': entity1,
                                'relation': relation,
                                'entity2': entity2,
                                'source_file': chunk.metadata.get('file_path'),
                                'node_type': 'relationship'
                            }
                        )
                        
                        self.graph_manager.add_node(relationship_node)
                        relationships_created += 1
                        
                        # Link to entities
                        edge1 = Edge(
                            source=relationship_id,
                            target=entity1_nodes[0].id,
                            relationship_type="involves_entity"
                        )
                        edge2 = Edge(
                            source=relationship_id,
                            target=entity2_nodes[0].id,
                            relationship_type="involves_entity"
                        )
                        
                        self.graph_manager.add_edge(edge1)
                        self.graph_manager.add_edge(edge2)
                        
                        # Direct relationship between entities
                        entity_edge = Edge(
                            source=entity1_nodes[0].id,
                            target=entity2_nodes[0].id,
                            relationship_type=relation,
                            metadata={'relationship_node': relationship_id}
                        )
                        self.graph_manager.add_edge(entity_edge)
                    else:
                        relationships_skipped += 1
                
                performance_logger.add_step_metadata(
                    relationships_created=relationships_created,
                    relationships_skipped=relationships_skipped
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
            performance_logger.add_step_metadata(error_message=str(e))
            return False
    
    def _phase_2_augmentation(self) -> Dict[str, Any]:
        """Phase II: Generate augmented nodes (A, H, O)."""
        logger.info("Starting Phase II: Graph Augmentation")
        
        try:
            # Step 1: Identify important entities
            with performance_logger.step("Identify Important Entities"):
                important_entities = self.graph_manager.get_important_entities(
                    Config.IMPORTANT_ENTITY_PERCENTAGE
                )
                logger.info(f"Identified {len(important_entities)} important entities")
                performance_logger.add_step_metadata(
                    important_entities_count=len(important_entities),
                    percentage_threshold=Config.IMPORTANT_ENTITY_PERCENTAGE
                )
            
            # Step 2: Generate Attribute nodes (A) for important entities
            with performance_logger.step("Generate Attribute Nodes", target_count=len(important_entities)):
                attribute_nodes_created = 0
                for i, entity in enumerate(important_entities):
                    with performance_logger.step(f"Create Attribute for Entity {i+1}", entity_id=entity.id):
                        success = self._create_attribute_node(entity)
                        if success:
                            attribute_nodes_created += 1
                        performance_logger.add_step_metadata(
                            success=success,
                            entity_content=entity.content[:100]  # First 100 chars
                        )
                
                logger.info(f"Created {attribute_nodes_created} attribute nodes")
                performance_logger.add_step_metadata(
                    attribute_nodes_created=attribute_nodes_created,
                    success_rate=(attribute_nodes_created / len(important_entities)) * 100 if important_entities else 0
                )
            
            # Step 3: Detect communities
            with performance_logger.step("Community Detection"):
                community_assignments = self.graph_manager.detect_communities(
                    resolution=Config.LEIDEN_RESOLUTION,
                    random_state=Config.LEIDEN_RANDOM_STATE
                )
                
                if not community_assignments:
                    logger.warning("Community detection failed")
                    performance_logger.add_step_metadata(detection_failed=True)
                    return {'success': False, 'error': 'Community detection failed'}
                
                communities = set(community_assignments.values())
                performance_logger.add_step_metadata(
                    communities_detected=len(communities),
                    nodes_assigned=len(community_assignments)
                )
            
            # Step 4: Generate High-Level (H) and Overview (O) nodes
            with performance_logger.step("Generate Community Summaries", communities_count=len(communities)):
                high_level_nodes_created = 0
                overview_nodes_created = 0
                
                for i, community_id in enumerate(communities):
                    with performance_logger.step(f"Create Community {i+1} Nodes", community_id=community_id):
                        h_success, o_success = self._create_community_nodes(community_id)
                        if h_success:
                            high_level_nodes_created += 1
                        if o_success:
                            overview_nodes_created += 1
                        
                        performance_logger.add_step_metadata(
                            high_level_success=h_success,
                            overview_success=o_success
                        )
                
                logger.info(f"Created {high_level_nodes_created} high-level nodes and {overview_nodes_created} overview nodes")
                performance_logger.add_step_metadata(
                    high_level_nodes_created=high_level_nodes_created,
                    overview_nodes_created=overview_nodes_created
                )
            
            return {
                'success': True,
                'important_entities': len(important_entities),
                'attribute_nodes': attribute_nodes_created,
                'communities_detected': len(communities),
                'high_level_nodes': high_level_nodes_created,
                'overview_nodes': overview_nodes_created
            }
            
        except Exception as e:
            logger.error(f"Error in Phase II augmentation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_attribute_node(self, entity: Node) -> bool:
        """Create an attribute node for an important entity."""
        try:
            # Gather context from connected nodes
            connected_nodes = self.graph_manager.get_connected_nodes(entity.id)
            context_chunks = []
            
            for node in connected_nodes:
                if node.type == NodeType.TEXT:
                    context_chunks.append(node.content)
            
            # Add the entity's own context
            if hasattr(entity, 'content'):
                context_chunks.append(f"Entity: {entity.content}")
            
            # Get relationships for this entity
            entity_relationships = []
            for neighbor in self.graph_manager.graph.neighbors(entity.id):
                neighbor_node = self.graph_manager.get_node(neighbor)
                if neighbor_node and neighbor_node.type == NodeType.RELATIONSHIP:
                    entity_relationships.append(neighbor_node.content)
            
            # Generate attributes using LLM with relationships
            attributes = self.llm_service.generate_entity_attributes(
                entity.content, 
                context_chunks[:5],  # Limit context to avoid token limits
                entity_relationships[:3]  # Include up to 3 relationships
            )
            
            # Create attribute node
            attribute_id = f"A_{entity.id}"
            attribute_node = Node(
                id=attribute_id,
                type=NodeType.ATTRIBUTE,
                content=attributes,
                metadata={
                    'entity_id': entity.id,
                    'entity_name': entity.content,
                    'node_type': 'attribute'
                }
            )
            
            self.graph_manager.add_node(attribute_node)
            
            # Link to entity
            edge = Edge(
                source=attribute_id,
                target=entity.id,
                relationship_type="describes_entity"
            )
            self.graph_manager.add_edge(edge)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating attribute node for entity {entity.id}: {e}")
            return False
    
    def _create_community_nodes(self, community_id: int) -> tuple[bool, bool]:
        """Create high-level and overview nodes for a community."""
        try:
            # Get community nodes
            community_nodes = self.graph_manager.get_community_nodes(community_id)
            
            if not community_nodes:
                return False, False
            
            # Convert to dict format for LLM service
            community_data = []
            for node in community_nodes:
                community_data.append({
                    'content': node.content,
                    'type': node.type.value,
                    'metadata': node.metadata
                })
            
            # Generate high-level summary
            high_level_summary = self.llm_service.generate_community_summary(community_data)
            
            # Create High-Level node
            high_level_id = f"H_community_{community_id}"
            high_level_node = Node(
                id=high_level_id,
                type=NodeType.HIGH_LEVEL,
                content=high_level_summary,
                metadata={
                    'community_id': community_id,
                    'node_count': len(community_nodes),
                    'node_type': 'high_level'
                }
            )
            
            h_success = self.graph_manager.add_node(high_level_node)
            
            # Generate overview title
            overview_title = self.llm_service.generate_community_overview(high_level_summary)
            
            # Create Overview node
            overview_id = f"O_community_{community_id}"
            overview_node = Node(
                id=overview_id,
                type=NodeType.OVERVIEW,
                content=overview_title,
                metadata={
                    'community_id': community_id,
                    'high_level_node': high_level_id,
                    'node_type': 'overview'
                }
            )
            
            o_success = self.graph_manager.add_node(overview_node)
            
            # Link overview to high-level
            if h_success and o_success:
                edge = Edge(
                    source=overview_id,
                    target=high_level_id,
                    relationship_type="summarizes"
                )
                self.graph_manager.add_edge(edge)
            
            # Link high-level to community nodes
            if h_success:
                for node in community_nodes[:10]:  # Limit connections
                    edge = Edge(
                        source=high_level_id,
                        target=node.id,
                        relationship_type="summarizes_content"
                    )
                    self.graph_manager.add_edge(edge)
            
            return h_success, o_success
            
        except Exception as e:
            logger.error(f"Error creating community nodes for community {community_id}: {e}")
            return False, False
    
    def save_graph(self, filepath: str = None) -> bool:
        """Save the current graph to disk."""
        if filepath is None:
            filepath = Config.GRAPH_DB_PATH
        
        return self.graph_manager.save_graph(filepath)
    
    def load_graph(self, filepath: str = None) -> bool:
        """Load a graph from disk."""
        if filepath is None:
            filepath = Config.GRAPH_DB_PATH
        
        return self.graph_manager.load_graph(filepath)
    
    def _phase_3_embedding_generation(self) -> Dict[str, Any]:
        """Phase III: Generate embeddings for all nodes."""
        logger.info("Starting Phase III: Embedding Generation")
        
        try:
            # Initialize HNSW service
            with performance_logger.step("Initialize HNSW Service"):
                hnsw_service = self._get_hnsw_service()
                logger.info(f"HNSW service initialized: {hnsw_service is not None}")
                performance_logger.add_step_metadata(hnsw_initialized=hnsw_service is not None)
            
            # Collect nodes for embedding
            with performance_logger.step("Collect Nodes for Embedding"):
                all_nodes = []
                node_type_counts = {}
                
                for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                                 NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                    nodes = self.graph_manager.get_nodes_by_type(node_type)
                    all_nodes.extend(nodes)
                    node_type_counts[node_type.value] = len(nodes)
                
                if not all_nodes:
                    logger.warning("No nodes found for embedding generation")
                    performance_logger.add_step_metadata(no_nodes_found=True)
                    return {'success': True, 'embeddings_generated': 0}
                
                logger.info(f"Generating embeddings for {len(all_nodes)} nodes")
                performance_logger.add_step_metadata(
                    total_nodes=len(all_nodes),
                    node_type_breakdown=node_type_counts
                )
            
            # Prepare embedding data
            with performance_logger.step("Prepare Embedding Data"):
                texts = [node.content for node in all_nodes]
                node_ids = [node.id for node in all_nodes]
                batch_size = Config.DEFAULT_BATCH_SIZE
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                performance_logger.add_step_metadata(
                    batch_size=batch_size,
                    total_batches=total_batches,
                    total_texts=len(texts)
                )
            
            # Generate embeddings in batches
            with performance_logger.step("Generate Embeddings", total_batches=total_batches):
                embeddings_generated = 0
                hnsw_indexed = 0
                
                for i in range(0, len(texts), batch_size):
                    batch_num = (i // batch_size) + 1
                    batch_texts = texts[i:i + batch_size]
                    batch_node_ids = node_ids[i:i + batch_size]
                    
                    with performance_logger.step(f"Process Batch {batch_num}", 
                                                batch_size=len(batch_texts),
                                                batch_range=f"{i}-{i+len(batch_texts)}"):
                        
                        # Generate embeddings for this batch
                        embeddings = self.llm_service.get_embeddings(batch_texts, len(batch_texts))
                        
                        if len(embeddings) != len(batch_texts):
                            logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(batch_texts)}")
                            performance_logger.add_step_metadata(
                                embedding_mismatch=True,
                                expected=len(batch_texts),
                                received=len(embeddings)
                            )
                            continue
                        
                        # Update nodes and add to HNSW index
                        batch_indexed = 0
                        for node_id, embedding in zip(batch_node_ids, embeddings):
                            node = self.graph_manager.get_node(node_id)
                            if node:
                                node.embeddings = embedding
                                self.graph_manager.update_node(node)
                                embeddings_generated += 1
                                
                                # Add to HNSW index for vector search
                                metadata = {
                                    'type': node.type.value,
                                    'content': node.content[:200]  # Store first 200 chars for metadata
                                }
                                success = hnsw_service.add_node_embedding(
                                    node_id=node_id,
                                    embedding=embedding,
                                    metadata=metadata
                                )
                                
                                if success:
                                    batch_indexed += 1
                                    hnsw_indexed += 1
                                else:
                                    logger.warning(f"Failed to add node {node_id} to HNSW index")
                        
                        performance_logger.add_step_metadata(
                            embeddings_in_batch=len(embeddings),
                            nodes_updated=len(batch_texts),
                            hnsw_indexed_in_batch=batch_indexed
                        )
                    
                    logger.info(f"Generated embeddings for batch {batch_num}/{total_batches}")
                
                performance_logger.add_step_metadata(
                    total_embeddings_generated=embeddings_generated,
                    total_hnsw_indexed=hnsw_indexed
                )
            
            # Save HNSW index to disk
            with performance_logger.step("Save HNSW Index"):
                try:
                    hnsw_saved = hnsw_service.save_index()
                    logger.info(f"HNSW index saved: {hnsw_saved}")
                    performance_logger.add_step_metadata(
                        save_success=hnsw_saved,
                        index_size=hnsw_service.current_count if hasattr(hnsw_service, 'current_count') else 0
                    )
                except Exception as e:
                    logger.warning(f"Failed to save HNSW index: {e}")
                    performance_logger.add_step_metadata(
                        save_failed=True,
                        error_message=str(e)
                    )
            
            logger.info(f"Phase III completed: {embeddings_generated} embeddings generated and indexed")
            
            return {
                'success': True,
                'embeddings_generated': embeddings_generated,
                'hnsw_indexed': hnsw_indexed
            }
            
        except Exception as e:
            logger.error(f"Error in Phase III embedding generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'embeddings_generated': 0
            }
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get statistics about the current indexed content."""
        return self.graph_manager.get_stats()
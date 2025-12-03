import logging
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

from ..graph.graph_manager import GraphManager
from ..graph.node_types import Node, Edge, NodeType
from ..llm.llm_service import LLMService
from .document_loader import DocumentLoader, ProcessedDocument, DocumentChunk
from .llamaparse_service import EnhancedDocumentLoader
from ..vector.hnsw_service import HNSWService
from ..config.settings import Config

logger = logging.getLogger(__name__)

class IndexingPipeline:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.llm_service = LLMService()
        self.hnsw_service = None  # Will be initialized when needed
        
        # Org-level state tracking
        self.is_incremental_mode = False
        self.current_org_id = None
        self.current_file_id = None
        
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
        
    def index_document(self, file_path: str) -> Dict[str, Any]:
        """Index a document through the complete pipeline."""
        logger.info(f"Starting indexing pipeline for {file_path}")
        start_time = time.time()
        
        # Phase 1: Load and chunk document
        processed_doc = self.document_loader.load_document(file_path)
        if not processed_doc:
            return {'success': False, 'error': 'Failed to load document'}
        
        # Phase 2: Graph Decomposition
        decomposition_result = self._phase_1_decomposition(processed_doc)
        if not decomposition_result['success']:
            return decomposition_result
        
        # Phase 2: Graph Augmentation
        augmentation_result = self._phase_2_augmentation()
        if not augmentation_result['success']:
            return augmentation_result
        
        # Phase 3: Embedding Generation
        embedding_result = self._phase_3_embedding_generation()
        if not embedding_result['success']:
            return embedding_result
        
        processing_time = time.time() - start_time
        stats = self.graph_manager.get_stats()
        
        logger.info(f"Document indexing completed in {processing_time:.2f} seconds")
        
        return {
            'success': True,
            'processing_time': processing_time,
            'document_metadata': processed_doc.metadata,
            'graph_stats': stats,
            'chunks_processed': len(processed_doc.chunks)
        }
    
    def _phase_1_decomposition(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """Phase I: Extract base nodes (T, S, N, R) from document chunks."""
        logger.info("Starting Phase I: Graph Decomposition")
        
        total_chunks = len(processed_doc.chunks)
        successful_chunks = 0
        failed_chunks = 0
        
        try:
            if Config.ENABLE_PARALLEL_PROCESSING and total_chunks >= Config.MAX_CONCURRENT_CHUNKS:
                # Parallel processing for better performance
                logger.info(f"🚀 Using parallel processing with {Config.MAX_CONCURRENT_CHUNKS} workers")
                
                # Thread-safe counters
                success_lock = threading.Lock()
                
                with ThreadPoolExecutor(max_workers=Config.MAX_CONCURRENT_CHUNKS) as executor:
                    # Submit all chunk processing tasks
                    future_to_chunk = {executor.submit(self._process_chunk, chunk): chunk for chunk in processed_doc.chunks}
                    
                    # Process completed tasks
                    for i, future in enumerate(as_completed(future_to_chunk)):
                        try:
                            success = future.result()
                            with success_lock:
                                if success:
                                    successful_chunks += 1
                                else:
                                    failed_chunks += 1
                                
                                # Log progress
                                total_processed = successful_chunks + failed_chunks
                                if total_processed % 10 == 0 or total_processed == total_chunks:
                                    logger.info(f"Processed {total_processed}/{total_chunks} chunks (parallel)")
                        except Exception as e:
                            logger.error(f"Error processing chunk: {e}")
                            with success_lock:
                                failed_chunks += 1
            else:
                # Sequential processing (fallback or small datasets)
                logger.info("🔄 Using sequential processing")
                for chunk in processed_doc.chunks:
                    success = self._process_chunk(chunk)
                    if success:
                        successful_chunks += 1
                    else:
                        failed_chunks += 1
                    
                    # Log progress
                    if (successful_chunks + failed_chunks) % 10 == 0:
                        logger.info(f"Processed {successful_chunks + failed_chunks}/{total_chunks} chunks")
            
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
            
            # Extract semantic units, entities, and relationships using LLM
            extraction_result = self.llm_service.extract_all_from_chunk(chunk.content)
            
            if not extraction_result.success:
                logger.warning(f"LLM extraction failed for chunk {chunk.chunk_index}: {extraction_result.error_message}")
                return False
            
            # Create Semantic Unit nodes (S)
            semantic_node_ids = []
            for i, semantic_unit in enumerate(extraction_result.semantic_units):
                semantic_id = f"S_{chunk_id}_{i}"
                semantic_node = Node(
                    id=semantic_id,
                    type=NodeType.SEMANTIC,
                    content=semantic_unit,
                    metadata={
                        **chunk.metadata,  # Include file_id, org_id, user_id etc.
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
            
            # Create Entity nodes (N) with deduplication
            entity_node_ids = []
            for i, entity in enumerate(extraction_result.entities):
                entity_id = f"N_{chunk_id}_{i}"
                entity_node = Node(
                    id=entity_id,
                    type=NodeType.ENTITY,
                    content=entity,
                    metadata={
                        **chunk.metadata,  # Include file_id, org_id, user_id etc.
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
                else:
                    # Entity was merged, find the existing entity ID
                    existing_mentions = self.graph_manager.get_entity_mentions(entity)
                    if existing_mentions:
                        entity_node_ids.append(existing_mentions[0].id)
                
                # Link to text node
                if entity_node_ids:
                    edge = Edge(
                        source=text_node.id,
                        target=entity_node_ids[-1],
                        relationship_type="mentions_entity"
                    )
                    self.graph_manager.add_edge(edge)
            
            # Create Relationship nodes (R)
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
                            **chunk.metadata,  # Include file_id, org_id, user_id etc.
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
            return False
    
    def _phase_2_augmentation(self) -> Dict[str, Any]:
        """Phase II: Generate augmented nodes (A, H, O)."""
        logger.info("Starting Phase II: Graph Augmentation")
        
        try:
            # Step 1: Identify important entities
            important_entities = self.graph_manager.get_important_entities(
                Config.IMPORTANT_ENTITY_PERCENTAGE
            )
            logger.info(f"Identified {len(important_entities)} important entities")
            
            # Step 2: Generate Attribute nodes (A) for important entities
            attribute_nodes_created = 0
            for entity in important_entities:
                success = self._create_attribute_node(entity)
                if success:
                    attribute_nodes_created += 1
            
            logger.info(f"Created {attribute_nodes_created} attribute nodes")
            
            # Step 3: Detect communities
            community_assignments = self.graph_manager.detect_communities(
                resolution=Config.LEIDEN_RESOLUTION,
                random_state=Config.LEIDEN_RANDOM_STATE
            )
            
            if not community_assignments:
                logger.warning("Community detection failed")
                return {'success': False, 'error': 'Community detection failed'}
            
            # Step 4: Generate High-Level (H) and Overview (O) nodes
            communities = set(community_assignments.values())
            high_level_nodes_created = 0
            overview_nodes_created = 0
            
            for community_id in communities:
                h_success, o_success = self._create_community_nodes(community_id)
                if h_success:
                    high_level_nodes_created += 1
                if o_success:
                    overview_nodes_created += 1
            
            logger.info(f"Created {high_level_nodes_created} high-level nodes and {overview_nodes_created} overview nodes")
            
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
            
            # Create attribute node with org-specific ID
            attribute_id = f"A_{entity.id}_{self.current_org_id}" if self.current_org_id else f"A_{entity.id}"
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
            
            # Create High-Level node with org-specific ID
            high_level_id = f"H_community_{community_id}_{self.current_org_id}" if self.current_org_id else f"H_community_{community_id}"
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
            
            # Create Overview node with org-specific ID
            overview_id = f"O_community_{community_id}_{self.current_org_id}" if self.current_org_id else f"O_community_{community_id}"
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
        
        # Get HNSW service instance
        hnsw_service = self._get_hnsw_service()
        logger.info(f"HNSW service initialized: {hnsw_service is not None}")
        
        try:
            # Get all nodes that need embeddings
            all_nodes = []
            for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                             NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                nodes = self.graph_manager.get_nodes_by_type(node_type)
                all_nodes.extend(nodes)
            
            if not all_nodes:
                logger.warning("No nodes found for embedding generation")
                return {'success': True, 'embeddings_generated': 0}
            
            logger.info(f"Generating embeddings for {len(all_nodes)} nodes")
            
            # Prepare texts for embedding
            texts = [node.content for node in all_nodes]
            node_ids = [node.id for node in all_nodes]
            
            # Generate ALL embeddings at once with optimized batching
            logger.info(f"🚀 Generating embeddings for {len(texts)} nodes using optimized batching...")
            
            embeddings = self.llm_service.get_embeddings(texts)
            
            if len(embeddings) != len(texts):
                logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")
                return {'success': False, 'error': 'Embedding generation failed'}
            
            logger.info(f"✅ Successfully generated {len(embeddings)} embeddings")
            
            # Update nodes with embeddings and add to HNSW index in optimized batches
            embeddings_generated = 0
            hnsw_batch_size = 500  # Process HNSW updates in batches for better performance
            
            for i in range(0, len(all_nodes), hnsw_batch_size):
                batch_start = i
                batch_end = min(i + hnsw_batch_size, len(all_nodes))
                
                # Update nodes with embeddings
                for j in range(batch_start, batch_end):
                    node_id = node_ids[j]
                    embedding = embeddings[j]
                    
                    node = self.graph_manager.get_node(node_id)
                    if node:
                        node.embeddings = embedding
                        self.graph_manager.update_node(node)
                        
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
                            embeddings_generated += 1
                        else:
                            logger.warning(f"Failed to add node {node_id} to HNSW index")
                
                batch_num = (batch_start // hnsw_batch_size) + 1
                total_batches = (len(all_nodes) + hnsw_batch_size - 1) // hnsw_batch_size
                logger.info(f"✅ HNSW batch {batch_num}/{total_batches}: {batch_end - batch_start} nodes indexed")
            
            # Save HNSW index to disk
            try:
                hnsw_saved = hnsw_service.save_index()
                logger.info(f"HNSW index saved: {hnsw_saved}")
            except Exception as e:
                logger.warning(f"Failed to save HNSW index: {e}")
            
            logger.info(f"Phase III completed: {embeddings_generated} embeddings generated and indexed")
            
            return {
                'success': True,
                'embeddings_generated': embeddings_generated,
                'hnsw_indexed': embeddings_generated
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
    
    # ==================== INCREMENTAL PROCESSING METHODS ====================
    
    def set_incremental_mode(self, org_id: str, file_id: str, existing_graph_data: bytes = None):
        """Set pipeline to incremental mode for org-level processing"""
        self.is_incremental_mode = True
        self.current_org_id = org_id
        self.current_file_id = file_id
        
        if existing_graph_data:
            # Load existing org graph
            success = self.graph_manager.load_from_data(existing_graph_data)
            if success:
                logger.info(f"Loaded existing org graph for {org_id}: {self.graph_manager.graph.number_of_nodes()} nodes")
            else:
                logger.error(f"Failed to load existing graph for {org_id}")
                self.is_incremental_mode = False
        else:
            logger.info(f"Starting fresh org graph for {org_id}")
    
    def _phase_2_incremental_augmentation(self, new_entities: List[Node]) -> Dict[str, Any]:
        """Incremental augmentation - only process new entities"""
        start_time = time.time()
        logger.info("🔄 Starting Phase II Incremental Augmentation")
        
        try:
            # Only create attribute nodes for truly new entities
            attribute_nodes_created = 0
            
            logger.info(f"Processing attributes for {len(new_entities)} new entities")
            
            for entity in new_entities:
                if self._create_attribute_node(entity):
                    attribute_nodes_created += 1
                    
                    if attribute_nodes_created % 10 == 0:
                        logger.info(f"Created {attribute_nodes_created} attribute nodes...")
            
            # Only run community detection if we have significant new content
            communities = {}
            high_level_nodes_created = 0
            overview_nodes_created = 0
            
            if len(new_entities) > 5:  # Only if substantial new content
                logger.info("Running incremental community detection")
                communities = self.graph_manager.detect_communities()
                
                # Create community nodes for new/updated communities only
                for community_id in set(communities.values()):
                    # Check if this community already has high-level nodes
                    existing_high_level = any(
                        node.metadata.get('community_id') == community_id
                        for node in self.graph_manager.get_nodes_by_type(NodeType.HIGH_LEVEL)
                    )
                    
                    if not existing_high_level:
                        high_level_created, overview_created = self._create_community_nodes(community_id)
                        if high_level_created:
                            high_level_nodes_created += 1
                        if overview_created:
                            overview_nodes_created += 1
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Phase II Incremental completed in {processing_time:.2f} seconds")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'new_entities_processed': len(new_entities),
                'attribute_nodes': attribute_nodes_created,
                'communities_detected': len(communities),
                'high_level_nodes': high_level_nodes_created,
                'overview_nodes': overview_nodes_created,
                'incremental_mode': True
            }
            
        except Exception as e:
            logger.error(f"Error in Phase II incremental augmentation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _phase_3_incremental_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings only for new nodes"""
        start_time = time.time()
        logger.info("🔄 Starting Phase III Incremental Embedding Generation")
        
        try:
            # Get nodes without embeddings (newly added)
            nodes_to_embed = []
            node_ids = []
            
            for node_id, node_data in self.graph_manager.graph.nodes(data=True):
                if not node_data.get('embeddings'):
                    # Check if this is from current file being processed
                    if (node_data.get('metadata', {}).get('file_id') == self.current_file_id or
                        not node_data.get('metadata', {}).get('file_id')):  # Handle nodes without file_id
                        
                        node = self.graph_manager.get_node(node_id)
                        if node and hasattr(node, 'content') and len(node.content.strip()) > 10:
                            nodes_to_embed.append(node.content)
                            node_ids.append(node_id)
            
            if not nodes_to_embed:
                logger.info("No new nodes need embedding generation")
                return {
                    'success': True,
                    'embeddings_generated': 0,
                    'hnsw_indexed': 0,
                    'processing_time': time.time() - start_time,
                    'incremental_mode': True
                }
            
            logger.info(f"Generating embeddings for {len(nodes_to_embed)} new nodes")
            
            # Generate embeddings in batches
            embeddings = self.llm_service.get_embeddings(nodes_to_embed)
            
            if not embeddings or len(embeddings) != len(nodes_to_embed):
                raise Exception(f"Embedding generation failed: expected {len(nodes_to_embed)}, got {len(embeddings or [])}")
            
            # Get HNSW service
            hnsw_service = self._get_hnsw_service()
            
            # Update nodes with embeddings and add to HNSW
            embeddings_generated = 0
            
            for node_id, embedding in zip(node_ids, embeddings):
                node = self.graph_manager.get_node(node_id)
                if node:
                    node.embeddings = embedding
                    self.graph_manager.update_node(node)
                    
                    # Add to HNSW index
                    metadata = {
                        'type': node.type.value,
                        'content': node.content[:200],
                        'file_id': self.current_file_id,
                        'org_id': self.current_org_id
                    }
                    success = hnsw_service.add_node_embedding(
                        node_id=node_id,
                        embedding=embedding,
                        metadata=metadata
                    )
                    
                    if success:
                        embeddings_generated += 1
            
            # Save HNSW index
            try:
                hnsw_saved = hnsw_service.save_index()
                logger.info(f"HNSW index saved: {hnsw_saved}")
            except Exception as e:
                logger.warning(f"Failed to save HNSW index: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Phase III Incremental completed: {embeddings_generated} embeddings in {processing_time:.2f} seconds")
            
            return {
                'success': True,
                'embeddings_generated': embeddings_generated,
                'hnsw_indexed': embeddings_generated,
                'processing_time': processing_time,
                'incremental_mode': True
            }
            
        except Exception as e:
            logger.error(f"Error in Phase III incremental embedding generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'embeddings_generated': 0,
                'incremental_mode': True
            }
    
    def get_new_entities_from_current_file(self) -> List[Node]:
        """Get entity nodes that were added from the current file"""
        new_entities = []
        
        # Debug: log what we're looking for
        logger.info(f"Looking for entities with file_id: {self.current_file_id}")
        logger.info(f"NodeType.ENTITY.value: {NodeType.ENTITY.value}")
        
        entity_count = 0
        current_file_count = 0
        
        for node_id, node_data in self.graph_manager.graph.nodes(data=True):
            node_type = node_data.get('type')
            node_file_id = node_data.get('metadata', {}).get('file_id')
            
            if node_type == NodeType.ENTITY.value:
                entity_count += 1
                if node_file_id == self.current_file_id:
                    current_file_count += 1
                    node = self.graph_manager.get_node(node_id)
                    if node:
                        new_entities.append(node)
        
        logger.info(f"Total entities in graph: {entity_count}")
        logger.info(f"Entities from current file ({self.current_file_id}): {current_file_count}")
        logger.info(f"New entities found: {len(new_entities)}")
        
        # Debug: show a few examples of what we found
        if entity_count > 0 and current_file_count == 0:
            logger.warning("No entities found for current file. Sample entity file_ids:")
            sample_count = 0
            for node_id, node_data in self.graph_manager.graph.nodes(data=True):
                if node_data.get('type') == NodeType.ENTITY.value and sample_count < 3:
                    sample_file_id = node_data.get('metadata', {}).get('file_id')
                    logger.warning(f"  Entity {node_id}: file_id={sample_file_id}")
                    sample_count += 1
        
        return new_entities
import logging
import uuid
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..graph.graph_manager import GraphManager
from ..graph.node_types import Node, Edge, NodeType
from ..llm.llm_service import LLMService, EnhancedExtractionResult
from .document_loader import DocumentLoader, ProcessedDocument, DocumentChunk
from .llamaparse_service import EnhancedDocumentLoader
from ..vector.hnsw_service import HNSWService
from ..config.settings import Config

logger = logging.getLogger(__name__)

class IndexingPipeline:
    def __init__(self):
        self.graph_manager = GraphManager()
        self.llm_service = LLMService(model_type="gemini")
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
        
    def index_document(self, file_path: str) -> Dict[str, Any]:
        """Index a document through the complete pipeline with async optimization."""
        print(f"🚀 Starting NodeRAG pipeline for: {os.path.basename(file_path)}")
        start_time = time.time()
        
        # Phase 1: Load and chunk document
        print("📄 Loading and chunking document...")
        processed_doc = self.document_loader.load_document(file_path)
        if not processed_doc:
            print("❌ Failed to load document")
            return {'success': False, 'error': 'Failed to load document'}
        
        print(f"✅ Document loaded: {len(processed_doc.chunks)} chunks created")
        
        # Run phases with async optimization
        try:
            print("⚡ Running optimized async pipeline...")
            result = asyncio.run(self._async_index_pipeline(processed_doc))
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['document_metadata'] = processed_doc.metadata
            result['chunks_processed'] = len(processed_doc.chunks)
            
            print(f"\n🎉 Pipeline completed in {processing_time:.2f}s")
            print(f"📊 Graph stats: {result['graph_stats']['total_nodes']} nodes, {result['graph_stats']['total_edges']} edges")
            
            # Print phase timing breakdown
            if 'phase_timing' in result:
                timing = result['phase_timing']
                print(f"\n⏱️  Phase Performance Breakdown:")
                print(f"   Phase 1 (Decomposition): {timing['phase1_time']:.2f}s")
                print(f"   Phase 2 (Augmentation):  {timing['phase2_time']:.2f}s") 
                print(f"   Phase 3 (Embeddings):    {timing['phase3_time']:.2f}s")
                print(f"   Total Phases:            {timing['total_phases_time']:.2f}s")
                overhead = processing_time - timing['total_phases_time']
                print(f"   Overhead (Loading etc):  {overhead:.2f}s")
            
            # Print LLM token usage statistics
            llm_stats = self.llm_service.get_usage_stats()
            if llm_stats['total_tokens'] > 0:
                print(f"\n🤖 Gemini Token Usage:")
                print(f"   Input tokens:  {llm_stats['input_tokens']:,}")
                print(f"   Output tokens: {llm_stats['output_tokens']:,}")
                print(f"   Total tokens:  {llm_stats['total_tokens']:,}")
                print(f"   API calls:     {llm_stats['api_calls']}")
            
            return result
            
        except Exception as e:
            print(f"⚠️  Async pipeline error: {e}")
            print("🔄 Falling back to sync processing...")
            return self._sync_index_pipeline(processed_doc, start_time)
    
    async def _async_index_pipeline(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """Async pipeline with batch processing and overlapping phases."""
        
        # Phase 1: Batch decomposition (async)
        print("\n🔗 Phase 1: Graph Decomposition (Batch Processing)")
        phase1_start = time.time()
        decomposition_result = await self._async_phase_1_decomposition(processed_doc)
        phase1_time = time.time() - phase1_start
        if not decomposition_result['success']:
            return decomposition_result
        print(f"✅ Phase 1 complete: {decomposition_result['chunks_processed']} chunks processed in {phase1_time:.2f}s")
        
        # Phase 2: Parallel augmentation (async)
        print("\n🌟 Phase 2: Graph Augmentation (Parallel Processing)")
        phase2_start = time.time()
        augmentation_result = await self._async_phase_2_augmentation()
        phase2_time = time.time() - phase2_start
        if not augmentation_result['success']:
            return augmentation_result
        print(f"✅ Phase 2 complete: {augmentation_result['attribute_nodes']} attributes, {augmentation_result['communities_detected']} communities in {phase2_time:.2f}s")
        
        # Phase 3: Embedding generation (can overlap with storage)
        print("\n🧠 Phase 3: Embedding Generation")
        phase3_start = time.time()
        embedding_result = self._phase_3_embedding_generation()
        phase3_time = time.time() - phase3_start
        if not embedding_result['success']:
            return embedding_result
        print(f"✅ Phase 3 complete: {embedding_result['embeddings_generated']} embeddings generated in {phase3_time:.2f}s")
        
        try:
            stats = self.graph_manager.get_stats()
        except Exception as e:
            print(f"   ⚠️  Error getting graph stats: {e}")
            stats = {'total_nodes': 0, 'total_edges': 0, 'node_type_counts': {}}
        
        return {
            'success': True,
            'graph_stats': stats,
            'decomposition': decomposition_result,
            'augmentation': augmentation_result,
            'embeddings': embedding_result,
            'phase_timing': {
                'phase1_time': phase1_time,
                'phase2_time': phase2_time,
                'phase3_time': phase3_time,
                'total_phases_time': phase1_time + phase2_time + phase3_time
            }
        }
    
    def _sync_index_pipeline(self, processed_doc: ProcessedDocument, start_time: float) -> Dict[str, Any]:
        """Fallback sync pipeline."""
        # Phase 1: Graph Decomposition  
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
        
        return {
            'success': True,
            'processing_time': processing_time,
            'document_metadata': processed_doc.metadata,
            'graph_stats': stats,
            'chunks_processed': len(processed_doc.chunks)
        }
    
    def _phase_1_decomposition(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """Phase I: Extract base nodes (T, S, N, R) from document chunks with parallel processing."""
        logger.info("Starting Phase I: Graph Decomposition (Parallel)")
        
        total_chunks = len(processed_doc.chunks)
        successful_chunks = 0
        failed_chunks = 0
        
        try:
            # Use parallel processing with ThreadPoolExecutor
            max_workers = min(4, total_chunks)  # Limit to 4 concurrent workers
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunk processing tasks
                future_to_chunk = {
                    executor.submit(self._process_chunk, chunk): chunk 
                    for chunk in processed_doc.chunks
                }
                
                # Process completed futures as they finish
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        success = future.result()
                        if success:
                            successful_chunks += 1
                        else:
                            failed_chunks += 1
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.chunk_index}: {e}")
                        failed_chunks += 1
                    
                    # Log progress
                    completed = successful_chunks + failed_chunks
                    if completed % 5 == 0 or completed == total_chunks:
                        logger.info(f"Processed {completed}/{total_chunks} chunks (Success: {successful_chunks}, Failed: {failed_chunks})")
            
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
        """Phase II: Generate augmented nodes (A, H, O) with parallel processing."""
        logger.info("Starting Phase II: Graph Augmentation (Parallel)")
        
        try:
            # Step 1: Identify important entities
            important_entities = self.graph_manager.get_important_entities(
                Config.IMPORTANT_ENTITY_PERCENTAGE
            )
            logger.info(f"Identified {len(important_entities)} important entities")
            
            # Step 2: Generate Attribute nodes (A) for important entities in parallel
            attribute_nodes_created = 0
            max_workers = min(3, len(important_entities))  # Limit workers to avoid overwhelming LLM
            
            if important_entities and max_workers > 0:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit attribute generation tasks
                    future_to_entity = {
                        executor.submit(self._create_attribute_node, entity): entity 
                        for entity in important_entities
                    }
                    
                    # Process completed futures
                    for future in as_completed(future_to_entity):
                        entity = future_to_entity[future]
                        try:
                            success = future.result()
                            if success:
                                attribute_nodes_created += 1
                        except Exception as e:
                            logger.error(f"Error creating attribute for entity {entity.id}: {e}")
            
            logger.info(f"Created {attribute_nodes_created} attribute nodes")
            
            # Step 3: Detect communities
            community_assignments = self.graph_manager.detect_communities(
                resolution=Config.LEIDEN_RESOLUTION,
                random_state=Config.LEIDEN_RANDOM_STATE
            )
            
            if not community_assignments:
                logger.warning("Community detection failed")
                return {'success': False, 'error': 'Community detection failed'}
            
            # Step 4: Generate High-Level (H) and Overview (O) nodes in parallel
            communities = list(set(community_assignments.values()))
            high_level_nodes_created = 0
            overview_nodes_created = 0
            
            if communities:
                print("   📝 Generating community summaries...")
                max_workers = min(3, len(communities))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit community processing tasks
                    future_to_community = {
                        executor.submit(self._create_community_nodes, community_id): community_id 
                        for community_id in communities
                    }
                    
                    # Process completed futures
                    for future in as_completed(future_to_community):
                        community_id = future_to_community[future]
                        try:
                            h_success, o_success = future.result()
                            if h_success:
                                high_level_nodes_created += 1
                            if o_success:
                                overview_nodes_created += 1
                        except Exception as e:
                            logger.error(f"Error creating community nodes for community {community_id}: {e}")
            
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
        
        print("💾 Saving graph to disk...")
        success = self.graph_manager.save_graph(filepath)
        if success:
            print("✅ Graph saved successfully")
        else:
            print("❌ Graph save failed")
        return success
    
    def load_graph(self, filepath: str = None) -> bool:
        """Load a graph from disk."""
        if filepath is None:
            filepath = Config.GRAPH_DB_PATH
        
        return self.graph_manager.load_graph(filepath)
    
    def _phase_3_embedding_generation(self) -> Dict[str, Any]:
        """Phase III: Generate embeddings for all nodes with checkpointing."""
        # Get HNSW service instance
        hnsw_service = self._get_hnsw_service()
        
        try:
            # Get all nodes that need embeddings
            all_nodes = []
            for node_type in [NodeType.SEMANTIC, NodeType.ENTITY, NodeType.RELATIONSHIP, 
                             NodeType.ATTRIBUTE, NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                nodes = self.graph_manager.get_nodes_by_type(node_type)
                all_nodes.extend(nodes)
            
            if not all_nodes:
                print("   ⚠️  No nodes found for embedding generation")
                return {'success': True, 'embeddings_generated': 0}
            
            print(f"   🔢 Processing {len(all_nodes)} nodes in batches...")
            
            # Check for existing checkpoint and resume if possible
            checkpoint_data = self._load_embedding_checkpoint()
            start_batch = 0
            embeddings_generated = 0
            
            if checkpoint_data and Config.ENABLE_CHECKPOINTS:
                start_batch = checkpoint_data.get('last_completed_batch', 0)
                embeddings_generated = checkpoint_data.get('embeddings_generated', 0)
                if start_batch > 0:
                    print(f"   📁 Resuming from checkpoint: batch {start_batch + 1}, {embeddings_generated} embeddings done")
            
            # Prepare texts for embedding
            texts = [node.content for node in all_nodes]
            node_ids = [node.id for node in all_nodes]
            
            # Generate embeddings in batches with adaptive sizing
            batch_size = Config.get_adaptive_batch_size(len(all_nodes))
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            print(f"   📊 Environment: {'Render' if Config.IS_RENDER else 'Cloud' if Config.IS_CLOUD else 'Local'}")
            print(f"   📏 Batch size: {batch_size} (adaptive), Total batches: {total_batches}")
            
            for i in range(start_batch * batch_size, len(texts), batch_size):
                current_batch_num = i // batch_size + 1
                batch_texts = texts[i:i + batch_size]
                batch_node_ids = node_ids[i:i + batch_size]
                
                # Generate embeddings for this batch
                embeddings = self.llm_service.get_embeddings(batch_texts, batch_size)
                
                if len(embeddings) != len(batch_texts):
                    logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(batch_texts)}")
                    # Save checkpoint even for failed batches to track progress
                    self._save_embedding_checkpoint(current_batch_num - 1, embeddings_generated)
                    continue
                
                # Update nodes with embeddings and add to HNSW index
                batch_success_count = 0
                for node_id, embedding in zip(batch_node_ids, embeddings):
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
                            batch_success_count += 1
                            embeddings_generated += 1
                        else:
                            logger.warning(f"Failed to add node {node_id} to HNSW index")
                
                # Save checkpoint every N batches
                if Config.ENABLE_CHECKPOINTS and current_batch_num % Config.CHECKPOINT_INTERVAL == 0:
                    self._save_embedding_checkpoint(current_batch_num, embeddings_generated)
                    if Config.IS_RENDER:
                        print(f"   💾 Checkpoint saved after batch {current_batch_num}")
                
                print(f"   ✓ Batch {current_batch_num}/{total_batches} completed ({batch_success_count} embeddings)")
                
                # Memory and performance monitoring for cloud environments
                if Config.IS_CLOUD and current_batch_num % Config.MEMORY_CHECK_INTERVAL == 0:
                    try:
                        import psutil
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 85:  # High memory usage warning
                            print(f"   ⚠️  High memory usage: {memory_percent:.1f}%")
                            import gc
                            gc.collect()
                    except ImportError:
                        pass  # psutil not available, skip memory check
            
            # Save final checkpoint
            if Config.ENABLE_CHECKPOINTS:
                self._save_embedding_checkpoint(total_batches, embeddings_generated, completed=True)
                print("   📁 Final checkpoint saved")
            
            # Save HNSW index to disk
            print("   💾 Saving HNSW index to disk...")
            try:
                hnsw_saved = hnsw_service.save_index()
                if hnsw_saved:
                    print("   ✅ HNSW index saved successfully")
                else:
                    print("   ⚠️  HNSW index save failed")
            except Exception as e:
                print(f"   ❌ Failed to save HNSW index: {e}")
            
            # Clean up checkpoint file on successful completion
            if Config.ENABLE_CHECKPOINTS:
                self._cleanup_embedding_checkpoint()
            
            print(f"   📈 Total: {embeddings_generated} embeddings generated and indexed")
            
            logger.info(f"Phase III completed: {embeddings_generated} embeddings generated and indexed")
            
            return {
                'success': True,
                'embeddings_generated': embeddings_generated,
                'hnsw_indexed': embeddings_generated,
                'batch_size_used': batch_size,
                'environment': 'render' if Config.IS_RENDER else 'cloud' if Config.IS_CLOUD else 'local'
            }
            
        except Exception as e:
            logger.error(f"Error in Phase III embedding generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'embeddings_generated': 0
            }
    
    async def _async_phase_1_decomposition(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """
        RESTRUCTURED: Two-Phase Approach for Better Graph Quality
        Phase 1a: Parallel LLM Extraction (collect all results)
        Phase 1b: Unified Graph Building (build graph from all results)
        """
        total_chunks = len(processed_doc.chunks)
        print(f"🚀 NEW TWO-PHASE APPROACH: {total_chunks} chunks")
        print(f"📋 Phase 1a: Parallel LLM Extraction (collect all results)")
        print(f"🔗 Phase 1b: Unified Graph Building (build from all results)")
        
        try:
            # PHASE 1a: Parallel LLM Extraction Only (no graph building)
            all_extraction_results = await self._parallel_extract_all_chunks(processed_doc.chunks)
            
            successful_extractions = sum(1 for result in all_extraction_results if result['success'])
            failed_extractions = len(all_extraction_results) - successful_extractions
            
            print(f"✅ Phase 1a Complete: {successful_extractions}/{total_chunks} successful extractions")
            
            # PHASE 1b: Unified Graph Building
            print(f"🔗 Phase 1b: Building unified graph from all extracted data...")
            graph_build_result = await self._build_unified_graph_from_extractions(all_extraction_results, processed_doc.chunks)
            
            print(f"✅ Phase 1b Complete: {graph_build_result['total_nodes']} nodes, {graph_build_result['total_edges']} edges")
            print(f"   📊 Entity consolidation: {graph_build_result['consolidated_entities']} duplicates merged")
            print(f"   🔗 Cross-chunk relationships: {graph_build_result['cross_chunk_relationships']} detected")
            
            logger.info(f"Two-phase async decomposition completed: {successful_extractions} successful, {failed_extractions} failed")
            
            return {
                'success': True,
                'chunks_processed': successful_extractions,
                'chunks_failed': failed_extractions,
                'graph_quality': graph_build_result,
                'processing_method': 'two_phase_unified'
            }
            
        except Exception as e:
            logger.error(f"Error in Two-Phase Async Decomposition: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _async_phase_2_augmentation(self) -> Dict[str, Any]:
        """Async Phase II with batch attribute generation."""
        try:
            # Step 1: Identify important entities
            print("   🎯 Identifying important entities...")
            important_entities = self.graph_manager.get_important_entities(
                Config.IMPORTANT_ENTITY_PERCENTAGE
            )
            print(f"   ✓ Found {len(important_entities)} important entities")
            
            # Step 2: Batch generate attributes
            print("   🔨 Generating entity attributes in batch...")
            attribute_nodes_created = 0
            if important_entities:
                # Prepare batch data for entities
                entities_with_context = []
                for entity in important_entities:
                    connected_nodes = self.graph_manager.get_connected_nodes(entity.id)
                    context_chunks = [node.content for node in connected_nodes if node.type == NodeType.TEXT][:3]
                    
                    relationships = []
                    for neighbor in self.graph_manager.graph.neighbors(entity.id):
                        neighbor_node = self.graph_manager.get_node(neighbor)
                        if neighbor_node and neighbor_node.type == NodeType.RELATIONSHIP:
                            relationships.append(neighbor_node.content)
                    
                    entities_with_context.append((entity.content, context_chunks, relationships[:3]))
                
                # Batch generate attributes
                print(f"   📝 Generating attributes for {len(entities_with_context)} entities...")
                batch_attributes = await self.llm_service.generate_entity_attributes_batch(entities_with_context)
                
                # Create attribute nodes
                for i, (entity, attributes) in enumerate(zip(important_entities, batch_attributes)):
                    print(f"   🔍 Entity {i+1}: {entity.content}")
                    print(f"      Attributes: {attributes[:100]}{'...' if len(attributes) > 100 else ''}")
                    
                    success = self._create_attribute_node_from_text(entity, attributes)
                    if success:
                        attribute_nodes_created += 1
                        print(f"      ✅ Attribute node created")
                    else:
                        print(f"      ❌ Attribute node creation failed")
            
            # Step 3: Community detection and processing (keep sync for graph algorithms)
            print("   🌐 Detecting communities...")
            community_assignments = self.graph_manager.detect_communities(
                resolution=Config.LEIDEN_RESOLUTION,
                random_state=Config.LEIDEN_RANDOM_STATE
            )
            
            # Step 4: Community processing (can be parallelized with sync executor)
            if not community_assignments:
                print("   ⚠️  No communities detected (likely due to insufficient entities)")
                communities = []
            else:
                communities = list(set(community_assignments.values()))
                print(f"   ✓ Found {len(communities)} communities")
            high_level_nodes_created = 0
            overview_nodes_created = 0
            
            if communities:
                # Use executor for CPU-bound community processing
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=3) as executor:
                    tasks = []
                    for community_id in communities:
                        task = loop.run_in_executor(executor, self._create_community_nodes_with_debug, community_id)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    
                    for i, (h_success, o_success, debug_info) in enumerate(results):
                        print(f"   🏘️  Community {i+1}: {debug_info}")
                        if h_success:
                            high_level_nodes_created += 1
                        if o_success:
                            overview_nodes_created += 1
            
            logger.info(f"Async Phase II completed: {attribute_nodes_created} attributes, {high_level_nodes_created} high-level, {overview_nodes_created} overview")
            
            return {
                'success': True,
                'important_entities': len(important_entities),
                'attribute_nodes': attribute_nodes_created,
                'communities_detected': len(communities),
                'high_level_nodes': high_level_nodes_created,
                'overview_nodes': overview_nodes_created,
                'processing_method': 'async_batch'
            }
            
        except Exception as e:
            logger.error(f"Error in Async Phase II: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_nodes_from_result(self, chunk: DocumentChunk, result: 'EnhancedExtractionResult', chunk_id: str) -> bool:
        """Create nodes from extraction result."""
        try:
            # Create Text Node (T)
            text_node = Node(
                id=f"T_{chunk_id}",
                type=NodeType.TEXT,
                content=chunk.content,
                metadata={**chunk.metadata, 'node_type': 'text', 'chunk_id': chunk_id}
            )
            self.graph_manager.add_node(text_node)
            
            # Create Semantic Unit nodes (S)
            for i, semantic_unit_data in enumerate(result.semantic_units):
                semantic_id = f"S_{chunk_id}_{i}"
                semantic_content = semantic_unit_data.get('semantic_unit', '') if isinstance(semantic_unit_data, dict) else str(semantic_unit_data)
                
                semantic_node = Node(
                    id=semantic_id,
                    type=NodeType.SEMANTIC,
                    content=semantic_content,
                    metadata={'chunk_id': chunk_id, 'semantic_index': i, 'node_type': 'semantic'}
                )
                
                self.graph_manager.add_node(semantic_node)
                
                # Link to text node
                self.graph_manager.add_edge(Edge(
                    source=text_node.id, target=semantic_id, relationship_type="contains_semantic_unit"
                ))
            
            # Create Entity and Relationship nodes
            self._create_entity_and_relationship_nodes(result, chunk_id, text_node.id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating nodes from result: {e}")
            return False
    
    def _create_entity_and_relationship_nodes(self, result: 'EnhancedExtractionResult', chunk_id: str, text_node_id: str):
        """Create entity and relationship nodes from extraction result."""
        entity_node_ids = []
        
        # Create Entity nodes
        for i, entity in enumerate(result.entities):
            entity_id = f"N_{chunk_id}_{i}"
            entity_node = Node(
                id=entity_id,
                type=NodeType.ENTITY,
                content=entity,
                metadata={'chunk_id': chunk_id, 'entity_index': i, 'node_type': 'entity'}
            )
            
            added = self.graph_manager.add_node(entity_node)
            if added:
                entity_node_ids.append(entity_id)
                # Link to text node
                self.graph_manager.add_edge(Edge(
                    source=text_node_id, target=entity_id, relationship_type="mentions_entity"
                ))
        
        # Create Relationship nodes
        for i, (entity1, relation, entity2) in enumerate(result.relationships):
            entity1_nodes = self.graph_manager.get_entity_mentions(entity1)
            entity2_nodes = self.graph_manager.get_entity_mentions(entity2)
            
            if entity1_nodes and entity2_nodes:
                relationship_id = f"R_{chunk_id}_{i}"
                relationship_node = Node(
                    id=relationship_id,
                    type=NodeType.RELATIONSHIP,
                    content=f"{entity1} {relation} {entity2}",
                    metadata={'chunk_id': chunk_id, 'relationship_index': i, 'node_type': 'relationship'}
                )
                
                self.graph_manager.add_node(relationship_node)
                
                # Link relationships
                self.graph_manager.add_edge(Edge(
                    source=relationship_id, target=entity1_nodes[0].id, relationship_type="involves_entity"
                ))
                self.graph_manager.add_edge(Edge(
                    source=relationship_id, target=entity2_nodes[0].id, relationship_type="involves_entity"
                ))
                self.graph_manager.add_edge(Edge(
                    source=entity1_nodes[0].id, target=entity2_nodes[0].id, 
                    relationship_type=relation, metadata={'relationship_node': relationship_id}
                ))
    
    def _create_attribute_node_from_text(self, entity: Node, attributes_text: str) -> bool:
        """Create attribute node from generated text."""
        try:
            attribute_id = f"A_{entity.id}"
            attribute_node = Node(
                id=attribute_id,
                type=NodeType.ATTRIBUTE,
                content=attributes_text,
                metadata={'entity_id': entity.id, 'entity_name': entity.content, 'node_type': 'attribute'}
            )
            
            self.graph_manager.add_node(attribute_node)
            self.graph_manager.add_edge(Edge(
                source=attribute_id, target=entity.id, relationship_type="describes_entity"
            ))
            
            return True
        except Exception as e:
            logger.error(f"Error creating attribute node: {e}")
            return False
    
    def _create_community_nodes_with_debug(self, community_id: int) -> tuple[bool, bool, str]:
        """Create high-level and overview nodes for a community with debug info."""
        try:
            # Get community nodes
            community_nodes = self.graph_manager.get_community_nodes(community_id)
            
            if not community_nodes:
                return False, False, f"No nodes in community {community_id}"
            
            debug_info = f"Community {community_id}: {len(community_nodes)} nodes"
            
            # Convert to dict format for LLM service
            community_data = []
            for node in community_nodes:
                community_data.append({
                    'content': node.content,
                    'type': node.type.value,
                    'metadata': node.metadata
                })
            
            print(f"   🔍 Processing community {community_id} with {len(community_nodes)} nodes")
            print(f"      Node types: {[node.type.value for node in community_nodes[:5]]}")
            print(f"      Sample content: {[node.content[:50] + '...' for node in community_nodes[:2]]}")
            
            # Generate high-level summary
            try:
                high_level_summary = self.llm_service.generate_community_summary(community_data)
                print(f"      High-level summary: {high_level_summary[:100]}{'...' if len(high_level_summary) > 100 else ''}")
                
                if "Error" in high_level_summary or "error" in high_level_summary:
                    print(f"      ⚠️  Generated summary contains error: {high_level_summary}")
                
            except Exception as e:
                print(f"      ❌ High-level summary generation failed: {e}")
                high_level_summary = f"Error generating community summary: {str(e)}"
            
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
            try:
                overview_title = self.llm_service.generate_community_overview(high_level_summary)
                print(f"      Overview title: {overview_title}")
                
                if "Error" in overview_title or "error" in overview_title:
                    print(f"      ⚠️  Generated overview contains error: {overview_title}")
                    
            except Exception as e:
                print(f"      ❌ Overview title generation failed: {e}")
                overview_title = f"Error generating community overview: {str(e)}"
            
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
            
            debug_info += f", H: {h_success}, O: {o_success}"
            print(f"      ✅ Community {community_id} processed: H={h_success}, O={o_success}")
            return h_success, o_success, debug_info
            
        except Exception as e:
            debug_info = f"Community {community_id} failed: {str(e)}"
            print(f"      ❌ Community {community_id} processing failed: {e}")
            return False, False, debug_info
    
    async def _parallel_extract_all_chunks(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Phase 1a: Pure parallel LLM extraction without any graph building.
        Collects all extraction results for unified processing later.
        """
        total_chunks = len(chunks)
        print(f"   ⚡ Extracting from {total_chunks} chunks in parallel (8 workers)...")
        
        try:
            # Process chunks in parallel using ThreadPoolExecutor
            semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent workers
            
            async def extract_single_chunk(chunk: DocumentChunk, chunk_index: int) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        print(f"   🔄 Worker processing chunk {chunk_index+1}/{total_chunks}...")
                        
                        # Pure LLM extraction - no graph operations
                        loop = asyncio.get_event_loop()
                        extraction_result = await loop.run_in_executor(
                            None, 
                            self.llm_service.extract_all_from_chunk, 
                            chunk.content
                        )
                        
                        if extraction_result.success:
                            print(f"   ✅ Chunk {chunk_index+1}: {len(extraction_result.entities)} entities, {len(extraction_result.relationships)} relationships")
                            return {
                                'success': True,
                                'chunk': chunk,
                                'chunk_index': chunk_index,
                                'semantic_units': extraction_result.semantic_units,
                                'entities': extraction_result.entities,
                                'relationships': extraction_result.relationships
                            }
                        else:
                            print(f"   ❌ Chunk {chunk_index+1}: LLM extraction failed - {extraction_result.error_message}")
                            return {
                                'success': False,
                                'chunk': chunk,
                                'chunk_index': chunk_index,
                                'error': extraction_result.error_message
                            }
                            
                    except Exception as e:
                        print(f"   ❌ Chunk {chunk_index+1}: Exception - {e}")
                        return {
                            'success': False,
                            'chunk': chunk,
                            'chunk_index': chunk_index,
                            'error': str(e)
                        }
            
            # Create all extraction tasks
            extraction_tasks = [
                extract_single_chunk(chunk, i) for i, chunk in enumerate(chunks)
            ]
            
            # Execute all extractions in parallel
            all_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for result in all_results:
                if isinstance(result, Exception):
                    logger.error(f"Extraction task failed: {result}")
                    processed_results.append({
                        'success': False,
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            
            successful_count = sum(1 for r in processed_results if r.get('success', False))
            print(f"   📊 Extraction Summary: {successful_count}/{total_chunks} successful")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in parallel extraction: {e}")
            raise
    
    async def _build_unified_graph_from_extractions(self, extraction_results: List[Dict[str, Any]], chunks: List[DocumentChunk] = None) -> Dict[str, Any]:
        """
        Phase 1b: Build unified graph from all extraction results.
        This fixes the fragmentation issue by processing all data together.
        """
        print(f"   🏗️  Building unified graph from {len(extraction_results)} extraction results...")
        
        try:
            # Collect all entities and relationships across chunks
            all_entities = {}  # entity_content -> list of (chunk_index, entity_data)
            all_relationships = []
            all_semantic_units = []
            text_nodes_created = 0
            
            # Process successful extractions
            for result in extraction_results:
                if not result.get('success', False):
                    continue
                
                chunk = result['chunk']
                chunk_index = result['chunk_index']
                chunk_id = str(chunk_index)
                
                # 1. Create Text Node (T) for each chunk
                text_node = Node(
                    id=f"T_{chunk_id}",
                    type=NodeType.TEXT,
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        'node_type': 'text',
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_index
                    }
                )
                self.graph_manager.add_node(text_node)
                text_nodes_created += 1
                
                # 2. Collect Semantic Units
                for i, semantic_unit in enumerate(result.get('semantic_units', [])):
                    all_semantic_units.append({
                        'content': semantic_unit,
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_index,
                        'semantic_index': i,
                        'text_node_id': text_node.id
                    })
                
                # 3. Collect Entities (with deduplication tracking)
                for entity_content in result.get('entities', []):
                    entity_key = entity_content.lower().strip()
                    if entity_key not in all_entities:
                        all_entities[entity_key] = []
                    
                    all_entities[entity_key].append({
                        'original_content': entity_content,
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_index,
                        'text_node_id': text_node.id
                    })
                
                # 4. Collect Relationships
                for relationship in result.get('relationships', []):
                    if isinstance(relationship, tuple) and len(relationship) == 3:
                        entity1, relation, entity2 = relationship
                        all_relationships.append({
                            'entity1': entity1,
                            'relation': relation, 
                            'entity2': entity2,
                            'chunk_id': chunk_id,
                            'chunk_index': chunk_index,
                            'text_node_id': text_node.id
                        })
            
            print(f"   📊 Collected: {len(all_semantic_units)} semantic units, {len(all_entities)} unique entities, {len(all_relationships)} relationships")
            
            # 5. Create Enhanced Semantic Unit Nodes (with rich context preservation)
            semantic_nodes_created = 0
            semantic_content_index = {}  # Track semantic content for cross-referencing
            
            print(f"   📝 Creating semantic units from {len(all_semantic_units)} extracted units...")
            
            for semantic_data in all_semantic_units:
                semantic_id = f"S_{semantic_data['chunk_id']}_{semantic_data['semantic_index']}"
                semantic_content = semantic_data['content']
                
                # Enhanced semantic node with richer metadata
                semantic_node = Node(
                    id=semantic_id,
                    type=NodeType.SEMANTIC,
                    content=semantic_content,
                    metadata={
                        'chunk_id': semantic_data['chunk_id'],
                        'chunk_index': semantic_data['chunk_index'],
                        'semantic_index': semantic_data['semantic_index'],
                        'content_length': len(semantic_content),
                        'contains_temporal': self._detect_temporal_content(semantic_content),
                        'contains_quantitative': self._detect_quantitative_content(semantic_content),
                        'contains_important': self._detect_important_content(semantic_content),
                        'node_type': 'semantic'
                    }
                )
                
                success = self.graph_manager.add_node(semantic_node)
                if success:
                    semantic_nodes_created += 1
                    semantic_content_index[semantic_content.lower()] = semantic_id
                    
                    # Link to text node
                    self.graph_manager.add_edge(Edge(
                        source=semantic_data['text_node_id'],
                        target=semantic_id,
                        relationship_type="contains_semantic_unit"
                    ))
                    
                    # Log important semantic units for debugging
                    if (semantic_node.metadata['contains_temporal'] or 
                        semantic_node.metadata['contains_quantitative'] or 
                        semantic_node.metadata['contains_important']):
                        print(f"   🕒 Important semantic unit: {semantic_content[:100]}...")
            
            print(f"   ✅ Created {semantic_nodes_created} semantic nodes with enhanced metadata")
            
            # 6. Create Consolidated Entity Nodes (avoiding duplicates)
            entity_mapping = {}  # original_entity_content -> consolidated_entity_id
            consolidated_entities = 0
            entity_nodes_created = 0
            
            print(f"   🔨 Creating consolidated entities from {len(all_entities)} unique entity groups...")
            
            for entity_key, entity_instances in all_entities.items():
                # Use the most complete version of the entity name
                best_entity_name = max(entity_instances, key=lambda x: len(x['original_content']))['original_content']
                
                # Create single entity node for all instances
                entity_id = f"N_{entity_key.replace(' ', '_').replace('-', '_').replace('.', '_')}"
                entity_node = Node(
                    id=entity_id,
                    type=NodeType.ENTITY,
                    content=best_entity_name,
                    metadata={
                        'mentions': [inst['chunk_id'] for inst in entity_instances],
                        'mention_count': len(entity_instances),
                        'variants': list(set(inst['original_content'] for inst in entity_instances)),
                        'chunks_mentioned': list(set(inst['chunk_index'] for inst in entity_instances)),
                        'node_type': 'entity'
                    }
                )
                
                success = self.graph_manager.add_node(entity_node)
                if success:
                    entity_nodes_created += 1
                    
                    # Map all variations to this consolidated entity
                    for instance in entity_instances:
                        entity_mapping[instance['original_content']] = entity_id
                        
                        # Link to text nodes where it appears
                        edge_success = self.graph_manager.add_edge(Edge(
                            source=instance['text_node_id'],
                            target=entity_id,
                            relationship_type="mentions_entity"
                        ))
                        if not edge_success:
                            logger.warning(f"Failed to create edge from {instance['text_node_id']} to {entity_id}")
                    
                    if len(entity_instances) > 1:
                        consolidated_entities += 1
                        print(f"   ✨ Consolidated entity: '{best_entity_name}' ({len(entity_instances)} mentions)")
                else:
                    logger.warning(f"Failed to create entity node: {entity_id}")
            
            print(f"   ✅ Created {entity_nodes_created} entity nodes, consolidated {consolidated_entities} duplicates")
            
            # 7. Create Enhanced Relationship Nodes (preserving temporal and contextual data)
            relationships_created = 0
            cross_chunk_relationships = 0
            temporal_relationships = 0
            
            print(f"   🔗 Creating relationships from {len(all_relationships)} extracted relationships...")
            
            for rel_data in all_relationships:
                entity1_id = entity_mapping.get(rel_data['entity1'])
                entity2_id = entity_mapping.get(rel_data['entity2'])
                
                if entity1_id and entity2_id and entity1_id != entity2_id:
                    relationship_id = f"R_{rel_data['chunk_id']}_{relationships_created}"
                    relationship_content = f"{rel_data['entity1']} {rel_data['relation']} {rel_data['entity2']}"
                    
                    # Enhanced relationship detection
                    is_temporal = any(term in rel_data['relation'].lower() for term in 
                                    ['born', 'founded', 'established', 'created', 'years', 'ago', 'since'])
                    is_quantitative = any(char.isdigit() for char in str(rel_data['relation']))
                    
                    relationship_node = Node(
                        id=relationship_id,
                        type=NodeType.RELATIONSHIP,
                        content=relationship_content,
                        metadata={
                            'entity1': rel_data['entity1'],
                            'relation': rel_data['relation'],
                            'entity2': rel_data['entity2'],
                            'chunk_id': rel_data['chunk_id'],
                            'chunk_index': rel_data['chunk_index'],
                            'consolidated_entity1': entity1_id,
                            'consolidated_entity2': entity2_id,
                            'is_temporal': is_temporal,
                            'is_quantitative': is_quantitative,
                            'relation_length': len(rel_data['relation']),
                            'node_type': 'relationship'
                        }
                    )
                    
                    success = self.graph_manager.add_node(relationship_node)
                    if success:
                        relationships_created += 1
                        
                        if is_temporal:
                            temporal_relationships += 1
                            print(f"   🕒 Temporal relationship: {relationship_content}")
                        
                        # Create relationship edges
                        self.graph_manager.add_edge(Edge(
                            source=relationship_id,
                            target=entity1_id,
                            relationship_type="involves_entity"
                        ))
                        self.graph_manager.add_edge(Edge(
                            source=relationship_id,
                            target=entity2_id,
                            relationship_type="involves_entity"
                        ))
                        
                        # Enhanced edge with temporal/quantitative metadata
                        self.graph_manager.add_edge(Edge(
                            source=entity1_id,
                            target=entity2_id,
                            relationship_type=rel_data['relation'],
                            metadata={
                                'relationship_node': relationship_id,
                                'is_temporal': is_temporal,
                                'is_quantitative': is_quantitative
                            }
                        ))
                        
                        # Cross-chunk relationship detection
                        entity1_mentions = all_entities.get(rel_data['entity1'].lower().strip(), [])
                        entity2_mentions = all_entities.get(rel_data['entity2'].lower().strip(), [])
                        
                        if (len(entity1_mentions) > 1 or len(entity2_mentions) > 1):
                            cross_chunk_relationships += 1
                            print(f"   🌉 Cross-chunk relationship: {relationship_content}")
                else:
                    if not entity1_id:
                        logger.warning(f"Missing entity1 mapping: {rel_data['entity1']}")
                    if not entity2_id:
                        logger.warning(f"Missing entity2 mapping: {rel_data['entity2']}")
                        
            # 8. Create Temporal Context Nodes (for timeline information)
            temporal_context_nodes = self._create_temporal_context_nodes(
                all_semantic_units, entity_mapping, semantic_content_index
            )
            
            print(f"   ✅ Created {relationships_created} relationship nodes ({cross_chunk_relationships} cross-chunk, {temporal_relationships} temporal)")
            print(f"   ⏰ Created {temporal_context_nodes} temporal context nodes")
            
            # Calculate accurate node and edge counts
            total_nodes = text_nodes_created + semantic_nodes_created + entity_nodes_created + relationships_created
            total_edges = semantic_nodes_created + len(entity_mapping) + (relationships_created * 3)  # More accurate edge count
            
            return {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'text_nodes': text_nodes_created,
                'semantic_nodes': semantic_nodes_created,
                'entity_nodes': entity_nodes_created,
                'relationship_nodes': relationships_created,
                'consolidated_entities': consolidated_entities,
                'cross_chunk_relationships': cross_chunk_relationships,
                'unique_entities': len(all_entities),
                'total_relationships': len(all_relationships),
                'entity_mapping_size': len(entity_mapping)
            }
            
        except Exception as e:
            logger.error(f"Error building unified graph: {e}")
            raise
    
    def _create_temporal_context_nodes(self, all_semantic_units: List[Dict], entity_mapping: Dict, semantic_content_index: Dict) -> int:
        """
        Create specialized temporal context nodes to preserve timeline information that might be lost.
        This addresses the data loss issue in parallel processing.
        """
        temporal_nodes_created = 0
        
        try:
            for semantic_data in all_semantic_units:
                semantic_content = semantic_data['content'].lower()
                
                # Comprehensive temporal pattern detection
                temporal_patterns = [
                    # Birth/founding patterns
                    r'(\w+.*?)\s+was\s+born\s+(\d+)\s+years?\s+ago',
                    r'(\w+.*?)\s+(founded|established|created|launched|started)\s+(\d+)\s+years?\s+ago',
                    r'(\d+)\s+years?\s+ago.*?(\w+.*?)\s+(was\s+)?(founded|born|established|created)',
                    
                    # Experience/tenure patterns  
                    r'for\s+over\s+(\d+)\s+years?.*?(leadership|team|experience|focus|placed)',
                    r'(leadership|team|management).*?has\s+(placed|focused|dedicated|spent).*?(\d+)\s+years?',
                    r'(\d+)\s+years?.*?(experience|focus|leadership|expertise|dedication)',
                    
                    # General temporal patterns
                    r'(\w+.*?)\s+(\d+)\s+years?\s+(old|ago|experience|history)',
                    r'(since|for|over|during|throughout)\s+(\d+)\s+years?.*?(\w+)',
                    r'(\w+.*?)\s+(began|started|commenced|initiated)\s+in\s+(\d{4})',
                    r'in\s+(\d{4}).*?(\w+.*?)\s+(was\s+)?(founded|established|created)',
                    
                    # Timeline and duration patterns
                    r'(timeline|history|background).*?(\d+)\s+(years?|decades?)',
                    r'(\d+)\s+(years?|decades?).*?(timeline|history|background|experience)',
                    r'(over|more than|nearly|approximately)\s+(\d+)\s+years?',
                    
                    # Age and maturity patterns
                    r'(\w+.*?)\s+is\s+(\d+)\s+years?\s+(old|mature)',
                    r'(\d+)[-\s]year[-\s](old|history|experience)',
                    r'(\w+.*?)\s+has\s+been\s+.*?for\s+(\d+)\s+years?'
                ]
                
                import re
                for pattern in temporal_patterns:
                    matches = re.findall(pattern, semantic_content, re.IGNORECASE)
                    
                    if matches:
                        for match in matches:
                            # Create temporal context node to preserve this information
                            temporal_id = f"TEMPORAL_{semantic_data['chunk_id']}_{temporal_nodes_created}"
                            
                            if isinstance(match, tuple) and len(match) >= 2:
                                if match[0].isdigit():  # Number first (e.g., "4 years ago")
                                    temporal_content = f"Timeline: {match[1]} occurred {match[0]} years ago"
                                elif match[1].isdigit():  # Number second (e.g., "Andor born 4 years ago")
                                    temporal_content = f"Timeline: {match[0]} {match[1]} years ago"
                                else:
                                    temporal_content = f"Timeline: {' '.join(match)}"
                            else:
                                temporal_content = f"Timeline: {semantic_data['content']}"
                            
                            temporal_node = Node(
                                id=temporal_id,
                                type=NodeType.ATTRIBUTE,  # Use attribute type for temporal context
                                content=temporal_content,
                                metadata={
                                    'chunk_id': semantic_data['chunk_id'],
                                    'chunk_index': semantic_data['chunk_index'],
                                    'original_semantic': semantic_data['content'],
                                    'temporal_pattern': pattern,
                                    'matched_data': str(match),
                                    'node_type': 'temporal_context'
                                }
                            )
                            
                            success = self.graph_manager.add_node(temporal_node)
                            if success:
                                temporal_nodes_created += 1
                                print(f"   ⏰ Preserved temporal context: {temporal_content}")
                                
                                # Link to relevant entities if they exist
                                for entity_key, entity_id in entity_mapping.items():
                                    if any(word in entity_key for word in ['andor', 'health', 'leadership']):
                                        self.graph_manager.add_edge(Edge(
                                            source=temporal_id,
                                            target=entity_id,
                                            relationship_type="provides_temporal_context"
                                        ))
                                        
                                # Link to text node
                                text_node_id = semantic_data.get('text_node_id')
                                if text_node_id:
                                    self.graph_manager.add_edge(Edge(
                                        source=text_node_id,
                                        target=temporal_id,
                                        relationship_type="contains_temporal_context"
                                    ))
            
            return temporal_nodes_created
            
        except Exception as e:
            logger.error(f"Error creating temporal context nodes: {e}")
            return temporal_nodes_created
    
    def _detect_temporal_content(self, content: str) -> bool:
        """
        Comprehensive temporal content detection - not just hardcoded terms.
        """
        content_lower = content.lower()
        
        # Basic temporal indicators
        temporal_terms = [
            'years', 'year', 'ago', 'founded', 'born', 'established', 'since', 'for',
            'during', 'when', 'before', 'after', 'until', 'while', 'throughout',
            'months', 'month', 'weeks', 'week', 'days', 'day', 'hours', 'hour',
            'decades', 'decade', 'centuries', 'century', 'recently', 'previously',
            'formerly', 'initially', 'originally', 'started', 'began', 'commenced',
            'launched', 'created', 'initiated', 'introduced', 'developed', 'first',
            'last', 'latest', 'current', 'ongoing', 'continuous', 'historical'
        ]
        
        # Date patterns
        import re
        date_patterns = [
            r'\b\d{4}\b',  # Years like 2023, 1975
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates like 12/25/2023
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\bover\s+\d+\s+(years?|months?|weeks?|days?)\b',
            r'\b\d+\s+(years?|months?|weeks?|days?)\s+(ago|later|old)\b'
        ]
        
        # Check for basic temporal terms
        if any(term in content_lower for term in temporal_terms):
            return True
        
        # Check for date patterns
        for pattern in date_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check for timeline expressions
        timeline_expressions = [
            r'\b(when|where|how long|duration|period|timeline|history|background)\b',
            r'\b(experience|tenure|service|employment|leadership|management)\b',
            r'\b(placement|focus|emphasis|concentration|dedication)\b'
        ]
        
        for pattern in timeline_expressions:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def _detect_quantitative_content(self, content: str) -> bool:
        """
        Comprehensive quantitative content detection.
        """
        import re
        content_lower = content.lower()
        
        # Basic digit check
        if any(char.isdigit() for char in content):
            return True
        
        # Quantitative terms
        quantitative_terms = [
            'percent', 'percentage', '%', 'ratio', 'proportion', 'rate', 'amount',
            'number', 'count', 'total', 'sum', 'average', 'mean', 'median',
            'maximum', 'minimum', 'range', 'scale', 'measure', 'measurement',
            'statistics', 'data', 'metrics', 'analytics', 'figures',
            'approximately', 'roughly', 'about', 'around', 'nearly', 'close to',
            'more than', 'less than', 'greater than', 'fewer than', 'over', 'under'
        ]
        
        # Number words
        number_words = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
            'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion',
            'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
            'many', 'several', 'few', 'multiple', 'various', 'numerous'
        ]
        
        # Check for quantitative terms
        if any(term in content_lower for term in quantitative_terms + number_words):
            return True
        
        # Check for measurement patterns
        measurement_patterns = [
            r'\b\d+([.,]\d+)?\s*(percent|%|dollars?|\$|euros?|pounds?|kg|lbs?|miles?|km|feet|ft|inches?|in|hours?|hrs?|minutes?|mins?|seconds?|secs?)\b',
            r'\b(increase|decrease|growth|decline|reduction|improvement|boost|drop|rise|fall)\b',
            r'\b(compared to|versus|vs\.?|against|relative to)\b'
        ]
        
        for pattern in measurement_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def _detect_important_content(self, content: str) -> bool:
        """
        Detect other important content that should be preserved.
        """
        content_lower = content.lower()
        
        # Important business/organizational concepts
        important_terms = [
            # Leadership and roles
            'ceo', 'cfo', 'cto', 'president', 'director', 'manager', 'executive', 'founder',
            'leadership', 'team', 'staff', 'employee', 'personnel', 'workforce',
            
            # Business concepts
            'mission', 'vision', 'goal', 'objective', 'strategy', 'approach', 'methodology',
            'platform', 'solution', 'service', 'product', 'offering', 'capability',
            'innovation', 'technology', 'development', 'implementation', 'deployment',
            'collaboration', 'partnership', 'alliance', 'relationship', 'network',
            
            # Outcomes and impact
            'outcome', 'result', 'achievement', 'success', 'impact', 'effect', 'benefit',
            'improvement', 'enhancement', 'optimization', 'efficiency', 'productivity',
            'quality', 'satisfaction', 'experience', 'performance', 'effectiveness',
            
            # Healthcare specific
            'patient', 'clinician', 'physician', 'doctor', 'nurse', 'healthcare', 'medical',
            'treatment', 'care', 'therapy', 'diagnosis', 'monitoring', 'consultation',
            'virtual', 'telehealth', 'telemedicine', 'remote', 'digital', 'ai', 'artificial intelligence',
            
            # Problem-solving
            'challenge', 'problem', 'issue', 'shortage', 'gap', 'need', 'requirement',
            'solution', 'address', 'solve', 'resolve', 'tackle', 'handle', 'manage'
        ]
        
        # Check for important terms
        if any(term in content_lower for term in important_terms):
            return True
        
        # Check for sentences with key action verbs
        import re
        action_patterns = [
            r'\b(enables?|allows?|provides?|offers?|delivers?|supports?|facilitates?)\b',
            r'\b(reduces?|increases?|improves?|enhances?|optimizes?|streamlines?)\b',
            r'\b(integrates?|combines?|unifies?|consolidates?|merges?|connects?)\b',
            r'\b(designed|built|created|developed|engineered|constructed)\b',
            r'\b(helps?|assists?|aids?|supports?|enables?|empowers?)\b'
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, content_lower):
                return True
        
        # Check for causal/explanatory language
        explanatory_patterns = [
            r'\b(because|since|due to|owing to|as a result|therefore|thus|hence|consequently)\b',
            r'\b(in order to|so that|to ensure|to enable|to provide|to support)\b',
            r'\b(by|through|via|using|utilizing|leveraging|employing)\b'
        ]
        
        for pattern in explanatory_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    async def _consolidate_entities_and_relationships(self) -> Dict[str, Any]:
        """
        Post-processing phase to consolidate entities and detect cross-chunk relationships.
        Maintains parallel optimizations while improving graph quality.
        """
        try:
            # 1. Entity Consolidation using semantic similarity
            entity_nodes = self.graph_manager.get_nodes_by_type(NodeType.ENTITY)
            consolidated_count = 0
            cross_chunk_relationships = 0
            
            print(f"   🔍 Analyzing {len(entity_nodes)} entities for consolidation...")
            
            # Group similar entities using embedding similarity
            if len(entity_nodes) > 1:
                entity_contents = [node.content for node in entity_nodes]
                
                # Generate embeddings in batch for efficiency
                embeddings = self.llm_service.get_embeddings(entity_contents, batch_size=32)
                
                # Find similar entities using cosine similarity
                if len(embeddings) == len(entity_nodes):
                    embeddings_matrix = np.array(embeddings)
                    similarity_matrix = cosine_similarity(embeddings_matrix)
                else:
                    logger.warning(f"Embedding mismatch: {len(embeddings)} embeddings for {len(entity_nodes)} entities")
                    return {'merged_entities': 0, 'cross_chunk_relationships': 0, 'error': 'embedding_mismatch'}
                
                # Consolidate entities with > 0.85 similarity
                processed_entities = set()
                for i, entity1 in enumerate(entity_nodes):
                    if entity1.id in processed_entities:
                        continue
                        
                    similar_entities = []
                    for j, entity2 in enumerate(entity_nodes):
                        if i != j and entity2.id not in processed_entities:
                            similarity = similarity_matrix[i][j]
                            if similarity > 0.85:  # High similarity threshold
                                similar_entities.append(entity2)
                    
                    if similar_entities:
                        # Merge entities
                        primary_entity = entity1
                        for similar_entity in similar_entities:
                            self.graph_manager.merge_entities(primary_entity.id, similar_entity.id)
                            processed_entities.add(similar_entity.id)
                            consolidated_count += 1
                        
                        processed_entities.add(primary_entity.id)
            
            # 2. Cross-chunk relationship detection
            print(f"   🔗 Detecting cross-chunk relationships...")
            
            # Get all entities that appear in multiple chunks
            multi_chunk_entities = []
            for entity in entity_nodes:
                if entity.id not in [e.id for e in entity_nodes if e.id in processed_entities]:
                    connected_chunks = set()
                    for neighbor_id in self.graph_manager.graph.neighbors(entity.id):
                        neighbor = self.graph_manager.get_node(neighbor_id)
                        if neighbor and neighbor.type == NodeType.TEXT:
                            chunk_id = neighbor.metadata.get('chunk_id')
                            if chunk_id:
                                connected_chunks.add(chunk_id)
                    
                    if len(connected_chunks) > 1:
                        multi_chunk_entities.append((entity, connected_chunks))
            
            # Find potential relationships between multi-chunk entities
            for i, (entity1, chunks1) in enumerate(multi_chunk_entities):
                for j, (entity2, chunks2) in enumerate(multi_chunk_entities[i+1:], i+1):
                    # Check if entities appear in overlapping or adjacent chunks
                    chunk_overlap = chunks1.intersection(chunks2)
                    if chunk_overlap:
                        # Use LLM to determine if there's a meaningful relationship
                        relationship = await self._detect_entity_relationship(entity1, entity2)
                        if relationship:
                            # Create cross-chunk relationship
                            self._create_cross_chunk_relationship(entity1, entity2, relationship, chunk_overlap)
                            cross_chunk_relationships += 1
            
            return {
                'merged_entities': consolidated_count,
                'cross_chunk_relationships': cross_chunk_relationships,
                'multi_chunk_entities': len(multi_chunk_entities)
            }
            
        except Exception as e:
            logger.error(f"Error in entity consolidation: {e}")
            return {'merged_entities': 0, 'cross_chunk_relationships': 0, 'error': str(e)}
    
    async def _detect_entity_relationship(self, entity1: Node, entity2: Node) -> Optional[str]:
        """Use LLM to detect potential relationship between two entities."""
        try:
            # Get context from both entities
            context1 = []
            context2 = []
            
            for neighbor_id in self.graph_manager.graph.neighbors(entity1.id):
                neighbor = self.graph_manager.get_node(neighbor_id)
                if neighbor and neighbor.type == NodeType.TEXT:
                    context1.append(neighbor.content[:500])  # Limit context
            
            for neighbor_id in self.graph_manager.graph.neighbors(entity2.id):
                neighbor = self.graph_manager.get_node(neighbor_id)
                if neighbor and neighbor.type == NodeType.TEXT:
                    context2.append(neighbor.content[:500])
            
            # Use LLM to detect relationship
            prompt = f"""
Analyze if there is a meaningful relationship between these two entities based on their context.

Entity 1: {entity1.content}
Context 1: {' '.join(context1[:2])}

Entity 2: {entity2.content}
Context 2: {' '.join(context2[:2])}

Return only the relationship type if one exists (e.g., "works for", "is located in", "partners with"), 
or "NONE" if no clear relationship exists.

Relationship:"""

            relationship = self.llm_service._chat_completion(prompt, temperature=0.1)
            relationship = relationship.strip()
            
            if relationship and relationship != "NONE" and len(relationship) < 100:
                return relationship
            return None
            
        except Exception as e:
            logger.error(f"Error detecting entity relationship: {e}")
            return None
    
    def _create_cross_chunk_relationship(self, entity1: Node, entity2: Node, relationship: str, chunk_overlap: set):
        """Create a cross-chunk relationship node."""
        try:
            relationship_id = f"R_cross_{entity1.id}_{entity2.id}"
            relationship_node = Node(
                id=relationship_id,
                type=NodeType.RELATIONSHIP,
                content=f"{entity1.content} {relationship} {entity2.content}",
                metadata={
                    'entity1': entity1.content,
                    'relation': relationship,
                    'entity2': entity2.content,
                    'cross_chunk': True,
                    'chunks': list(chunk_overlap),
                    'node_type': 'relationship'
                }
            )
            
            self.graph_manager.add_node(relationship_node)
            
            # Create edges
            self.graph_manager.add_edge(Edge(
                source=relationship_id, target=entity1.id, relationship_type="involves_entity"
            ))
            self.graph_manager.add_edge(Edge(
                source=relationship_id, target=entity2.id, relationship_type="involves_entity"
            ))
            self.graph_manager.add_edge(Edge(
                source=entity1.id, target=entity2.id, relationship_type=relationship,
                metadata={'relationship_node': relationship_id, 'cross_chunk': True}
            ))
            
        except Exception as e:
            logger.error(f"Error creating cross-chunk relationship: {e}")

    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get statistics about the current indexed content."""
        return self.graph_manager.get_stats()
    
    def _get_checkpoint_path(self) -> str:
        """Get the path for embedding checkpoint file."""
        import os
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        return os.path.join(Config.DATA_DIR, "embedding_checkpoint.json")
    
    def _load_embedding_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load embedding generation checkpoint if it exists."""
        if not Config.ENABLE_CHECKPOINTS:
            return None
        
        checkpoint_path = self._get_checkpoint_path()
        
        try:
            import json
            import os
            
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    print(f"   📁 Found checkpoint: {checkpoint_data}")
                    return checkpoint_data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        
        return None
    
    def _save_embedding_checkpoint(self, batch_num: int, embeddings_generated: int, completed: bool = False) -> None:
        """Save embedding generation checkpoint."""
        if not Config.ENABLE_CHECKPOINTS:
            return
        
        checkpoint_path = self._get_checkpoint_path()
        
        try:
            import json
            import time
            
            checkpoint_data = {
                'last_completed_batch': batch_num,
                'embeddings_generated': embeddings_generated,
                'timestamp': time.time(),
                'environment': 'render' if Config.IS_RENDER else 'cloud' if Config.IS_CLOUD else 'local',
                'completed': completed
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _cleanup_embedding_checkpoint(self) -> None:
        """Clean up checkpoint file after successful completion."""
        if not Config.ENABLE_CHECKPOINTS:
            return
        
        checkpoint_path = self._get_checkpoint_path()
        
        try:
            import os
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print("   🧹 Checkpoint file cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoint: {e}")
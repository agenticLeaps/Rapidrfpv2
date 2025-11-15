import os
import time
import logging
from typing import List, Dict, Any, Optional
from glob import glob
import asyncio

from .state_manager import StateManager
from ..document_processing.indexing_pipeline import IndexingPipeline
from ..vector.hnsw_service import HNSWService
from ..search.advanced_search import AdvancedSearchSystem
from ..config.settings import Config

logger = logging.getLogger(__name__)

class IncrementalIndexingPipeline:
    """
    Incremental indexing pipeline supporting resume and incremental updates.
    Implements NodeRAG's incremental processing approach.
    """
    
    def __init__(self):
        """Initialize incremental indexing pipeline."""
        self.state_manager = StateManager()
        self.indexing_pipeline = IndexingPipeline()
        self.hnsw_service = None
        self.search_system = None
        
        # Pipeline phases
        self.phases = [
            "DOCUMENT_DISCOVERY",
            "DOCUMENT_PROCESSING", 
            "GRAPH_AUGMENTATION",
            "EMBEDDING_GENERATION",
            "HNSW_INDEXING",
            "SEARCH_SYSTEM_UPDATE",
            "FINISHED"
        ]
        
        logger.info("Incremental indexing pipeline initialized")
    
    def discover_documents(self, 
                          root_dirs: List[str],
                          file_patterns: List[str] = None) -> List[str]:
        """
        Discover documents in specified directories.
        
        Args:
            root_dirs: List of root directories to search
            file_patterns: File patterns to match (e.g., ["*.pdf", "*.docx"])
            
        Returns:
            List of discovered document file paths
        """
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.docx", "*.txt", "*.md"]
        
        discovered_files = []
        
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                logger.warning(f"Directory not found: {root_dir}")
                continue
            
            for pattern in file_patterns:
                search_pattern = os.path.join(root_dir, "**", pattern)
                files = glob(search_pattern, recursive=True)
                discovered_files.extend(files)
        
        # Remove duplicates and sort
        discovered_files = sorted(list(set(discovered_files)))
        
        logger.info(f"Discovered {len(discovered_files)} documents")
        return discovered_files
    
    def check_incremental_mode(self, all_documents: List[str]) -> bool:
        """
        Check if we can run in incremental mode.
        
        Args:
            all_documents: List of all discovered documents
            
        Returns:
            True if incremental processing is possible
        """
        # Clean up state for deleted documents
        current_docs_set = set(all_documents)
        self.state_manager.cleanup_deleted_documents(current_docs_set)
        
        # Check which documents need processing
        docs_to_process = self.state_manager.get_documents_to_process(all_documents)
        
        # Incremental mode if we have existing state and only some docs need processing
        can_resume = self.state_manager.can_resume_processing()
        has_changes = len(docs_to_process) > 0
        has_existing_work = len(self.state_manager.document_states) > 0
        
        is_incremental = has_existing_work and (not has_changes or len(docs_to_process) < len(all_documents))
        
        self.state_manager.set_incremental_mode(is_incremental)
        
        logger.info(f"Incremental mode: {is_incremental} "
                   f"(can_resume={can_resume}, docs_to_process={len(docs_to_process)}/{len(all_documents)})")
        
        return is_incremental
    
    async def process_documents_incremental(self, documents_to_process: List[str]) -> Dict[str, Any]:
        """
        Process documents incrementally with proper state management.
        
        Args:
            documents_to_process: List of documents that need processing
            
        Returns:
            Processing results summary
        """
        if not documents_to_process:
            logger.info("No documents to process")
            return {'processed': 0, 'failed': 0, 'skipped': len(documents_to_process)}
        
        self.state_manager.update_pipeline_phase("DOCUMENT_PROCESSING")
        
        processed = 0
        failed = 0
        
        for i, file_path in enumerate(documents_to_process):
            logger.info(f"Processing document {i+1}/{len(documents_to_process)}: {file_path}")
            
            # Start processing
            self.state_manager.start_document_processing(file_path)
            start_time = time.time()
            
            try:
                # Process document
                result = self.indexing_pipeline.index_document(file_path)
                processing_time = time.time() - start_time
                
                if result['success']:
                    # Extract node IDs from result (implementation would need to track this)
                    node_ids_created = []  # Would need to implement node ID tracking
                    chunks_processed = result.get('chunks_processed', 0)
                    
                    self.state_manager.complete_document_processing(
                        file_path=file_path,
                        chunks_processed=chunks_processed,
                        node_ids_created=node_ids_created,
                        processing_time=processing_time,
                        success=True
                    )
                    processed += 1
                    
                else:
                    self.state_manager.complete_document_processing(
                        file_path=file_path,
                        chunks_processed=0,
                        node_ids_created=[],
                        processing_time=processing_time,
                        success=False,
                        error_message=result.get('error', 'Unknown error')
                    )
                    failed += 1
                    
            except Exception as e:
                processing_time = time.time() - start_time
                error_msg = str(e)
                
                self.state_manager.complete_document_processing(
                    file_path=file_path,
                    chunks_processed=0,
                    node_ids_created=[],
                    processing_time=processing_time,
                    success=False,
                    error_message=error_msg
                )
                failed += 1
                logger.error(f"Error processing {file_path}: {error_msg}")
        
        result_summary = {
            'processed': processed,
            'failed': failed,
            'skipped': 0,
            'total_time': sum(
                s.processing_time for s in self.state_manager.document_states.values()
                if s.success
            )
        }
        
        logger.info(f"Document processing completed: {processed} processed, {failed} failed")
        return result_summary
    
    async def generate_embeddings_incremental(self) -> Dict[str, Any]:
        """Generate embeddings for new nodes incrementally."""
        self.state_manager.update_pipeline_phase("EMBEDDING_GENERATION")
        
        try:
            # Get all nodes that need embeddings
            # This would need to be implemented to track which nodes are new
            all_nodes = []
            for node_type in ['semantic_unit', 'entity', 'attribute', 'high_level']:
                nodes = self.indexing_pipeline.graph_manager.get_nodes_by_type(
                    getattr(self.indexing_pipeline.graph_manager.NodeType, node_type.upper())
                )
                for node in nodes:
                    if not hasattr(node, 'embeddings') or node.embeddings is None:
                        all_nodes.append(node)
            
            if not all_nodes:
                logger.info("No nodes need embeddings")
                return {'embeddings_generated': 0}
            
            # Generate embeddings in batches
            batch_size = Config.DEFAULT_BATCH_SIZE
            embeddings_generated = 0
            
            for i in range(0, len(all_nodes), batch_size):
                batch = all_nodes[i:i + batch_size]
                texts = [node.content for node in batch]
                
                # Generate embeddings
                embeddings = self.indexing_pipeline.llm_service.get_embeddings(texts)
                
                # Update nodes with embeddings
                for node, embedding in zip(batch, embeddings):
                    node.embeddings = embedding
                    embeddings_generated += 1
                
                logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            
            logger.info(f"Generated {embeddings_generated} embeddings")
            return {'embeddings_generated': embeddings_generated}
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {'embeddings_generated': 0, 'error': str(e)}
    
    async def update_hnsw_index_incremental(self) -> Dict[str, Any]:
        """Update HNSW index with new embeddings incrementally."""
        self.state_manager.update_pipeline_phase("HNSW_INDEXING")
        
        try:
            # Initialize HNSW service if not exists
            if self.hnsw_service is None:
                self.hnsw_service = HNSWService(
                    dimension=Config.HNSW_DIMENSION,
                    max_elements=Config.HNSW_MAX_ELEMENTS,
                    ef_construction=Config.HNSW_EF_CONSTRUCTION,
                    m=Config.HNSW_M,
                    space=Config.HNSW_SPACE
                )
                
                # Try to load existing index
                if os.path.exists(Config.HNSW_INDEX_PATH + ".hnsw"):
                    self.hnsw_service.load_index(Config.HNSW_INDEX_PATH)
            
            # Get nodes with embeddings that aren't in HNSW yet
            nodes_to_add = []
            
            for node_type_name in ['SEMANTIC', 'ENTITY', 'ATTRIBUTE', 'HIGH_LEVEL']:
                node_type = getattr(self.indexing_pipeline.graph_manager.NodeType, node_type_name)
                nodes = self.indexing_pipeline.graph_manager.get_nodes_by_type(node_type)
                
                for node in nodes:
                    if (hasattr(node, 'embeddings') and 
                        node.embeddings is not None and
                        not self.hnsw_service.get_node_by_id(node.id)):
                        
                        metadata = {
                            'node_type': node.type.value,
                            'content': node.content
                        }
                        nodes_to_add.append((node.id, node.embeddings, metadata))
            
            if not nodes_to_add:
                logger.info("No new nodes to add to HNSW index")
                return {'nodes_added': 0}
            
            # Add nodes to HNSW index
            added_count = self.hnsw_service.add_batch_embeddings(nodes_to_add)
            
            # Save updated index
            self.hnsw_service.save_index(Config.HNSW_INDEX_PATH)
            
            logger.info(f"Added {added_count} nodes to HNSW index")
            return {'nodes_added': added_count}
            
        except Exception as e:
            logger.error(f"Error updating HNSW index: {e}")
            return {'nodes_added': 0, 'error': str(e)}
    
    async def update_search_system(self) -> Dict[str, Any]:
        """Update search system with new graph and index."""
        self.state_manager.update_pipeline_phase("SEARCH_SYSTEM_UPDATE")
        
        try:
            if self.search_system is None and self.hnsw_service is not None:
                self.search_system = AdvancedSearchSystem(
                    graph_manager=self.indexing_pipeline.graph_manager,
                    hnsw_service=self.hnsw_service,
                    llm_service=self.indexing_pipeline.llm_service
                )
                logger.info("Search system initialized")
            elif self.search_system is not None:
                self.search_system.update_graph(self.indexing_pipeline.graph_manager)
                logger.info("Search system updated")
            
            return {'search_system_updated': True}
            
        except Exception as e:
            logger.error(f"Error updating search system: {e}")
            return {'search_system_updated': False, 'error': str(e)}
    
    async def run_incremental_pipeline(self, 
                                     root_dirs: List[str],
                                     file_patterns: List[str] = None,
                                     force_full_rebuild: bool = False) -> Dict[str, Any]:
        """
        Run the complete incremental indexing pipeline.
        
        Args:
            root_dirs: Directories to search for documents
            file_patterns: File patterns to match
            force_full_rebuild: Force full rebuild instead of incremental
            
        Returns:
            Pipeline execution results
        """
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: Document Discovery
            self.state_manager.update_pipeline_phase("DOCUMENT_DISCOVERY")
            all_documents = self.discover_documents(root_dirs, file_patterns)
            
            if not all_documents:
                logger.warning("No documents found")
                return {'status': 'no_documents', 'message': 'No documents found'}
            
            # Check incremental mode
            if force_full_rebuild:
                self.state_manager.clear_all_state()
                is_incremental = False
                documents_to_process = all_documents
            else:
                is_incremental = self.check_incremental_mode(all_documents)
                documents_to_process = self.state_manager.get_documents_to_process(all_documents)
            
            logger.info(f"Pipeline mode: {'Incremental' if is_incremental else 'Full'}")
            logger.info(f"Documents to process: {len(documents_to_process)}")
            
            results = {
                'mode': 'incremental' if is_incremental else 'full',
                'total_documents': len(all_documents),
                'documents_to_process': len(documents_to_process),
                'phases': {}
            }
            
            # Phase 2: Document Processing
            if documents_to_process:
                doc_results = await self.process_documents_incremental(documents_to_process)
                results['phases']['document_processing'] = doc_results
                
                # Phase 3: Graph Augmentation (if new documents were processed)
                if doc_results['processed'] > 0:
                    self.state_manager.update_pipeline_phase("GRAPH_AUGMENTATION")
                    # Run Phase II augmentation for new content
                    augmentation_result = self.indexing_pipeline._phase_2_augmentation()
                    results['phases']['graph_augmentation'] = augmentation_result
                
                # Phase 4: Embedding Generation
                embedding_results = await self.generate_embeddings_incremental()
                results['phases']['embedding_generation'] = embedding_results
                
                # Phase 5: HNSW Indexing
                hnsw_results = await self.update_hnsw_index_incremental()
                results['phases']['hnsw_indexing'] = hnsw_results
            
            # Phase 6: Search System Update
            search_results = await self.update_search_system()
            results['phases']['search_system_update'] = search_results
            
            # Mark as finished
            self.state_manager.update_pipeline_phase("FINISHED")
            
            total_time = time.time() - pipeline_start_time
            results['total_time'] = total_time
            results['status'] = 'completed'
            
            # Get final stats
            processing_stats = self.state_manager.get_processing_stats()
            results['final_stats'] = processing_stats
            
            logger.info(f"Incremental pipeline completed in {total_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in incremental pipeline: {e}")
            self.state_manager.update_pipeline_phase("ERROR")
            return {
                'status': 'error',
                'error': str(e),
                'total_time': time.time() - pipeline_start_time
            }
    
    def get_search_system(self) -> Optional[AdvancedSearchSystem]:
        """Get the search system instance."""
        return self.search_system
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.state_manager.get_processing_stats()
    
    def reset_failed_documents(self):
        """Reset failed documents for reprocessing."""
        self.state_manager.reset_failed_documents()
    
    def clear_all_state(self):
        """Clear all processing state."""
        self.state_manager.clear_all_state()
import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import numpy as np

from ..vector.hnsw_service import HNSWService, SearchResult
from .personalized_pagerank import PersonalizedPageRank
from ..llm.llm_service import LLMService
from ..graph.graph_manager import GraphManager
from ..graph.node_types import NodeType
from ..config.settings import Config

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Comprehensive retrieval result matching NodeRAG's structure."""
    query: str
    hnsw_results: List[SearchResult]
    accurate_results: List[str]  # Exact entity matches
    ppr_results: List[Tuple[str, float]]  # PPR ranked nodes
    final_nodes: List[str]  # Final selected nodes for response
    entity_nodes: List[str]
    relationship_nodes: List[str]
    high_level_nodes: List[str]
    search_metadata: Dict[str, Any]

class AdvancedSearchSystem:
    """
    Advanced search system combining HNSW, entity matching, and PPR.
    Implements NodeRAG's multi-signal retrieval approach.
    """
    
    def __init__(self, 
                 graph_manager: GraphManager,
                 hnsw_service: HNSWService,
                 llm_service: LLMService):
        """
        Initialize advanced search system.
        
        Args:
            graph_manager: Graph management service
            hnsw_service: HNSW vector similarity service
            llm_service: LLM service for query processing
        """
        self.graph_manager = graph_manager
        self.hnsw_service = hnsw_service
        self.llm_service = llm_service
        
        # Initialize PPR
        self.ppr = PersonalizedPageRank(
            graph=graph_manager.graph,
            modified=True,
            weight='weight'
        )
        
        # Build entity lookup for exact matching
        self._build_entity_lookup()
        
        logger.info("Advanced search system initialized")
    
    def _build_entity_lookup(self):
        """Build entity lookup for exact string matching."""
        self.entity_lookup = {}
        
        # Get all entity nodes
        entity_nodes = self.graph_manager.get_nodes_by_type(NodeType.ENTITY)
        
        for entity in entity_nodes:
            # Index by lowercase content for case-insensitive search
            entity_text = entity.content.lower()
            if entity_text not in self.entity_lookup:
                self.entity_lookup[entity_text] = []
            self.entity_lookup[entity_text].append(entity.id)
        
        logger.info(f"Built entity lookup with {len(self.entity_lookup)} unique entities")
    
    def search(self, 
               query: str,
               k_hnsw: int = None,
               k_final: int = None,
               entity_nodes_limit: int = None,
               relationship_nodes_limit: int = None,
               high_level_nodes_limit: int = None) -> RetrievalResult:
        """
        Perform comprehensive search using multiple signals.
        
        Args:
            query: Search query
            k_hnsw: Number of HNSW results
            k_final: Final number of nodes to return
            entity_nodes_limit: Max entity nodes
            relationship_nodes_limit: Max relationship nodes  
            high_level_nodes_limit: Max high-level nodes
            
        Returns:
            RetrievalResult with comprehensive search results
        """
        # Set defaults
        k_hnsw = k_hnsw or Config.DEFAULT_SEARCH_K
        k_final = k_final or (Config.DEFAULT_SEARCH_K * 2)
        entity_nodes_limit = entity_nodes_limit or 10
        relationship_nodes_limit = relationship_nodes_limit or 30
        high_level_nodes_limit = high_level_nodes_limit or 10
        
        logger.info(f"Advanced search for query: '{query}'")
        
        # Step 1: HNSW semantic similarity search
        hnsw_results = self._hnsw_search(query, k_hnsw)
        
        # Step 2: Query decomposition and exact entity matching
        accurate_results = self._exact_entity_search(query)
        
        # Step 3: Personalized PageRank
        ppr_results = self._personalized_pagerank_search(
            hnsw_results, accurate_results, k_final
        )
        
        # Step 4: Post-process and categorize results
        final_result = self._post_process_results(
            query=query,
            hnsw_results=hnsw_results,
            accurate_results=accurate_results,
            ppr_results=ppr_results,
            entity_limit=entity_nodes_limit,
            relationship_limit=relationship_nodes_limit,
            high_level_limit=high_level_nodes_limit
        )
        
        logger.info(f"Search completed: {len(final_result.final_nodes)} final nodes")
        return final_result
    
    def _hnsw_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform database vector similarity search (replaces HNSW)."""
        try:
            # Get query embedding
            query_embeddings = self.llm_service.get_embeddings([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.warning("Failed to get query embedding")
                return []
            
            query_embedding = query_embeddings[0]
            
            # Get current org_id from graph metadata or default
            org_id = getattr(self, '_current_org_id', None)
            if not org_id:
                # Try to get org_id from first node's metadata
                for node_id, node_data in self.graph_manager.graph.nodes(data=True):
                    metadata = node_data.get('metadata', {})
                    if 'org_id' in metadata:
                        org_id = metadata['org_id']
                        self._current_org_id = org_id
                        break
                
                if not org_id:
                    logger.warning("No org_id found for database vector search")
                    return []
            
            # Perform database vector search
            from src.storage.neon_storage import NeonDBStorage
            storage = NeonDBStorage()
            
            search_results = storage.vector_similarity_search_sync(
                query_embedding=query_embedding,
                org_id=org_id,
                k=k,
                similarity_threshold=0.5  # Lower threshold for more results
            )
            
            # Convert database results to SearchResult format
            results = []
            for result in search_results:
                # Create SearchResult-like object with both similarity and distance
                search_result = type('SearchResult', (), {
                    'node_id': result['node_id'],
                    'similarity': result['similarity_score'],
                    'distance': 1.0 - result['similarity_score'],  # Convert similarity to distance
                    'content': result['content'],
                    'node_type': result['node_type'],
                    'metadata': result['metadata']
                })()
                results.append(search_result)
            
            logger.debug(f"Database vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in database vector search: {e}")
            return []
    
    def _exact_entity_search(self, query: str) -> List[str]:
        """Perform exact entity matching using query decomposition."""
        try:
            # Decompose query into entities
            decomposed_entities = self.llm_service.decompose_query(query)
            
            matched_entities = []
            
            # Search for exact matches
            for entity in decomposed_entities:
                entity_lower = entity.lower().strip()
                
                # Direct lookup
                if entity_lower in self.entity_lookup:
                    matched_entities.extend(self.entity_lookup[entity_lower])
                    continue
                
                # Fuzzy word-based matching
                entity_words = entity_lower.split()
                if len(entity_words) > 1:
                    # Create regex pattern for phrase matching
                    pattern = re.compile(r'\b' + r'\s+'.join(map(re.escape, entity_words)) + r'\b')
                    
                    for entity_text, node_ids in self.entity_lookup.items():
                        if pattern.search(entity_text):
                            matched_entities.extend(node_ids)
            
            # Remove duplicates while preserving order
            unique_entities = []
            seen = set()
            for entity_id in matched_entities:
                if entity_id not in seen:
                    unique_entities.append(entity_id)
                    seen.add(entity_id)
            
            logger.debug(f"Exact entity search found {len(unique_entities)} matches")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error in exact entity search: {e}")
            return []
    
    def _personalized_pagerank_search(self, 
                                    hnsw_results: List[SearchResult],
                                    accurate_results: List[str],
                                    k: int) -> List[Tuple[str, float]]:
        """Perform personalized PageRank search."""
        try:
            # Build personalization vector
            personalization = {}
            
            # Add HNSW results with similarity weight
            for result in hnsw_results:
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1.0 - result.distance
                personalization[result.node_id] = similarity * Config.SIMILARITY_WEIGHT
            
            # Add exact matches with accuracy weight
            for node_id in accurate_results:
                personalization[node_id] = Config.ACCURACY_WEIGHT
            
            if not personalization:
                logger.warning("No seed nodes for PPR")
                return []
            
            # Compute PPR
            ppr_results = self.ppr.search_top_k(
                personalization=personalization,
                k=k * 2,  # Get more results for filtering
                alpha=Config.PPR_ALPHA,
                max_iter=Config.PPR_MAX_ITERATIONS
            )
            
            logger.debug(f"PPR search returned {len(ppr_results)} results")
            return ppr_results
            
        except Exception as e:
            logger.error(f"Error in PPR search: {e}")
            return []
    
    def _post_process_results(self,
                            query: str,
                            hnsw_results: List[SearchResult],
                            accurate_results: List[str],
                            ppr_results: List[Tuple[str, float]],
                            entity_limit: int,
                            relationship_limit: int,
                            high_level_limit: int) -> RetrievalResult:
        """Post-process and categorize search results."""
        
        # Track already selected nodes
        selected_nodes = set()
        
        # Add HNSW and exact match results first
        for result in hnsw_results:
            selected_nodes.add(result.node_id)
        
        for node_id in accurate_results:
            selected_nodes.add(node_id)
        
        # Categorize nodes by type
        entity_nodes = []
        relationship_nodes = []
        high_level_nodes = []
        other_nodes = []
        
        # Process PPR results and categorize
        for node_id, score in ppr_results:
            if node_id in selected_nodes:
                continue
            
            node = self.graph_manager.get_node(node_id)
            if not node:
                continue
            
            # Categorize by node type
            if node.type == NodeType.ENTITY:
                if len(entity_nodes) < entity_limit:
                    entity_nodes.append(node_id)
                    
            elif node.type == NodeType.RELATIONSHIP:
                if len(relationship_nodes) < relationship_limit:
                    relationship_nodes.append(node_id)
                    
            elif node.type in [NodeType.HIGH_LEVEL, NodeType.OVERVIEW]:
                if len(high_level_nodes) < high_level_limit:
                    high_level_nodes.append(node_id)
                    
            else:
                other_nodes.append(node_id)
            
            selected_nodes.add(node_id)
            
            # Stop if we have enough in all categories
            if (len(entity_nodes) >= entity_limit and 
                len(relationship_nodes) >= relationship_limit and 
                len(high_level_nodes) >= high_level_limit):
                break
        
        # Add entity attributes for selected entities
        final_nodes = list(selected_nodes)
        for entity_id in entity_nodes:
            # Find attribute nodes for this entity
            connected_nodes = self.graph_manager.get_connected_nodes(entity_id)
            for connected in connected_nodes:
                if (connected.type == NodeType.ATTRIBUTE and 
                    connected.id not in selected_nodes):
                    final_nodes.append(connected.id)
                    selected_nodes.add(connected.id)
        
        # Add high-level content for high-level titles
        for hl_title_id in high_level_nodes:
            connected_nodes = self.graph_manager.get_connected_nodes(hl_title_id)
            for connected in connected_nodes:
                if (connected.type == NodeType.HIGH_LEVEL and 
                    connected.id not in selected_nodes):
                    final_nodes.append(connected.id)
                    selected_nodes.add(connected.id)
                    break  # Only add one high-level content per title
        
        # Add some other nodes if we haven't reached the limit
        remaining_slots = max(0, (entity_limit + relationship_limit + high_level_limit) - len(final_nodes))
        for node_id in other_nodes[:remaining_slots]:
            if node_id not in selected_nodes:
                final_nodes.append(node_id)
        
        # Build metadata
        search_metadata = {
            'hnsw_count': len(hnsw_results),
            'exact_match_count': len(accurate_results),
            'ppr_count': len(ppr_results),
            'entity_count': len(entity_nodes),
            'relationship_count': len(relationship_nodes),
            'high_level_count': len(high_level_nodes),
            'total_selected': len(final_nodes)
        }
        
        return RetrievalResult(
            query=query,
            hnsw_results=hnsw_results,
            accurate_results=accurate_results,
            ppr_results=ppr_results,
            final_nodes=final_nodes,
            entity_nodes=entity_nodes,
            relationship_nodes=relationship_nodes,
            high_level_nodes=high_level_nodes,
            search_metadata=search_metadata
        )
    
    def answer_query(self,
                    query: str,
                    use_structured_prompt: bool = True) -> Dict[str, Any]:
        """
        Generate answer for query using retrieved information.

        Args:
            query: User query
            use_structured_prompt: Whether to use structured prompt format

        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Perform search
            retrieval_result = self.search(query)

            # Build context from retrieved nodes and track sources
            context_parts = []
            source_files = set()

            for node_id in retrieval_result.final_nodes:
                node = self.graph_manager.get_node(node_id)
                if node:
                    if use_structured_prompt:
                        context_parts.append(f"[{node.type.value}] {node.content}")
                    else:
                        context_parts.append(node.content)

                    # Extract file_id from node metadata
                    if hasattr(node, 'metadata') and node.metadata:
                        file_id = node.metadata.get('file_id')
                        if file_id:
                            # Handle both string and list types
                            if isinstance(file_id, list):
                                source_files.update(file_id)
                            else:
                                source_files.add(file_id)

            retrieved_info = "\n\n".join(context_parts)

            # Generate answer with usage tracking
            answer_prompt = self.llm_service.prompt_manager.answer_generation.format(
                info=retrieved_info,
                query=query
            )

            response_data = self.llm_service._chat_completion_with_usage(answer_prompt, temperature=0.7)

            return {
                'query': query,
                'answer': response_data['response'],
                'retrieval_metadata': retrieval_result.search_metadata,
                'retrieved_nodes': len(retrieval_result.final_nodes),
                'context_length': len(retrieved_info),
                'sources': list(source_files),  # Add sources list
                'usage': response_data['usage']  # Add token usage
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'query': query,
                'answer': f"Error generating answer: {str(e)}",
                'error': True,
                'sources': [],
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
    
    def update_graph(self, new_graph_manager: GraphManager):
        """Update the search system with a new graph."""
        self.graph_manager = new_graph_manager
        self.ppr.update_graph(new_graph_manager.graph)
        self._build_entity_lookup()
        logger.info("Search system updated with new graph")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search system statistics."""
        return {
            'hnsw_stats': self.hnsw_service.get_stats(),
            'ppr_stats': self.ppr.get_stats(),
            'entity_lookup_size': len(self.entity_lookup),
            'graph_stats': self.graph_manager.get_stats()
        }
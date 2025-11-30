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
        
        # Step 2: AGENTIC QUERY UNDERSTANDING & EXPANSION
        # First understand the query intent and expand semantically
        expanded_queries = self._agentic_query_expansion(query)
        decomposed_queries = self._decompose_query(query)
        
        # Combine all query variations for comprehensive search
        all_search_queries = [query] + expanded_queries + decomposed_queries
        accurate_results = []
        
        for search_query in all_search_queries:
            sub_results = self._exact_entity_search(search_query)
            accurate_results.extend(sub_results)
        
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
        
        # Step 5: Enhanced relationship expansion for cross-chunk connections
        final_result.final_nodes = self._expand_with_cross_chunk_relationships(final_result.final_nodes)
        
        # Step 6: NodeRAG-style attribute expansion for richer context
        final_result.final_nodes = self._expand_with_attributes(final_result.final_nodes)
        
        logger.info(f"Search completed: {len(final_result.final_nodes)} final nodes (with attributes)")
        return final_result
    
    def _hnsw_search(self, query: str, k: int) -> List[SearchResult]:
        """Perform HNSW semantic similarity search."""
        try:
            # Get query embedding
            query_embeddings = self.llm_service.get_embeddings([query])
            if not query_embeddings or not query_embeddings[0]:
                logger.warning("Failed to get query embedding")
                return []
            
            query_embedding = np.array(query_embeddings[0], dtype=np.float32)
            
            # Perform HNSW search
            results = self.hnsw_service.search(
                query_embedding=query_embedding,
                k=k,
                ef=max(200, k * 2)  # Dynamic ef based on k
            )
            
            logger.debug(f"HNSW search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in HNSW search: {e}")
            return []
    
    def _exact_entity_search(self, query: str) -> List[str]:
        """Perform exact entity matching using query decomposition."""
        try:
            # Decompose query into entities
            decomposed_entities = self._decompose_query(query)
            
            matched_entities = []
            
            # Search for exact matches with enhanced coverage
            search_terms = decomposed_entities + [query.lower().strip()]  # Include original query
            
            for entity in search_terms:
                entity_lower = entity.lower().strip()
                
                # Direct lookup
                if entity_lower in self.entity_lookup:
                    matched_entities.extend(self.entity_lookup[entity_lower])
                    continue
                
                # Partial matching for company names like "Andor" matching "Andor Health"
                for entity_text, node_ids in self.entity_lookup.items():
                    if (entity_lower in entity_text or 
                        entity_text in entity_lower or
                        any(word in entity_text for word in entity_lower.split() if len(word) > 3)):
                        matched_entities.extend(node_ids)
                
                # Fuzzy word-based matching for phrases
                entity_words = entity_lower.split()
                if len(entity_words) > 1:
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
            
            # Build context from retrieved nodes
            context_parts = []
            
            for node_id in retrieval_result.final_nodes:
                node = self.graph_manager.get_node(node_id)
                if node:
                    if use_structured_prompt:
                        context_parts.append(f"[{node.type.value}] {node.content}")
                    else:
                        context_parts.append(node.content)
            
            retrieved_info = "\n\n".join(context_parts)
            
            # Generate answer
            answer_prompt = self.llm_service.prompt_manager.answer_generation.format(
                info=retrieved_info,
                query=query
            )
            
            response = self.llm_service._chat_completion(answer_prompt, temperature=0.1)  # Low temperature for deterministic responses
            
            return {
                'query': query,
                'answer': response,
                'retrieval_metadata': retrieval_result.search_metadata,
                'retrieved_nodes': len(retrieval_result.final_nodes),
                'context_length': len(retrieved_info)
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'query': query,
                'answer': f"Error generating answer: {str(e)}",
                'error': True
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
    
    def _decompose_query(self, query: str) -> List[str]:
        """
        NodeRAG-style query decomposition into searchable components.
        Breaks complex queries into entities and sub-queries.
        """
        try:
            # Use the LLM to decompose query into searchable elements
            decomposition_prompt = self.llm_service.prompt_manager.query_decomposition.format(
                query=query
            )
            
            response = self.llm_service._chat_completion(decomposition_prompt, temperature=0.3)
            
            # Parse JSON response
            import json
            decomposition_data = json.loads(response)
            elements = decomposition_data.get('elements', [])
            
            # Filter and clean elements
            cleaned_elements = []
            for element in elements:
                element = element.strip()
                if len(element) > 2 and element.lower() not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                    cleaned_elements.append(element)
            
            logger.info(f"Query decomposition: '{query}' → {cleaned_elements}")
            return cleaned_elements[:5]  # Limit to 5 elements
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            # Fallback: simple keyword extraction
            words = query.lower().split()
            keywords = [w for w in words if len(w) > 3 and w.isalpha()]
            return keywords[:3]
    
    def _expand_with_attributes(self, node_ids: List[str]) -> List[str]:
        """
        NodeRAG-style attribute expansion.
        Automatically adds related attribute nodes for richer context.
        """
        expanded_nodes = set(node_ids)
        
        for node_id in node_ids:
            try:
                node = self.graph_manager.get_node(node_id)
                if not node:
                    continue
                
                # Find connected attribute nodes
                if hasattr(self.graph_manager.graph, 'neighbors'):
                    neighbors = list(self.graph_manager.graph.neighbors(node_id))
                    
                    for neighbor_id in neighbors:
                        neighbor = self.graph_manager.get_node(neighbor_id)
                        if neighbor and neighbor.type == NodeType.ATTRIBUTE:
                            expanded_nodes.add(neighbor_id)
                            
                            # Limit attribute expansion to prevent explosion
                            if len(expanded_nodes) > len(node_ids) * 2:
                                break
                                
            except Exception as e:
                logger.warning(f"Attribute expansion failed for node {node_id}: {e}")
                continue
        
        expanded_list = list(expanded_nodes)
        logger.info(f"Expanded {len(node_ids)} nodes to {len(expanded_list)} with attributes")
        return expanded_list
    
    def _agentic_query_expansion(self, query: str) -> List[str]:
        """
        Advanced agentic query expansion - understands intent and generates semantic variations.
        This is the core of true NodeRAG-style intelligent retrieval.
        """
        try:
            # Agentic prompt that understands query intent and semantics
            expansion_prompt = f"""
You are an intelligent query expansion system. Given a user question, generate 5-8 semantically equivalent variations that would help find the same information.

Rules:
1. Understand the INTENT behind the question
2. Generate synonyms and alternative phrasings  
3. Include both formal and informal variations
4. Handle entity name variations (e.g., "Andor" → "Andor Health")
5. Convert temporal expressions (e.g., "born" → "founded", "established", "created")

Original Query: "{query}"

Generate query variations as a JSON array:
{{"variations": ["variation1", "variation2", ...]}}

Focus on:
- Synonym replacement
- Entity variations  
- Temporal/action alternatives
- Formal vs informal phrasing
"""

            response = self.llm_service._chat_completion(expansion_prompt, temperature=0.3)
            
            import json
            expansion_data = json.loads(response)
            variations = expansion_data.get('variations', [])
            
            # Clean and filter variations
            clean_variations = []
            for variation in variations:
                variation = variation.strip().lower()
                if variation != query.lower() and len(variation) > 5:
                    clean_variations.append(variation)
            
            logger.info(f"Query expansion: '{query}' → {len(clean_variations)} variations")
            return clean_variations[:6]  # Limit to best 6 variations
            
        except Exception as e:
            logger.warning(f"Agentic query expansion failed: {e}")
            # Fallback: basic synonym expansion
            return self._basic_query_expansion(query)
    
    def _basic_query_expansion(self, query: str) -> List[str]:
        """Fallback query expansion with predefined synonyms"""
        query_lower = query.lower()
        expansions = []
        
        # Company/organization founding synonyms
        if any(word in query_lower for word in ['born', 'founded', 'established', 'created', 'started']):
            entity = self._extract_entity_from_query(query)
            if entity:
                expansions.extend([
                    f"when was {entity} founded",
                    f"when did {entity} start", 
                    f"{entity} establishment date",
                    f"{entity} creation timeline",
                    f"founding of {entity}"
                ])
        
        # Age/time related queries
        if any(word in query_lower for word in ['age', 'old', 'years']):
            entity = self._extract_entity_from_query(query) 
            if entity:
                expansions.extend([
                    f"how long has {entity} existed",
                    f"{entity} company age",
                    f"years since {entity} founded"
                ])
        
        return expansions[:4]
    
    def _expand_with_cross_chunk_relationships(self, node_ids: List[str]) -> List[str]:
        """
        Enhanced expansion that includes cross-chunk relationship nodes for better context.
        This addresses the parallel processing quality issue by surfacing relationships
        that span multiple document chunks.
        """
        expanded_nodes = set(node_ids)
        
        for node_id in node_ids:
            try:
                node = self.graph_manager.get_node(node_id)
                if not node:
                    continue
                
                # Look for cross-chunk relationships involving this node
                if hasattr(self.graph_manager.graph, 'neighbors'):
                    neighbors = list(self.graph_manager.graph.neighbors(node_id))
                    
                    for neighbor_id in neighbors:
                        neighbor = self.graph_manager.get_node(neighbor_id)
                        if (neighbor and neighbor.type == NodeType.RELATIONSHIP and 
                            neighbor.metadata.get('cross_chunk', False)):
                            
                            # Add the cross-chunk relationship node
                            expanded_nodes.add(neighbor_id)
                            
                            # Add the other entities involved in this relationship
                            relationship_neighbors = list(self.graph_manager.graph.neighbors(neighbor_id))
                            for rel_neighbor_id in relationship_neighbors:
                                rel_neighbor = self.graph_manager.get_node(rel_neighbor_id)
                                if (rel_neighbor and rel_neighbor.type == NodeType.ENTITY and 
                                    rel_neighbor.id != node_id):
                                    expanded_nodes.add(rel_neighbor_id)
                            
                            # Limit expansion to prevent explosion
                            if len(expanded_nodes) > len(node_ids) * 3:
                                break
                                
            except Exception as e:
                logger.warning(f"Cross-chunk relationship expansion failed for node {node_id}: {e}")
                continue
        
        expanded_list = list(expanded_nodes)
        added_count = len(expanded_list) - len(node_ids)
        if added_count > 0:
            logger.info(f"Added {added_count} cross-chunk relationship nodes for richer context")
        return expanded_list
    
    def _extract_entity_from_query(self, query: str) -> str:
        """Extract main entity from query for expansion"""
        query_words = query.lower().split()
        
        # Common entity patterns
        for word in query_words:
            if word not in ['what', 'when', 'where', 'how', 'is', 'the', 'of', 'did', 'was', 'age', 'born', 'founded']:
                return word.title()
        
        return ""
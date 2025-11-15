import os
import pickle
import numpy as np
import hnswlib
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx

from ..config.settings import Config

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """HNSW search result with metadata."""
    node_id: str
    distance: float
    node_type: str
    content: str

@dataclass
class HNSWIndex:
    """HNSW index metadata."""
    dimension: int
    max_elements: int
    ef_construction: int
    m: int
    space: str
    id_mapping: Dict[int, str]  # internal_id -> node_id
    reverse_mapping: Dict[str, int]  # node_id -> internal_id

class HNSWService:
    """
    HNSW (Hierarchical Navigable Small World) vector similarity search service.
    Implements fast approximate nearest neighbor search for node embeddings.
    """
    
    def __init__(self, 
                 dimension: int = 1536,  # OpenAI embedding dimension
                 max_elements: int = 100000,
                 ef_construction: int = 200,
                 m: int = 50,
                 space: str = 'cosine'):
        """
        Initialize HNSW service.
        
        Args:
            dimension: Embedding vector dimension
            max_elements: Maximum number of elements in index
            ef_construction: Size of dynamic candidate list
            m: Number of bi-directional links created for every new element
            space: Distance metric ('cosine', 'l2', 'ip')
        """
        self.dimension = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = m
        self.space = space
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space=space, dim=dimension)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m
        )
        
        # ID mappings
        self.id_mapping: Dict[int, str] = {}  # internal_id -> node_id
        self.reverse_mapping: Dict[str, int] = {}  # node_id -> internal_id
        self.node_metadata: Dict[str, Dict[str, Any]] = {}  # node_id -> metadata
        
        # Current index state
        self.current_count = 0
        self.is_built = False
        
        logger.info(f"HNSW service initialized: dim={dimension}, space={space}")
    
    def add_node_embedding(self, 
                          node_id: str, 
                          embedding: np.ndarray, 
                          metadata: Dict[str, Any] = None) -> bool:
        """
        Add a single node embedding to the index.
        
        Args:
            node_id: Unique node identifier
            embedding: Node embedding vector
            metadata: Additional node metadata
            
        Returns:
            bool: True if successfully added
        """
        try:
            if node_id in self.reverse_mapping:
                logger.warning(f"Node {node_id} already exists in index")
                return False
            
            if self.current_count >= self.max_elements:
                logger.error(f"HNSW index is full (max: {self.max_elements})")
                return False
            
            # Validate embedding
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            
            if embedding.shape[0] != self.dimension:
                raise ValueError(f"Embedding dimension {embedding.shape[0]} != expected {self.dimension}")
            
            # Add to index
            internal_id = self.current_count
            self.index.add_items(embedding.reshape(1, -1), np.array([internal_id]))
            
            # Update mappings
            self.id_mapping[internal_id] = node_id
            self.reverse_mapping[node_id] = internal_id
            self.node_metadata[node_id] = metadata or {}
            
            self.current_count += 1
            logger.debug(f"Added node {node_id} to HNSW index (internal_id: {internal_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding node {node_id} to HNSW index: {e}")
            return False
    
    def add_batch_embeddings(self, 
                            nodes: List[Tuple[str, np.ndarray, Dict[str, Any]]]) -> int:
        """
        Add multiple node embeddings in batch.
        
        Args:
            nodes: List of (node_id, embedding, metadata) tuples
            
        Returns:
            int: Number of successfully added nodes
        """
        successful_adds = 0
        
        # Prepare batch data
        embeddings_batch = []
        internal_ids_batch = []
        
        for node_id, embedding, metadata in nodes:
            if node_id in self.reverse_mapping:
                logger.warning(f"Skipping duplicate node {node_id}")
                continue
                
            if self.current_count >= self.max_elements:
                logger.warning(f"HNSW index is full, stopping batch add")
                break
            
            try:
                # Validate embedding
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                if embedding.shape[0] != self.dimension:
                    logger.error(f"Invalid embedding dimension for {node_id}: {embedding.shape[0]}")
                    continue
                
                # Prepare for batch add
                internal_id = self.current_count
                embeddings_batch.append(embedding)
                internal_ids_batch.append(internal_id)
                
                # Update mappings
                self.id_mapping[internal_id] = node_id
                self.reverse_mapping[node_id] = internal_id
                self.node_metadata[node_id] = metadata or {}
                
                self.current_count += 1
                successful_adds += 1
                
            except Exception as e:
                logger.error(f"Error preparing node {node_id} for batch add: {e}")
                continue
        
        # Add batch to index
        if embeddings_batch:
            try:
                embeddings_array = np.array(embeddings_batch, dtype=np.float32)
                internal_ids_array = np.array(internal_ids_batch)
                
                self.index.add_items(embeddings_array, internal_ids_array)
                logger.info(f"Successfully added {successful_adds} nodes to HNSW index")
                
            except Exception as e:
                logger.error(f"Error in batch add to HNSW index: {e}")
                # Rollback mappings
                for internal_id in internal_ids_batch:
                    node_id = self.id_mapping.pop(internal_id, None)
                    if node_id:
                        self.reverse_mapping.pop(node_id, None)
                        self.node_metadata.pop(node_id, None)
                        self.current_count -= 1
                successful_adds = 0
        
        return successful_adds
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 10,
               ef: int = None) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_embedding: Query vector
            k: Number of nearest neighbors to return
            ef: Size of dynamic candidate list (None = use default)
            
        Returns:
            List[SearchResult]: Search results with metadata
        """
        try:
            if self.current_count == 0:
                logger.warning("HNSW index is empty")
                return []
            
            # Validate query embedding
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            if query_embedding.shape[0] != self.dimension:
                raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} != expected {self.dimension}")
            
            # Set ef parameter for search
            if ef is not None:
                self.index.set_ef(ef)
            else:
                # Use default ef = max(ef_construction, k)
                self.index.set_ef(max(self.ef_construction, k))
            
            # Perform search
            k = min(k, self.current_count)  # Don't search for more than available
            internal_ids, distances = self.index.knn_query(
                query_embedding.reshape(1, -1), 
                k=k
            )
            
            # Convert results
            results = []
            for internal_id, distance in zip(internal_ids[0], distances[0]):
                node_id = self.id_mapping.get(internal_id)
                if node_id:
                    metadata = self.node_metadata.get(node_id, {})
                    results.append(SearchResult(
                        node_id=node_id,
                        distance=float(distance),
                        node_type=metadata.get('node_type', 'unknown'),
                        content=metadata.get('content', '')[:200] + '...' if len(metadata.get('content', '')) > 200 else metadata.get('content', '')
                    ))
            
            logger.debug(f"HNSW search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in HNSW search: {e}")
            return []
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node metadata by ID."""
        return self.node_metadata.get(node_id)
    
    def get_embedding_by_id(self, node_id: str) -> Optional[np.ndarray]:
        """Get node embedding by ID."""
        internal_id = self.reverse_mapping.get(node_id)
        if internal_id is not None:
            try:
                # HNSW doesn't directly support getting embeddings by ID
                # This would require storing embeddings separately
                logger.warning("Getting embeddings by ID not implemented - requires separate storage")
                return None
            except Exception as e:
                logger.error(f"Error getting embedding for {node_id}: {e}")
        return None
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove node from index.
        Note: HNSW doesn't support deletion, so this marks as inactive.
        """
        if node_id in self.reverse_mapping:
            # Mark as deleted in metadata
            if node_id in self.node_metadata:
                self.node_metadata[node_id]['deleted'] = True
            logger.warning(f"Node {node_id} marked as deleted (HNSW doesn't support true deletion)")
            return True
        return False
    
    def save_index(self, filepath: str = None) -> bool:
        """Save HNSW index and metadata to disk."""
        try:
            if filepath is None:
                filepath = os.path.join(Config.DATA_DIR, "processed", "hnsw_index")
            
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save HNSW index
            index_file = f"{filepath}.hnsw"
            self.index.save_index(index_file)
            
            # Save metadata
            metadata_file = f"{filepath}.metadata"
            metadata = {
                'dimension': self.dimension,
                'max_elements': self.max_elements,
                'ef_construction': self.ef_construction,
                'm': self.m,
                'space': self.space,
                'current_count': self.current_count,
                'id_mapping': self.id_mapping,
                'reverse_mapping': self.reverse_mapping,
                'node_metadata': self.node_metadata
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"HNSW index saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving HNSW index: {e}")
            return False
    
    def load_index(self, filepath: str = None) -> bool:
        """Load HNSW index and metadata from disk."""
        try:
            if filepath is None:
                filepath = os.path.join(Config.DATA_DIR, "processed", "hnsw_index")
            index_file = f"{filepath}.hnsw"
            metadata_file = f"{filepath}.metadata"
            
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                logger.error(f"HNSW index files not found: {filepath}")
                return False
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Validate metadata
            if metadata['dimension'] != self.dimension:
                logger.warning(f"Dimension mismatch: {metadata['dimension']} != {self.dimension}")
                logger.info(f"Clearing old index and creating new one with dimension {self.dimension}")
                # Delete old index files
                try:
                    os.remove(index_file)
                    os.remove(metadata_file)
                    logger.info("Old HNSW index files deleted")
                except Exception as e:
                    logger.warning(f"Could not delete old index files: {e}")
                return False
            
            # Load HNSW index
            self.index = hnswlib.Index(space=metadata['space'], dim=metadata['dimension'])
            self.index.load_index(index_file, max_elements=metadata['max_elements'])
            
            # Restore state
            self.dimension = metadata['dimension']
            self.max_elements = metadata['max_elements']
            self.ef_construction = metadata['ef_construction']
            self.m = metadata['m']
            self.space = metadata['space']
            self.current_count = metadata['current_count']
            self.id_mapping = metadata['id_mapping']
            self.reverse_mapping = metadata['reverse_mapping']
            self.node_metadata = metadata['node_metadata']
            
            logger.info(f"HNSW index loaded from {filepath}, {self.current_count} elements")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HNSW index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HNSW index statistics."""
        return {
            'dimension': self.dimension,
            'max_elements': self.max_elements,
            'current_count': self.current_count,
            'ef_construction': self.ef_construction,
            'm': self.m,
            'space': self.space,
            'memory_usage_mb': self.current_count * self.dimension * 4 / (1024 * 1024),  # Rough estimate
            'fill_percentage': (self.current_count / self.max_elements) * 100
        }
    
    def build_nx_graph(self) -> nx.Graph:
        """
        Build NetworkX graph from HNSW connections (Layer 0).
        This creates a similarity-based graph structure.
        """
        try:
            # Get layer 0 graph from HNSW
            # Note: This requires accessing internal HNSW structure
            # which may not be available in all HNSW implementations
            
            G = nx.Graph()
            
            # Add all nodes
            for node_id in self.reverse_mapping.keys():
                metadata = self.node_metadata.get(node_id, {})
                if not metadata.get('deleted', False):
                    G.add_node(node_id, **metadata)
            
            # This is a simplified version - actual implementation would need
            # to access HNSW's internal graph structure
            logger.warning("HNSW graph construction is simplified - may need custom HNSW implementation")
            
            return G
            
        except Exception as e:
            logger.error(f"Error building NetworkX graph from HNSW: {e}")
            return nx.Graph()
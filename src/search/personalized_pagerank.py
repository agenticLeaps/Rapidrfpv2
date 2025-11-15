import networkx as nx
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional
import logging
from operator import itemgetter

logger = logging.getLogger(__name__)

class PersonalizedPageRank:
    """
    Personalized PageRank implementation for graph-based search.
    Based on NodeRAG's sparse PPR implementation with optimizations.
    """
    
    def __init__(self, graph: nx.Graph, modified: bool = True, weight: str = 'weight'):
        """
        Initialize Personalized PageRank.
        
        Args:
            graph: NetworkX graph
            modified: Whether to use modified PPR (handles disconnected components)
            weight: Edge weight attribute name
        """
        self.graph = graph
        self.nodes = list(self.graph.nodes())
        self.modified = modified
        self.weight = weight
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        
        # Build transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
        logger.info(f"PPR initialized with {self.n_nodes} nodes, modified={modified}")
    
    def _build_transition_matrix(self) -> sp.csc_matrix:
        """Build sparse transition matrix for PPR computation."""
        try:
            if self.n_nodes == 0:
                return sp.csc_matrix((0, 0))
            
            # Get adjacency matrix with weights
            adjacency_matrix = nx.adjacency_matrix(
                self.graph, 
                nodelist=self.nodes,
                weight=self.weight
            )
            
            # Make symmetric (undirected graph)
            adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
            
            if self.modified:
                # Handle disconnected nodes by adding self-loops
                out_degree = np.array(adjacency_matrix.sum(1)).flatten()
                
                # Convert to lil_matrix for efficient modification
                adjacency_matrix = sp.lil_matrix(adjacency_matrix)
                
                # Add uniform connections for isolated nodes
                isolated_nodes = (out_degree == 0)
                if np.any(isolated_nodes):
                    adjacency_matrix[isolated_nodes, :] = 1.0 / self.n_nodes
                    adjacency_matrix.setdiag(0)  # Remove self-loops
                
                # Convert back to csc for efficient arithmetic
                adjacency_matrix = sp.csc_matrix(adjacency_matrix)
                
                # Recalculate out-degree
                out_degree = np.array(adjacency_matrix.sum(1)).flatten()
            
            else:
                out_degree = np.array(adjacency_matrix.sum(1)).flatten()
            
            # Avoid division by zero
            out_degree[out_degree == 0] = 1.0
            
            # Create transition matrix (column-stochastic)
            # Each column sums to 1, representing transition probabilities
            transition_matrix = adjacency_matrix.multiply(1.0 / out_degree[:, np.newaxis])
            
            # Transpose to get row-stochastic matrix
            transition_matrix = transition_matrix.T
            
            return sp.csc_matrix(transition_matrix)
            
        except Exception as e:
            logger.error(f"Error building transition matrix: {e}")
            return sp.csc_matrix((self.n_nodes, self.n_nodes))
    
    def compute_ppr(self, 
                   personalization: Dict[str, float],
                   alpha: float = 0.85,
                   max_iter: int = 100,
                   epsilon: float = 1e-6) -> List[Tuple[str, float]]:
        """
        Compute Personalized PageRank scores.
        
        Args:
            personalization: Dictionary mapping node IDs to personalization weights
            alpha: Damping factor (probability of continuing random walk)
            max_iter: Maximum number of iterations
            epsilon: Convergence threshold
            
        Returns:
            List of (node_id, score) tuples sorted by score (descending)
        """
        try:
            if self.n_nodes == 0:
                return []
            
            # Initialize personalization vector
            personalization_vector = np.zeros(self.n_nodes)
            
            # Set personalization weights
            total_weight = 0.0
            for node_id, weight in personalization.items():
                if node_id in self.node_to_idx:
                    idx = self.node_to_idx[node_id]
                    personalization_vector[idx] = weight
                    total_weight += weight
            
            # Normalize personalization vector
            if total_weight > 0:
                personalization_vector /= total_weight
            else:
                # Uniform distribution if no valid personalization
                personalization_vector.fill(1.0 / self.n_nodes)
            
            # Initialize PageRank vector
            pagerank_vector = personalization_vector.copy()
            
            # Power iteration
            for iteration in range(max_iter):
                prev_pagerank = pagerank_vector.copy()
                
                # PPR update: α * M^T * v + (1-α) * s
                pagerank_vector = (alpha * self.transition_matrix.dot(pagerank_vector) + 
                                 (1 - alpha) * personalization_vector)
                
                # Check convergence
                diff = np.linalg.norm(pagerank_vector - prev_pagerank, ord=1)
                if diff < epsilon:
                    logger.debug(f"PPR converged after {iteration + 1} iterations (diff={diff:.2e})")
                    break
            else:
                logger.warning(f"PPR did not converge after {max_iter} iterations (diff={diff:.2e})")
            
            # Convert to node_id, score pairs and sort
            results = []
            for idx, score in enumerate(pagerank_vector):
                node_id = self.idx_to_node[idx]
                results.append((node_id, float(score)))
            
            # Sort by score (descending)
            results.sort(key=itemgetter(1), reverse=True)
            
            logger.debug(f"PPR computed for {len(personalization)} seed nodes, "
                        f"top score: {results[0][1]:.6f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing PPR: {e}")
            return []
    
    def search_top_k(self, 
                     personalization: Dict[str, float],
                     k: int = 50,
                     alpha: float = 0.85,
                     max_iter: int = 100) -> List[Tuple[str, float]]:
        """
        Get top-k nodes by PPR score.
        
        Args:
            personalization: Seed nodes and weights
            k: Number of top results to return
            alpha: PPR damping factor
            max_iter: Maximum iterations
            
        Returns:
            Top-k (node_id, score) pairs
        """
        all_results = self.compute_ppr(
            personalization=personalization,
            alpha=alpha,
            max_iter=max_iter
        )
        
        return all_results[:k]
    
    def get_node_neighbors(self, node_id: str, max_neighbors: int = None) -> List[str]:
        """Get direct neighbors of a node."""
        if node_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(node_id))
        
        if max_neighbors:
            neighbors = neighbors[:max_neighbors]
        
        return neighbors
    
    def compute_multi_source_ppr(self,
                                seed_nodes: List[str],
                                weights: Optional[List[float]] = None,
                                alpha: float = 0.85,
                                k: int = 50) -> List[Tuple[str, float]]:
        """
        Compute PPR from multiple seed nodes.
        
        Args:
            seed_nodes: List of seed node IDs
            weights: Optional weights for seed nodes (uniform if None)
            alpha: PPR damping factor
            k: Number of results to return
            
        Returns:
            Top-k (node_id, score) pairs
        """
        if not seed_nodes:
            return []
        
        # Create personalization dictionary
        if weights is None:
            weights = [1.0] * len(seed_nodes)
        
        personalization = {
            node_id: weight 
            for node_id, weight in zip(seed_nodes, weights)
            if node_id in self.node_to_idx
        }
        
        return self.search_top_k(
            personalization=personalization,
            k=k,
            alpha=alpha
        )
    
    def update_graph(self, new_graph: nx.Graph):
        """Update the graph and rebuild transition matrix."""
        self.graph = new_graph
        self.nodes = list(self.graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        
        # Rebuild transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
        logger.info(f"PPR graph updated: {self.n_nodes} nodes")
    
    def get_stats(self) -> Dict[str, any]:
        """Get PPR statistics."""
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.graph.number_of_edges(),
            'modified_ppr': self.modified,
            'weight_attribute': self.weight,
            'transition_matrix_nnz': self.transition_matrix.nnz if self.transition_matrix is not None else 0,
            'density': nx.density(self.graph) if self.n_nodes > 0 else 0.0
        }
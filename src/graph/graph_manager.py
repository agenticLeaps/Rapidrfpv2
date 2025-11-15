import networkx as nx
import pickle
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import logging

from .node_types import Node, Edge, NodeType

logger = logging.getLogger(__name__)

class GraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()  # Changed from MultiDiGraph to DiGraph for community detection
        self.entity_index = defaultdict(set)  # For entity deduplication
        self.community_assignments = {}
        
    def add_node(self, node: Node) -> bool:
        """Add a node to the graph. Returns True if added, False if duplicate."""
        if node.type == NodeType.ENTITY:
            # Check for entity deduplication
            existing_entity = self._find_duplicate_entity(node)
            if existing_entity:
                # Merge with existing entity
                self._merge_entities(existing_entity, node)
                return False
        
        self.graph.add_node(
            node.id,
            type=node.type.value,
            content=node.content,
            metadata=node.metadata,
            embeddings=node.embeddings
        )
        
        if node.type == NodeType.ENTITY:
            self.entity_index[node.content.lower()].add(node.id)
        
        logger.debug(f"Added node {node.id} of type {node.type.value}")
        return True
    
    def update_node(self, node: Node) -> bool:
        """Update an existing node in the graph."""
        if not self.graph.has_node(node.id):
            logger.warning(f"Cannot update node {node.id}: node doesn't exist")
            return False
        
        # Update node attributes
        self.graph.nodes[node.id].update({
            'type': node.type.value,
            'content': node.content,
            'metadata': node.metadata,
            'embeddings': node.embeddings
        })
        
        logger.debug(f"Updated node {node.id} of type {node.type.value}")
        return True
    
    def add_edge(self, edge: Edge) -> bool:
        """Add an edge to the graph."""
        if not self.graph.has_node(edge.source) or not self.graph.has_node(edge.target):
            logger.warning(f"Cannot add edge: nodes {edge.source} or {edge.target} don't exist")
            return False
        
        # Prevent self-loops
        if edge.source == edge.target:
            logger.debug(f"Skipping self-loop edge: {edge.source} -> {edge.target}")
            return False
        
        # For DiGraph, if edge already exists, update it instead of creating duplicate
        if self.graph.has_edge(edge.source, edge.target):
            # Update existing edge with new relationship type if different
            existing_data = self.graph[edge.source][edge.target]
            if existing_data.get('relationship_type') != edge.relationship_type:
                # Combine relationship types
                existing_rel = existing_data.get('relationship_type', '')
                new_rel = f"{existing_rel}; {edge.relationship_type}" if existing_rel else edge.relationship_type
                self.graph[edge.source][edge.target]['relationship_type'] = new_rel
        else:
            self.graph.add_edge(
                edge.source,
                edge.target,
                relationship_type=edge.relationship_type,
                weight=edge.weight,
                metadata=edge.metadata
            )
        
        logger.debug(f"Added edge {edge.source} -> {edge.target}")
        return True
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        if not self.graph.has_node(node_id):
            return None
        
        data = self.graph.nodes[node_id]
        return Node(
            id=node_id,
            type=NodeType(data['type']),
            content=data['content'],
            metadata=data['metadata'],
            embeddings=data.get('embeddings')
        )
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data['type'] == node_type.value:
                nodes.append(Node(
                    id=node_id,
                    type=NodeType(data['type']),
                    content=data['content'],
                    metadata=data['metadata'],
                    embeddings=data.get('embeddings')
                ))
        return nodes
    
    def get_connected_nodes(self, node_id: str, relationship_types: Optional[List[str]] = None) -> List[Node]:
        """Get nodes connected to the given node."""
        if not self.graph.has_node(node_id):
            return []
        
        connected_nodes = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph[node_id][neighbor]  # DiGraph has single edge data, not dict of edges
            if relationship_types is None or edge_data.get('relationship_type') in relationship_types:
                node = self.get_node(neighbor)
                if node:
                    connected_nodes.append(node)
        
        return connected_nodes
    
    def get_entity_mentions(self, entity_name: str) -> List[Node]:
        """Get all mentions of an entity (case-insensitive)."""
        entity_ids = self.entity_index.get(entity_name.lower(), set())
        return [self.get_node(entity_id) for entity_id in entity_ids if self.get_node(entity_id)]
    
    def calculate_node_importance(self) -> Dict[str, float]:
        """Calculate node importance using betweenness centrality and k-core."""
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.graph.to_undirected())
        
        # K-core decomposition
        core_numbers = nx.core_number(self.graph.to_undirected())
        
        # Combine both metrics
        importance_scores = {}
        max_betweenness = max(betweenness.values()) if betweenness else 1
        max_core = max(core_numbers.values()) if core_numbers else 1
        
        # Avoid division by zero
        if max_betweenness == 0:
            max_betweenness = 1
        if max_core == 0:
            max_core = 1
        
        for node_id in self.graph.nodes():
            betweenness_norm = betweenness.get(node_id, 0) / max_betweenness
            core_norm = core_numbers.get(node_id, 0) / max_core
            importance_scores[node_id] = (betweenness_norm + core_norm) / 2
        
        return importance_scores
    
    def get_important_entities(self, percentage: float = 0.2) -> List[Node]:
        """Get the most important entities based on graph metrics."""
        importance_scores = self.calculate_node_importance()
        entity_nodes = self.get_nodes_by_type(NodeType.ENTITY)
        
        # Sort entities by importance
        entity_scores = [(node, importance_scores.get(node.id, 0)) for node in entity_nodes]
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top percentage
        num_important = max(1, int(len(entity_scores) * percentage))
        return [node for node, score in entity_scores[:num_important]]
    
    def detect_communities(self, resolution: float = 1.0, random_state: int = 42) -> Dict[str, int]:
        """Detect communities using Leiden algorithm."""
        try:
            import community as community_louvain
            import networkx as nx
            
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Remove self-loops if they exist
            if nx.number_of_selfloops(undirected_graph) > 0:
                logger.info(f"Removing {nx.number_of_selfloops(undirected_graph)} self-loops from graph")
                undirected_graph.remove_edges_from(nx.selfloop_edges(undirected_graph))
            
            # Use Louvain algorithm (substitute for Leiden)
            self.community_assignments = community_louvain.best_partition(
                undirected_graph, 
                resolution=resolution,
                random_state=random_state
            )
            
            logger.info(f"Detected {len(set(self.community_assignments.values()))} communities")
            return self.community_assignments
            
        except ImportError:
            logger.error("Community detection library not available")
            return {}
    
    def get_community_nodes(self, community_id: int) -> List[Node]:
        """Get all nodes belonging to a specific community."""
        node_ids = [node_id for node_id, comm_id in self.community_assignments.items() 
                   if comm_id == community_id]
        return [self.get_node(node_id) for node_id in node_ids if self.get_node(node_id)]
    
    def save_graph(self, filepath: str) -> bool:
        """Save the graph to disk."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'entity_index': dict(self.entity_index),
                    'community_assignments': self.community_assignments
                }, f)
            logger.info(f"Graph saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
            return False
    
    def load_graph(self, filepath: str) -> bool:
        """Load the graph from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.entity_index = defaultdict(set, data.get('entity_index', {}))
                self.community_assignments = data.get('community_assignments', {})
            logger.info(f"Graph loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        node_type_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_type_counts[data['type']] += 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_type_counts': dict(node_type_counts),
            'num_communities': len(set(self.community_assignments.values())) if self.community_assignments else 0,
            'is_connected': nx.is_weakly_connected(self.graph)
        }
    
    def _find_duplicate_entity(self, entity_node: Node) -> Optional[str]:
        """Find if an entity already exists (case-insensitive)."""
        existing_ids = self.entity_index.get(entity_node.content.lower())
        return next(iter(existing_ids)) if existing_ids else None
    
    def _merge_entities(self, existing_id: str, new_node: Node):
        """Merge a new entity node with an existing one."""
        existing_node = self.get_node(existing_id)
        if not existing_node:
            return
        
        # Merge metadata
        merged_metadata = existing_node.metadata.copy()
        for key, value in new_node.metadata.items():
            if key in merged_metadata:
                if isinstance(merged_metadata[key], list):
                    if isinstance(value, list):
                        merged_metadata[key].extend(value)
                    else:
                        merged_metadata[key].append(value)
                elif isinstance(value, list):
                    merged_metadata[key] = [merged_metadata[key]] + value
                else:
                    merged_metadata[key] = [merged_metadata[key], value]
            else:
                merged_metadata[key] = value
        
        # Update the existing node
        self.graph.nodes[existing_id]['metadata'] = merged_metadata
        logger.debug(f"Merged entity {new_node.id} into {existing_id}")
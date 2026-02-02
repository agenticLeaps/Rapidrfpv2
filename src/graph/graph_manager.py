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

        # Ensure file_ids is tracked as a set in metadata
        metadata = node.metadata.copy()
        file_id = metadata.get('file_id')
        if file_id:
            # Convert single file_id to a set of file_ids for multi-file tracking
            metadata['file_ids'] = {file_id}
        else:
            metadata['file_ids'] = set()

        self.graph.add_node(
            node.id,
            type=node.type.value,
            content=node.content,
            metadata=metadata,
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

        # First, handle file_ids specially to track all files referencing this entity
        existing_file_ids = merged_metadata.get('file_ids', set())
        new_file_id = new_node.metadata.get('file_id')
        new_file_ids = new_node.metadata.get('file_ids', set())

        # Ensure existing_file_ids is a set
        if not isinstance(existing_file_ids, set):
            existing_file_ids = {existing_file_ids} if existing_file_ids else set()

        # Combine file_ids from both nodes
        combined_file_ids = existing_file_ids.copy()
        if new_file_id:
            combined_file_ids.add(new_file_id)
        if new_file_ids:
            combined_file_ids.update(new_file_ids)

        merged_metadata['file_ids'] = combined_file_ids

        # Merge other metadata fields
        for key, value in new_node.metadata.items():
            if key in ['file_id', 'file_ids']:
                continue  # Already handled above

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
        logger.debug(f"Merged entity {new_node.id} into {existing_id}, now referenced by {len(combined_file_ids)} files")
    
    # ==================== ORG-LEVEL GRAPH METHODS ====================
    
    def load_from_data(self, graph_data: bytes) -> bool:
        """Load existing graph data into current instance"""
        try:
            import pickle
            from collections import defaultdict
            
            data = pickle.loads(graph_data)
            
            # Load the graph
            self.graph = data['graph']
            
            # Rebuild entity index
            self.entity_index = defaultdict(set)
            for node_id, node_data in self.graph.nodes(data=True):
                if node_data.get('type') == NodeType.ENTITY.value:
                    entity_content = node_data.get('content', '').lower()
                    self.entity_index[entity_content].add(node_id)
            
            # Load community assignments if available
            self.community_assignments = data.get('community_assignments', {})
            
            logger.info(f"Loaded existing graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error loading graph data: {e}")
            return False
    
    def merge_file_nodes(self, new_nodes: List[Node], file_id: str) -> Dict[str, Any]:
        """Merge new file nodes with entity deduplication"""
        stats = {
            'nodes_added': 0,
            'entities_merged': 0,
            'edges_added': 0,
            'file_id': file_id
        }
        
        node_mapping = {}  # Map new node IDs to actual IDs in graph
        
        try:
            # Add all non-entity nodes first
            for node in new_nodes:
                if node.type != NodeType.ENTITY:
                    if self.add_node(node):
                        stats['nodes_added'] += 1
                        node_mapping[node.id] = node.id
                    else:
                        node_mapping[node.id] = node.id  # Already exists
            
            # Add entity nodes with deduplication
            for node in new_nodes:
                if node.type == NodeType.ENTITY:
                    # Check for existing entity
                    existing_entity = self._find_duplicate_entity(node)
                    if existing_entity:
                        # Merge with existing
                        self._merge_entities(existing_entity, node)
                        node_mapping[node.id] = existing_entity
                        stats['entities_merged'] += 1
                        logger.debug(f"Merged entity {node.id} -> {existing_entity}")
                    else:
                        # Add new entity
                        if self.add_node(node):
                            stats['nodes_added'] += 1
                            node_mapping[node.id] = node.id
                        else:
                            node_mapping[node.id] = node.id
            
            logger.info(f"File {file_id} merge complete: {stats['nodes_added']} new nodes, {stats['entities_merged']} entities merged")
            return stats
            
        except Exception as e:
            logger.error(f"Error merging file nodes for {file_id}: {e}")
            stats['error'] = str(e)
            return stats
    
    def get_org_stats(self, processed_files: List[str] = None) -> Dict[str, Any]:
        """Get stats with file-level breakdown"""
        base_stats = self.get_stats()
        
        # Add org-specific stats
        base_stats.update({
            'org_level': True,
            'processed_files': processed_files or [],
            'files_count': len(processed_files or [])
        })
        
        # Get file-level node distribution if metadata available
        if processed_files:
            file_stats = {}
            for node_id, data in self.graph.nodes(data=True):
                file_id = data.get('metadata', {}).get('file_id')
                if file_id in processed_files:
                    if file_id not in file_stats:
                        file_stats[file_id] = {'nodes': 0, 'types': {}}
                    file_stats[file_id]['nodes'] += 1
                    node_type = data.get('type', 'unknown')
                    file_stats[file_id]['types'][node_type] = file_stats[file_id]['types'].get(node_type, 0) + 1
            
            base_stats['file_distribution'] = file_stats
        
        return base_stats
    
    def clear_file_nodes(self, file_id: str) -> Dict[str, Any]:
        """
        Remove or update nodes associated with a specific file.
        Preserves nodes that are referenced by multiple files (e.g., merged entities).
        """
        stats = {
            'nodes_removed': 0,
            'nodes_preserved': 0,
            'edges_removed': 0
        }

        try:
            nodes_to_remove = []
            nodes_to_update = []

            # Categorize nodes: remove vs update
            for node_id, data in self.graph.nodes(data=True):
                metadata = data.get('metadata', {})

                # Check both file_ids (set) and file_id (legacy single value)
                file_ids = metadata.get('file_ids', set())
                single_file_id = metadata.get('file_id')

                # Ensure file_ids is a set
                if not isinstance(file_ids, set):
                    file_ids = {file_ids} if file_ids else set()

                # Include legacy single file_id if it exists
                if single_file_id and single_file_id not in file_ids:
                    file_ids.add(single_file_id)

                # Check if this node references the file being deleted
                if file_id in file_ids:
                    if len(file_ids) > 1:
                        # Node is referenced by multiple files - preserve it
                        nodes_to_update.append((node_id, file_ids))
                        stats['nodes_preserved'] += 1
                    else:
                        # Node is only referenced by this file - delete it
                        nodes_to_remove.append(node_id)

            # Update multi-file nodes (remove file_id from their file_ids set)
            for node_id, file_ids in nodes_to_update:
                updated_file_ids = file_ids - {file_id}
                self.graph.nodes[node_id]['metadata']['file_ids'] = updated_file_ids
                logger.debug(f"Node {node_id} preserved (still referenced by {len(updated_file_ids)} files)")

            # Remove single-file nodes
            edges_before = self.graph.number_of_edges()
            for node_id in nodes_to_remove:
                if self.graph.has_node(node_id):
                    # Remove from entity index if it's an entity
                    node_data = self.graph.nodes[node_id]
                    if node_data.get('type') == NodeType.ENTITY.value:
                        entity_content = node_data.get('content', '').lower()
                        if entity_content in self.entity_index:
                            self.entity_index[entity_content].discard(node_id)
                            if not self.entity_index[entity_content]:
                                del self.entity_index[entity_content]

                    self.graph.remove_node(node_id)
                    stats['nodes_removed'] += 1

            stats['edges_removed'] = edges_before - self.graph.number_of_edges()

            logger.info(
                f"Cleared file {file_id}: {stats['nodes_removed']} nodes removed, "
                f"{stats['nodes_preserved']} nodes preserved (multi-file), "
                f"{stats['edges_removed']} edges removed"
            )
            return stats

        except Exception as e:
            logger.error(f"Error clearing file nodes for {file_id}: {e}")
            import traceback
            traceback.print_exc()
            stats['error'] = str(e)
            return stats
    
    def get_file_node_count(self, file_id: str) -> int:
        """Get count of nodes for a specific file"""
        count = 0
        for node_id, data in self.graph.nodes(data=True):
            metadata = data.get('metadata', {})
            file_ids = metadata.get('file_ids', set())
            single_file_id = metadata.get('file_id')

            # Handle legacy single file_id
            if not isinstance(file_ids, set):
                file_ids = {file_ids} if file_ids else set()
            if single_file_id and single_file_id not in file_ids:
                file_ids.add(single_file_id)

            if file_id in file_ids:
                count += 1
        return count

    def get_node_file_references(self, node_id: str) -> Set[str]:
        """
        Get all file IDs that reference a specific node.
        Useful for understanding which files are using a particular entity.
        """
        if not self.graph.has_node(node_id):
            return set()

        metadata = self.graph.nodes[node_id].get('metadata', {})
        file_ids = metadata.get('file_ids', set())
        single_file_id = metadata.get('file_id')

        # Ensure file_ids is a set
        if not isinstance(file_ids, set):
            file_ids = {file_ids} if file_ids else set()

        # Include legacy single file_id
        if single_file_id and single_file_id not in file_ids:
            file_ids.add(single_file_id)

        return file_ids

    def get_shared_nodes_stats(self) -> Dict[str, Any]:
        """
        Get statistics about nodes shared across multiple files.
        Useful for understanding entity deduplication effectiveness.
        """
        single_file_nodes = 0
        multi_file_nodes = 0
        max_file_count = 0
        file_distribution = defaultdict(int)

        for node_id, data in self.graph.nodes(data=True):
            file_refs = self.get_node_file_references(node_id)
            num_files = len(file_refs)

            if num_files == 1:
                single_file_nodes += 1
            elif num_files > 1:
                multi_file_nodes += 1
                max_file_count = max(max_file_count, num_files)

            file_distribution[num_files] += 1

        return {
            'single_file_nodes': single_file_nodes,
            'multi_file_nodes': multi_file_nodes,
            'max_files_per_node': max_file_count,
            'file_distribution': dict(file_distribution),
            'deduplication_ratio': multi_file_nodes / (single_file_nodes + multi_file_nodes) if (single_file_nodes + multi_file_nodes) > 0 else 0
        }
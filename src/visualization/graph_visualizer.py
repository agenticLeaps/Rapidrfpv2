import os
import pickle
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import networkx as nx
from pyvis.network import Network
import tempfile

from ..graph.graph_manager import GraphManager
from ..graph.node_types import NodeType
from ..config.settings import Config

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""
    height: str = "100vh"
    width: str = "100vw"
    bgcolor: str = "#1e1e1e"
    font_color: str = "white"
    max_nodes: int = 2000
    show_physics: bool = True
    show_buttons: bool = True
    notebook: bool = False
    
    # Node styling
    node_size_multiplier: float = 20.0
    min_node_size: float = 10.0
    max_node_size: float = 50.0
    
    # Edge styling
    edge_color: str = "#cccccc"
    edge_width: float = 1.0
    
    # Layout settings
    physics_enabled: bool = True
    stabilization_iterations: int = 100
    spring_length: int = 200
    spring_constant: float = 0.05
    damping: float = 0.4

class GraphVisualizer:
    """Interactive graph visualization using PyVis."""
    
    def __init__(self, graph_manager: GraphManager, config: VisualizationConfig = None):
        self.graph_manager = graph_manager
        self.config = config or VisualizationConfig()
        self.node_colors = {
            NodeType.TEXT: "#FF6B6B",           # Red
            NodeType.SEMANTIC: "#4ECDC4",       # Teal
            NodeType.ENTITY: "#45B7D1",         # Blue
            NodeType.RELATIONSHIP: "#96CEB4",   # Green
            NodeType.ATTRIBUTE: "#FFEAA7",      # Yellow
            NodeType.HIGH_LEVEL: "#DDA0DD",     # Plum
            NodeType.OVERVIEW: "#FFB347"        # Orange
        }
        
    def _calculate_node_importance_scores(self) -> Dict[str, float]:
        """Calculate importance scores for node sizing."""
        try:
            importance_scores = self.graph_manager.calculate_node_importance()
            if not importance_scores:
                # Fallback: use degree centrality
                graph = self.graph_manager.graph
                degree_centrality = nx.degree_centrality(graph)
                max_score = max(degree_centrality.values()) if degree_centrality else 1.0
                importance_scores = {
                    node_id: score / max_score 
                    for node_id, score in degree_centrality.items()
                }
            
            return importance_scores
        except Exception as e:
            logger.warning(f"Failed to calculate importance scores: {e}")
            # Return uniform scores
            return {node_id: 1.0 for node_id in self.graph_manager.graph.nodes()}
    
    def _filter_graph_by_importance(self, max_nodes: int = None) -> nx.Graph:
        """Filter graph to most important nodes."""
        if max_nodes is None:
            max_nodes = self.config.max_nodes
        
        graph = self.graph_manager.graph
        
        if len(graph.nodes()) <= max_nodes:
            return graph.copy()
        
        # Get importance scores
        importance_scores = self._calculate_node_importance_scores()
        
        # Sort nodes by importance
        sorted_nodes = sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Take top nodes
        top_nodes = [node_id for node_id, _ in sorted_nodes[:max_nodes]]
        
        # Create subgraph
        subgraph = graph.subgraph(top_nodes).copy()
        
        # Ensure connectivity by adding bridge nodes if needed
        if not nx.is_connected(subgraph.to_undirected()):
            subgraph = self._ensure_connectivity(graph, subgraph, importance_scores)
        
        logger.info(f"Filtered graph from {len(graph.nodes())} to {len(subgraph.nodes())} nodes")
        return subgraph
    
    def _ensure_connectivity(self, original_graph: nx.Graph, subgraph: nx.Graph, 
                           importance_scores: Dict[str, float]) -> nx.Graph:
        """Ensure subgraph connectivity by adding bridge nodes."""
        try:
            # Find connected components
            components = list(nx.connected_components(subgraph.to_undirected()))
            
            if len(components) <= 1:
                return subgraph
            
            # Find shortest paths between components in original graph
            bridge_nodes = set()
            main_component = max(components, key=len)
            
            for component in components:
                if component == main_component:
                    continue
                
                # Find shortest path from this component to main component
                try:
                    source = next(iter(component))
                    target = next(iter(main_component))
                    
                    if nx.has_path(original_graph, source, target):
                        path = nx.shortest_path(original_graph, source, target)
                        bridge_nodes.update(path)
                except (nx.NetworkXNoPath, StopIteration):
                    continue
            
            # Add bridge nodes to subgraph
            all_nodes = set(subgraph.nodes()) | bridge_nodes
            connected_subgraph = original_graph.subgraph(all_nodes).copy()
            
            logger.info(f"Added {len(bridge_nodes)} bridge nodes for connectivity")
            return connected_subgraph
            
        except Exception as e:
            logger.warning(f"Failed to ensure connectivity: {e}")
            return subgraph
    
    def _get_node_color(self, node_type: NodeType) -> str:
        """Get color for node type."""
        return self.node_colors.get(node_type, "#CCCCCC")
    
    def _calculate_node_size(self, importance_score: float) -> float:
        """Calculate node size based on importance."""
        # Normalize and scale
        size = (
            self.config.min_node_size + 
            importance_score * (self.config.max_node_size - self.config.min_node_size)
        )
        return max(self.config.min_node_size, min(size, self.config.max_node_size))
    
    def _create_node_title(self, node) -> str:
        """Create HTML title for node hover."""
        title_parts = [
            f"<b>ID:</b> {node.id}",
            f"<b>Type:</b> {node.type.value}",
            f"<b>Content:</b> {node.content[:100]}{'...' if len(node.content) > 100 else ''}",
        ]
        
        # Add metadata info
        if node.metadata:
            title_parts.append("<b>Metadata:</b>")
            for key, value in list(node.metadata.items())[:3]:
                title_parts.append(f"  {key}: {str(value)[:50]}")
        
        return "<br>".join(title_parts)
    
    def create_visualization(self, 
                           output_path: str = None,
                           max_nodes: int = None,
                           filter_node_types: List[NodeType] = None,
                           highlight_communities: bool = True) -> str:
        """
        Create interactive graph visualization.
        
        Args:
            output_path: Output HTML file path
            max_nodes: Maximum number of nodes to display
            filter_node_types: Node types to include (None for all)
            highlight_communities: Whether to highlight communities
            
        Returns:
            Path to generated HTML file
        """
        try:
            # Filter graph
            filtered_graph = self._filter_graph_by_importance(max_nodes)
            
            # Apply node type filter
            if filter_node_types:
                filtered_nodes = []
                for node_id in filtered_graph.nodes():
                    node = self.graph_manager.get_node(node_id)
                    if node and node.type in filter_node_types:
                        filtered_nodes.append(node_id)
                filtered_graph = filtered_graph.subgraph(filtered_nodes)
            
            # Get importance scores for sizing
            importance_scores = self._calculate_node_importance_scores()
            
            # Create PyVis network
            net = Network(
                height=self.config.height,
                width=self.config.width,
                bgcolor=self.config.bgcolor,
                font_color=self.config.font_color,
                notebook=self.config.notebook
            )
            
            # Add nodes
            for node_id in filtered_graph.nodes():
                node = self.graph_manager.get_node(node_id)
                if not node:
                    continue
                
                # Calculate node size
                importance = importance_scores.get(node_id, 0.5)
                size = self._calculate_node_size(importance)
                
                # Get node color
                color = self._get_node_color(node.type)
                
                # Create title
                title = self._create_node_title(node)
                
                # Add node
                net.add_node(
                    node_id,
                    label=f"{node.type.value}",
                    title=title,
                    color=color,
                    size=size,
                    shape="dot"
                )
            
            # Add edges
            for source, target in filtered_graph.edges():
                # Get edge data
                edge_data = filtered_graph.get_edge_data(source, target, {})
                
                # Create edge title
                edge_title = f"Connection: {source} → {target}"
                if 'relationship_type' in edge_data:
                    edge_title += f"<br>Type: {edge_data['relationship_type']}"
                
                net.add_edge(
                    source,
                    target,
                    title=edge_title,
                    color=self.config.edge_color,
                    width=self.config.edge_width
                )
            
            # Set physics options
            if self.config.physics_enabled:
                physics_options = {
                    "physics": {
                        "enabled": True,
                        "stabilization": {
                            "enabled": True,
                            "iterations": self.config.stabilization_iterations
                        },
                        "barnesHut": {
                            "gravitationalConstant": -2000,
                            "centralGravity": 0.3,
                            "springLength": self.config.spring_length,
                            "springConstant": self.config.spring_constant,
                            "damping": self.config.damping
                        }
                    }
                }
                net.set_options(json.dumps(physics_options))
            
            # Generate output path if not provided
            if output_path is None:
                output_path = os.path.join(Config.DATA_DIR, "processed", "graph_visualization.html")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save visualization
            net.show(output_path, notebook=self.config.notebook)
            
            logger.info(f"Graph visualization saved to: {output_path}")
            logger.info(f"Nodes: {len(filtered_graph.nodes())}, Edges: {len(filtered_graph.edges())}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            raise
    
    def create_community_visualization(self, output_path: str = None) -> str:
        """Create visualization highlighting communities."""
        try:
            # Detect communities
            communities = self.graph_manager.detect_communities()
            
            if not communities:
                logger.warning("No communities detected, creating standard visualization")
                return self.create_visualization(output_path)
            
            # Filter graph
            filtered_graph = self._filter_graph_by_importance()
            importance_scores = self._calculate_node_importance_scores()
            
            # Create community color map
            community_colors = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
                "#DDA0DD", "#FFB347", "#98D8C8", "#F7DC6F", "#BB8FCE"
            ]
            
            # Create PyVis network
            net = Network(
                height=self.config.height,
                width=self.config.width,
                bgcolor=self.config.bgcolor,
                font_color=self.config.font_color,
                notebook=self.config.notebook
            )
            
            # Add nodes with community colors
            for node_id in filtered_graph.nodes():
                node = self.graph_manager.get_node(node_id)
                if not node:
                    continue
                
                # Get community ID for this node
                community_id = communities.get(node_id, None)
                
                # Get community color
                if community_id is not None:
                    color_idx = int(community_id) % len(community_colors)
                    color = community_colors[color_idx]
                else:
                    color = "#CCCCCC"
                
                # Calculate size
                importance = importance_scores.get(node_id, 0.5)
                size = self._calculate_node_size(importance)
                
                # Create title
                title = self._create_node_title(node)
                if community_id is not None:
                    title += f"<br><b>Community:</b> {community_id}"
                
                # Add node
                net.add_node(
                    node_id,
                    label=f"{node.type.value}",
                    title=title,
                    color=color,
                    size=size,
                    shape="dot"
                )
            
            # Add edges
            for source, target in filtered_graph.edges():
                edge_data = filtered_graph.get_edge_data(source, target, {})
                
                edge_title = f"Connection: {source} → {target}"
                if 'relationship_type' in edge_data:
                    edge_title += f"<br>Type: {edge_data['relationship_type']}"
                
                net.add_edge(
                    source,
                    target,
                    title=edge_title,
                    color=self.config.edge_color,
                    width=self.config.edge_width
                )
            
            # Set physics options
            if self.config.physics_enabled:
                physics_options = {
                    "physics": {
                        "enabled": True,
                        "stabilization": {"enabled": True},
                        "barnesHut": {
                            "gravitationalConstant": -2000,
                            "centralGravity": 0.3,
                            "springLength": self.config.spring_length
                        }
                    }
                }
                net.set_options(json.dumps(physics_options))
            
            # Generate output path if not provided
            if output_path is None:
                output_path = os.path.join(Config.DATA_DIR, "processed", "community_visualization.html")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save visualization
            net.show(output_path, notebook=self.config.notebook)
            
            logger.info(f"Community visualization saved to: {output_path}")
            logger.info(f"Communities: {len(communities)}, Nodes: {len(filtered_graph.nodes())}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create community visualization: {e}")
            raise
    
    def create_node_type_visualization(self, node_types: List[NodeType], 
                                     output_path: str = None) -> str:
        """Create visualization focusing on specific node types."""
        return self.create_visualization(
            output_path=output_path,
            filter_node_types=node_types
        )
    
    def get_visualization_stats(self) -> Dict[str, Any]:
        """Get statistics about the current graph for visualization."""
        try:
            graph = self.graph_manager.graph
            
            # Basic stats
            stats = {
                'total_nodes': len(graph.nodes()),
                'total_edges': len(graph.edges()),
                'is_connected': nx.is_connected(graph.to_undirected()),
                'num_components': nx.number_connected_components(graph.to_undirected()),
                'density': nx.density(graph),
                'avg_clustering': nx.average_clustering(graph.to_undirected())
            }
            
            # Node type breakdown
            node_type_counts = {}
            for node_id in graph.nodes():
                node = self.graph_manager.get_node(node_id)
                if node:
                    node_type = node.type.value
                    node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
            
            stats['node_type_counts'] = node_type_counts
            
            # Community stats
            try:
                communities = self.graph_manager.detect_communities()
                stats['num_communities'] = len(communities) if communities else 0
                if communities:
                    community_sizes = [len(nodes) for nodes in communities.values()]
                    stats['avg_community_size'] = sum(community_sizes) / len(community_sizes)
                    stats['max_community_size'] = max(community_sizes)
            except:
                stats['num_communities'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get visualization stats: {e}")
            return {'error': str(e)}